from __future__ import annotations

import asyncio
import gc
import re
import time
from collections import Counter

import numpy as np
import polars as pl
from codecarbon import EmissionsTracker

from mds_norm.paths import EMISSIONS_LOG
from mds_norm.pipeline.vocab_indexes import (
    CONCEPT_GROUPS,
    GROUP_VOCABS,
    KIND_PRIORITY,
    english_only,
    gloss_index,
    group_index,
)
from mds_norm.utils.atomise import DATE_LIKE, GROUP_DESC

EMBED_MODEL = "LiquidAI/LFM2.5-Embedding-350M"
ENCODE_BATCH = 128  # batch and chunk sizes set for a 32GB GPU
SIM_CHUNK = 4096

TOP_K = 5
# Cosines run low here; the floor only excludes hopeless candidates
RETRIEVE_FLOOR = 0.20
MIN_LEN = 3
MAX_LEN = 100  # longer values are prose, not vocabulary terms
COVERAGE = 0.99  # occurrence share of each group's queue that earns a call

LLM_MODEL = "gpt-oss-20b"
LLM_API_BASE = "http://localhost:30000/v1"
LLM_CONCURRENCY = 200
# Generous cap: a truncated call must not read as rejection
LLM_MAX_NEW_TOKENS = 3072
LLM_TEMPERATURE = 0.2
LLM_EXTRA_BODY = {"top_p": 1.0, "reasoning_effort": "low"}

PROMPT = """A UK museum catalogue record records "{value}" in a field holding a {desc}.

Candidate terms from the field's controlled vocabularies:

{options}

Which candidate names the same thing as the recorded value? Rules:
- Choose a candidate only if it denotes the same concept. A broader, narrower or merely associated term is not a
  match: for `oak` neither `wood` nor `oak gall` is the concept.
- Spelling, plural and word-form differences do not matter (`engraved` and `engraving` are the same concept), and
  neither does word order.
- Judge the concept, not the string: high string similarity between different things (`Bronze Age` and `bronze`,
  `tin` and `tine`) is not a match.
- If the value names something no candidate denotes, or you cannot tell which, answer 0.

Answer with only the number of the single best candidate, or 0 for none of them."""

CHOICE_RX = re.compile(r"\d+")

CANDIDATE_SCHEMA = {
    "norm": pl.String,
    "score": pl.Float64,
    "rank": pl.Int64,
    "vocab": pl.String,
    "subject": pl.String,
    "matched_term": pl.String,
    "gloss": pl.String,
    "group": pl.String,
}
DECISION_SCHEMA = {
    "group": pl.String,
    "norm": pl.String,
    "answered": pl.Boolean,
    "pick": pl.Int64,
    "n_candidates": pl.UInt32,
    "vocab": pl.String,
    "subject": pl.String,
    "matched_term": pl.String,
    "score": pl.Float64,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rerank_queue(pending: pl.DataFrame) -> pl.DataFrame:
    """Term-shaped deferred atoms, capped at the values covering most of each group's mass"""
    queue = (
        pending.filter(
            pl.col("group").is_in(sorted(CONCEPT_GROUPS))
            & pl.col("norm").str.len_chars().is_between(MIN_LEN, MAX_LEN)
            & pl.col("norm").str.contains(r"[a-z]")
            & ~((pl.col("group") == "periodo") & pl.col("norm").str.contains(DATE_LIKE))
        )
        .group_by("group", "norm")
        .agg(occ=pl.col("count").sum(), atom=pl.col("atom").sort_by("count", descending=True).first())
        # ties broken on the norm, so queuing is stable
        .sort(["occ", "norm"], descending=[True, False])
        # Mass the commoner atoms cover, so single-atom groups still queue
        .with_columns(above=((pl.col("occ").cum_sum() - pl.col("occ")) / pl.col("occ").sum()).over("group"))
    )
    capped = queue.filter(pl.col("above") < COVERAGE).drop("above").sort("group", "norm")
    log(
        f"rerank queue: {len(capped):,} of {len(queue):,} distinct atoms "
        f"({capped['occ'].sum():,} of {queue['occ'].sum():,} occurrences)"
    )
    return capped


def documents(group: str) -> pl.DataFrame:
    """One row per subject in the group's vocabularies: its preferred term and one line of context"""
    index = english_only(group_index(group))
    best = (
        index.filter(pl.col("kind").is_in(["prefLabelGVP", "prefLabel"]))
        .sort(pl.col("kind").replace_strict(KIND_PRIORITY, return_dtype=pl.Int8), pl.col("term"))
        .unique(["vocab", "subject"], keep="first")
        .select("vocab", "subject", matched_term=pl.col("term"))
    )
    glosses = pl.concat(
        [gloss_index(vocab).with_columns(vocab=pl.lit(vocab)) for vocab in GROUP_VOCABS[group]], how="diagonal"
    )
    return best.join(glosses, on=["vocab", "subject"], how="left").sort("vocab", "subject").collect(engine="streaming")


def top_k(queries: list[str], q_emb: np.ndarray, d_emb: np.ndarray) -> pl.DataFrame:
    """The TOP_K best documents per query, ranked, in bounded chunks — the full matrix OOMs"""
    k = min(TOP_K, d_emb.shape[0])
    idx, score = [], []
    for i in range(0, len(queries), SIM_CHUNK):
        sims = q_emb[i : i + SIM_CHUNK] @ d_emb.T
        part = np.argpartition(sims, -k, axis=1)[:, -k:]
        part_scores = np.take_along_axis(sims, part, axis=1)
        # Ties break toward the lower index so ranking reproduces
        order = np.lexsort((part, -part_scores), axis=1)
        idx.append(np.take_along_axis(part, order, axis=1))
        score.append(np.take_along_axis(part_scores, order, axis=1))
    return (
        pl.DataFrame(
            {"norm": queries, "doc_idx": np.concatenate(idx).tolist(), "score": np.concatenate(score).tolist()}
        )
        .explode("doc_idx", "score")
        .with_columns(rank=pl.int_range(pl.len()).over("norm") + 1, doc_idx=pl.col("doc_idx").cast(pl.Int64))
        .filter(pl.col("score") >= RETRIEVE_FLOOR)
    )


def candidates(queue: pl.DataFrame) -> pl.DataFrame:
    """Retrieve each queued atom's TOP_K candidate subjects by cosine, with their glosses"""
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device="cuda")
    encode = {
        "normalize_embeddings": True,
        "batch_size": ENCODE_BATCH,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    frames = []
    for group in sorted(CONCEPT_GROUPS):
        queries = queue.filter(pl.col("group") == group)["norm"].to_list()
        if not queries:
            continue
        docs = documents(group)
        d_emb = model.encode(docs["matched_term"].to_list(), prompt_name="document", **encode)
        q_emb = model.encode(queries, prompt_name="query", **encode)
        frames.append(
            top_k(queries, q_emb, d_emb)
            .join(docs.with_row_index("doc_idx").with_columns(pl.col("doc_idx").cast(pl.Int64)), on="doc_idx")
            .drop("doc_idx")
            .with_columns(group=pl.lit(group))
        )
        log(f"  {group}: {len(queries):,} atoms against {len(docs):,} subjects")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return pl.concat(frames).sort("group", "norm", "rank") if frames else pl.DataFrame(schema=CANDIDATE_SCHEMA)


def render_options(terms: list[str], glosses: list[str | None]) -> str:
    return "\n".join(
        f"{i}. {term}" + (f" — {gloss}" if gloss else "")
        for i, (term, gloss) in enumerate(zip(terms, glosses, strict=True), 1)
    )


def parse_choice(completion: str | None, n_options: int) -> int | None:
    """The option the model picked; None for a refusal, a rejection or an unusable answer"""
    if not completion:
        return None
    numbers = CHOICE_RX.findall(completion)
    if not numbers:
        return None
    choice = int(numbers[-1])  # the last integer is the answer, even after restatement
    return choice if 1 <= choice <= n_options else None


async def choose(
    cands: pl.DataFrame, queue: pl.DataFrame, model: str = LLM_MODEL, prompt: str = PROMPT, **decoding: object
) -> pl.DataFrame:
    """One row per queued atom: the candidate the model selected, if it selected one"""
    from mds_norm.utils.inference import Inference

    # A `context` column is optional; the production prompt ignores it
    carry = ["group", "norm", "atom"] + (["context"] if "context" in queue.columns else [])
    prompts = (
        cands.group_by("group", "norm")
        .agg(pl.col("matched_term"), pl.col("gloss"), pl.col("subject"), pl.col("vocab"), pl.col("score"))
        .join(queue.select(carry), on=["group", "norm"])
        .sort("group", "norm")
    )
    if not len(prompts):
        return pl.DataFrame(schema=DECISION_SCHEMA)

    samples = [
        {
            "desc": GROUP_DESC[row["group"]],
            "value": row["atom"],
            "context": row.get("context") or "",
            "options": render_options(row["matched_term"], row["gloss"]),
        }
        for row in prompts.iter_rows(named=True)
    ]
    body = {"max_tokens": LLM_MAX_NEW_TOKENS, "temperature": LLM_TEMPERATURE, "extra_body": LLM_EXTRA_BODY} | decoding
    inf = Inference(model=model, base_url=LLM_API_BASE, concurrency=LLM_CONCURRENCY, timeout=600.0)
    with EmissionsTracker(
        project_name="vocab_rerank_llm", output_dir=str(EMISSIONS_LOG), log_level="error", tracking_mode="machine"
    ):
        answers = await inf.generate(samples, prompt, usage=True, **body)

    # A failed call is worth its reason
    errors = Counter(a["error"].split("(")[0] for a in answers if a["error"])
    for kind, n in errors.most_common(3):
        log(f"  rerank: {n:,} calls failed with {kind}")
    completions = [a["content"] for a in answers]
    picks = [
        parse_choice(c, len(row["matched_term"]))
        for c, row in zip(completions, prompts.iter_rows(named=True), strict=True)
    ]
    return prompts.with_columns(
        # An empty completion is unanswered, not a rejection
        answered=pl.Series([bool(c) for c in completions]),
        pick=pl.Series(picks, dtype=pl.Int64),
    ).select(
        "group",
        "norm",
        "answered",
        "pick",
        n_candidates=pl.col("matched_term").list.len().cast(pl.UInt32),
        vocab=pl.col("vocab").list.get(pl.col("pick") - 1),
        subject=pl.col("subject").list.get(pl.col("pick") - 1),
        matched_term=pl.col("matched_term").list.get(pl.col("pick") - 1),
        score=pl.col("score").list.get(pl.col("pick") - 1),
    )


def rerank_tier(pending: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Retrieve, then let the model select one candidate or reject them all"""
    queue = rerank_queue(pending)
    cands = candidates(queue)
    decisions = asyncio.run(choose(cands, queue))
    hits = decisions.filter(pl.col("subject").is_not_null()).drop("answered", "pick")
    unanswered = int((~decisions["answered"]).sum()) if len(decisions) else 0
    if unanswered:
        log(f"  rerank: {unanswered:,} calls returned no usable answer — those atoms stay deferred")
    log(f"  rerank: {len(hits):,} of {len(queue):,} queued atoms selected a candidate")
    return hits.with_columns(sub_component=pl.lit("rerank"), resolved_by=pl.lit("llm_choice")), queue
