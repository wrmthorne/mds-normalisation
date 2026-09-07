from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from codecarbon import EmissionsTracker
from mds_data_model.introspection import all_vocab_fields

from mds_norm.evaluation.reports import OWNED_MIN_SHARE
from mds_norm.paths import COMPILED, EMISSIONS_LOG, FIELD_STATS, ROOT, VOCABS
from mds_norm.pipeline.build_local_termlists import load_termlist_index
from mds_norm.pipeline.vocab_indexes import GROUP_FOR
from mds_norm.utils.atomise import (
    ATOM_PROSE_MAX_CHARS,
    ATOM_PROSE_MAX_TOKENS,
    GROUP_DESC,
    LLM_COVERAGE,
    LLM_MAX_NEW_TOKENS,
    LLM_PROMPT,
    NULL_MARKERS,
    PLACEHOLDER_MARKERS,
    PROSE_MAX_CHARS,
    PROSE_MAX_TOKENS,
    SEMANTIC_MARKERS,
    atomise,
    morph_variants,
    parse_llm_atoms,
    separator_regex,
    us_variant,
)

CENSUS = COMPILED / "coverage_census.parquet"
VOCAB_MAP = VOCABS / "field_vocab_map.json"
OUT_DIR = ROOT / "experiments" / "termlist_pricing"
EMISSIONS = EMISSIONS_LOG

# The cascade constants; pricing must run production settings
MIN_ATTEST_COUNT = 5
ATTEST_THRESHOLD = 0.5
MIN_DISTINCT_FRAGMENTS = 10
MIN_SUPPORT_RECORDS = 200
# The closed candidate set this variant was costed with
FIXED_CANDIDATES = (",", "/", "&", "+", "and", "or")
FUZZY_ACCEPT = 93.0
FUZZY_MIN_LEN = 4
SEMANTIC_ACCEPT = 0.35
SEMANTIC_MARGIN = 0.10
SEMANTIC_MIN_LEN = 3
SEMANTIC_MAX_LEN = 100
ENCODE_BATCH = 128
LLM_BATCH = 64

# Fields never automated, or too varied for a controlled vocabulary
TERMLIST_EXCLUDED = {
    "spectrum/object_name_type",
    "spectrum/other_number_type",
    "spectrum/text_reason",
    "spectrum/responsible_department_section",
    "spectrum/dimension_measured_part",
}
ADMIT_OCC, ADMIT_INST = 100_000, 10


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_term(col: pl.Expr) -> pl.Expr:
    return col.str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars(" .,;:")


def pnorm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip(" .,;:")


class Rung:
    """Times + energy-tracks one cascade rung and logs its coverage gain"""

    def __init__(self, costs: list, name: str) -> None:
        self.costs, self.name = costs, name

    def __enter__(self) -> Rung:
        self.t0 = time.time()
        self.tracker = EmissionsTracker(
            project_name=f"termlist_{self.name}", output_dir=str(EMISSIONS), log_level="error"
        )
        self.tracker.start()
        return self

    def __exit__(self, *exc: object) -> bool:
        self.tracker.stop()
        d = self.tracker.final_emissions_data
        self.costs.append(
            {
                "rung": self.name,
                "seconds": round(time.time() - self.t0, 2),
                "energy_kwh": d.energy_consumed,
                "emissions_kg": d.emissions,
            }
        )
        return False


def llm_worker(queue_path: str, out_path: str) -> None:
    """Generate atomiser completions for the queued atoms (subprocess body)"""
    import torch
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    payload = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    desc, queue = payload["desc"], payload["atoms"]
    mid = "LiquidAI/LFM2.5-350M"
    # This transformers version cannot resolve the declared tokenizer class
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(hf_hub_download(mid, "tokenizer.json")))
    tok.chat_template = Path(hf_hub_download(mid, "chat_template.jinja")).read_text()
    with Path(hf_hub_download(mid, "tokenizer_config.json")).open(encoding="utf-8") as f:
        tcfg = json.load(f)
    for attr in ("eos_token", "pad_token", "bos_token"):
        if tcfg.get(attr):
            v = tcfg[attr]
            setattr(tok, attr, v["content"] if isinstance(v, dict) else v)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("cuda").eval()
    out: dict[str, str] = {}
    for i in range(0, len(queue), LLM_BATCH):
        batch = queue[i : i + LLM_BATCH]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": LLM_PROMPT.format(desc=desc, value=v)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for v in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left", return_token_type_ids=False).to(
            "cuda"
        )
        with torch.no_grad():
            # Production sampling; greedy decode echoes the prompt's example atoms
            gen = lm.generate(
                **enc,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.1,
                top_k=50,
                repetition_penalty=1.05,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        for v, seq in zip(batch, gen, strict=True):
            out[v] = tok.decode(seq[enc["input_ids"].shape[1] :], skip_special_tokens=True)
        if i % (LLM_BATCH * 4) == 0:
            log(f"llm worker: {min(i + LLM_BATCH, len(queue))}/{len(queue)}")
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Price a local termlist field through the whole cascade, rung by rung.")
    ap.add_argument("--field", default="persons_association")
    ap.add_argument("--skip-llm", action="store_true", help="price the deterministic + embedding rungs only")
    ap.add_argument(
        "--endpoint",
        action="store_true",
        help="run the LLM rung against the served vllm endpoint "
        "(LFM2.5-350M at localhost:30000) instead of the "
        "transformers harness — required for a valid yield",
    )
    ap.add_argument("--llm-worker", nargs=2, metavar=("QUEUE", "OUT"), help="internal: run the generation subprocess")
    args = ap.parse_args()
    if args.llm_worker:
        llm_worker(*args.llm_worker)
        return
    field = args.field
    # A field's group is its whole target list
    group = GROUP_FOR[f"spectrum/{field}"]
    desc = GROUP_DESC[group]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index = load_termlist_index(field)
    index_norms = set(index["norm"].to_list())
    # norm -> canonical term (prefLabel of the subject)
    pref = {
        r["norm"]: r["term"]
        for r in index.sort(pl.col("kind") != "prefLabel").unique("norm", keep="first").iter_rows(named=True)
    }
    log(f"{field}: {index['subject'].n_unique()} terms, {len(index_norms)} surface norms")

    costs: list[dict] = []
    resolved: dict[str, str] = {}  # atom norm -> rung name (first hit wins)

    with Rung(costs, "harvest"):
        vals = (
            pl.scan_parquet(FIELD_STATS)
            .filter((pl.col("field_type") == f"spectrum/{field}") & pl.col("value").is_not_null())
            .with_columns(pl.col("value").str.strip_chars().str.replace_all(r"\s+", " "))
            .filter(pl.col("value") != "")
            .with_columns(pl.col("data_source").cast(pl.String))
            .group_by("data_source", "value")
            .agg(count=pl.len())
            .with_columns(norm=norm_term(pl.col("value")))
            .collect(engine="streaming")
        )
    total_occ = int(vals["count"].sum())
    log(f"{field}: {vals.height:,} distinct (institution, value), {total_occ:,} occurrences")

    vals = vals.with_columns(
        route=pl.when(pl.col("norm").is_in(sorted(PLACEHOLDER_MARKERS)) | (pl.col("norm") == ""))
        .then(pl.lit("null_marker"))
        .when(pl.col("norm").is_in(sorted(SEMANTIC_MARKERS)))
        .then(pl.lit("semantic_marker"))
        .when(
            (pl.col("value").str.len_chars() > PROSE_MAX_CHARS)
            | (pl.col("value").str.split(" ").list.len() > PROSE_MAX_TOKENS)
        )
        .then(pl.lit("prose"))
        .otherwise(pl.lit("cascade"))
    )

    with Rung(costs, "delimiter_induction"):
        cvals = vals.filter(pl.col("route") == "cascade")
        attested = (
            set(
                cvals.group_by("norm")
                .agg(pl.col("count").sum())
                .filter(pl.col("count") >= MIN_ATTEST_COUNT)["norm"]
                .to_list()
            )
            | index_norms
        )
        accepted: dict[str, list[str]] = {}
        for name in FIXED_CANDIDATES:
            rx = re.compile(separator_regex(name))
            gate = f" {name} " if name.isalpha() else name
            subset = cvals.filter(pl.col("value").str.contains(gate, literal=True))
            if not len(subset):
                continue
            for ds, part in subset.group_by("data_source"):
                frags_total = frags_attested = 0
                distinct_attested: set[str] = set()
                support = 0
                for v, c in zip(part["value"], part["count"], strict=True):
                    frags = [pnorm(p) for p in rx.split(v) if p.strip()]
                    if len(frags) <= 1:
                        continue
                    support += c
                    frags_total += len(frags) * c
                    for fr in frags:
                        if fr in attested:
                            frags_attested += c
                            distinct_attested.add(fr)
                if (
                    frags_total
                    and frags_attested / frags_total >= ATTEST_THRESHOLD
                    and len(distinct_attested) >= MIN_DISTINCT_FRAGMENTS
                    and support >= MIN_SUPPORT_RECORDS
                ):
                    accepted.setdefault(ds[0], []).append(name)
    log(f"{field}: separators accepted for {len(accepted)} institutions")

    with Rung(costs, "atomise"):
        whole_hit = pl.col("norm").is_in(sorted(index_norms))
        cascade = vals.filter(pl.col("route") == "cascade")
        keep_whole = cascade.filter(whole_hit | ~pl.col("value").str.contains(r"[;|,/&+]| and | or "))
        to_split = cascade.join(keep_whole.select("data_source", "value"), on=["data_source", "value"], how="anti")
        rows = []
        for ds, v, c in zip(to_split["data_source"], to_split["value"], to_split["count"], strict=True):
            rows.extend((ds, v, c, a["atom"]) for a in atomise(v, accepted.get(ds)))
        atoms = pl.concat(
            [
                keep_whole.select("data_source", "value", "count", atom=pl.col("value")),
                pl.DataFrame(
                    rows,
                    schema={"data_source": pl.String, "value": pl.String, "count": pl.UInt32, "atom": pl.String},
                    orient="row",
                ),
            ]
        ).with_columns(norm=norm_term(pl.col("atom")))
        atoms = atoms.with_columns(
            atom_route=pl.when(pl.col("norm").is_in(sorted(NULL_MARKERS)) | (pl.col("norm") == ""))
            .then(pl.lit("null_marker"))
            .when(
                (pl.col("atom").str.len_chars() > ATOM_PROSE_MAX_CHARS)
                | (pl.col("atom").str.split(" ").list.len() > ATOM_PROSE_MAX_TOKENS)
            )
            .then(pl.lit("prose"))
            .otherwise(pl.lit("cascade"))
        )
    cascade_atoms = atoms.filter(pl.col("atom_route") == "cascade")
    atom_occ = int(cascade_atoms["count"].sum())
    log(
        f"{field}: {atoms.height:,} atoms, {cascade_atoms['norm'].n_unique():,} "
        f"distinct cascade norms / {atom_occ:,} atom-occurrences"
    )

    def pending() -> pl.DataFrame:
        return cascade_atoms.filter(~pl.col("norm").is_in(list(resolved))).select("norm").unique()

    def occ_of(norms: set[str]) -> int:
        return int(cascade_atoms.filter(pl.col("norm").is_in(sorted(norms)))["count"].sum())

    def take(name: str, hit_norms: set[str]) -> None:
        new = {n for n in hit_norms if n not in resolved}
        for n in new:
            resolved[n] = name
        costs[-1].update(new_norms=len(new), new_occ=occ_of(new))
        log(f"  {name}: +{len(new):,} norms / +{costs[-1]['new_occ']:,} occ")

    with Rung(costs, "exact"):
        hits = {n for n in pending()["norm"] if n in index_norms}
    take("exact", hits)

    with Rung(costs, "exact_variant"):
        hits = {n for n in pending()["norm"] if (v := us_variant(n)) and v in index_norms}
    take("exact_variant", hits)

    with Rung(costs, "exact_paren"):
        hits = set()
        for n in pending()["norm"]:
            m = re.match(r"^(.+?) ?\((.+)\)$", n)
            if m and pnorm(m.group(1)) in index_norms:
                hits.add(n)
    take("exact_paren", hits)

    with Rung(costs, "exact_morph"):
        hits = {n for n in pending()["norm"] if any(v in index_norms for v in morph_variants(n))}
    take("exact_morph", hits)

    with Rung(costs, "fuzzy"):
        from rapidfuzz import fuzz
        from rapidfuzz.process import cdist

        queries = [n for n in pending()["norm"] if len(n) >= FUZZY_MIN_LEN]
        choices = sorted(index_norms)
        hits = set()
        if queries:
            scores = cdist(queries, choices, scorer=fuzz.ratio, score_cutoff=FUZZY_ACCEPT, workers=-1, dtype=np.uint8)
            best = scores.max(axis=1)
            hits = {q for q, s in zip(queries, best, strict=True) if s >= FUZZY_ACCEPT}
    take("fuzzy", hits)

    with Rung(costs, "semantic"):
        import torch
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("LiquidAI/LFM2.5-Embedding-350M", trust_remote_code=True, device="cuda")
        docs = sorted({pref[n] for n in index_norms if n in pref})
        queries = [
            n for n in pending()["norm"] if SEMANTIC_MIN_LEN <= len(n) <= SEMANTIC_MAX_LEN and re.search(r"[a-z]", n)
        ]
        hits = set()
        if queries:
            d_emb = model.encode(
                docs,
                prompt_name="document",
                normalize_embeddings=True,
                batch_size=ENCODE_BATCH,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            q_emb = model.encode(
                queries,
                prompt_name="query",
                normalize_embeddings=True,
                batch_size=ENCODE_BATCH,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            sims = q_emb @ d_emb.T
            if sims.shape[1] > 1:
                part = np.argpartition(sims, -2, axis=1)[:, -2:]
                ps = np.take_along_axis(sims, part, axis=1)
                top = ps.max(axis=1)
                margin = np.abs(ps[:, 1] - ps[:, 0])
                hits = {
                    q
                    for q, s, m in zip(queries, top, margin, strict=True)
                    if s >= SEMANTIC_ACCEPT and m >= SEMANTIC_MARGIN
                }
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    take("semantic", hits)

    llm_queue = (
        cascade_atoms.filter(~pl.col("norm").is_in(list(resolved)) & pl.col("atom").str.contains(" ", literal=True))
        .group_by("atom")
        .agg(occ=pl.col("count").sum())
        .sort("occ", descending=True)
        .with_columns(cum=pl.col("occ").cum_sum() / pl.col("occ").sum())
    )
    llm_queue = llm_queue.filter(pl.col("cum") <= LLM_COVERAGE)
    log(f"{field}: LLM queue {llm_queue.height:,} atoms ({int(llm_queue['occ'].sum()):,} occ)")
    if not args.skip_llm and llm_queue.height:
        with Rung(costs, "llm_atomiser"):
            if args.endpoint:
                # The cascade's tier-4 rung verbatim: same model and sampling
                import asyncio

                from mds_norm.utils.inference import Inference

                inf = Inference(model="LFM2.5-350M", concurrency=256, timeout=600.0)
                queued = llm_queue["atom"].to_list()
                comps = asyncio.run(
                    inf.generate(
                        [{"desc": desc, "value": v} for v in queued],
                        LLM_PROMPT,
                        max_tokens=LLM_MAX_NEW_TOKENS,
                        temperature=0.1,
                        extra_body={"top_k": 50, "repetition_penalty": 1.05},
                    )
                )
                completions = dict(zip(queued, comps, strict=True))
            else:
                # A fresh process: the embedding model patches transformers globally
                import subprocess
                import tempfile

                with tempfile.TemporaryDirectory() as td:
                    qpath, opath = Path(td) / "queue.json", Path(td) / "out.json"
                    qpath.write_text(
                        json.dumps({"desc": desc, "atoms": llm_queue["atom"].to_list()}), encoding="utf-8"
                    )
                    subprocess.run([sys.executable, __file__, "--llm-worker", str(qpath), str(opath)], check=True)
                    completions = json.loads(opath.read_text(encoding="utf-8"))
            hits, sub_hits = set(), 0
            for v, completion in completions.items():
                subs = parse_llm_atoms(v, completion or "") or []
                ok = []
                for s in subs:
                    sn = pnorm(s["sub_atom"])
                    if (
                        sn in index_norms
                        or ((uv := us_variant(sn)) and uv in index_norms)
                        or any(m in index_norms for m in morph_variants(sn))
                    ):
                        ok.append(sn)
                if subs and ok:
                    # Resolved only when every sub-atom links
                    sub_hits += len(ok)
                    if len(ok) == len(subs):
                        hits.add(pnorm(v))
        take("llm_atomiser", hits)
        costs[-1]["sub_atoms_linked"] = sub_hits
        costs[-1]["harness"] = "endpoint" if args.endpoint else "transformers"
        if args.endpoint:
            costs[-1]["hits_valid"] = True
            report_llm_note = (
                "llm rung yield measured against the served vllm endpoint "
                "(LFM2.5-350M at localhost:30000, production sampling); "
                "energy is machine-scope codecarbon while the endpoint "
                "serves the rung's calls."
            )
        else:
            # The transformers harness echoes example atoms, so yield is invalid
            costs[-1]["hits_valid"] = False
            report_llm_note = (
                "llm rung cost is measured (throughput/energy of LFM2.5-350M on "
                "this GPU) but its hit count is NOT valid: the transformers "
                "harness degenerates (prompt-example echoes) where the served "
                "endpoint does not; re-measure the rung's yield against the "
                "localhost:30000 endpoint when it is up. Upside is bounded by "
                "the queue mass, and the queue head is the deliberately-excluded "
                "field-misuse values (dorset worthys, association details, …)."
            )
    else:
        report_llm_note = "llm rung skipped"

    res_norms = set(resolved)
    resolved_occ = occ_of(res_norms)
    whole = vals.with_columns(whole_hit=pl.col("norm").is_in(sorted(index_norms)))
    exact_whole_occ = int(whole.filter(pl.col("whole_hit"))["count"].sum())
    route_occ = {
        r: int(vals.filter(pl.col("route") == r)["count"].sum())
        for r in ("null_marker", "semantic_marker", "prose", "cascade")
    }

    per_rung = {}
    for e in costs:
        if "new_occ" in e:
            wh = e["energy_kwh"] * 1000
            per_rung[e["rung"]] = {
                **{k: e[k] for k in ("seconds", "energy_kwh", "emissions_kg", "new_norms", "new_occ")},
                "wh_per_1k_occ": round(wh / (e["new_occ"] / 1000), 4) if e["new_occ"] else None,
            }

    report = {
        "field": field,
        "seed_terms": int(index["subject"].n_unique()),
        "total_occurrences": total_occ,
        "route_occurrences": route_occ,
        "pilot_exact_norm_whole_value": {"occ": exact_whole_occ, "share": round(exact_whole_occ / total_occ, 4)},
        "cascade_atom_occurrences": atom_occ,
        "resolved_atom_occurrences": resolved_occ,
        "cascade_coverage_atom_occ": round(resolved_occ / atom_occ, 4),
        "llm_queue": {"atoms": llm_queue.height, "occ": int(llm_queue["occ"].sum())},
        "llm_note": report_llm_note,
        "rungs": per_rung,
        "unit_cost": {
            "total_seconds": round(sum(e["seconds"] for e in costs), 1),
            "total_kwh": round(sum(e["energy_kwh"] for e in costs), 6),
            "wh_per_1k_occ_resolved": round(sum(e["energy_kwh"] for e in costs) * 1000 / (resolved_occ / 1000), 4)
            if resolved_occ
            else None,
        },
    }

    fs = (
        pl.scan_parquet(FIELD_STATS)
        .filter(pl.col("value").is_not_null())
        .group_by("field_type")
        .agg(occ=pl.len(), insts=pl.col("data_source").n_unique())
        .collect(engine="streaming")
    )
    corpus_occ = int(fs["occ"].sum())
    vocab_fields = {name for names in all_vocab_fields().values() for name in names}
    mapped = {"spectrum/" + f for f in json.loads(VOCAB_MAP.read_text())}
    admitted = fs.filter(
        pl.col("field_type").is_in(sorted(vocab_fields - mapped - TERMLIST_EXCLUDED))
        & (pl.col("occ") >= ADMIT_OCC)
        & (pl.col("insts") >= ADMIT_INST)
    ).sort("occ", descending=True)
    touched = None
    if CENSUS.exists():
        # The `worked` states must match reports.py coverage exactly
        cen = pl.read_parquet(CENSUS)
        per_field = (
            cen.group_by("field_type")
            .agg(
                total=pl.col("occurrences").sum(),
                worked=pl.col("occurrences")
                .filter(pl.col("disposition").is_in(["applied", "qualified", "deferred"]))
                .sum(),
            )
            .with_columns(share=pl.col("worked") / pl.col("total"))
        )
        stage_fields = per_field.filter(pl.col("share") >= OWNED_MIN_SHARE).select("field_type")
        touched_occ = int(fs.join(stage_fields, on="field_type", how="semi")["occ"].sum())
        touched = {
            "owned_min_share": OWNED_MIN_SHARE,
            "fields": stage_fields.height,
            "occ": touched_occ,
            "share": round(touched_occ / corpus_occ, 4),
        }
        # A field some stage already owns is no gap
        admitted = admitted.join(stage_fields, on="field_type", how="anti")
    report["target_restatement"] = {
        "definition": (
            "share of non-null occurrence mass in fields a "
            "value-level stage owns (>= owned_min_share of the "
            "field's occurrences left `untouched` in "
            "coverage_census); termlist-admissible = ControlledVocab "
            "fields (data model) not already cascaded or owned, "
            "N>=100k occ, M>=10 institutions, minus the "
            "2026-07-14/16 exclusions"
        ),
        "corpus_occurrences": corpus_occ,
        "currently_touched": touched,
        "admissible_termlist_fields": [
            {"field": f, "occ": int(o), "institutions": int(i)} for f, o, i in admitted.iter_rows()
        ],
        "admissible_termlist_occ": int(admitted["occ"].sum()),
        "restated_target_share": round(
            ((touched["occ"] if touched else 0) + int(admitted["occ"].sum())) / corpus_occ, 4
        )
        if touched
        else None,
    }

    out = OUT_DIR / f"{field}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(json.dumps({k: report[k] for k in ("cascade_coverage_atom_occ", "unit_cost")}, indent=2))
    log(f"done → {out}")


if __name__ == "__main__":
    main()
