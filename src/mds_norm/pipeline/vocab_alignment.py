from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections.abc import Callable

import numpy as np
import polars as pl
from codecarbon import EmissionsTracker
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

from mds_norm.paths import EMISSIONS_LOG, FIELD_STATS, VOCAB_ANNOTATIONS, VOCAB_DECISIONS, VOCAB_OUT, VOCABS
from mds_norm.pipeline.apply_homograph_verdicts import RUNG_VERDICTS, apply_verdicts
from mds_norm.pipeline.institutional_vocab_detect import BUILDERS as HOUSE_BUILDERS
from mds_norm.pipeline.vocab_indexes import (
    CASCADE_FIELDS,
    CONCEPT_GROUPS,
    GROUP_FOR,
    GROUP_VOCABS,
    KIND_PRIORITY,
    PLACE_GROUPS,
    english_only,
    group_index,
    house_index,
    periodo_spatial,
    tgn_children,
    tgn_spatial,
)
from mds_norm.pipeline.vocab_rerank import rerank_tier
from mds_norm.utils.atomise import (
    ATOM_PROSE_MAX_CHARS,
    ATOM_PROSE_MAX_TOKENS,
    BASE_SEPARATORS,
    COMPOUND_FIELDS,
    DATE_LIKE,
    GROUP_DESC,
    LLM_COVERAGE,
    LLM_MAX_NEW_TOKENS,
    LLM_PROMPT,
    NO_ATOMISE_FIELDS,
    NULL_MARKERS,
    PLACEHOLDER_MARKERS,
    PROSE_MAX_CHARS,
    PROSE_MAX_TOKENS,
    SEMANTIC_MARKERS,
    WORD_SEPARATORS,
    atomise,
    compound_head,
    morph_variants,
    norm_term,
    parse_llm_atoms,
    separator_literal,
    separator_regex,
    separators_regex,
    us_variant,
)
from mds_norm.utils.masking import MASK

REVIEW_SAMPLE = VOCAB_OUT / "vocab_review_sample.csv"
RESIDUE_WORKLIST = VOCAB_OUT / "vocab_residue_worklist.parquet"
REPORT = VOCAB_OUT / "vocab_report.json"
# Every scored (group, institution, separator) candidate with its evidence
SEPARATORS = VOCAB_OUT / "induced_separators.parquet"
# Every candidate separator with the record support behind it
SEPARATOR_CANDIDATES = VOCAB_OUT / "separator_candidates.parquet"

COMPONENT = "vocab_alignment"
# House rungs carry their pooled counterparts' confidences
CONFIDENCE = {
    "exact": 0.97,
    "exact_variant": 0.93,
    "exact_morph": 0.88,
    "exact_paren": 0.85,
    "exact_compound": 0.85,
    "house_exact": 0.97,
    "house_variant": 0.93,
    "house_morph": 0.88,
    "house_paren": 0.85,
    "house_compound": 0.85,
    "fuzzy": 0.85,
    "rerank": 0.7,
    "llm_exact": 0.75,
    "llm_variant": 0.72,
    "llm_morph": 0.7,
}
RESOLVED_BY_FACTOR = {"unique": 1.0, "kind_tier": 0.95, "spatial": 0.92, "prominent": 0.92, "llm_choice": 1.0}

# Delimiter induction (fragment attestation)
MIN_ATTEST_COUNT = 5  # occurrences for a whole value to count as attested
ATTEST_THRESHOLD = 0.5  # record-weighted share of fragments that must be attested
MIN_SPLIT_FRAGMENTS = 2  # a value splitting into fewer parts is no evidence
MIN_DISTINCT_FRAGMENTS = 10
MIN_SUPPORT_RECORDS = 200  # institution records containing the separator
# A literal below the per-institution floor can never pass
MIN_LITERAL_SUPPORT = MIN_SUPPORT_RECORDS
CONNECTIVE_MIN_RECORDS = 200  # a word is a connective when frequent inside values ...
CONNECTIVE_MAX_WHOLE_RATIO = 0.02  # ... yet almost never a whole value
MAX_ROUNDS = 3  # rescore on remaining fragments until nothing new passes

# Homograph ladder
PROMINENCE_MIN = 10
PROMINENCE_RATIO = 3
DISAMBIG_SORT = {
    "by": ["preference", "pref_tiebreak", "prominence", "lang_p", "subject"],
    "descending": [False, False, True, False, False],
}

PAREN = r"^(.+?) ?\((.+)\)$"

FUZZY_ACCEPT = 93.0
FUZZY_MIN_LEN = 4
FUZZY_CHUNK = 2_000

LLM_MODEL = "LFM2.5-350M"
# Mechanical output, so the small model on its own port
LLM_API_BASE = "http://localhost:30001/v1"
LLM_CONCURRENCY = 256
# https://docs.vllm.ai/projects/recipes/en/latest/LiquidAI/LFM2.5.html#recommended-sampling
LLM_TEMPERATURE = 0.1
LLM_EXTRA_BODY = {"top_k": 50, "repetition_penalty": 1.05}

REVIEW_N = 12
COVERAGE_TARGET = 0.85

HIT_SCHEMA = {
    "norm": pl.String,
    "vocab": pl.String,
    "subject": pl.String,
    "matched_term": pl.String,
    "n_candidates": pl.UInt32,
    "resolved_by": pl.String,
    "group": pl.String,
    "sub_component": pl.String,
    "qualifier": pl.String,
    "score": pl.Float64,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def no_hits() -> pl.DataFrame:
    """An empty tier result — carries the full hit schema so assembly never loses a column"""
    return pl.DataFrame(schema=HIT_SCHEMA)


def value_frame(field_stats: pl.LazyFrame) -> pl.LazyFrame:
    """Every cascade-field occurrence, whitespace-normalised and tagged with its group"""
    return (
        field_stats.filter(pl.col("field_type").is_in(CASCADE_FIELDS) & pl.col("value").is_not_null())
        .select("record_id", "node_id", "data_source", "field_type", "value")
        .with_columns(pl.col("value").str.strip_chars().str.replace_all(r"\s+", " "))
        .filter(pl.col("value") != "")
        .with_columns(group=pl.col("field_type").replace_strict(GROUP_FOR, return_dtype=pl.String))
    )


def distinct_values(values: pl.LazyFrame) -> pl.DataFrame:
    """One row per (group, institution, value), with the two atomiser eligibility flags"""
    return (
        values.group_by("group", "data_source", "value")
        .agg(
            count=pl.len(),
            split_ok=(~pl.col("field_type").is_in(sorted(NO_ATOMISE_FIELDS))).all(),
            compound_ok=pl.col("field_type").is_in(sorted(COMPOUND_FIELDS)).all(),
        )
        .collect(engine="streaming")
    )


def route_values(distinct: pl.DataFrame) -> pl.DataFrame:
    """null_marker / semantic_marker / bilingual / prose / cascade per whole value"""
    return distinct.with_columns(norm=norm_term(pl.col("value"))).with_columns(
        # A knowledge-state marker is kept as recorded but never cascaded
        route=pl.when(pl.col("norm").is_in(sorted(PLACEHOLDER_MARKERS)) | (pl.col("norm") == ""))
        .then(pl.lit("null_marker"))
        .when(pl.col("norm").is_in(sorted(SEMANTIC_MARKERS)))
        .then(pl.lit("semantic_marker"))
        # Amgueddfa Cymru's single-pipe values are one concept written twice
        .when(
            (pl.col("data_source") == "Amgueddfa Cymru - Museum Wales")
            & (pl.col("value").str.count_matches(r"\|") == 1)
            & pl.col("value").str.contains(r"\S\s*\|\s*\S")
        )
        .then(pl.lit("bilingual"))
        .when(
            pl.col("group").is_in(sorted(CONCEPT_GROUPS))
            & (
                (pl.col("value").str.len_chars() > PROSE_MAX_CHARS)
                | (pl.col("value").str.split(" ").list.len() > PROSE_MAX_TOKENS)
            )
        )
        .then(pl.lit("prose"))
        .otherwise(pl.lit("cascade"))
    )


PAIRING_MARKS = r"[()\[\]{}\"'«»]"  # open or close a qualifier, never join a list
# Tight marks join compounds, so only spaced ones are candidates
TIGHT_MARKS = ("-", "'", "\u2019")
FULL_STOP = "."
# The certainty stage reads it; a split would drop it
UNCERTAINTY_MARK = "?"
SEP = "\x1f"


def discover_candidates(concept_vals: pl.DataFrame) -> pl.DataFrame:
    """Per group, every literal run between two content runs and every connective word, with record support"""
    literals = (
        concept_vals.select("group", "data_source", "value", "count", masked=MASK)
        .with_columns(pieces=pl.col("masked").str.replace_all(r"[ds]+", SEP).str.split(SEP))
        # only pieces between two content runs can separate
        .filter(pl.col("pieces").list.len() > 1)
        .with_columns(pieces=pl.col("pieces").list.slice(1, pl.col("pieces").list.len() - 2))
        .with_columns(pieces=pl.col("pieces").list.eval(pl.element().str.replace_all(r"\s+", " ")))
        .explode("pieces", empty_as_null=True)
        .with_columns(stripped=pl.col("pieces").str.strip_chars())
        # spacing is part of a hyphen's identity, not others'
        .with_columns(
            separator=pl.when(pl.col("stripped").is_in(TIGHT_MARKS) & (pl.col("pieces") != pl.col("stripped")))
            .then(" " + pl.col("stripped") + " ")
            .otherwise(pl.col("stripped"))
        )
        .unique(subset=["group", "data_source", "value", "separator"])
        # letters outside A-Z are unmasked and surface as literals
        .filter(~pl.col("separator").str.contains(r"[\p{L}\p{N}]"))
        .group_by("group", "separator")
        .agg(records=pl.col("count").sum(), n_values=pl.len())
        .with_columns(kind=pl.when(pl.col("separator") == "").then(pl.lit("whitespace")).otherwise(pl.lit("literal")))
    )
    tokens = (
        concept_vals.select("group", "count", tok=pl.col("value").str.to_lowercase().str.extract_all(r"[a-z]{2,}"))
        .with_columns(tok=pl.col("tok").list.unique())
        .explode("tok", empty_as_null=True)
        .drop_nulls("tok")
        .group_by("group", "tok")
        .agg(records=pl.col("count").sum(), n_values=pl.len())
    )
    whole_values = concept_vals.group_by("group", "norm").agg(whole_records=pl.col("count").sum())
    connectives = (
        tokens.join(whole_values, left_on=["group", "tok"], right_on=["group", "norm"], how="left")
        .with_columns(pl.col("whole_records").fill_null(0))
        .filter(
            (pl.col("records") >= CONNECTIVE_MIN_RECORDS)
            & (pl.col("whole_records") <= CONNECTIVE_MAX_WHOLE_RATIO * pl.col("records"))
        )
        .select("group", separator="tok", records="records", n_values="n_values", kind=pl.lit("word"))
    )
    base_rx = "|".join(re.escape(b) for b in BASE_SEPARATORS)
    literal_reason = (
        pl.when(pl.col("records") < MIN_LITERAL_SUPPORT)
        .then(pl.lit("support"))
        .when(pl.col("separator").str.contains(PAIRING_MARKS))
        .then(pl.lit("pairing mark"))
        .when(pl.col("separator").str.contains(base_rx))
        .then(pl.lit("base separator"))
        .when(pl.col("separator").is_in(TIGHT_MARKS) | pl.col("separator").str.contains(FULL_STOP, literal=True))
        .then(pl.lit("word orthography"))
        .when(pl.col("separator").str.contains(UNCERTAINTY_MARK, literal=True))
        .then(pl.lit("uncertainty mark"))
    )
    candidates = pl.concat([literals, connectives]).with_columns(
        reason=pl.when(pl.col("kind") == "literal")
        .then(literal_reason)
        .when(pl.col("kind") == "word")
        .then(pl.when(~pl.col("separator").is_in(WORD_SEPARATORS)).then(pl.lit("content word")))
        .otherwise(pl.lit("whitespace"))
    )
    candidates = candidates.with_columns(admitted=pl.col("reason").is_null())
    candidates.sort("group", "kind", "records", descending=[False, False, True]).write_parquet(SEPARATOR_CANDIDATES)
    admitted = candidates.filter("admitted")
    log(
        f"{candidates.height:,} separator candidates discovered, {admitted.height:,} admitted "
        f"({admitted['separator'].n_unique()} distinct) → {SEPARATOR_CANDIDATES}"
    )
    return admitted.select("group", "separator")


def _fragments(rx: re.Pattern) -> Callable[[str], list[str]]:
    return lambda v: [p.strip() for p in rx.split(v) if p.strip()]


def split_on_accepted(vals: pl.DataFrame, accepted: pl.DataFrame) -> pl.DataFrame:
    """The values as the accepted separators leave them: each fragment carries its value's record count"""
    if accepted.is_empty():
        return vals
    sep_sets = accepted.group_by("group", "data_source").agg(seps=pl.col("separator").sort())
    keyed = vals.join(sep_sets, on=["group", "data_source"], how="left")
    untouched = keyed.filter(pl.col("seps").is_null()).drop("seps")
    frames = [untouched]
    for (seps,), part in keyed.filter(pl.col("seps").is_not_null()).group_by("seps"):
        rx = re.compile(separators_regex(seps))
        frames.append(
            part.drop("seps")
            .with_columns(value=pl.col("value").map_elements(_fragments(rx), return_dtype=pl.List(pl.String)))
            .explode("value", empty_as_null=True)
            .drop_nulls("value")
        )
    return pl.concat(frames)


def score_candidates(vals: pl.DataFrame, candidates: pl.DataFrame, attested: pl.DataFrame) -> pl.DataFrame:
    """Fragment attestation of every (group, institution, candidate) over the values as currently split"""
    split_frames = []
    for group, sep in candidates.select("group", "separator").unique().iter_rows():
        lit = separator_literal(sep)
        gate = f" {lit} " if lit.isalpha() else lit
        subset = vals.filter((pl.col("group") == group) & pl.col("value").str.contains(gate, literal=True))
        if subset.is_empty():
            continue
        rx = re.compile(separator_regex(sep))
        subset = subset.with_columns(
            separator=pl.lit(sep), frags=pl.col("value").map_elements(_fragments(rx), return_dtype=pl.List(pl.String))
        ).filter(pl.col("frags").list.len() >= MIN_SPLIT_FRAGMENTS)
        split_frames.append(subset.select("group", "data_source", "count", "separator", "frags"))
    if not split_frames:
        return pl.DataFrame()

    candidate_splits = pl.concat(split_frames)
    exploded = (
        candidate_splits.explode("frags", empty_as_null=True)
        .rename({"frags": "frag"})
        .with_columns(frag_norm=norm_term(pl.col("frag")))
        .join(attested.with_columns(attested=pl.lit(True)), on=["group", "frag_norm"], how="left")
        .with_columns(pl.col("attested").fill_null(False))
    )
    return (
        candidate_splits.group_by("group", "data_source", "separator")
        .agg(support_records=pl.col("count").sum(), frags_total=(pl.col("frags").list.len() * pl.col("count")).sum())
        .join(
            exploded.group_by("group", "data_source", "separator").agg(
                frags_attested=(pl.col("attested").cast(pl.UInt32) * pl.col("count")).sum(),
                n_fragments=pl.col("frag_norm").filter(pl.col("attested")).n_unique(),
            ),
            on=["group", "data_source", "separator"],
        )
        .with_columns(attestation=pl.col("frags_attested") / pl.col("frags_total"))
        .with_columns(
            is_separator=(pl.col("attestation") >= ATTEST_THRESHOLD)
            & (pl.col("n_fragments") >= MIN_DISTINCT_FRAGMENTS)
            & (pl.col("support_records") >= MIN_SUPPORT_RECORDS)
        )
    )


def induce_separators(whole: pl.DataFrame) -> pl.DataFrame:
    """The (group, institution, separator) triples fragment attestation certifies"""
    concept_vals = whole.filter((pl.col("route") == "cascade") & pl.col("group").is_in(sorted(CONCEPT_GROUPS)))
    candidates = discover_candidates(concept_vals)
    attested = pl.concat(
        [
            concept_vals.group_by("group", "norm")
            .agg(pl.col("count").sum())
            .filter(pl.col("count") >= MIN_ATTEST_COUNT)
            .select("group", frag_norm="norm"),
            *[
                candidates.filter(pl.col("group") == g)
                .select("group")
                .unique()
                .lazy()
                .join(group_index(g).select(frag_norm="norm").unique(), how="cross")
                .collect(engine="streaming")
                for g in sorted(CONCEPT_GROUPS)
            ],
        ]
    ).unique()

    vals = concept_vals.select("group", "data_source", "value", "count")
    accepted = pl.DataFrame(
        schema={"group": pl.String, "data_source": vals.schema["data_source"], "separator": pl.String}
    )
    rounds = []
    for round_no in range(1, MAX_ROUNDS + 1):
        # Two separators attest neither alone, so rescore on the fragments
        scores = score_candidates(split_on_accepted(vals, accepted), candidates, attested)
        if scores.is_empty():
            break
        scores = scores.join(
            accepted.with_columns(done=pl.lit(True)), on=["group", "data_source", "separator"], how="left"
        )
        scores = (
            scores.filter(pl.col("done").is_null()).drop("done").with_columns(round=pl.lit(round_no, dtype=pl.Int8))
        )
        rounds.append(scores)
        new = scores.filter("is_separator").select("group", "data_source", "separator")
        log(f"round {round_no}: {new.height} acceptances of {scores.height:,} scored candidates")
        if new.is_empty():
            break
        accepted = pl.concat([accepted, new])

    VOCAB_OUT.mkdir(parents=True, exist_ok=True)
    all_scores = pl.concat(rounds)
    # a candidate keeps only its final round's verdict
    all_scores = all_scores.sort("round", descending=True).unique(
        subset=["group", "data_source", "separator"], keep="first"
    )
    all_scores.sort("group", "data_source", "separator").write_parquet(SEPARATORS)
    log(
        f"{accepted.height} (group, institution, separator) acceptances of {all_scores.height:,} scored → {SEPARATORS}"
    )
    return accepted


def atomise_values(whole: pl.DataFrame, accepted: pl.DataFrame) -> pl.DataFrame:
    """Split the cascade values into atoms with char spans, then re-guard each atom"""
    cascade_vals = whole.filter(pl.col("route") == "cascade")
    bilingual = whole.filter(pl.col("route") == "bilingual").with_columns(
        en_atom=pl.col("value").str.split("|").list.last().str.strip_chars(),
        norm=norm_term(pl.col("value").str.split("|").list.last().str.strip_chars()),
    )

    whole_hits = pl.concat(
        [
            cascade_vals.filter(pl.col("group") == g)
            .lazy()
            .join(group_index(g).select("norm").unique(), on="norm", how="semi")
            .collect(engine="streaming")
            for g in GROUP_VOCABS
        ]
    )

    sep_sets = accepted.group_by("group", "data_source").agg(seps=pl.col("separator").sort())
    # a cheap gate before python splitting: any known literal
    literals = {separator_literal(s) for s in accepted["separator"]} | set(BASE_SEPARATORS)
    prefilter = "|".join(re.escape(f" {lit} " if lit.isalpha() else lit) for lit in sorted(literals))
    to_split = (
        cascade_vals.filter(
            pl.col("group").is_in(sorted(CONCEPT_GROUPS))
            & pl.col("split_ok")
            & pl.col("value").str.contains(prefilter)
        )
        .join(whole_hits.select("group", "data_source", "value"), on=["group", "data_source", "value"], how="anti")
        .join(sep_sets, on=["group", "data_source"], how="left")
    )
    no_split = whole.filter(pl.col("route") != "bilingual").join(
        to_split.select("group", "data_source", "value"), on=["group", "data_source", "value"], how="anti"
    )

    atoms = (
        pl.concat(
            [
                no_split.select(
                    "group",
                    "data_source",
                    "value",
                    "count",
                    "route",
                    "split_ok",
                    "compound_ok",
                    atom=pl.col("value"),
                    span_start=pl.lit(0, dtype=pl.Int64),
                    span_end=pl.col("value").str.len_chars().cast(pl.Int64),
                ),
                # Bilingual: the English half is the atom, spanning the value
                bilingual.select(
                    "group",
                    "data_source",
                    "value",
                    "count",
                    route=pl.lit("cascade"),
                    split_ok=pl.col("split_ok"),
                    compound_ok=pl.col("compound_ok"),
                    atom=pl.col("en_atom"),
                    span_start=pl.lit(0, dtype=pl.Int64),
                    span_end=pl.col("value").str.len_chars().cast(pl.Int64),
                ),
                to_split.select(
                    "group",
                    "data_source",
                    "value",
                    "count",
                    "route",
                    "split_ok",
                    "compound_ok",
                    parts=pl.struct(["value", "seps"]).map_elements(
                        lambda row: atomise(row["value"], row["seps"]),
                        return_dtype=pl.List(
                            pl.Struct({"atom": pl.String, "span_start": pl.Int64, "span_end": pl.Int64})
                        ),
                    ),
                )
                .explode("parts", empty_as_null=True)
                .unnest("parts"),
            ]
        )
        .with_columns(
            # A value of only separators explodes to a null atom
            empty_split=pl.col("atom").is_null(),
            atom=pl.col("atom").fill_null(pl.col("value")),
            span_start=pl.col("span_start").fill_null(0),
            span_end=pl.col("span_end").fill_null(pl.col("value").str.len_chars().cast(pl.Int64)),
        )
        .with_columns(norm=norm_term(pl.col("atom")))
    )

    # Null markers hide inside lists, and long atoms are sentences
    atoms = atoms.with_columns(
        atom_route=pl.when(pl.col("route") != "cascade")
        .then(pl.col("route"))
        .when(pl.col("empty_split") | (pl.col("norm") == "") | pl.col("norm").is_in(sorted(NULL_MARKERS)))
        .then(pl.lit("null_marker"))
        .when(
            pl.col("group").is_in(sorted(CONCEPT_GROUPS))
            & (
                (pl.col("atom").str.len_chars() > ATOM_PROSE_MAX_CHARS)
                | (pl.col("atom").str.split(" ").list.len() > ATOM_PROSE_MAX_TOKENS)
            )
        )
        .then(pl.lit("prose"))
        .otherwise(pl.lit("cascade"))
    ).drop("route", "empty_split")

    log(f"{len(whole):,} distinct (group, institution, value) rows → {len(atoms):,} atoms ({len(to_split):,} split)")
    return atoms


def resolve_norms(
    index: pl.LazyFrame,
    norms: pl.DataFrame | None,
    prominence: pl.LazyFrame | None = None,
    preference: pl.LazyFrame | None = None,
) -> pl.DataFrame:
    """Best (vocab, subject, term) per norm, plus how any homograph was settled"""
    lf = index if norms is None else index.join(norms.lazy(), on="norm", how="semi")
    cand = (
        lf.with_columns(
            kind_p=pl.col("kind").replace_strict(KIND_PRIORITY, return_dtype=pl.Int8),
            lang_p=(~pl.col("lang").str.starts_with("en")).cast(pl.Int8).fill_null(1),
        )
        .group_by("norm", "vocab", "vocab_priority", "subject")
        .agg(
            kind_p=pl.col("kind_p").min(),
            lang_p=pl.col("lang_p").min(),
            matched_term=pl.col("term").sort_by(["kind_p", "lang_p"]).first(),
        )
    )
    cand = (
        cand.join(prominence, on="subject", how="left")
        if prominence is not None
        else cand.with_columns(prominence=pl.lit(0, dtype=pl.UInt32))
    )
    cand = (
        cand.join(preference, on="subject", how="left")
        if preference is not None
        else cand.with_columns(preference=pl.lit(0, dtype=pl.Int8), pref_tiebreak=pl.lit(0, dtype=pl.Int32))
    )
    cand = cand.with_columns(
        pl.col("prominence").fill_null(0), pl.col("preference").fill_null(2), pl.col("pref_tiebreak").fill_null(0)
    )
    key = ["norm", "vocab", "vocab_priority"]
    counts = cand.group_by(key).agg(
        n_candidates=pl.col("subject").n_unique().cast(pl.UInt32), pref_min=pl.col("preference").min()
    )
    # The prior speaks first: Getty prefers a small settlement elsewhere
    narrowed = cand.join(counts, on=key).filter(pl.col("preference") == pl.col("pref_min"))
    kinds = narrowed.group_by(key).agg(n_pref_best=pl.col("subject").n_unique(), min_kind=pl.col("kind_p").min())

    return (
        narrowed.join(kinds, on=key)
        .filter(pl.col("kind_p") == pl.col("min_kind"))
        .group_by(key)
        .agg(
            n_candidates=pl.col("n_candidates").first(),
            n_best=pl.col("subject").n_unique(),
            subject=pl.col("subject").sort_by(**DISAMBIG_SORT).first(),
            matched_term=pl.col("matched_term").sort_by(**DISAMBIG_SORT).first(),
            n_pref_best=pl.col("n_pref_best").first(),
            pref_min=pl.col("pref_min").first(),
            prom_top=pl.col("prominence").sort(descending=True).head(2),
        )
        .with_columns(
            prom0=pl.col("prom_top").list.get(0, null_on_oob=True).fill_null(0),
            prom1=pl.col("prom_top").list.get(1, null_on_oob=True).fill_null(0),
        )
        .with_columns(
            # Rungs in the order their measured precision earns
            resolved_by=pl.when(pl.col("n_candidates") == 1)
            .then(pl.lit("unique"))
            .when(
                pl.lit(prominence is not None)
                & (pl.col("prom0") >= PROMINENCE_MIN)
                & (pl.col("prom0") >= PROMINENCE_RATIO * (pl.col("prom1") + 1))
            )
            .then(pl.lit("prominent"))
            .when(pl.lit(preference is not None) & ((pl.col("n_pref_best") == 1) | (pl.col("pref_min") == 0)))
            .then(pl.lit("spatial"))
            .when(pl.col("n_best") == 1)
            .then(pl.lit("kind_tier"))
        )
        .sort("vocab_priority")
        .unique("norm", keep="first", maintain_order=True)
        .select("norm", "vocab", "subject", "matched_term", "n_candidates", "resolved_by")
        .collect(engine="streaming")
    )


def disambiguators(group: str) -> dict[str, pl.LazyFrame]:
    kw = {}
    if group in PLACE_GROUPS:
        kw["prominence"] = tgn_children()
        kw["preference"] = tgn_spatial()
    if "periodo" in GROUP_VOCABS[group]:
        kw["preference"] = periodo_spatial()
    return kw


def pending_atoms(cascade_atoms: pl.DataFrame, hits: list[pl.DataFrame]) -> pl.DataFrame:
    matched = pl.concat([f.select("group", "norm") for f in hits]).unique()
    return cascade_atoms.join(matched, on=["group", "norm"], how="anti")


def variant_hits(atoms: pl.DataFrame, index: pl.LazyFrame) -> pl.DataFrame:
    """GB→US respelling retry, anchored on the vocabulary: a variant counts only if it matches"""
    norms = atoms.select("norm").unique()["norm"]
    pairs = pl.DataFrame({"norm": norms, "variant": [us_variant(n) if n else None for n in norms]}).drop_nulls(
        "variant"
    )
    if not len(pairs):
        return no_hits()
    lookup = resolve_norms(index, pairs.select(norm=pl.col("variant")).unique())
    return pairs.join(lookup.rename({"norm": "variant"}), on="variant").drop("variant")


def paren_hits(atoms: pl.DataFrame, index: pl.LazyFrame) -> pl.DataFrame:
    """`metal (unknown)`, `film (photographic)`: match the head, keep the qualifier"""
    heads = (
        atoms.select("norm")
        .unique()
        .with_columns(head=pl.col("norm").str.extract(PAREN, 1), qualifier=pl.col("norm").str.extract(PAREN, 2))
        .drop_nulls("head")
        .with_columns(head=norm_term(pl.col("head")))
        .filter(pl.col("head") != "")
    )
    if not len(heads):
        return no_hits()
    lookup = resolve_norms(index, heads.select(norm=pl.col("head")).unique())
    return heads.join(lookup.rename({"norm": "head"}), on="head").drop("head")


def morph_hits(atoms: pl.DataFrame, index: pl.LazyFrame) -> pl.DataFrame:
    """`engraved` → `engraving`: the deverbal noun the vocabularies actually list"""
    pairs = (
        atoms.select("norm")
        .unique()
        .with_columns(variant=pl.col("norm").map_elements(morph_variants, return_dtype=pl.List(pl.String)))
        .explode("variant")
        .drop_nulls("variant")
        .with_row_index("prio")
    )
    if not len(pairs):
        return no_hits()
    lookup = resolve_norms(index, pairs.select(norm=pl.col("variant")).unique())
    return (
        pairs.join(lookup.rename({"norm": "variant"}), on="variant")
        .sort("prio")
        .unique("norm", keep="first")
        .drop("variant", "prio")
    )


def compound_hits(atoms: pl.DataFrame, index: pl.LazyFrame) -> pl.DataFrame:
    """`engraving on paper`: read the head, keep the rest as the qualifier"""
    parts = (
        atoms.group_by("norm")
        .agg(compound_ok=pl.col("compound_ok").all())
        .filter("compound_ok")
        .with_columns(parts=pl.col("norm").map_elements(compound_head, return_dtype=pl.List(pl.String)))
        .drop_nulls("parts")
        .select(
            "norm",
            head=pl.col("parts").list.get(0),
            tail=pl.col("parts").list.get(1),
            qualifier=pl.col("parts").list.get(2),
        )
    )
    if not len(parts):
        return no_hits()
    attested = index.select("norm").unique().collect(engine="streaming")["norm"]
    parts = parts.filter(pl.col("tail").is_in(attested.implode()))
    if not len(parts):
        return no_hits()
    lookup = resolve_norms(index, parts.select(norm=pl.col("head")).unique())
    return parts.join(lookup.rename({"norm": "head"}), on="head").drop("head", "tail")


# A retry counts only if the rewrite exact-matches
RETRIES = (("variant", variant_hits), ("paren", paren_hits), ("morph", morph_hits), ("compound", compound_hits))
POOLED_NAMES = {step: f"exact_{step}" for step, _ in RETRIES} | {"exact": "exact"}
HOUSE_NAMES = {step: f"house_{step}" for step, _ in RETRIES} | {"exact": "house_exact"}


def ladder(
    atoms: pl.DataFrame, index: pl.LazyFrame, names: dict[str, str], retries: bool, **disambig: pl.LazyFrame
) -> list[pl.DataFrame]:
    """Verbatim exact, then each anchored retry over what the tiers before it left"""
    hits = [
        resolve_norms(index, atoms.select("norm").unique(), **disambig).with_columns(
            sub_component=pl.lit(names["exact"])
        )
    ]
    for step, retry in RETRIES if retries else ():
        matched = pl.concat([h.select("norm") for h in hits]).unique()
        remaining = atoms.join(matched, on="norm", how="anti")
        if not len(remaining):
            break
        hits.append(retry(remaining, index).with_columns(sub_component=pl.lit(names[step])))
    return [h for h in hits if len(h)]


def pooled_ladder(cascade_atoms: pl.DataFrame) -> list[pl.DataFrame]:
    """The ladder against each group's own vocabularies, keyed on (group, norm)"""
    frames = []
    for g in GROUP_VOCABS:
        atoms = cascade_atoms.filter(pl.col("group") == g)
        if not len(atoms):
            continue
        # Places take the exact tier only: retries rewrite English morphology
        hits = ladder(atoms, group_index(g), POOLED_NAMES, g in CONCEPT_GROUPS, **disambiguators(g))
        frames += [h.with_columns(group=pl.lit(g)) for h in hits]
    return frames


def house_vocabs(prediction: dict, group: str) -> list[str]:
    """The lists credited to one (institution, field), most-evidenced first"""
    # Exclusive evidence is a stronger claim than base coverage
    specific = list(prediction.get("specific_evidence", {}))
    ordered = specific + [v for v in prediction.get("vocabs", []) if v not in specific]
    return [v for v in ordered if v in HOUSE_BUILDERS and v not in GROUP_VOCABS[group]]


def house_ranks() -> pl.DataFrame:
    """(group, institution, vocabulary) the detection credits, in preference order"""
    inst_vocab_map = json.loads((VOCABS / "institution_vocab_map.json").read_text())
    rows = []
    for inst, fields in inst_vocab_map.items():
        for field, pred in fields.items():
            group = GROUP_FOR.get("spectrum/" + field)
            if group is None:
                continue
            rows += [(group, inst, vocab, rank) for rank, vocab in enumerate(house_vocabs(pred, group))]
    return pl.DataFrame(rows, schema=["group", "data_source", "vocab", "house_rank"], orient="row")


def house_tier(cascade_atoms: pl.DataFrame) -> pl.DataFrame:
    """The ladder against the list each institution catalogues from, keyed on (group, institution, norm)"""
    ranks = house_ranks()
    seed = no_hits().with_columns(data_source=pl.lit(None, dtype=pl.String), house_step=pl.lit(None, dtype=pl.Int32))
    frames = [seed]
    for group, vocab in ranks.select("group", "vocab").unique().sort("group", "vocab").iter_rows():
        insts = ranks.filter((pl.col("group") == group) & (pl.col("vocab") == vocab))["data_source"].to_list()
        atoms = cascade_atoms.filter((pl.col("group") == group) & pl.col("data_source").cast(pl.String).is_in(insts))
        if not len(atoms):
            continue
        # The corpus carries institutions as an Enum, house_ranks as strings
        pairs = atoms.select(pl.col("data_source").cast(pl.String), "norm").unique()
        for step, hits in enumerate(ladder(atoms, house_index(vocab), HOUSE_NAMES, group in CONCEPT_GROUPS)):
            frames.append(
                pairs.join(hits, on="norm").with_columns(group=pl.lit(group), house_step=pl.lit(step, dtype=pl.Int32))
            )
    return (
        pl.concat(frames, how="diagonal")
        .join(ranks, on=["group", "data_source", "vocab"])
        .sort("house_rank", "house_step")
        .unique(["group", "data_source", "norm"], keep="first", maintain_order=True)
        .drop("house_rank", "house_step")
    )


def fuzzy_tier(pending: pl.DataFrame) -> pl.DataFrame:
    """Indel ratio against the full English vocabulary of each group"""
    frames = []
    for group in sorted(CONCEPT_GROUPS):
        queries = (
            pending.filter((pl.col("group") == group) & (pl.col("norm").str.len_chars() >= FUZZY_MIN_LEN))
            .select("norm")
            .unique()["norm"]
            .sort()
            .to_list()
        )
        if not queries:
            continue
        # sorted so equal-scoring choices always resolve to the same one
        table = resolve_norms(english_only(group_index(group)), None).sort("norm")
        choices = table["norm"].to_list()
        best_idx, best_score = [], []
        for i in range(0, len(queries), FUZZY_CHUNK):
            scores = cdist(
                queries[i : i + FUZZY_CHUNK],
                choices,
                scorer=fuzz.ratio,
                score_cutoff=FUZZY_ACCEPT,
                workers=-1,
                dtype=np.uint8,
            )
            best_idx.append(scores.argmax(axis=1))
            best_score.append(scores.max(axis=1))
        best_idx, best_score = np.concatenate(best_idx), np.concatenate(best_score)
        frames.append(
            pl.DataFrame(
                {"norm": queries, "match_norm": [choices[j] for j in best_idx], "score": best_score.astype(np.float64)}
            )
            .filter(pl.col("score") >= FUZZY_ACCEPT)
            .join(table.rename({"norm": "match_norm"}), on="match_norm")
            .with_columns(group=pl.lit(group), sub_component=pl.lit("fuzzy"), score=pl.col("score") / 100.0)
            .drop("match_norm")
        )
    return pl.concat(frames) if frames else no_hits()


def llm_queue(pending: pl.DataFrame) -> pl.DataFrame:
    """Multi-word deferred atoms, capped at the values covering 95% of occurrence mass"""
    queue = (
        pending.filter(
            pl.col("group").is_in(sorted(CONCEPT_GROUPS))
            & pl.col("split_ok")  # no-atomise fields never reach the LLM splitter
            & pl.col("atom").str.contains(" ", literal=True)
            & ~((pl.col("group") == "periodo") & pl.col("norm").str.contains(DATE_LIKE))
        )
        .group_by("group", "atom")
        .agg(occ=pl.col("count").sum())
        .sort("occ", descending=True)
        .with_columns(cum_share=(pl.col("occ").cum_sum() / pl.col("occ").sum()).over("group"))
    )
    capped = queue.filter(pl.col("cum_share") <= LLM_COVERAGE)
    log(
        f"LLM queue: {len(capped):,} of {len(queue):,} distinct atoms "
        f"({capped['occ'].sum():,} of {queue['occ'].sum():,} occurrences)"
    )
    return capped


async def llm_atomise(queue: pl.DataFrame) -> pl.DataFrame:
    """Segment each queued atom into verbatim sub-atoms, validated mechanically"""
    from mds_norm.utils.inference import Inference

    inf = Inference(model=LLM_MODEL, base_url=LLM_API_BASE, concurrency=LLM_CONCURRENCY, timeout=600.0)
    samples = [{"desc": GROUP_DESC[g], "value": v} for g, v in zip(queue["group"], queue["atom"], strict=True)]
    with EmissionsTracker(
        project_name="vocab_atomise_llm", output_dir=str(EMISSIONS_LOG), log_level="error", tracking_mode="machine"
    ):
        completions = await inf.generate(
            samples, LLM_PROMPT, max_tokens=LLM_MAX_NEW_TOKENS, temperature=LLM_TEMPERATURE, extra_body=LLM_EXTRA_BODY
        )

    rows = [
        {"group": g, "atom": v, **sub}
        for g, v, c in zip(queue["group"], queue["atom"], completions, strict=True)
        for sub in parse_llm_atoms(v, c or "") or []
    ]
    split = (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "group": pl.String,
                "atom": pl.String,
                "sub_atom": pl.String,
                "sub_start": pl.Int64,
                "sub_end": pl.Int64,
            }
        )
    )
    log(f"{split['atom'].n_unique():,} of {len(queue):,} queued atoms split into {len(split):,} sub-atoms")
    return split.with_columns(sub_norm=norm_term(pl.col("sub_atom")))


def llm_tier(llm_split: pl.DataFrame) -> pl.DataFrame:
    """Sub-atoms re-enter the exact tier plus the two anchored retries; nothing fuzzier"""
    frames = []
    for g in sorted(CONCEPT_GROUPS):
        norms = llm_split.filter(pl.col("group") == g).select(norm=pl.col("sub_norm")).unique()
        if not len(norms):
            continue
        lookup = resolve_norms(group_index(g), norms).with_columns(group=pl.lit(g), sub_component=pl.lit("llm_exact"))
        frames.append(lookup)

        unmatched = norms.join(lookup.select("norm"), on="norm", how="anti")
        pairs = pl.DataFrame(
            {"norm": unmatched["norm"], "variant": [us_variant(n) if n else None for n in unmatched["norm"]]}
        ).drop_nulls("variant")
        if len(pairs):
            variants = resolve_norms(group_index(g), pairs.select(norm=pl.col("variant")).unique())
            frames.append(
                pairs.join(variants.rename({"norm": "variant"}), on="variant")
                .drop("variant")
                .with_columns(group=pl.lit(g), sub_component=pl.lit("llm_variant"))
            )
        morphs = (
            unmatched.with_columns(
                variant=pl.col("norm").map_elements(morph_variants, return_dtype=pl.List(pl.String))
            )
            .explode("variant", empty_as_null=True)
            .drop_nulls("variant")
            .with_row_index("prio")
        )
        if len(morphs):
            lookup = resolve_norms(group_index(g), morphs.select(norm=pl.col("variant")).unique())
            frames.append(
                morphs.join(lookup.rename({"norm": "variant"}), on="variant")
                .sort("prio")
                .unique("norm", keep="first")
                .drop("variant", "prio")
                .with_columns(group=pl.lit(g), sub_component=pl.lit("llm_morph"))
            )
    return pl.concat(frames).unique(["group", "norm"], keep="first") if frames else no_hits()


HOUSE_COLS = ["vocab", "subject", "matched_term", "n_candidates", "resolved_by", "sub_component", "score", "qualifier"]


def house_overlay(base: pl.DataFrame, house_hits: pl.DataFrame) -> pl.DataFrame:
    """House hits displace the pooled match; a verbatim pooled match survives as the crosswalk"""
    if not len(house_hits):
        return base.with_columns(xref_vocab=pl.lit(None, dtype=pl.String), xref_subject=pl.lit(None, dtype=pl.String))
    use_house = pl.col("subject_house").is_not_null()
    crosswalk = use_house & pl.col("subject").is_not_null() & pl.col("sub_component").str.starts_with("exact")
    # Join on a cast copy rather than changing the column
    house = house_hits.rename({c: f"{c}_house" for c in HOUSE_COLS}).rename({"data_source": "data_source_key"})
    return (
        base.with_columns(data_source_key=pl.col("data_source").cast(pl.String))
        .join(house, on=["group", "data_source_key", "norm"], how="left")
        .drop("data_source_key")
        .with_columns(
            xref_vocab=pl.when(crosswalk).then(pl.col("vocab")),
            xref_subject=pl.when(crosswalk).then(pl.col("subject")),
        )
        .with_columns(
            [pl.when(use_house).then(pl.col(f"{c}_house")).otherwise(pl.col(c)).alias(c) for c in HOUSE_COLS]
        )
        .drop([f"{c}_house" for c in HOUSE_COLS])
    )


def assemble(
    atoms: pl.DataFrame,
    tier_frames: list[pl.DataFrame],
    house_hits: pl.DataFrame,
    llm_split: pl.DataFrame,
    llm_processed: pl.DataFrame,
    llm_lookup: pl.DataFrame,
) -> pl.DataFrame:
    """Fold every tier back onto the atom table and stamp the sidecar contract"""
    # The empty frame carries the full hit schema
    tier_hits = pl.concat([no_hits(), *tier_frames], how="diagonal").unique(
        ["group", "norm"], keep="first", maintain_order=True
    )
    base = house_overlay(atoms.join(tier_hits, on=["group", "norm"], how="left"), house_hits)

    # LLM-split parents never hit; replace with their sub-atoms
    marked = base.join(
        llm_split.select("group", "atom").unique().with_columns(rep=pl.lit(True)), on=["group", "atom"], how="left"
    )
    parents = marked.filter(pl.col("rep").is_not_null() & pl.col("subject").is_null() & pl.col("split_ok"))
    kept = marked.filter(pl.col("rep").is_null() | pl.col("subject").is_not_null() | ~pl.col("split_ok")).drop("rep")

    sub_atoms = (
        parents.select(
            "group",
            "data_source",
            "value",
            "count",
            "atom_route",
            "atom",
            "split_ok",
            parent_start=pl.col("span_start"),
        )
        .join(llm_split.drop("sub_norm"), on=["group", "atom"])
        .with_columns(
            atom=pl.col("sub_atom"),
            span_start=pl.col("parent_start") + pl.col("sub_start"),
            span_end=pl.col("parent_start") + pl.col("sub_end"),
            norm=norm_term(pl.col("sub_atom")),
        )
        .drop("sub_atom", "sub_start", "sub_end", "parent_start")
        .join(llm_lookup, on=["group", "norm"], how="left")
        .with_columns(from_llm=pl.lit(True))
    )

    return (
        pl.concat([kept, sub_atoms], how="diagonal")
        .join(llm_processed.with_columns(llm_done=pl.lit(True)), on=["group", "atom"], how="left")
        .with_columns(
            status=pl.when(pl.col("atom_route") == "null_marker")
            .then(pl.lit("rejected"))
            .when(pl.col("subject").is_null())
            .then(pl.lit("deferred"))
            .when(pl.col("resolved_by").is_null())
            .then(pl.lit("flagged"))
            .otherwise(pl.lit("resolved"))
        )
        .with_columns(
            defer_reason=pl.when(pl.col("status") != "deferred")
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("atom_route") == "prose")
            .then(pl.lit("prose"))
            .when(pl.col("atom_route") == "semantic_marker")
            .then(pl.lit("semantic_marker"))
            .when(pl.col("group").is_in(sorted(PLACE_GROUPS)))
            .then(pl.lit("place_pipeline"))
            .when((pl.col("group") == "periodo") & pl.col("norm").str.contains(DATE_LIKE))
            .then(pl.lit("date_parser"))
            .when(pl.col("llm_done").fill_null(False) | pl.col("from_llm").fill_null(False))
            .then(pl.lit("no_match"))
            .when(pl.col("atom").str.contains(" ", literal=True))
            .then(pl.lit("atomiser"))
            .otherwise(pl.lit("no_match")),
            component=pl.lit(COMPONENT),
            tier=pl.when(pl.col("sub_component").str.starts_with("llm") | (pl.col("sub_component") == "rerank"))
            .then(4)
            .when(pl.col("sub_component").is_not_null())
            .then(2),
            confidence=pl.when(pl.col("status") == "resolved").then(
                pl.col("sub_component").replace_strict(CONFIDENCE, return_dtype=pl.Float64, default=None)
                * pl.col("resolved_by").replace_strict(RESOLVED_BY_FACTOR, return_dtype=pl.Float64, default=1.0)
            ),
        )
        .drop("llm_done", "from_llm")
    )


def write_sidecar(values: pl.LazyFrame, decisions: pl.DataFrame) -> int:
    """Join the decisions back onto every occurrence and sink straight to parquet"""
    (
        values.join(
            decisions.drop("count", "norm", "atom_route", "split_ok", "compound_ok").lazy(),
            on=["group", "data_source", "value"],
            how="left",
        ).sink_parquet(VOCAB_ANNOTATIONS)
    )
    return pl.scan_parquet(VOCAB_ANNOTATIONS).select(pl.len()).collect(engine="streaming").item()


def coverage(annotations: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Occurrence- and distinct-weighted linkage, per field and per institution"""
    active = annotations.filter(pl.col("status") != "rejected")
    per_field = (
        active.group_by("field_type")
        .agg(
            occurrences=pl.len(),
            resolved=(pl.col("status") == "resolved").mean(),
            flagged=(pl.col("status") == "flagged").mean(),
            deferred=(pl.col("status") == "deferred").mean(),
        )
        .sort("resolved")
        .collect(engine="streaming")
        .with_columns(meets_target=pl.col("resolved") >= COVERAGE_TARGET)
    )
    per_institution = (
        active.group_by("data_source", "group")
        .agg(occurrences=pl.len(), resolved_occ=(pl.col("status") == "resolved").mean())
        .collect(engine="streaming")
        .join(
            active.unique(["data_source", "group", "value"])
            .group_by("data_source", "group")
            .agg(distinct=pl.len(), resolved_distinct=(pl.col("status") == "resolved").mean())
            .collect(engine="streaming"),
            on=["data_source", "group"],
        )
        .sort("resolved_occ")
    )
    return per_field, per_institution


def residue_worklist(decisions: pl.DataFrame) -> pl.DataFrame:
    """Deferred atoms grouped by shape — what the next normalisation pass acts on"""
    return (
        decisions.filter(pl.col("status") == "deferred")
        .with_columns(shape=pl.col("atom").str.replace_all(r"[A-Za-z]+", "s").str.replace_all(r"\d+", "d"))
        .group_by("group", "defer_reason", "shape")
        .agg(distinct=pl.len(), occurrences=pl.col("count").sum(), examples=pl.col("atom").head(3))
        .sort("occurrences", descending=True)
    )


REVIEW_COLS = [
    "group",
    "sub_component",
    "resolved_by",
    "status",
    "data_source",
    "value",
    "atom",
    "qualifier",
    "vocab",
    "subject",
    "matched_term",
    "score",
    "n_candidates",
    "count",
]


def review_sample(decisions: pl.DataFrame) -> pl.DataFrame:
    """Head and tail per (group, sub_component, resolved_by) over resolved and flagged atoms"""
    reviewable = decisions.filter(pl.col("status").is_in(["resolved", "flagged"]))
    return (
        pl.concat(
            [
                pl.concat(
                    [part.sort("count", descending=True).head(REVIEW_N), part.sample(min(REVIEW_N, len(part)), seed=0)]
                )
                for _, part in reviewable.group_by("group", "sub_component", "resolved_by")
            ]
        )
        .unique(["group", "atom"], maintain_order=True)
        .select(REVIEW_COLS)
    )


def counts_by(frame: pl.DataFrame, *by: str) -> list[dict]:
    return (
        frame.group_by(*by)
        .agg(atoms=pl.len(), occurrences=pl.col("count").sum())
        .sort("occurrences", descending=True)
        .to_dicts()
    )


def run_tiers(cascade_atoms: pl.DataFrame) -> tuple[list[pl.DataFrame], pl.DataFrame]:
    """Every deterministic tier in priority order, plus the house overlay computed alongside them"""
    with EmissionsTracker(project_name="vocab_align_exact", output_dir=str(EMISSIONS_LOG), log_level="error"):
        house_hits = house_tier(cascade_atoms)
        tier_frames = pooled_ladder(cascade_atoms)
    with EmissionsTracker(project_name="vocab_align_fuzzy", output_dir=str(EMISSIONS_LOG), log_level="error"):
        tier_frames.append(fuzzy_tier(pending_atoms(cascade_atoms, tier_frames)))

    # Key on the name: two frames can share a sub_component
    for name, frame in sorted(((f["sub_component"][0], f) for f in tier_frames if len(f)), key=lambda pair: pair[0]):
        log(f"  {name}: {len(frame):,} (group, norm) hits")
    log(f"  house: {len(house_hits):,} (group, institution, norm) hits")
    return tier_frames, house_hits


def run_llm_tier(pending: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """The tier-4 splitter queue, its segmentations and the matches they earn"""
    queue = llm_queue(pending)
    llm_split = asyncio.run(llm_atomise(queue))
    llm_lookup = llm_tier(llm_split)
    log(f"  llm: {len(llm_lookup):,} sub-atom norms matched")
    return llm_split, queue.select("group", "atom"), llm_lookup


def no_llm_tier() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    empty = pl.DataFrame(schema={"group": pl.String, "atom": pl.String})
    split = empty.with_columns(
        sub_atom=pl.lit(None, dtype=pl.String),
        sub_start=pl.lit(None, dtype=pl.Int64),
        sub_end=pl.lit(None, dtype=pl.Int64),
        sub_norm=pl.lit(None, dtype=pl.String),
    )
    return split, empty, no_hits()


def rerank_pending(pending: pl.DataFrame, llm_split: pl.DataFrame, llm_lookup: pl.DataFrame) -> pl.DataFrame:
    """What the splitter left: the atoms it did not segment, plus the sub-atoms that matched nothing"""
    whole = pending.join(llm_split.select("group", "atom").unique(), on=["group", "atom"], how="anti").select(
        "group", "atom", "norm", "count"
    )
    subs = (
        # one weight per split parent: the occurrences its sub-atoms inherit
        llm_split.join(pending.group_by("group", "atom").agg(count=pl.col("count").sum()), on=["group", "atom"])
        .join(
            llm_lookup.select("group", norm=pl.col("norm")),
            left_on=["group", "sub_norm"],
            right_on=["group", "norm"],
            how="anti",
        )
        .group_by("group", atom=pl.col("sub_atom"), norm=pl.col("sub_norm"))
        .agg(count=pl.col("count").sum())
    )
    return pl.concat([whole, subs.select(whole.columns)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align vocabulary values to authorities: house lists, then exact, fuzzy and two model rungs."
    )
    parser.add_argument("--skip-llm", action="store_true", help="run the cascade without tier 4 (no local server)")
    parser.add_argument(
        "--separators-only",
        action="store_true",
        help="induce the per-institution separators, write them and stop, without running the cascade",
    )
    args = parser.parse_args()

    VOCAB_OUT.mkdir(parents=True, exist_ok=True)
    EMISSIONS_LOG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: dict = {}

    values = value_frame(pl.scan_parquet(FIELD_STATS))
    distinct = distinct_values(values)
    log(f"{len(distinct):,} distinct (group, institution, value) rows / {distinct['count'].sum():,} occurrences")

    whole = route_values(distinct)
    report["routes"] = counts_by(whole, "group", "route")

    accepted = induce_separators(whole)
    report["separators"] = accepted.group_by("group", "separator").agg(institutions=pl.len()).sort("group").to_dicts()
    if args.separators_only:
        log(f"separators only: {json.dumps(report['separators'])}")
        return

    atoms = atomise_values(whole, accepted)
    report["atom_routes"] = counts_by(atoms, "group", "atom_route")

    cascade_atoms = atoms.filter(pl.col("atom_route") == "cascade")
    tier_frames, house_hits = run_tiers(cascade_atoms)
    pending = pending_atoms(cascade_atoms, tier_frames)

    if args.skip_llm:
        log("tier 4 skipped — values it would have segmented stay deferred as `atomiser`")
        llm_split, llm_processed, llm_lookup = no_llm_tier()
        rerank_hits, rerank_queued = no_hits(), pl.DataFrame(schema={"group": pl.String, "norm": pl.String})
    else:
        llm_split, llm_processed, llm_lookup = run_llm_tier(pending)
        rerank_hits, rerank_queued = rerank_tier(rerank_pending(pending, llm_split, llm_lookup))
    report["rerank"] = {"queued": len(rerank_queued), "selected": len(rerank_hits)}

    # The rerank rung runs last, reaching atoms and sub-atoms alike
    sub_atom_lookup = pl.concat([llm_lookup, rerank_hits], how="diagonal").unique(
        ["group", "norm"], keep="first", maintain_order=True
    )
    decisions = assemble(atoms, [*tier_frames, rerank_hits], house_hits, llm_split, llm_processed, sub_atom_lookup)
    if RUNG_VERDICTS.exists():
        decisions = apply_verdicts(decisions)
    decisions.write_parquet(VOCAB_DECISIONS)
    report["decisions"] = counts_by(decisions, "group", "status")
    log(f"{len(decisions):,} atom decisions → {VOCAB_DECISIONS}")

    n_ann = write_sidecar(values, decisions)
    log(f"{n_ann:,} annotation rows → {VOCAB_ANNOTATIONS}")

    per_field, per_institution = coverage(pl.scan_parquet(VOCAB_ANNOTATIONS))
    below = per_institution.filter(pl.col("resolved_occ") < COVERAGE_TARGET)
    report["per_field"] = per_field.to_dicts()
    report["below_target"] = {"pairs": len(below), "of": len(per_institution)}
    log(f"{len(below)}/{len(per_institution)} (institution, group) pairs below the {COVERAGE_TARGET:.0%} target")

    residue_worklist(decisions).write_parquet(RESIDUE_WORKLIST)
    review_sample(decisions).write_csv(REVIEW_SAMPLE)
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    log(f"review sample → {REVIEW_SAMPLE}; report → {REPORT}")


if __name__ == "__main__":
    main()
