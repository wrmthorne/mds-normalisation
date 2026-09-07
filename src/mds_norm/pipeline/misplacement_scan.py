from __future__ import annotations

import argparse
import time

import polars as pl
from mds_data_model.introspection import (
    agent_fields,
    date_fields,
    free_text_fields,
    measurement_fields,
    monetary_fields,
    vocab_fields,
)
from scipy.stats import binomtest, false_discovery_control

from mds_norm.paths import (
    FIELD_STATS,
    MISPLACEMENT_EXAMPLES,
    MISPLACEMENT_HITS,
    MISPLACEMENT_OUT,
    MISPLACEMENT_PAIRS,
    PROBE_CANDIDATES,
)
from mds_norm.utils.patches import DEST

# Detector 1: probe spans must cover most of the cell
CELL_COVERAGE = 0.6

# Detector 2: candidate values, and what claims a home field
MAX_VALUE_CHARS = 60
MIN_HOME_INSTITUTIONS = 5
HOME_DOMINANCE = 0.8

# Reporting floors for an (institution, field, home field) pair
MIN_HITS = 30
MIN_SHARE = 0.005
MIN_LIFT = 5.0
ALPHA = 0.01
# Both fields populated is a distinction, not a mapping error
MIN_HOME_ABSENT = 0.5
EXAMPLES_PER_PAIR = 15

TYPED_FIELDS = sorted(
    set(vocab_fields()) | set(agent_fields()) | set(date_fields()) | set(measurement_fields()) | set(monetary_fields())
)
CANDIDATE_FIELDS = sorted(set(TYPED_FIELDS) | set(free_text_fields()))

NORM = pl.col("value").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()
HIT_SCHEMA = ["data_source", "field_type", "home_field", "record_id", "value", "detector"]
PAIR_KEYS = ["data_source", "field_type", "home_field", "detector"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def typed_hits(field_stats: pl.LazyFrame, candidates: pl.LazyFrame) -> pl.LazyFrame:
    """Cells the probe spans account for whole: the value is the structured type, not prose mentioning it"""
    return (
        candidates.join(field_stats.select("node_id", "value", "char_count"), on="node_id")
        .group_by("node_id", "record_id", "data_source", "field_type", "group")
        .agg(
            covered=pl.col("candidate").str.len_chars().sum(),
            char_count=pl.col("char_count").first(),
            value=pl.col("value").first(),
        )
        .filter(pl.col("covered") / pl.col("char_count") >= CELL_COVERAGE)
        .with_columns(home_field=DEST, detector=pl.lit("typed"))
        .filter(pl.col("home_field") != pl.col("field_type"))
        .select(HIT_SCHEMA)
    )


def field_usage(field_stats: pl.LazyFrame) -> pl.DataFrame:
    """Occurrences of each short normalised value, per institution and field"""
    return (
        field_stats.filter(pl.col("field_type").is_in(CANDIDATE_FIELDS) & (pl.col("char_count") <= MAX_VALUE_CHARS))
        .select("data_source", "field_type", norm=NORM)
        .group_by("data_source", "field_type", "norm")
        .len("n")
        .collect(engine="streaming")
    )


def consensus_home(usage: pl.DataFrame) -> pl.DataFrame:
    """Per (institution, field, value): the field the rest of the corpus keeps that value in"""
    support = (
        usage.filter(pl.col("field_type").is_in(TYPED_FIELDS))
        .group_by("norm", "field_type")
        .agg(institutions=pl.len())
    )
    multi = support.filter(pl.len().over("norm") > 1)
    return (
        usage.select("data_source", "norm", used_field="field_type")
        .join(multi, on="norm")
        .with_columns(
            institutions=pl.col("institutions") - (pl.col("used_field") == pl.col("field_type")).cast(pl.UInt32)
        )
        .group_by("data_source", "norm", "used_field")
        .agg(
            home_field=pl.col("field_type").sort_by("institutions", "field_type", descending=[True, False]).first(),
            home_institutions=pl.col("institutions").max(),
            total_institutions=pl.col("institutions").sum(),
        )
        .filter(
            (pl.col("home_field") != pl.col("used_field"))
            & (pl.col("home_institutions") >= MIN_HOME_INSTITUTIONS)
            & (pl.col("home_institutions") / pl.col("total_institutions") >= HOME_DOMINANCE)
        )
    )


def distributional_hits(field_stats: pl.LazyFrame, consensus: pl.DataFrame) -> pl.LazyFrame:
    """Every occurrence of a value its own institution files away from the corpus consensus"""
    keys = consensus.select("data_source", "norm", "home_field", field_type=pl.col("used_field")).lazy()
    return (
        field_stats.filter(pl.col("field_type").is_in(CANDIDATE_FIELDS) & (pl.col("char_count") <= MAX_VALUE_CHARS))
        .select("data_source", "field_type", "record_id", "value", norm=NORM)
        .join(keys, on=["data_source", "field_type", "norm"])
        .with_columns(detector=pl.lit("distributional"))
        .select(HIT_SCHEMA)
    )


def home_present(field_stats: pl.LazyFrame, hits: pl.DataFrame) -> pl.DataFrame:
    """Whether each hit record already carries a value in the field the content belongs to"""
    homes = hits["home_field"].unique().to_list()
    record_home = (
        field_stats.filter(pl.col("field_type").is_in(homes))
        .select("record_id", home_field="field_type")
        .unique()
        .with_columns(home_present=pl.lit(True))
    )
    return (
        hits.lazy()
        .join(record_home, on=["record_id", "home_field"], how="left")
        .with_columns(pl.col("home_present").fill_null(False))
        .collect(engine="streaming")
    )


def score_pairs(hits: pl.DataFrame, denominators: pl.DataFrame) -> pl.DataFrame:
    """Rank (institution, field, home field) pairs by how far they depart from the rest of the corpus"""
    pairs = (
        hits.group_by(PAIR_KEYS)
        .agg(
            hits=pl.len(),
            values=pl.col("value").n_unique(),
            records=pl.col("record_id").n_unique(),
            home_absent=1 - pl.col("home_present").mean(),
        )
        .join(denominators, on=["data_source", "field_type"])
        .join(
            denominators.rename({"field_type": "home_field", "occ": "home_occ"}),
            on=["data_source", "home_field"],
            how="left",
        )
        .with_columns(pl.col("home_occ").fill_null(0), share=pl.col("hits") / pl.col("occ"))
        .with_columns(over_floor=pl.col("share") >= MIN_SHARE)
        .with_columns(peers=pl.sum("over_floor").over("field_type", "home_field", "detector") - pl.col("over_floor"))
    )
    corpus = pairs.group_by("field_type", "home_field", "detector").agg(
        all_hits=pl.col("hits").sum(), all_occ=pl.col("occ").sum()
    )
    field_occ = denominators.group_by("field_type").agg(field_occ=pl.col("occ").sum())
    scored = (
        pairs.join(corpus, on=["field_type", "home_field", "detector"])
        .join(field_occ, on="field_type")
        # Leave-one-out baseline; half an occurrence keeps the rate nonzero
        .with_columns(baseline=(pl.col("all_hits") - pl.col("hits") + 0.5) / (pl.col("field_occ") - pl.col("occ") + 1))
        .with_columns(lift=pl.col("share") / pl.col("baseline"))
        .filter((pl.col("hits") >= MIN_HITS) & pl.col("over_floor"))
    )
    return (
        scored.with_columns(q=_adjusted_p(scored))
        .with_columns(
            mapping_issue=(pl.col("lift") >= MIN_LIFT)
            & (pl.col("q") <= ALPHA)
            & (pl.col("home_absent") >= MIN_HOME_ABSENT)
        )
        .sort("hits", descending=True)
        .select(
            "data_source",
            "field_type",
            "home_field",
            "detector",
            "hits",
            "values",
            "records",
            "occ",
            "share",
            "baseline",
            "lift",
            "q",
            "peers",
            "home_absent",
            "home_occ",
            "mapping_issue",
        )
    )


def _adjusted_p(scored: pl.DataFrame) -> pl.Series:
    """One-sided binomial test of each pair's rate against its leave-one-out baseline, BH-adjusted"""
    if scored.is_empty():
        return pl.Series("q", [], dtype=pl.Float64)
    p = [
        binomtest(row["hits"], row["occ"], row["baseline"], alternative="greater").pvalue
        for row in scored.iter_rows(named=True)
    ]
    return pl.Series("q", false_discovery_control(p, method="bh"))


def pair_examples(hits: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """The values that fired, most frequent first, for review"""
    return (
        hits.join(pairs.select(PAIR_KEYS), on=PAIR_KEYS)
        .group_by(*PAIR_KEYS, "value")
        .agg(n=pl.len(), home_absent=1 - pl.col("home_present").mean())
        .sort("n", descending=True)
        .group_by(PAIR_KEYS, maintain_order=True)
        .head(EXAMPLES_PER_PAIR)
    )


def run(detectors: list[str]) -> pl.DataFrame:
    MISPLACEMENT_OUT.mkdir(parents=True, exist_ok=True)
    field_stats = pl.scan_parquet(FIELD_STATS)
    denominators = field_stats.group_by("data_source", "field_type").len("occ").collect(engine="streaming")

    frames = []
    if "typed" in detectors:
        hits = typed_hits(field_stats, pl.scan_parquet(PROBE_CANDIDATES)).collect(engine="streaming")
        log(f"typed: {len(hits):,} whole-cell hits over {hits['field_type'].n_unique()} fields")
        frames.append(hits)
    if "distributional" in detectors:
        usage = field_usage(field_stats)
        log(f"distributional: {len(usage):,} (institution, field, value) cells")
        consensus = consensus_home(usage)
        log(f"distributional: {len(consensus):,} values the corpus homes elsewhere")
        hits = distributional_hits(field_stats, consensus).collect(engine="streaming")
        log(f"distributional: {len(hits):,} hits")
        frames.append(hits)

    hits = home_present(field_stats, pl.concat(frames))
    hits.write_parquet(MISPLACEMENT_HITS)
    pairs = score_pairs(hits, denominators)
    pairs.write_parquet(MISPLACEMENT_PAIRS)
    pair_examples(hits, pairs).write_parquet(MISPLACEMENT_EXAMPLES)
    flagged = pairs.filter("mapping_issue")
    log(f"{len(pairs):,} pairs above the floors, {len(flagged):,} flagged → {MISPLACEMENT_PAIRS}")
    log(f"{flagged['data_source'].n_unique()} institutions, {flagged['hits'].sum():,} occurrences")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Institution-level field mapping errors: content filed in the wrong field."
    )
    ap.add_argument("--detector", nargs="+", default=["typed", "distributional"], choices=["typed", "distributional"])
    args = ap.parse_args()
    with pl.Config(tbl_rows=40, fmt_str_lengths=50, tbl_width_chars=200):
        print(run(args.detector).filter("mapping_issue").head(40))


if __name__ == "__main__":
    main()
