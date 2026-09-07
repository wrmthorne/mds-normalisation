from __future__ import annotations

import argparse
import json
import time

import polars as pl
from mds_data_model.introspection import date_fields, monetary_fields

from mds_norm.paths import FIELD_STATS, PATTERNS_OUT
from mds_norm.pipeline.consistency_induction import OBJECT_NUMBER_FIELDS, induce_lookup

OUT_DIR = PATTERNS_OUT

# Values kept per (institution, pattern, slot); slot-role tests sample
SLOT_SAMPLE = 10_000

# Slots below this evidence floor are not written
MIN_SLOT_SUPPORT = 50

# Merged patterns per family carrying a worked example
EXAMPLE_TOP_N = 12

# Maximal runs of one character class, in slot order
SLOT_RUNS = r"[0-9]+|[A-Za-z]+"

# Each family pools into the group field_census.merge_group writes
FAMILIES = {
    "date": ("__date__", sorted(date_fields())),
    "dimension": ("spectrum/dimension", ["spectrum/dimension"]),
    "money": ("__price__", sorted(monetary_fields())),
    "object_number": ("__object_number__", OBJECT_NUMBER_FIELDS),
}

# Stages a value passes through, each an `induced` column
STAGES = ("source_pattern", "skeleton", "merged_pattern")

SLOT_KIND = pl.when(pl.col("slot").str.contains(r"^[0-9]+$")).then(pl.lit("d")).otherwise(pl.lit("s"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def family_values(family: str) -> pl.LazyFrame:
    """Census rows for one family, tagged with the induction group they pool into"""
    group, fields = FAMILIES[family]
    return (
        pl.scan_parquet(FIELD_STATS)
        .filter(pl.col("field_type").is_in(fields) & pl.col("value").is_not_null())
        .select(pl.col("data_source").cast(pl.String), "field_type", "value", merge_group=pl.lit(group))
    )


def induced(family: str, case_sensitive: bool = False) -> pl.DataFrame:
    """Family occurrences carrying the pattern they hold at each stage of the induction"""
    base = family_values(family)
    lookup = induce_lookup(
        base.select("merge_group", "value"), keep_features=True, case_sensitive=case_sensitive
    ).select(
        "value",
        # institutional_fingerprints writes a single-character run as the bare marker
        source_pattern=pl.col("pattern").str.replace_all(r"\{1\}", ""),
        skeleton=pl.col("pattern").str.replace_all(r"\{\d+\}", ""),
        merged_pattern="merged_pattern",
    )
    return base.collect(engine="streaming").join(lookup, on="value", how="left").drop_nulls("merged_pattern")


def slot_values(values: pl.DataFrame) -> pl.DataFrame:
    """A sample of each (institution, merged pattern, slot)'s values, for the slot-role tests"""
    return (
        values.lazy()
        .with_columns(slot=pl.col("value").str.extract_all(SLOT_RUNS))
        .with_columns(slot_idx=pl.int_ranges(0, pl.col("slot").list.len()).cast(pl.List(pl.Int32)))
        .explode("slot", "slot_idx")
        .drop_nulls("slot")
        .with_columns(slot_kind=SLOT_KIND)
        .group_by("data_source", "merged_pattern", "slot_idx", "slot_kind")
        .agg(values=pl.col("slot").head(SLOT_SAMPLE), support=pl.len())
        .filter(pl.col("support") >= MIN_SLOT_SUPPORT)
        .drop("support")
        .collect(engine="streaming")
    )


def pattern_dist(values: pl.DataFrame) -> pl.DataFrame:
    """Each institution's distribution over the family's merged patterns"""
    return (
        values.group_by("data_source", "merged_pattern")
        .agg(count=pl.len().cast(pl.UInt32))
        .with_columns(prob=pl.col("count") / pl.col("count").sum().over("data_source"))
    )


def patterns_merged(values: pl.DataFrame) -> pl.DataFrame:
    """One row per merged pattern, carrying the (field type, source pattern) members it absorbed"""
    members = (
        values.group_by("merged_pattern", "field_type", "source_pattern")
        .agg(count=pl.len())
        .sort("count", descending=True)
        .group_by("merged_pattern")
        .agg(
            total_count=pl.col("count").sum(),
            n_members=pl.len(),
            members=pl.struct(field_type="field_type", pattern="source_pattern", count="count"),
        )
    )
    return members.with_columns(
        pl.col("members").map_elements(lambda m: json.dumps(m.to_list()), return_dtype=pl.String)
    ).sort("total_count", descending=True)


def pattern_examples(family: str) -> pl.DataFrame:
    """The most common real value behind each of the family's highest-mass merged patterns"""
    top = pl.read_parquet(OUT_DIR / f"{family}_patterns_merged.parquet").head(EXAMPLE_TOP_N)
    counts = family_values(family).group_by("merge_group", "value").agg(count=pl.len()).collect(engine="streaming")
    lookup = induce_lookup(counts.lazy().select("merge_group", "value"))
    return (
        top.select("merged_pattern", "total_count")
        .with_columns(rank=pl.int_range(1, pl.len() + 1, dtype=pl.UInt32))
        .join(counts.join(lookup, on=["merge_group", "value"]), on="merged_pattern", how="left")
        .sort("count", descending=True, nulls_last=True)
        .group_by("merged_pattern")
        .first()
        .select("rank", "merged_pattern", "total_count", example="value", example_count="count")
        .sort("rank")
    )


def coverage_curve(values: pl.DataFrame, family: str) -> pl.DataFrame:
    """Patterns ranked by mass at each induction stage, with the share of the family's values they cover"""
    curves = []
    for stage in STAGES:
        ranked = (
            values.group_by(stage)
            .agg(count=pl.len())
            .sort("count", descending=True)
            .with_columns(rank=pl.int_range(1, pl.len() + 1, dtype=pl.UInt32))
            .with_columns(cum_share=pl.col("count").cum_sum() / pl.col("count").sum())
        )
        curves.append(
            ranked.select(pl.lit(family).alias("family"), pl.lit(stage).alias("stage"), "rank", "count", "cum_share")
        )
    return pl.concat(curves)


def stage_counts(curve: pl.DataFrame) -> pl.DataFrame:
    """How many patterns each stage leaves, and how many of them cover four fifths of the values"""
    return (
        curve.group_by("family", "stage")
        .agg(
            n_patterns=pl.col("rank").max(), n_values=pl.col("count").sum(), k_80=pl.col("cum_share").lt(0.8).sum() + 1
        )
        .with_columns(stage=pl.col("stage").cast(pl.Enum(STAGES)))
        .sort("family", "stage")
    )


def case_comparison() -> pl.DataFrame:
    """Pattern counts per family when letters mask to one class against upper and lower kept apart"""
    rows = []
    for family in FAMILIES:
        for case_sensitive in (False, True):
            curve = coverage_curve(induced(family, case_sensitive=case_sensitive), family)
            rows.append(
                stage_counts(curve).with_columns(masking=pl.lit("mixed case" if case_sensitive else "one class"))
            )
    return pl.concat(rows).sort("family", "masking", "stage")


def export(family: str) -> None:
    values = induced(family)
    log(f"{family}: {values.height:,} occurrences, {values['merged_pattern'].n_unique():,} merged patterns")

    slots = slot_values(values)
    slots.write_parquet(OUT_DIR / f"{family}_institutional_slot_values.parquet")
    log(f"{family}: {slots.height:,} (institution, pattern, slot) cells")

    dist = pattern_dist(values)
    dist.write_parquet(OUT_DIR / f"{family}_institutional_pattern_dist.parquet")
    log(f"{family}: {dist.height:,} (institution, pattern) cells over {dist['data_source'].n_unique()} institutions")

    merged = patterns_merged(values)
    merged.write_parquet(OUT_DIR / f"{family}_patterns_merged.parquet")
    log(f"{family}: {merged.height:,} merged patterns -> {OUT_DIR / f'{family}_patterns_merged.parquet'}")

    examples = pattern_examples(family)
    examples.write_parquet(OUT_DIR / f"{family}_pattern_examples.parquet")
    log(f"{family}: worked examples for the top {examples.height} merged patterns")

    curve = coverage_curve(values, family)
    curve.write_parquet(OUT_DIR / f"{family}_coverage_curve.parquet")
    counts = stage_counts(curve)
    log(f"{family}: " + ", ".join(f"{r['stage']} {r['n_patterns']:,}" for r in counts.iter_rows(named=True)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pattern artefacts the institutional priors and fingerprints read.")
    ap.add_argument("--family", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES))
    ap.add_argument(
        "--examples",
        action="store_true",
        help="write only the worked examples, leaving the sampled artefacts of a full export untouched",
    )
    ap.add_argument(
        "--case-comparison",
        action="store_true",
        help="write the pattern counts under both maskings and exit; the shipped induction masks letters as one class",
    )
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.case_comparison:
        comparison = case_comparison()
        comparison.write_parquet(OUT_DIR / "case_sensitivity.parquet")
        log(f"case comparison -> {OUT_DIR / 'case_sensitivity.parquet'}")
        print(comparison)
        return

    for family in args.family:
        if args.examples:
            examples = pattern_examples(family)
            examples.write_parquet(OUT_DIR / f"{family}_pattern_examples.parquet")
            log(f"{family}: worked examples for the top {examples.height} merged patterns")
        else:
            export(family)


if __name__ == "__main__":
    main()
