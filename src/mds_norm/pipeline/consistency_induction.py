from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
from mds_data_model.introspection import date_fields, measurement_fields, monetary_fields, reference_number_fields

from mds_norm.paths import COMPILED, CONSISTENCY_OUT, RAW_RECORDS

OUT_DIR = CONSISTENCY_OUT
CORPORA = {"raw": RAW_RECORDS, "compiled": COMPILED / "mds-normalised.parquet"}

OBJECT_NUMBER_FIELDS = [
    "spectrum/object_number",
    "spectrum/other_number",
    "spectrum/related_object_number",
    "spectrum/disposal_new_object_number",
    "spectrum/catalogue_number",
]
REFERENCE_NUMBER_FIELDS = list(reference_number_fields())
PRICE_FIELDS = list(monetary_fields())
MEASUREMENT_FIELDS = list(measurement_fields())
DATE_FIELDS = pl.col("field_type").is_in(date_fields())
STRUCTURED_FIELDS = pl.col("field_type").is_in(PRICE_FIELDS + MEASUREMENT_FIELDS)
# Fields the census induces patterns for, feeding the probes
PATTERN_FIELDS = DATE_FIELDS | STRUCTURED_FIELDS
# Object numbers qualify too: the scheme is the convention
CONSISTENCY_FIELDS = PATTERN_FIELDS | pl.col("field_type").is_in(OBJECT_NUMBER_FIELDS)

MARKERS = ["d", "s", "S"]
SEP = "\x1f"
WIDE_MIN = 3
SPLIT_THRESHOLD = 50
TAU = 0.05  # head mass threshold
# k_80 covers this share of a field's values
K80_COVERAGE = 0.80
# Floors the sensitivity check reads, TAU among them
TAU_SWEEP = (0.02, 0.05, 0.10)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def induce_lookup(scoped: pl.LazyFrame, keep_features: bool = False, case_sensitive: bool = False) -> pl.DataFrame:
    """(merge_group, value) -> merged_pattern"""
    el = pl.element()
    if case_sensitive:
        masked = (
            pl.col("value")
            .str.replace_all(r"[a-z]", "s")
            .str.replace_all(r"[A-Z]", "S")
            .str.replace_all(r"[0-9]", "d")
        )
        slot_pat = r"[0-9]+|[A-Z]+|[a-z]+"
        run_pat = r"d+|s+|S+|[\s\S]"
    else:
        masked = pl.col("value").str.replace_all(r"[A-Za-z]", "s").str.replace_all(r"[0-9]", "d")
        slot_pat = r"[A-Za-z]+|[0-9]+"
        run_pat = r"d+|s+|[\s\S]"

    pattern_expr = (
        pl.col("runs")
        .list.eval(
            pl.when(el.str.slice(0, 1).is_in(MARKERS))
            .then(el.str.slice(0, 1) + "{" + el.str.len_chars().cast(pl.String) + "}")
            .otherwise(el)
        )
        .list.join("")
    )
    width_sig_expr = (
        pl.when(pl.col("runs").list.len() == 1 & pl.col("runs").list.first().str.slice(0, 1).is_in(MARKERS))
        .then("L" + pl.col("runs").list.first().str.len_chars().cast(pl.String))
        .otherwise(
            pl.col("runs")
            .list.eval(
                pl.when(~el.str.slice(0, 1).is_in(MARKERS))
                .then(pl.lit("."))
                .when(el.str.len_chars() >= WIDE_MIN)
                .then(pl.lit("W"))
                .otherwise(pl.lit("N"))
            )
            .list.join("")
        )
    )
    merge_key = (
        pl.when(pl.col("merge_group") == "__date__")
        .then(pl.col("skeleton") + SEP + pl.col("width_sig"))
        .otherwise(pl.col("skeleton"))
    )

    # Stage 1 — distinct value -> features
    value_patterns = (
        scoped.select("value")
        .unique()
        .with_columns(masked=masked, slot_values=pl.col("value").str.extract_all(slot_pat))
        .with_columns(runs=pl.col("masked").str.extract_all(run_pat))
        .with_columns(pattern=pattern_expr, width_sig=width_sig_expr)
        .with_columns(skeleton=pl.col("pattern").str.replace_all(r"\{\d+\}", ""))
        .select("value", "pattern", "skeleton", "width_sig", "runs", *(["slot_values"] if keep_features else []))
        .collect()
    )
    group_values = scoped.unique(subset=["merge_group", "value"]).collect()

    skeleton_patterns = group_values.join(value_patterns, on="value", how="left").with_columns(merge_key=merge_key)

    # Stage 2 — digit-slot spread detector
    digit_slots = (
        value_patterns.select("value", "pattern", "skeleton", "width_sig")
        .with_columns(d=pl.col("value").str.extract_all(r"[0-9]+"))
        .with_columns(
            slot=pl.int_ranges(0, pl.col("d").list.len()), slot_len=pl.col("d").list.eval(el.str.len_chars())
        )
        .explode("d", "slot", "slot_len", empty_as_null=True)
        .with_columns(slot_int=pl.col("d").cast(pl.Int64, strict=False))
        .join(group_values, on="value", how="inner")
        .with_columns(merge_key=merge_key)
    )
    pattern_medians = digit_slots.group_by("merge_group", "merge_key", "slot", "slot_len", "pattern").agg(
        medians=pl.col("slot_int").median()
    )
    group_score = (
        pattern_medians.group_by("merge_group", "merge_key", "slot", "slot_len")
        .agg(spread=pl.col("medians").max() - pl.col("medians").min())
        .group_by("merge_group", "merge_key", "slot")
        .agg(score=pl.col("spread").max())
        .group_by("merge_group", "merge_key")
        .agg(
            max_score=pl.col("score").max(),
            # Slots tie on spread, so the lowest index wins
            worst_slot=pl.col("slot").filter(pl.col("score") == pl.col("score").max()).min(),
        )
    )

    # Stage 3 — split map on the worst slot
    flagged = group_score.filter(pl.col("max_score") >= SPLIT_THRESHOLD)
    split_map = (
        pattern_medians.join(
            flagged.select("merge_group", "merge_key", "worst_slot"),
            left_on=["merge_group", "merge_key", "slot"],
            right_on=["merge_group", "merge_key", "worst_slot"],
            how="inner",
        )
        # The gap walk reads neighbours, so order must be total
        .sort("medians", "pattern")
        .with_columns(
            gap=(pl.col("medians") - pl.col("medians").shift(1)).over("merge_group", "merge_key", "slot_len")
        )
        .with_columns(
            is_break=(pl.col("gap") == pl.col("gap").max().over("merge_group", "merge_key", "slot_len")).fill_null(
                False
            )
        )
        .with_columns(cluster=pl.col("is_break").cast(pl.Int32).cum_sum().over("merge_group", "merge_key", "slot_len"))
        .select(
            "merge_group",
            "merge_key",
            "pattern",
            split_id=pl.col("slot_len").cast(pl.String) + ":" + pl.col("cluster").cast(pl.String),
        )
    )
    skeleton_patterns_2 = skeleton_patterns.join(
        split_map, on=["merge_group", "merge_key", "pattern"], how="left"
    ).with_columns(refined_key=pl.col("merge_key") + SEP + pl.col("split_id").fill_null(""))

    # Stage 4 — merge width-variants per refined_key
    keyed = skeleton_patterns_2.select("merge_group", "refined_key", "pattern", "runs").unique(
        subset=["merge_group", "refined_key", "pattern"]
    )
    m, c = pl.col("runs"), pl.col("c")
    merged = (
        keyed.with_columns(idx=pl.int_ranges(0, pl.col("runs").list.len()))
        .explode("runs", "idx", empty_as_null=True)
        .group_by("merge_group", "refined_key", "idx")
        .agg(c=m.str.slice(0, 1).first(), mn=m.str.len_chars().min(), mx=m.str.len_chars().max())
        .with_columns(
            tok_str=pl.when(~c.is_in(MARKERS))
            .then(c)
            .when(pl.col("mn") == pl.col("mx"))
            .then(c + "{" + pl.col("mn").cast(pl.String) + "}")
            .otherwise(c + "{" + pl.col("mn").cast(pl.String) + "," + pl.col("mx").cast(pl.String) + "}")
        )
        .group_by("merge_group", "refined_key")
        .agg(merged_pattern=pl.col("tok_str").sort_by("idx").str.join(""))
    )

    out = skeleton_patterns_2.select("merge_group", "value", "refined_key").join(
        merged, on=["merge_group", "refined_key"], how="left"
    )
    if keep_features:
        return out.join(value_patterns.select("value", "pattern", "slot_values"), on="value", how="left").select(
            "merge_group", "value", "pattern", "slot_values", "merged_pattern"
        )
    return out.select("merge_group", "value", "merged_pattern")


def mask_corpus(path: Path) -> pl.DataFrame:
    """Per (record_id, data_source, field_type) merged_pattern for the inducible fields"""
    group_expr = (
        pl.when(pl.col("field_type").str.contains("date"))
        .then(pl.lit("__date__"))
        .when(pl.col("field_type").is_in(OBJECT_NUMBER_FIELDS))
        .then(pl.lit("__object_number__"))
        .when(pl.col("field_type").is_in(REFERENCE_NUMBER_FIELDS))
        .then(pl.lit("__reference_number__"))
        .when(pl.col("field_type").is_in(PRICE_FIELDS))
        .then(pl.lit("__price__"))
        .otherwise(pl.col("field_type"))
        .alias("merge_group")
    )

    base = (
        pl.scan_parquet(path)
        .filter(CONSISTENCY_FIELDS & pl.col("value").is_not_null())
        .with_columns(pl.col("value").str.strip_chars().str.replace_all(r"\s+", " "))
        .filter(pl.col("value") != "")
        .select("record_id", "data_source", "field_type", "value")
        .with_columns(group_expr)
    )

    scoped = base.select("merge_group", "value")
    lookup = induce_lookup(scoped)
    return (
        base.collect()
        .join(lookup, on=["merge_group", "value"], how="left")
        .select("record_id", "data_source", "field_type", "merged_pattern")
    )


def score_corpus(masked: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(per-institution-field concentration, per-record consistency counts, per-field verdict counts)"""
    counts = masked.group_by("data_source", "field_type", "merged_pattern").agg(pl.len().alias("count"))
    totals = counts.group_by("data_source", "field_type").agg(
        pl.col("count").sum().alias("n_records"),
        pl.col("count").filter(pl.col("merged_pattern").is_null()).sum().alias("n_anomaly"),
    )
    real = (
        counts.filter(pl.col("merged_pattern").is_not_null())
        .join(totals, on=["data_source", "field_type"])
        .with_columns((pl.col("count") / pl.col("n_records")).alias("share"))
        .with_columns((pl.col("share") >= TAU).alias("in_head"))
    )

    # k_80: merged patterns, largest first, covering 80%
    k80 = (
        real.sort(["data_source", "field_type", "share"], descending=[False, False, True])
        .with_columns(pl.col("share").cum_sum().over("data_source", "field_type").alias("cum_share"))
        .group_by("data_source", "field_type")
        .agg(
            (pl.col("cum_share") < K80_COVERAGE).sum().alias("_below"),
            (pl.col("cum_share").max() >= K80_COVERAGE).alias("_reaches"),
        )
        .with_columns(pl.when(pl.col("_reaches")).then(pl.col("_below") + 1).otherwise(None).alias("k_80"))
        .select("data_source", "field_type", "k_80")
    )
    concentration = (
        real.group_by("data_source", "field_type")
        .agg(
            pl.col("in_head").sum().alias("n_head_patterns"),
            pl.len().alias("n_patterns"),
            pl.col("n_records").first(),
            pl.col("n_anomaly").first(),
            pl.col("count").filter(pl.col("in_head")).sum().alias("head_count"),
        )
        .join(k80, on=["data_source", "field_type"])
        .with_columns(
            (pl.col("head_count") / pl.col("n_records")).alias("head_share"),
            (pl.col("n_anomaly") / pl.col("n_records")).alias("anomaly_share"),
        )
        .select(
            "data_source",
            "field_type",
            "k_80",
            "n_head_patterns",
            "n_patterns",
            "n_records",
            "head_share",
            "anomaly_share",
        )
        .sort(["field_type", "k_80"])
    )

    head_set = real.filter(pl.col("in_head")).select("data_source", "field_type", "merged_pattern")
    verdicts = masked.join(
        head_set.with_columns(pl.lit(True).alias("in_head")),
        on=["data_source", "field_type", "merged_pattern"],
        how="left",
    ).with_columns(
        pl.col("merged_pattern").is_not_null().alias("well_formed"),
        (pl.col("merged_pattern").is_not_null() & pl.col("in_head").fill_null(False)).alias("conventional"),
    )
    well_formed = verdicts.filter(pl.col("well_formed"))  # malformed excluded from consistency
    consistency_rec = (
        well_formed.group_by("record_id", "data_source")
        .agg(pl.len().alias("n_conventional_applicable"), pl.col("conventional").sum().alias("n_conventional_ok"))
        .with_columns((pl.col("n_conventional_ok") / pl.col("n_conventional_applicable")).alias("consistency"))
    )
    # Per-field counts so consistency can be recomputed over a subset of a record's fields
    consistency_field = well_formed.group_by("record_id", "data_source", "field_type").agg(
        pl.len().alias("n_conventional_applicable"), pl.col("conventional").sum().alias("n_conventional_ok")
    )
    return concentration, consistency_rec, consistency_field


def head_floor_sweep(masked: pl.DataFrame) -> pl.DataFrame:
    """The divergent share at each candidate head floor, corpus-wide and per institution-field pair"""
    counts = masked.group_by("data_source", "field_type", "merged_pattern").agg(pl.len().alias("count"))
    shares = counts.filter(pl.col("merged_pattern").is_not_null()).with_columns(
        share=pl.col("count") / pl.col("count").sum().over("data_source", "field_type")
    )
    rows = []
    for tau in TAU_SWEEP:
        pairs = (
            shares.with_columns(in_head=pl.col("share") >= tau)
            .group_by("data_source", "field_type")
            .agg(
                n_values=pl.col("count").sum(),
                head_count=pl.col("count").filter("in_head").sum(),
                n_head_patterns=pl.col("in_head").sum(),
            )
            .with_columns(divergent_share=1 - pl.col("head_count") / pl.col("n_values"))
        )
        rows.append(pairs.with_columns(tau=pl.lit(tau)))
    return pl.concat(rows).select("tau", "data_source", "field_type", "n_values", "n_head_patterns", "divergent_share")


def main(corpora: list[str] | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for corpus in corpora or list(CORPORA):
        path = CORPORA[corpus]
        log(f"masking {corpus} corpus {path.name}…")
        masked = mask_corpus(path)
        log(f"{corpus}: {masked.height:,} inducible field-values")
        concentration, rec, field_verdicts = score_corpus(masked)
        concentration.write_parquet(OUT_DIR / f"pattern_consistency_{corpus}.parquet")
        field_verdicts.write_parquet(OUT_DIR / f"consistency_field_{corpus}.parquet")
        sweep = head_floor_sweep(masked)
        sweep.write_parquet(OUT_DIR / f"head_floor_sweep_{corpus}.parquet")
        pooled = sweep.group_by("tau").agg(
            divergent_share=1 - (pl.col("n_values") * (1 - pl.col("divergent_share"))).sum() / pl.col("n_values").sum()
        )
        log(
            f"{corpus}: divergent share by head floor "
            + ", ".join(f"{r['tau']:.0%} {r['divergent_share']:.4f}" for r in pooled.sort("tau").iter_rows(named=True))
        )
        rec.write_parquet(OUT_DIR / f"consistency_record_{corpus}.parquet")
        rate = rec["n_conventional_ok"].sum() / rec["n_conventional_applicable"].sum()
        log(
            f"{corpus}: {rec.height:,} records scored, corpus conventional rate {rate:.4f}, "
            f"anomaly-share median {concentration['anomaly_share'].median():.4f}"
        )
        print(f"  → consistency_record_{corpus}.parquet  ({rec.height:,} records)")
    print(f"\n→ {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Induce per-field patterns and score record consistency.")
    ap.add_argument(
        "--corpus", choices=list(CORPORA), action="append", help="restrict to these corpora (default: all)"
    )
    main(ap.parse_args().corpus)
