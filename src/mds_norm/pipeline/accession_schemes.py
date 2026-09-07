from __future__ import annotations

import argparse
import json
import time
from itertools import pairwise

import polars as pl

from mds_norm.paths import PATTERNS_OUT, RAW_RECORDS
from mds_norm.utils.masking import MASK, signature_of

OUT_PATH = PATTERNS_OUT / "object_number_year_encoding.parquet"
DIFFS_PATH = PATTERNS_OUT / "accession_year_diffs.parquet"

# Four-digit runs outside this window are not years
YEAR_MIN, YEAR_MAX = 1750, 2026
YEAR_DIGITS = 4

# Year slot: >=80% four-digit values, >=90% of those in range
FRAC4_MIN, YEARFRAC_MIN = 0.80, 0.90

# Record share in a year-carrying pattern; the fallback reads it
YEAR_PLAUSIBLE = 0.2

# Below this many dated records, structure decides instead
MIN_DATED = 30
# No dated records means never rejected, only unconfirmed
AGREE_CONFIRM, AGREE_REJECT = 0.70, 0.30
# Accessioning and recording can fall in different years
AGREE_TOLERANCE = 1
# Production after accession is impossible; this share rejects a scheme
IMPOSSIBLE_MAX = 0.20

# Structural fallback where too few dated records exist to test
STRUCT_MIN_YEARS = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slot_is_year(values: list[str]) -> bool:
    """Does this slot's sample read as a year: mostly four digits, and those mostly in the calendar window"""
    nums4 = [v for v in values if v.isdigit() and len(v) == YEAR_DIGITS]
    if not values or len(nums4) / len(values) < FRAC4_MIN:
        return False
    in_range = [v for v in nums4 if YEAR_MIN <= int(v) <= YEAR_MAX]
    return len(in_range) / len(nums4) >= YEARFRAC_MIN


def slot_lookup() -> pl.DataFrame:
    """(field_type, source_pattern) -> merged_pattern, from the pattern merge"""
    merged = pl.read_parquet(PATTERNS_OUT / "object_number_patterns_merged.parquet")
    return pl.DataFrame(
        [
            {"field_type": m["field_type"], "source_pattern": m["pattern"], "merged_pattern": r["merged_pattern"]}
            for r in merged.iter_rows(named=True)
            for m in json.loads(r["members"])
        ]
    )


def year_at_slot(row: dict) -> int | None:
    """The four-digit value of slot `year_slot_idx`, walking runs of d and s in order"""
    value, masked, idx = row["value"], row["masked"], row["year_slot_idx"]
    i = slot = 0
    while i < len(masked):
        c = masked[i]
        if c in "ds":
            j = i
            while j < len(masked) and masked[j] == c:
                j += 1
            if slot == idx:
                v = value[i:j]
                return int(v) if c == "d" and len(v) == YEAR_DIGITS else None
            slot += 1
            i = j
        else:
            i += 1
    return None


def year_slots() -> pl.DataFrame:
    """(institution, pattern) pairs whose digit slot reads as a year, from the slot-value export"""
    slots = pl.read_parquet(PATTERNS_OUT / "object_number_institutional_slot_values.parquet").filter(
        pl.col("slot_kind") == "d"
    )
    rows = [
        {"data_source": r["data_source"], "merged_pattern": r["merged_pattern"], "year_slot_idx": r["slot_idx"]}
        for r in slots.iter_rows(named=True)
        if slot_is_year(r["values"])
    ]
    return pl.DataFrame(
        rows, schema={"data_source": pl.String, "merged_pattern": pl.String, "year_slot_idx": pl.Int32}
    )


def objnum_years(institutions: list[str] | None = None) -> pl.DataFrame:
    """The year each record's object number encodes, wherever its pattern carries a year-like slot"""
    scope = pl.col("data_source").cast(pl.String)
    rows = pl.scan_parquet(RAW_RECORDS).filter(
        (pl.col("field_type") == "spectrum/object_number") & pl.col("value").is_not_null()
    )
    if institutions is not None:
        rows = rows.filter(scope.is_in(institutions))
    return (
        rows.with_columns(scope, masked=MASK)
        .with_columns(source_pattern=pl.col("masked").map_elements(signature_of, return_dtype=pl.String))
        .join(slot_lookup().lazy(), on=["field_type", "source_pattern"])
        .join(year_slots().lazy(), on=["data_source", "merged_pattern"])
        .with_columns(
            year=pl.struct("value", "masked", "year_slot_idx").map_elements(year_at_slot, return_dtype=pl.Int32)
        )
        .filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX))
        .group_by("record_id")
        .agg(pl.col("data_source").first(), pl.col("year").min(), pl.col("value").first())
        .collect(engine="streaming")
    )


def field_years(field: str) -> pl.DataFrame:
    """The earliest four-digit year each record records in one date field"""
    return (
        pl.scan_parquet(RAW_RECORDS)
        .filter((pl.col("field_type") == field) & pl.col("value").is_not_null())
        .select("record_id", pl.col("data_source").cast(pl.String), yrs=pl.col("value").str.extract_all(r"\d{4}"))
        .explode("yrs")
        .drop_nulls("yrs")
        .with_columns(year=pl.col("yrs").cast(pl.Int32))
        .filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX))
        .group_by("record_id")
        .agg(pl.col("data_source").first(), pl.col("year").min())
        .collect(engine="streaming")
    )


def year_coverage() -> pl.DataFrame:
    """How much of each institution's numbering sits in a pattern that carries a year-like slot"""
    slots = pl.read_parquet(PATTERNS_OUT / "object_number_institutional_slot_values.parquet").filter(
        pl.col("slot_kind") == "d"
    )
    year_like = year_slots().select("data_source", "merged_pattern").unique()
    # A pattern never inspected is not evidence against the institution
    inspected = slots.select("data_source", "merged_pattern").unique()

    dist = pl.read_parquet(PATTERNS_OUT / "object_number_institutional_pattern_dist.parquet")
    flagged = (
        dist.join(year_like.with_columns(year_enc=pl.lit(True)), on=["data_source", "merged_pattern"], how="left")
        .join(inspected.with_columns(seen=pl.lit(True)), on=["data_source", "merged_pattern"], how="left")
        .with_columns(pl.col("year_enc").fill_null(False), pl.col("seen").fill_null(False))
    )
    return (
        flagged.group_by("data_source")
        .agg(
            n_records=pl.col("count").sum().cast(pl.UInt32),
            year_coverage=pl.col("count").filter("year_enc").sum() / pl.col("count").sum(),
            inspected_share=pl.col("count").filter("seen").sum() / pl.col("count").sum(),
            n_year_patterns=pl.col("merged_pattern").filter("year_enc").n_unique().cast(pl.UInt32),
        )
        .with_columns(year_coverage=pl.col("year_coverage").fill_null(0.0))
        .sort("year_coverage", descending=True)
    )


def differences(objnum: pl.DataFrame) -> pl.DataFrame:
    """Per record, the object-number year set against the recorded accession year and the production year"""
    recorded = (
        pl.concat(
            [
                field_years("spectrum/accession_date").with_columns(source=pl.lit("accession_date")),
                field_years("spectrum/acquisition_date").with_columns(source=pl.lit("acquisition_date")),
            ]
        )
        .sort("source")
        .unique(subset=["record_id"], keep="first")
        .rename({"year": "recorded_year"})
        .drop("data_source")
    )
    production = field_years("spectrum/object_production_date").rename({"year": "production_year"}).drop("data_source")
    return (
        objnum.join(recorded, on="record_id", how="left")
        .join(production, on="record_id", how="left")
        .with_columns(
            recorded_diff=pl.col("year") - pl.col("recorded_year"),
            production_diff=pl.col("year") - pl.col("production_year"),
        )
    )


def structural(objnum: pl.DataFrame) -> pl.DataFrame:
    """The two tests a scheme with too few dated records can still be held to"""
    return (
        objnum.with_columns(
            # A within-year scheme restarts the sequence rather than growing it
            tail=pl.col("value").str.extract(r"(?:19|20)\d{2}\D+(\d+)", 1).cast(pl.Int64, strict=False)
        )
        .group_by("data_source")
        .agg(
            n_records=pl.len(),
            n_years=pl.col("year").n_unique(),
            year_span=pl.col("year").max() - pl.col("year").min(),
            tails=pl.col("tail"),
            years=pl.col("year"),
        )
        .with_columns(restart_rate=pl.struct("years", "tails").map_elements(_restart_rate, return_dtype=pl.Float64))
        .drop("tails", "years")
    )


def _restart_rate(row: dict) -> float:
    """Share of consecutive year pairs where the sequence number falls back rather than carrying on"""
    pairs: dict[int, int] = {}
    for year, tail in zip(row["years"], row["tails"], strict=True):
        if tail is not None:
            pairs[year] = max(pairs.get(year, 0), tail)
    steps = list(pairwise(pairs[y] for y in sorted(pairs)))
    if not steps:
        return 0.0
    return sum(1 for a, b in steps if b < a) / len(steps)


def verdicts(diffs: pl.DataFrame, struct: pl.DataFrame, coverage: pl.DataFrame) -> pl.DataFrame:
    """One verdict per institution: confirmed on agreement, rejected only on evidence against"""
    agreement = (
        diffs.group_by("data_source")
        .agg(
            n_dated=pl.col("recorded_diff").drop_nulls().len(),
            agree=(pl.col("recorded_diff").abs() <= AGREE_TOLERANCE).mean(),
            early=(pl.col("recorded_diff") < -AGREE_TOLERANCE).mean(),
            late=(pl.col("recorded_diff") > AGREE_TOLERANCE).mean(),
            n_production=pl.col("production_diff").drop_nulls().len(),
            impossible=(pl.col("production_diff") < 0).mean(),
        )
        .with_columns(pl.col("agree", "early", "late", "impossible").fill_null(0.0))
    )
    joined = coverage.join(agreement, on="data_source", how="left").join(struct, on="data_source", how="left")
    tested = pl.col("n_dated") >= MIN_DATED
    return joined.with_columns(
        verdict=pl.when(tested & (pl.col("agree") >= AGREE_CONFIRM) & (pl.col("impossible") <= IMPOSSIBLE_MAX))
        .then(pl.lit("confirmed"))
        .when(tested & ((pl.col("agree") <= AGREE_REJECT) | (pl.col("impossible") > IMPOSSIBLE_MAX)))
        .then(pl.lit("rejected"))
        .when(tested)
        .then(pl.lit("plausible"))
        # no dated records: only the number structure remains
        .when((pl.col("n_years").fill_null(0) >= STRUCT_MIN_YEARS) & (pl.col("year_coverage") >= YEAR_PLAUSIBLE))
        .then(pl.lit("plausible"))
        .otherwise(pl.lit("rejected")),
        evidence=pl.when(tested).then(pl.lit("dated records")).otherwise(pl.lit("structure")),
    ).sort("verdict", "year_coverage", descending=[False, True])


def main() -> None:
    ap = argparse.ArgumentParser(description="Whether each institution's object numbers encode an accession year.")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    coverage = year_coverage()
    log(f"{coverage.height} institutions carry a year-like object-number pattern")

    objnum = objnum_years()
    log(f"{objnum.height:,} records carry a year in their object number")

    diffs = differences(objnum)
    diffs.write_parquet(DIFFS_PATH)
    log(
        f"{diffs['recorded_diff'].drop_nulls().len():,} records also record an accession year; "
        f"{diffs['production_diff'].drop_nulls().len():,} a production year → {DIFFS_PATH}"
    )

    out = verdicts(diffs, structural(objnum), coverage)
    out.write_parquet(args.out)
    counts = dict(out.group_by("verdict").len().iter_rows())
    log(f"verdicts {counts} → {args.out}")
    with pl.Config(tbl_rows=20, fmt_str_lengths=40):
        print(out.select("data_source", "n_records", "year_coverage", "n_dated", "agree", "impossible", "verdict"))


if __name__ == "__main__":
    main()
