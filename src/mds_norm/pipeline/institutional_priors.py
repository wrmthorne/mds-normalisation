from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from codecarbon import track_emissions
from mds_data_model.introspection import date_fields

from mds_norm.parsers.parse_dates import Conventions, parse_date
from mds_norm.paths import EMISSIONS_LOG, INSTITUTIONAL, PATTERNS_OUT, RAW_RECORDS
from mds_norm.pipeline.institutional_fingerprints import FP_KINDS

RAW_PATH = RAW_RECORDS
PATTERNS = PATTERNS_OUT
OUT_PATH = INSTITUTIONAL / "date_conventions.parquet"
OUT_UNITS = INSTITUTIONAL / "unit_conventions.parquet"
OUT_SLOT_ROLES = INSTITUTIONAL / "slot_roles.parquet"
NEIGHBOURS_PATH = INSTITUTIONAL / "fingerprint_neighbours.parquet"
# Past this Jensen-Shannon distance no convention carries across
TRANSFER_MAX_DISTANCE = 0.6
EMISSIONS_LOG_PATH = EMISSIONS_LOG

# A slot regularly exceeding 12 cannot be a month
MIN_SLOT_N = 50
GT12_FRAC = 0.05
GT31_FRAC = 0.01
MAX_MONTH, MAX_DAY = 12, 31

# Three-component numeric families: two short slots and a year anchor, with one separator throughout. Any short-slot
# width qualifies, so a shape whose day and month are always two digits is tested alongside one where they vary
SHORT_SLOT, YEAR_SLOT, SEPARATOR = r"d\{(?:1|2|1,2)\}", r"d\{(?:4|3,4)\}", r"([./-])"
FAMILIES = {
    "year_last": (re.compile(rf"^{SHORT_SLOT}{SEPARATOR}{SHORT_SLOT}\1{YEAR_SLOT}$"), [0, 1], 2),
    "year_first": (re.compile(rf"^{YEAR_SLOT}{SEPARATOR}{SHORT_SLOT}\1{SHORT_SLOT}$"), [1, 2], 0),
}

# Zero-placeholder convention: a short slot mostly reading 0
ZERO_FRAC_MIN = 0.25

# '=' separates a range only where the pairs ascend
EQ_MIN_OCC = 5
EQ_MIN_FRAC = 0.9

# A prime verdict needs near-unanimous occurrences matching one reading
PRIME_MIN_OCC = 5
PRIME_MIN_FRAC = 0.9
PRIME_TOL = 0.05

# Manual verdicts where the evidence sits below the floors
MANUAL_PRIME = {"Royal Armouries": "in"}

# Families whose slot cells are read for their roles
PATTERN_FAMILIES = ("date", "dimension", "money", "object_number")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slot_stats(values: list[str]) -> dict:
    x = np.array([int(v) for v in values if v.isdigit()], dtype=np.int64)
    if len(x) == 0:
        return {"n": 0, "max": 0, "frac_gt12": 0.0, "frac_gt31": 0.0, "frac_zero": 0.0}
    return {
        "n": len(x),
        "max": int(x.max()),
        "frac_gt12": float((x > MAX_MONTH).mean()),
        "frac_gt31": float((x > MAX_DAY).mean()),
        "frac_zero": float((x == 0).mean()),
    }


def slot_role(s: dict | None) -> str | None:
    if s is None or s["n"] < MIN_SLOT_N:
        return None
    if s["frac_gt31"] > GT31_FRAC:  # truncated year leaking in, not day/month
        return "year?"
    if s["frac_gt12"] > GT12_FRAC:  # exceeds 12 -> must be day-of-month
        return "day"
    return "month"  # bounded to 12 -> month


# Terms kept per letter slot read as vocabulary
SLOT_TERMS = 10


def letter_slot_terms(values: list[str]) -> dict:
    """A letter slot read as a vocabulary: how many terms it draws on and which dominate"""
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:SLOT_TERMS]
    n = len(values)
    return {"n_terms": len(counts), "top_terms": [t for t, _ in top], "top_share": top[0][1] / n if top else 0.0}


def slot_roles(family: str) -> pl.DataFrame:
    """What each (institution, pattern, slot) cell holds, read from the cell's own value distribution"""
    cells = pl.read_parquet(PATTERNS / f"{family}_institutional_slot_values.parquet")
    rows = []
    for r in cells.iter_rows(named=True):
        row = {
            "family": family,
            "data_source": r["data_source"],
            "merged_pattern": r["merged_pattern"],
            "slot_idx": r["slot_idx"],
            "slot_kind": r["slot_kind"],
        }
        if r["slot_kind"] == "d":
            s = slot_stats(r["values"])
            role = slot_role(s)
            rows.append(
                row
                | {
                    "n": s["n"],
                    "role": role,
                    "max": s["max"],
                    "frac_gt12": s["frac_gt12"],
                    "frac_gt31": s["frac_gt31"],
                    "frac_zero": s["frac_zero"],
                    "n_terms": None,
                    "top_terms": None,
                    "top_share": None,
                }
            )
        else:
            v = letter_slot_terms(r["values"])
            rows.append(
                row
                | {
                    "n": len(r["values"]),
                    "role": "vocabulary",
                    "max": None,
                    "frac_gt12": None,
                    "frac_gt31": None,
                    "frac_zero": None,
                    "n_terms": v["n_terms"],
                    "top_terms": v["top_terms"],
                    "top_share": v["top_share"],
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "family": pl.String,
            "data_source": pl.String,
            "merged_pattern": pl.String,
            "slot_idx": pl.Int32,
            "slot_kind": pl.String,
            "n": pl.Int64,
            "role": pl.String,
            "max": pl.Int64,
            "frac_gt12": pl.Float64,
            "frac_gt31": pl.Float64,
            "frac_zero": pl.Float64,
            "n_terms": pl.Int64,
            "top_terms": pl.List(pl.String),
            "top_share": pl.Float64,
        },
    )


def load_slot_values() -> pl.DataFrame:
    return pl.read_parquet(PATTERNS / "date_institutional_slot_values.parquet").with_columns(
        pl.col("merged_pattern").str.replace_all(r"[\[\]?]", "").str.strip_chars().alias("shape")
    )


def date_orders(inst: pl.DataFrame) -> pl.DataFrame:
    """One row per institution: day/month-order verdict per family"""
    rows: dict[str, dict] = {}
    for family, (pattern, short, year) in FAMILIES.items():
        # Matched in Python because the separator backreference is beyond the regex engine polars uses
        shapes = [s for s in inst["shape"].unique() if pattern.match(s)]
        pooled = (
            inst.filter(pl.col("shape").is_in(shapes))
            .group_by("data_source", "slot_idx")
            .agg(pl.col("values").list.explode())
        )
        stats = {(r["data_source"], r["slot_idx"]): slot_stats(r["values"]) for r in pooled.iter_rows(named=True)}
        for src in {s for s, _ in stats}:
            roles = {year: "Y"}
            for idx in short:
                roles[idx] = {"day": "D", "month": "M"}.get(slot_role(stats.get((src, idx))))
            order = "".join(roles[i] for i in sorted(roles)) if all(roles.values()) else None
            if order and (order.count("D") != 1 or order.count("M") != 1):
                order = None
            n = min((stats[(src, i)]["n"] for i in short if (src, i) in stats), default=0)
            rows.setdefault(src, {"data_source": src})
            rows[src][f"order_{family}"] = order
            rows[src][f"n_{family}"] = n
    return pl.DataFrame(
        list(rows.values()),
        schema={
            "data_source": pl.String,
            "order_year_last": pl.String,
            "n_year_last": pl.Int64,
            "order_year_first": pl.String,
            "n_year_first": pl.Int64,
        },
    )


def zero_placeholders(inst: pl.DataFrame) -> pl.DataFrame:
    """Institutions with a non-year date slot dominated by 0/00"""
    rows = []
    for r in inst.filter((pl.col("slot_kind") == "d") & ~pl.col("shape").str.contains(":")).iter_rows(named=True):
        s = slot_stats(r["values"])
        if s["n"] < MIN_SLOT_N or s["frac_zero"] < ZERO_FRAC_MIN:
            continue
        nz = [int(v) for v in r["values"] if v.isdigit() and int(v) > 0]
        if nz and max(nz) > MAX_DAY:  # year slot, not a day/month position
            continue
        rows.append(
            {
                "data_source": r["data_source"],
                "zero_frac": s["frac_zero"],
                "zero_cell": f"{r['shape']} slot {r['slot_idx']}",
                "zero_n": s["n"],
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={"data_source": pl.String, "zero_frac": pl.Float64, "zero_cell": pl.String, "zero_n": pl.Int64}
        )
    return pl.DataFrame(rows).sort("zero_frac", descending=True).unique(subset=["data_source"], keep="first")


def eq_rangelike(value: str) -> bool | None:
    """Does '=' read as an ascending date range in this value?"""
    try:
        with_eq = parse_date(value.strip(), Conventions(eq_range=True))
        without = parse_date(value.strip(), Conventions())
    except Exception:
        return None
    if with_eq is None or with_eq == without:
        return None
    # The parser orders endpoints, so read ascendingness as written
    before, sep, after = value.partition("=")
    if not sep:
        return None
    sides = [parse_date(part.strip(), Conventions()) for part in (before, after)]
    if any(s is None for s in sides):
        return None
    lo, hi = (s.get("date_earliest_single") for s in sides)
    return lo is not None and hi is not None and not lo.startswith("-") and lo <= hi


def eq_verdicts() -> pl.DataFrame:
    """Per institution: does '=' behave as a range separator in its date fields?"""
    eq_vals = (
        pl.scan_parquet(RAW_PATH)
        .filter(pl.col("field_type").is_in(list(date_fields())) & pl.col("value").str.contains("="))
        .group_by(pl.col("data_source").cast(pl.String), "value")
        .agg(pl.len().alias("count"))
        .collect(engine="streaming")
    )
    verdict = {v: eq_rangelike(v) for v in eq_vals["value"].unique()}
    return (
        eq_vals.with_columns(rangelike=pl.col("value").replace_strict(verdict, return_dtype=pl.Boolean))
        .filter(pl.col("rangelike").is_not_null())
        .group_by("data_source")
        .agg(
            eq_occ=pl.col("count").sum(), eq_frac=(pl.col("count") * pl.col("rangelike")).sum() / pl.col("count").sum()
        )
        .with_columns(eq_means_range=(pl.col("eq_occ") >= EQ_MIN_OCC) & (pl.col("eq_frac") >= EQ_MIN_FRAC))
    )


# a primed number with a metric gloss ("2.98' (75.7mm)")
_PRIME_DUAL = re.compile(r"(\d+(?:\.\d+)?)\s*['′‘’]\s*\(\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)\b\s*\)")
_METRIC_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
_PRIME_MM = {"in": 25.4, "ft": 304.8}


def prime_evidence(value: str) -> str | None:
    """'in' or 'ft' when every dual annotation in the value agrees on exactly one reading of the prime"""
    verdicts = set()
    for x, y, unit in _PRIME_DUAL.findall(value):
        target = float(y) * _METRIC_MM[unit]
        if target == 0:
            return None
        fits = [u for u, mm in _PRIME_MM.items() if abs(float(x) * mm - target) / target <= PRIME_TOL]
        if len(fits) != 1:
            return None
        verdicts.add(fits[0])
    return verdicts.pop() if len(verdicts) == 1 else None


def unit_verdicts() -> pl.DataFrame:
    """Per institution - what a prime after a digit means in dimension values"""
    vals = (
        pl.scan_parquet(RAW_PATH)
        .filter(
            pl.col("field_type").is_in(["spectrum/dimension", "spectrum/dimension_value"])
            & pl.col("value").str.contains("['′‘’]")
        )
        .group_by(pl.col("data_source").cast(pl.String), "value")
        .agg(pl.len().alias("count"))
        .collect(engine="streaming")
    )
    verdict = {v: prime_evidence(v) for v in vals["value"].unique()}
    derived = (
        vals.with_columns(reading=pl.col("value").replace_strict(verdict, return_dtype=pl.String))
        .filter(pl.col("reading").is_not_null())
        .group_by("data_source")
        .agg(
            prime_occ=pl.col("count").sum(),
            in_frac=(pl.col("count") * (pl.col("reading") == "in")).sum() / pl.col("count").sum(),
        )
        .with_columns(
            prime_unit=pl.when(pl.col("in_frac") >= PRIME_MIN_FRAC)
            .then(pl.lit("in"))
            .when(pl.col("in_frac") <= 1 - PRIME_MIN_FRAC)
            .then(pl.lit("ft"))
            .otherwise(pl.lit(None)),
            source=pl.lit("derived"),
        )
        .filter((pl.col("prime_occ") >= PRIME_MIN_OCC) & pl.col("prime_unit").is_not_null())
    )
    manual = (
        pl.DataFrame({"data_source": list(MANUAL_PRIME), "prime_unit": list(MANUAL_PRIME.values())})
        .with_columns(
            prime_occ=pl.lit(None, dtype=pl.UInt32), in_frac=pl.lit(None, dtype=pl.Float64), source=pl.lit("manual")
        )
        .select(derived.columns)
    )
    clash = derived.join(manual, on="data_source", how="inner", suffix="_manual").filter(
        pl.col("prime_unit") != pl.col("prime_unit_manual")
    )
    if clash.height:
        raise SystemExit(f"manual prime verdicts contradict derived evidence:\n{clash}")
    return pl.concat([derived, manual.join(derived, on="data_source", how="anti")]).sort("data_source")


TRANSFERABLE = ("dm_order", "zero_placeholder", "eq_means_range")


def transfer_conventions(conv: pl.DataFrame, kind: str, max_distance: float, neighbours: Path) -> pl.DataFrame:
    """Fill a convention from the nearest fingerprint neighbour; off by default, measured not to help"""
    ranked = (
        pl.read_parquet(neighbours)
        .filter((pl.col("kind") == kind) & (pl.col("js_distance") <= max_distance))
        .sort("data_source", "rank")
    )
    out = conv
    for column in TRANSFERABLE:
        donors = out.filter(pl.col(column).is_not_null()).select(neighbour="data_source", donor_value=column)
        filled = (
            ranked.join(donors, on="neighbour", how="inner")
            .sort("data_source", "rank")
            .group_by("data_source", maintain_order=True)
            .agg(pl.col("donor_value").first(), pl.col("neighbour").first())
        )
        out = (
            out.join(filled, on="data_source", how="left")
            .with_columns(
                pl.when(pl.col(column).is_null())
                .then(pl.col("neighbour"))
                .otherwise(pl.lit(None))
                .alias(f"{column}_source"),
                pl.coalesce(column, "donor_value").alias(column),
            )
            .drop("donor_value", "neighbour")
        )
        log(
            f"transferred {column}: {out[f'{column}_source'].is_not_null().sum()} institutions filled from a neighbour"
        )
    return out


@track_emissions(project_name="institutional_priors", output_dir=str(EMISSIONS_LOG_PATH), log_level="error")
def main() -> None:
    ap = argparse.ArgumentParser(description="Per-institution date and unit conventions, derived from the raw corpus.")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--out-units", type=Path, default=OUT_UNITS)
    ap.add_argument("--out-slot-roles", type=Path, default=OUT_SLOT_ROLES)
    ap.add_argument(
        "--transfer",
        choices=FP_KINDS,
        help="fill missing conventions from the nearest institution under this fingerprint (off by default)",
    )
    ap.add_argument("--transfer-max-distance", type=float, default=TRANSFER_MAX_DISTANCE)
    args = ap.parse_args()

    inst = load_slot_values()
    log(f"slot-value cells: {inst.height:,} across {inst['data_source'].n_unique()} institutions")

    orders = date_orders(inst)
    zeros = zero_placeholders(inst)
    eq = eq_verdicts()
    log(
        f"order verdicts: {orders.filter(pl.col('order_year_last').is_not_null()).height} "
        f"year-last, {orders.filter(pl.col('order_year_first').is_not_null()).height} "
        f"year-first; zero-placeholder: {zeros.height}; '=' evidence: {eq.height}"
    )

    conv = (
        orders.join(zeros, on="data_source", how="full", coalesce=True)
        .join(eq, on="data_source", how="full", coalesce=True)
        .with_columns(
            # Day/month order comes from the year-last family's written order
            dm_order=pl.col("order_year_last").replace_strict({"DMY": "DM", "MDY": "MD"}, default=None),
            zero_placeholder=pl.col("zero_frac").is_not_null(),
        )
        .sort("data_source")
    )
    if args.transfer:
        conv = transfer_conventions(conv, args.transfer, args.transfer_max_distance, NEIGHBOURS_PATH)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    conv.write_parquet(args.out)
    log(f"{conv.height} institutions → {args.out}")

    roles = pl.concat([slot_roles(f) for f in PATTERN_FAMILIES])
    args.out_slot_roles.parent.mkdir(parents=True, exist_ok=True)
    roles.write_parquet(args.out_slot_roles)
    counts = dict(roles.group_by("role").len().iter_rows())
    log(f"slot roles: {counts} over {roles.height:,} cells → {args.out_slot_roles}")

    units = unit_verdicts()
    args.out_units.parent.mkdir(parents=True, exist_ok=True)
    units.write_parquet(args.out_units)
    log(f"unit-symbol verdicts: {units.height} institutions → {args.out_units}")
    with pl.Config(fmt_str_lengths=45):
        print(units)
    with pl.Config(tbl_rows=20, fmt_str_lengths=45):
        print(
            conv.filter(
                pl.col("dm_order").is_not_null()
                | pl.col("zero_placeholder")
                | pl.col("eq_means_range").fill_null(False)
            ).select(
                "data_source",
                "dm_order",
                "n_year_last",
                "order_year_first",
                "zero_placeholder",
                "zero_frac",
                "eq_means_range",
                "eq_occ",
                "eq_frac",
            )
        )


if __name__ == "__main__":
    main()
