from __future__ import annotations

import argparse

import polars as pl

from experiments.harness import EXP_OUT, ROOT, Variant, log, measure, write_run
from mds_norm.parsers.parse_dates import Conventions, parse_date
from mds_norm.pipeline.compile_records import DATE_FIELDS
from mds_norm.pipeline.institutional_priors import (
    EQ_MIN_FRAC,
    EQ_MIN_OCC,
    GT12_FRAC,
    GT31_FRAC,
    MIN_SLOT_N,
    ZERO_FRAC_MIN,
    eq_rangelike,
)

EXP = "fingerprint_routing"
RAW = ROOT / "data" / "mds-flat-records.parquet"
INSTITUTIONAL = ROOT / "data" / "institutional"
PROD_CACHE = ROOT / "data" / "compiled" / "date_parse_cache.parquet"
EXP_CACHE = EXP_OUT / EXP / "parse_cache.parquet"

CONV_KEYS = ["dm_order", "eq", "z0"]

# Slot-role bounds, as in institutional_priors
MONTH_MAX = 12
DAY_MAX = 31

# Two numeric families with a 4-digit year anchor, per separator
YEAR_LAST = [rf"^\s*(\d{{1,2}}){s}(\d{{1,2}}){s}\d{{4}}\s*$" for s in ("/", r"\.", "-")]
YEAR_FIRST = [rf"^\s*\d{{4}}{s}(\d{{1,2}}){s}(\d{{1,2}})\s*$" for s in ("-", r"\.", "/")]


def date_nodes() -> pl.DataFrame:
    """Date-field occurrence counts per (institution, stratum, value)"""
    strata_years = (
        pl.read_parquet(INSTITUTIONAL / "practice_strata.parquet")
        .select("data_source", "stratum_idx", accession_year=pl.int_ranges("start_year", pl.col("end_year") + 1))
        .explode("accession_year")
    )
    rec_stratum = (
        pl.read_parquet(INSTITUTIONAL / "accession_years.parquet")
        .join(strata_years, on=["data_source", "accession_year"])
        .select("record_id", "stratum_idx")
    )
    return (
        pl.scan_parquet(RAW)
        .filter(pl.col("field_type").is_in(list(DATE_FIELDS)) & pl.col("value").is_not_null())
        .select("record_id", pl.col("data_source").cast(pl.String), "value")
        .join(rec_stratum.lazy(), on="record_id", how="left")
        .group_by("data_source", "stratum_idx", "value")
        .agg(n_occ=pl.len())
        .collect(engine="streaming")
    )


def stratum_conventions(nodes: pl.DataFrame) -> pl.DataFrame:
    """R2's prior table: dm_order / eq / z0 per (institution, stratum)"""
    dated = nodes.filter(pl.col("stratum_idx").is_not_null())

    def family_slots(regexes: list[str]) -> pl.DataFrame:
        """Short-slot values per (institution, stratum) with their occurrence weight"""
        parts = []
        for rx in regexes:
            m = (
                dated.with_columns(g=pl.col("value").str.extract_groups(rx))
                .filter(pl.col("g").struct["1"].is_not_null())
                .unnest("g")
            )
            parts += [
                m.select("data_source", "stratum_idx", "n_occ", slot=pl.lit(i), val=pl.col(str(i + 1)).cast(pl.Int32))
                for i in (0, 1)
            ]
        return pl.concat(parts)

    # dm order from the year-last family, occurrence-weighted per slot
    roles = (
        family_slots(YEAR_LAST)
        .group_by("data_source", "stratum_idx", "slot")
        .agg(
            n=pl.col("n_occ").sum(),
            gt12=(pl.col("n_occ") * (pl.col("val") > MONTH_MAX)).sum(),
            gt31=(pl.col("n_occ") * (pl.col("val") > DAY_MAX)).sum(),
        )
        .with_columns(
            role=pl.when(pl.col("n") < MIN_SLOT_N)
            .then(pl.lit(None))
            .when(pl.col("gt31") / pl.col("n") > GT31_FRAC)
            .then(pl.lit("year?"))
            .when(pl.col("gt12") / pl.col("n") > GT12_FRAC)
            .then(pl.lit("day"))
            .otherwise(pl.lit("month"))
        )
    )
    dm = (
        roles.select("data_source", "stratum_idx", "slot", "role")
        .pivot(on="slot", index=["data_source", "stratum_idx"], values="role")
        .rename({"0": "role0", "1": "role1"})
        .with_columns(
            dm_order=pl.when((pl.col("role0") == "day") & (pl.col("role1") == "month"))
            .then(pl.lit("DM"))
            .when((pl.col("role0") == "month") & (pl.col("role1") == "day"))
            .then(pl.lit("MD"))
            .otherwise(pl.lit(None))
        )
        .select("data_source", "stratum_idx", "dm_order")
    )

    # Zero placeholders are positive evidence only; silence is no verdict
    zero = (
        pl.concat([family_slots(YEAR_LAST), family_slots(YEAR_FIRST)])
        .group_by("data_source", "stratum_idx", "slot")
        .agg(n=pl.col("n_occ").sum(), zeros=(pl.col("n_occ") * (pl.col("val") == 0)).sum())
        .filter(pl.col("n") >= MIN_SLOT_N)
        .with_columns(frac=pl.col("zeros") / pl.col("n"))
        .group_by("data_source", "stratum_idx")
        .agg(z0=(pl.col("frac") >= ZERO_FRAC_MIN).any())
        .filter(pl.col("z0"))
    )

    # '=' as a range separator, occurrence-weighted over distinct values
    eq_vals = dated.filter(pl.col("value").str.contains("="))
    verdict = {v: eq_rangelike(v) for v in eq_vals["value"].unique()}
    eq = (
        eq_vals.with_columns(rangelike=pl.col("value").replace_strict(verdict, return_dtype=pl.Boolean))
        .filter(pl.col("rangelike").is_not_null())
        .group_by("data_source", "stratum_idx")
        .agg(
            eq_occ=pl.col("n_occ").sum(), eq_frac=(pl.col("n_occ") * pl.col("rangelike")).sum() / pl.col("n_occ").sum()
        )
        .with_columns(
            eq=pl.when(pl.col("eq_occ") >= EQ_MIN_OCC).then(pl.col("eq_frac") >= EQ_MIN_FRAC).otherwise(pl.lit(None))
        )
        .select("data_source", "stratum_idx", "eq")
    )

    # Null means no stratum evidence; False is a measured negative
    return dm.join(zero, on=["data_source", "stratum_idx"], how="full", coalesce=True).join(
        eq, on=["data_source", "stratum_idx"], how="full", coalesce=True
    )


def institution_conventions() -> pl.DataFrame:
    """R1's prior table, as production loads it"""
    return pl.read_parquet(INSTITUTIONAL / "date_conventions.parquet").select(
        "data_source",
        dm_order=pl.col("dm_order").fill_null(""),
        eq=pl.col("eq_means_range").fill_null(False),
        z0=pl.col("zero_placeholder").fill_null(False),
    )


def parse_outcomes(combos: pl.DataFrame) -> pl.DataFrame:
    """Outcome per distinct (value, dm_order, eq, z0): resolved, qualified or unparsed"""
    schema = {
        "value": pl.String,
        "dm_order": pl.String,
        "eq": pl.Boolean,
        "z0": pl.Boolean,
        "edtf": pl.String,
        "dm_ambig": pl.Boolean,
    }
    caches = [pl.read_parquet(PROD_CACHE).select(list(schema)) if PROD_CACHE.exists() else pl.DataFrame(schema=schema)]
    if EXP_CACHE.exists():
        caches.append(pl.read_parquet(EXP_CACHE))
    cached = pl.concat(caches).unique(subset=["value", *CONV_KEYS])

    todo = combos.join(cached, on=["value", *CONV_KEYS], how="anti")
    if todo.height:
        log(f"parsing {todo.height:,} uncached (value, conventions) combos")
        rows = []
        for v, o, e, z in todo.select("value", *CONV_KEYS).iter_rows():
            try:
                p = parse_date(v, Conventions(dm_order=o or None, eq_range=e, zero_null=z))
            except Exception:
                p = None
            rows.append((v, o, e, z, p.get("value_edtf") if p else None, bool(p and p.get("dm_ambiguous"))))
        new = pl.DataFrame(rows, schema=schema, orient="row")
        EXP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        exp_cache = pl.concat([pl.read_parquet(EXP_CACHE), new]) if EXP_CACHE.exists() else new
        exp_cache.write_parquet(EXP_CACHE)
        cached = pl.concat([cached, new])

    return combos.join(cached, on=["value", *CONV_KEYS], how="left").with_columns(
        outcome=pl.when(pl.col("edtf").is_null())
        .then(pl.lit("unparsed"))
        .when(pl.col("dm_ambig"))
        .then(pl.lit("qualified"))
        .otherwise(pl.lit("resolved"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare three convention-prior regimes over the tier-1 date parse.")
    ap.parse_args()

    with measure(EXP, "regimes") as cost:
        nodes = date_nodes()
        log(
            f"population: {nodes.height:,} (institution, stratum, value) cells, "
            f"{nodes['n_occ'].sum():,} occurrences; "
            f"{nodes.filter(pl.col('stratum_idx').is_not_null())['n_occ'].sum():,} "
            f"in a dated stratum"
        )

        inst_conv = institution_conventions()
        strat_conv = stratum_conventions(nodes)
        log(
            f"stratum priors: {strat_conv.height} (institution, stratum) rows "
            f"({strat_conv['dm_order'].is_not_null().sum()} with a dm verdict, "
            f"{strat_conv['z0'].sum()} zero-placeholder, "
            f"{strat_conv['eq'].sum()} eq-range)"
        )

        # R0 defaults; R1 institution prior; R2 stratum prior with fallback
        assigned = (
            nodes.join(inst_conv.rename({k: f"{k}_r1" for k in CONV_KEYS}), on="data_source", how="left")
            .with_columns(
                pl.col("dm_order_r1").fill_null(""), pl.col("eq_r1").fill_null(False), pl.col("z0_r1").fill_null(False)
            )
            .join(strat_conv.rename({k: f"{k}_r2" for k in CONV_KEYS}), on=["data_source", "stratum_idx"], how="left")
            .with_columns(
                dm_order_r0=pl.lit(""),
                eq_r0=pl.lit(False),
                z0_r0=pl.lit(False),
                dm_order_r2=pl.coalesce("dm_order_r2", "dm_order_r1"),
                eq_r2=pl.coalesce("eq_r2", "eq_r1"),
                z0_r2=pl.coalesce("z0_r2", "z0_r1"),
            )
        )

        combos = pl.concat(
            [
                assigned.select("value", dm_order=f"dm_order_{r}", eq=f"eq_{r}", z0=f"z0_{r}")
                for r in ("r0", "r1", "r2")
            ]
        ).unique()
        outcomes = parse_outcomes(combos)

        per_regime = {}
        for r in ("r0", "r1", "r2"):
            joined = assigned.join(
                outcomes, left_on=["value", f"dm_order_{r}", f"eq_{r}", f"z0_{r}"], right_on=["value", *CONV_KEYS]
            )
            per_regime[r] = joined.select(
                "data_source", "stratum_idx", "value", "n_occ", outcome=pl.col("outcome"), edtf=pl.col("edtf")
            )
            summary = joined.group_by("outcome").agg(occ=pl.col("n_occ").sum()).sort("outcome")
            log(f"{r}: " + "; ".join(f"{o} {c:,}" for o, c in summary.iter_rows()))

        # regime disagreements: same cell, different parse or outcome
        pair_cols = ["data_source", "stratum_idx", "value", "n_occ"]
        diffs = {}
        for a, b in (("r0", "r1"), ("r1", "r2")):
            d = (
                per_regime[a]
                .rename({"outcome": f"outcome_{a}", "edtf": f"edtf_{a}"})
                .join(per_regime[b].rename({"outcome": f"outcome_{b}", "edtf": f"edtf_{b}"}), on=pair_cols)
                .filter(
                    (pl.col(f"outcome_{a}") != pl.col(f"outcome_{b}"))
                    | (pl.col(f"edtf_{a}") != pl.col(f"edtf_{b}")).fill_null(
                        pl.col(f"edtf_{a}").is_null() != pl.col(f"edtf_{b}").is_null()
                    )
                )
            )
            diffs[f"{a}_vs_{b}"] = d
            log(f"{a}→{b}: {d.height:,} cells / {d['n_occ'].sum():,} occurrences change parse or status")
        disagreements = pl.concat(
            [
                d.with_columns(pair=pl.lit(k)).select(
                    "pair",
                    *pair_cols,
                    outcome_a=pl.col(f"outcome_{k.split('_vs_')[0]}"),
                    outcome_b=pl.col(f"outcome_{k.split('_vs_')[1]}"),
                    edtf_a=pl.col(f"edtf_{k.split('_vs_')[0]}"),
                    edtf_b=pl.col(f"edtf_{k.split('_vs_')[1]}"),
                )
                for k, d in diffs.items()
            ]
        )

    out = EXP_OUT / EXP
    out.mkdir(parents=True, exist_ok=True)
    strat_conv.write_parquet(out / "stratum_conventions.parquet")
    disagreements.write_parquet(out / "regime_disagreements.parquet")

    regime_occ = pl.concat(
        [
            per_regime[r]
            .group_by("data_source", "outcome")
            .agg(occ=pl.col("n_occ").sum())
            .with_columns(regime=pl.lit(r))
            for r in per_regime
        ]
    )
    metrics = {
        "population_occ": int(nodes["n_occ"].sum()),
        "stratified_occ": int(nodes.filter(pl.col("stratum_idx").is_not_null())["n_occ"].sum()),
        "strata_with_priors": strat_conv.height,
        **{
            f"{r}_{o}": int(c)
            for r in per_regime
            for o, c in per_regime[r].group_by("outcome").agg(pl.col("n_occ").sum()).iter_rows()
        },
        **{f"diff_{k}_occ": int(d["n_occ"].sum()) for k, d in diffs.items()},
        **cost,
    }
    write_run(EXP, Variant(name="regimes", model=None, params={"regimes": ["r0", "r1", "r2"]}), regime_occ, metrics)


if __name__ == "__main__":
    main()
