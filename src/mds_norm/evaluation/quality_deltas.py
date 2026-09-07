from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from mds_norm.metrics import completeness, conformance, consistency, load_base, thinness
from mds_norm.paths import COMPILED, CONSISTENCY_OUT, EVAL_OUT, METRICS_OUT, RAW_RECORDS

RAW_PATH = RAW_RECORDS
COMPILED_PATH = COMPILED / "mds-normalised.parquet"
FROZEN = METRICS_OUT
WEIGHTS = FROZEN / "field_weights.parquet"
PROPENSITY = FROZEN / "decomposition_propensity.parquet"
# the head is re-induced per corpus, so counts are too
CONSISTENCY_DIR = CONSISTENCY_OUT
CONSISTENCY = {
    "raw": CONSISTENCY_DIR / "consistency_record_raw.parquet",
    "compiled": CONSISTENCY_DIR / "consistency_record_compiled.parquet",
}
OUT_DIR = EVAL_OUT

# metric column -> axis name; falling thinness is an improvement
AXES = {
    "completeness": "completeness",
    "thinness": "thinness",
    "decomposition_rate": "decomposition",
    "conformance": "conformance",
    "consistency": "consistency",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_record(
    path: Path, weights: pl.DataFrame, propensity: pl.DataFrame, consistency_path: Path | None
) -> pl.DataFrame:
    """All axis columns joined per (record_id, data_source) for one corpus"""
    base = load_base(path)

    def to_str(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("data_source").cast(pl.String))

    c = to_str(completeness.compute(base, weights=weights))
    t = to_str(
        thinness.compute(base, propensity=propensity).select(
            "record_id", "data_source", "thinness", "decomposition_rate"
        )
    )
    f = to_str(conformance.compute(base).select("record_id", "data_source", "conformance"))
    out = c.join(t, on=["record_id", "data_source"], how="full", coalesce=True).join(
        f, on=["record_id", "data_source"], how="full", coalesce=True
    )
    if consistency_path and consistency_path.exists():
        s = to_str(consistency.compute(consistency_path).select("record_id", "data_source", "consistency"))
        out = out.join(s, on=["record_id", "data_source"], how="full", coalesce=True)
    else:
        out = out.with_columns(consistency=pl.lit(None, dtype=pl.Float64))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = pl.read_parquet(WEIGHTS)
    propensity = pl.read_parquet(PROPENSITY)

    log(f"scoring raw corpus {RAW_PATH.name}…")
    before = per_record(RAW_PATH, weights, propensity, CONSISTENCY["raw"])
    log(f"raw: {before.height:,} scored records")
    log(f"scoring compiled corpus {COMPILED_PATH.name}…")
    after = per_record(COMPILED_PATH, weights, propensity, CONSISTENCY["compiled"])
    log(f"compiled: {after.height:,} scored records")
    consistency_ran = CONSISTENCY["raw"].exists() and CONSISTENCY["compiled"].exists()

    # pair on the shared record population so deltas are like-for-like
    metric_cols = list(AXES)
    paired = before.select("record_id", "data_source", *metric_cols).join(
        after.select("record_id", "data_source", *metric_cols),
        on=["record_id", "data_source"],
        how="inner",
        suffix="_after",
    )
    log(f"paired records: {paired.height:,}")

    # the paired delta covers only records defined on both sides
    corpus = {}
    for col, axis in AXES.items():
        b_def = paired[col].is_not_null()
        a_def = paired[f"{col}_after"].is_not_null()
        both = paired.filter(b_def & a_def)
        pb, pa = both[col].mean(), both[f"{col}_after"].mean()
        corpus[axis] = {
            "before_mean": paired[col].mean(),
            "before_n": int(b_def.sum()),
            "after_mean": paired[f"{col}_after"].mean(),
            "after_n": int(a_def.sum()),
            "paired_before": pb,
            "paired_after": pa,
            "paired_delta": (pa - pb) if (pa is not None and pb is not None) else None,
            "n_both_defined": both.height,
            "gained_definition": int((~b_def & a_def).sum()),
            "lost_definition": int((b_def & ~a_def).sum()),
        }

    aggs = []
    for col, axis in AXES.items():
        both = pl.col(col).is_not_null() & pl.col(f"{col}_after").is_not_null()
        aggs += [
            pl.col(col).filter(both).mean().alias(f"{axis}_before"),
            pl.col(f"{col}_after").filter(both).mean().alias(f"{axis}_after"),
            both.sum().alias(f"{axis}_n_both"),
        ]
    per_inst = paired.group_by("data_source").agg(pl.len().alias("n_records"), *aggs).sort("data_source")
    for axis in AXES.values():
        per_inst = per_inst.with_columns((pl.col(f"{axis}_after") - pl.col(f"{axis}_before")).alias(f"{axis}_delta"))

    per_inst.write_parquet(OUT_DIR / "quality_deltas_per_institution.parquet")
    # the fill delta excludes records the pipeline made core-complete
    comp = corpus["completeness"]
    completeness_components = {
        "fill_paired_delta": comp["paired_delta"],
        "fill_n_paired": comp["n_both_defined"],
        "coverage_gained_records": comp["gained_definition"],
        "coverage_lost_records": comp["lost_definition"],
        "scored_population_before": comp["before_n"],
        "scored_population_after": comp["after_n"],
        "population_mean_before": comp["before_mean"],
        "population_mean_after": comp["after_mean"],
    }
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_corpus": str(RAW_PATH),
        "compiled_corpus": str(COMPILED_PATH),
        "frozen_weights": str(WEIGHTS),
        "n_records_raw": before.height,
        "n_records_compiled": after.height,
        "n_paired": paired.height,
        "n_institutions": per_inst.height,
        "corpus_wide": corpus,
        "completeness_components": completeness_components,
        "consistency_note": (
            "re-induced per corpus (consistency_induction.py); head τ=0.05"
            if consistency_ran
            else "pending — run consistency_induction.py to produce the per-corpus counts"
        ),
        "conformance_note": (
            "rule + model-typed slot checks (wrmthorne/ channel excluded); well-formedness-count layer pending"
        ),
    }
    (OUT_DIR / "quality_deltas_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt(x: float | None) -> str:
        return "   —  " if x is None else f"{x:8.4f}"

    print(
        f"\nX1 before/after — {paired.height:,} paired records, "
        f"{per_inst.height} institutions "
        f"(paired delta over records defined on both corpora)\n"
    )
    print(f"{'axis':<26}{'before':>10}{'after':>10}{'delta':>10}{'n_both':>12}{'gained':>9}")
    for axis, v in corpus.items():
        print(
            f"{axis:<26}{fmt(v['paired_before'])}{fmt(v['paired_after'])}"
            f"{fmt(v['paired_delta'])}{v['n_both_defined']:>12,}"
            f"{v['gained_definition']:>9,}"
        )
    cc = completeness_components
    print(
        f"\ncompleteness = coverage + fill (quote both):"
        f"\n  coverage  {cc['coverage_gained_records']:+,} records gained core-completeness "
        f"({cc['scored_population_before']:,} → {cc['scored_population_after']:,} scored)"
        f"\n  fill      {cc['fill_paired_delta']:+.4f} paired delta over "
        f"{cc['fill_n_paired']:,} both-defined records"
    )
    # dispersion of the per-institution completeness gain
    d = per_inst["completeness_delta"].drop_nulls()
    if d.len():
        print(
            f"\nper-institution completeness gain: "
            f"median {d.median():.4f}, "
            f"IQR [{d.quantile(0.25):.4f}, {d.quantile(0.75):.4f}], "
            f"min {d.min():.4f}, max {d.max():.4f}"
        )
    print(f"\n→ {OUT_DIR / 'quality_deltas_summary.json'}\n→ {OUT_DIR / 'quality_deltas_per_institution.parquet'}")


if __name__ == "__main__":
    main()
