from __future__ import annotations

import json
import time

import polars as pl

from mds_norm.paths import ANALYSIS_OUTPUT, CONSISTENCY_OUT, EVAL_OUT, EXP_OUT, INSTITUTIONAL, PROBE_CANDIDATES

OUT_DIR = EVAL_OUT / "remediation"

INFORMATIVENESS = EXP_OUT / "metric_informativeness" / "metric_informativeness.json"
CONCENTRATION = CONSISTENCY_OUT / "pattern_consistency_raw.parquet"
NULL_MARKERS = INSTITUTIONAL / "null_marker_candidates.parquet"
UNCERTAINTY = INSTITUTIONAL / "uncertainty_marking.parquet"
PROSE_COMPOSITION = ANALYSIS_OUTPUT / "prose_composition.parquet"

# rule-fixable when the record already carries what a fix needs
RULE_FIXABLE = {
    "minority convention": True,
    "malformed": False,
    "placeholder": True,
    "structured content in prose": True,
    "marked uncertainty": True,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def variance_shares() -> pl.DataFrame:
    """Panel one: how much of each measure's variation lies between institutions rather than within them"""
    report = json.loads(INFORMATIVENESS.read_text())
    shares = report["between_institution_variance_share"]
    return pl.DataFrame({"measure": list(shares), "between_institution": [shares[m] for m in shares]}).sort(
        "between_institution", descending=True
    )


def per_field() -> pl.DataFrame:
    """Panel two: where divergence sits, by field and by institution"""
    concentration = pl.read_parquet(CONCENTRATION)
    return (
        concentration.with_columns(
            divergent=((1 - pl.col("head_share")) * pl.col("n_records")).round(0).cast(pl.Int64),
            malformed=(pl.col("anomaly_share") * pl.col("n_records")).round(0).cast(pl.Int64),
        )
        .group_by("field_type")
        .agg(
            values=pl.col("n_records").sum(),
            divergent=pl.col("divergent").sum(),
            malformed=pl.col("malformed").sum(),
            institutions=pl.col("data_source").n_unique(),
        )
        .with_columns(divergent_share=pl.col("divergent") / pl.col("values"))
        .sort("divergent", descending=True)
    )


def by_kind() -> pl.DataFrame:
    """Panel three: the divergent values divided into the kinds a remediation would treat differently"""
    concentration = pl.read_parquet(CONCENTRATION)
    conventional = concentration.select(
        (((1 - pl.col("head_share")) - pl.col("anomaly_share")) * pl.col("n_records")).sum().alias("n"),
        pl.col("data_source").n_unique().alias("institutions"),
    ).row(0)
    malformed = concentration.select(
        (pl.col("anomaly_share") * pl.col("n_records")).sum().alias("n"),
        pl.col("data_source").filter(pl.col("anomaly_share") > 0).n_unique().alias("institutions"),
    ).row(0)

    markers = pl.read_parquet(NULL_MARKERS)
    placeholders = (int(markers["occ"].sum()), markers["data_source"].n_unique())

    prose = pl.read_parquet(PROBE_CANDIDATES)
    empty_target = prose.filter(pl.col("status").is_in(["novel", "unparsed_field"]))
    misplaced = (empty_target.height, empty_target["data_source"].n_unique())

    uncertainty = pl.read_parquet(UNCERTAINTY)
    marked = uncertainty.select(
        (pl.col("n_values") * (pl.col("question") + pl.col("brackets") + pl.col("lexical"))).sum().alias("n"),
        pl.col("data_source").filter(pl.col("style") != "no marker").n_unique().alias("institutions"),
    ).row(0)

    rows = [
        {"kind": "minority convention", "values": int(conventional[0]), "institutions": conventional[1]},
        {"kind": "malformed", "values": int(malformed[0]), "institutions": malformed[1]},
        {"kind": "placeholder", "values": placeholders[0], "institutions": placeholders[1]},
        {"kind": "structured content in prose", "values": misplaced[0], "institutions": misplaced[1]},
        {"kind": "marked uncertainty", "values": int(marked[0]), "institutions": marked[1]},
    ]
    return (
        pl.DataFrame(rows)
        .with_columns(rule_fixable=pl.col("kind").replace_strict(RULE_FIXABLE))
        .sort("values", descending=True)
    )


def targets(kinds: pl.DataFrame, fields: pl.DataFrame) -> pl.DataFrame:
    """The ranked list: what would pay, by how much of the corpus it touches and whether a rule could do it"""
    field_rows = fields.head(10).select(
        target=pl.lit("field: ") + pl.col("field_type").cast(pl.String),
        values=pl.col("divergent").cast(pl.Int64),
        institutions=pl.col("institutions").cast(pl.Int64),
        rule_fixable=pl.lit(True),
    )
    kind_rows = kinds.select(
        target=pl.lit("kind: ") + pl.col("kind"),
        values=pl.col("values").cast(pl.Int64),
        institutions=pl.col("institutions").cast(pl.Int64),
        rule_fixable="rule_fixable",
    )
    return pl.concat([kind_rows, field_rows]).sort("values", descending=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shares = variance_shares()
    shares.write_parquet(OUT_DIR / "variance_shares.parquet")

    fields = per_field()
    fields.write_parquet(OUT_DIR / "divergence_per_field.parquet")

    kinds = by_kind()
    kinds.write_parquet(OUT_DIR / "divergence_by_kind.parquet")

    ranked = targets(kinds, fields)
    ranked.write_parquet(OUT_DIR / "remediation_targets.parquet")
    log(f"→ {OUT_DIR}")
    with pl.Config(tbl_rows=20, fmt_str_lengths=48):
        print(shares)
        print(kinds)
        print(ranked)


if __name__ == "__main__":
    main()
