import json

import polars as pl

from mds_norm.paths import EXP_OUT, FIELD_STATS, VOCAB_ANNOTATIONS, VOCAB_DECISIONS
from mds_norm.pipeline.vocab_indexes import CASCADE_FIELDS, GROUP_FOR

RUNG_VERDICTS = EXP_OUT / "homograph" / "rung_verdicts.json"
DECISIONS = VOCAB_DECISIONS
ANNOTATIONS = VOCAB_ANNOTATIONS

# rungs keyed by sub_component rather than resolved_by
GUARD_RUNGS = {"fuzzy", "rerank"}


def apply_verdicts(decisions: pl.DataFrame) -> pl.DataFrame:
    """Flip decision statuses per the measured rung verdicts"""
    rung_verdicts = json.loads(RUNG_VERDICTS.read_text())
    # an unscored guard rung is not trusted to resolve
    demote = [r for r, v in rung_verdicts.items() if v["action"] == "demote"]
    demote += [r for r in GUARD_RUNGS if r not in rung_verdicts]
    demote_ladder = [r for r in demote if r not in GUARD_RUNGS]  # keyed by resolved_by
    demote_guard = [r for r in demote if r in GUARD_RUNGS]  # keyed by sub_component
    promote = rung_verdicts.get("flagged_best", {}).get("action") == "promote"
    flagged_conf = rung_verdicts.get("flagged_best", {}).get("precision")
    was_flagged = pl.col("status") == "flagged"
    is_demoted = pl.col("resolved_by").is_in(demote_ladder) | pl.col("sub_component").is_in(demote_guard)
    out = decisions.with_columns(
        status=pl.when((pl.col("status") == "resolved") & is_demoted)
        .then(pl.lit("flagged"))
        .when(pl.lit(promote) & was_flagged)
        .then(pl.lit("resolved"))
        .otherwise(pl.col("status")),
        confidence=pl.when((pl.col("status") == "resolved") & is_demoted)
        .then(pl.lit(None, dtype=pl.Float64))
        .when(pl.lit(promote) & was_flagged)
        .then(pl.lit(flagged_conf, dtype=pl.Float64))
        .otherwise(pl.col("confidence")),
    )
    print("rung verdicts applied:", {r: v["action"] for r, v in rung_verdicts.items()})
    return out


def regenerate_annotations(decisions: pl.DataFrame) -> None:
    """Re-join decisions onto every value occurrence — vocab_alignment's sidecar step"""
    value_ldf = (
        pl.scan_parquet(FIELD_STATS)
        .filter(pl.col("field_type").is_in(CASCADE_FIELDS) & pl.col("value").is_not_null())
        .select("record_id", "node_id", "data_source", "field_type", "value")
        .with_columns(pl.col("value").str.strip_chars().str.replace_all(r"\s+", " "))
        .filter(pl.col("value") != "")
        .with_columns(group=pl.col("field_type").replace_strict(GROUP_FOR, return_dtype=pl.String))
    )
    (
        value_ldf.join(
            # atomiser flags exist only in current-cascade decisions
            decisions.drop("count", "norm", "atom_route", "split_ok", "compound_ok", strict=False).lazy(),
            on=["group", "data_source", "value"],
            how="left",
        ).sink_parquet(ANNOTATIONS)
    )
    n = pl.scan_parquet(ANNOTATIONS).select(pl.len()).collect(engine="streaming").item()
    print(f"{n:,} annotation rows -> {ANNOTATIONS}")


def main() -> None:
    decisions = pl.read_parquet(DECISIONS)
    before = decisions.group_by("status").agg(atoms=pl.len(), occ=pl.col("count").sum()).sort("status")
    print("BEFORE:\n", before)

    decisions = apply_verdicts(decisions)
    decisions.write_parquet(DECISIONS)

    after = decisions.group_by("status").agg(atoms=pl.len(), occ=pl.col("count").sum()).sort("status")
    print("AFTER:\n", after)

    regenerate_annotations(decisions)


if __name__ == "__main__":
    main()
