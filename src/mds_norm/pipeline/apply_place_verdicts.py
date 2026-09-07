import json

import polars as pl

from mds_norm.paths import EXP_OUT, PLACES_OUT
from mds_norm.pipeline import places_fallback, places_pipeline

RUNG_VERDICTS = EXP_OUT / "places" / "rung_verdicts.json"
TGN_DECISIONS = PLACES_OUT / "place_value_decisions.parquet"
FALLBACK_DECISIONS = PLACES_OUT / "place_fallback_decisions.parquet"

# Fallback rungs name the gazetteer: `unique` differs sharply between them
TGN_RUNG = pl.lit("tgn/") + pl.coalesce("resolved_by", "ctx_state")
FALLBACK_RUNG = pl.lit("fallback/") + pl.col("vocab") + "/" + pl.col("resolved_by").fill_null("flagged")


def apply_verdicts(decisions: pl.DataFrame, rung: pl.Expr, verdicts: dict) -> pl.DataFrame:
    """Flag the values of every rung the review sample demoted, and drop their confidence"""
    action = rung.replace_strict({r: v["action"] for r, v in verdicts.items()}, default="insufficient")
    demoted = (pl.col("status") == "resolved") & (action == "demote")
    return decisions.with_columns(
        status=pl.when(demoted).then(pl.lit("flagged")).otherwise(pl.col("status")),
        confidence=pl.when(demoted).then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.col("confidence")),
    )


def census(decisions: pl.DataFrame, rung: pl.Expr) -> pl.DataFrame:
    return (
        decisions.filter(pl.col("status").is_in(["resolved", "flagged"]))
        .group_by(rung.alias("rung"), "status")
        .agg(norms=pl.len(), occurrences=pl.col("count").sum())
        .sort("rung", "status")
    )


def main() -> None:
    verdicts = json.loads(RUNG_VERDICTS.read_text())
    print("rung verdicts:", {rung: v["action"] for rung, v in verdicts.items()})
    # Promotions add assertions, so they stay a release decision
    promotions = [rung for rung, v in verdicts.items() if v["action"] == "promote"]
    if promotions:
        print("promotions not applied, left as a release decision:", promotions)

    # TGN first: the fallback queues what that stage deferred
    for path, rung, stage in (
        (TGN_DECISIONS, TGN_RUNG, places_pipeline),
        (FALLBACK_DECISIONS, FALLBACK_RUNG, places_fallback),
    ):
        decisions = pl.read_parquet(path)
        print(f"\nBEFORE {path.name}:\n", census(decisions, rung))
        decisions = apply_verdicts(decisions, rung, verdicts)
        decisions.write_parquet(path)
        print(f"AFTER {path.name}:\n", census(decisions, rung))
        rows = stage.write_sidecar(stage.load_queue(), decisions)
        print(f"{rows:,} annotation rows rewritten")


if __name__ == "__main__":
    main()
