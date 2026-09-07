from __future__ import annotations

import json
import time

import polars as pl

from mds_norm.paths import EXP_OUT, GOLD_CASES

FINGERPRINTS = EXP_OUT / "fingerprint_routing"
DISAGREEMENTS = FINGERPRINTS / "regime_disagreements.parquet"
GOLD_DATES = GOLD_CASES / "gold_dates.json"
OUT = FINGERPRINTS / "regime_precision.json"


def main() -> None:
    r12 = (
        pl.read_parquet(DISAGREEMENTS)
        .filter(pl.col("pair") == "r1_vs_r2")
        .with_columns(
            kind=pl.when(pl.col("edtf_a").is_null())
            .then(pl.lit("new_parse"))
            .when(pl.col("edtf_a") == pl.col("edtf_b"))
            .then(pl.lit("same_edtf_promotion"))
            .otherwise(pl.lit("edtf_changed"))
        )
    )
    structure = {
        k: {"rows": int(n), "occ": int(o)}
        for k, n, o in r12.group_by("kind").agg(pl.len(), pl.col("n_occ").sum()).iter_rows()
    }

    gold = json.loads(GOLD_DATES.read_text())["cases"]
    gold_edtf = {}
    for c in gold:
        # A value labelled twice with different expectations adjudicates nothing
        e = (c.get("expected") or {}).get("edtf")
        if c["raw"] in gold_edtf and gold_edtf[c["raw"]] != e:
            gold_edtf[c["raw"]] = None
        else:
            gold_edtf[c["raw"]] = e
    gv = pl.DataFrame({"value": list(gold_edtf), "gold_edtf": list(gold_edtf.values())})
    scored = (
        r12.join(gv, on="value")
        .filter(pl.col("gold_edtf").is_not_null())
        .with_columns(r2_correct=pl.col("edtf_b") == pl.col("gold_edtf"))
    )
    gold_axis = {
        "distinct_values": scored.height,
        "occ": int(scored["n_occ"].sum()),
        "r2_matches_gold_occ": int(scored.filter(pl.col("r2_correct"))["n_occ"].sum()),
        "mismatches": [
            {"value": v, "gold": g, "r2": b, "occ": int(o)}
            for v, g, b, o in scored.filter(~pl.col("r2_correct"))
            .select("value", "gold_edtf", "edtf_b", "n_occ")
            .iter_rows()
        ],
        "note": "same-EDTF promotions mean R1 and R2 agree on the value, so "
        "this axis tests whether the shared parse is right on the "
        "overlapping gold head, not which regime wins",
    }

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "pair": "r1_vs_r2",
        "structure": structure,
        "gold_overlap": gold_axis,
        "verdict": {
            "wire_stratum_keys": False,
            "reason": "no disagreement changes a published value — the whole "
            "R1->R2 effect is dropping ambiguous_dm on the "
            "same-EDTF mass plus 846 occ of new parses. The upside "
            "is confidence relabelling only, and whether the "
            "stratum dm_order assertion is right still needs the "
            "disagreement-stratum labels. Conservative "
            "framing (decisions.md): the qualification is the "
            "honest state until precision is measured.",
            "reopen_when": "the value panel labels the disagreement stratum and the promotion precision "
            "clears the bar",
        },
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
