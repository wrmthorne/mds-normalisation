from __future__ import annotations

import json

import polars as pl

from experiments.harness import SAMPLES, load_predictions, log
from experiments.nuextract_trial import EXP as NX_EXP
from experiments.nuextract_trial import VARIANTS as NX
from experiments.variants import EXTRACTION_VARIANTS, REPRESENTATION_VARIANTS

# Pooling every variant keeps precision comparable; absent runs are skipped
EXPS = {"extraction_variants": EXTRACTION_VARIANTS, "representation_variants": REPRESENTATION_VARIANTS, NX_EXP: NX}


def pooled_candidates() -> dict[str, list[dict]]:
    pools: dict[str, dict[tuple, dict]] = {}
    n_runs = 0
    for exp, variants in EXPS.items():
        for name in variants:
            preds = load_predictions(exp, name)
            if preds is None:
                continue
            n_runs += 1
            for row in preds.filter(pl.col("status") == "resolved").iter_rows(named=True):
                key = (row["field"], row["value"], row["source_field"])
                pool = pools.setdefault(row["id"], {})
                entry = pool.setdefault(
                    key,
                    {
                        "field": row["field"],
                        "value": row["value"],
                        "source_field": row["source_field"],
                        "variants": [],
                    },
                )
                tag = f"{exp.split('_')[0]}:{name}"
                if tag not in entry["variants"]:
                    entry["variants"].append(tag)
    if not n_runs:
        raise SystemExit(
            "no cached predictions under analysis_output/experiments/ — run the extraction and "
            "representation variants first"
        )
    log(
        f"pooled {sum(len(p) for p in pools.values())} distinct candidate ops "
        f"from {n_runs} variant runs over {len(pools)} records"
    )
    return {rid: sorted(pool.values(), key=lambda c: (c["field"], c["value"])) for rid, pool in pools.items()}


def main() -> None:
    path = SAMPLES / "extraction.jsonl"
    if not path.exists():
        raise SystemExit(f"no sample at {path} — draw the extraction sample first")
    pools = pooled_candidates()
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    n_with = 0
    for it in items:
        it["candidates"] = pools.get(it["id"], [])
        n_with += bool(it["candidates"])
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(path)
    log(f"{n_with}/{len(items)} sample items now carry candidates → {path}")


if __name__ == "__main__":
    main()
