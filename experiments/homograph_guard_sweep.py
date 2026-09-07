from __future__ import annotations

import json
import math

from experiments.harness import EXP_OUT, gold_cases, sample_items

OUT = EXP_OUT / "homograph"
TARGET = 0.95
GUARDS = ("fuzzy", "semantic")
# the current floors, for the report header
CURRENT_MIN_LEN = {"fuzzy": 4, "semantic": 3}
CURRENT_ACCEPT = {"fuzzy": 0.93, "semantic": 0.35}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def rows_for(guard: str, items: dict, gold: dict) -> list[dict]:
    out = []
    for iid, case in gold.items():
        it = items.get(iid)
        if it is None or it.get("sub_component") != guard:
            continue
        v = (case.get("expected") or {}).get("verdict")
        if not v or v == "cant_tell":
            continue
        out.append(
            {"len": len(it.get("norm") or ""), "score": it.get("score"), "correct": v == "correct", "verdict": v}
        )
    return out


def sweep(rows: list[dict], key: str) -> list[dict]:
    """Precision on the atoms a `key >= t` guard would still admit"""
    vals = sorted({r[key] for r in rows if r[key] is not None})
    table = []
    for t in vals:
        kept = [r for r in rows if r[key] is not None and r[key] >= t]
        n = len(kept)
        corr = sum(r["correct"] for r in kept)
        lo, hi = wilson(corr, n)
        dropped = [r for r in rows if r[key] is not None and r[key] < t]
        table.append(
            {
                "threshold": t,
                "n_admitted": n,
                "correct": corr,
                "precision": corr / n if n else None,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "n_dropped": len(dropped),
                "dropped_correct": sum(r["correct"] for r in dropped),
            }
        )
    return table


def best_reaching_target(table: list[dict]) -> dict | None:
    """The lowest threshold whose Wilson lower bound clears the target while still admitting anything"""
    for row in table:
        if row["n_admitted"] and row["wilson_lo"] is not None and row["wilson_lo"] >= TARGET:
            return row
    return None


def main() -> None:
    items = {i["id"]: i for i in sample_items("homograph")}
    gold = gold_cases("homograph")

    report = {"target": TARGET, "guards": {}}
    for guard in GUARDS:
        rows = rows_for(guard, items, gold)
        lens = sorted(r["len"] for r in rows)
        n = len(rows)
        base = sum(r["correct"] for r in rows)
        blo, bhi = wilson(base, n)
        len_sweep = sweep(rows, "len")
        score_sweep = sweep(rows, "score")
        from collections import Counter

        report["guards"][guard] = {
            "current_min_len": CURRENT_MIN_LEN[guard],
            "current_accept": CURRENT_ACCEPT[guard],
            "n": n,
            "baseline_precision": base / n if n else None,
            "baseline_wilson": [blo, bhi],
            "norm_len": {"min": min(lens), "median": lens[len(lens) // 2], "max": max(lens)},
            "verdicts": dict(Counter(r["verdict"] for r in rows)),
            "len_guard_binds": min(lens) < CURRENT_MIN_LEN[guard],
            "len_sweep_reaches_target": best_reaching_target(len_sweep),
            "score_sweep_reaches_target": best_reaching_target(score_sweep),
            "len_sweep": len_sweep,
            "score_sweep": score_sweep,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "guard_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fmt(v: float | None) -> str:
        return "  —  " if v is None else f"{v:5.3f}"

    for guard, g in report["guards"].items():
        print(
            f"\n{guard}: n={g['n']}, baseline precision {fmt(g['baseline_precision'])} "
            f"[{fmt(g['baseline_wilson'][0])},{fmt(g['baseline_wilson'][1])}], "
            f"norm-len {g['norm_len']}, verdicts {g['verdicts']}"
        )
        print(
            f"  length floor currently {g['current_min_len']}; "
            f"binds on this sample: {g['len_guard_binds']}; "
            f"a length guard reaching {TARGET}: "
            f"{'none' if not g['len_sweep_reaches_target'] else g['len_sweep_reaches_target']['threshold']}"
        )
        print(
            f"  score floor currently {g['current_accept']}; "
            f"a score guard reaching {TARGET}: "
            f"{'none' if not g['score_sweep_reaches_target'] else g['score_sweep_reaches_target']}"
        )
    print(f"\n→ {OUT / 'guard_sweep.json'}")


if __name__ == "__main__":
    main()
