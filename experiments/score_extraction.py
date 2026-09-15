from __future__ import annotations

import polars as pl

from experiments.harness import EXP_OUT, ROOT, gold_cases, load_predictions, log, sample_items
from experiments.variants import EXTRACTION_VARIANTS, REPRESENTATION_VARIANTS
from mds_norm.utils.patches import strip_prefix

EXPS = {"extraction_variants": EXTRACTION_VARIANTS, "representation_variants": REPRESENTATION_VARIANTS}
FRAME = ROOT / "data" / "gold" / "frames" / "extraction.parquet"


def _key(field: str | None, value: str | None) -> tuple[str, str]:
    f = strip_prefix(field or "").strip().lower()
    v = " ".join((value or "").split()).lower()
    return f, v


def _overlaps(a: str, b: str) -> bool:
    """Two values name the same thing up to where the annotator cut the span"""
    return a in b or b in a


def _tolerant_pairs(pred_keys: set[tuple[str, str]], gold_keys: set[tuple[str, str]]) -> int:
    """Greedy one-to-one count of predictions whose span overlaps an unclaimed gold span in the same field"""
    unclaimed = set(gold_keys)
    hits = 0
    for pf, pv in sorted(pred_keys):
        match = next((g for g in sorted(unclaimed) if g[0] == pf and _overlaps(pv, g[1])), None)
        if match is not None:
            unclaimed.discard(match)
            hits += 1
    return hits


def score_variant(exp: str, name: str, gold: dict[str, dict], per_task: list[dict] | None = None) -> dict | None:
    preds = load_predictions(exp, name)
    if preds is None:
        return None
    resolved = preds.filter(pl.col("status") == "resolved")

    tp = fp = fn = misfiled = value_ok = 0
    loose_tp = 0
    tasks: dict[str, dict[str, int]] = {}
    n_items = 0
    for item_id, case in gold.items():
        if case["expected"] is None:
            continue
        n_items += 1
        gold_ops = case["expected"].get("ops", [])
        gold_keys = {_key(o.get("field"), o.get("value")) for o in gold_ops}
        gold_values = {v for _, v in gold_keys}
        pred_keys = {
            _key(r["field"], r["value"]) for r in resolved.filter(pl.col("id") == item_id).iter_rows(named=True)
        }
        item_loose = _tolerant_pairs(pred_keys, gold_keys)
        loose_tp += item_loose
        for k in pred_keys:
            t = tasks.setdefault(k[0], dict.fromkeys(("tp", "fp", "fn", "loose_tp", "n_pred", "n_gold"), 0))
            t["n_pred"] += 1
            if k in gold_keys:
                tp += 1
                value_ok += 1
                t["tp"] += 1
            elif k[1] in gold_values:
                fp += 1
                value_ok += 1
                misfiled += 1
                t["fp"] += 1
            else:
                fp += 1
                t["fp"] += 1
        for k in gold_keys:
            t = tasks.setdefault(k[0], dict.fromkeys(("tp", "fp", "fn", "loose_tp", "n_pred", "n_gold"), 0))
            t["n_gold"] += 1
            if k not in pred_keys:
                t["fn"] += 1
        for field in {k[0] for k in pred_keys}:
            tasks[field]["loose_tp"] += _tolerant_pairs(
                {k for k in pred_keys if k[0] == field}, {k for k in gold_keys if k[0] == field}
            )
        fn += len(gold_keys - pred_keys)

    if not n_items:
        return None
    if per_task is not None:
        for field, t in sorted(tasks.items()):
            per_task.append(
                {
                    "exp": exp,
                    "variant": name,
                    "task": field,
                    "n_pred": t["n_pred"],
                    "n_gold": t["n_gold"],
                    "precision_strict": t["tp"] / t["n_pred"] if t["n_pred"] else None,
                    "recall_strict": t["tp"] / t["n_gold"] if t["n_gold"] else None,
                    "precision_tolerant": t["loose_tp"] / t["n_pred"] if t["n_pred"] else None,
                    "recall_tolerant": t["loose_tp"] / t["n_gold"] if t["n_gold"] else None,
                }
            )
    runs = pl.read_ndjson(EXP_OUT / exp / "runs.jsonl").filter(pl.col("variant") == name).tail(1).to_dicts()[0]
    n_pred = tp + fp
    return {
        "exp": exp,
        "variant": name,
        "n_items": n_items,
        "pred_ops": n_pred,
        "gold_ops_missed": fn,
        "precision_strict": tp / n_pred if n_pred else None,
        "precision_value": value_ok / n_pred if n_pred else None,
        "recall_strict": tp / (tp + fn) if (tp + fn) else None,
        "precision_tolerant": loose_tp / n_pred if n_pred else None,
        "recall_tolerant": loose_tp / (tp + fn) if (tp + fn) else None,
        "placement_error": misfiled / value_ok if value_ok else None,
        "prompt_tokens": runs.get("prompt_tokens"),
        "completion_tokens": runs.get("completion_tokens"),
        "tokens_per_accepted_op": runs.get("tokens_per_accepted_op"),
        "energy_wh": runs.get("energy_wh"),
        "wh_per_record": runs.get("wh_per_record"),
    }


def main() -> None:
    gold = gold_cases("extraction")
    # Unlabelled sample ids are absent from gold, so report coverage
    n_sample = len(sample_items("extraction"))
    log(f"gold covers {len(gold)}/{n_sample} sampled records")

    queue_records = pl.read_parquet(FRAME).height if FRAME.exists() else None
    rows = []
    task_rows: list[dict] = []
    for exp, variants in EXPS.items():
        for name in variants:
            if r := score_variant(exp, name, gold, task_rows):
                if queue_records and r["wh_per_record"] is not None:
                    r["projected_queue_wh"] = r["wh_per_record"] * queue_records
                rows.append(r)
    if not rows:
        raise SystemExit("nothing to score — run variants and label the pooled gold")
    results = pl.DataFrame(rows).sort("precision_strict", descending=True, nulls_last=True)
    out = EXP_OUT / "extraction_results.parquet"
    results.write_parquet(out)
    per_task = EXP_OUT / "extraction_per_task.parquet"
    pl.DataFrame(task_rows).write_parquet(per_task)
    log(f"per-task precision → {per_task}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=220, fmt_str_lengths=30):
        print(results)
    log(f"results → {out}" + (f" (queue projection over {queue_records:,} records)" if queue_records else ""))


if __name__ == "__main__":
    main()
