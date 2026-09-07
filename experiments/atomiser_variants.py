from __future__ import annotations

import argparse
import asyncio
import re
from collections.abc import Iterable
from functools import cache

import polars as pl

from experiments.harness import (
    EXP_OUT,
    ROOT,
    Variant,
    gold_cases,
    ht_weight,
    load_predictions,
    log,
    measure,
    sample_items,
    write_run,
)
from experiments.variants import ATOMISER_VARIANTS
from mds_norm.utils.atomise import (
    ATOM_PROSE_MAX_CHARS,
    ATOM_PROSE_MAX_TOKENS,
    BASE_SEPARATORS,
    DATE_LIKE,
    GROUP_DESC,
    LLM_PROMPT,
    NULL_MARKERS,
    atomise,
    parse_llm_atoms,
    separator_literal,
)

EXP = "atomiser_variants"
SEPARATORS = ROOT / "data" / "gold" / "frames" / "atomiser_separators.parquet"
FRAME = ROOT / "data" / "gold" / "frames" / "atomiser.parquet"

_DATE_LIKE = re.compile(DATE_LIKE)


def sep_map() -> dict[tuple[str, str], list[str]]:
    if not SEPARATORS.exists():
        raise SystemExit(f"no {SEPARATORS} — build the atomiser frame first")
    return {(g, ds): s for g, ds, s in pl.read_parquet(SEPARATORS).iter_rows()}


@cache
def split_gate() -> re.Pattern:
    """Cheap gate before splitting: any separator the frame attests"""
    literals = {separator_literal(s) for seps in sep_map().values() for s in seps} | set(BASE_SEPARATORS)
    return re.compile("|".join(re.escape(f" {lit} " if lit.isalpha() else lit) for lit in sorted(literals)))


def guard(atom: str) -> str:
    """Post-split guards mirroring the cascade: null markers hide inside lists, and a long atom is prose"""
    norm = re.sub(r"\s+", " ", atom.strip().lower())
    if not norm or norm in NULL_MARKERS:
        return "null_marker"
    if len(atom) > ATOM_PROSE_MAX_CHARS or len(atom.split(" ")) > ATOM_PROSE_MAX_TOKENS:
        return "prose"
    return "cascade"


def det_split(item: dict, seps: dict) -> list[str]:
    """The deterministic fast path: attested-separator split plus guards"""
    value = item["value"]
    if not split_gate().search(value):
        return [value]
    parts = atomise(value, seps.get((item["field_type"], item["data_source"])))
    return [p["atom"] for p in parts if guard(p["atom"]) != "null_marker"]


def llm_eligible(item: dict, atom: str) -> bool:
    """Which post-split atoms the LLM rung sees"""
    return (
        item["field_type"] in GROUP_DESC
        and " " in atom
        and guard(atom) == "cascade"
        and not (item["field_type"] == "periodo" and _DATE_LIKE.search(atom.lower()))
    )


async def predict(variant: Variant, limit: int | None) -> None:
    items = sample_items("atomiser", limit)

    rows, metrics = [], {}
    if variant.params.get("composition") == "everywhere":
        from mds_norm.utils.inference import Inference

        samples = [{"desc": GROUP_DESC.get(it["field_type"], it["field_type"]), "value": it["value"]} for it in items]
        inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)
        with measure(EXP, variant.name) as cost:
            replies = await inf.generate(samples, LLM_PROMPT, usage=True, **variant.decoding)
        for it, r in zip(items, replies, strict=True):
            # No fast path: an unparsed reply passes the value whole
            parsed = parse_llm_atoms(it["value"], r["content"] or "")
            rows.append(
                {
                    "id": it["id"],
                    "value": it["value"],
                    "field_type": it["field_type"],
                    "data_source": it["data_source"],
                    "stratum": it["stratum"],
                    "predicted": [d["sub_atom"] for d in parsed] if parsed else [it["value"]],
                    "completion": r["content"] or None,
                    "parsed": bool(parsed) or None,
                }
            )
        metrics = {
            "n_llm_calls": len(items),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in replies),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in replies),
            "n_errors": sum(1 for r in replies if r["error"]),
        }
    elif variant.model is None:
        seps = sep_map()
        with measure(EXP, variant.name) as cost:
            base = [(it, det_split(it, seps)) for it in items]
            for it, atoms in base:
                rows.append(
                    {
                        "id": it["id"],
                        "value": it["value"],
                        "field_type": it["field_type"],
                        "data_source": it["data_source"],
                        "stratum": it["stratum"],
                        "predicted": atoms,
                        "completion": None,
                        "parsed": None,
                    }
                )
        metrics = {"n_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    else:
        from mds_norm.utils.inference import Inference

        base = [(it, det_split(it, sep_map())) for it in items]
        queue = [
            (i, j, atom)
            for i, (it, atoms) in enumerate(base)
            for j, atom in enumerate(atoms)
            if llm_eligible(base[i][0], atom)
        ]
        samples = [{"desc": GROUP_DESC[base[i][0]["field_type"]], "value": atom} for i, _, atom in queue]
        inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)
        with measure(EXP, variant.name) as cost:
            replies = await inf.generate(samples, LLM_PROMPT, usage=True, **variant.decoding)
        splits: dict[tuple[int, int], list[str]] = {}
        completions: dict[int, list[str]] = {}
        for (i, j, atom), r in zip(queue, replies, strict=True):
            completions.setdefault(i, []).append(r["content"] or "")
            parsed = parse_llm_atoms(atom, r["content"] or "")
            if parsed:
                splits[(i, j)] = [d["sub_atom"] for d in parsed]
        for i, (it, atoms) in enumerate(base):
            final = [s for j, a in enumerate(atoms) for s in splits.get((i, j), [a])]
            rows.append(
                {
                    "id": it["id"],
                    "value": it["value"],
                    "field_type": it["field_type"],
                    "data_source": it["data_source"],
                    "stratum": it["stratum"],
                    "predicted": final,
                    "completion": "\n---\n".join(completions.get(i, [])) or None,
                    "parsed": any((i, j) in splits for j in range(len(atoms))) or None,
                }
            )
        metrics = {
            "n_llm_calls": len(queue),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in replies),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in replies),
            "n_errors": sum(1 for r in replies if r["error"]),
        }

    metrics |= {"n_items": len(items), **cost, "wh_per_1k_values": 1e3 * cost["energy_wh"] / max(len(items), 1)}
    preds = pl.DataFrame(rows, schema_overrides={"completion": pl.String, "parsed": pl.Boolean})
    write_run(EXP, variant, preds, metrics)


def _norm_atoms(atoms: Iterable[str]) -> set[str]:
    return {re.sub(r"\s+", " ", a.strip().lower()) for a in atoms if a and a.strip()}


def _decision(atoms: Iterable[str]) -> str:
    n = len(_norm_atoms(atoms))
    return "no_terms" if n == 0 else ("keep_whole" if n == 1 else "split")


def score() -> None:
    gold = gold_cases("atomiser")
    items = {it["id"]: it for it in sample_items("atomiser")}
    # Cost projection: the everywhere variant has no fast path
    frame = pl.read_parquet(FRAME)
    queue = frame.filter(pl.col("issues").list.set_intersection(pl.lit(["llm_queued", "tail_dropped"])).list.len() > 0)
    queue_values = queue.height
    surface_values = frame.height

    out = []
    for name, variant in ATOMISER_VARIANTS.items():
        preds = load_predictions(EXP, name)
        if preds is None:
            log(f"{name}: no cached predictions — skipped")
            continue
        runs = pl.read_ndjson(EXP_OUT / EXP / "runs.jsonl").filter(pl.col("variant") == name).tail(1).to_dicts()[0]
        n = tp = fp = fn = exact = dec_ok = 0
        w_sum = w_exact = 0.0
        n_refused = 0
        for row in preds.iter_rows(named=True):
            case = gold.get(row["id"])
            if case is None:
                continue
            if case["expected"] is None:  # annotator refused the item
                n_refused += 1
                continue
            g = _norm_atoms(case["expected"]["atoms"])
            p = _norm_atoms(row["predicted"])
            n += 1
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
            ok = g == p
            exact += ok
            dec_ok += _decision(case["expected"]["atoms"]) == _decision(row["predicted"])
            w = ht_weight(items[row["id"]]) if row["id"] in items else None
            if w is not None:
                w_sum += w
                w_exact += w * ok
        if not n:
            log(f"{name}: no scored items (gold not labelled yet?)")
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        everywhere = variant.params.get("composition") == "everywhere"
        proj_values = surface_values if everywhere else queue_values
        out.append(
            {
                "variant": name,
                "n_scored": n,
                "n_refused": n_refused,
                "exact_match": exact / n,
                "decision_acc": dec_ok / n,
                "atom_precision": prec,
                "atom_recall": rec,
                "atom_f1": 2 * prec * rec / max(prec + rec, 1e-9),
                "ht_exact_match": (w_exact / w_sum) if w_sum else None,
                "prompt_tokens": runs.get("prompt_tokens"),
                "completion_tokens": runs.get("completion_tokens"),
                "energy_wh": runs.get("energy_wh"),
                "wh_per_1k_values": runs.get("wh_per_1k_values"),
                "projected_queue_wh": (runs.get("wh_per_1k_values") or 0) * proj_values / 1e3,
                "projection_values": proj_values,
            }
        )
    if not out:
        raise SystemExit("nothing to score")
    results = pl.DataFrame(out).sort("atom_f1", descending=True)
    results.write_parquet(EXP_OUT / EXP / "results.parquet")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=30):
        print(results)
    log(
        f"results → {EXP_OUT / EXP / 'results.parquet'} "
        f"(projection over {queue_values:,} queue values; "
        f"{surface_values:,} surface values for the everywhere variant)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(ATOMISER_VARIANTS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.score:
        score()
    elif args.variant:
        asyncio.run(predict(ATOMISER_VARIANTS[args.variant], args.limit))
    else:
        ap.error("--variant NAME to predict, or --score")


if __name__ == "__main__":
    main()
