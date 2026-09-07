from __future__ import annotations

import argparse
import asyncio
import json
import math
import re

import polars as pl

from experiments.harness import EXP_OUT, Variant, gold_cases, measure, sample_items, write_run

OUT = EXP_OUT / "homograph"
TARGET = 0.95  # the cascade's stated bar
MAX_CANDIDATES = 15  # cap the displayed set; pipeline + gold subject always kept
NONE = "none"

# rung membership, as the rung scorer groups them
LADDER = ("kind_tier", "prominent", "spatial")
GUARDS = ("fuzzy", "semantic")


def rungs_of(item: dict) -> list[str]:
    out = []
    if item.get("resolved_by") in LADDER:
        out.append(item["resolved_by"])
    if item.get("status") == "flagged":
        out.append("flagged_best")
    if item.get("sub_component") in GUARDS:
        out.append(item["sub_component"])
    return out


SYSTEM = (
    "You are a museum cataloguing vocabulary specialist. A term was extracted from a catalogue record and "
    "tentatively matched to a controlled-vocabulary concept. Your job is to decide which of the listed concepts "
    "the term genuinely denotes IN THIS record, using each concept's hierarchy/authority as evidence — or to "
    "reject them all.\n"
    "\n"
    "A concept is only correct if the extracted term actually names it. A merely similar-sounding or "
    'near-spelled concept is NOT a match: prefer "none" whenever the term does not clearly denote one of the '
    "listed concepts (for example a term carrying a qualifier the concept lacks, or a different "
    "material/period/place than any concept describes).\n"
    "\n"
    'Answer with only the option number, e.g. "2".'
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def gold_answer(item: dict, case: dict) -> str | None:
    """The gold subject to score against, or None when excluded (cant_tell)"""
    exp = case.get("expected") or {}
    v = exp.get("verdict")
    if v == "correct":
        return exp.get("subject") or item.get("subject")
    if v in ("wrong_candidate", "match"):  # the rerank task's, judged with no pick shown
        return exp.get("subject")
    if v == "not_in_candidates":
        return NONE
    return None  # cant_tell -> excluded


def describe(c: dict) -> str:
    """One line per candidate: term, label kind and disambiguating context"""
    term = c.get("term") or "?"
    bits = [f'"{term}"']
    kind = c.get("kind")
    if kind and kind not in ("prefLabel", "prefLabelGVP"):
        bits.append(f"[{kind}]")
    if c.get("parents"):
        bits.append(f"— {c['parents']}")
    else:
        meta = []
        if c.get("authority"):
            meta.append(str(c["authority"]))
        sy, ey = c.get("start_year"), c.get("stop_year")
        if sy is not None or ey is not None:
            meta.append(f"dates {sy}..{ey}")
        if meta:
            bits.append("— " + "; ".join(meta))
    return " ".join(bits)


def display_candidates(item: dict, gold: str | None) -> list[dict]:
    """Distinct candidates, capped, always retaining the pipeline and gold subjects"""
    seen: dict[str, dict] = {}
    for c in item.get("candidates") or []:
        sub = c.get("subject")
        if sub is not None and sub not in seen:
            seen[sub] = c
    keep_first = [s for s in (item.get("subject"), gold) if s in seen and s != NONE]
    ordered = list(dict.fromkeys(keep_first + list(seen)))
    if len(ordered) > MAX_CANDIDATES:
        forced = list(keep_first)
        rest = [s for s in ordered if s not in forced][: MAX_CANDIDATES - len(forced)]
        ordered = forced + rest
    return [seen[s] for s in ordered]


def build_prompt(item: dict, cands: list[dict]) -> tuple[str, list[str]]:
    group = item.get("field_type") or item.get("vocab") or "vocabulary"
    value = item.get("value") or item.get("atom")
    lines = [f"  {i + 1}. {describe(c)}" for i, c in enumerate(cands)]
    lines.append(f"  {len(cands) + 1}. none — the term does not denote any of these concepts")
    prompt = (
        f"Record source ({item['data_source']}):\n"
        f"  field group: {group}\n"
        f"  full recorded value: {value!r}\n"
        f"  term extracted from it: {item['atom']!r}\n\n"
        f"Which controlled-vocabulary concept does the term {item['atom']!r} denote here?\n"
        + "\n".join(lines)
        + "\n\nAnswer with only the option number."
    )
    choices = [c["subject"] for c in cands] + [NONE]
    return prompt, choices


def parse_choice(content: str | None, choices: list[str]) -> str | None:
    if not content:
        return None
    m = re.search(r"\d+", content)
    if not m:
        return None
    idx = int(m.group()) - 1
    return choices[idx] if 0 <= idx < len(choices) else None


async def run(variant: Variant, limit: int | None, rungs: set[str]) -> pl.DataFrame:
    items = {i["id"]: i for i in sample_items("homograph")}
    gold = gold_cases("homograph")

    work = []
    for iid, case in gold.items():
        it = items.get(iid)
        if it is None:
            continue
        ga = gold_answer(it, case)
        if ga is None:  # cant_tell / unlabelled
            continue
        if rungs and not (set(rungs_of(it)) & rungs):
            continue
        work.append((it, case, ga))
    if limit:
        work = work[:limit]

    units, prompts = [], []
    for it, case, ga in work:
        cands = display_candidates(it, ga)
        if not cands:
            continue
        prompt, choices = build_prompt(it, cands)
        units.append((it, case, ga, choices))
        prompts.append(prompt)

    from mds_norm.utils.inference import Inference

    inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)
    with measure("homograph_verifier", variant.name) as cost:
        replies = await inf.generate(prompts, system=SYSTEM, usage=True, **variant.decoding)

    rows = []
    for (it, case, ga, choices), r in zip(units, replies, strict=True):
        pick = parse_choice(r["content"], choices)
        rows.append(
            {
                "id": it["id"],
                "atom": it["atom"],
                "value": it.get("value"),
                "rungs": rungs_of(it),
                "sub_component": it.get("sub_component"),
                "resolved_by": it.get("resolved_by"),
                "status": it.get("status"),
                "verdict": (case.get("expected") or {}).get("verdict"),
                "pipeline_subject": it.get("subject"),
                "gold": ga,
                "pick": pick,
                "error": r["error"],
            }
        )
    preds = pl.from_dicts(rows)

    tokens = sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in replies)
    metrics = {
        "n_items": len(rows),
        "n_errors": sum(1 for r in replies if r["error"]),
        "tokens": tokens,
        **cost,
        "wh_per_op": cost["energy_wh"] / max(len(rows), 1),
    }
    write_run("homograph_verifier", variant, preds, metrics)
    return preds


def score_rung(rows: list[dict]) -> dict:
    n = len(rows)
    acc = sum(r["pick"] == r["gold"] for r in rows)
    lo, hi = wilson(acc, n)
    # A verification gate keeps only picks the verifier confirms
    confirm = [r for r in rows if r["pick"] == r["pipeline_subject"] and r["pick"] != NONE]
    conf_ok = sum(r["gold"] == r["pipeline_subject"] for r in confirm)
    clo, chi = wilson(conf_ok, len(confirm))
    pipe_ok = sum(r["gold"] == r["pipeline_subject"] for r in rows)  # rung's own precision
    return {
        "n": n,
        "verifier_accuracy": acc / n if n else None,
        "verifier_wilson": [lo, hi],
        "pipeline_precision": pipe_ok / n if n else None,
        "n_confirm": len(confirm),
        "confirm_precision": conf_ok / len(confirm) if confirm else None,
        "confirm_wilson": [clo, chi],
        "reject_rate": sum(r["pick"] == NONE for r in rows) / n if n else None,
        "n_errors": sum(r["error"] is not None for r in rows),
        "verdict": (
            "verify_gate"
            if (confirm and clo is not None and clo >= TARGET)
            else "route"
            if (n and lo is not None and lo >= TARGET)
            else "demote"
        ),
    }


def score(preds: pl.DataFrame) -> dict:
    rows = preds.to_dicts()
    report = {"target": TARGET, "n_items": len(rows), "rungs": {}, "overall": {}}
    by_rung: dict[str, list[dict]] = {}
    for r in rows:
        for rung in r["rungs"]:
            by_rung.setdefault(rung, []).append(r)
    for rung, rr in sorted(by_rung.items()):
        report["rungs"][rung] = score_rung(rr)
    report["overall"] = score_rung(rows)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rungs", default="fuzzy,semantic", help="comma list of rungs to score, or 'all'")
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument(
        "--reasoning",
        default="low",
        choices=["low", "medium", "high"],
        help="gpt-oss reasoning_effort; the run name carries it so variants don't clobber",
    )
    args = ap.parse_args()
    rungs = set() if args.rungs == "all" else set(args.rungs.split(","))

    # gpt-oss reasons before answering, so higher effort needs room
    name = "gpt-oss-20b" if args.reasoning == "low" else f"gpt-oss-20b_{args.reasoning}"
    max_tokens = 512 if args.reasoning == "low" else 1024
    variant = Variant(
        name=name,
        model=args.model,
        decoding={"temperature": 0.0, "top_p": 1.0, "reasoning_effort": args.reasoning, "max_tokens": max_tokens},
        params={"rungs": sorted(rungs) or ["all"]},
        notes="candidate-disambiguation verifier over homograph gold",
    )
    preds = asyncio.run(run(variant, args.limit, rungs))
    report = score(preds)

    OUT.mkdir(parents=True, exist_ok=True)
    out_name = "verifier_precision.json" if args.reasoning == "low" else f"verifier_precision_{args.reasoning}.json"
    (OUT / out_name).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fmt(v: float | None) -> str:
        return "  —  " if v is None else f"{v:5.3f}"

    print(f"\ncandidate-disambiguation verifier over homograph gold — {preds.height} items\n")
    print(f"{'rung':>14}  {'n':>4}  verifier [wilson95]     pipe   conf_n  conf_prec [wilson95]     rej   verdict")
    for rung, r in report["rungs"].items():
        vlo, vhi = r["verifier_wilson"]
        clo, chi = r["confirm_wilson"]
        print(
            f"{rung:>14}  {r['n']:>4}  {fmt(r['verifier_accuracy'])} "
            f"[{fmt(vlo)},{fmt(vhi)}]  {fmt(r['pipeline_precision'])}  "
            f"{r['n_confirm']:>6}  {fmt(r['confirm_precision'])} "
            f"[{fmt(clo)},{fmt(chi)}]  {fmt(r['reject_rate'])}  {r['verdict']}"
        )
    o = report["overall"]
    print(
        f"\noverall verifier {fmt(o['verifier_accuracy'])} vs pipeline "
        f"{fmt(o['pipeline_precision'])}  (n={o['n']}, errors={o['n_errors']})"
    )
    print(f"→ {OUT / out_name}")


if __name__ == "__main__":
    main()
