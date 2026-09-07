from __future__ import annotations

import argparse
import asyncio
import json
import statistics

import polars as pl
from scipy.stats import binomtest

from experiments.harness import EXP_OUT, Variant, gold_cases, gold_records, measure, sample_items, write_run
from experiments.homograph_verifier import NONE, gold_answer, wilson
from mds_norm.pipeline import vocab_rerank as vr
from mds_norm.pipeline.vocab_indexes import GROUP_FOR

OUT = EXP_OUT / "rerank"
TARGET = 0.95

# The qualified rule treats a qualified value as its concept
STRICT_RULE = (
    "- Choose a candidate only if it denotes the same concept. A broader, narrower or merely associated term"
    " is not a\n  match: for `oak` neither `wood` nor `oak gall` is the concept."
)
QUALIFIED_RULE = (
    "- Choose a candidate only if the value denotes it. A genuinely broader, narrower or merely associated term"
    " is not a\n  match: for `oak` neither `wood` nor `oak gall` is the concept.\n"
    "- A value that names a candidate concept and then qualifies it *is* that concept (`laminated printed paper`"
    " is\n  laminated paper; `carved oak` is oak). Choose the candidate the qualifier attaches to."
)

# `{context}` is empty on "none", reproducing the production prompt exactly
CONTEXT_SLOT = ("\n\nCandidate terms", "\n{context}\nCandidate terms")
PROMPTS = {
    "strict": vr.PROMPT.replace(*CONTEXT_SLOT, 1),
    "qualified": vr.PROMPT.replace(STRICT_RULE, QUALIFIED_RULE).replace(*CONTEXT_SLOT, 1),
}


TITLE_FIELDS = ("spectrum/object_name", "spectrum/title")
DESC_FIELDS = ("spectrum/brief_description", "spectrum/physical_description", "spectrum/description")
DESC_CHARS = 600
MAX_SIBLINGS = 8

# Cumulative contexts; cosine is the rung the model replaced
COSINE = "cosine"
CONTEXTS = {
    COSINE: (),
    "none": (),
    "entry": ("entry",),
    "title": ("entry", "title"),
    "siblings": ("entry", "siblings"),
    "description": ("entry", "description"),
    "all": ("entry", "title", "siblings", "description"),
}

# The embedding rung's accept rule, as the cascade ran it
COSINE_ACCEPT = 0.35
COSINE_MARGIN = 0.10


def truncate(text: str, limit: int = DESC_CHARS) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + " …"


def record_lines(record: dict, item: dict, parts: tuple[str, ...]) -> list[tuple[str, str]]:
    """The requested slices of one record the atom's value appears in"""
    nodes = [n for n in record["nodes"] if n.get("value")]
    out = []
    if "title" in parts:
        for field in TITLE_FIELDS:
            for n in nodes:
                if n["field_type"] == field:
                    out.append((n["label"], truncate(n["value"], 120)))
                    break
    if "siblings" in parts:
        seen: dict[str, str] = {}
        for n in nodes:
            if n["field_type"] in GROUP_FOR and n["value"] != item.get("value") and n["label"] not in seen:
                seen[n["label"]] = truncate(n["value"], 120)
        out += list(seen.items())[:MAX_SIBLINGS]
    if "description" in parts:
        for field in DESC_FIELDS:
            n = next((n for n in nodes if n["field_type"] == field), None)
            if n:
                out.append((n["label"], truncate(n["value"])))
                break
    return out


def context_of(item: dict, records: dict[str, dict], parts: tuple[str, ...]) -> str:
    """The context block for one atom, empty when nothing it asks for is present"""
    lines = []
    if "entry" in parts and item.get("value") and item["value"] != item["atom"]:
        lines.append(("field entry", truncate(item["value"], 200)))
    record = next((records[r] for r in item.get("exemplars") or [] if r in records), None)
    if record and parts:
        lines += record_lines(record, item, parts)
    if not lines:
        return ""
    body = "\n".join(f"  {label}: {value}" for label, value in lines)
    return f"\nThe record it appears in says:\n\n{body}\n"


def work_items(task: str, rung: str | None) -> list[dict]:
    """Labelled gold items, each with the answer to score against"""
    items = {i["id"]: i for i in sample_items(task)}
    out = []
    for iid, case in gold_cases(task).items():
        item = items.get(iid)
        if item is None or (rung and item.get("sub_component") != rung):
            continue
        answer = gold_answer(item, case)
        if answer is None:  # cant_tell
            continue
        verdict = (case.get("expected") or {}).get("verdict")
        # Rerank gold was labelled against the same candidates shown
        out.append(item | {"gold": answer, "verdict": verdict, "adjudicated": task == "rerank"})
    return out


def stored_candidates(work: list[dict]) -> pl.DataFrame:
    """The candidate sets the gold was labelled against, as the frame recorded them"""
    rows = [
        {
            "group": w["field_type"],
            "norm": w["norm"],
            "subject": c["subject"],
            "matched_term": c["term"],
            "vocab": c["vocab"],
            "gloss": c["gloss"],
            "score": c["score"],
            "rank": c["rank"],
        }
        for w in work
        for c in w["candidates"]
    ]
    return pl.from_dicts(rows).unique(["group", "norm", "rank"]).sort("group", "norm", "rank")


def queue_frame(work: list[dict], records: dict[str, dict], parts: tuple[str, ...]) -> pl.DataFrame:
    """The gold atoms shaped like the rung's own queue, carrying the requested context"""
    return (
        pl.DataFrame(
            [
                {
                    "group": w["field_type"],
                    "norm": w["norm"],
                    "atom": w["atom"],
                    "occ": w["n_occ"],
                    "context": context_of(w, records, parts),
                }
                for w in work
            ]
        )
        .group_by("group", "norm")
        .agg(atom=pl.col("atom").first(), occ=pl.col("occ").sum(), context=pl.col("context").first())
        .sort("group", "norm")
    )


def decide_cosine(cands: pl.DataFrame, accept: float = COSINE_ACCEPT, margin: float = COSINE_MARGIN) -> pl.DataFrame:
    """The top candidate where it clears the floor and beats the runner-up by the margin"""
    ranked = (
        cands.sort("rank")
        .group_by("group", "norm")
        .agg(pl.col("subject"), pl.col("matched_term"), pl.col("vocab"), pl.col("score"))
    )
    top = pl.col("score").list.get(0)
    # An unlisted runner-up sat below the retrieval floor
    second = pl.col("score").list.get(1, null_on_oob=True).fill_null(vr.RETRIEVE_FLOOR)
    take = (top >= accept) & ((top - second) >= margin)
    return ranked.select(
        "group",
        "norm",
        # The rule always decides; a cosine cannot answer nothing
        answered=pl.lit(True),
        pick=pl.when(take).then(1).cast(pl.Int64),
        n_candidates=pl.col("subject").list.len().cast(pl.UInt32),
        vocab=pl.when(take).then(pl.col("vocab").list.get(0)),
        subject=pl.when(take).then(pl.col("subject").list.get(0)),
        matched_term=pl.when(take).then(pl.col("matched_term").list.get(0)),
        score=pl.when(take).then(top),
    ).sort("group", "norm")


def classify(row: dict) -> str:
    """What the rung did about this item, against what the gold says the answer is"""
    pick, gold = row["pick"], row["gold"]
    if not row["answered"]:
        return "unanswered"
    if pick is None:
        return "rejected_correctly" if gold == NONE else "missed"
    if pick == gold:
        return "selected_correctly"
    if gold != NONE:
        return "selected_wrongly"  # gold names a different concept for this atom
    if row["adjudicated"]:
        return "selected_wrongly"  # the labeller saw this candidate and did not choose it
    # Replayed gold judged "none" against the old candidate set
    return "repeated_the_old_error" if pick == row["pipeline_subject"] else "unlabelled"


def score(preds: pl.DataFrame) -> dict:
    rows = preds.to_dicts()
    n = len(rows)
    counts = preds["outcome"].value_counts().sort("count", descending=True)

    answered = [r for r in rows if r["answered"]]
    asserted = [r for r in rows if r["pick"] is not None]
    sure = sum(r["outcome"] == "selected_correctly" for r in rows)
    unlabelled = sum(r["outcome"] == "unlabelled" for r in rows)
    lo_lo, lo_hi = wilson(sure, len(asserted))
    hi_lo, hi_hi = wilson(sure + unlabelled, len(asserted))

    agree = sum(r["outcome"] in ("selected_correctly", "rejected_correctly") for r in rows)
    agree_lo, agree_hi = wilson(agree, len(answered))

    # Retrieval caps the decision: an unsurfaced concept is never offered
    with_subject = [r for r in rows if r["gold"] != NONE]
    retrieved = sum(r["gold_retrieved"] for r in with_subject)
    found = sum(r["outcome"] == "selected_correctly" for r in with_subject)
    rec_lo, rec_hi = wilson(found, len(with_subject))

    report = {
        "target": TARGET,
        "n": n,
        "outcomes": {r["outcome"]: r["count"] for r in counts.iter_rows(named=True)},
        "rung": {
            "n_asserted": len(asserted),
            "reject_rate": (len(answered) - len(asserted)) / len(answered) if answered else None,
            "precision_lower": sure / len(asserted) if asserted else None,
            "precision_lower_wilson": [lo_lo, lo_hi],
            "precision_upper": (sure + unlabelled) / len(asserted) if asserted else None,
            "precision_upper_wilson": [hi_lo, hi_hi],
            "n_unadjudicated": unlabelled,
            "n_answered": len(answered),
            "agreement_with_gold": agree / len(answered) if answered else None,
            "agreement_wilson": [agree_lo, agree_hi],
            "link_recall": found / len(with_subject) if with_subject else None,
            "link_recall_wilson": [rec_lo, rec_hi],
        },
        "retrieval": {
            "n_with_known_subject": len(with_subject),
            "recall_at_k": retrieved / len(with_subject) if with_subject else None,
            "k": vr.TOP_K,
        },
    }
    # Only the replayed homograph gold carries the old rung
    if any(r["pipeline_subject"] for r in rows):
        old_ok = sum(r["gold"] == r["pipeline_subject"] and r["gold"] != NONE for r in rows)
        old_lo, old_hi = wilson(old_ok, n)
        report["old_rung"] = {"precision": old_ok / n, "wilson": [old_lo, old_hi], "n_asserted": n}
        report["vs_old_rung"] = mcnemar(
            [r["outcome"] in ("selected_correctly", "rejected_correctly") for r in rows],
            [r["gold"] == r["pipeline_subject"] and r["gold"] != NONE for r in rows],
        )
    return report


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact paired test on the items the two variants disagree about"""
    wins = sum(x and not y for x, y in zip(a, b, strict=True))
    losses = sum(y and not x for x, y in zip(a, b, strict=True))
    p = binomtest(wins, wins + losses).pvalue if wins + losses else None
    return {"wins": wins, "losses": losses, "p_value": p}


def predictions(work: list[dict], decisions: pl.DataFrame, cands: pl.DataFrame) -> pl.DataFrame:
    """One row per gold item: what the matcher decided, and what the gold makes of it"""
    picked = {(r["group"], r["norm"]): r for r in decisions.iter_rows(named=True)}
    offered = {
        (g, nrm): set(sub)
        for g, nrm, sub in cands.group_by("group", "norm")
        .agg(pl.col("subject"))
        .select("group", "norm", "subject")
        .iter_rows()
    }
    rows = []
    for w in work:
        key = (w["field_type"], w["norm"])
        hit = picked.get(key)
        row = {
            "id": w["id"],
            "group": w["field_type"],
            "atom": w["atom"],
            "value": w.get("value"),
            "n_occ": w["n_occ"],
            "stratum": w.get("stratum"),
            "verdict": w["verdict"],
            "gold": w["gold"],
            "adjudicated": w["adjudicated"],
            "pipeline_subject": w.get("subject"),
            "answered": bool(hit and hit["answered"]),
            "pick": hit["subject"] if hit else None,
            "pick_term": hit["matched_term"] if hit else None,
            "pick_cosine": hit["score"] if hit else None,
            "n_offered": len(offered.get(key, ())),
            "gold_retrieved": w["gold"] in offered.get(key, ()),
        }
        rows.append(row | {"outcome": classify(row)})
    return pl.from_dicts(rows)


async def run(
    variant: Variant, work: list[dict], queue: pl.DataFrame, cands: pl.DataFrame, prompt: str
) -> tuple[pl.DataFrame, dict]:
    with measure("rerank", variant.name) as cost:
        if variant.model is None:
            decisions = decide_cosine(cands)
        else:
            decisions = await vr.choose(cands, queue, model=variant.model, prompt=prompt, **variant.decoding)

    preds = predictions(work, decisions, cands)
    metrics = {"n_items": len(preds), **cost, "wh_per_item": cost["energy_wh"] / len(preds)}
    write_run("rerank", variant, preds, metrics)
    return preds, metrics


def cosine_sweep(work: list[dict], cands: pl.DataFrame) -> list[dict]:
    """The rule's whole operating curve, since its cascade setting decides almost nothing here"""

    def fmt(v: float | None) -> str:
        return "  —  " if v is None else f"{v:5.3f}"

    out = []
    print(f"\n{'accept':>7} {'margin':>7} {'assert':>7} {'prec':>7} {'agree':>7} {'recall':>7}")
    for accept in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        for margin in (0.0, 0.01, 0.02, 0.05, 0.10):
            rung = score(predictions(work, decide_cosine(cands, accept, margin), cands))["rung"]
            out.append({"accept": accept, "margin": margin} | rung)
            print(
                f"{accept:7.2f} {margin:7.2f} {rung['n_asserted']:7d} {fmt(rung['precision_lower'])} "
                f"{fmt(rung['agreement_with_gold'])} {fmt(rung['link_recall'])}"
            )
    return out


def context_variant(args: argparse.Namespace, context: str, replicate: int) -> Variant:
    if context == COSINE:
        return Variant(
            name=f"{COSINE}:{args.task}",
            params={"context": COSINE, "accept": COSINE_ACCEPT, "margin": COSINE_MARGIN},
            notes="the embedding rung the model replaced, over the same retrieved candidates",
        )
    name = f"{args.model}:{args.task}:{args.prompt}:{context}" + (f":r{replicate}" if args.replicates > 1 else "")
    extra_body = dict(vr.LLM_EXTRA_BODY)
    if args.no_think:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    return Variant(
        name=name,
        model=args.model,
        base_url=vr.LLM_API_BASE,
        concurrency=vr.LLM_CONCURRENCY,
        decoding={"temperature": vr.LLM_TEMPERATURE, "extra_body": extra_body},
        params={"context": context, "parts": list(CONTEXTS[context]), "replicate": replicate},
        notes="production rerank configuration, one record context",
    )


def consensus(preds: list[pl.DataFrame]) -> dict[str, bool]:
    """Per item, whether the context answered it correctly in most of its replicates"""
    votes: dict[str, list[bool]] = {}
    for p in preds:
        for r in p.iter_rows(named=True):
            ok = r["outcome"] in ("selected_correctly", "rejected_correctly")
            votes.setdefault(r["id"], []).append(ok)
    return {iid: sum(v) * 2 > len(v) for iid, v in votes.items()}


def mean_of(reports: list[dict], key: str) -> float | None:
    vals = [r["rung"][key] for r in reports if r["rung"][key] is not None]
    return statistics.mean(vals) if vals else None


def summarise(contexts: dict[str, list[dict]], preds: dict[str, list[pl.DataFrame]], baseline: str) -> dict:
    """Mean over replicates per context, each paired against the baseline on majority verdicts"""

    def fmt(v: float | None) -> str:
        return "  —  " if v is None else f"{v:5.3f}"

    base = consensus(preds[baseline]) if baseline in preds else {}
    head = (
        f"{'context':>12} {'runs':>4}  {'agree':>7} {'sd':>6}  "
        f"{'assert':>6}  {'prec':>7}  {'recall':>7}  {'wh/item':>8}"
    )
    print(f"\n{head}   vs {baseline} (win/loss, p)")
    out = {}
    for context, reports in contexts.items():
        agree = [r["rung"]["agreement_with_gold"] for r in reports]
        sd = statistics.stdev(agree) if len(agree) > 1 else 0.0
        context_ok = consensus(preds[context])
        shared = sorted(set(context_ok) & set(base))
        paired = mcnemar([context_ok[i] for i in shared], [base[i] for i in shared]) if context != baseline else None
        out[context] = {
            "n_runs": len(reports),
            "agreement_mean": statistics.mean(agree),
            "agreement_sd": sd,
            "n_asserted_mean": statistics.mean(r["rung"]["n_asserted"] for r in reports),
            "precision_mean": mean_of(reports, "precision_lower"),
            "link_recall_mean": mean_of(reports, "link_recall"),
            "wh_per_item_mean": statistics.mean(r["wh_per_item"] for r in reports),
            f"vs_{baseline}": paired,
        }
        o = out[context]
        pair = "" if paired is None else f"   {paired['wins']}/{paired['losses']}, p={paired['p_value']:.3f}"
        print(
            f"{context:>12} {len(reports):>4}  {fmt(o['agreement_mean'])} {sd:6.3f}  "
            f"{o['n_asserted_mean']:6.1f}  {fmt(o['precision_mean'])}  "
            f"{fmt(o['link_recall_mean'])}  {o['wh_per_item_mean']:8.4f}{pair}"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the rerank rung against gold, one record context at a time.")
    ap.add_argument("--task", default="rerank", choices=["rerank", "homograph"], help="which gold sample to score on")
    ap.add_argument("--rung", default=None, help="restrict to one sub_component of the homograph gold, e.g. semantic")
    ap.add_argument("--contexts", default="none", help="comma list of record contexts, or 'all'")
    ap.add_argument(
        "--replicates", type=int, default=1, help="repeats per context; the contexts differ by less than the sd"
    )
    ap.add_argument(
        "--baseline", default="none", choices=sorted(CONTEXTS), help="context the others are paired against"
    )
    ap.add_argument("--model", default=vr.LLM_MODEL)
    ap.add_argument("--prompt", default="strict", choices=sorted(PROMPTS))
    # LFM2.5-8B-A1B ignores this and reasons anyway
    ap.add_argument("--no-think", action="store_true", help="ask the chat template to skip the thinking block")
    ap.add_argument("--cosine-sweep", action="store_true", help="score the cosine rule over a floor/margin grid")
    args = ap.parse_args()
    contexts = list(CONTEXTS) if args.contexts == "all" else args.contexts.split(",")

    work = work_items(args.task, args.rung)
    records = gold_records()
    print(f"{len(work)} labelled {args.rung or args.task} items")

    OUT.mkdir(parents=True, exist_ok=True)
    # Retrieval ignores context, so all contexts share one candidate set
    cands = stored_candidates(work) if args.task == "rerank" else vr.candidates(queue_frame(work, records, ()))
    if args.cosine_sweep:
        sweep = cosine_sweep(work, cands)
        (OUT / f"cosine_sweep_{args.task}.json").write_text(json.dumps(sweep, indent=2))

    reports: dict[str, list[dict]] = {}
    preds_by_variant: dict[str, list[pl.DataFrame]] = {}
    for context in contexts:
        queue = queue_frame(work, records, CONTEXTS[context])
        with_context = int((queue["context"].str.len_chars() > 0).sum())
        print(f"\n--- {context}: {len(queue)} queries, {with_context} carrying context")
        for replicate in range(1 if context == COSINE else args.replicates):  # the cosine rule is deterministic
            variant = context_variant(args, context, replicate)
            preds, metrics = asyncio.run(run(variant, work, queue, cands, PROMPTS[args.prompt]))
            report = score(preds) | {"variant": context, "parts": list(CONTEXTS[context]), "replicate": replicate}
            if not report["rung"]["n_answered"]:
                raise SystemExit(f"{variant.name}: every call came back empty — is {variant.base_url} serving?")
            report["wh_per_item"] = metrics["wh_per_item"]
            reports.setdefault(context, []).append(report)
            preds_by_variant.setdefault(context, []).append(preds)
            (OUT / f"{variant.name.replace(':', '_')}_comparison.json").write_text(json.dumps(report, indent=2))
            r = report["rung"]

            # A dead endpoint answers nothing, so rates may be None
            def fmt(v: float | None) -> str:
                return "—" if v is None else f"{v:.3f}"

            print(
                f"  r{replicate}: agreement {fmt(r['agreement_with_gold'])} "
                f"({r['n_asserted']} asserted, precision {fmt(r['precision_lower'])}, "
                f"recall {fmt(r['link_recall'])})"
            )

    summary = summarise(reports, preds_by_variant, args.baseline)
    out = OUT / f"context_variants_{args.task}_{args.prompt}.json"
    out.write_text(json.dumps({"summary": summary, "runs": reports}, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
