from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re

import polars as pl

from experiments.harness import EXP_OUT, SEED, Variant, gold_cases, measure, sample_items, write_run
from mds_norm.paths import FIELD_STATS, RAW_RECORDS

RAW = RAW_RECORDS

OUT = EXP_OUT / "routing"
TARGET = 0.95  # the bar a family must clear to earn production routing
HOLDOUT = 0.5  # test share of the deterministic dev/test split

NONE = "none"


def split_of(item_id: str, holdout: float = HOLDOUT) -> str:
    """Deterministic dev/test assignment, stable across runs and independent of PYTHONHASHSEED"""
    h = int(hashlib.blake2b(f"{SEED}:{item_id}".encode(), digest_size=8).hexdigest(), 16)
    return "test" if (h % 10_000) / 10_000.0 < holdout else "dev"


# Value family -> candidate fields, each with a meaning
FAMILY_FIELDS = {
    "date": [
        ("spectrum/object_production_date", "the date this object was made or produced"),
        ("spectrum/acquisition_date", "when the museum acquired the object"),
        ("spectrum/field_collection_date", "when the object was collected in the field"),
        (
            "spectrum/association_date",
            "a date the object is associated with, commemorates, or depicts, or the date of an original it reproduces",
        ),
    ],
    "dimension": [
        ("spectrum/dimension", "a physical measurement of the whole object (height, width, length, diameter, weight)"),
        (
            "spectrum/technical_attribute_measurement",
            "a technical specification of the object or a part (calibre, gauge, voltage, focal length)",
        ),
    ],
    "price": [
        ("spectrum/object_purchase_price", "the price the museum paid to acquire the object"),
        ("spectrum/object_valuation", "an appraised or insurance value, not a purchase price"),
    ],
}

FAMILY_OF = {}
for fam, fields in FAMILY_FIELDS.items():
    for f, _ in fields:
        FAMILY_OF[f] = fam
# extra proposed-destination aliases that map into a family
FAMILY_OF["spectrum/technical_attribute"] = "dimension"

SYSTEM = (
    "You are a museum cataloguing assistant. A value has already been extracted verbatim from a catalogue "
    "record; your only job is to decide which catalogue field it belongs in, using the surrounding record as "
    "evidence. You never change or re-extract the value.\n"
    "\n"
    'Choose exactly one option by its number. Choose "none" when the value is not a genuine member of the '
    "offered field family for THIS object (for example a date the object merely depicts when no field fits, "
    "or a measurement that is really a scale ratio), or when it belongs in none of the listed fields.\n"
    "\n"
    'Answer with only the number, e.g. "2".'
)

# Per-family guidance, empty where the generic instruction sufficed
FAMILY_GUIDANCE = {
    "date": (
        "Guidance for dates. In museum cataloguing a date recorded against an "
        "object is, by default, when the object was MADE — choose "
        "object_production_date unless the record gives specific evidence the date "
        "belongs elsewhere:\n"
        "  • an acquisition/accession/purchase/donation context → acquisition_date;\n"
        "  • a field-collection/expedition/excavation context → field_collection_date;\n"
        "  • the date is of an event, person or place the object only depicts, "
        "commemorates or reproduces (not when it was made) → association_date.\n"
        "Prefer object_production_date when the evidence is ambiguous. Choose none "
        "only if the value is not a date of this object at all."
    )
}


def family_of(item: dict) -> str | None:
    return FAMILY_OF.get(item["field_type"])


def gold_destination(item: dict, case: dict) -> str | None:
    """The known-correct field for scoring, or None when unknown/excluded"""
    v = (case.get("expected") or {}).get("verdict")
    if v == "correct_destination":
        return item["field_type"]
    if v == "not_extractable":
        return NONE
    if v == "wrong_destination":
        return (case.get("expected") or {}).get("correct_field")  # may be None
    return None  # wrong_value -> excluded


def render_record(rows: list[dict], source_field: str) -> str:
    lines = []
    for r in rows:
        val = r["value"]
        if val is None or val == "":
            continue
        field = r["field_type"].removeprefix("spectrum/")
        mark = "  <-- value extracted from here" if r["field_type"] == source_field else ""
        lines.append(f"  {field}: {val}{mark}")
    return "\n".join(lines[:40])


def build_prompt(item: dict, rows: list[dict]) -> tuple[str, list[str]]:
    fam = family_of(item)
    fields = FAMILY_FIELDS[fam]
    source_field = (item.get("op_detail") or {}).get("source_field") or item["field_type"]
    record = render_record(rows, source_field)
    opts = [f"  {i + 1}. {f.removeprefix('spectrum/')} — {desc}" for i, (f, desc) in enumerate(fields)]
    opts.append(
        f"  {len(fields) + 1}. none — the value is not a genuine {fam} of this object, or belongs in no field above"
    )
    guidance = FAMILY_GUIDANCE.get(fam, "")
    guidance_block = f"{guidance}\n\n" if guidance else ""
    prompt = (
        f"Record ({item['data_source']}):\n{record}\n\n"
        f'The value "{item["value"]}" was extracted from the '
        f"'{source_field.removeprefix('spectrum/')}' field.\n"
        f"{guidance_block}"
        f"Which field does this value belong in?\n" + "\n".join(opts) + "\n\n"
        "Answer with only the option number."
    )
    choices = [f for f, _ in fields] + [NONE]
    return prompt, choices


def parse_choice(content: str | None, choices: list[str]) -> str | None:
    if not content:
        return None
    m = re.search(r"\d+", content)
    if not m:
        return None
    idx = int(m.group()) - 1
    return choices[idx] if 0 <= idx < len(choices) else None


def fetch_records(record_ids: list[str]) -> dict[str, list[dict]]:
    flat = (
        pl.scan_parquet(RAW)
        .filter(pl.col("record_id").is_in(record_ids) & pl.col("field_type").str.starts_with("spectrum/"))
        .select("record_id", pl.col("data_source").cast(pl.String), "node_id", "field_type")
        .join(pl.scan_parquet(FIELD_STATS).select("node_id", "value"), on="node_id", how="left")
        .collect(engine="streaming")
    )
    return {
        (k if isinstance(k, str) else k[0]): sub.to_dicts()
        for k, sub in flat.partition_by("record_id", as_dict=True).items()
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


async def run(variant: Variant, limit: int | None) -> pl.DataFrame:
    items = {i["id"]: i for i in sample_items("placement")}
    gold = gold_cases("placement")

    # routable, in-family items with a value and a recognised family
    work = []
    for iid, case in gold.items():
        it = items.get(iid)
        if it is None or family_of(it) is None or not it.get("value"):
            continue
        work.append((it, case))
    if limit:
        work = work[:limit]

    records = fetch_records(sorted({it["record_id"] for it, _ in work}))
    units, prompts = [], []
    for it, case in work:
        rows = records.get(it["record_id"])
        if not rows:
            continue
        prompt, choices = build_prompt(it, rows)
        units.append((it, case, choices))
        prompts.append(prompt)

    from mds_norm.utils.inference import Inference

    inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)
    with measure("routing", variant.name) as cost:
        replies = await inf.generate(prompts, system=SYSTEM, usage=True, **variant.decoding)

    rows = []
    for (it, case, choices), r in zip(units, replies, strict=True):
        pick = parse_choice(r["content"], choices)
        rows.append(
            {
                "id": it["id"],
                "record_id": it["record_id"],
                "family": family_of(it),
                "value": it["value"],
                "proposed": it["field_type"],
                "verdict": (case.get("expected") or {}).get("verdict"),
                "gold_dest": gold_destination(it, case),
                "router_pick": pick,
                "split": split_of(it["id"]),
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
    write_run("routing", variant, preds, metrics)
    return preds


def _score_rows(rows: list[dict]) -> dict:
    """Per-family and overall routing precision for one set of prediction rows"""
    report = {"families": {}, "overall": {}}
    fam_rows = {f: [] for f in FAMILY_FIELDS}
    for row in rows:
        if row["family"] in fam_rows:
            fam_rows[row["family"]].append(row)
    overall_in = overall_router_ok = overall_pipe_ok = 0
    for fam, frows in fam_rows.items():
        cands = {f for f, _ in FAMILY_FIELDS[fam]} | {NONE}
        scored = [r for r in frows if r["gold_dest"] in cands]
        router_ok = sum(r["router_pick"] == r["gold_dest"] for r in scored)
        pipe_ok = sum(r["proposed"] == r["gold_dest"] for r in scored)
        n = len(scored)
        lo, hi = wilson(router_ok, n)
        avoid_pool = [r for r in frows if r["verdict"] == "wrong_destination" and r["gold_dest"] is None]
        avoided = sum(r["router_pick"] != r["proposed"] for r in avoid_pool)
        report["families"][fam] = {
            "n_scored": n,
            "router_accuracy": router_ok / n if n else None,
            "router_wilson": [lo, hi],
            "pipeline_baseline": pipe_ok / n if n else None,
            "n_avoidance": len(avoid_pool),
            "avoidance_rate": avoided / len(avoid_pool) if avoid_pool else None,
            "clears_target": (lo is not None and lo >= TARGET),
            "verdict": (
                "route"
                if (lo is not None and lo >= TARGET)
                else "beats_pipeline"
                if (n and pipe_ok and router_ok > pipe_ok)
                else "insufficient"
            ),
        }
        overall_in += n
        overall_router_ok += router_ok
        overall_pipe_ok += pipe_ok
    lo, hi = wilson(overall_router_ok, overall_in)
    report["overall"] = {
        "n_scored": overall_in,
        "router_accuracy": overall_router_ok / overall_in if overall_in else None,
        "router_wilson": [lo, hi],
        "pipeline_baseline": overall_pipe_ok / overall_in if overall_in else None,
    }
    return report


def score(preds: pl.DataFrame) -> dict:
    """Report on the full gold and on the deterministic dev/test split"""
    rows = preds.to_dicts()
    return {
        "target": TARGET,
        "n_items": preds.height,
        "holdout": HOLDOUT,
        "all": _score_rows(rows),
        "dev": _score_rows([r for r in rows if r["split"] == "dev"]),
        "test": _score_rows([r for r in rows if r["split"] == "test"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="gpt-oss-20b")
    args = ap.parse_args()

    # gpt-oss reasons before answering, so leave room for content
    variant = Variant(
        name="gpt-oss-20b",
        model=args.model,
        decoding={"temperature": 0.0, "top_p": 1.0, "reasoning_effort": "low", "max_tokens": 512},
        notes="candidate routing over placement gold",
    )
    preds = asyncio.run(run(variant, args.limit))
    report = score(preds)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "routing_precision.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fmt(v: float | None) -> str:
        return "  —  " if v is None else f"{v:5.3f}"

    print(
        f"\ncandidate routing over placement gold — {preds.height} routable ops (dev/test split, holdout={HOLDOUT})\n"
    )
    for split in ("all", "dev", "test"):
        sub = report[split]
        print(f"[{split}]  {'family':>10}  {'n':>4}  router  [wilson95]      pipeline  avoid  verdict")
        for fam, r in sub["families"].items():
            lo, hi = r["router_wilson"]
            print(
                f"       {fam:>10}  {r['n_scored']:>4}  {fmt(r['router_accuracy'])}  "
                f"[{fmt(lo)},{fmt(hi)}]  {fmt(r['pipeline_baseline'])}  "
                f"{fmt(r['avoidance_rate'])}  {r['verdict']}"
            )
        o = sub["overall"]
        print(
            f"       {'overall':>10}  {o['n_scored']:>4}  {fmt(o['router_accuracy'])}  "
            f"[{fmt(o['router_wilson'][0])},{fmt(o['router_wilson'][1])}]  "
            f"{fmt(o['pipeline_baseline'])}\n"
        )
    print(f"→ {OUT / 'routing_precision.json'}")


if __name__ == "__main__":
    main()
