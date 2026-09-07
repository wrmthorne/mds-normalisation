from __future__ import annotations

import argparse
import asyncio
import json
import time

import polars as pl

from experiments.extraction_common import load_requests, parse_ops_fields
from experiments.harness import (
    EXP_OUT,
    SAMPLES,
    Variant,
    gold_cases,
    load_predictions,
    log,
    measure,
    sample_items,
    write_run,
)
from experiments.score_extraction import _key as op_key
from mds_norm.utils.patches import FIELD_FOR_TASK, TASK_TEXT_V2, strip_prefix, validate_op

EXP = "nuextract_trial"
BASE_URL = "http://localhost:30000/v1"
QUEUE_RECORDS = 1_289_514  # data/extraction/queue.parquet (the run this must pay for)

# The production variant's measured rate, not a re-derived estimate
PRODUCTION = {"wh_per_record": 0.00963, "records_per_s": 16.1, "source": "data/extraction/progress.jsonl"}

TASK_KEYS = ("date", "material", "dimension")


# One typed leaf per task; a list holds repeats

LEAF = "verbatim-string"
TEMPLATE_KEY = {k: strip_prefix(v) for k, v in FIELD_FOR_TASK.items()}

# Contract v2's task text without its `- add <field>:` framing
INSTRUCTION_HEAD = (
    "The document is one UK museum object record, rendered as JSON. Extract only"
    " facts stated about the catalogued object itself. Record text often describes"
    " other things: general background essays, the class of objects this one"
    " belongs to, an original that this object reproduces, the subject it depicts,"
    " or a container it is kept in. Facts about those are not facts about this"
    " object. Many records state nothing for a field; leaving it empty is a"
    " correct and common answer."
)


def instructions_for(tasks: list[str]) -> str:
    body = [TASK_TEXT_V2[k].split(":", 1)[1].strip() for k in tasks]
    return "\n".join([INSTRUCTION_HEAD] + [f"{TEMPLATE_KEY[k]}: {t}" for k, t in zip(tasks, body, strict=True)])


def template_for(tasks: list[str]) -> str:
    return json.dumps({TEMPLATE_KEY[k]: [LEAF] for k in tasks}, ensure_ascii=False)


def unit_key(tasks: tuple[str, ...]) -> str:
    return tasks[0] if len(tasks) == 1 else "composed"


def _tasks_of(req: dict) -> list[str]:
    """The task keys a request carries, recovered from its admitted fields"""
    return [k for k in TASK_KEYS if TEMPLATE_KEY[k] in req["allowed"]]


PRED_SCHEMA = {
    "id": pl.String,
    "record_id": pl.String,
    "unit": pl.String,
    "op": pl.String,
    "field": pl.String,
    "value": pl.String,
    "source_field": pl.String,
    "span_start": pl.UInt32,
    "span_end": pl.UInt32,
    "status": pl.String,
    "reason": pl.String,
}


async def predict_nuextract(variant: Variant, limit: int | None) -> None:
    """One call per (record, task group)"""
    from mds_norm.utils.inference import Inference

    items = sample_items("extraction", limit)
    reqs = load_requests(items)
    scale = variant.params.get("prompt_scale", "composed")

    groups: dict[tuple[str, ...], list[dict]] = {}
    for req in reqs:
        tasks = _tasks_of(req)
        if not tasks:
            continue
        for group in [tuple(tasks)] if scale == "composed" else [(t,) for t in tasks]:
            groups.setdefault(group, []).append(req)

    inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)
    replies: list[dict] = []
    units: list[tuple[dict, tuple[str, ...]]] = []
    with measure(EXP, variant.name) as cost:
        for tasks, group_reqs in groups.items():
            body = dict(variant.decoding)
            kwargs = {
                "template": template_for(list(tasks)),
                "instructions": instructions_for(list(tasks)),
                **body.pop("chat_template_kwargs", {}),
            }
            docs = [json.dumps(r["record_json"], indent=1, ensure_ascii=False) for r in group_reqs]
            out = await inf.generate(docs, usage=True, progress=True, chat_template_kwargs=kwargs, **body)
            replies.extend(out)
            units.extend((req, tasks) for req in group_reqs)

    rows = []
    for (req, tasks), reply in zip(units, replies, strict=True):
        base = {"id": req["item_id"], "record_id": req["record_id"], "unit": unit_key(tasks)}
        ops = parse_ops_fields(reply["content"])
        if ops is None:
            rows.append(
                base | {"status": "deferred", "reason": "llm_error" if reply["error"] else "unparseable_response"}
            )
            continue
        for op in ops:
            parsed, reason = validate_op(req, op)
            if parsed is None:
                rows.append(
                    base | {k: op.get(k) for k in ("op", "field", "value")} | {"status": "rejected", "reason": reason}
                )
            else:
                parsed.pop("node_id", None)
                rows.append(base | parsed | {"status": "resolved"})

    _write(variant, items, units, replies, rows, cost)


# The Liquid Nanos take the schema in the system prompt

COPY_RULE = (
    "Every value must be copied character-for-character from the record."
    " Omit a key entirely when the record states nothing for it;"
    " an empty object is a correct and common answer."
)

# The same rules with contract v2's illustrations removed

TASK_TEXT_TERSE = {
    "date": "the date this object itself was made or produced, if stated. Not"
    " acquisition, donation or association dates; not a date the object"
    " commemorates, depicts or carries in its content; not the date of an"
    " original it reproduces. Keep any uncertainty qualifier that appears"
    " with the date.",
    "material": "each material this object is made of, one entry per material."
    " Copy the bare material name without colour, pattern or form"
    " words, unless the compound is itself the name of the material."
    " Techniques, processes, implements, colours, places, the names of"
    " the object or its parts, and its contents are not materials"
    " unless the record states what they are made of.",
    "dimension": "this object's stated measurements, copied as written including"
    " units, one entry per measurement. Where the same measurement is"
    " stated twice, copy the unit-bearing form. A scale ratio or a"
    " size designation is not a measurement.",
}


def terse_rules(tasks: list[str]) -> str:
    return "\n".join(f"{TEMPLATE_KEY[k]}: {TASK_TEXT_TERSE[k]}" for k in tasks)


def schema_for(tasks: list[str]) -> str:
    return json.dumps({TEMPLATE_KEY[k]: ["string"] for k in tasks}, ensure_ascii=False)


def schema_system(tasks: list[str]) -> str:
    """Schema first, in the phrasing the Nanos were tuned on, then the rules"""
    return (
        "Return data as a JSON object with the following schema:\n"
        + schema_for(tasks)
        + "\n\n"
        + COPY_RULE
        + "\n\nThe document is one UK museum object record. Extract only facts"
        " stated about the catalogued object itself, not about background"
        " essays, the class of objects it belongs to, an original it"
        " reproduces, the subject it depicts, or a container it is kept in."
        "\n\n" + terse_rules(tasks)
    )


async def predict_schema_prompt(variant: Variant, limit: int | None) -> None:
    """One call per (record, task group) against a schema-prompted model"""
    from mds_norm.utils.inference import Inference

    items = sample_items("extraction", limit)
    reqs = load_requests(items)
    scale = variant.params.get("prompt_scale", "composed")
    schema_in = variant.params.get("schema_in", "system")

    groups: dict[tuple[str, ...], list[dict]] = {}
    for req in reqs:
        if tasks := _tasks_of(req):
            for g in [tuple(tasks)] if scale == "composed" else [(t,) for t in tasks]:
                groups.setdefault(g, []).append(req)

    inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)
    replies: list[dict] = []
    units: list[tuple[dict, tuple[str, ...]]] = []
    with measure(EXP, variant.name) as cost:
        for tasks, group_reqs in groups.items():
            prompts, system = _schema_prompts(list(tasks), group_reqs, schema_in)
            replies.extend(await inf.generate(prompts, system=system, usage=True, **variant.decoding))
            units.extend((req, tasks) for req in group_reqs)

    rows = []
    for (req, tasks), reply in zip(units, replies, strict=True):
        base = {"id": req["item_id"], "record_id": req["record_id"], "unit": unit_key(tasks)}
        ops = parse_ops_fields(reply["content"])
        if ops is None:
            rows.append(
                base | {"status": "deferred", "reason": "llm_error" if reply["error"] else "unparseable_response"}
            )
            continue
        for op in ops:
            parsed, reason = validate_op(req, op)
            if parsed is None:
                rows.append(
                    base | {k: op.get(k) for k in ("op", "field", "value")} | {"status": "rejected", "reason": reason}
                )
            else:
                parsed.pop("node_id", None)
                rows.append(base | parsed | {"status": "resolved"})

    _write(variant, items, units, replies, rows, cost)


def flatten_record(record: dict, prefix: str = "") -> str:
    """`field: value` lines instead of indented JSON"""
    lines: list[str] = []
    for key, value in record.items():
        if isinstance(value, dict):
            lines.append(flatten_record(value, f"{prefix}{key}."))
        elif isinstance(value, list):
            lines.extend(
                flatten_record(item, f"{prefix}{key}.") if isinstance(item, dict) else f"{prefix}{key}: {item}"
                for item in value
            )
        elif value is not None:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(line for line in lines if line)


def _schema_prompts(tasks: list[str], reqs: list[dict], schema_in: str) -> tuple[list[str], str | None]:
    docs = [flatten_record(r["record_json"]) for r in reqs]
    if schema_in == "system":
        return docs, schema_system(tasks)
    tail = "\n\nExtract, from the record above.\n" + schema_system(tasks)
    return ["Record:\n" + d + tail for d in docs], None


async def predict_baseline(variant: Variant, limit: int | None) -> None:
    """The incumbent, re-measured into this experiment's output directory"""
    from experiments.extraction_common import predict

    await predict(EXP, variant, limit)


# This variant won, so the renderer below is production's


def render_record_first(tasks: list[str], record_json: dict) -> str:
    from mds_norm.utils.patches import RECORD_FIRST, render_prompt

    return render_prompt(tasks, record_json, RECORD_FIRST)


async def predict_record_first(variant: Variant, limit: int | None) -> None:
    """`gpt-oss-20b:single_task_v2` with the record ahead of the task"""
    from mds_norm.utils.inference import Inference
    from mds_norm.utils.patches import SYSTEM_V2, parse_ops

    items = sample_items("extraction", limit)
    reqs = load_requests(items)

    # Requests grouped by record, so tasks share a cached prefix
    units = [
        (req, k, render_record_first([TASK_TEXT_V2[k]], req["record_json"])) for req in reqs for k in _tasks_of(req)
    ]

    inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)
    with measure(EXP, variant.name) as cost:
        replies = await inf.generate([u[2] for u in units], system=SYSTEM_V2, usage=True, **variant.decoding)

    rows = []
    for (req, unit, _), reply in zip(units, replies, strict=True):
        base = {"id": req["item_id"], "record_id": req["record_id"], "unit": unit}
        ops = parse_ops(reply["content"])
        if ops is None:
            rows.append(
                base | {"status": "deferred", "reason": "llm_error" if reply["error"] else "unparseable_response"}
            )
            continue
        for op in ops:
            parsed, reason = validate_op(req, op)
            if parsed is None:
                proposed = op if isinstance(op, dict) else {}
                rows.append(
                    base
                    | {k: v for k, v in proposed.items() if k in ("op", "field", "value") and isinstance(v, str)}
                    | {"status": "rejected", "reason": reason}
                )
            else:
                parsed.pop("node_id", None)
                rows.append(base | parsed | {"status": "resolved"})

    _write(variant, items, units, replies, rows, cost)


def _write(
    variant: Variant, items: list[dict], units: list, replies: list[dict], rows: list[dict], cost: dict
) -> None:
    preds = pl.from_dicts(rows, schema=PRED_SCHEMA) if rows else pl.DataFrame(schema=PRED_SCHEMA)
    accepted = preds.filter(pl.col("status") == "resolved").height
    prompt_tokens = sum(r["prompt_tokens"] or 0 for r in replies)
    completion_tokens = sum(r["completion_tokens"] or 0 for r in replies)
    tokens = prompt_tokens + completion_tokens
    duration = cost["duration_s"] or 1.0
    metrics = {
        "n_items": len(items),
        "n_requests": len(units),
        "n_errors": sum(1 for r in replies if r["error"]),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "accepted_ops": accepted,
        "tokens_per_accepted_op": tokens / accepted if accepted else None,
        **cost,
        "wh_per_record": cost["energy_wh"] / max(len(items), 1),
        "records_per_s": len(items) / duration,
        "requests_per_s": len(units) / duration,
        "completion_tokens_per_s": completion_tokens / duration,
    }
    write_run(EXP, variant, preds, metrics)


# Gold cannot fill the endpoint, so bench a real shard

BENCH_SHARD = 3  # first shard the production run has not consumed
BENCH_RECORDS = 2_000
BENCH_LOG = "bench.jsonl"


def bench_requests(n: int) -> list[dict]:
    """Production requests from a shard the production run has not reached"""
    from mds_norm.pipeline.extraction import QUEUE, shard_requests

    queue = pl.read_parquet(QUEUE)
    reqs, _ = shard_requests(BENCH_SHARD, queue, "v2")
    return reqs[:n]


def bench_calls(variant: Variant, reqs: list[dict]) -> list[tuple[dict, int]]:
    """The variant's calls as (generate kwargs, n_records) groups"""
    from mds_norm.utils.patches import SYSTEM_V2, render_prompt

    scale = variant.params.get("prompt_scale", "composed")
    body = dict(variant.decoding)

    if variant.model != "gpt-oss-20b":
        groups: dict[tuple[str, ...], list[dict]] = {}
        for req in reqs:
            if tasks := _tasks_of(req):
                for g in [tuple(tasks)] if scale == "composed" else [(t,) for t in tasks]:
                    groups.setdefault(g, []).append(req)
        if variant.model in SCHEMA_PROMPT_MODELS:
            schema_in = variant.params.get("schema_in", "system")
            calls = []
            for tasks, group in groups.items():
                prompts, system = _schema_prompts(list(tasks), group, schema_in)
                calls.append(({"samples": prompts, "system": system, **body}, len(group)))
            return calls
        kwargs = body.pop("chat_template_kwargs", {})
        return [
            (
                {
                    "samples": [json.dumps(r["record_json"], indent=1, ensure_ascii=False) for r in group],
                    "chat_template_kwargs": {
                        "template": template_for(list(tasks)),
                        "instructions": instructions_for(list(tasks)),
                        **kwargs,
                    },
                    **body,
                },
                len(group),
            )
            for tasks, group in groups.items()
        ]

    render = render_record_first if variant.params.get("prompt_order") == "record_first" else render_prompt
    prompts = [render([task], req["record_json"]) for req in reqs for task in req["tasks"]]
    return [({"samples": prompts, "system": SYSTEM_V2, **body}, len(reqs))]


async def bench(variant: Variant, n: int) -> dict:
    from mds_norm.utils.inference import Inference

    reqs = bench_requests(n)
    calls = bench_calls(variant, reqs)
    inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)

    replies: list[dict] = []
    records = 0
    with measure(EXP, f"bench:{variant.name}") as cost:
        for kwargs, n_records in calls:
            replies.extend(await inf.generate(usage=True, **kwargs))
            records += n_records

    duration = cost["duration_s"] or 1.0
    prompt_tokens = sum(r["prompt_tokens"] or 0 for r in replies)
    completion_tokens = sum(r["completion_tokens"] or 0 for r in replies)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variant": variant.name,
        "model": variant.model,
        "shard": BENCH_SHARD,
        "records": len(reqs),
        "requests": len(replies),
        "concurrency": variant.concurrency,
        "errors": sum(1 for r in replies if r["error"]),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_record": (prompt_tokens + completion_tokens) / max(len(reqs), 1),
        "duration_s": round(duration, 1),
        "records_per_s": len(reqs) / duration,
        "energy_wh": round(cost["energy_wh"], 4),
        "wh_per_record": cost["energy_wh"] / max(len(reqs), 1),
    }
    entry["projected_queue_kwh"] = entry["wh_per_record"] * QUEUE_RECORDS / 1e3
    entry["projected_queue_h"] = QUEUE_RECORDS / entry["records_per_s"] / 3600
    out = EXP_OUT / EXP / BENCH_LOG
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log(
        f"{variant.name}: {entry['records_per_s']:.1f} rec/s, "
        f"{entry['wh_per_record']:.5f} Wh/record, "
        f"{entry['tokens_per_record']:.0f} tokens/record → "
        f"{entry['projected_queue_kwh']:.1f} kWh / {entry['projected_queue_h']:.1f} h "
        f"over the queue"
    )
    return entry


def pooled_keys() -> dict[str, set[tuple[str, str]]]:
    """Per item, the (field, value) keys that were put in front of an annotator"""
    path = SAMPLES / "extraction.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        it = json.loads(line)
        out[it["id"]] = {op_key(c["field"], c["value"]) for c in (it.get("candidates") or [])}
    return out


def score_variant(name: str, gold: dict[str, dict], pool: dict[str, set], exp: str = EXP) -> dict | None:
    preds = load_predictions(exp, name)
    if preds is None:
        return None
    resolved = preds.filter(pl.col("status") == "resolved")

    tp = fp = fn = misfiled = value_ok = novel = 0
    n_items = 0
    for item_id, case in gold.items():
        if case["expected"] is None:
            continue
        n_items += 1
        gold_keys = {op_key(o.get("field"), o.get("value")) for o in case["expected"].get("ops", [])}
        gold_values = {v for _, v in gold_keys}
        pred_keys = {
            op_key(r["field"], r["value"]) for r in resolved.filter(pl.col("id") == item_id).iter_rows(named=True)
        }
        for k in pred_keys:
            if k in gold_keys:
                tp += 1
                value_ok += 1
                continue
            fp += 1
            if k[1] in gold_values:
                value_ok += 1
                misfiled += 1
            if k not in pool.get(item_id, set()):
                novel += 1
        fn += len(gold_keys - pred_keys)

    if not n_items:
        return None
    runs = pl.read_ndjson(EXP_OUT / exp / "runs.jsonl").filter(pl.col("variant") == name).tail(1).to_dicts()[0]
    n_pred = tp + fp
    judged = n_pred - novel
    wh_per_record = runs.get("wh_per_record")
    # The extraction runner journals no rate; derive it
    rate = runs.get("records_per_s") or (runs.get("n_items") or 0) / max(runs.get("duration_s") or 0, 1e-9)
    return {
        "variant": name,
        "model": runs.get("model"),
        "n_items": n_items,
        "pred_ops": n_pred,
        "gold_ops_missed": fn,
        "precision_strict": tp / n_pred if n_pred else None,
        "precision_judged": tp / judged if judged else None,
        "unjudged_ops": novel,
        "recall_strict": tp / (tp + fn) if (tp + fn) else None,
        "f1_strict": 2 * tp / (2 * tp + fp + fn) if n_pred or fn else None,
        "placement_error": misfiled / value_ok if value_ok else None,
        "deferred": preds.filter(pl.col("status") == "deferred").height,
        "rejected": preds.filter(pl.col("status") == "rejected").height,
        "prompt_tokens": runs.get("prompt_tokens"),
        "completion_tokens": runs.get("completion_tokens"),
        "tokens_per_accepted_op": runs.get("tokens_per_accepted_op"),
        "duration_s": runs.get("duration_s"),
        "records_per_s": rate,
        "energy_wh": runs.get("energy_wh"),
        "wh_per_record": wh_per_record,
        "projected_queue_kwh": (wh_per_record * QUEUE_RECORDS / 1e3 if wh_per_record else None),
    }


def score(names: list[str] | None = None, exp: str = EXP) -> pl.DataFrame:
    gold = gold_cases("extraction")
    pool = pooled_keys()
    if names is None:
        runs = pl.read_ndjson(EXP_OUT / exp / "runs.jsonl")
        names = runs["variant"].unique(maintain_order=True).to_list()
    rows = [r for name in names if (r := score_variant(name, gold, pool, exp))]
    if not rows:
        raise SystemExit(f"nothing to score under {EXP_OUT / exp}")
    results = pl.DataFrame(rows).sort("f1_strict", descending=True, nulls_last=True)
    out = EXP_OUT / exp / "results.parquet"
    results.write_parquet(out)
    log(f"results → {out}")
    return results


# Local to the trial; `variants.py` definitions must not change

_NX_DECODING = {"temperature": 0.0, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False}}
_GPT_OSS_DECODING = {"temperature": 0.2, "top_p": 1.0, "reasoning_effort": "low", "max_tokens": 1024}

VARIANTS = {
    v.name: v
    for v in [
        Variant(
            "nuextract3:composed",
            model="NuExtract3",
            base_url=BASE_URL,
            concurrency=256,
            decoding=_NX_DECODING,
            params={"prompt_scale": "composed", "representation": "template"},
            notes="one call per record; template carries every missing task",
        ),
        Variant(
            "nuextract3:single_task",
            model="NuExtract3",
            base_url=BASE_URL,
            concurrency=256,
            decoding=_NX_DECODING,
            params={"prompt_scale": "single_task", "representation": "template"},
            notes="one call per task — the scale the production variant uses",
        ),
        Variant(
            "nuextract3:composed_think",
            model="NuExtract3",
            base_url=BASE_URL,
            concurrency=256,
            decoding=_NX_DECODING | {"chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 4096},
            params={"prompt_scale": "composed", "representation": "template"},
            notes="reasoning on: the accuracy ceiling, at completion-token cost",
        ),
        Variant(
            "gpt-oss-20b:single_task_v2",
            model="gpt-oss-20b",
            base_url=BASE_URL,
            concurrency=200,
            decoding=_GPT_OSS_DECODING,
            params={"prompt_scale": "single_task", "representation": "json_patch", "prompt_rev": "v2"},
            notes="the incumbent, running the production queue, re-measured on this machine for a same-day comparison",
        ),
        Variant(
            "gpt-oss-20b:single_task_v2_record_first",
            model="gpt-oss-20b",
            base_url=BASE_URL,
            concurrency=200,
            decoding=_GPT_OSS_DECODING,
            params={
                "prompt_scale": "single_task",
                "representation": "json_patch",
                "prompt_rev": "v2",
                "prompt_order": "record_first",
            },
            notes="same contract, record before task, so a record's three "
            "requests share a cached prefill; adopted into production "
            "2026-08-01 as utils.patches.PROMPT_ORDER",
        ),
    ]
    + [
        # Liquid Nanos with every economy this trial measured
        Variant(
            f"{model.lower()}:{scale}",
            model=model,
            base_url=BASE_URL,
            concurrency=256,
            decoding={"temperature": 0.0, "max_tokens": 512, "extra_body": {"repetition_penalty": 1.05}},
            params={
                "prompt_scale": scale,
                "representation": "schema_prompt",
                "schema_in": "system" if scale == "composed" else "user_suffix",
            },
            notes="extraction-tuned Nano"
            if "Extract" in model
            else "general instruct control: does the Extract tune pay?",
        )
        for model in ("LFM2-1.2B-Extract", "LFM2-350M-Extract", "LFM2.5-350M")
        for scale in ("composed", "single_task")
    ]
}

SCHEMA_PROMPT_MODELS = {"LFM2-1.2B-Extract", "LFM2-350M-Extract", "LFM2.5-350M"}

RUNNERS = {
    "gpt-oss-20b:single_task_v2": predict_baseline,
    "gpt-oss-20b:single_task_v2_record_first": predict_record_first,
} | {v.name: predict_schema_prompt for v in VARIANTS.values() if v.model in SCHEMA_PROMPT_MODELS}


def main() -> None:
    ap = argparse.ArgumentParser(description="Is a task-specific 4B extraction model cheaper than the incumbent?")
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--limit", type=int, help="first N gold items (smoke test)")
    ap.add_argument(
        "--bench",
        nargs="?",
        type=int,
        const=BENCH_RECORDS,
        help=f"throughput only, over N unconsumed production records (default {BENCH_RECORDS:,}); no gold, no scoring",
    )
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    if args.variant:
        variant = VARIANTS[args.variant]
        t0 = time.time()
        if args.bench:
            asyncio.run(bench(variant, args.bench))
        else:
            asyncio.run(RUNNERS.get(variant.name, predict_nuextract)(variant, args.limit))
        log(f"{variant.name} in {time.time() - t0:.0f}s")
    if args.score or not args.variant:
        with pl.Config(tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=30):
            print(score())


if __name__ == "__main__":
    main()
