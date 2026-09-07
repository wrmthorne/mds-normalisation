from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import polars as pl

from mds_norm.paths import FIELD_STATS, PROBE_CANDIDATES, RAW_RECORDS
from mds_norm.utils.patches import SYSTEM_V2, TASK_TEXT_V2, build_request, render_prompt

if TYPE_CHECKING:
    from experiments.harness import Variant

RAW = RAW_RECORDS


TASK_KEYS = ("date", "material", "dimension")


def missing_of(item: dict) -> list[str]:
    return [k for k in TASK_KEYS if f"missing_{k}" in item["issues"]]


def load_requests(items: list[dict]) -> list[dict]:
    """One production-contract request per gold item, tagged with its item id"""
    ids = sorted({it["record_id"] for it in items})
    flat = (
        pl.scan_parquet(RAW)
        .filter(pl.col("record_id").is_in(ids) & pl.col("field_type").str.starts_with("spectrum/"))
        .select("record_id", pl.col("data_source").cast(pl.String), "node_id", "parent_id", "depth", "field_type")
        .join(pl.scan_parquet(FIELD_STATS).select("node_id", "value"), on="node_id", how="left")
        .collect(engine="streaming")
    )
    by_record = {
        (k if isinstance(k, str) else k[0]): sub for k, sub in flat.partition_by("record_id", as_dict=True).items()
    }
    reqs = []
    for it in items:
        rows = by_record.get(it["record_id"])
        if rows is None:
            continue
        req = build_request(it["record_id"], rows.to_dicts(), missing_of(it), [])
        if req is None:
            continue
        req["item_id"] = it["id"]
        reqs.append(req)
    return reqs


DIFF_SYSTEM = """\
You extract catalogue data from UK museum object records. You only locate and copy text that is \
already in the record; you never compose, rephrase, or normalise values.

Respond with one line per finding, in this exact format:
+ <target field>: <verbatim contiguous substring> (from <record key the value was copied from>)

Rules:
- The value must be copied character-for-character from the text of the named source key.
- Only use target fields named in the tasks.
- If a task finds nothing, emit no line for it; an empty response is valid.
- Output only these lines, no markdown, no commentary."""

FIELDS_SYSTEM = """\
You extract catalogue data from UK museum object records. You only locate and copy text that is \
already in the record; you never compose, rephrase, or normalise values.

Respond with a single JSON object mapping each target field to the list of verbatim values found, e.g.:
{"material": ["oak", "brass"], "object_production_date": ["1850"]}

Rules:
- Every value must be copied character-for-character from the record text.
- Only use target fields named in the tasks as keys.
- If a task finds nothing, omit its key; {} is valid.
- Output only the JSON object, no markdown."""

_DIFF_LINE = re.compile(r"^\+\s*([\w./ -]+?)\s*:\s*(.+?)\s*\(from\s+([^)]+)\)\s*$")
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_ops_diff(content: str | None) -> list | None:
    """`+ field: value (from key)` lines to op dicts, None when no line parses"""
    if content is None:
        return None
    ops, saw_plus = [], False
    for raw in content.splitlines():
        line = raw.strip()
        if not line.startswith("+"):
            continue
        saw_plus = True
        if m := _DIFF_LINE.match(line):
            ops.append({"op": "add", "field": m.group(1), "value": m.group(2), "source_field": m.group(3).strip()})
    if saw_plus and not ops:
        return None  # emitted findings but none in-format: unparseable
    return ops  # no "+" lines at all is a valid empty response


def parse_ops_fields(content: str | None) -> list | None:
    """`{"field": ["value", ...]}` to add-ops without source attribution"""
    if content is None or (m := _JSON_OBJ.search(content)) is None:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    ops = []
    for field, raw_vals in obj.items():
        vals = [raw_vals] if isinstance(raw_vals, str) else raw_vals
        if not isinstance(vals, list):
            continue
        ops.extend({"op": "add", "field": field, "value": v, "source_field": None} for v in vals if isinstance(v, str))
    return ops


def single_task_prompts(req: dict) -> list[tuple[str, str]]:
    """(task_bullet_key, prompt) pairs — one targeted request per composed task"""
    return [(task, render_prompt([task], req["record_json"])) for task in req["tasks"]]


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


async def predict(exp: str, variant: Variant, limit: int | None) -> None:
    """Run one variant over the frozen extraction sample"""
    from experiments.harness import measure, sample_items, write_run
    from mds_norm.utils.inference import Inference
    from mds_norm.utils.patches import SYSTEM, parse_ops, validate_op

    system = {"json_patch": SYSTEM, "diff": DIFF_SYSTEM, "fields_only": FIELDS_SYSTEM}
    parser = {"json_patch": parse_ops, "diff": parse_ops_diff, "fields_only": parse_ops_fields}
    rep = variant.params.get("representation", "json_patch")
    scale = variant.params.get("prompt_scale", "composed")
    rev = variant.params.get("prompt_rev")

    items = sample_items("extraction", limit)
    reqs = load_requests(items)
    if rev == "v2":
        from mds_norm.utils.patches import TASK_TEXT

        remap = {TASK_TEXT[k]: TASK_TEXT_V2[k] for k in TASK_TEXT}
        for req in reqs:
            req["tasks"] = [remap.get(t, t) for t in req["tasks"]]
            req["prompt"] = render_prompt(req["tasks"], req["record_json"])
        system = system | {"json_patch": SYSTEM_V2}
    units = (
        [(req, "composed", req["prompt"]) for req in reqs]
        if scale == "composed"
        else [(req, task, prompt) for req in reqs for task, prompt in single_task_prompts(req)]
    )

    inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)
    with measure(exp, variant.name) as cost:
        replies = await inf.generate([u[2] for u in units], system=system[rep], usage=True, **variant.decoding)

    rows = []
    for (req, unit, _), r in zip(units, replies, strict=True):
        base = {"id": req["item_id"], "record_id": req["record_id"], "unit": unit}
        ops = parser[rep](r["content"])
        if ops is None:
            rows.append(base | {"status": "deferred", "reason": "llm_error" if r["error"] else "unparseable_response"})
            continue
        for op in ops:
            parsed, reason = validate_op(req, op)
            if parsed is None:
                proposed = op if isinstance(op, dict) else {}
                clean = {
                    k: proposed.get(k)
                    for k in ("op", "field", "value", "source_field")
                    if isinstance(proposed.get(k), str)
                }
                rows.append(base | clean | {"status": "rejected", "reason": reason})
            else:
                parsed.pop("node_id", None)
                rows.append(base | parsed | {"status": "resolved"})

    preds = pl.from_dicts(rows, schema=PRED_SCHEMA) if rows else pl.DataFrame(schema=PRED_SCHEMA)
    accepted = preds.filter(pl.col("status") == "resolved").height
    tokens = sum(r["prompt_tokens"] or 0 for r in replies) + sum(r["completion_tokens"] or 0 for r in replies)
    metrics = {
        "n_items": len(items),
        "n_requests": len(units),
        "n_errors": sum(1 for r in replies if r["error"]),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in replies),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in replies),
        "accepted_ops": accepted,
        "tokens_per_accepted_op": tokens / accepted if accepted else None,
        **cost,
        "wh_per_record": cost["energy_wh"] / max(len(items), 1),
    }
    write_run(exp, variant, preds, metrics)


def predict_mechanical(exp: str, variant: Variant, limit: int | None) -> None:
    """The mechanical-op baseline: replay the probe-queue ops with no LLM"""
    from mds_data_model.introspection import free_text_fields

    from experiments.harness import measure, sample_items, write_run
    from mds_norm.utils.patches import DEST, FIELD_FOR_TASK

    items = sample_items("extraction", limit)
    allowed = {it["record_id"]: {FIELD_FOR_TASK[k] for k in missing_of(it)} for it in items}
    item_id = {it["record_id"]: it["id"] for it in items}

    def norm(c: pl.Expr) -> pl.Expr:
        return c.str.to_lowercase().str.replace_all(r"\s+", "")

    with measure(exp, variant.name) as cost:
        values = pl.scan_parquet(FIELD_STATS).select("node_id", "value")
        pc = (
            pl.scan_parquet(PROBE_CANDIDATES)
            .filter(
                pl.col("record_id").is_in(list(allowed))
                & pl.col("status").is_in(["refine", "novel"])
                & ~pl.col("suspect_recent").fill_null(False)
            )
            .join(values, on="node_id")
            .with_columns(span_start=pl.col("value").str.find(pl.col("candidate"), literal=True))
            .drop_nulls("span_start")
            .with_columns(
                span_end=(pl.col("span_start") + pl.col("candidate").str.len_chars()),
                is_text=pl.col("field_type").is_in(free_text_fields()),
                whole_cell=norm(pl.col("candidate")) == norm(pl.col("value")),
                field=DEST,
            )
            .filter(pl.col("is_text") | pl.col("whole_cell"))
            .collect(engine="streaming")
        )

    rows = [
        {
            "id": item_id[r["record_id"]],
            "record_id": r["record_id"],
            "unit": "mechanical",
            "op": "add" if r["is_text"] else "move",
            "field": r["field"],
            "value": r["candidate"],
            "source_field": r["field_type"],
            "span_start": r["span_start"],
            "span_end": r["span_end"],
            "status": "resolved",
            "reason": None,
        }
        for r in pc.iter_rows(named=True)
        if r["field"] in allowed[r["record_id"]]
    ]

    preds = pl.from_dicts(rows, schema=PRED_SCHEMA) if rows else pl.DataFrame(schema=PRED_SCHEMA)
    metrics = {
        "n_items": len(items),
        "n_requests": 0,
        "n_errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "accepted_ops": preds.height,
        "tokens_per_accepted_op": None,
        **cost,
        "wh_per_record": cost["energy_wh"] / max(len(items), 1),
    }
    write_run(exp, variant, preds, metrics)
