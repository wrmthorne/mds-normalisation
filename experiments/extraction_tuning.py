from __future__ import annotations

import argparse
import asyncio
import json
import re
import time

import polars as pl

from experiments.extraction_common import load_requests
from experiments.harness import EXP_OUT, Variant, log, measure, sample_items, write_run
from experiments.nuextract_trial import PRED_SCHEMA, QUEUE_RECORDS, _tasks_of
from mds_norm.utils.patches import SYSTEM_V2, TASK_TEXT_V2, parse_ops, validate_op

EXP = "extraction_tuning"
BASE_URL = "http://localhost:30000/v1"
BENCH_SHARD = 4  # unconsumed, and not the shard the trial benched on

# The incumbent's measured production rate, not a re-derived estimate
PRODUCTION = {"wh_per_record": 0.00963, "records_per_s": 16.1, "source": "data/extraction/progress.jsonl"}


# Three rewritten task bullets; the system prompt is unchanged

TASK_TEXT_V3 = {
    "date": "- add object_production_date: the date this object itself was made"
    " or produced, if the record states one. Acquisition, donation and"
    " association dates are not it, and neither is the date of an"
    " original that this object reproduces. A date the record gives for"
    " the object — in a title, a description, a dating field, or marked"
    " on the object as made — is it. Copy the date with any uncertainty"
    " qualifier and any range attached to it, not a bare year lifted out"
    " of a longer date expression.",
    "material": "- add material: each material this object is made of, one op per"
    " material. Copy the whole material phrase the record uses, then"
    " remove only the words describing how this object looks —"
    " colour used descriptively, pattern, weave decoration, finish,"
    " condition — and words naming the object's form rather than its"
    " substance. Keep every remaining word: where the record names a"
    " specific material, do not shorten it to the general one it"
    " belongs to, and do not shorten a two-word substance name to"
    " its head noun. Techniques and processes, implements, colours"
    " alone, places, names of the object or its parts, and its"
    " contents are not materials unless the record states what they"
    " are made of.",
    "dimension": "- add dimension: this object's stated measurements, copied as"
    " written including units. Where a field holds several"
    " measurements together as one value, copy that whole value as"
    " a single op — do not split it into one op per measurement."
    " Where measurements are embedded in running prose, copy each"
    " labelled measurement separately, with its label. Where the"
    " record states the same measurement twice, copy the"
    " unit-bearing form. A scale ratio or a size designation is not"
    " a measurement.",
}

# The suppression sentence, ablated alone to attribute the date silence
_BAR = " Never lower the bar to produce an op."
SYSTEM_V3_NOBAR = SYSTEM_V2.replace(_BAR, "")
if SYSTEM_V3_NOBAR == SYSTEM_V2:
    raise ValueError("suppression sentence not found in SYSTEM_V2")


# Three spellings of the same record; the values are byte-identical


def render_json_indent(record: dict) -> str:
    return json.dumps(record, indent=1, ensure_ascii=False)


def render_json_compact(record: dict) -> str:
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False)


def render_flat(record: dict) -> str:
    from experiments.nuextract_trial import flatten_record

    return flatten_record(record)


RENDERERS = {"json_indent": render_json_indent, "json_compact": render_json_compact, "flat": render_flat}


def build_prompt(tasks: list[str], record: dict, renderer: str) -> str:
    """`render_prompt(..., RECORD_FIRST)` with the record spelling swappable"""
    return "Record:\n" + RENDERERS[renderer](record) + "\n\nExtract, from the record above:\n" + "\n".join(tasks)


def variant_prompts(variant: Variant, reqs: list[dict]) -> list[tuple[dict, str, str]]:
    """(request, unit, prompt) for one variant"""
    text = _text_for(variant.params.get("contract", "v2"))
    renderer = variant.params.get("renderer", "json_indent")
    mark = variant.params.get("freetext_note", False)
    units = []
    for req in reqs:
        tasks = _tasks_of(req)
        if not tasks:
            continue
        note = freetext_note(req["record_json"]) if mark else ""
        if variant.params.get("prompt_scale") == "composed":
            units.append(
                (req, "composed", build_prompt([text[k] for k in tasks], req["record_json"], renderer) + note)
            )
        else:
            units.extend((req, k, build_prompt([text[k]], req["record_json"], renderer) + note) for k in tasks)
    return units


def system_for(variant: Variant) -> str:
    return SYSTEM_V3_NOBAR if variant.params.get("system") == "nobar" else SYSTEM_V2


def freetext_note(record: dict) -> str:
    from mds_norm.utils.patches import FREE_TEXT, strip_prefix

    short = {strip_prefix(f) for f in FREE_TEXT}
    prose = [k for k in record if k in short]
    cells = [k for k in record if k not in short and k != "data_source"]
    if not prose or not cells:
        return ""
    return (
        "\nIn this record, these fields hold running prose: "
        + ", ".join(prose)
        + ". Every other field is a single stored cell."
    )


async def predict(variant: Variant, limit: int | None, tag: str = "") -> None:
    """One variant over the frozen extraction gold"""
    from mds_norm.utils.inference import Inference

    if tag:
        variant = Variant(
            variant.name + tag,
            model=variant.model,
            base_url=variant.base_url,
            concurrency=variant.concurrency,
            decoding=variant.decoding,
            params=variant.params,
            notes=variant.notes + f" (repeat {tag.lstrip('#')})",
        )
    items = sample_items("extraction", limit)
    reqs = load_requests(items)
    units = variant_prompts(variant, reqs)

    inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)
    with measure(EXP, variant.name) as cost:
        replies = await inf.generate([u[2] for u in units], system=system_for(variant), usage=True, **variant.decoding)

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

    preds = pl.from_dicts(rows, schema=PRED_SCHEMA) if rows else pl.DataFrame(schema=PRED_SCHEMA)
    accepted = preds.filter(pl.col("status") == "resolved").height
    prompt_tokens = sum(r["prompt_tokens"] or 0 for r in replies)
    completion_tokens = sum(r["completion_tokens"] or 0 for r in replies)
    duration = cost["duration_s"] or 1.0
    write_run(
        EXP,
        variant,
        preds,
        {
            "n_items": len(items),
            "n_requests": len(units),
            "n_errors": sum(1 for r in replies if r["error"]),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "accepted_ops": accepted,
            "tokens_per_accepted_op": ((prompt_tokens + completion_tokens) / accepted if accepted else None),
            **cost,
            "wh_per_record": cost["energy_wh"] / max(len(items), 1),
            "records_per_s": len(items) / duration,
            "requests_per_s": len(units) / duration,
        },
    )


# `parse_dimensions` proposes spans, so this scores like an LLM variant

# A labelled or bare measurement inside running prose
_UNIT_WORD = (
    r"mm|cm|m|in|ins|inch|inches|ft|foot|feet|g|kg|lb|lbs|oz"
    r"|[\"”″′']"
)
_LABEL = (
    r"(?:height|width|depth|length|diameter|diam|dia|thickness|breadth"
    r"|weight|circumference|radius|h|w|d|l)"
)
_NUM = r"\d+(?:\.\d+)?(?:\s*\d+/\d+)?|\d+/\d+"
_PROSE_SPAN = re.compile(
    rf"(?:{_LABEL})?\s*[:.]?\s*(?:{_NUM})\s*(?:{_UNIT_WORD})\b\.?"
    rf"(?:\s*[x×]\s*(?:{_NUM})\s*(?:{_UNIT_WORD})?\b)*",
    re.IGNORECASE,
)


# Fields that cannot hold a measurement, however well they parse
_NOT_MEASUREMENT = re.compile(r"number|identifier|date|_id$|reference|location|url|person|place|name$")


def _measured(text: str) -> bool:
    """Only spans carrying a unit, so a bare year is not read as a measurement"""
    from mds_norm.parsers.parse_dimensions import parse_dimensions

    r = parse_dimensions(text)
    return bool(r) and any(m["dimension_measurement_unit"] for m in r["measurements"])


def dimension_proposals(req: dict) -> list[dict]:
    """Deterministic `add` ops for the dimension task, one per span offered"""
    from mds_norm.utils.patches import FIELD_FOR_TASK, FREE_TEXT, strip_prefix

    target = strip_prefix(FIELD_FOR_TASK["dimension"])
    ops = []
    for key, nodes in req["nodes"].items():
        if _NOT_MEASUREMENT.search(key):
            continue
        for _node_id, field_type, value in nodes:
            if not value or not any(ch.isdigit() for ch in value):
                continue
            if field_type in FREE_TEXT:
                for m in _PROSE_SPAN.finditer(value):
                    span = m.group().strip(" .,;:")
                    if span and _measured(span):
                        ops.append({"op": "add", "field": target, "value": span, "source_field": key})
            elif _measured(value):
                ops.append({"op": "add", "field": target, "value": value, "source_field": key})
    return ops


def predict_dimension_parser(variant: Variant, limit: int | None) -> None:
    """The no-LLM dimension variant"""
    items = sample_items("extraction", limit)
    reqs = load_requests(items)

    rows = []
    with measure(EXP, variant.name) as cost:
        for req in reqs:
            if "dimension" not in _tasks_of(req):
                continue
            base = {"id": req["item_id"], "record_id": req["record_id"], "unit": "dimension"}
            for op in dimension_proposals(req):
                parsed, reason = validate_op(req, op)
                if parsed is None:
                    rows.append(
                        base | {k: op[k] for k in ("op", "field", "value")} | {"status": "rejected", "reason": reason}
                    )
                else:
                    parsed.pop("node_id", None)
                    rows.append(base | parsed | {"status": "resolved"})

    preds = pl.from_dicts(rows, schema=PRED_SCHEMA) if rows else pl.DataFrame(schema=PRED_SCHEMA)
    duration = cost["duration_s"] or 1.0
    write_run(
        EXP,
        variant,
        preds,
        {
            "n_items": len(items),
            "n_requests": 0,
            "n_errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "accepted_ops": preds.filter(pl.col("status") == "resolved").height,
            "tokens_per_accepted_op": None,
            **cost,
            "wh_per_record": cost["energy_wh"] / max(len(items), 1),
            "records_per_s": len(items) / duration,
            "requests_per_s": 0.0,
        },
    )


# Gold cannot fill the endpoint, so bench a real shard

BENCH_RECORDS = 2_000


def bench_requests(n: int) -> list[dict]:
    from mds_norm.pipeline.extraction import QUEUE, shard_requests

    queue = pl.read_parquet(QUEUE)
    reqs, _ = shard_requests(BENCH_SHARD, queue, "v2")
    return reqs[:n]


async def bench(variant: Variant, n: int) -> dict:
    from mds_norm.utils.inference import Inference

    reqs = bench_requests(n)
    units = variant_prompts(variant, reqs)
    inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)

    with measure(EXP, f"bench:{variant.name}") as cost:
        replies = await inf.generate([u[2] for u in units], system=system_for(variant), usage=True, **variant.decoding)

    duration = cost["duration_s"] or 1.0
    prompt_tokens = sum(r["prompt_tokens"] or 0 for r in replies)
    completion_tokens = sum(r["completion_tokens"] or 0 for r in replies)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variant": variant.name,
        "model": variant.model,
        "shard": BENCH_SHARD,
        "records": len(reqs),
        "requests": len(units),
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
    out = EXP_OUT / EXP / "bench.jsonl"
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


async def cache_probe(variant: Variant, n: int = 400) -> None:
    import httpx

    from mds_norm.utils.inference import Inference

    async def counters() -> dict[str, float]:
        async with httpx.AsyncClient(timeout=10.0) as c:
            body = (await c.get(variant.base_url.replace("/v1", "") + "/metrics")).text
        out = {}
        for line in body.splitlines():
            for k in (
                "vllm:prefix_cache_queries_total",
                "vllm:prefix_cache_hits_total",
                "vllm:gpu_prefix_cache_queries_total",
                "vllm:gpu_prefix_cache_hits_total",
            ):
                if line.startswith(k):
                    out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[-1])
        return out

    reqs = bench_requests(n)
    inf = Inference(model=variant.model, base_url=variant.base_url, concurrency=variant.concurrency, timeout=600.0)

    for label, scale in (("single_task (production)", "single_task"), ("composed", "composed")):
        v = Variant(
            variant.name,
            model=variant.model,
            base_url=variant.base_url,
            concurrency=variant.concurrency,
            decoding=variant.decoding,
            params=variant.params | {"prompt_scale": scale},
        )
        units = variant_prompts(v, reqs)
        before = await counters()
        t0 = time.time()
        await inf.generate([u[2] for u in units], system=system_for(v), usage=True, **v.decoding)
        dt = time.time() - t0
        after = await counters()
        q = sum(after.get(k, 0) - before.get(k, 0) for k in after if k.endswith("queries_total"))
        h = sum(after.get(k, 0) - before.get(k, 0) for k in after if k.endswith("hits_total"))
        log(
            f"{label}: {len(units)} requests over {len(reqs)} records in {dt:.1f}s "
            f"({len(reqs) / dt:.1f} rec/s) — prefix-cache blocks queried {q:,.0f}, "
            f"hit {h:,.0f} ({h / q if q else 0:.1%})"
        )


_DECODING = {"temperature": 0.2, "top_p": 1.0, "reasoning_effort": "low", "max_tokens": 1024}
_BASE = {"model": "gpt-oss-20b", "base_url": BASE_URL, "concurrency": 200, "decoding": _DECODING}
_SINGLE = {"prompt_scale": "single_task", "representation": "json_patch"}

VARIANTS = {
    v.name: v
    for v in [
        # The bar: the production configuration, re-run here
        Variant(
            "incumbent",
            **_BASE,
            params=_SINGLE | {"contract": "v2"},
            notes="gpt-oss-20b:single_task_v2_record_first as production runs it; "
            "byte-identical prompts, this experiment's output directory",
        ),
        # Accuracy: the contract, one change at a time
        Variant(
            "v3_date",
            **_BASE,
            params=_SINGLE | {"contract": "v3_date"},
            notes="v2 everywhere except the date bullet: does the rewritten date "
            "rule recover the 37 silent date misses?",
        ),
        Variant(
            "v3_material",
            **_BASE,
            params=_SINGLE | {"contract": "v3_material"},
            notes="v2 everywhere except the material bullet: keep-by-default, against the 20 too-short material spans",
        ),
        Variant(
            "v3_dimension",
            **_BASE,
            params=_SINGLE | {"contract": "v3_dimension"},
            notes="v2 everywhere except the dimension bullet: one op per "
            "measurement *cell*, against the 30 too-short spans",
        ),
        Variant("contract_v3", **_BASE, params=_SINGLE | {"contract": "v3"}, notes="all three rewritten bullets"),
        Variant(
            "contract_v3_nobar",
            **_BASE,
            params=_SINGLE | {"contract": "v3", "system": "nobar"},
            notes="v3 plus the 'never lower the bar' sentence ablated — isolates "
            "how much silence the suppression line is buying",
        ),
        Variant(
            "compact_json",
            **_BASE,
            params=_SINGLE | {"contract": "v2", "renderer": "json_compact"},
            notes="13.3% fewer record tokens, identical values; does dropping the "
            "indentation cost the model anything?",
        ),
        Variant(
            "flat_record",
            **_BASE,
            params=_SINGLE | {"contract": "v2", "renderer": "flat"},
            notes="11.1% fewer record tokens as 'field: value' lines",
        ),
        Variant(
            "composed_v2",
            **_BASE,
            params={"prompt_scale": "composed", "representation": "json_patch", "contract": "v2"},
            notes="one request per record instead of 1.84; the extraction variants rejected this under "
            "task-first (F1 0.531 vs 0.601) — re-asked under record-first",
        ),
        Variant(
            "composed_v3",
            **_BASE,
            params={"prompt_scale": "composed", "representation": "json_patch", "contract": "v3"},
            notes="the cheap scale and the rewritten contract together",
        ),
        Variant(
            "contract_v3_compact",
            **_BASE,
            params=_SINGLE | {"contract": "v3", "renderer": "json_compact"},
            notes="the candidate configuration: v3 rules, compact record",
        ),
        # Second round: only the date bullet earned its place
        Variant(
            "v3_date_nobar",
            **_BASE,
            params=_SINGLE | {"contract": "v3_date", "system": "nobar"},
            notes="v2 contract, v3 date bullet, suppression sentence ablated — the two silence levers together",
        ),
        Variant(
            "v3_date_flat",
            **_BASE,
            params=_SINGLE | {"contract": "v3_date", "renderer": "flat"},
            notes="the date win on the cheaper record spelling",
        ),
        # Greedy: the cheapest attempt at narrowing the noise floor
        Variant(
            "incumbent_greedy",
            **_BASE | {"decoding": _DECODING | {"temperature": 0.0}},
            params=_SINGLE | {"contract": "v2"},
            notes="the incumbent, greedy: how much of the spread is sampling?",
        ),
        Variant(
            "v3_date_greedy",
            **_BASE | {"decoding": _DECODING | {"temperature": 0.0}},
            params=_SINGLE | {"contract": "v3_date"},
            notes="the date win, greedy",
        ),
        # Third round, after the parser fixes raised the gate's recall
        Variant(
            "incumbent_pf",
            **_BASE,
            params=_SINGLE | {"contract": "v2"},
            notes="the incumbent again, against the repaired gate",
        ),
        Variant(
            "v3_dimension_pf",
            **_BASE,
            params=_SINGLE | {"contract": "v3_dimension"},
            notes="the whole-cell rule re-asked now that whole cells parse — the "
            "first round tested it while the gate was rejecting them",
        ),
        Variant(
            "v3_dim_freetext",
            **_BASE,
            params=_SINGLE | {"contract": "v3_dimension", "freetext_note": True},
            notes="whole-cell rule plus the gate's own free-text classification, "
            "so the model can tell a cell from prose",
        ),
        Variant(
            "dimension_parser",
            model=None,
            params={"prompt_scale": "none", "representation": "parser"},
            notes="no LLM: parse_dimensions proposes the spans it can read, on "
            "the gold's own convention. Does the model earn its place?",
        ),
    ]
}

# Single-bullet variants swap one v2 entry for its v3 rewrite
_ONE_BULLET = {"v3_date": "date", "v3_material": "material", "v3_dimension": "dimension"}


def _text_for(contract: str) -> dict[str, str]:
    if contract == "v3":
        return TASK_TEXT_V3
    if k := _ONE_BULLET.get(contract):
        return TASK_TEXT_V2 | {k: TASK_TEXT_V3[k]}
    return TASK_TEXT_V2


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Variants over the extraction contract clauses and the record rendering, with the model fixed."
    )
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--all", action="store_true", help="run every variant in order")
    ap.add_argument("--limit", type=int, help="first N gold items (smoke test)")
    ap.add_argument(
        "--bench",
        nargs="?",
        type=int,
        const=BENCH_RECORDS,
        help="throughput only, over N unconsumed production records",
    )
    ap.add_argument("--cache-probe", action="store_true", help="report vLLM's prefix-cache hit rate per request scale")
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run each variant N times as name#2, name#3, ... to measure the run-to-run spread at temperature 0.2",
    )
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    names = sorted(VARIANTS) if args.all else ([args.variant] if args.variant else [])
    for name in names:
        variant = VARIANTS[name]
        for rep in range(1, args.repeat + 1):
            t0 = time.time()
            if args.bench:
                asyncio.run(bench(variant, args.bench))
            elif variant.model is None:
                predict_dimension_parser(variant, args.limit)
            else:
                asyncio.run(predict(variant, args.limit, tag="" if rep == 1 else f"#{rep}"))
            log(f"{name}{'' if rep == 1 else f'#{rep}'} in {time.time() - t0:.0f}s")

    if args.cache_probe:
        asyncio.run(cache_probe(VARIANTS["incumbent"]))

    if args.score or not (names or args.cache_probe):
        from experiments.nuextract_trial import score

        with pl.Config(tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=30):
            print(score(exp=EXP))


if __name__ == "__main__":
    main()
