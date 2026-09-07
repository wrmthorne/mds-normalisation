from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

import polars as pl

from mds_norm.paths import (
    DATA,
    EMISSIONS_LOG,
    FIELD_STATS,
    GOLD_FRAMES,
    PROBE_CANDIDATES,
    RAW_RECORDS,
    RECORD_FIXES_OUT,
    ROOT,
)
from mds_norm.utils.patches import (
    DEST,
    LLM_CONFIDENCE,
    MIN_TEXT_CHARS,
    PROMPT_ORDER,
    RECORD_FIRST,
    SYSTEM,
    SYSTEM_V2,
    TARGET_PRESENCE,
    TASK_FIRST,
    TASK_OF_FIELD,
    TASK_TEXT,
    TASK_TEXT_V2,
    build_request,
    parse_ops,
    render_prompt,
    validate_op,
)

OUT = DATA / "extraction"
QUEUE = OUT / "queue.parquet"
ROWS = OUT / "record_rows.parquet"
SHARDS = OUT / "shards"
PROGRESS = OUT / "progress.jsonl"
LLM_OPS = OUT / "llm_ops.parquet"
# Replies keyed by record content, so re-runs pay for changes
RESPONSE_CACHE = OUT / "response_cache.parquet"

RF_OUT = RECORD_FIXES_OUT
PATCHES = RF_OUT / "record_patches.parquet"
REPORT = RF_OUT / "extraction_report.json"
REVIEW = RF_OUT / "record_fixes_review_sample.csv"

COMPONENT = "record_fixes"
TIER = 5

# Measured configuration; `max_tokens` only truncates a runaway generation
MODEL = "gpt-oss-20b"
BASE_URL = "http://localhost:30000/v1"
DECODING = {"temperature": 0.2, "top_p": 1.0, "reasoning_effort": "low", "max_tokens": 1024}
CONCURRENCY = 200
TIMEOUT = 600.0
RETRIES = 2

SHARD_RECORDS = 10_000  # ~18k requests, ~10 min of endpoint time per shard
RELOC_HITS = 5  # relocation hits offered per record
MAX_ERROR_RATE = 0.5  # above this a shard is a failure, not a result

# Suffix-only caps, so a recovered span keeps its original offset
MAX_VALUE_CHARS = 8_000
MAX_PROMPT_CHARS = 48_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


@contextmanager
def measure(project: str, energy_url: str | None = None) -> Iterator[dict]:
    """Codecarbon around one shard, logging to the production emissions log"""
    from codecarbon import EmissionsTracker

    from mds_norm.utils.energy import RemoteMeter

    EMISSIONS_LOG.mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(project_name=project, output_dir=str(EMISSIONS_LOG), log_level="error")
    remote = RemoteMeter(energy_url) if energy_url else None
    cost: dict = {}
    tracker.start()
    if remote:
        remote.start(project)
    try:
        yield cost
    finally:
        emissions_kg = tracker.stop() or 0.0
        data = tracker.final_emissions_data
        cost["energy_wh"] = round(data.energy_consumed * 1e3, 4)
        cost["co2_g"] = round(emissions_kg * 1e3, 4)
        cost["duration_s"] = round(data.duration, 1)
        if remote:
            served = remote.stop()
            cost["energy_wh_client"] = cost["energy_wh"]
            cost["co2_g_client"] = cost["co2_g"]
            cost["energy_wh_server"] = served["energy_wh"]
            cost["gpu_energy_wh_server"] = served["gpu_energy_wh"]
            cost["co2_g_server"] = served["co2_g"]
            cost["session_wh_server"] = served["session"]["energy_wh"]
            cost["energy_wh"] = round(cost["energy_wh"] + served["energy_wh"], 4)
            cost["co2_g"] = round(cost["co2_g"] + served["co2_g"], 4)


def probe_ops() -> tuple[pl.DataFrame, pl.DataFrame]:
    """The probe scan's verified spans as mechanical patch ops, plus the relocation queue"""
    from mds_data_model.introspection import free_text_fields

    def norm(c: pl.Expr) -> pl.Expr:
        return c.str.to_lowercase().str.replace_all(r"\s+", "")

    pc = (
        pl.scan_parquet(PROBE_CANDIDATES)
        .filter(pl.col("status").is_in(["refine", "novel", "additional"]) & ~pl.col("suspect_recent").fill_null(False))
        .join(pl.scan_parquet(FIELD_STATS).select("node_id", "value"), on="node_id")
        .with_columns(span_start=pl.col("value").str.find(pl.col("candidate"), literal=True))
        .drop_nulls("span_start")
        .with_columns(
            span_end=pl.col("span_start") + pl.col("candidate").str.len_chars(),
            is_text=pl.col("field_type").is_in(free_text_fields()),
            whole_cell=norm(pl.col("candidate")) == norm(pl.col("value")),
            field=DEST,
            data_source=pl.col("data_source").cast(pl.String),
        )
        .drop("value")
        .collect(engine="streaming")
    )

    mechanical = pc.filter(pl.col("is_text") | pl.col("whole_cell")).select(
        "record_id",
        "data_source",
        "node_id",
        source_field=pl.col("field_type"),
        op=pl.when(pl.col("is_text")).then(pl.lit("add")).otherwise(pl.lit("move")),
        field=pl.col("field"),
        value=pl.col("candidate"),
        span_start=pl.col("span_start").cast(pl.UInt32),
        span_end=pl.col("span_end").cast(pl.UInt32),
        task=pl.lit("probe_") + pl.col("status"),
        status=pl.when(pl.col("status") == "additional").then(pl.lit("flagged")).otherwise(pl.lit("resolved")),
        confidence=pl.col("status").replace_strict(
            {"refine": 0.9, "novel": 0.8, "additional": None}, return_dtype=pl.Float64
        ),
        sub_component=pl.lit("mechanical"),
    )

    reloc = (
        pc.filter(~pl.col("is_text") & ~pl.col("whole_cell"))
        .group_by("record_id")
        .agg(hits=pl.struct("node_id", "field_type", "candidate", "field").head(RELOC_HITS))
    )
    return mechanical, reloc


def record_targets() -> pl.DataFrame:
    """Every record's free-text mass and which extraction targets it already fills, before admission"""
    from mds_data_model.introspection import free_text_fields

    text = pl.col("field_type").is_in(free_text_fields())

    def presence(k: str) -> pl.Expr:
        return pl.col("field_type").is_in(TARGET_PRESENCE[k]).any()

    return (
        pl.scan_parquet(FIELD_STATS)
        .group_by("record_id")
        .agg(
            data_source=pl.col("data_source").first().cast(pl.String),
            text_chars=pl.col("char_count").filter(text).sum(),
            has_date=presence("date"),
            has_material=presence("material"),
            has_dimension=presence("dimension"),
        )
        .with_columns(
            n_missing=pl.sum_horizontal(~pl.col("has_date"), ~pl.col("has_material"), ~pl.col("has_dimension"))
        )
        .collect(engine="streaming")
    )


def admission_queue() -> pl.DataFrame:
    """Records with free-text mass whose extraction targets are empty"""
    return record_targets().filter((pl.col("text_chars") >= MIN_TEXT_CHARS) & (pl.col("n_missing") > 0))


EXTRACTION_FRAME = GOLD_FRAMES / "extraction.parquet"


def build_queue(shard_records: int, frame_only: bool = False) -> pl.DataFrame:
    """Extraction + relocation admissions, ordered by work per record (tasks, then text mass) and cut into shards"""
    extract_q = admission_queue()
    log(f"extraction admission: {extract_q.height:,} records")
    if frame_only:
        frame = pl.read_parquet(EXTRACTION_FRAME, columns=["record_id"]).unique()
        extract_q = extract_q.join(frame, on="record_id", how="semi")
        log(f"restricted to the frozen frame: {extract_q.height:,} records")
    _, reloc_q = probe_ops()
    log(f"relocation admission: {reloc_q.height:,} records")

    queue = (
        extract_q.join(reloc_q, on="record_id", how="full", coalesce=True)
        .with_columns(
            n_missing=pl.col("n_missing").fill_null(0),
            text_chars=pl.col("text_chars").fill_null(0),
            n_hits=pl.col("hits").list.len().fill_null(0),
            has_date=pl.col("has_date").fill_null(True),
            has_material=pl.col("has_material").fill_null(True),
            has_dimension=pl.col("has_dimension").fill_null(True),
        )
        .with_columns(n_tasks=pl.col("n_missing") + pl.col("n_hits"))
        .sort("n_tasks", "text_chars", descending=True)
        .with_row_index("priority")
        .with_columns(shard=(pl.col("priority") // shard_records).cast(pl.UInt32))
    )

    # data_source is only null for a relocation-only record
    if queue["data_source"].null_count():
        src = (
            pl.scan_parquet(PROBE_CANDIDATES)
            .select("record_id", ds=pl.col("data_source").cast(pl.String))
            .unique(subset="record_id")
            .collect(engine="streaming")
        )
        queue = (
            queue.join(src, on="record_id", how="left")
            .with_columns(data_source=pl.col("data_source").fill_null(pl.col("ds")))
            .drop("ds")
        )
    return queue


def materialise_rows(queue: pl.DataFrame) -> int:
    """Flat rows with their values tagged with the shard that will consume them"""
    keys = queue.lazy().select("record_id", "shard")
    values = (
        pl.scan_parquet(FIELD_STATS)
        .select("record_id", "node_id", "value")
        .join(keys.select("record_id"), on="record_id", how="semi")
        .select("node_id", "value")
    )
    rows = (
        pl.scan_parquet(RAW_RECORDS)
        .filter(pl.col("field_type").str.starts_with("spectrum/"))
        .select("record_id", pl.col("data_source").cast(pl.String), "node_id", "parent_id", "depth", "field_type")
        .join(keys, on="record_id", how="inner")
        .join(values, on="node_id", how="left")
    )
    # Sort so a record gets the same prompt every prepare
    rows.sort("record_id", "depth", "field_type", "value").sink_parquet(ROWS)
    return pl.scan_parquet(ROWS).select(pl.len()).collect().item()


def _truncate(rows: list[dict]) -> bool:
    """Cap each value at MAX_VALUE_CHARS in place"""
    cut = False
    for r in rows:
        v = r["value"]
        if v is not None and len(v) > MAX_VALUE_CHARS:
            r["value"] = v[:MAX_VALUE_CHARS]
            cut = True
    return cut


def shard_requests(
    shard: int, queue: pl.DataFrame, contract: str, order: str = PROMPT_ORDER
) -> tuple[list[dict], dict]:
    """The requests for one shard: one per admitted record, carrying its task list"""
    task_text = TASK_TEXT_V2 if contract == "v2" else TASK_TEXT
    work = queue.filter(pl.col("shard") == shard)
    rows = pl.scan_parquet(ROWS).filter(pl.col("shard") == shard).drop("shard").collect(engine="streaming")
    by_record = {
        (k if isinstance(k, str) else k[0]): sub.to_dicts()
        for k, sub in rows.partition_by("record_id", as_dict=True).items()
    }

    stats = {"records": work.height, "missing_rows": 0, "value_truncated": 0}
    requests = []
    for r in work.iter_rows(named=True):
        record_rows = by_record.get(r["record_id"])
        if not record_rows:
            stats["missing_rows"] += 1
            continue
        stats["value_truncated"] += _truncate(record_rows)
        missing = [k for k in ("date", "material", "dimension") if not r[f"has_{k}"]]
        req = build_request(r["record_id"], record_rows, missing, r["hits"] or [], task_text=task_text, order=order)
        if req is not None:
            requests.append(req)
    return requests, stats


def units_with_tasks(req: dict, scale: str, order: str = PROMPT_ORDER) -> list[tuple[str, str, str]]:
    """(unit, task, prompt) triples for one request"""
    if scale == "composed":
        return [("composed", "\n".join(req["tasks"]), req["prompt"])]
    return [
        (unit, task, render_prompt([task], req["record_json"], order))
        for unit, task in zip(req["units"], req["tasks"], strict=True)
    ]


def units_of(req: dict, scale: str, order: str = PROMPT_ORDER) -> list[tuple[str, str]]:
    """(unit, prompt) pairs for one request"""
    return [(unit, prompt) for unit, _, prompt in units_with_tasks(req, scale, order)]


def _clip(prompt: str) -> str:
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt
    return prompt[:MAX_PROMPT_CHARS] + "\n… (record truncated)"


OPS_SCHEMA = {
    "record_id": pl.String,
    "data_source": pl.String,
    "node_id": pl.Binary,
    "source_field": pl.String,
    "op": pl.String,
    "field": pl.String,
    "value": pl.String,
    "span_start": pl.UInt32,
    "span_end": pl.UInt32,
    "task": pl.String,
    "status": pl.String,
    "confidence": pl.Float64,
    "sub_component": pl.String,
    "rationale": pl.String,
    "reason": pl.String,
    "unit": pl.String,
    "shard": pl.UInt32,
}
RESPONSE_SCHEMA = {
    "record_id": pl.String,
    "unit": pl.String,
    "content": pl.String,
    "prompt_tokens": pl.Int64,
    "completion_tokens": pl.Int64,
    "error": pl.String,
    "prompt_chars": pl.UInt32,
    "shard": pl.UInt32,
    "content_sha": pl.String,
}
CACHE_SCHEMA = {
    "content_sha": pl.String,
    "unit": pl.String,
    "content": pl.String,
    "prompt_tokens": pl.Int64,
    "completion_tokens": pl.Int64,
}


type Json = dict[str, "Json"] | list["Json"] | str | int | float | bool | None


def _canonical(node: Json) -> Json:
    """A record's content with every ordering choice removed"""
    if isinstance(node, dict):
        return {k: _canonical(v) for k, v in sorted(node.items())}
    if isinstance(node, list):
        return sorted((_canonical(v) for v in node), key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))
    return node


def content_sha(record_json: dict, task: str, contract: str, order: str) -> str:
    """What a record asks of the model, independent of the order its rows happened to arrive in"""
    payload = json.dumps([_canonical(record_json), task, contract, order], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@cache
def response_cache() -> dict[tuple[str, str], dict]:
    """(content hash, unit) -> the reply that content already earned"""
    if not RESPONSE_CACHE.exists():
        return {}
    stored = pl.read_parquet(RESPONSE_CACHE)
    return {
        (r["content_sha"], r["unit"]): {
            "content": r["content"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "error": None,
        }
        for r in stored.iter_rows(named=True)
    }


def gate(units: list[tuple[dict, str]], replies: list[dict], shard: int) -> pl.DataFrame:
    """Every proposal through the unchanged validation gate"""
    rows = []
    for (req, unit), reply in zip(units, replies, strict=True):
        base = {
            "record_id": req["record_id"],
            "data_source": req["data_source"],
            "sub_component": "llm",
            "unit": unit,
            "shard": shard,
        }
        ops = parse_ops(reply["content"])
        if ops is None:
            rows.append(
                base | {"status": "deferred", "reason": "llm_error" if reply["error"] else "unparseable_response"}
            )
            continue
        for op in ops:
            proposed = op if isinstance(op, dict) else {}
            rationale = proposed.get("rationale")
            parsed, reason = validate_op(req, op)
            if parsed is None:
                rows.append(
                    base
                    | {k: v for k, v in proposed.items() if k in ("op", "field", "value") and isinstance(v, str)}
                    | {
                        "status": "rejected",
                        "reason": reason,
                        "rationale": rationale if isinstance(rationale, str) else None,
                    }
                )
            else:
                task = "relocate" if parsed["op"] == "move" else TASK_OF_FIELD.get(parsed["field"], "relocate")
                rows.append(
                    base
                    | parsed
                    | {
                        "task": task,
                        "status": "resolved",
                        "confidence": LLM_CONFIDENCE,
                        "rationale": rationale if isinstance(rationale, str) else None,
                    }
                )
    ops_df = pl.from_dicts(rows, schema=OPS_SCHEMA) if rows else pl.DataFrame(schema=OPS_SCHEMA)
    return dedupe_resolved(ops_df)


OP_KEY = ["record_id", "node_id", "op", "field", "value", "span_start"]


def dedupe_resolved(ops: pl.DataFrame) -> pl.DataFrame:
    """One applied op is enough. Rejections and deferrals are not deduplicated"""
    indexed = ops.with_row_index("_i")
    resolved = pl.col("status") == "resolved"
    return (
        pl.concat([indexed.filter(resolved).unique(subset=OP_KEY, keep="first"), indexed.filter(~resolved)])
        .sort("_i")
        .drop("_i")
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    queue = build_queue(args.shard_records, frame_only=args.frame)
    queue.write_parquet(QUEUE)
    n_shards = int(queue["shard"].max()) + 1 if queue.height else 0
    requests = int(queue["n_tasks"].sum())
    log(f"queue: {queue.height:,} records / {requests:,} single-task requests in {n_shards} shards → {QUEUE}")

    n_rows = materialise_rows(queue)
    log(f"record rows: {n_rows:,} → {ROWS} ({ROWS.stat().st_size / 1e9:.1f} GB, {time.time() - t0:.0f}s total)")
    log(f"next: uv run python -m {__spec__.name} run")


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}


def _is_local(base_url: str) -> bool:
    from urllib.parse import urlsplit

    return (urlsplit(base_url).hostname or "") in LOCAL_HOSTS


def _energy_ready(args: argparse.Namespace) -> None:
    """Check whether remote energy tracker is ready"""
    from mds_norm.utils.energy import RemoteMeter

    if args.energy_url:
        health = RemoteMeter(args.energy_url).health()
        hw = ", ".join(health["hardware"])
        if "GPU" not in health["hardware"]:
            raise SystemExit(
                f"the energy agent at {args.energy_url} sees no GPU "
                f"({hw}) — it would attribute none of the model's draw"
            )
        log(f"energy agent at {args.energy_url}: {hw} in {health['country']}")
    elif not _is_local(args.base_url) and not args.allow_unmetered:
        raise SystemExit(
            f"{args.base_url} is served from another machine, but codecarbon only "
            "measures this one. Start the agent there —\n"
            "    python -m mds_norm.utils.energy serve --country-iso-code GBR\n"
            "— and pass --energy-url http://<host>:8770, or --allow-unmetered to "
            "run with client-side energy only (the journal records which)."
        )


def _endpoint_ready(base_url: str, model: str) -> bool:
    import httpx

    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=10.0)
        r.raise_for_status()
        served = [m["id"] for m in r.json().get("data", [])]
    except Exception as exc:
        log(f"endpoint {base_url} unreachable: {exc!r}")
        return False
    if model not in served:
        log(f"endpoint serves {served}, not {model!r}")
        return False
    return True


def done_shards() -> set[int]:
    if not PROGRESS.exists():
        return set()
    return {json.loads(line)["shard"] for line in PROGRESS.read_text(encoding="utf-8").splitlines() if line}


def journal(entry: dict) -> None:
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def run_shard(shard: int, queue: pl.DataFrame, args: argparse.Namespace) -> dict:
    from mds_norm.utils.inference import Inference

    requests, stats = shard_requests(shard, queue, args.contract, args.order)
    # grouped by record to ensure cache is reused between samples
    work = [
        (req, unit, task, prompt)
        for req in requests
        for unit, task, prompt in units_with_tasks(req, args.scale, args.order)
    ]
    if args.limit:
        work = work[: args.limit]
    units = [(req, unit) for req, unit, _, _ in work]
    clipped = sum(len(p) > MAX_PROMPT_CHARS for *_, p in work)
    prompts = [_clip(p) for *_, p in work]

    if args.dry_run:
        chars = sum(len(p) for p in prompts)
        log(
            f"shard {shard}: {len(requests):,} records, {len(prompts):,} requests, "
            f"{chars / 4 / 1e6:.2f}M prompt tokens (est), {clipped} clipped"
        )
        return {}

    shas = [content_sha(req["record_json"], task, args.contract, args.order) for req, _, task, _ in work]
    cached = response_cache() if args.reuse else {}
    replies: list[dict | None] = [cached.get((s, u)) for s, (_, u) in zip(shas, units, strict=True)]
    todo = [i for i, reply in enumerate(replies) if reply is None]
    if cached:
        log(f"shard {shard}: {len(prompts) - len(todo):,} of {len(prompts):,} prompts answered from the cache")

    inf = Inference(
        model=args.model, base_url=args.base_url, concurrency=args.concurrency, timeout=TIMEOUT, retries=RETRIES
    )
    system = SYSTEM_V2 if args.contract == "v2" else SYSTEM
    with measure("record_fixes_llm", args.energy_url) as cost:
        fresh = (
            await inf.generate(
                [prompts[i] for i in todo], system=system, usage=True, progress=not args.quiet, **DECODING
            )
            if todo
            else []
        )
    for i, reply in zip(todo, fresh, strict=True):
        replies[i] = reply

    ops = gate(units, replies, shard)
    responses = pl.from_dicts(
        [
            {
                "record_id": req["record_id"],
                "unit": unit,
                "shard": shard,
                "prompt_chars": len(prompt),
                "content_sha": sha,
                **reply,
            }
            for (req, unit), prompt, sha, reply in zip(units, prompts, shas, replies, strict=True)
        ],
        schema=RESPONSE_SCHEMA,
    )
    SHARDS.mkdir(parents=True, exist_ok=True)
    ops.write_parquet(SHARDS / f"ops_{shard:04d}.parquet")
    responses.write_parquet(SHARDS / f"responses_{shard:04d}.parquet")

    resolved = ops.filter(pl.col("status") == "resolved").height
    errors = int(responses["error"].is_not_null().sum())
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shard": shard,
        "git": git_sha(),
        "model": args.model,
        "scale": args.scale,
        "contract": args.contract,
        "order": args.order,
        "decoding": DECODING,
        "concurrency": args.concurrency,
        "base_url": args.base_url,
        "energy_url": args.energy_url,
        **stats,
        "requests": len(prompts),
        "requests_called": len(todo),
        "requests_reused": len(prompts) - len(todo),
        "prompts_clipped": clipped,
        "errors": errors,
        # tokens the endpoint served, so costs track the work
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in fresh),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in fresh),
        "resolved_ops": resolved,
        "rejected_ops": ops.filter(pl.col("status") == "rejected").height,
        "deferred": ops.filter(pl.col("status") == "deferred").height,
        **cost,
    }
    if errors > len(todo) * MAX_ERROR_RATE:
        raise SystemExit(
            f"shard {shard}: {errors:,} of {len(todo):,} calls failed — not "
            "journalled. Check the endpoint, then re-run `run` to retry this shard."
        )

    journal(entry)
    log(
        f"shard {shard}: {entry['requests']:,} requests ({len(todo):,} called) → {resolved:,} accepted ops "
        f"({errors} errors) in {cost['duration_s']:.0f}s, "
        f"{cost['energy_wh']:.1f} Wh"
    )
    return entry


def cmd_run(args: argparse.Namespace) -> None:
    import asyncio

    if not QUEUE.exists():
        raise SystemExit(f"no queue at {QUEUE} — run `prepare` first")
    queue = pl.read_parquet(QUEUE)
    all_shards = sorted(queue["shard"].unique().to_list())
    if args.shards:
        wanted = set(_parse_shards(args.shards))
        all_shards = [s for s in all_shards if s in wanted]
    pending = [s for s in all_shards if s not in done_shards()] if not args.redo else all_shards
    if not pending:
        log("nothing pending — all requested shards are journalled as done")
        return
    log(f"{len(pending)} shard(s) pending of {len(all_shards)}: {pending[0]}..{pending[-1]}")

    if not args.dry_run:
        if not _endpoint_ready(args.base_url, args.model):
            raise SystemExit("endpoint not ready — start the server, or pass --dry-run")
        _energy_ready(args)

    for n, shard in enumerate(pending, start=1):
        asyncio.run(run_shard(shard, queue, args))
        if args.max_shards and n >= args.max_shards:
            log(f"stopping after {n} shard(s) as asked; resume with `run`")
            break
    _summarise()


def _parse_shards(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _summarise() -> dict:
    if not PROGRESS.exists():
        return {}
    latest = {
        json.loads(line)["shard"]: json.loads(line)
        for line in PROGRESS.read_text(encoding="utf-8").splitlines()
        if line
    }
    runs = list(latest.values())
    keys = (
        "records",
        "requests",
        "resolved_ops",
        "rejected_ops",
        "deferred",
        "errors",
        "prompt_tokens",
        "completion_tokens",
    )
    total = {k: sum(r.get(k) or 0 for r in runs) for k in keys}
    total["shards"] = len({r["shard"] for r in runs})
    for k in ("energy_wh", "energy_wh_client", "energy_wh_server", "gpu_energy_wh_server", "duration_s"):
        total[k] = round(sum(r.get(k) or 0 for r in runs), 1)
    total["unmetered_shards"] = sum(
        not r.get("energy_url") and not _is_local(r.get("base_url") or BASE_URL) for r in runs
    )
    return total


def _requests_now(args: argparse.Namespace) -> pl.DataFrame:
    """Every (record, unit) the current queue would ask, with its content hash and prompt length"""
    queue = pl.read_parquet(QUEUE)
    rows = [
        {
            "record_id": req["record_id"],
            "unit": unit,
            "content_sha": content_sha(req["record_json"], task, args.contract, args.order),
            "chars_now": len(_clip(prompt)),
        }
        for shard in sorted(queue["shard"].unique().to_list())
        for req in shard_requests(shard, queue, args.contract, args.order)[0]
        for unit, task, prompt in units_with_tasks(req, args.scale, args.order)
    ]
    return pl.from_dicts(
        rows, schema={"record_id": pl.String, "unit": pl.String, "content_sha": pl.String, "chars_now": pl.UInt32}
    )


def cmd_cache(args: argparse.Namespace) -> None:
    files = sorted(SHARDS.glob("responses_*.parquet"))
    if not files:
        raise SystemExit(f"no shard responses under {SHARDS}")

    hashed, legacy = [], []
    for path in files:
        frame = pl.read_parquet(path)
        (hashed if "content_sha" in frame.columns else legacy).append(frame)
    if legacy and not args.from_snapshot:
        raise SystemExit(
            f"{len(legacy)} shard file(s) predate the content hash, and the rows they were rendered from "
            "are gone. They can only be re-keyed against the current queue, which assumes the corpus has "
            "not moved under them — pass --from-snapshot to accept that."
        )
    if legacy:
        old = pl.concat(legacy, how="diagonal_relaxed").filter(
            pl.col("error").is_null() & pl.col("content").is_not_null()
        )
        log(f"re-keying {old.height:,} replies from {len(legacy)} shard file(s) against the current queue")
        now = _requests_now(args)
        # Repeated unit labels are ambiguous; re-key only single occurrences
        old_once = old.filter(pl.len().over("record_id", "unit") == 1)
        now_once = now.filter(pl.len().over("record_id", "unit") == 1)
        rekeyed = old_once.join(now_once, on=["record_id", "unit"], how="inner")
        # Prompt length is the evidence the content is unchanged
        kept = rekeyed.filter(pl.col("prompt_chars") == pl.col("chars_now"))
        log(
            f"  {old.height - old_once.height:,} replies sit under a repeated unit label and cannot be re-keyed; "
            f"{old_once.height - rekeyed.height:,} are no longer asked"
        )
        log(f"  {kept.height:,} of {rekeyed.height:,} still render the same prompt length")
        hashed.append(kept.drop("chars_now"))

    stored = pl.concat(hashed, how="diagonal_relaxed")
    cached = (
        stored.filter(pl.col("error").is_null() & pl.col("content").is_not_null())
        .unique(subset=["content_sha", "unit"], keep="first")
        .select(list(CACHE_SCHEMA))
    )
    cached.write_parquet(RESPONSE_CACHE)
    log(f"{cached.height:,} replies of {stored.height:,} stored → {RESPONSE_CACHE}")


def cmd_status(_args: argparse.Namespace) -> None:
    if not QUEUE.exists():
        raise SystemExit(f"no queue at {QUEUE} — run `prepare` first")
    queue = pl.read_parquet(QUEUE)
    n_shards = int(queue["shard"].max()) + 1
    done = done_shards()
    total = _summarise()
    if not total:
        log(
            f"queue ready: {queue.height:,} records / {int(queue['n_tasks'].sum()):,} "
            f"requests in {n_shards} shards; nothing run yet"
        )
        return
    left = n_shards - len(done)
    per_shard_s = total["duration_s"] / max(total["shards"], 1)
    per_shard_wh = total["energy_wh"] / max(total["shards"], 1)
    log(
        f"{len(done)}/{n_shards} shards done — {total['records']:,} records, "
        f"{total['requests']:,} requests, {total['resolved_ops']:,} accepted ops, "
        f"{total['rejected_ops']:,} rejected, {total['deferred']:,} deferred, "
        f"{total['errors']:,} errors"
    )
    log(
        f"cost so far: {total['energy_wh'] / 1e3:.2f} kWh over "
        f"{total['duration_s'] / 3600:.1f} h "
        f"({total['prompt_tokens'] + total['completion_tokens']:,} tokens)"
    )
    if total["energy_wh_server"]:
        log(
            f"  of which {total['energy_wh_server'] / 1e3:.2f} kWh on the serving "
            f"host ({total['gpu_energy_wh_server'] / 1e3:.2f} kWh GPU), "
            f"{total['energy_wh_client'] / 1e3:.2f} kWh on this one"
        )
    if total["unmetered_shards"]:
        log(
            f"  WARNING: {total['unmetered_shards']} shard(s) ran against a remote "
            "endpoint with no energy agent — their energy is the client's only"
        )
    if left:
        log(f"remaining: {left} shards ≈ {left * per_shard_s / 3600:.1f} h, {left * per_shard_wh / 1e3:.2f} kWh")
    else:
        log(f"queue complete — next: uv run python -m {__spec__.name} finalise")


def cmd_finalise(args: argparse.Namespace) -> None:
    if not QUEUE.exists():
        raise SystemExit(f"no queue at {QUEUE} — run `prepare` first")
    queue = pl.read_parquet(QUEUE)
    n_shards = int(queue["shard"].max()) + 1
    done = done_shards()
    if len(done) < n_shards and not args.partial:
        raise SystemExit(
            f"only {len(done)}/{n_shards} shards are done — finish the run, or pass --partial to assemble what exists"
        )

    shard_files = sorted(SHARDS.glob("ops_*.parquet"))
    if not shard_files:
        raise SystemExit(f"no shard outputs under {SHARDS}")
    llm = dedupe_resolved(pl.concat([pl.read_parquet(p) for p in shard_files], how="vertical"))
    llm.write_parquet(LLM_OPS)
    log(f"{llm.height:,} LLM op rows from {len(shard_files)} shards → {LLM_OPS}")

    mechanical, _ = probe_ops()
    log(f"{mechanical.height:,} mechanical patch rows")

    patches = pl.concat([mechanical, llm.drop("unit", "shard")], how="diagonal_relaxed").with_columns(
        component=pl.lit(COMPONENT), tier=pl.lit(TIER, dtype=pl.Int32)
    )
    RF_OUT.mkdir(parents=True, exist_ok=True)
    backup = PATCHES.with_name("record_patches_phase1.parquet")
    if PATCHES.exists() and not backup.exists():
        PATCHES.rename(backup)
        log(f"previous sidecar kept as {backup.name}")
    patches.write_parquet(PATCHES)
    log(f"{patches.height:,} patch rows → {PATCHES}")

    accepted = llm.filter(pl.col("status") == "resolved")
    total = _summarise()
    tokens = total.get("prompt_tokens", 0) + total.get("completion_tokens", 0)
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "git": git_sha(),
        "configuration": {
            "model": args.model,
            "scale": args.scale,
            "contract": args.contract,
            "decoding": DECODING,
            "selected_by": "analysis_output/experiments/extraction_results.parquet",
        },
        "queue": {
            "records": queue.height,
            "requests": int(queue["n_tasks"].sum()),
            "shards": n_shards,
            "shards_done": len(done),
        },
        "run": total,
        "ops": {
            "llm_resolved": accepted.height,
            "llm_rejected": llm.filter(pl.col("status") == "rejected").height,
            "llm_deferred": llm.filter(pl.col("status") == "deferred").height,
            "records_with_accepted_op": accepted["record_id"].n_unique(),
            "by_task": {
                r["task"]: r["len"]
                for r in accepted.group_by("task").len().sort("len", descending=True).iter_rows(named=True)
            },
            "rejection_reasons": {
                r["reason"]: r["len"]
                for r in llm.filter(pl.col("status").is_in(["rejected", "deferred"]))
                .group_by("reason")
                .len()
                .sort("len", descending=True)
                .iter_rows(named=True)
            },
            "mechanical": {
                r["task"]: r["len"]
                for r in mechanical.group_by("task").len().sort("len", descending=True).iter_rows(named=True)
            },
        },
        "unit_cost": {
            "tokens_per_record": round(tokens / max(total.get("records", 0), 1), 1),
            "tokens_per_accepted_op": round(tokens / max(accepted.height, 1), 1),
            "wh_per_record": round(total.get("energy_wh", 0) / max(total.get("records", 0), 1), 5),
            "wh_per_accepted_op": round(total.get("energy_wh", 0) / max(accepted.height, 1), 5),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"report → {REPORT}")

    review_cols = [
        "record_id",
        "sub_component",
        "task",
        "op",
        "source_field",
        "field",
        "value",
        "rationale",
        "confidence",
    ]
    mech_ok = mechanical.filter(pl.col("status") == "resolved")
    review = pl.concat(
        [
            accepted.sample(min(args.review, accepted.height), seed=0),
            mech_ok.sample(min(args.review, mech_ok.height), seed=0),
        ],
        how="diagonal_relaxed",
    ).select(review_cols)
    review.write_csv(REVIEW)
    log(f"review sample ({review.height} rows) → {REVIEW}")
    log(
        "next: recompile — systemd-run --user --scope -p MemoryMax=48G "
        "-p MemorySwapMax=0 .venv/bin/python3 -m mds_norm.pipeline.compile_records"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM extraction over the full admission queue.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="build the queue and materialise its rows")
    p.add_argument("--shard-records", type=int, default=SHARD_RECORDS)
    p.add_argument(
        "--frame",
        action="store_true",
        help="restrict to the frozen sampling frame, the 1.02M-record queue the projection is stated over",
    )
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="run pending shards against the endpoint")
    p.add_argument("--shards", help="subset, e.g. 0-9 or 3,7 (default: all pending)")
    p.add_argument("--max-shards", type=int, help="stop after this many shards")
    p.add_argument("--limit", type=int, help="requests per shard (smoke test)")
    p.add_argument("--redo", action="store_true", help="re-run journalled shards")
    p.add_argument(
        "--reuse", action="store_true", help="take any prompt the response cache already holds an answer for"
    )
    p.add_argument("--dry-run", action="store_true", help="build prompts and report size; no endpoint calls")
    p.add_argument("--scale", choices=("single_task", "composed"), default="single_task")
    p.add_argument("--contract", choices=("v2", "v1"), default="v2")
    p.add_argument(
        "--order",
        choices=(RECORD_FIRST, TASK_FIRST),
        default=PROMPT_ORDER,
        help="prompt layout; record-first shares one prefill across a "
        "record's tasks (default, measured 2026-08-01). "
        "task-first is the earlier layout",
    )
    p.add_argument("--model", default=MODEL)
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument(
        "--energy-url",
        default=None,
        help="energy agent on the serving host (utils/energy.py), e.g. "
        "http://gpu-box:8770 — required when --base-url is remote",
    )
    p.add_argument("--allow-unmetered", action="store_true", help="run a remote endpoint with client-side energy only")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    p.add_argument("--quiet", action="store_true", help="no progress bar")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("cache", help="key the shard replies by their prompt, so a re-run pays only for what changed")
    p.add_argument(
        "--from-snapshot",
        action="store_true",
        help="rebuild the prompts of shards recorded before the hash, from the queue and rows on disk",
    )
    p.add_argument("--scale", choices=("single_task", "composed"), default="single_task")
    p.add_argument("--contract", choices=("v2", "v1"), default="v2")
    p.add_argument("--order", choices=(RECORD_FIRST, TASK_FIRST), default=PROMPT_ORDER)
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("status", help="progress, cost so far, projected remainder")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("finalise", help="assemble the sidecar, report, review sample")
    p.add_argument("--partial", action="store_true", help="assemble from an unfinished run")
    p.add_argument("--review", type=int, default=40, help="review rows per source")
    p.add_argument("--scale", choices=("single_task", "composed"), default="single_task")
    p.add_argument("--contract", choices=("v2", "v1"), default="v2")
    p.add_argument("--model", default=MODEL)
    p.set_defaults(func=cmd_finalise)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
