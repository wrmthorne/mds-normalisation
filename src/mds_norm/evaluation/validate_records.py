from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import polars as pl
from mds_data_model.models.object import Object
from pydantic import ValidationError

from mds_norm.paths import COMPILED, EVAL_OUT, RAW_RECORDS, ROOT
from mds_norm.utils.patches import reform, strip_prefix

VALIDATION_OUT = EVAL_OUT / "validation"
COMPILED_PATH = COMPILED / "mds-normalised.parquet"
COLUMNS = ["record_id", "data_source", "node_id", "parent_id", "depth", "source_array_pos", "field_type", "value"]
DEFAULT_SAMPLE = 100_000
BATCH_RECORDS = 500_000
"""Records held in memory at once; each batch costs one pass over the corpus"""
TOP_EXAMPLES = 5
KNOWN_FIELDS = frozenset(strip_prefix(field.alias or name) for name, field in Object.model_fields.items())

# pydantic reports a rejected union per member; drop the wrappers
SCALAR_TAGS = frozenset({"str", "int", "float", "bool", "none"})
WRAPPER_ERRORS = frozenset({"list_type", "model_type", "dict_type", "union_tag_invalid", "union_tag_not_found"})
EXAMPLE_CHARS = 120


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _location(loc: tuple) -> str:
    """The error's position as a record path, with pydantic's union tags removed"""
    parts = []
    for raw in loc:
        if isinstance(raw, int):
            parts.append(str(raw))
            continue
        part = strip_prefix(str(raw))
        if part not in SCALAR_TAGS and part.islower() and not set("[-") & set(part):
            parts.append(part)
    return ".".join(parts)


def _violation(field: str, err: dict) -> dict:
    """One violation from the error that best names it: its kind, where it sits, and what was there"""
    value = err.get("input")
    if err["type"] == "missing":
        return {"field": field, "error": "missing", "location": field, "value": None}
    if isinstance(value, dict | list):
        # every union member rejected a container: object or repeat
        kind = "multiple_values" if isinstance(value, list) else "unexpected_object"
        return {"field": field, "error": kind, "location": _location(err["loc"]), "value": repr(value)[:EXAMPLE_CHARS]}
    return {
        "field": field,
        "error": err["type"],
        "location": _location(err["loc"]),
        "value": repr(value)[:EXAMPLE_CHARS],
    }


def record_errors(rows: list[dict]) -> list[dict]:
    """One entry per schema violation in a record's nodes; empty when the record validates"""
    record = reform(rows)
    out = [
        {"field": field, "error": "unknown_field", "location": field, "value": None}
        for field in record
        if field not in KNOWN_FIELDS
    ]
    try:
        Object.model_validate(record)
    except ValidationError as exc:
        by_field: dict[str, list[dict]] = {}
        for err in exc.errors():
            by_field.setdefault(strip_prefix(str(err["loc"][0])), []).append(err)
        for field, errors in by_field.items():
            if required := [e for e in errors if e["type"] == "missing" and len(e["loc"]) == 1]:
                out.append(_violation(field, required[0]))
                continue
            leaves = [e for e in errors if not isinstance(e.get("input"), dict | list)] or [
                max(errors, key=lambda e: len(e["loc"]))
            ]
            best: dict[str, dict] = {}
            for err in leaves:
                key = repr(err.get("input"))[:EXAMPLE_CHARS]
                if key not in best or (best[key]["type"] in WRAPPER_ERRORS and err["type"] not in WRAPPER_ERRORS):
                    best[key] = err
            out += [_violation(field, e) for e in best.values()]
    return out


def check(rows: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Validate every record in `rows`; returns its violations and one verdict per record"""
    violations, verdicts = [], []
    for (record_id, data_source), group in rows.sort("depth", "source_array_pos", nulls_last=True).group_by(
        ["record_id", "data_source"]
    ):
        errors = record_errors(group.to_dicts())
        violations += [{"record_id": record_id, "data_source": data_source} | e for e in errors]
        verdicts.append({"record_id": record_id, "data_source": data_source, "valid": not errors})
    schema = {
        "record_id": pl.String,
        "data_source": pl.String,
        "field": pl.String,
        "error": pl.String,
        "location": pl.String,
        "value": pl.String,
    }
    return pl.DataFrame(violations, schema=schema), pl.DataFrame(
        verdicts, schema={"record_id": pl.String, "data_source": pl.String, "valid": pl.Boolean}
    )


def _batches(record_ids: pl.DataFrame, size: int) -> Iterator[pl.DataFrame]:
    for start in range(0, record_ids.height, size):
        yield record_ids.slice(start, size)


def _record_ids(path: Path, sample: int, seed: int) -> pl.DataFrame:
    ids = pl.scan_parquet(path).select("record_id").unique().collect(engine="streaming")
    if sample and sample < ids.height:
        return ids.sample(sample, seed=seed)
    return ids


def summarise(violations: pl.DataFrame) -> pl.DataFrame:
    """Violations collapsed to one row per (field, error), commonest first"""
    return (
        violations.group_by("field", "error")
        .agg(
            pl.col("record_id").n_unique().alias("records"),
            pl.col("location").unique().head(TOP_EXAMPLES).alias("locations"),
            pl.col("value").drop_nulls().unique().head(TOP_EXAMPLES).alias("examples"),
        )
        .sort("records", descending=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate compiled records against the MDS pydantic model.",
        prog="python -m mds_norm.evaluation.validate_records",
    )
    ap.add_argument("--path", type=Path, default=COMPILED_PATH, help=f"corpus to validate (raw: {RAW_RECORDS})")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="records to check, 0 for every record")
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--out", type=Path, default=VALIDATION_OUT)
    args = ap.parse_args()

    if not args.path.exists():
        raise SystemExit(f"no corpus at {args.path}")
    out = args.out / args.path.stem
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    ids = _record_ids(args.path, args.sample, args.seed)
    log(f"validating {ids.height:,} records from {args.path.name}")

    violations, verdicts = [], []
    for i, batch in enumerate(_batches(ids, BATCH_RECORDS), start=1):
        rows = (
            pl.scan_parquet(args.path)
            .select(COLUMNS)
            .join(batch.lazy(), on="record_id", how="semi")
            .collect(engine="streaming")
        )
        v, d = check(rows)
        violations.append(v)
        verdicts.append(d)
        log(f"  batch {i}: {d.height:,} records, {d.height - int(d['valid'].sum()):,} invalid")

    violations, verdicts = pl.concat(violations), pl.concat(verdicts)
    by_field = summarise(violations)
    by_institution = (
        verdicts.group_by("data_source")
        .agg(pl.len().alias("records"), pl.col("valid").sum().alias("valid"))
        .with_columns((pl.col("valid") / pl.col("records")).alias("valid_rate"))
        .sort("valid_rate")
    )
    violations.write_parquet(out / "violations.parquet")
    by_field.write_parquet(out / "violations_by_field.parquet")
    by_institution.write_parquet(out / "validity_by_institution.parquet")

    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "corpus": str(args.path.relative_to(ROOT)) if args.path.is_relative_to(ROOT) else str(args.path),
        "records": verdicts.height,
        "valid": int(verdicts["valid"].sum()),
        "valid_rate": round(float(verdicts["valid"].mean()), 4),
        "records_missing_required": int(violations.filter(pl.col("error") == "missing")["record_id"].n_unique()),
        "fields_violated": by_field.height,
        "seconds": round(time.time() - started, 1),
    }
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    with pl.Config(tbl_rows=25, fmt_str_lengths=60):
        print(by_field.head(25))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
