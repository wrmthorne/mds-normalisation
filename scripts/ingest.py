from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from mds_norm.paths import RAW_RECORDS, RECORD_ADMIN

type Json = dict[str, "Json"] | list["Json"] | str | int | float | None

BATCH_SIZE = 50_000
MAX_DEPTH = 4
ID_BYTES = 12  # the widest payload arrow stores inline in a binary view

INT_FIELDS = {"added", "processed", "sequence"}
TOP_KEYS = {"@admin", "@document"}
ADMIN_KEYS = {"added", "data_source", "id", "processed", "sequence", "source", "stream", "uid", "uuid"}
DATA_SOURCE_KEYS = {"code", "group", "name", "organisation"}
DOCUMENT_KEYS = {"type", "units"}
UNIT_KEYS = {"label", "path", "type", "units", "value"}

NODE_SCHEMA = {
    "record_id": pl.String(),
    "data_source": pl.String(),
    "node_id": pl.Binary(),
    "parent_id": pl.Binary(),
    "depth": pl.UInt8(),
    "source_array_pos": pl.UInt16(),
    "label": pl.String(),
    "path": pl.String(),
    "field_type": pl.String(),
    "value": pl.String(),
    "extra": pl.String(),
}
NODE_COLUMNS = list(NODE_SCHEMA)


def unit_struct(depth: int) -> pa.DataType:
    """The recursive `units` struct, `depth` levels deep"""
    leaf = [("extra", pa.string()), ("label", pa.string()), ("path", pa.string()), ("type", pa.string())]
    struct = pa.struct([*leaf, ("value", pa.string())])
    for _ in range(depth - 1):
        struct = pa.struct([*leaf, ("units", pa.list_(struct)), ("value", pa.string())])
    return struct


SCHEMA = pa.schema(
    [
        (
            "@admin",
            pa.struct(
                [
                    ("added", pa.int64()),
                    (
                        "data_source",
                        pa.struct(
                            [
                                ("code", pa.string()),
                                ("group", pa.string()),
                                ("name", pa.string()),
                                ("organisation", pa.string()),
                            ]
                        ),
                    ),
                    ("id", pa.string()),
                    ("processed", pa.int64()),
                    ("sequence", pa.int64()),
                    ("source", pa.string()),
                    ("stream", pa.string()),
                    ("uid", pa.string()),
                    ("uuid", pa.string()),
                ]
            ),
        ),
        ("@document", pa.struct([("type", pa.string()), ("units", pa.list_(unit_struct(MAX_DEPTH)))])),
        ("extra", pa.string()),
    ]
)


def text(value: Json) -> str | None:
    """Render a scalar as the schema's string, since the API sends numbers for some of them"""
    return value if value is None or isinstance(value, str) else str(value)


def spill(obj: dict[str, Json], known: set[str], prefix: str = "") -> dict[str, Json]:
    """Collect whatever `obj` carries beyond `known`, under dotted keys"""
    return {f"{prefix}{k}": v for k, v in obj.items() if k not in known}


def normalise_unit(unit: dict[str, Json], depth: int) -> dict[str, Json]:
    """Shape one unit to the schema, keeping anything the schema cannot hold in `extra`"""
    extra = spill(unit, UNIT_KEYS)
    children = unit.get("units")
    deepest = depth == MAX_DEPTH - 1
    if deepest and children:
        extra["units"] = children  # nesting past the modelled depth is kept verbatim
    out = {
        "extra": json.dumps(extra, ensure_ascii=False) if extra else None,
        "label": text(unit.get("label")),
        "path": text(unit.get("path")),
        "type": text(unit.get("type")),
        "value": text(unit.get("value")),
    }
    if not deepest:
        out["units"] = [normalise_unit(u, depth + 1) for u in children] if children else None
    return out


def normalise(record: dict[str, Json]) -> dict[str, Json]:
    """Shape one record to the schema, keeping anything the schema cannot hold in `extra`"""
    admin = record.get("@admin") or {}
    data_source = admin.get("data_source") or {}
    document = record.get("@document") or {}
    extra = (
        spill(record, TOP_KEYS)
        | spill(admin, ADMIN_KEYS, "@admin.")
        | spill(data_source, DATA_SOURCE_KEYS, "@admin.data_source.")
        | spill(document, DOCUMENT_KEYS, "@document.")
    )
    units = document.get("units")
    return {
        "@admin": {
            **{k: text(admin.get(k)) for k in ADMIN_KEYS - INT_FIELDS - {"data_source"}},
            **{k: int(admin[k]) if admin.get(k) is not None else None for k in INT_FIELDS},
            "data_source": {k: text(data_source.get(k)) for k in DATA_SOURCE_KEYS},
        },
        "@document": {
            "type": text(document.get("type")),
            "units": [normalise_unit(u, 0) for u in units] if units else None,
        },
        "extra": json.dumps(extra, ensure_ascii=False) if extra else None,
    }


@contextmanager
def open_jsonl(path: Path) -> Iterator[Iterator[str]]:
    """Yield the export's lines, decompressing on the way if it is zstd"""
    if path.suffix in {".zstd", ".zst"}:
        with path.open("rb") as handle, zstd.ZstdDecompressor().stream_reader(handle) as raw:
            yield io.TextIOWrapper(raw, encoding="utf-8")
    else:
        with path.open(encoding="utf-8") as stream:
            yield stream


def superseded(source: Path) -> set[int]:
    """The lines holding a copy of a record that a later line replaces"""
    last: dict[str, int] = {}
    stale: set[int] = set()
    with open_jsonl(source) as stream:
        for i, line in enumerate(stream):
            uuid = json.loads(line)["@admin"]["uuid"]
            if (earlier := last.get(uuid)) is not None:
                stale.add(earlier)
            last[uuid] = i
    print(f"[{time.strftime('%H:%M:%S')}] {len(last):,} records; {len(stale):,} superseded copies dropped", flush=True)
    return stale


def jsonl_batches(source: Path, batch_size: int) -> Iterator[pa.Table]:
    """Read the JSONL export in batches of `batch_size` records, newest copy per record"""
    stale = superseded(source)
    with open_jsonl(source) as stream:
        batch: list[Json] = []
        for i, line in enumerate(stream):
            if i in stale:
                continue
            batch.append(normalise(json.loads(line)))
            if len(batch) >= batch_size:
                yield pa.Table.from_pylist(batch, schema=SCHEMA)
                batch.clear()
        if batch:
            yield pa.Table.from_pylist(batch, schema=SCHEMA)


def parquet_batches(source: Path, batch_size: int) -> Iterator[pa.Table]:
    """Read the nested parquet in batches of `batch_size` records"""
    reader = pq.ParquetFile(source)
    for batch in reader.iter_batches(batch_size=batch_size):
        yield pa.Table.from_batches([batch])


def batches(source: Path, batch_size: int) -> Iterator[pa.Table]:
    """Read whichever form the source is in"""
    if source.suffix == ".parquet":
        return parquet_batches(source, batch_size)
    return jsonl_batches(source, batch_size)


def node_ids(n: int) -> pl.Series:
    """Draw `n` random identifiers, narrow enough for arrow to hold them inline"""
    raw = os.urandom(ID_BYTES * n)
    return pl.Series("node_id", [raw[i : i + ID_BYTES] for i in range(0, ID_BYTES * n, ID_BYTES)], dtype=pl.Binary)


def record_columns(batch: pl.DataFrame) -> pl.DataFrame:
    """Project (record_id, data_source, units) out of the MDS export schema"""
    return batch.select(
        pl.col("@admin").struct.field("uuid").alias("record_id"),
        pl.col("@admin").struct.field("data_source").struct.field("organisation").alias("data_source"),
        pl.col("@document").struct.field("units").alias("units"),
    )


def admin_columns(batch: pl.DataFrame) -> pl.DataFrame:
    """One row per record: everything in @admin, plus the document type"""
    admin = pl.col("@admin")
    data_source = admin.struct.field("data_source")
    return batch.select(
        admin.struct.field("uuid").alias("record_id"),
        admin.struct.field("uid").alias("uid"),
        admin.struct.field("id").alias("source_record_id"),
        admin.struct.field("source").alias("source"),
        admin.struct.field("stream").alias("stream"),
        data_source.struct.field("code").alias("code"),
        data_source.struct.field("group").alias("group"),
        data_source.struct.field("name").alias("name"),
        data_source.struct.field("organisation").alias("organisation"),
        pl.from_epoch(admin.struct.field("added"), time_unit="ms").dt.replace_time_zone("UTC").alias("added"),
        pl.from_epoch(admin.struct.field("processed"), time_unit="ms").dt.replace_time_zone("UTC").alias("processed"),
        admin.struct.field("sequence").alias("sequence"),
        pl.col("@document").struct.field("type").alias("document_type"),
        pl.col("extra").alias("extra"),
    )


def flatten_batch(df: pl.DataFrame, max_depth: int = MAX_DEPTH) -> pl.DataFrame:
    """Explode the nested units level by level, one row per node"""
    fields = [
        pl.col("_units").struct.field("label").alias("label"),
        pl.col("_units").struct.field("path").alias("path"),
        pl.col("_units").struct.field("type").alias("field_type"),
        pl.col("_units").struct.field("value").alias("value"),
        pl.col("_units").struct.field("extra").alias("extra"),
    ]
    positions = pl.int_ranges(pl.col("_units").list.len(), dtype=pl.UInt16).alias("source_array_pos")

    level = (
        df.select("record_id", "data_source", pl.col("units").alias("_units"))
        .with_row_index("_rec")
        .with_columns(positions)
        .explode("_units", "source_array_pos", empty_as_null=False)
        .filter(pl.col("_units").is_not_null())
        .with_columns(
            *fields,
            pl.col("_units").struct.field("units").alias("_children"),
            pl.lit(0, dtype=pl.UInt8).alias("depth"),
            pl.lit(None, dtype=pl.Binary).alias("parent_id"),
        )
    )
    if level.height == 0:
        return pl.DataFrame(schema=NODE_SCHEMA)

    level = level.with_columns(node_ids(level.height))
    parts = [level.select("_rec", *NODE_COLUMNS)]

    for depth in range(1, max_depth):
        children = level.select("_rec", "record_id", "data_source", "node_id", "_children").rename(
            {"node_id": "parent_id", "_children": "_units"}
        )
        level = (
            children.with_columns(positions)
            .explode("_units", "source_array_pos", empty_as_null=False)
            .filter(pl.col("_units").is_not_null())
            .with_columns(*fields, pl.lit(depth, dtype=pl.UInt8).alias("depth"))
        )
        if depth < max_depth - 1:
            level = level.with_columns(pl.col("_units").struct.field("units").alias("_children"))
        if level.height == 0:
            break
        level = level.with_columns(node_ids(level.height))
        parts.append(level.select("_rec", *NODE_COLUMNS))

    return pl.concat(parts).sort("_rec", maintain_order=True).drop("_rec")


def sink_staged(staging: Path, target: Path) -> None:
    """Merge the staged batches into one file"""
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.scan_parquet(staging / "*.parquet").sink_parquet(target)
    shutil.rmtree(staging)


def stage(target: Path, name: str) -> Path:
    """Make an empty staging directory beside `target` for the per-batch writes"""
    staging = target.with_name(f"_{target.stem}_{name}_staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def write_nested(source: Path, target: Path, batch_size: int) -> int:
    """Write the records in their original shape, returning how many were read"""
    if source.suffix == ".parquet":
        raise SystemExit("--form nested needs a JSONL source; the nested parquet is already in that shape")
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(target, SCHEMA)
    total = 0
    try:
        for table in jsonl_batches(source, batch_size):
            writer.write_table(table)
            total += table.num_rows
            print(f"[{time.strftime('%H:%M:%S')}] {total:,} records", flush=True)
    finally:
        writer.close()
    return total


def write_flat(source: Path, target: Path, admin_target: Path, batch_size: int) -> tuple[int, int]:
    """Write the node table and the record table, returning how many of each were read"""
    nodes_staging, admin_staging = stage(target, "nodes"), stage(admin_target, "admin")
    records = nodes = 0

    for i, table in enumerate(batches(source, batch_size)):
        batch = pl.from_arrow(table)
        admin_columns(batch).write_parquet(admin_staging / f"{i:05d}.parquet")
        flat = flatten_batch(record_columns(batch))
        if flat.height:
            flat.write_parquet(nodes_staging / f"{i:05d}.parquet")
            nodes += flat.height
        records += table.num_rows
        print(f"[{time.strftime('%H:%M:%S')}] {records:,} records, {nodes:,} nodes", flush=True)

    sink_staged(admin_staging, admin_target)
    sink_staged(nodes_staging, target)
    return records, nodes


def main() -> None:
    """Run the ingest from the command line"""
    ap = argparse.ArgumentParser(
        description="Read an MDS export and write either the nested record parquet or the flat node table."
    )
    ap.add_argument("source", type=Path, help="MDS export (.jsonl, .jsonl.zstd) or the nested parquet")
    ap.add_argument("target", type=Path, nargs="?", help=f"output file (flat default {RAW_RECORDS})")
    ap.add_argument("--form", choices=("flat", "nested"), default="flat", help="output shape (default flat)")
    ap.add_argument("--admin", type=Path, default=RECORD_ADMIN, help=f"record table (default {RECORD_ADMIN})")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    if args.form == "nested":
        if args.target is None:
            raise SystemExit("--form nested needs an explicit target")
        total = write_nested(args.source, args.target, args.batch_size)
        print(f"{total:,} records → {args.target}")
    else:
        target = args.target or RAW_RECORDS
        records, nodes = write_flat(args.source, target, args.admin, args.batch_size)
        print(f"{nodes:,} nodes → {target}")
        print(f"{records:,} records → {args.admin}")


if __name__ == "__main__":
    main()
