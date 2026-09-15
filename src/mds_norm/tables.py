from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from mds_norm.paths import RAW_RECORDS

# export-only: array position, and what the schema could not hold
SOURCE_ONLY = {"source_array_pos": pl.UInt16, "extra": pl.String}

LICENCE_FIELD = "ciim/license"
CC0_LICENCE = "CC 0"


def cc0_licences(path: Path | None = None) -> pl.LazyFrame:
    """Record ids and institutions of every record whose licence unit says CC0"""
    return (
        pl.scan_parquet(path or RAW_RECORDS)
        .filter(pl.col("field_type") == LICENCE_FIELD, pl.col("value") == CC0_LICENCE)
        .select("record_id", "data_source")
        .unique()
    )


def source_only(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Give a frame of generated nodes the source-only columns, null throughout"""
    return frame.with_columns(pl.lit(None, dtype=dtype).alias(name) for name, dtype in SOURCE_ONLY.items())


def record_nodes(record_ids: Iterable[str], path: Path | None = None) -> dict[str, dict]:
    """Each record's nodes in document order"""
    view = ("depth", "path", "label", "field_type", "value")
    nodes = (
        pl.scan_parquet(path or RAW_RECORDS)
        .filter(pl.col("record_id").is_in(list(record_ids)))
        .select(
            "record_id",
            "data_source",
            "source_array_pos",
            *view,
            pl.col("node_id").bin.encode("hex").alias("node_id"),
            pl.col("parent_id").bin.encode("hex").alias("parent_id"),
        )
        .collect(engine="streaming")
        .sort("depth", "source_array_pos", nulls_last=True)
    )
    out: dict[str, dict] = {}
    for (rid, ds), g in nodes.group_by(["record_id", "data_source"], maintain_order=True):
        out[rid] = {
            "record_id": rid,
            "data_source": ds,
            "nodes": [
                {"id": nid, "parent": pid, "depth": dep, "path": pth, "label": lab, "field_type": ft, "value": val}
                for nid, pid, dep, pth, lab, ft, val in g.select("node_id", "parent_id", *view).iter_rows()
            ],
        }
    return out
