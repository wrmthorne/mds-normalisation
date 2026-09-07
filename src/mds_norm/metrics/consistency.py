from __future__ import annotations

from pathlib import Path

import polars as pl

from .common import data_source_enum

_COLUMNS = ("record_id", "data_source", "n_conventional_applicable", "n_conventional_ok", "consistency")


def compute(consistency: pl.DataFrame | pl.LazyFrame | str | Path, base: pl.LazyFrame | None = None) -> pl.DataFrame:
    """Load the per-record consistency counts, keyed consistently with the metric frames"""
    if isinstance(consistency, (str, Path)):
        consistency = pl.read_parquet(consistency)
    elif isinstance(consistency, pl.LazyFrame):
        consistency = consistency.collect()
    consistency = consistency.select(_COLUMNS)
    if base is not None:
        consistency = consistency.with_columns(pl.col("data_source").cast(data_source_enum(base)))
    return consistency
