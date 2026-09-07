from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import get_args

import annotated_types
import polars as pl
from mds_data_model.introspection import all_free_text_fields, date_fields, fields_of_type
from mds_data_model.models import all_models
from mds_data_model.models.object import Object

from mds_norm import paths

# Default location of the flat-records parquet
DATA_PATH = paths.RAW_RECORDS

# Core fields; a record missing any is excluded from completeness
CORE_FIELDS = ("spectrum/object_number", "spectrum/object_name", "spectrum/brief_description")


# ``ciim/`` is administrative provenance, excluded from every descriptive metric
ADMIN = pl.col("field_type").cast(pl.String).str.starts_with("ciim/")

# ``wrmthorne/`` is release apparatus, excluded from every content metric
RELEASE = pl.col("field_type").cast(pl.String).str.starts_with("wrmthorne/")

# The core fields, handled separately from the weighted descriptive fields
COMPULSORY = pl.col("field_type").is_in(list(CORE_FIELDS))


def load_base(data_path: str | Path = DATA_PATH) -> pl.LazyFrame:
    """Scan the flat-records parquet, casting the low-cardinality keys to ``Enum`` to reduce memory overhead"""
    ldf = pl.scan_parquet(data_path)
    field_types = ldf.select(pl.col("field_type").unique()).collect(engine="streaming")["field_type"].to_list()
    data_sources = ldf.select(pl.col("data_source").unique()).collect(engine="streaming")["data_source"].to_list()
    return ldf.with_columns(
        pl.col("field_type").cast(pl.Enum(field_types)), pl.col("data_source").cast(pl.Enum(data_sources))
    )


def data_source_enum(base: pl.LazyFrame) -> pl.Enum:
    return base.collect_schema()["data_source"]


@cache
def date_field_names() -> frozenset[str]:
    """Prefixed names of every field that can hold a :class:`Date`, at any depth"""
    return frozenset(name for model in all_models() for name in date_fields(model))


@cache
def free_text_field_names() -> frozenset[str]:
    """Prefixed names of every free-text field across the model graph"""
    return frozenset(n for names in all_free_text_fields().values() for n in names)


def _has_gt0(tp: object) -> bool:
    """Whether the annotation carries a Gt(0) constraint anywhere in its tree"""
    return any((isinstance(a, annotated_types.Gt) and a.gt == 0) or _has_gt0(a) for a in get_args(tp))


@cache
def numeric_field_names() -> frozenset[str]:
    """Prefixed names of every field whose declared type admits a number"""
    names = {
        model.model_fields[name].alias or name
        for model in all_models()
        for t in (int, float)
        for name in fields_of_type(model, t)
    }
    return frozenset(n for n in names - {"spectrum/value"} if not n.startswith("wrmthorne/"))


@cache
def positive_int_field_names() -> frozenset[str]:
    """Prefixed names of fields constrained to positive integers"""
    return (
        frozenset(
            model.model_fields[name].alias or name
            for model in all_models()
            for name in fields_of_type(model, int)
            if _has_gt0(model.model_fields[name].annotation)
        )
        & numeric_field_names()
    )


@cache
def known_field_names() -> frozenset[str]:
    """Every recognised field alias across the model graph"""
    return frozenset(
        (model.model_fields[name].alias or name) for model in (Object, *all_models()) for name in model.model_fields
    )
