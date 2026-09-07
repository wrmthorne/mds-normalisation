from __future__ import annotations

import polars as pl
from mds_data_model.introspection import all_free_text_fields

from .common import ADMIN, RELEASE

# Skew is undefined below this many characters of content
SKEW_MIN_CHARS = 50
# Institution percentile that information density normalises against
DENSITY_PERCENTILE = 0.9


def _char_stats(base: pl.LazyFrame) -> pl.DataFrame:
    """Per-record descriptive character counts, split into free-text and structured"""
    # cast to String: the set names fields the data lacks
    free_text = (
        pl.col("field_type").cast(pl.String).is_in([n for names in all_free_text_fields().values() for n in names])
    )
    return (
        base.filter(~ADMIN & ~RELEASE & (pl.col("depth") == 0))
        .with_columns(pl.col("value").str.len_chars().fill_null(0).alias("clen"))
        .with_columns(pl.when(free_text).then("clen").otherwise(0).alias("clen_free"))
        .group_by("record_id", "data_source")
        .agg(pl.col("clen").sum().alias("c_total"), pl.col("clen_free").sum().alias("c_free"))
        .with_columns((pl.col("c_total") - pl.col("c_free")).alias("c_structured"))
        .collect(engine="streaming")
    )


def _fragments(base: pl.LazyFrame) -> pl.LazyFrame:
    # a fragment is decomposed when it parents any depth-1 node
    parent_ids = (
        base.filter((pl.col("depth") == 1) & ~ADMIN & ~RELEASE)
        .select("parent_id")
        .unique()
        .collect(engine="streaming")["parent_id"]
    )
    return (
        base.filter((pl.col("depth") == 0) & ~ADMIN & ~RELEASE)
        .with_columns(pl.col("node_id").is_in(parent_ids.implode()).alias("is_decomposed"))
        .filter(pl.col("value").is_not_null() | pl.col("is_decomposed"))  # populated or container roots
    )


def decomposition_propensity(base: pl.LazyFrame) -> pl.DataFrame:
    """Corpus-wide p_f per field: the fraction of its fragments that are decomposed"""
    return (
        _fragments(base)
        .group_by("field_type")
        .agg(pl.col("is_decomposed").mean().alias("p_f"))
        .collect(engine="streaming")
    )


def _decomposition(base: pl.LazyFrame, propensity: pl.DataFrame | None = None) -> pl.LazyFrame:
    """Per-record decomposition rate and depth"""
    fragments = _fragments(base)
    if propensity is None:
        propensity = (
            fragments.group_by("field_type")
            .agg(pl.col("is_decomposed").mean().alias("p_f"))
            .collect(engine="streaming")
        )
    else:
        propensity = propensity.with_columns(pl.col("field_type").cast(base.collect_schema()["field_type"]))
    rate = (
        fragments.join(propensity.lazy(), on="field_type", how="left")
        .with_columns(pl.col("p_f").fill_null(0.0))
        .group_by("record_id", "data_source")
        .agg((pl.col("p_f") * pl.col("is_decomposed")).sum().alias("num"), pl.col("p_f").sum().alias("den"))
        .with_columns(
            pl.when(pl.col("den") > 0).then(pl.col("num") / pl.col("den")).otherwise(None).alias("decomposition_rate")
        )
        .select("record_id", "data_source", "decomposition_rate")
    )
    depth = (
        base.filter((pl.col("depth") == 1) & ~ADMIN & ~RELEASE)
        .group_by("record_id", "data_source")
        .agg(pl.col("parent_id").n_unique().alias("n_decomposed"), pl.len().alias("total_sub"))
        .with_columns((pl.col("total_sub") / pl.col("n_decomposed")).alias("decomposition_depth"))
        .select("record_id", "data_source", "decomposition_depth")
    )
    return rate.join(depth, on=["record_id", "data_source"], how="left")


def compute(base: pl.LazyFrame, propensity: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-record thinness sub-metrics"""
    char_stats = _char_stats(base)
    thin = (
        char_stats.with_columns(
            # c_total over the institution's 90th percentile, capped at 1
            pl.min_horizontal(
                pl.col("c_total") / pl.col("c_total").quantile(DENSITY_PERCENTILE).over("data_source"), 1.0
            ).alias("information_density"),
            # skew = c_free / c_total; undefined below the character threshold
            pl.when(pl.col("c_total") >= SKEW_MIN_CHARS)
            .then(pl.col("c_free") / pl.col("c_total"))
            .otherwise(None)
            .alias("skew"),
        )
        .with_columns(
            # the share of the usual volume this record leaves unstructured
            (1 - pl.col("information_density") * (1 - pl.col("skew").fill_null(0.0))).alias("thinness")
        )
        .select("record_id", "data_source", "information_density", "skew", "thinness")
    )

    decomposition = _decomposition(base, propensity).collect(engine="streaming")
    return thin.join(decomposition, on=["record_id", "data_source"], how="left")
