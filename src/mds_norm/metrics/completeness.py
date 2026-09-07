from __future__ import annotations

import polars as pl

from .common import ADMIN, COMPULSORY, CORE_FIELDS, RELEASE

# A field counts for an institution when fill exceeds this
APPLICABILITY = 0.10

# weight follows a retype, so retyping reads as no loss
RETYPE_SOURCES = {"spectrum/object_production_date": "spectrum/associated_date"}


def field_weights(base: pl.LazyFrame) -> pl.DataFrame:
    """Per-(field, institution) applicable weight `w = max(G, I)` over the full institution-by-field grid"""
    # Denominator of the local fill rate I
    record_counts = base.select("record_id", "data_source").unique().group_by("data_source").len(name="total_records")
    descriptive = base.filter(pl.col("value").is_not_null() & ~(COMPULSORY | ADMIN | RELEASE))
    # Distinct records populating each (field_type, data_source): the numerator of I
    field_counts = descriptive.group_by("field_type", "data_source").agg(
        pl.col("record_id").n_unique().alias("records_with_field")
    )
    # w = max(global fill, local fill) over the applicable fields
    return (
        record_counts.join(descriptive.select("field_type").unique(), how="cross")
        .join(field_counts, on=["field_type", "data_source"], how="left")
        .with_columns((pl.col("records_with_field").fill_null(0) / pl.col("total_records")).alias("local_fill"))
        .with_columns(pl.col("local_fill").mean().over("field_type").alias("global_fill"))
        .filter((pl.col("global_fill") > APPLICABILITY) | (pl.col("local_fill") > APPLICABILITY))
        .with_columns(pl.max_horizontal("global_fill", "local_fill").alias("w"))
        .select("field_type", "data_source", "w")
        .collect(engine="streaming")
    )


def compute(base: pl.LazyFrame, weights: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-record weighted completeness C, over records carrying all three core fields"""
    if weights is None:
        weights = field_weights(base)
    else:
        sch = base.collect_schema()
        weights = weights.with_columns(
            pl.col("field_type").cast(sch["field_type"]), pl.col("data_source").cast(sch["data_source"])
        )
    # Denominator of C per institution: total applicable weight (record-independent)
    denom = weights.group_by("data_source").agg(pl.col("w").sum().alias("total_weight"))

    # Records missing any core field are excluded from the metric
    core_records = (
        base.filter(pl.col("value").is_not_null() & COMPULSORY)
        .group_by("record_id", "data_source")
        .agg(pl.col("field_type").n_unique().alias("n_core"))
        .filter(pl.col("n_core") == len(CORE_FIELDS))
        .select("record_id", "data_source")
    )
    # numerator: w summed over the applicable fields the record populates
    ft_dtype = base.collect_schema()["field_type"]
    populated = (
        base.filter(pl.col("value").is_not_null() & ~(COMPULSORY | ADMIN | RELEASE))
        .group_by("record_id", "data_source")
        .agg(pl.col("field_type").unique())
        .explode("field_type")
    )
    direct = populated.join(weights.lazy(), on=["field_type", "data_source"], how="left")
    is_dst = pl.col("field_type").cast(pl.String).is_in(list(RETYPE_SOURCES))
    src_populated = populated.filter(pl.col("field_type").cast(pl.String).is_in(list(RETYPE_SOURCES.values()))).select(
        "record_id", "data_source", src=pl.col("field_type").cast(pl.String)
    )
    inherited = (
        direct.filter(is_dst)
        .with_columns(src=pl.col("field_type").cast(pl.String).replace_strict(RETYPE_SOURCES, default=None))
        .join(src_populated, on=["record_id", "data_source", "src"], how="anti")
        .with_columns(pl.col("src").cast(ft_dtype))
        .join(weights.lazy().rename({"field_type": "src", "w": "w_src"}), on=["src", "data_source"], how="left")
    )
    kept = pl.concat(
        [
            direct.filter(~is_dst),
            # destinations that also populate the source keep their own row
            direct.filter(is_dst).join(
                inherited.select("record_id", "data_source", "field_type"),
                on=["record_id", "data_source", "field_type"],
                how="anti",
            ),
        ]
    )
    numerator = (
        pl.concat(
            [kept, inherited.select("record_id", "data_source", field_type="src", w=pl.max_horizontal("w", "w_src"))]
        )
        .filter(pl.col("w").is_not_null())
        .unique(subset=["record_id", "data_source", "field_type"])
        .group_by("record_id", "data_source")
        .agg(pl.col("w").sum().alias("weighted_filled"))
    )
    # left join so records with no applicable field score 0
    return (
        core_records.join(numerator, on=["record_id", "data_source"], how="left")
        .with_columns(pl.col("weighted_filled").fill_null(0.0))
        .join(denom.lazy(), on="data_source", how="left")
        .with_columns((pl.col("weighted_filled") / pl.col("total_weight")).alias("completeness"))
        .select("record_id", "data_source", "completeness")
        .collect(engine="streaming")
    )
