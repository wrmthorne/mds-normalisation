from __future__ import annotations

from collections.abc import Callable

import polars as pl

from mds_norm import paths
from mds_norm.metrics import completeness, conformance, kiraly, thinness

OUT = paths.EXP_OUT / "kiraly"

# A balanced sample, so no institution's habits dominate
MIN_RECORDS, PER_INSTITUTION = 300, 600
SEED = 20260806


def cached(name: str, build: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    """Read the artefact if it exists, otherwise build and write it"""
    path = OUT / f"{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = build()
    frame.write_parquet(path)
    return frame


def institution_counts() -> pl.DataFrame:
    return cached(
        "institution_counts",
        lambda: (
            pl.scan_parquet(paths.RAW_RECORDS)
            .select("record_id", "data_source")
            .unique()
            .group_by("data_source")
            .len(name="n_records")
            .collect(engine="streaming")
        ),
    )


def sample_records() -> pl.DataFrame:
    """(record_id, data_source) for the balanced sample"""
    counts = institution_counts()
    eligible = counts.filter(pl.col("n_records") >= MIN_RECORDS)["data_source"].to_list()
    return cached(
        "sample_records",
        lambda: (
            pl.scan_parquet(paths.RAW_RECORDS)
            .select("record_id", "data_source")
            .unique()
            .filter(pl.col("data_source").is_in(eligible))
            .collect(engine="streaming")
            .sample(fraction=1.0, shuffle=True, seed=SEED)
            .group_by("data_source")
            .head(PER_INSTITUTION)
        ),
    )


def sample_rows() -> pl.DataFrame:
    """Every statement of every sampled record"""
    sample = sample_records()
    return cached(
        "sample_rows",
        lambda: (
            pl.scan_parquet(paths.RAW_RECORDS)
            .join(sample.lazy(), on=["record_id", "data_source"], how="semi")
            .collect(engine="streaming")
        ),
    )


def typed(rows: pl.DataFrame, counts: pl.DataFrame, wide: bool = False) -> pl.LazyFrame:
    """The sample as the metrics want it: field and institution as enums, optionally widened to the frozen tables"""
    fields = set(rows["field_type"].cast(pl.String))
    if wide:
        frozen_weights = pl.read_parquet(paths.METRICS_OUT / "field_weights.parquet")
        frozen_propensity = pl.read_parquet(paths.METRICS_OUT / "decomposition_propensity.parquet")
        fields |= set(frozen_weights["field_type"]) | set(frozen_propensity["field_type"])
    return rows.lazy().with_columns(
        pl.col("field_type").cast(pl.String).cast(pl.Enum(sorted(fields))),
        pl.col("data_source").cast(pl.String).cast(pl.Enum(sorted(counts["data_source"]))),
    )


def record_metrics() -> pl.DataFrame:
    """Every per-record measure over the sample: ours, and Kiraly's beside them"""
    counts, rows, sample = institution_counts(), sample_rows(), sample_records()
    base, wide = typed(rows, counts), typed(rows, counts, wide=True)
    corpus = pl.scan_parquet(paths.RAW_RECORDS)

    value_frequencies = cached("value_frequencies", lambda: kiraly.value_frequencies(corpus, restrict_to=base))
    importance = cached("importance_weights", lambda: kiraly.importance_weights(corpus))
    idf = cached("idf", lambda: kiraly.idf_weights(corpus, n_chunks=16))

    def build() -> pl.DataFrame:
        frozen_weights = pl.read_parquet(paths.METRICS_OUT / "field_weights.parquet")
        frozen_propensity = pl.read_parquet(paths.METRICS_OUT / "decomposition_propensity.parquet")
        ours = (
            completeness.compute(wide, weights=frozen_weights)
            .join(
                thinness.compute(wide, propensity=frozen_propensity),
                on=["record_id", "data_source"],
                how="full",
                coalesce=True,
            )
            .join(
                conformance.compute(wide).select("record_id", "data_source", "conformance"),
                on=["record_id", "data_source"],
                how="full",
                coalesce=True,
            )
            .join(
                pl.scan_parquet(paths.CONSISTENCY_OUT / "consistency_record_raw.parquet")
                .join(sample.lazy(), on="record_id", how="semi")
                .select("record_id", "consistency")
                .collect(engine="streaming"),
                on="record_id",
                how="left",
            )
        )
        theirs = (
            kiraly.completeness(base, weights=importance)
            .join(
                kiraly.conformance_to_expectations(base, frequencies=value_frequencies),
                on=["record_id", "data_source"],
                how="full",
                coalesce=True,
            )
            .join(kiraly.coherence(base, idf=idf).drop("data_source"), on="record_id", how="left")
        )
        return ours.join(theirs, on=["record_id", "data_source"], how="full", coalesce=True).with_columns(
            pl.col("data_source").cast(pl.String)
        )

    return cached("record_metrics", build)
