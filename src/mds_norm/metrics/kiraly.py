from __future__ import annotations

import polars as pl

from .common import ADMIN, RELEASE, free_text_field_names, known_field_names

# Occurrence shares the QA Catalogue profile reports coverage at
PARETO_QUANTILES = (0.2, 0.5, 0.8)
# Terms covering this share of a field's occurrences define k_80
COVERAGE = 0.80
# Coherence is undefined below this many populated textual fields
MIN_TEXT_FIELDS = 2
# scikit-learn's default word token: two or more word characters
WORD_PATTERN = r"\b\w\w+\b"

DESCRIPTIVE = pl.col("value").is_not_null() & ~(ADMIN | RELEASE)


def schema_fields() -> list[str]:
    """The fields the standard defines, less the CIIM administrative and release namespaces"""
    return sorted(f for f in known_field_names() if not f.startswith(("ciim/", "wrmthorne/")))


def categorical() -> pl.Expr:
    """Populated descriptive values outside the free-text fields, where a value-frequency reading is meaningful"""
    return DESCRIPTIVE & ~pl.col("field_type").cast(pl.String).is_in(sorted(free_text_field_names()))


def importance_weights(base: pl.LazyFrame) -> pl.DataFrame:
    """One coefficient per field, the corpus-global fill rate, identical for every institution"""
    records = base.select("record_id", "data_source").unique().group_by("data_source").len(name="total")
    counts = (
        base.filter(DESCRIPTIVE)
        .group_by("field_type", "data_source")
        .agg(pl.col("record_id").n_unique().alias("n_with"))
    )
    return (
        records.join(base.filter(DESCRIPTIVE).select("field_type").unique(), how="cross")
        .join(counts, on=["field_type", "data_source"], how="left")
        .with_columns((pl.col("n_with").fill_null(0) / pl.col("total")).alias("fill"))
        .group_by("field_type")
        .agg(pl.col("fill").mean().alias("w"))
        .collect(engine="streaming")
    )


def completeness(base: pl.LazyFrame, weights: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-record completeness against a fixed denominator of every field the standard defines"""
    if weights is None:
        weights = importance_weights(base)
    fields = schema_fields()
    total_w = weights.filter(pl.col("field_type").is_in(fields))["w"].sum()
    return (
        base.filter(DESCRIPTIVE)
        .select("record_id", "data_source", pl.col("field_type").cast(pl.String))
        .unique()
        .filter(pl.col("field_type").is_in(fields))
        .join(weights.lazy(), on="field_type", how="left")
        .group_by("record_id", "data_source")
        .agg(pl.len().alias("n_populated"), pl.col("w").fill_null(0.0).sum().alias("sum_w"))
        .with_columns(
            (pl.col("n_populated") / len(fields)).alias("k_completeness_ratio"),
            (pl.col("sum_w") / total_w).alias("k_completeness_weighted"),
        )
        .select("record_id", "data_source", "k_completeness_ratio", "k_completeness_weighted")
        .collect(engine="streaming")
    )


def value_frequencies(base: pl.LazyFrame, restrict_to: pl.LazyFrame | None = None) -> pl.DataFrame:
    """Corpus occurrence counts per (field, value) and per field, the input to information content"""
    values = base.filter(categorical())
    if restrict_to is not None:
        pairs = restrict_to.select("field_type", "value").unique()
        values = values.join(pairs, on=["field_type", "value"], how="semi")
    counts = values.group_by("field_type", "value").len(name="times")
    totals = base.filter(categorical()).group_by("field_type").len(name="n_f")
    return counts.join(totals, on="field_type", how="left").collect(engine="streaming")


def information_content(frequencies: pl.DataFrame) -> pl.DataFrame:
    """Per-value information content 1 - log(times) / log(n_f); 0 for a value everyone shares, 1 for a unique one"""
    return frequencies.with_columns(
        (1.0 - pl.col("times").log() / pl.col("n_f").log()).clip(0.0, 1.0).alias("ic")
    ).select("field_type", "value", "ic")


def conformance_to_expectations(base: pl.LazyFrame, frequencies: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-record mean information content over its categorical values"""
    if frequencies is None:
        frequencies = value_frequencies(base)
    return (
        base.filter(categorical())
        .select("record_id", "data_source", pl.col("field_type").cast(pl.String), "value")
        .collect(engine="streaming")
        .join(information_content(frequencies), on=["field_type", "value"], how="left")
        .group_by("record_id", "data_source")
        .agg(pl.col("ic").mean().alias("k_information_content"), pl.len().alias("n_categorical_values"))
    )


def documents(base: pl.LazyFrame) -> pl.LazyFrame:
    """One document per populated textual field of each record: its values joined with a space"""
    free_text = pl.col("field_type").cast(pl.String).is_in(sorted(free_text_field_names()))
    return (
        base.filter(DESCRIPTIVE & free_text)
        .group_by("record_id", "data_source", "field_type")
        .agg(pl.col("value").str.join(" ").alias("text"))
        .filter(pl.col("text").str.len_chars() > 0)
    )


def chunk_of(chunk: int, n_chunks: int) -> pl.Expr:
    """Membership of a record in one of ``n_chunks`` hash partitions"""
    return pl.col("record_id").hash() % n_chunks == chunk


# Lower-cased word tokens of a document's text
TOKENS = pl.col("text").str.to_lowercase().str.extract_all(WORD_PATTERN).alias("token")


def term_counts(docs: pl.LazyFrame) -> pl.LazyFrame:
    """Word tokens per document with their in-document counts"""
    return (
        docs.select("record_id", "field_type", TOKENS)
        .explode("token", empty_as_null=True)
        .drop_nulls("token")
        .group_by("record_id", "field_type", "token")
        .len(name="tf")
    )


def idf_weights(base: pl.LazyFrame, min_df: int = 3, max_features: int = 50_000, n_chunks: int = 1) -> pl.DataFrame:
    """Smoothed inverse document frequency per token, over every textual document ``base`` holds"""
    docs = documents(base)
    n_docs = docs.select(pl.len()).collect(engine="streaming").item()

    def count(chunk: int) -> pl.DataFrame:
        listed = docs.filter(chunk_of(chunk, n_chunks)).select(TOKENS)
        distinct = listed.select(pl.col("token").list.unique()).explode("token", empty_as_null=True).drop_nulls()
        total = listed.explode("token", empty_as_null=True).drop_nulls().group_by("token").len("total")
        return distinct.group_by("token").len("df").join(total, on="token").collect(engine="streaming")

    return (
        pl.concat([count(chunk) for chunk in range(n_chunks)])
        .lazy()
        .group_by("token")
        .agg(pl.col("df").sum(), pl.col("total").sum())
        .filter(pl.col("df") >= min_df)
        .sort("total", "token", descending=[True, False])
        .head(max_features)
        .select("token", (((1 + n_docs) / (1 + pl.col("df"))).log() + 1).alias("idf"))
        .with_row_index("token_id")
        .collect()
    )


def coherence(
    base: pl.LazyFrame, idf: pl.DataFrame | None = None, min_df: int = 3, max_features: int = 50_000, n_chunks: int = 1
) -> pl.DataFrame:
    """Per-record mean pairwise cosine distance between the tf-idf vectors of its textual fields"""
    if idf is None:
        idf = idf_weights(base, min_df=min_df, max_features=max_features)
    vocab = idf.lazy().select("token", "token_id", "idf")
    docs = documents(base)
    multi = docs.group_by("record_id", "data_source").len(name="n_docs").filter(pl.col("n_docs") >= MIN_TEXT_FIELDS)
    docs = docs.join(multi, on=["record_id", "data_source"], how="semi")

    def score(chunk: int) -> pl.DataFrame:
        in_chunk = chunk_of(chunk, n_chunks)
        weights = (
            term_counts(docs.filter(in_chunk))
            .join(vocab, on="token")
            .select("record_id", "field_type", "token_id", ((1 + pl.col("tf").log()) * pl.col("idf")).alias("w"))
            .with_columns((pl.col("w") / (pl.col("w") ** 2).sum().sqrt().over("record_id", "field_type")).alias("w"))
        )
        similarity = (
            weights.join(weights, on=["record_id", "token_id"], suffix="_b")
            .filter(pl.col("field_type") < pl.col("field_type_b"))
            .group_by("record_id")
            .agg((pl.col("w") * pl.col("w_b")).sum().alias("similarity"))
        )
        n_pairs = pl.col("n_docs") * (pl.col("n_docs") - 1) / 2
        distance = (1 - pl.col("similarity").fill_null(0.0) / n_pairs).clip(0.0, 1.0)
        return (
            multi.filter(in_chunk)
            .join(similarity, on="record_id", how="left")
            .select("record_id", "data_source", distance.alias("k_coherence"))
            .collect(engine="streaming")
        )

    return pl.concat([score(chunk) for chunk in range(n_chunks)])


def provenance(scores: pl.DataFrame, column: str) -> pl.DataFrame:
    """Per-source reputation: the mean of a record-level quality score over everything the source contributed"""
    return scores.group_by("data_source").agg(pl.col(column).mean().alias("k_provenance"), pl.len().alias("n_records"))


def field_frequency_profile(base: pl.LazyFrame) -> pl.DataFrame:
    """QA Catalogue field frequency pattern: each institution's schema elements ranked by occurrence"""
    return (
        base.filter(DESCRIPTIVE)
        .group_by("data_source", "field_type")
        .len(name="occurrences")
        .with_columns((pl.col("occurrences") / pl.col("occurrences").sum().over("data_source")).alias("share"))
        .sort("data_source", "share", descending=[False, True])
        .with_columns(
            pl.col("share").cum_sum().over("data_source").alias("cum_share"),
            pl.int_range(1, pl.len() + 1).over("data_source").alias("rank"),
        )
        .collect(engine="streaming")
    )


def pareto_summary(profile: pl.DataFrame) -> pl.DataFrame:
    """Whether an institution's element usage is Pareto-shaped: how few fields carry most of its statements"""
    quantile_cols = [
        (pl.col("cum_share") < q).sum().add(1).clip(upper_bound=pl.len()).alias(f"k_{int(q * 100)}")
        for q in PARETO_QUANTILES
    ]
    return profile.group_by("data_source").agg(
        pl.len().alias("n_fields"),
        *quantile_cols,
        # share of statements held by the top fifth of fields
        pl.col("share").filter(pl.col("rank") <= (pl.len() / 5).ceil()).sum().alias("top_fifth_share"),
    )


def field_vocabulary(base: pl.LazyFrame, frequencies: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-field vocabulary concentration alongside mean information content, over the same value distribution"""
    if frequencies is None:
        frequencies = value_frequencies(base)
    ranked = (
        frequencies.sort("field_type", "times", descending=[False, True])
        .with_columns((pl.col("times") / pl.col("times").sum().over("field_type")).alias("share"))
        .with_columns(pl.col("share").cum_sum().over("field_type").alias("cum_share"))
    )
    ic = information_content(frequencies).join(
        frequencies.select("field_type", "value", "times"), on=["field_type", "value"]
    )
    return (
        ranked.group_by("field_type")
        .agg(
            pl.col("times").sum().alias("n_occurrences"),
            pl.len().alias("n_distinct"),
            (pl.col("times") == 1).mean().alias("singleton_rate"),
            (pl.col("cum_share") < COVERAGE).sum().add(1).clip(upper_bound=pl.len()).alias("k_80"),
        )
        .join(
            ic.group_by("field_type").agg(
                pl.col("ic").mean().alias("mean_ic"),
                (pl.col("ic") * pl.col("times")).sum().truediv(pl.col("times").sum()).alias("mean_ic_weighted"),
            ),
            on="field_type",
        )
    )
