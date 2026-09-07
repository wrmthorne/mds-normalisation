from __future__ import annotations

import time
from pathlib import Path

import ftfy
import polars as pl
from codecarbon import EmissionsTracker

from mds_norm.paths import EMISSIONS_LOG, FIELD_STATS, RAW_RECORDS
from mds_norm.pipeline.consistency_induction import (
    OBJECT_NUMBER_FIELDS,
    PATTERN_FIELDS,
    PRICE_FIELDS,
    REFERENCE_NUMBER_FIELDS,
    induce_lookup,
)

STOPWORDS = frozenset(
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
    ]
)
STOPWORD_REGEX = r"(?i)\b(?:" + "|".join(sorted(STOPWORDS)) + r")\b"

_TLD = r"com|org|net|edu|gov|mil|int|io|co|ai|app|dev|info|biz|xyz|uk|us|ca|de|fr|jp|cn|au|in|ru|nl|eu"
URL_REGEX = r"(?i)\b(?:https?://)?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+" rf"(?:{_TLD})\b(?:/[^\s]*)?"
EMAIL_REGEX = (
    r"(?i)\b[a-z0-9._%+-]+@[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\.[a-z]{2,}\b"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def flatten_verbose(pattern: str) -> str:
    """Strip re.VERBOSE whitespace and comments so Rust's regex engine accepts the pattern"""
    out, in_class, escaped, in_comment = [], False, False, False
    for ch in pattern:
        if in_comment:
            if ch == "\n":
                in_comment = False
        elif escaped:
            out.append(ch)
            escaped = False
        elif ch == "\\":
            out.append(ch)
            escaped = True
        elif in_class:
            out.append(ch)
            if ch == "]":
                in_class = False
        elif ch == "[":
            in_class = True
            out.append(ch)
        elif ch == "#":
            in_comment = True
        elif not ch.isspace():
            out.append(ch)
    return "".join(out)


MOJIBAKE_REGEX = flatten_verbose(ftfy.badness.BADNESS_RE.pattern)


def fix_mojibake(values: pl.Series) -> pl.Series:
    return pl.Series([ftfy.fix_text(v) for v in values], dtype=pl.String())


def merge_group() -> pl.Expr:
    """Fields sharing an induction group; dates and identifiers pool across field types"""
    return (
        pl.when(pl.col("field_type").str.contains("date"))
        .then(pl.lit("__date__"))
        .when(pl.col("field_type").is_in(OBJECT_NUMBER_FIELDS))
        .then(pl.lit("__object_number__"))
        .when(pl.col("field_type").is_in(REFERENCE_NUMBER_FIELDS))
        .then(pl.lit("__reference_number__"))
        .when(pl.col("field_type").is_in(PRICE_FIELDS))
        .then(pl.lit("__price__"))
        .otherwise(pl.col("field_type"))
        .alias("merge_group")
    )


def standardise(path: Path | None = None) -> pl.LazyFrame:
    """Trimmed, mojibake-repaired spectrum values with their per-value features"""
    source = path or RAW_RECORDS
    data_sources = (
        pl.scan_parquet(source).select(pl.col("data_source").unique()).collect(engine="streaming")["data_source"]
    ).to_list()

    base = (
        pl.scan_parquet(source)
        # an enum keeps institution labels off the string heap
        .with_columns(pl.col("data_source").cast(pl.Enum(data_sources)))
        .filter(pl.col("field_type").str.starts_with("spectrum/") & pl.col("value").is_not_null())
        .with_columns(pl.col("value").str.strip_chars())
        .filter(pl.col("value") != "")
        .with_columns(contains_mojibake=pl.col("value").str.contains(MOJIBAKE_REGEX))
    )

    # ftfy is a Python call: flagged subset only
    repaired = base.filter(pl.col("contains_mojibake")).select(
        "node_id",
        pl.col("value").map_batches(fix_mojibake, return_dtype=pl.String(), is_elementwise=True).alias("value_fixed"),
    )

    return (
        base.join(repaired, on="node_id", how="left")
        .with_columns(pl.coalesce("value_fixed", "value").alias("value"))
        .drop("value_fixed")
        .with_columns(
            char_count=pl.col("value").str.len_chars(),
            digit_chars=pl.col("value").str.count_matches(r"\d"),
            alpha_chars=pl.col("value").str.count_matches(r"[A-Za-z]"),
            punct_chars=pl.col("value").str.count_matches(r"[^\w\s]"),
            delim_chars=pl.col("value").str.count_matches(r"[,;/|]"),
            paren_count=pl.col("value").str.count_matches(r"\("),
            token_count=pl.col("value").str.count_matches(r"\b\w+\b"),
            stop_count=pl.col("value").str.count_matches(STOPWORD_REGEX),
            # Europeana contamination: a closing tag is rarely incidental
            contains_html=pl.col("value").str.contains(r"</[a-zA-Z][^>]*>"),
            contains_url=pl.col("value").str.contains(URL_REGEX),
            contains_email=pl.col("value").str.contains(EMAIL_REGEX),
            escape_count=pl.col("value").str.count_matches(r"[\n\t\r]"),
            # distinguishes identifiers and enums from prose
            no_lowercase=pl.col("value").str.contains(r"^[^a-z]*$"),
        )
    )


def census(standardised: pl.LazyFrame) -> pl.LazyFrame:
    """Attach the induced pattern columns to the inducible fields, pass the rest through"""
    scoped = standardised.filter(PATTERN_FIELDS).select(merge_group(), "value")
    lookup = induce_lookup(scoped, keep_features=True).lazy()

    grouped = standardised.with_columns(merge_group())
    return pl.concat(
        [
            grouped.filter(PATTERN_FIELDS).join(lookup, on=["merge_group", "value"], how="left"),
            grouped.filter(~PATTERN_FIELDS),
        ],
        how="diagonal_relaxed",
    )


def main() -> None:
    FIELD_STATS.parent.mkdir(parents=True, exist_ok=True)
    EMISSIONS_LOG.mkdir(parents=True, exist_ok=True)
    with EmissionsTracker(project_name="tier0_standardise", output_dir=str(EMISSIONS_LOG), log_level="error"):
        log(f"scanning {RAW_RECORDS.name}…")
        census(standardise()).sink_parquet(FIELD_STATS)
    total = pl.scan_parquet(FIELD_STATS).select(pl.len()).collect(engine="streaming").item()
    log(f"{total:,} nodes → {FIELD_STATS}")


if __name__ == "__main__":
    main()
