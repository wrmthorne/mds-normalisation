from __future__ import annotations

from pathlib import Path

import polars as pl

from mds_norm.utils.markup import HTML_TAG_RE

from .common import (
    ADMIN,
    RELEASE,
    data_source_enum,
    date_field_names,
    free_text_field_names,
    known_field_names,
    numeric_field_names,
    positive_int_field_names,
)

CURRENT_YEAR = 2026
SHORT_DESC = 20  # chars: a description shorter than this is "very short"
LONG_TITLE = 150  # chars: a title longer than this is "extremely long"
TITLE_REUSE = 25  # a title this many records share is "systematically reused"

# a real tag only; bracketed content is a recording convention
HTML_RE = HTML_TAG_RE
ENC_RE = r"&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);|â€™|â€œ|â€|Ã©|Ã¨|Ã |Ã¼|Ã¶|Ã¤|Ã±|Ã³|Ã­|Â£|Â©|Â°|�"
URI_RE = r"^\s*https?://\S+\s*$"
WORD_RE = r"[A-Za-z]{3,}"

TITLE, DESCR, OBJNUM = "spectrum/title", "spectrum/brief_description", "spectrum/object_number"

NUMERIC_RE = r"^-?\d+(\.\d+)?$"
POSINT_RE = r"^[1-9]\d*$"

WellformedInput = "pl.DataFrame | pl.LazyFrame | str | Path | None"


def _load_wellformed(wellformed: WellformedInput, enum: pl.Enum) -> pl.DataFrame | None:
    """Normalise the well-formedness input to a DataFrame keyed by the base enum"""
    if wellformed is None:
        return None
    if isinstance(wellformed, (str, Path)):
        wellformed = pl.read_parquet(wellformed)
    elif isinstance(wellformed, pl.LazyFrame):
        wellformed = wellformed.collect()
    return wellformed.select("record_id", "data_source", "n_wellformed_applicable", "n_wellformed_ok").with_columns(
        pl.col("data_source").cast(enum)
    )


def compute(base: pl.LazyFrame, wellformed: WellformedInput = None) -> pl.DataFrame:
    """Per-record conformance and per-check pass flags"""
    date_fields = list(date_field_names())
    known_fields = known_field_names()

    # Title, description and object_number per record, pivoted wide
    kf = (
        base.filter(pl.col("field_type").is_in([TITLE, DESCR, OBJNUM]) & pl.col("value").is_not_null())
        .with_columns(pl.col("field_type").cast(pl.String))
        .group_by("record_id", "data_source", "field_type")
        .agg(pl.col("value").first().alias("v"))
        .collect(engine="streaming")
        .pivot(values="v", index=["record_id", "data_source"], on="field_type")
    )
    # pivot omits absent fields; rename what exists, backfill the rest
    kf = kf.rename({k: v for k, v in {TITLE: "title", DESCR: "descr", OBJNUM: "objnum"}.items() if k in kf.columns})
    for col in ("title", "descr", "objnum"):
        if col not in kf.columns:
            kf = kf.with_columns(pl.lit(None, dtype=pl.String).alias(col))
    # Title reuse counts within each institution
    reuse = kf.filter(pl.col("title").is_not_null()).group_by("data_source", "title").len(name="title_count")
    kf = kf.join(reuse, on=["data_source", "title"], how="left")

    # HTML and encoding flags, plus URL-only free-text values
    free_text = pl.col("field_type").cast(pl.String).is_in(sorted(free_text_field_names()))
    flags = (
        base.filter(~ADMIN & ~RELEASE & pl.col("value").is_not_null())
        .group_by("record_id", "data_source")
        .agg(
            pl.col("value").str.contains(HTML_RE).any().alias("has_html"),
            pl.col("value").str.contains(ENC_RE).any().alias("has_enc"),
            free_text.any().alias("has_text"),
            (free_text & pl.col("value").str.contains(URI_RE)).any().alias("has_uri_text"),
        )
        .collect(engine="streaming")
    )
    # only a same-group repeat counts; siblings legitimately share values
    dup = (
        base.filter(~ADMIN & ~RELEASE & pl.col("value").is_not_null())
        .with_columns(pl.struct("record_id", "parent_id", "field_type", "value").hash().alias("h"))
        .with_columns(pl.col("h").is_duplicated().alias("isdup"))
        .group_by("record_id", "data_source")
        .agg(pl.col("isdup").any().alias("has_dup"))
        .collect(engine="streaming")
    )
    # Impossible (future) dates from any date field
    dates = (
        base.filter(pl.col("field_type").cast(pl.String).is_in(date_fields) & pl.col("value").is_not_null())
        .with_columns(pl.col("value").str.extract(r"(\d{3,4})").cast(pl.Int32, strict=False).alias("yr"))
        .group_by("record_id", "data_source")
        .agg(pl.col("yr").max().alias("max_yr"), pl.col("yr").is_not_null().any().alias("has_yr"))
        .collect(engine="streaming")
    )
    # Every populated field should be one the model recognises
    schema = (
        base.filter(~ADMIN & ~RELEASE & pl.col("value").is_not_null())
        .with_columns(pl.col("field_type").cast(pl.String).is_in(known_fields).not_().alias("unknown"))
        .group_by("record_id", "data_source")
        .agg(pl.col("unknown").any().alias("has_unknown_field"))
        .collect(engine="streaming")
    )
    # numeric slots must hold numbers, Gt(0) slots positive integers
    posint_fields = sorted(positive_int_field_names())
    num_fields = sorted(numeric_field_names() - positive_int_field_names())
    typed = (
        base.filter(
            pl.col("field_type").cast(pl.String).is_in(num_fields + posint_fields) & pl.col("value").is_not_null()
        )
        .with_columns(
            ok=pl.when(pl.col("field_type").cast(pl.String).is_in(posint_fields))
            .then(pl.col("value").str.strip_chars().str.contains(POSINT_RE))
            .otherwise(pl.col("value").str.strip_chars().str.contains(NUMERIC_RE))
        )
        .group_by("record_id", "data_source")
        .agg(n_typed_applicable=pl.len().cast(pl.Int64), n_typed_ok=pl.col("ok").sum().cast(pl.Int64))
        .collect(engine="streaming")
    )

    df = (
        flags.join(dup, on=["record_id", "data_source"])
        .join(kf, on=["record_id", "data_source"], how="left")
        .join(dates, on=["record_id", "data_source"], how="left")
        .join(schema, on=["record_id", "data_source"], how="left")
        .join(typed, on=["record_id", "data_source"], how="left")
    )

    def norm(c: str) -> pl.Expr:
        return pl.col(c).str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ")

    # (applicable, passes) per check
    title_p, descr_p = pl.col("title").is_not_null(), pl.col("descr").is_not_null()
    checks = {
        "reuse": (title_p, pl.col("title_count") < TITLE_REUSE),
        "near_identical": (title_p & descr_p, norm("title") != norm("descr")),
        "duplicate_stmt": (pl.lit(True), ~pl.col("has_dup")),
        "nondescriptive_title": (
            title_p,
            pl.col("title").str.contains(WORD_RE) & (pl.col("title") != pl.col("objnum").fill_null("")),
        ),
        # a URL-only value in any free-text field, per record
        "uri_only_text": (pl.col("has_text").fill_null(False), ~pl.col("has_uri_text").fill_null(False)),
        "short_descr": (descr_p, pl.col("descr").str.len_chars() >= SHORT_DESC),
        "long_title": (title_p, pl.col("title").str.len_chars() <= LONG_TITLE),
        "impossible_date": (pl.col("has_yr").fill_null(False), pl.col("max_yr").fill_null(0) <= CURRENT_YEAR),
        "html": (pl.lit(True), ~pl.col("has_html")),
        "encoding": (pl.lit(True), ~pl.col("has_enc")),
        "schema_field": (pl.lit(True), ~pl.col("has_unknown_field").fill_null(False)),
    }
    df = df.with_columns(
        [a.cast(pl.Int32).alias(f"_app_{k}") for k, (a, _) in checks.items()]
        + [(a & p).cast(pl.Int32).alias(f"_ok_{k}") for k, (a, p) in checks.items()]
        + [pl.when(a).then(p).otherwise(None).alias(f"pass_{k}") for k, (a, p) in checks.items()]
    )

    # fold pre-computed well-formedness counts onto the rule-based sums
    wf = _load_wellformed(wellformed, data_source_enum(base))
    if wf is not None:
        df = df.join(wf, on=["record_id", "data_source"], how="left")
    df = df.with_columns(
        pl.col("n_wellformed_applicable").fill_null(0)
        if wf is not None
        else pl.lit(0).alias("n_wellformed_applicable"),
        pl.col("n_wellformed_ok").fill_null(0) if wf is not None else pl.lit(0).alias("n_wellformed_ok"),
        pl.col("n_typed_applicable").fill_null(0),
        pl.col("n_typed_ok").fill_null(0),
    )

    return df.with_columns(
        pl.when(pl.col("n_wellformed_applicable") > 0)
        .then(pl.col("n_wellformed_ok") / pl.col("n_wellformed_applicable"))
        .otherwise(None)
        .alias("pass_wellformed"),
        pl.when(pl.col("n_typed_applicable") > 0)
        .then(pl.col("n_typed_ok") / pl.col("n_typed_applicable"))
        .otherwise(None)
        .alias("pass_model_typed"),
        (
            (pl.sum_horizontal(f"_ok_{k}" for k in checks) + pl.col("n_wellformed_ok") + pl.col("n_typed_ok"))
            / (
                pl.sum_horizontal(f"_app_{k}" for k in checks)
                + pl.col("n_wellformed_applicable")
                + pl.col("n_typed_applicable")
            )
        ).alias("conformance"),
    ).select(
        "record_id",
        "data_source",
        "conformance",
        *[f"pass_{k}" for k in checks],
        "pass_wellformed",
        "pass_model_typed",
    )


# Checks whose failure belongs to a value, not the record
CHECK_FIELDS = {"html": HTML_RE, "encoding": ENC_RE}


def failure_breakdown(base: pl.LazyFrame, scored: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(per-check failure counts, the fields whose values drive the value-level checks)"""
    checks = [c.removeprefix("pass_") for c in scored.columns if c.startswith("pass_")]
    per_check = pl.DataFrame(
        [
            {
                "check": check,
                "applicable": int(scored[f"pass_{check}"].is_not_null().sum()),
                # the ratio checks score a share, so anything short fails
                "failed": int((scored[f"pass_{check}"].cast(pl.Float64) < 1.0).sum()),
            }
            for check in checks
        ]
    ).with_columns(fail_rate=pl.col("failed") / pl.col("applicable"))
    # a check with nothing applicable was not run
    per_check = per_check.filter(pl.col("applicable") > 0)

    populated = base.filter(~ADMIN & ~RELEASE & pl.col("value").is_not_null())
    per_field = (
        pl.concat(
            [
                populated.filter(pl.col("value").str.contains(regex))
                .group_by("field_type")
                .agg(values=pl.len())
                .with_columns(check=pl.lit(check))
                for check, regex in CHECK_FIELDS.items()
            ]
        )
        .with_columns(pl.col("field_type").cast(pl.String))
        .sort("check", "values", descending=[False, True])
        .collect(engine="streaming")
    )
    return per_check.sort("fail_rate", descending=True), per_field
