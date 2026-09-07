from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from codecarbon import track_emissions
from mds_data_model.introspection import (
    all_free_text_fields,
    date_fields,
    measurement_fields,
    monetary_fields,
    reference_number_fields,
)

from mds_norm.parsers.parse_certainty import PREFILTER, parse_certainty
from mds_norm.parsers.parse_counts import parse_count
from mds_norm.parsers.parse_dates import PARSER_VERSION as DATE_PARSER_VERSION
from mds_norm.parsers.parse_dates import Conventions, load_periods, parse_date
from mds_norm.parsers.parse_dimensions import (
    AMBIGUOUS_KEYWORDS,
    KEYWORDS,
    PAREN,
    QUALIFIERS,
    TO_BASE,
    UNIT_CLASS,
    UNITS,
    join_slots,
    parse_dimensions,
)
from mds_norm.parsers.parse_dimensions import PARSER_VERSION as DIM_PARSER_VERSION
from mds_norm.paths import (
    COMPILED,
    EMISSIONS_LOG,
    FIELD_STATS,
    INSTITUTIONAL,
    MOJIBAKE_REPAIRS,
    PERSON_ANNOTATIONS,
    PERSON_LINKS,
    PLACE_ANNOTATIONS,
    PLACE_FALLBACK_ANNOTATIONS,
    RAW_RECORDS,
    RECORD_PATCHES,
    VOCAB_ANNOTATIONS,
    VOCABS,
)
from mds_norm.tables import source_only
from mds_norm.utils.atomise import PLACEHOLDER_MARKERS, SEMANTIC_MARKERS
from mds_norm.utils.markup import HTML_ENTITY_RE, HTML_TAG_RE

RAW_PATH = RAW_RECORDS
EMISSIONS_LOG_PATH = EMISSIONS_LOG
VOCAB_ANN = VOCAB_ANNOTATIONS
PERSON_ANN = PERSON_ANNOTATIONS
PERSON_LINK_ANN = PERSON_LINKS
PLACE_ANN = PLACE_ANNOTATIONS
PLACE_FALLBACK_ANN = PLACE_FALLBACK_ANNOTATIONS
PATCHES = RECORD_PATCHES
CONVENTIONS = INSTITUTIONAL / "date_conventions.parquet"
ACCESSION_YEARS = INSTITUTIONAL / "accession_years.parquet"
# The only shape whose century the accession year settles
_TWO_DIGIT_YEAR = r"\b\d{1,2}[./]\d{1,2}[./]\d{2}\b"
PERIODS_CSV = VOCABS / "periods.csv"

PROTECTED = ("spectrum/object_name",)
# Date sub-fields no stage reads; bound slots take ISO points
DATE_BOUNDS = {"spectrum/date_earliest_single": "earliest", "spectrum/date_latest": "latest"}
PERSON_DATE_FIELDS = ("spectrum/persons_birth_date", "spectrum/persons_death_date")
DATE_LIKE = (*DATE_BOUNDS, *PERSON_DATE_FIELDS)
DATE_FIELDS = tuple(date_fields()) + DATE_LIKE
EDTF_FIELD = "wrmthorne/date_edtf"
# A Spectrum date slot holds one ISO point, nothing richer
ISO_POINT = r"^-?\d{4}(-\d{2}(-\d{2})?)?$"
# flatten to the field names across every model
FREE_TEXT = tuple(n for names in all_free_text_fields().values() for n in names)
IDENTIFIERS = tuple(reference_number_fields())
MEASUREMENT = tuple(measurement_fields())
MONETARY = tuple(monetary_fields())

# The node table's own columns, in its order, first
BASE_COLS = [
    "record_id",
    "data_source",
    "node_id",
    "parent_id",
    "depth",
    "source_array_pos",
    "label",
    "path",
    "field_type",
    "value",
    "extra",
]
OUT_COLS = [*BASE_COLS, "as_recorded", "component"]

# Per-node release disposition, most trusted first
DISPOSITION = ("applied", "qualified", "deferred", "enriched", "verified", "untouched")

# Spectrum-native sub-fields only; `display` is not decomposition for a person
PERSON_PARTS = {
    "prefix": "spectrum/persons_title",
    "given": "spectrum/persons_forenames",
    "middle": "spectrum/persons_forenames",
    "nickname": "spectrum/persons_name_notes",
    "surname": "spectrum/persons_surname",
    "suffix": "spectrum/persons_additions_to_name",
}
ORG_PARTS = {"display": "spectrum/organisations_main_body"}
AGENT_PARTS = (*PERSON_PARTS, *ORG_PARTS)  # columns read off the annotations
DIM_FIELD = "spectrum/dimension"
DIM_PARTS = {
    "dimension_value": "spectrum/dimension_value",
    "dimension_measurement_unit": "spectrum/dimension_measurement_unit",
    "dimension_value_qualifier": "spectrum/dimension_value_qualifier",
    "dimension_measured_part": "spectrum/dimension_measured_part",
}
# Ambiguous letters dropped: 'd' is depth or diameter
DIM_TERMS = {k: v for k, v in KEYWORDS.items() if k not in AMBIGUOUS_KEYWORDS}

CERT_PARTS = {
    "kind": "wrmthorne/kind",
    "notation": "wrmthorne/notation",
    "source": "wrmthorne/source",
    "confidence": "wrmthorne/confidence",
    "best_candidate": "wrmthorne/best_candidate",
}

# Notation the date parser strips, kept in the certainty annotation
_DATE_NOTE = r"(?i)\(circa\)|\bcirca\b|\bca?\.|\bapprox\b|\babout\b|~|\?+|\[|\]"

# A bound year beyond this is erroneous as recorded
IMPLAUSIBLE_YEAR = 2026

# Matched after lowercase and strip, so 'n.d.' is stored 'n.d'
DATE_SEMANTIC_MARKERS = SEMANTIC_MARKERS | {
    "n/k",
    "n.k",
    "unknown date",
    "date unknown",
    "date not known",
    "uncertain period",
    "undated",
    "not dated",
    "n.d",
    "no date",
}
_SEMANTIC_NORM = pl.col("base_value").str.to_lowercase().str.strip_chars().str.strip_chars_end(" .")

CERT_SCHEMA = {
    "target_node_id": pl.Binary,
    "record_id": pl.String,
    "data_source": pl.String,
    "kind": pl.String,
    "notation": pl.String,
    "source": pl.String,
    "confidence": pl.Float64,
    "best_candidate": pl.String,
    "detail": pl.String,
}
PROP_SCHEMA = {
    "node_id": pl.Binary,
    "expect_value": pl.String,
    "new_value": pl.String,
    "component": pl.String,
    "tier": pl.Int32,
    "confidence": pl.Float64,
}
CONFLICT_SCHEMA = {"node_id": pl.Binary, "component": pl.String, "reason": pl.String, "detail": pl.String}


def bin16(*exprs: pl.Expr | str) -> pl.Expr:
    """A 16-byte node id derived from the given columns (two 64-bit hashes)"""
    s = pl.struct(*exprs)

    def _combine(batch: pl.Series) -> pl.Series:
        df = batch.struct.unnest()
        arr = np.empty((len(batch), 2), dtype="<u8")
        arr[:, 0] = df["h1"].to_numpy()
        arr[:, 1] = df["h2"].to_numpy()
        buf = arr.tobytes()
        return pl.Series([buf[i * 16 : (i + 1) * 16] for i in range(len(batch))], dtype=pl.Binary)

    return pl.struct(s.hash(seed=101).alias("h1"), s.hash(seed=202).alias("h2")).map_batches(
        _combine, return_dtype=pl.Binary
    )


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# A bare `<1g` has no closing `>` and never matches
_HTML_TAG = HTML_TAG_RE
_HTML = rf"{_HTML_TAG}|{HTML_ENTITY_RE}"
# Phrase text in angle brackets is content, so unwrap it
_BRACKETED_CONTENT = r"^\s*<\s*([a-zA-Z][^<>=/\"]*\s[^<>=/\"]*)>\s*$"


def _unescape(col: pl.Expr) -> pl.Expr:
    """Decode the character entities the compiler knows. `&amp;` goes first to handle doubly-escaped forms"""
    return (
        col.str.replace_all(r"&nbsp;", " ")
        .str.replace_all(r"&amp;", "&")
        .str.replace_all(r"&lt;", "<")
        .str.replace_all(r"&gt;", ">")
        .str.replace_all(r"&quot;", '"')
        .str.replace_all(r"&apos;|&#0*39;", "'")
    )


def _dehtml(col: pl.Expr) -> pl.Expr:
    """Unescape entities and strip HTML tags"""
    text = _unescape(col)
    unwrapped = text.str.extract(_BRACKETED_CONTENT, 1).str.strip_chars()
    cleaned = text.str.replace_all(_HTML_TAG, " ").str.replace_all(r"[ \t]+", " ").str.strip_chars()
    return (
        pl.when(unwrapped.is_not_null())
        .then(unwrapped)
        .when(col.str.contains(_HTML))
        .then(pl.when(cleaned.str.len_chars() > 0).then(cleaned).otherwise(pl.lit(None, dtype=pl.String)))
        .otherwise(col)
    )


# Knowledge-state markers are excluded; they assert something and persist
TIER0_PLACEHOLDERS = PLACEHOLDER_MARKERS | {
    "x",
    # An export artefact standing in for an empty source column
    "no entry in input file",
    "[no entry in input file]",
}
# An empty form template may spell each label's unit
_DIM_TEMPLATE = (
    r"(?i)^\s*(?:(?:length|width|height|depth|diameter|breadth|"
    r"thickness|weight|circumference|diam)\s*(?:/\s*[a-z]{1,4}\s*)?"
    r":\s*)+$|^\s*measures needed\s*:?\s*$"
)
# A value holding only a unit token records no measurement
_UNIT_LEAK_FIELD = "spectrum/dimension_value"

# '-' and '/' absent: BCE years and identifier structure
_DANGLING = " \t\r\n,;|:"
# A label with no number, in number-carrying fields only
_LABEL_ONLY = r"^[^0-9]*:+\s*$"
NUMBER_CARRYING = (*IDENTIFIERS, DIM_FIELD, _UNIT_LEAK_FIELD)
# Manual-only and free-text fields take no strip, only feedback
MANUAL_ONLY = ("spectrum/object_name_type", "spectrum/other_number_type")
NO_EDIT = (*PROTECTED, *MANUAL_ONLY, *FREE_TEXT)


def _strip_dangling(col: pl.Expr) -> pl.Expr:
    """Trim orphaned separators off either end; reads field_type"""
    return pl.when(pl.col("field_type").is_in(NO_EDIT)).then(col).otherwise(col.str.strip_chars(_DANGLING))


def _placeholder(col: pl.Expr) -> pl.Expr:
    norm = col.str.strip_chars().str.to_lowercase()
    return (norm.is_in(sorted(TIER0_PLACEHOLDERS)) | norm.str.contains(_DIM_TEMPLATE)).fill_null(False)


def _echo_norm(col: pl.Expr) -> pl.Expr:
    return col.str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()


def _field_name_echo(col: pl.Expr) -> pl.Expr:
    """Whole value that just names its own field; reads label and field_type"""
    v = _echo_norm(col)
    return (
        (v == _echo_norm(pl.col("label"))) | (v == _echo_norm(pl.col("field_type").str.split("/").list.last()))
    ).fill_null(False)


def _label_only(col: pl.Expr) -> pl.Expr:
    """Whole-value label in a number-carrying field; reads field_type"""
    return (pl.col("field_type").is_in(NUMBER_CARRYING) & col.str.contains(_LABEL_ONLY)).fill_null(False)


def _unit_leak(col: pl.Expr) -> pl.Expr:
    """A dimension_value carrying no digit; reads field_type"""
    return ((pl.col("field_type") == _UNIT_LEAK_FIELD) & ~col.str.contains(r"\d")).fill_null(False)


def screen_generated(part: pl.LazyFrame) -> pl.LazyFrame:
    """Apply the tier-0 no-content tests to nodes a later stage generated"""
    release = pl.col("field_type").str.starts_with("wrmthorne/")
    stripped = _strip_dangling(pl.col("value"))
    return part.with_columns(value=pl.when(release).then(pl.col("value")).otherwise(stripped)).filter(
        release
        | pl.col("value").is_null()
        | ~(
            _placeholder(pl.col("value"))
            | _unit_leak(pl.col("value"))
            | _label_only(pl.col("value"))
            | _field_name_echo(pl.col("value"))
            | (pl.col("value").str.len_chars() == 0)
        )
    )


def build_base(tmp_path: Path, sources: list[str] | None) -> pl.LazyFrame:
    """Tier-0-applied node table: raw columns plus base_value and contains_url"""
    raw = pl.scan_parquet(RAW_PATH)
    if sources:
        raw = raw.filter(pl.col("data_source").is_in(sources))
    fs = pl.scan_parquet(FIELD_STATS).select("node_id", pl.col("value").alias("value_t0"), "contains_url")
    # Unattested damage keeps its value and its encoding_damage annotation
    repairs = (
        pl.scan_parquet(MOJIBAKE_REPAIRS).select(
            pl.col("value").alias("value_t0"), pl.col("repaired").alias("value_fixed")
        )
        if MOJIBAKE_REPAIRS.exists()
        else pl.LazyFrame(schema={"value_t0": pl.String, "value_fixed": pl.String})
    )
    # Materialised so the no-content tests re-read instead of re-evaluating
    cleaned = _dehtml(pl.coalesce("value_fixed", "value_t0", "value"))
    c, s = pl.col("_cleaned"), pl.col("_stripped")
    (
        raw.join(fs, on="node_id", how="left", maintain_order="left")
        .join(repairs, on="value_t0", how="left", maintain_order="left")
        .with_columns(_cleaned=cleaned)
        # The tests read the unstripped value too, so strip last
        .with_columns(_stripped=_strip_dangling(c))
        .with_columns(
            base_value=pl.when(pl.col("field_type").is_in(PROTECTED))
            .then(pl.col("value"))
            .when(
                _placeholder(c)
                | _placeholder(s)
                | _unit_leak(c)
                | _unit_leak(s)
                | _label_only(c)
                | _field_name_echo(s)
                # separators all the way down: null, never the empty string
                | (s.str.len_chars() == 0).fill_null(False)
            )
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(s),
            contains_url=pl.col("contains_url").fill_null(False),
        )
        .drop("value_t0", "value_fixed", "_cleaned", "_stripped")
        .sink_parquet(tmp_path)
    )
    return pl.scan_parquet(tmp_path)


def accession_ceilings() -> pl.LazyFrame:
    """record_id -> the latest year a two-digit production year may complete to"""
    schema = {"record_id": pl.String, "ceiling": pl.Int32}
    if not ACCESSION_YEARS.exists():
        log(f"dates: no accession years at {ACCESSION_YEARS} — two-digit years keep their refusal")
        return pl.LazyFrame(schema=schema)
    # The accession year is the ceiling; the earliest is tightest
    return (
        pl.scan_parquet(ACCESSION_YEARS)
        .group_by("record_id")
        .agg(ceiling=pl.col("accession_year").min().cast(pl.Int32))
    )


def load_conventions() -> pl.DataFrame:
    """Per-institution date conventions, exported by institutional_priors.py"""
    schema = {"data_source": pl.String, "dm_order": pl.String, "eq": pl.Boolean, "z0": pl.Boolean}
    if not CONVENTIONS.exists():
        log(f"dates: no conventions at {CONVENTIONS} — every institution reads under the corpus default")
        return pl.DataFrame(schema=schema)
    return pl.read_parquet(CONVENTIONS).select(
        "data_source",
        dm_order=pl.col("dm_order").fill_null(""),
        eq=pl.col("eq_means_range").fill_null(False),
        z0=pl.col("zero_placeholder").fill_null(False),
    )


def compile_dates(
    base: pl.LazyFrame, cache_path: Path, report: dict
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.LazyFrame]:
    """Tier-1 date parse over distinct (value, conventions) into in-place EDTF proposals"""
    conv = load_conventions()
    conv_keys = ["dm_order", "eq", "z0", "ceiling"]
    dnodes = (
        base.filter(pl.col("field_type").is_in(DATE_FIELDS) & pl.col("base_value").is_not_null())
        .with_columns(pl.col("data_source").cast(pl.String).alias("_ds"))
        .join(conv.lazy().rename({"data_source": "_ds"}), on="_ds", how="left")
        .join(accession_ceilings(), on="record_id", how="left")
        .with_columns(
            dm_order=pl.col("dm_order").fill_null(""),
            eq=pl.col("eq").fill_null(False),
            z0=pl.col("z0").fill_null(False),
            # 0 stands for "no ceiling"; a null would never join
            ceiling=pl.when(pl.col("base_value").str.contains(_TWO_DIGIT_YEAR))
            .then(pl.col("ceiling").fill_null(0))
            .otherwise(pl.lit(0, dtype=pl.Int32))
            .cast(pl.Int32),
        )
    )
    # A knowledge-state marker is not a date and never parses
    semantic = (
        dnodes.filter(_SEMANTIC_NORM.is_in(sorted(DATE_SEMANTIC_MARKERS)))
        .select("node_id")
        .collect(engine="streaming")
    )
    dnodes = dnodes.filter(~_SEMANTIC_NORM.is_in(sorted(DATE_SEMANTIC_MARKERS)))
    log(f"dates: {semantic.height:,} semantic-marker values routed to deferred")
    vals = dnodes.select(pl.col("base_value").alias("value"), *conv_keys).unique().collect(engine="streaming")
    log(f"dates: {vals.height:,} distinct (value, conventions) ({conv.height} institutions with priors)")

    cache_schema = {
        "value": pl.String,
        "dm_order": pl.String,
        "eq": pl.Boolean,
        "z0": pl.Boolean,
        "ceiling": pl.Int32,
        "edtf": pl.String,
        "earliest": pl.String,
        "latest": pl.String,
        "cert": pl.String,
        "e_qual": pl.String,
        "l_qual": pl.String,
        "dm_ambig": pl.Boolean,
        "period": pl.String,
        "pver": pl.String,
    }
    cached = pl.read_parquet(cache_path) if cache_path.exists() else pl.DataFrame(schema=cache_schema)
    if set(cached.columns) != set(cache_schema):
        log("dates: cache pre-dates its current key set; reparsing from scratch")
        cached = pl.DataFrame(schema=cache_schema)
    # rows from an older parser reparse
    cached = cached.filter(pl.col("pver") == DATE_PARSER_VERSION)
    todo = vals.join(cached, on=["value", *conv_keys], how="anti")
    if todo.height:
        rows = []
        for v, o, e, z, ceiling in todo.select("value", *conv_keys).iter_rows():
            try:
                p = parse_date(
                    v, Conventions(dm_order=o or None, eq_range=e, zero_null=z, century_ceiling=ceiling or None)
                )
            except Exception:
                p = None
            edtf = p.get("value_edtf") if p else None
            earliest = p.get("date_earliest_single") if p else None
            cert = (p.get("date_earliest_single_certainty") or p.get("date_latest_certainty")) if p else None
            rows.append(
                (
                    v,
                    o,
                    e,
                    z,
                    ceiling,
                    edtf,
                    earliest,
                    (p.get("date_latest") or earliest) if p else None,
                    cert,
                    p.get("date_earliest_single_qualifier") if p else None,
                    p.get("date_latest_qualifier") if p else None,
                    bool(p and p.get("dm_ambiguous")),
                    p.get("date_period") if p else None,
                    DATE_PARSER_VERSION,
                )
            )
        cached = pl.concat([cached, pl.DataFrame(rows, schema=cache_schema, orient="row")])
        cached.write_parquet(cache_path)
    log(f"dates: parsed {todo.height:,} new values ({cached['edtf'].is_not_null().sum():,}/{cached.height:,} parse)")

    # Only an ISO point is written; ranges stay as recorded
    written = (
        pl.when(pl.col("field_type") == "spectrum/date_earliest_single")
        .then(pl.col("earliest"))
        .when(pl.col("field_type") == "spectrum/date_latest")
        .then(pl.col("latest"))
        .when(pl.col("edtf").str.contains(ISO_POINT))
        .then(pl.col("edtf"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    parsed = dnodes.join(
        cached.lazy(), left_on=["base_value", *conv_keys], right_on=["value", *conv_keys]
    ).with_columns(written=written)
    # fill_null because a signed BCE year is not future
    future = pl.col("field_type").is_in(list(DATE_BOUNDS)) & (
        pl.col("written").str.extract(r"^(\d{3,4})").cast(pl.Int32, strict=False) > IMPLAUSIBLE_YEAR
    ).fill_null(False)
    implausible = (
        parsed.filter(pl.col("edtf").is_not_null() & future)
        .select(
            target_node_id="node_id",
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            kind=pl.lit("implausible_date"),
            notation=pl.lit(None, dtype=pl.String),
            source=pl.lit("dates"),
            confidence=pl.lit(None, dtype=pl.Float64),
            best_candidate=pl.lit(None, dtype=pl.String),
            detail=pl.lit("bound year is in the future; likely an unsigned BCE year or a typo"),
        )
        .collect(engine="streaming")
    )
    proposals = (
        parsed.filter(pl.col("written").is_not_null() & ~future & (pl.col("written") != pl.col("base_value")))
        .select(
            node_id="node_id",
            expect_value=pl.lit(None, dtype=pl.String),
            new_value="written",
            component=pl.lit("dates"),
            tier=pl.lit(1, dtype=pl.Int32),
            confidence=pl.lit(1.0),
        )
        .collect(engine="streaming")
    )
    # Read and confirmed parseable, so `verified` rather than `untouched`
    verified = (
        parsed.filter(pl.col("written").is_not_null() & ~future & (pl.col("written") == pl.col("base_value")))
        .select("node_id")
        .collect(engine="streaming")
    )
    certainty = (
        parsed.filter(pl.col("edtf").is_not_null() & pl.col("cert").is_not_null())
        .select(
            target_node_id="node_id",
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            kind=pl.lit("cataloguer_marked"),
            notation=pl.col("base_value").str.extract(_DATE_NOTE, 0),
            source=pl.lit("cataloguer"),
            confidence=pl.lit(None, dtype=pl.Float64),
            best_candidate=pl.lit(None, dtype=pl.String),
            detail=pl.col("cert"),
        )
        .collect(engine="streaming")
    )
    ambiguous = (
        parsed.filter(pl.col("edtf").is_not_null() & pl.col("dm_ambig"))
        .select(
            target_node_id="node_id",
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            kind=pl.lit("ambiguous_dm"),
            notation=pl.lit(None, dtype=pl.String),
            source=pl.lit("dates"),
            confidence=pl.lit(None, dtype=pl.Float64),
            best_candidate=pl.lit(None, dtype=pl.String),
            detail=pl.lit("day-first assumed; institution has no day/month order prior"),
        )
        .collect(engine="streaming")
    )
    period_nodes = date_period_nodes(parsed, base)
    bound_nodes = date_substructure(parsed, base)
    n_bound = bound_nodes.select(pl.len()).collect(engine="streaming").item()
    log(f"dates: {n_bound:,} substructure nodes (bound children + native certainty/qualifier slots)")
    edtf_nodes = date_edtf_nodes(parsed, base)
    n_edtf = edtf_nodes.select(pl.len()).collect(engine="streaming").item()
    log(f"dates: {n_edtf:,} EDTF nodes")
    report["dates"] = {
        "distinct_values": vals.height,
        "proposals": proposals.height,
        "certainty": certainty.height,
        "ambiguous_dm": ambiguous.height,
        "implausible_future": implausible.height,
        "period_nodes": period_nodes.height,
        "substructure_nodes": n_bound,
        "edtf_nodes": n_edtf,
        "semantic_markers": semantic.height,
        "verified": verified.height,
    }
    return (
        proposals,
        pl.concat([certainty, ambiguous, implausible]),
        period_nodes,
        semantic,
        verified,
        bound_nodes,
        edtf_nodes,
    )


# Only group fields; a bound slot is itself a child
PERIOD_GROUP_FIELDS = tuple(date_fields())


def date_period_nodes(parsed: pl.LazyFrame, base: pl.LazyFrame) -> pl.DataFrame:
    """One spectrum/date_period child per date-group node carrying a recognised period label"""
    has_period_child = (
        base.filter(pl.col("field_type") == "spectrum/date_period")
        .select(pl.col("parent_id").alias("node_id"))
        .unique()
        .drop_nulls()
    )
    return (
        parsed.filter(pl.col("period").is_not_null() & pl.col("field_type").is_in(PERIOD_GROUP_FIELDS))
        .join(has_period_child, on="node_id", how="anti")
        .select(
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            node_id=bin16("node_id", pl.lit("date_period")),
            parent_id="node_id",
            depth=(pl.col("depth") + 1).cast(pl.UInt8),
            label=pl.lit("Date Period"),
            path=pl.lit(None, dtype=pl.String),
            field_type=pl.lit("spectrum/date_period"),
            value="period",
            as_recorded=pl.lit(None, dtype=pl.String),
            component=pl.lit("dates"),
        )
        .collect(engine="streaming")
    )


def date_edtf_nodes(parsed: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """One wrmthorne/date_edtf child per parsed date-group node, carrying the whole expression"""
    has_edtf_child = (
        base.filter(pl.col("field_type") == EDTF_FIELD)
        .select(pl.col("parent_id").alias("node_id"))
        .unique()
        .drop_nulls()
    )
    return (
        parsed.filter(pl.col("edtf").is_not_null() & pl.col("field_type").is_in(PERIOD_GROUP_FIELDS))
        .join(has_edtf_child, on="node_id", how="anti")
        .select(
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            node_id=bin16("node_id", pl.lit("date_edtf")),
            parent_id="node_id",
            depth=(pl.col("depth") + 1).cast(pl.UInt8),
            label=pl.lit("Date EDTF"),
            path=pl.lit(None, dtype=pl.String),
            field_type=pl.lit(EDTF_FIELD),
            value="edtf",
            as_recorded=pl.lit(None, dtype=pl.String),
            component=pl.lit("dates"),
        )
    )


# Native per-bound slots are populated beside the wrmthorne/certainty channel
BOUND_CHILD_FIELDS = {"earliest": "spectrum/date_earliest_single", "latest": "spectrum/date_latest"}
BOUND_SUB_FIELDS = {
    "spectrum/date_earliest_single": (
        "spectrum/date_earliest_single_certainty",
        "spectrum/date_earliest_single_qualifier",
    ),
    "spectrum/date_latest": ("spectrum/date_latest_certainty", "spectrum/date_latest_qualifier"),
}


def _flabel(field: str) -> str:
    return field.split("/")[1].replace("_", " ").title()


def _bound_year_ok(col: str) -> pl.Expr:
    return pl.col(col).str.extract(r"^-?(\d{3,4})").cast(pl.Int32, strict=False).fill_null(0) <= IMPLAUSIBLE_YEAR


def date_substructure(parsed: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """Spectrum substructure for parsed date values: bound children and native certainty slots"""
    out_schema = {
        "record_id": pl.String,
        "data_source": pl.String,
        "node_id": pl.Binary,
        "parent_id": pl.Binary,
        "depth": pl.UInt8,
        "label": pl.String,
        "path": pl.String,
        "field_type": pl.String,
        "value": pl.String,
        "as_recorded": pl.String,
        "component": pl.String,
    }

    def rows(
        sel: pl.LazyFrame, node: pl.Expr, parent: pl.Expr, rel_depth: int, field: str, value: pl.Expr
    ) -> pl.LazyFrame:
        return (
            sel.filter(value.is_not_null())
            .select(
                record_id="record_id",
                data_source=pl.col("data_source").cast(pl.String),
                node_id=node,
                parent_id=parent,
                depth=(pl.col("depth") + rel_depth).cast(pl.UInt8),
                label=pl.lit(_flabel(field)),
                path=pl.lit(None, dtype=pl.String),
                field_type=pl.lit(field),
                value=value,
                as_recorded=pl.lit(None, dtype=pl.String),
                component=pl.lit("dates"),
            )
            .cast(out_schema)
        )

    parts: list[pl.LazyFrame] = []

    has_bound = (
        base.filter(pl.col("field_type").is_in(list(DATE_BOUNDS)))
        .select(pl.col("parent_id").alias("node_id"))
        .unique()
        .drop_nulls()
    )
    groups = parsed.filter(
        pl.col("edtf").is_not_null()
        & pl.col("field_type").is_in(PERIOD_GROUP_FIELDS)
        & _bound_year_ok("earliest")
        & _bound_year_ok("latest")
    ).join(has_bound, on="node_id", how="anti")
    # The cache copies a point's earliest into `latest`
    is_range = pl.col("latest").is_not_null() & (
        pl.col("earliest").is_null() | (pl.col("latest") != pl.col("earliest"))
    )
    for which, fld in BOUND_CHILD_FIELDS.items():
        val = pl.col("earliest" if which == "earliest" else "latest")
        qual = pl.col("e_qual" if which == "earliest" else "l_qual")
        sel = groups.filter(val.is_not_null() if which == "earliest" else is_range)
        cid = bin16("node_id", pl.lit(f"date_bound_{which}"))
        parts.append(rows(sel, cid, pl.col("node_id"), 1, fld, val))
        cert_f, qual_f = BOUND_SUB_FIELDS[fld]
        for sub_f, sub_val, tag in ((cert_f, pl.col("cert"), "cert"), (qual_f, qual, "qual")):
            parts.append(rows(sel, bin16("node_id", pl.lit(f"date_bound_{which}_{tag}")), cid, 2, sub_f, sub_val))

    sub_fields = [f for pair in BOUND_SUB_FIELDS.values() for f in pair]
    has_sub = (
        base.filter(pl.col("field_type").is_in(sub_fields))
        .select(pl.col("parent_id").alias("node_id"))
        .unique()
        .drop_nulls()
    )
    slots = parsed.filter(pl.col("edtf").is_not_null() & pl.col("field_type").is_in(list(DATE_BOUNDS))).join(
        has_sub, on="node_id", how="anti"
    )
    for fld, which in DATE_BOUNDS.items():
        # a point in either slot carries the one qualifier
        qual = pl.coalesce("e_qual", "l_qual") if which == "earliest" else pl.coalesce("l_qual", "e_qual")
        sel = slots.filter(pl.col("field_type") == fld)
        cert_f, qual_f = BOUND_SUB_FIELDS[fld]
        for sub_f, sub_val, tag in ((cert_f, pl.col("cert"), "cert"), (qual_f, qual, "qual")):
            parts.append(
                rows(sel, bin16("node_id", pl.lit(f"bound_native_{tag}")), pl.col("node_id"), 1, sub_f, sub_val)
            )

    return pl.concat(parts)


ASSOC_DATE = "spectrum/associated_date"
PROD_DATE_FIELD = "spectrum/object_production_date"
# Ambiguous and non-production association labels are left alone
PRODUCTION_ASSOCIATIONS = {
    "made",
    "creation",
    "execution",
    "production",
    "produced",
    "manufactured",
    "made and designed",
    "date of creation",
    "date made",
    "year of production",
    "date of production",
}


def compile_associated_dates(base: pl.LazyFrame, report: dict) -> pl.DataFrame:
    """Retype an associated_date whose date_association child states the association is production"""
    assoc = base.filter(
        (pl.col("field_type") == "spectrum/date_association") & pl.col("base_value").is_not_null()
    ).select(pl.col("parent_id").alias("node_id"), assoc=pl.col("base_value").str.to_lowercase().str.strip_chars(" ."))
    unanimous = (
        assoc.group_by("node_id")
        .agg(n=pl.col("assoc").n_unique(), assoc=pl.col("assoc").first())
        .filter((pl.col("n") == 1) & pl.col("assoc").is_in(sorted(PRODUCTION_ASSOCIATIONS)))
    )
    retypes = (
        base.filter(pl.col("field_type") == ASSOC_DATE)
        .join(unanimous.select("node_id"), on="node_id", how="semi")
        .select(
            "node_id",
            retype_field=pl.lit(PROD_DATE_FIELD),
            retype_label=pl.lit("Object Production Date"),
            retype_component=pl.lit("associated_dates"),
        )
        .collect(engine="streaming")
    )
    report["associated_dates"] = {"retyped_nodes": retypes.height}
    log(f"associated dates: {retypes.height:,} nodes retyped to object_production_date")
    return retypes


MEASURED_PART_FIELD = "spectrum/dimension_measured_part"
QUALIFIER_FIELD = "spectrum/dimension_value_qualifier"


def compile_measured_part_qualifiers(base: pl.LazyFrame, report: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Retype a whole-value qualifier sitting in dimension_measured_part"""
    canon = (
        pl.col("base_value")
        .str.strip_chars()
        .str.to_lowercase()
        .str.strip_chars_end(" .")
        .replace_strict(QUALIFIERS, default=None)
    )
    has_qual = (
        base.filter(pl.col("field_type") == QUALIFIER_FIELD)
        .select(pl.col("parent_id").alias("_gid"))
        .unique()
        .drop_nulls()
    )
    nodes = (
        base.filter((pl.col("field_type") == MEASURED_PART_FIELD) & pl.col("base_value").is_not_null())
        .with_columns(canon=canon)
        .filter(pl.col("canon").is_not_null())
        .join(has_qual, left_on="parent_id", right_on="_gid", how="anti")
        .select("node_id", "base_value", "canon")
        .collect(engine="streaming")
    )
    retypes = nodes.select(
        "node_id",
        retype_field=pl.lit(QUALIFIER_FIELD),
        retype_label=pl.lit("Dimension Value Qualifier"),
        retype_component=pl.lit("dimensions"),
    )
    proposals = nodes.filter(pl.col("canon") != pl.col("base_value")).select(
        node_id="node_id",
        expect_value="base_value",
        new_value="canon",
        component=pl.lit("dimensions"),
        tier=pl.lit(1, dtype=pl.Int32),
        confidence=pl.lit(1.0),
    )
    report["measured_part_qualifiers"] = {"retyped_nodes": retypes.height, "canonicalised": proposals.height}
    log(f"measured parts: {retypes.height:,} qualifier values retyped to dimension_value_qualifier")
    return retypes, proposals


# The closed shapes only
ADMIN_CONCEPT_FIELD = "spectrum/associated_concept"
ADMIN_NOTE_FIELD = "spectrum/comments"
ADMIN_CONCEPT_RX = (
    r"(?i)^\s*(?:record verified by\b.+|(?:not\s+)?verified|"
    r"legal status to be verified)\s*$"
)


def compile_admin_concepts(base: pl.LazyFrame, report: dict) -> pl.DataFrame:
    retypes = (
        base.filter(
            (pl.col("field_type") == ADMIN_CONCEPT_FIELD) & pl.col("base_value").str.contains(ADMIN_CONCEPT_RX)
        )
        .select(
            "node_id",
            retype_field=pl.lit(ADMIN_NOTE_FIELD),
            retype_label=pl.lit("Comments"),
            retype_component=pl.lit("admin_note_routing"),
        )
        .collect(engine="streaming")
    )
    report["admin_note_routing"] = {"retyped_nodes": retypes.height}
    log(f"admin concepts: {retypes.height:,} verification notes retyped to comments")
    return retypes


BIRTH_DATE = "spectrum/persons_birth_date"
DEATH_DATE = "spectrum/persons_death_date"
# an activity or reign range is not a lifespan
_ACTIVITY = re.compile(r"(?i)\bactive\b|\bfl\.?\b|floruit|\breign|\bruled\b")
_DASH_SPACED = re.compile(r"\s+[-–—]\s+")
_DASH_TIGHT = re.compile(r"(?<=\d)[-–—]")
# A side that is a closed range is not one
_CLOSED_RANGE = re.compile(r"(?<!\.)/(?!\.)")
MIN_LIFESPAN = 15  # a shorter gap reads as an activity/reign range
MAX_LIFESPAN = 110
# a lifespan has two sides; more parts is not one
LIFESPAN_SIDES = 2
# The parser leaves the verb; the destination decision belongs here
_DEATH_VERB = re.compile(r"(?i)^\s*(?:died|d\.)\s*")


def _iso_year(iso: str | None) -> int | None:
    if not iso:
        return None
    neg = iso.startswith("-")
    y = int(iso.lstrip("-").split("-")[0])
    return -y if neg else y


def _split_lifespan(raw: str) -> tuple[str, str] | None:
    """Split a dual birth-death expression into its two sides, or None"""
    if _ACTIVITY.search(raw):
        return None
    for rx in (_DASH_SPACED, _DASH_TIGHT):
        parts = rx.split(raw)
        if len(parts) == LIFESPAN_SIDES and all(re.search(r"\d", p) for p in parts):
            return parts[0].strip(), parts[1].strip()
    return None


def parse_lifespan(raw: str) -> tuple[str, str] | None:
    """(birth EDTF, death EDTF) where raw reads as two independent life dates, else None"""
    sides = _split_lifespan(raw)
    if sides is None:
        return None
    edtfs, years = [], []
    for side in sides:
        try:
            p = parse_date(side)
        except Exception:
            return None
        edtf = p.get("value_edtf") if p else None
        if not edtf or _CLOSED_RANGE.search(edtf):
            return None
        y = _iso_year(p.get("date_earliest_single") or p.get("date_latest"))
        if y is None:
            return None
        edtfs.append(edtf)
        years.append(y)
    if not MIN_LIFESPAN <= years[1] - years[0] <= MAX_LIFESPAN:
        return None
    return edtfs[0], edtfs[1]


def parse_death_only(raw: str) -> str | None:
    """EDTF where a birth-date slot holds nothing but a death date ('died 1831'), else None"""
    rest = _DEATH_VERB.sub("", raw, count=1)
    if rest == raw or not re.search(r"\d", rest):
        return None
    try:
        p = parse_date(rest)
    except Exception:
        return None
    edtf = p.get("value_edtf") if p else None
    return edtf if edtf and not _CLOSED_RANGE.search(edtf) else None


def compile_person_dates(base: pl.LazyFrame, report: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split a persons_birth_date holding a lifespan, and retype one holding only a death date"""
    dd = base.filter(pl.col("field_type") == DEATH_DATE).select(pl.col("parent_id").alias("pid")).unique().drop_nulls()
    births = base.filter(pl.col("field_type") == BIRTH_DATE).join(dd, left_on="parent_id", right_on="pid", how="anti")
    cands = births.filter(pl.col("base_value").str.contains(r"\d.*[-–—].*\d"))
    vals = cands.select(pl.col("base_value").alias("value")).unique().collect(engine="streaming")
    rows = [(v, *s) for v in vals["value"] if (s := parse_lifespan(v))]
    parsed = pl.DataFrame(rows, orient="row", schema={"value": pl.String, "birth": pl.String, "death": pl.String})
    matched = cands.join(parsed.lazy(), left_on="base_value", right_on="value").collect(engine="streaming")

    proposals = matched.select(
        node_id="node_id",
        expect_value="base_value",
        new_value="birth",
        component=pl.lit("person_dates"),
        tier=pl.lit(1, dtype=pl.Int32),
        confidence=pl.lit(1.0),
    )
    death_nodes = matched.select(
        record_id="record_id",
        data_source=pl.col("data_source").cast(pl.String),
        node_id=bin16("node_id", pl.lit("person_death_split")),
        parent_id="parent_id",
        depth=pl.col("depth").cast(pl.UInt8),
        label=pl.lit("Persons Death Date"),
        path=pl.lit(None, dtype=pl.String),
        field_type=pl.lit(DEATH_DATE),
        value="death",
        as_recorded="base_value",
        component=pl.lit("person_dates"),
    )
    # The year is right and the destination wrong, so retype
    death_only = births.filter(pl.col("base_value").str.contains(r"(?i)^\s*(?:died|d\.)"))
    dvals = death_only.select(pl.col("base_value").alias("value")).unique().collect(engine="streaming")
    drows = [(v, e) for v in dvals["value"] if (e := parse_death_only(v))]
    dparsed = pl.DataFrame(drows, orient="row", schema={"value": pl.String, "edtf": pl.String})
    dmatched = death_only.join(dparsed.lazy(), left_on="base_value", right_on="value").collect(engine="streaming")

    death_props = dmatched.select(
        node_id="node_id",
        expect_value="base_value",
        new_value="edtf",
        component=pl.lit("person_dates"),
        tier=pl.lit(1, dtype=pl.Int32),
        confidence=pl.lit(1.0),
    )
    death_retypes = dmatched.select(
        "node_id",
        retype_field=pl.lit(DEATH_DATE),
        retype_label=pl.lit("Persons Death Date"),
        retype_component=pl.lit("person_dates"),
    )
    report["person_dates"] = {
        "candidates": vals.height,
        "split_values": parsed.height,
        "split_nodes": proposals.height,
        "death_only_values": dparsed.height,
        "death_only_nodes": death_retypes.height,
    }
    log(f"person dates: {proposals.height:,} birth nodes split into birth + death")
    log(f"person dates: {death_retypes.height:,} birth nodes retyped to death date")
    return pl.concat([proposals, death_props]), death_nodes, death_retypes


def _replace_spans(value: str, atoms: list[dict]) -> str | None:
    """Replace each resolved atom span with its matched term; None on overlap"""
    spans = sorted(atoms, key=lambda a: a["span_start"])
    out, cursor = [], 0
    for a in spans:
        s, e = a["span_start"], a["span_end"]
        if s < cursor or e > len(value):
            return None
        out.append(value[cursor:s])
        out.append(a["matched_term"])
        cursor = e
    out.append(value[cursor:])
    return "".join(out)


# Standing annotations here are filtered until the cascade is rerun
NON_VOCAB_FIELDS = ("spectrum/dimension_measured_part",)
NO_SPLIT_VOCAB_FIELDS = ("spectrum/associated_concept",)


def _filtered_annotations(ann_path: Path) -> pl.LazyFrame:
    """The vocabulary and places annotation sidecar, minus the rows compile must ignore"""
    fragment = (pl.col("span_start") > 0) | (pl.col("span_end") < pl.col("value").str.len_chars())
    placeholder_term = (
        pl.col("matched_term").str.strip_chars().str.to_lowercase().is_in(sorted(PLACEHOLDER_MARKERS)).fill_null(False)
    )
    return (
        pl.scan_parquet(ann_path)
        .with_columns(pl.col("data_source").cast(pl.String))
        .filter(
            ~pl.col("field_type").is_in(NON_VOCAB_FIELDS)
            & ~(pl.col("field_type").is_in(NO_SPLIT_VOCAB_FIELDS) & fragment.fill_null(False))
            & ~placeholder_term
        )
    )


def compile_vocab(
    report: dict,
    ann_path: Path = VOCAB_ANN,
    component: str = "vocab_alignment",
    certainty_kind: str = "ambiguous_homograph",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Resolved atoms rewrite the value in place; flagged atoms emit ambiguity certainty rows"""
    va = _filtered_annotations(ann_path)
    res = va.filter(pl.col("status") == "resolved")

    uniq = (
        res.unique(subset=["data_source", "group", "value", "span_start", "span_end"])
        .group_by("data_source", "group", "value")
        .agg(
            pl.struct("span_start", "span_end", "matched_term").alias("atoms"),
            pl.col("tier").min().alias("tier"),
            pl.col("confidence").min().alias("confidence"),
        )
        .collect(engine="streaming")
    )
    log(f"{component}: {uniq.height:,} distinct resolved values")

    new_vals, bad = [], 0
    for value, atoms in zip(uniq["value"], uniq["atoms"], strict=True):
        nv = _replace_spans(value, atoms)
        bad += nv is None
        new_vals.append(nv)
    uniq = uniq.with_columns(new_value=pl.Series(new_vals, dtype=pl.String)).filter(
        pl.col("new_value").is_not_null() & (pl.col("new_value") != pl.col("value"))
    )

    nodes = (
        res.unique(subset=["node_id"]).select("node_id", "data_source", "group", "value").collect(engine="streaming")
    )
    proposals = nodes.join(uniq, on=["data_source", "group", "value"], how="inner").select(
        node_id="node_id",
        expect_value="value",
        new_value="new_value",
        component=pl.lit(component),
        tier=pl.col("tier").cast(pl.Int32),
        confidence="confidence",
    )

    certainty = (
        va.filter(pl.col("status") == "flagged")
        .select(
            target_node_id="node_id",
            record_id="record_id",
            data_source="data_source",
            kind=pl.lit(certainty_kind),
            notation=pl.lit(None, dtype=pl.String),
            source=pl.lit(component),
            confidence="score",
            best_candidate="matched_term",
            detail=pl.format("{}:{}", "vocab", "subject"),
        )
        .collect(engine="streaming")
    )
    report[component] = {
        "distinct_resolved_values": uniq.height,
        "overlapping_span_values": bad,
        "proposals": proposals.height,
        "certainty": certainty.height,
    }
    return proposals, certainty, nodes.select("node_id")


AUTH_FIELD = "wrmthorne/authority"
AUTH_PARTS = {
    "authority_source": "wrmthorne/authority_source",
    "authority_id": "wrmthorne/authority_id",
    "authority_uri": "wrmthorne/authority_uri",
}
# uri is prefix plus id; local termlists have no prefix
AUTH_URI_PREFIXES = {
    "aat": "http://vocab.getty.edu/aat/",
    "tgn": "http://vocab.getty.edu/tgn/",
    "ulan": "http://vocab.getty.edu/ulan/",
    "isni": "https://isni.org/isni/",
    "periodo": "https://n2t.net/ark:/99152/",
    "fish_event_types": "http://purl.org/heritagedata/schemes/agl_et/concepts/",
    "fish_building_materials": "http://purl.org/heritagedata/schemes/eh_tbm/concepts/",
    "fish_archaeological_objects": "http://purl.org/heritagedata/schemes/mda_obj/concepts/",
    "geonames": "https://www.geonames.org/",
    "os_open_names": "http://data.ordnancesurvey.co.uk/id/",
}


def _authority_links(ann: pl.LazyFrame) -> list[pl.LazyFrame]:
    """The alignment's own link, plus the crosswalk a house-vocabulary hit keeps beside it"""
    resolved = ann.filter(pl.col("status") == "resolved")
    links = [
        resolved.filter(pl.col("vocab").is_not_null() & pl.col("subject").is_not_null()).select(
            "node_id", "record_id", "data_source", "vocab", "subject"
        )
    ]
    if "xref_vocab" in ann.collect_schema().names():
        links.append(
            resolved.filter(pl.col("xref_vocab").is_not_null() & pl.col("xref_subject").is_not_null()).select(
                "node_id", "record_id", "data_source", vocab="xref_vocab", subject="xref_subject"
            )
        )
    return links


def authority_nodes(base: pl.LazyFrame, ann_paths: list[Path], report: dict) -> pl.LazyFrame:
    """A wrmthorne/authority group beneath every resolved vocabulary or place alignment"""
    res = pl.concat([link for p in ann_paths for link in _authority_links(_filtered_annotations(p))])
    refs = (
        res.unique(subset=["node_id", "vocab", "subject"])
        .join(base.select("node_id", "depth"), on="node_id", how="inner")
        .with_columns(
            gid=bin16("node_id", "vocab", "subject", pl.lit("authority")),
            uri=pl.col("vocab").replace_strict(AUTH_URI_PREFIXES, default=None) + pl.col("subject"),
        )
    )

    groups = refs.select(
        record_id="record_id",
        data_source="data_source",
        node_id="gid",
        parent_id="node_id",
        depth=(pl.col("depth") + 1).cast(pl.UInt8),
        label=pl.lit("Authority"),
        path=pl.lit(None, dtype=pl.String),
        field_type=pl.lit(AUTH_FIELD),
        value=pl.lit(None, dtype=pl.String),
        as_recorded=pl.lit(None, dtype=pl.String),
        component=pl.lit("authorities"),
    )

    children = (
        refs.rename({"vocab": "authority_source", "subject": "authority_id", "uri": "authority_uri"})
        .select("record_id", "data_source", "gid", "depth", *AUTH_PARTS.keys())
        .unpivot(index=["record_id", "data_source", "gid", "depth"], variable_name="part", value_name="part_value")
        .filter(pl.col("part_value").is_not_null())
        .select(
            record_id="record_id",
            data_source="data_source",
            node_id=bin16("gid", "part", pl.lit("authority_child")),
            parent_id="gid",
            depth=(pl.col("depth") + 2).cast(pl.UInt8),
            label=pl.col("part").str.replace_all("_", " ").str.to_titlecase(),
            path=pl.lit(None, dtype=pl.String),
            field_type=pl.col("part").replace_strict(AUTH_PARTS),
            value="part_value",
            as_recorded=pl.lit(None, dtype=pl.String),
            component=pl.lit("authorities"),
        )
    )

    nodes = pl.concat([groups, children])
    n_refs = refs.select(pl.len()).collect(engine="streaming").item()
    report["authorities"] = {"references": n_refs}
    log(f"authorities: {n_refs:,} concept references published")
    return nodes


def compile_persons(base: pl.LazyFrame, report: dict) -> tuple[pl.LazyFrame, pl.DataFrame, pl.DataFrame]:
    """Resolved agent parses become sub-field child nodes"""
    pres = (
        pl.scan_parquet(PERSON_ANN)
        .filter(pl.col("status") == "resolved")
        .with_columns(pl.col("data_source").cast(pl.String))
    )
    pj = pres.join(base.select("node_id", "base_value", pl.col("depth")), on="node_id", how="inner")

    # Agent outputs are strings, not spans, so ignore whitespace differences
    def ws(c: str) -> pl.Expr:
        return pl.col(c).str.replace_all(r"\s+", " ").str.strip_chars()

    stale = (
        pj.filter(ws("value") != ws("base_value"))
        .select(node_id="node_id", component=pl.lit("agents"), reason=pl.lit("stale_snapshot"), detail="value")
        .collect(engine="streaming")
    )
    valid = pj.filter(ws("value") == ws("base_value"))

    children = (
        valid.select("node_id", "record_id", "data_source", "depth", "entity_type", *AGENT_PARTS)
        .unpivot(
            index=["node_id", "record_id", "data_source", "depth", "entity_type"],
            variable_name="part",
            value_name="part_value",
        )
        # A part with no letter is residue, never a name
        .filter(
            pl.col("part_value").is_not_null()
            & (pl.col("part_value") != "")
            & pl.col("part_value").str.contains(r"\p{L}")
        )
        .select(
            record_id="record_id",
            data_source="data_source",
            node_id=bin16("node_id", "part", pl.lit("person_child")),
            parent_id="node_id",
            depth=(pl.col("depth") + 1).cast(pl.UInt8),
            label=pl.col("part").str.replace("_", " ").str.to_titlecase(),
            path=pl.lit(None, dtype=pl.String),
            # Organisations decompose into Organisation fields, everyone else into Person fields
            field_type=pl.when(pl.col("entity_type") == "organisation")
            .then(pl.col("part").replace_strict(ORG_PARTS, default=None))
            .otherwise(pl.col("part").replace_strict(PERSON_PARTS, default=None)),
            value="part_value",
            as_recorded=pl.lit(None, dtype=pl.String),
            component=pl.lit("agents"),
        )
        .filter(pl.col("field_type").is_not_null())
    )

    touched = valid.select("node_id").collect(engine="streaming")
    n_children = children.select(pl.len()).collect(engine="streaming").item()
    report["agents"] = {"resolved_nodes": touched.height, "stale_snapshot": stale.height, "subfield_nodes": n_children}
    log(f"persons: {touched.height:,} resolved nodes → {n_children:,} sub-field nodes")
    return children, stale, touched


def _strip_spans(value: str, spans: list[dict]) -> str:
    for sp in sorted(spans, key=lambda s: -s["span_start"]):
        value = value[: sp["span_start"]] + value[sp["span_end"] :]
    return re.sub(r"\s{2,}", " ", value).strip(" ;,")


# Acquisition and field-collection dates are source-determined and left alone
UNCERTAIN_DEST_FIELDS = ("object_production_date", "technical_attribute_measurement", "object_purchase_price")
# Calibre, weight and shot-size read the same shape as dimensions
UNCERTAIN_MOVES = (("spectrum/technical_attribute", DIM_FIELD),)


def demote_uncertain_destinations(rp: pl.LazyFrame) -> pl.LazyFrame:
    """Demote resolved ops whose destination rests on a context-free assumption to flagged"""
    uncertain_move = pl.any_horizontal(
        (pl.col("op") == "move") & (pl.col("source_field") == src) & (pl.col("field") == dst)
        for src, dst in UNCERTAIN_MOVES
    )
    return rp.with_columns(
        _uncertain_dest=(pl.col("status") == "resolved")
        & (pl.col("field").str.split("/").list.last().is_in(UNCERTAIN_DEST_FIELDS) | uncertain_move)
    ).with_columns(status=pl.when(pl.col("_uncertain_dest")).then(pl.lit("flagged")).otherwise(pl.col("status")))


def compile_patches(
    base: pl.LazyFrame, sources: list[str] | None, report: dict
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Resolved add/move ops apply; flagged probe extractions publish as unranked_extraction certainty"""
    rp = pl.scan_parquet(PATCHES)
    if sources:
        rp = rp.filter(pl.col("data_source").is_in(sources))
    rp = demote_uncertain_destinations(rp)
    res = rp.filter(pl.col("status") == "resolved")

    moves = (
        res.filter(pl.col("op") == "move")
        .join(base.select("node_id", "base_value"), on="node_id", how="inner")
        .with_columns(
            span_text=pl.col("base_value").str.slice("span_start", pl.col("span_end") - pl.col("span_start"))
        )
        .collect(engine="streaming")
    )
    bad_moves = moves.filter(pl.col("span_text") != pl.col("value"))
    moves = moves.filter(pl.col("span_text") == pl.col("value"))

    strip_groups = moves.group_by("node_id").agg(
        pl.col("base_value").first(),
        pl.struct("span_start", "span_end").alias("spans"),
        pl.col("confidence").min().alias("confidence"),
    )
    stripped = [_strip_spans(v, s) for v, s in zip(strip_groups["base_value"], strip_groups["spans"], strict=True)]
    strip_props = strip_groups.with_columns(new_value=pl.Series(stripped, dtype=pl.String)).select(
        node_id="node_id",
        expect_value="base_value",
        new_value="new_value",
        component=pl.lit("record_fixes"),
        tier=pl.lit(5, dtype=pl.Int32),
        confidence="confidence",
    )

    adds = (
        pl.concat(
            [
                res.filter(pl.col("op") == "add").collect(engine="streaming"),
                moves.select(res.collect_schema().names()),
            ],
            how="vertical_relaxed",
        )
        .unique(subset=["record_id", "field", "value"])
        .filter(~pl.col("field").is_in(PROTECTED))
    )
    add_nodes = adds.select(
        record_id="record_id",
        data_source=pl.col("data_source").cast(pl.String),
        node_id=bin16("record_id", "field", "value", pl.lit("patch_add")),
        parent_id=pl.lit(None, dtype=pl.Binary),
        depth=pl.lit(0, dtype=pl.UInt8),
        label=(pl.col("field").str.split("/").list.last().str.replace_all("_", " ").str.to_titlecase()),
        path=pl.lit(None, dtype=pl.String),
        field_type="field",
        value="value",
        as_recorded=pl.lit(None, dtype=pl.String),
        component=pl.lit("record_fixes"),
    )

    certainty = (
        rp.filter(pl.col("status") == "flagged")
        .select(
            target_node_id="node_id",
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            kind=pl.when(pl.col("_uncertain_dest"))
            .then(pl.lit("uncertain_destination"))
            .otherwise(pl.lit("unranked_extraction")),
            notation=pl.lit(None, dtype=pl.String),
            source=pl.lit("record_fixes"),
            confidence="confidence",
            best_candidate="value",
            detail="field",
        )
        .collect(engine="streaming")
    )

    conflicts = bad_moves.select(
        node_id="node_id", component=pl.lit("record_fixes"), reason=pl.lit("bad_span"), detail="value"
    )
    demoted = int((certainty["kind"] == "uncertain_destination").sum())
    report["record_fixes"] = {
        "add_nodes": add_nodes.height,
        "move_strips": strip_props.height,
        "bad_move_spans": bad_moves.height,
        "certainty": certainty.height,
        "uncertain_destination_demoted": demoted,
    }
    return add_nodes, strip_props, certainty, conflicts


def edtf_patch_dates(add_nodes: pl.DataFrame, report: dict) -> pl.DataFrame:
    """Parse dates a record_fixes move landed in a date field after the date stage had read the corpus"""
    dated = add_nodes.filter(pl.col("field_type").is_in(DATE_FIELDS))
    if not dated.height:
        report["patch_date_reparse"] = {"candidates": 0, "reparsed": 0}
        return add_nodes
    rest = add_nodes.filter(~pl.col("field_type").is_in(DATE_FIELDS))
    conv = load_conventions()
    conv_keys = ["dm_order", "eq", "z0"]
    dated = dated.join(conv, on="data_source", how="left").with_columns(
        dm_order=pl.col("dm_order").fill_null(""), eq=pl.col("eq").fill_null(False), z0=pl.col("z0").fill_null(False)
    )
    semantic = (
        pl.col("value")
        .str.to_lowercase()
        .str.strip_chars()
        .str.strip_chars_end(" .")
        .is_in(sorted(DATE_SEMANTIC_MARKERS))
    )
    vals = dated.filter(~semantic).select("value", *conv_keys).unique()
    rows = []
    for v, o, e, z in vals.iter_rows():
        try:
            p = parse_date(v, Conventions(dm_order=o or None, eq_range=e, zero_null=z))
        except Exception:
            p = None
        earliest = p.get("date_earliest_single") if p else None
        rows.append(
            (
                v,
                o,
                e,
                z,
                p.get("value_edtf") if p else None,
                earliest,
                (p.get("date_latest") or earliest) if p else None,
            )
        )
    parsed = pl.DataFrame(
        rows,
        orient="row",
        schema={
            "value": pl.String,
            "dm_order": pl.String,
            "eq": pl.Boolean,
            "z0": pl.Boolean,
            "edtf": pl.String,
            "earliest": pl.String,
            "latest": pl.String,
        },
    )
    # Same slot rule as compile_dates; moved nodes get no substructure
    written = (
        pl.when(pl.col("field_type") == "spectrum/date_earliest_single")
        .then(pl.col("earliest"))
        .when(pl.col("field_type") == "spectrum/date_latest")
        .then(pl.col("latest"))
        .when(pl.col("edtf").str.contains(ISO_POINT))
        .then(pl.col("edtf"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    # Same far-future guard as compile_dates: the text stays instead
    future = pl.col("field_type").is_in(list(DATE_BOUNDS)) & (
        written.str.extract(r"^(\d{3,4})").cast(pl.Int32, strict=False) > IMPLAUSIBLE_YEAR
    ).fill_null(False)
    out = (
        dated.join(parsed, on=["value", *conv_keys], how="left")
        .with_columns(_new=pl.when(~future).then(written))
        .with_columns(
            as_recorded=pl.when(pl.col("_new").is_not_null() & (pl.col("_new") != pl.col("value")))
            .then(pl.col("value"))
            .otherwise(pl.col("as_recorded")),
            value=pl.when(pl.col("_new").is_not_null()).then(pl.col("_new")).otherwise(pl.col("value")),
        )
        .select(add_nodes.columns)
    )
    n = out.filter(pl.col("as_recorded").is_not_null()).height
    report["patch_date_reparse"] = {"candidates": dated.height, "reparsed": n}
    log(f"patch dates: {n:,} moved-in date values reparsed to EDTF")
    return pl.concat([rest, out])


OBJNUM_TYPES = ("accession number", "object number")


def compile_object_numbers(base: pl.LazyFrame, report: dict) -> pl.DataFrame:
    """Type-gated object-number recovery from an other_number"""
    typed = base.filter(
        (pl.col("field_type") == "spectrum/other_number_type")
        & pl.col("base_value").str.to_lowercase().str.strip_chars().is_in(OBJNUM_TYPES)
    ).select(pl.col("parent_id").alias("node_id"))
    cands = base.filter((pl.col("field_type") == "spectrum/other_number") & pl.col("base_value").is_not_null()).join(
        typed, on="node_id", how="semi"
    )
    have = (
        base.filter((pl.col("field_type") == "spectrum/object_number") & pl.col("base_value").is_not_null())
        .select("record_id")
        .unique()
    )
    picked = (
        cands.join(have, on="record_id", how="anti")
        .group_by("record_id")
        .agg(
            pl.col("data_source").first(),
            pl.col("base_value").n_unique().alias("n_cands"),
            pl.col("base_value").first().alias("value"),
        )
        .collect(engine="streaming")
    )
    take = picked.filter(pl.col("n_cands") == 1)
    add_nodes = take.select(
        record_id="record_id",
        data_source=pl.col("data_source").cast(pl.String),
        node_id=bin16("record_id", "value", pl.lit("objnum_transfer")),
        parent_id=pl.lit(None, dtype=pl.Binary),
        depth=pl.lit(0, dtype=pl.UInt8),
        label=pl.lit("Object Number"),
        path=pl.lit(None, dtype=pl.String),
        field_type=pl.lit("spectrum/object_number"),
        value="value",
        as_recorded=pl.lit(None, dtype=pl.String),
        component=pl.lit("object_number_transfer"),
    )
    report["object_number_transfer"] = {"records": take.height, "ambiguous_skipped": picked.height - take.height}
    log(f"object numbers: {take.height:,} transferred, {picked.height - take.height:,} ambiguous skipped")
    return add_nodes


def _canonical(mapping: dict[str, str]) -> pl.Expr:
    return pl.col("base_value").str.strip_chars().str.to_lowercase().replace_strict(mapping, default=None)


UNIT_CONVENTIONS = INSTITUTIONAL / "unit_conventions.parquet"


def load_unit_conventions() -> pl.DataFrame:
    """Per-institution prime-symbol reading ('in' or 'ft'), exported by institutional_priors.py"""
    schema = {"data_source": pl.String, "prime": pl.String}
    if not UNIT_CONVENTIONS.exists():
        log(f"dimensions: no unit conventions at {UNIT_CONVENTIONS} — every prime reads as the parser's default")
        return pl.DataFrame(schema=schema)
    return pl.read_parquet(UNIT_CONVENTIONS).select("data_source", prime="prime_unit")


def _dimension_cache(pairs: pl.DataFrame, cache_path: Path) -> pl.DataFrame:
    """Parse cache keyed by (value, prime-reading), versioned by `pver`"""
    schema = {
        "value": pl.String,
        "prime": pl.String,
        "status": pl.String,
        "measurements": pl.String,
        "residue": pl.String,
        "pver": pl.String,
    }
    cached = pl.read_parquet(cache_path) if cache_path.exists() else pl.DataFrame(schema=schema)
    if set(cached.columns) != set(schema):
        log("dimensions: cache pre-dates its current key set; reparsing")
        cached = pl.DataFrame(schema=schema)
    # rows from an older parser reparse
    cached = cached.filter(pl.col("pver") == DIM_PARSER_VERSION)
    todo = pairs.join(cached, on=["value", "prime"], how="anti")
    if todo.height:
        rows = []
        for v, prime in todo.select("value", "prime").iter_rows():
            try:
                p = parse_dimensions(v, prime_unit=prime)
            except Exception:
                p = None
            rows.append(
                (v, prime, "unparsed", None, None, DIM_PARSER_VERSION)
                if p is None
                else (
                    v,
                    prime,
                    p["status"],
                    json.dumps(p["measurements"]),
                    json.dumps(p["residue"]) if p["residue"] else None,
                    DIM_PARSER_VERSION,
                )
            )
        cached = pl.concat([cached, pl.DataFrame(rows, schema=schema, orient="row")])
        cached.write_parquet(cache_path)
    log(f"dimensions: parsed {todo.height:,} new values")
    return cached.drop("pver")


# A newline is a type separator: CRLF-aligned parallel lists
_TYPE_SPLIT = re.compile(r"(?i)\bx\b|×|,|/|[\r\n]+")
# 'wxh' and 'lxwxh' glue the separator to the single-letter types
_TYPE_UNGLUE = re.compile(r"(?i)(?<=[a-z])x(?=[a-z])")

MEASURE_SCHEMA = {
    "value": pl.String,
    "ptype": pl.String,
    "prime": pl.String,
    "axis": pl.Int32,
    "dimension_type": pl.String,
    "dimension_value": pl.String,
    "dimension_measurement_unit": pl.String,
    "dimension_value_qualifier": pl.String,
    "dimension_measured_part": pl.String,
    "ambiguous": pl.Boolean,
}
DEFER_SCHEMA = {"value": pl.String, "ptype": pl.String, "prime": pl.String, "reason": pl.String, "residue": pl.String}


_HELD_PAREN = re.compile(r"\x00(\d+)\x00")


def _split_type_expr(expr: str) -> list[tuple[str, str | None]]:
    """Segments of a parent type expression, each as (type words, its parenthetical)"""
    held: list[str] = []

    def hold(m: re.Match) -> str:
        held.append(m.group(1).strip(" ."))
        return f"\x00{len(held) - 1}\x00"

    out = []
    for seg in _TYPE_SPLIT.split(_TYPE_UNGLUE.sub(" x ", PAREN.sub(hold, expr))):
        idx = _HELD_PAREN.findall(seg)
        bare = _HELD_PAREN.sub(" ", seg).strip()
        if bare:
            out.append((bare, held[int(idx[0])] if len(idx) == 1 else None))
    return out


def _stated_types(expr: str, ms: list[dict]) -> list[tuple[str, str | None, str | None]] | None:
    """A parent Dimension node's value read as the (type, qualifier, measured part) of each measurement"""
    stated = []
    for bare, paren in _split_type_expr(expr):
        typ = DIM_TERMS.get(bare.lower())
        if typ is None:
            return None
        qual = QUALIFIERS.get(paren.lower()) if paren else None
        stated.append((typ, qual, paren.lower() if paren and not qual else None))
    if not stated:
        return None
    # group into positions; a min→max pair collapses into one
    positions, i = [], 0
    while i < len(ms):
        q = ms[i].get("dimension_value_qualifier") or ""
        if "minimum" in q and i + 1 < len(ms) and "maximum" in (ms[i + 1].get("dimension_value_qualifier") or ""):
            positions.append((i, i + 1))
            i += 2
        else:
            positions.append((i,))
            i += 1
    if len(stated) != len(positions):
        return None
    out: list[tuple[str, str | None, str | None] | None] = [None] * len(ms)
    for st, idxs in zip(stated, positions, strict=True):
        for j in idxs:
            out[j] = st
    return out


# Sources round the conversion they print, so magnitudes match loosely
_RESTATE_TOL = 0.02


def _base_magnitude(m: dict) -> tuple[str, float] | None:
    unit = m["dimension_measurement_unit"]
    if unit not in TO_BASE or m["dimension_value"] is None:
        return None
    return UNIT_CLASS[unit], m["dimension_value"] * TO_BASE[unit]


def _inherit_restated(ms: list[dict], types: list[str | None], parts: list[str | None]) -> None:
    """Type each measurement that restates a typed one in another unit, in place"""
    typed = [i for i, t in enumerate(types) if t is not None]
    for j, t in enumerate(types):
        b = _base_magnitude(ms[j]) if t is None else None
        if b is None:
            continue
        cand = {
            (types[i], parts[i])
            for i in typed
            if (a := _base_magnitude(ms[i])) is not None
            and a[0] == b[0]
            and ms[i]["dimension_measurement_unit"] != ms[j]["dimension_measurement_unit"]
            and abs(a[1] - b[1]) <= _RESTATE_TOL * max(a[1], b[1])
        }
        if len(cand) == 1:
            types[j], parts[j] = cand.pop()


def _measurement_rows(cached: pl.DataFrame, pairs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Resolve each (value, parent-type-expression) pair into measurement rows"""
    lookup = {(v, pr): (s, p, r) for v, pr, s, p, r in cached.iter_rows()}
    rows, deferred = [], []
    for value, ptype, prime in pairs.iter_rows():
        status, payload, residue = lookup[(value, prime)]
        ms = json.loads(payload) if payload else []
        if status != "resolved" or not ms:
            deferred.append((value, ptype, prime, status, residue))
            continue
        stated = _stated_types(ptype, ms) if ptype else None
        types, quals, parts, ambiguous, disagree = [], [], [], [], False
        for i, m in enumerate(ms):
            explicit = None if m["type_ambiguous"] else m["dimension_type"]
            parent, p_qual, p_part = stated[i] if stated else (None, None, None)
            disagree |= bool(explicit and parent and explicit != parent)
            types.append(explicit or parent or m["dimension_type"])
            quals.append(join_slots(m["dimension_value_qualifier"], p_qual))
            parts.append(join_slots(m["dimension_measured_part"], p_part))
            # Magnitudes are evidence for the pairing, not the type
            ambiguous.append(explicit is None and parent is None)
        _inherit_restated(ms, types, parts)
        # Publish the pairing untyped rather than a glued string
        untyped_single = (
            len(ms) == 1 and ms[0]["dimension_value"] is not None and ms[0]["dimension_measurement_unit"] is not None
        )
        if disagree or (any(t is None for t in types) and not untyped_single):
            deferred.append((value, ptype, prime, "type_disagreement" if disagree else "untyped", residue))
            continue
        rows.extend(
            (
                value,
                ptype,
                prime,
                axis,
                types[axis],
                f"{m['dimension_value']:.6g}",
                m["dimension_measurement_unit"],
                quals[axis],
                parts[axis],
                ambiguous[axis],
            )
            for axis, m in enumerate(ms)
        )
    return (
        pl.DataFrame(rows, orient="row", schema=MEASURE_SCHEMA),
        pl.DataFrame(deferred, orient="row", schema=DEFER_SCHEMA),
    )


def compile_dimensions(
    base: pl.LazyFrame, cache_path: Path, out_dir: Path, report: dict
) -> tuple[pl.DataFrame, pl.LazyFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Tier-1 dimension stage: the largest decomposition available"""
    dim = base.filter(pl.col("field_type") == DIM_FIELD)
    parent_ids = base.select(pl.col("parent_id").alias("node_id")).unique().drop_nulls()

    # One node in population A, parent and child in B
    pop_a = (
        dim.join(parent_ids, on="node_id", how="anti")
        .filter(pl.col("base_value").str.contains(r"\d"))
        .select(
            "record_id",
            "data_source",
            "parent_id",
            "depth",
            gid="node_id",
            gvalue="base_value",
            raw_node="node_id",
            raw_value="base_value",
            # One node here, so no separately-stated type to read
            ptype=pl.lit("", dtype=pl.String),
        )
    )
    # Population C: correspondence between siblings is already lost
    dv_counts = (
        base.filter(pl.col("field_type") == "spectrum/dimension_value")
        .group_by("parent_id")
        .agg(n_values=pl.len())
        .rename({"parent_id": "node_id"})
    )
    pop_b = (
        base.filter(
            (pl.col("field_type") == "spectrum/dimension_value")
            & pl.col("base_value").is_not_null()
            & ~pl.col("base_value").str.strip_chars().str.contains(r"^\d+(\.\d+)?$")
        )
        .join(
            dv_counts.filter(pl.col("n_values") == 1).select("node_id"),
            left_on="parent_id",
            right_on="node_id",
            how="semi",
        )
        .join(
            dim.select("node_id", gvalue="base_value", gparent="parent_id", gdepth="depth"),
            left_on="parent_id",
            right_on="node_id",
            how="inner",
        )
        .select(
            "record_id",
            "data_source",
            parent_id="gparent",
            depth="gdepth",
            gid="parent_id",
            gvalue="gvalue",
            raw_node="node_id",
            raw_value="base_value",
            ptype=pl.col("gvalue").fill_null(""),
        )
    )
    # Values without a prime keep the default cache key
    work = (
        pl.concat([pop_a, pop_b])
        .with_columns(_ds=pl.col("data_source").cast(pl.String))
        .join(load_unit_conventions().lazy().rename({"data_source": "_ds"}), on="_ds", how="left")
        .with_columns(
            prime=pl.when(pl.col("raw_value").str.contains("['′‘’]"))
            .then(pl.col("prime").fill_null("ft"))
            .otherwise(pl.lit("ft"))
        )
        .drop("_ds")
    )

    pairs = work.select(value="raw_value", ptype="ptype", prime="prime").unique().collect(engine="streaming")
    log(f"dimensions: {pairs.height:,} distinct (value, stated type, prime) pairs")
    cached = _dimension_cache(pairs.select("value", "prime").unique(), cache_path)
    mrows, deferred = _measurement_rows(cached, pairs)

    # Existing children: never create twice, and agree with the unit
    kids = (
        base.filter(pl.col("field_type").is_in(list(DIM_PARTS.values())))
        .select(
            gid="parent_id",
            cft="field_type",
            cunit=pl.when(pl.col("field_type") == "spectrum/dimension_measurement_unit").then(_canonical(UNITS)),
        )
        .group_by("gid")
        .agg(have=pl.col("cft").unique(), unit=pl.col("cunit").drop_nulls().first())
    )

    joined = work.join(
        mrows.lazy(), left_on=["raw_value", "ptype", "prime"], right_on=["value", "ptype", "prime"]
    ).join(kids, on="gid", how="left")
    clash = (
        joined.filter(
            (pl.col("axis") == 0)
            & pl.col("unit").is_not_null()
            & pl.col("dimension_measurement_unit").is_not_null()
            & (pl.col("unit") != pl.col("dimension_measurement_unit"))
        )
        .select("gid", "raw_value", "unit", "dimension_measurement_unit")
        .collect(engine="streaming")
    )
    joined = joined.join(clash.lazy().select("gid"), on="gid", how="anti").with_columns(
        axis_id=pl.when(pl.col("axis") == 0)
        .then(pl.col("gid"))
        .otherwise(bin16("gid", "axis", pl.lit("dimension_axis")))
    )

    types = joined.filter(
        (pl.col("axis") == 0)
        & pl.col("dimension_type").is_not_null()
        & (pl.col("gvalue").is_null() | (pl.col("gvalue") != pl.col("dimension_type")))
    ).select(
        node_id="gid",
        expect_value="gvalue",
        new_value="dimension_type",
        component=pl.lit("dimensions"),
        tier=pl.lit(1, dtype=pl.Int32),
        confidence=pl.lit(1.0),
    )
    numbers = joined.filter((pl.col("axis") == 0) & (pl.col("raw_node") != pl.col("gid"))).select(
        node_id="raw_node",
        expect_value="raw_value",
        new_value="dimension_value",
        component=pl.lit("dimensions"),
        tier=pl.lit(1, dtype=pl.Int32),
        confidence=pl.lit(1.0),
    )
    decomposed = pl.concat([types, numbers]).collect(engine="streaming")

    siblings = joined.filter(pl.col("axis") > 0).select(
        record_id="record_id",
        data_source=pl.col("data_source").cast(pl.String),
        node_id="axis_id",
        parent_id="parent_id",
        depth="depth",
        label=pl.lit("Dimension"),
        path=pl.lit(None, dtype=pl.String),
        field_type=pl.lit(DIM_FIELD),
        value="dimension_type",
        as_recorded="raw_value",
        component=pl.lit("dimensions"),
    )

    children = (
        joined.select("record_id", "data_source", "axis_id", "axis", "depth", "have", *DIM_PARTS)
        .unpivot(
            index=["record_id", "data_source", "axis_id", "axis", "depth", "have"],
            variable_name="part",
            value_name="part_value",
        )
        .with_columns(cft=pl.col("part").replace_strict(DIM_PARTS))
        .filter(
            pl.col("part_value").is_not_null()
            # A group may already hold the sub-field; never duplicate
            & ((pl.col("axis") > 0) | pl.col("have").is_null() | ~pl.col("have").list.contains(pl.col("cft")))
        )
        .select(
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            node_id=bin16("axis_id", "part", pl.lit("dimension_child")),
            parent_id="axis_id",
            depth=(pl.col("depth") + 1).cast(pl.UInt8),
            label=pl.col("part").str.replace_all("_", " ").str.to_titlecase(),
            path=pl.lit(None, dtype=pl.String),
            field_type="cft",
            value="part_value",
            as_recorded=pl.lit(None, dtype=pl.String),
            component=pl.lit("dimensions"),
        )
    )

    proposed = decomposed.lazy().select("node_id")
    canon = (
        pl.concat(
            [
                base.filter(pl.col("field_type") == "spectrum/dimension_measurement_unit").with_columns(
                    canon=_canonical(UNITS)
                ),
                dim.with_columns(canon=_canonical(DIM_TERMS)),
            ]
        )
        .join(proposed, on="node_id", how="anti")
        .filter(pl.col("canon").is_not_null() & (pl.col("canon") != pl.col("base_value")))
        .select(
            node_id="node_id",
            expect_value="base_value",
            new_value="canon",
            component=pl.lit("dimensions"),
            tier=pl.lit(1, dtype=pl.Int32),
            confidence=pl.lit(1.0),
        )
        .collect(engine="streaming")
    )

    # Qualify every sibling: a consumer may read `width` alone
    def _cert(rows: pl.LazyFrame, kind: str, notation: str | None, source: str, detail: str) -> pl.DataFrame:
        return (
            rows.unique(subset=["axis_id"])
            .select(
                target_node_id="axis_id",
                record_id="record_id",
                data_source=pl.col("data_source").cast(pl.String),
                kind=pl.lit(kind),
                notation=pl.lit(notation, dtype=pl.String),
                source=pl.lit(source),
                confidence=pl.lit(None, dtype=pl.Float64),
                best_candidate=pl.lit(None, dtype=pl.String),
                detail=pl.lit(detail),
            )
            .collect(engine="streaming")
        )

    parallel = (
        dim.join(dv_counts.filter(pl.col("n_values") > 1), on="node_id")
        .select(
            target_node_id="node_id",
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            kind=pl.lit("ambiguous_correspondence"),
            notation=pl.lit(None, dtype=pl.String),
            source=pl.lit("dimensions"),
            confidence=pl.lit(None, dtype=pl.Float64),
            best_candidate=pl.lit(None, dtype=pl.String),
            detail=pl.format("{} dimension_value children share one Dimension node", "n_values"),
        )
        .collect(engine="streaming")
    )

    certainty = pl.concat(
        [
            _cert(
                joined.filter(pl.col("ambiguous")),
                "ambiguous_dimension_type",
                None,
                "dimensions",
                "type read by the h x w x d convention; neither the value nor its parent states it",
            ),
            _cert(
                joined.filter(pl.col("dimension_value_qualifier").str.contains("uncertain")),
                "cataloguer_marked",
                "?",
                "cataloguer",
                "measurement marked uncertain by the cataloguer",
            ),
            parallel,
        ]
    )

    conflicts = clash.select(
        node_id="gid",
        component=pl.lit("dimensions"),
        reason=pl.lit("unit_disagreement"),
        detail=pl.format("{} parses as {}, recorded unit is {}", "raw_value", "dimension_measurement_unit", "unit"),
    )

    n_children = children.select(pl.len()).collect(engine="streaming").item()
    sib_targets = siblings.select("node_id", "depth").collect(engine="streaming")
    deferred.write_parquet(out_dir / "dimension_residue.parquet")
    report["dimensions"] = {
        "distinct_pairs": pairs.height,
        "deferred_pairs": deferred.height,
        "type_proposals": types.select(pl.len()).collect(engine="streaming").item(),
        "value_proposals": numbers.select(pl.len()).collect(engine="streaming").item(),
        "sibling_nodes": sib_targets.height,
        "subfield_nodes": n_children,
        "canonicalised_terms": canon.height,
        "certainty": certainty.height,
        "unit_disagreements": clash.height,
        "parallel_value_groups": parallel.height,
    }
    log(
        f"dimensions: {decomposed.height:,} nodes rewritten → {sib_targets.height:,} "
        f"sibling + {n_children:,} sub-field nodes; {canon.height:,} terms "
        f"canonicalised; {clash.height:,} unit disagreements deferred"
    )

    reconcile_dimensions(base, joined, out_dir, report)
    return (
        pl.concat([decomposed, canon]),
        pl.concat([siblings, children]),
        certainty,
        decomposed.select("node_id"),
        sib_targets,
        conflicts,
    )


def reconcile_dimensions(base: pl.LazyFrame, joined: pl.LazyFrame, out_dir: Path, report: dict) -> None:
    """Free verification over records holding a free-string dimension and an already-atomic one"""
    kids = (
        base.filter(pl.col("field_type").is_in(list(DIM_PARTS.values())))
        .select(node_id="parent_id", cft="field_type", cval="base_value")
        .group_by("node_id")
        .agg(
            a_value=pl.col("cval").filter(pl.col("cft") == "spectrum/dimension_value").first(),
            a_unit=pl.col("cval").filter(pl.col("cft") == "spectrum/dimension_measurement_unit").first(),
        )
    )
    atomic = (
        base.filter(pl.col("field_type") == DIM_FIELD)
        .join(kids, on="node_id", how="inner")
        .select(
            "record_id",
            a_type=_canonical(DIM_TERMS),
            a_value=pl.col("a_value").cast(pl.Float64, strict=False),
            a_unit=pl.col("a_unit").str.strip_chars().str.to_lowercase().replace_strict(UNITS, default=None),
        )
        .drop_nulls()
        .unique()
    )
    parsed = (
        joined.filter(pl.col("raw_node") == pl.col("gid"))
        .select(
            "record_id",
            "data_source",
            p_type="dimension_type",
            p_value=pl.col("dimension_value").cast(pl.Float64),
            p_unit="dimension_measurement_unit",
        )
        .drop_nulls()
        .unique()
    )

    # Magnitudes compare in the class's base unit, not as recorded
    def base_of(c: str) -> pl.Expr:
        return pl.col(c).replace_strict(TO_BASE, default=None)

    pairs = (
        parsed.join(atomic, left_on=["record_id", "p_type"], right_on=["record_id", "a_type"], how="inner")
        .with_columns(
            p_base=pl.col("p_value") * base_of("p_unit"),
            a_base=pl.col("a_value") * base_of("a_unit"),
            same_class=(
                pl.col("p_unit").replace_strict(UNIT_CLASS, default=None)
                == pl.col("a_unit").replace_strict(UNIT_CLASS, default=None)
            ),
        )
        .with_columns(
            agree=(
                pl.col("same_class") & ((pl.col("p_base") - pl.col("a_base")).abs() <= 1e-3 * pl.col("a_base").abs())
            ).fill_null(False)
        )
        # Whether the parsed one matches some atomic length, not every
        .group_by("data_source", "record_id", "p_type", "p_value", "p_unit")
        .agg(agree=pl.col("agree").any())
        .collect(engine="streaming")
    )
    by_inst = (
        pairs.group_by("data_source")
        .agg(measurements=pl.len(), agree=pl.col("agree").sum())
        .with_columns(rate=pl.col("agree") / pl.col("measurements"))
        .sort("measurements", descending=True)
    )
    by_inst.write_parquet(out_dir / "dimension_reconciliation.parquet")
    n = pairs.height
    report["dimensions"]["reconciliation"] = {
        "matched_measurements": n,
        "agree": int(pairs["agree"].sum()),
        "rate": round(pairs["agree"].mean(), 4) if n else None,
        "institutions": by_inst.height,
    }
    log(
        f"dimensions: reconciled {n:,} parsed measurements against atomic siblings "
        f"of the same type, {pairs['agree'].mean():.1%} agreement"
        if n
        else "dimensions: no pairs to reconcile"
    )


COUNT_FIELD = "spectrum/number_of_objects"


def compile_counts(base: pl.LazyFrame, report: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Tier-1 counts stage over number_of_objects"""
    nodes = base.filter((pl.col("field_type") == COUNT_FIELD) & pl.col("base_value").is_not_null())
    vals = nodes.select(pl.col("base_value").alias("value")).unique().collect(engine="streaming")
    rows = []
    for v in vals["value"]:
        parts = parse_count(v)
        # A compound must never collapse to its first term
        if parts is None or len(parts) != 1 or parts[0].count is None:
            continue
        rows.append((v, str(parts[0].count), parts[0].qualifier))
    parsed = pl.DataFrame(rows, orient="row", schema={"value": pl.String, "count": pl.String, "qualifier": pl.String})
    matched = nodes.join(parsed.lazy(), left_on="base_value", right_on="value")

    proposals = (
        matched.filter(pl.col("count") != pl.col("base_value"))
        .select(
            node_id="node_id",
            expect_value="base_value",
            new_value="count",
            component=pl.lit("counts"),
            tier=pl.lit(1, dtype=pl.Int32),
            confidence=pl.lit(1.0),
        )
        .collect(engine="streaming")
    )
    certainty = (
        matched.filter(pl.col("qualifier").is_not_null())
        .select(
            target_node_id="node_id",
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            kind=pl.lit("cataloguer_marked"),
            notation=pl.col("base_value").str.extract(
                r"(?i)^\s*(c\.?|ca\.?|approx\.?|approximately|about|est\.?|estimated)\b", 1
            ),
            source=pl.lit("cataloguer"),
            confidence=pl.lit(None, dtype=pl.Float64),
            best_candidate=pl.lit(None, dtype=pl.String),
            detail="qualifier",
        )
        .collect(engine="streaming")
    )
    report["counts"] = {
        "distinct_values": vals.height,
        "parsed_values": parsed.height,
        "proposals": proposals.height,
        "certainty": certainty.height,
    }
    log(f"counts: {proposals.height:,} nodes normalised to an integer")
    return proposals, certainty, proposals.select("node_id")


# Digits keep prose out and make a trailing '?' epistemic
_ID_TOKEN = r"[0-9A-Za-z][0-9A-Za-z._/\-]*"
_ID_MARKED = re.compile(rf"^\s*(?P<id>{_ID_TOKEN})\?+\s*$")
# Two or more marked ids in one cell
_ID_ALTS = re.compile(rf"^\s*{_ID_TOKEN}\?+(?:\s*[,;]?\s+{_ID_TOKEN}\?+)+\s*$")
_ID_HAS_DIGIT = re.compile(r"\d")
# Cheap prefilter; the Python rules above do the reading
_ID_PREFILTER = r"\?"


def read_marked_identifier(value: str) -> tuple[str | None, int] | None:
    """Read a cataloguer's uncertainty mark off a reference number"""
    if not value or not _ID_HAS_DIGIT.search(value):
        return None
    if m := _ID_MARKED.match(value):
        return m["id"], 1
    if _ID_ALTS.match(value):
        return None, len(re.findall(r"\?+", value))
    return None


def compile_identifier_certainty(base: pl.LazyFrame, report: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cataloguer uncertainty marked on reference numbers ('1934.02?')"""
    elig = base.filter(
        pl.col("field_type").is_in(IDENTIFIERS)
        & pl.col("base_value").is_not_null()
        & pl.col("base_value").str.contains(_ID_PREFILTER)
    )
    vals = elig.select(pl.col("base_value").alias("value")).unique().collect(engine="streaming")
    rows = [(v, *hit) for v in vals["value"] if (hit := read_marked_identifier(v)) is not None]
    parsed = pl.DataFrame(rows, orient="row", schema={"value": pl.String, "clean": pl.String, "n_alts": pl.Int32})
    matched = elig.join(parsed.lazy(), left_on="base_value", right_on="value")

    single = matched.filter(pl.col("clean").is_not_null())
    proposals = single.select(
        node_id="node_id",
        expect_value="base_value",
        new_value="clean",
        component=pl.lit("identifier_certainty"),
        tier=pl.lit(1, dtype=pl.Int32),
        confidence=pl.lit(1.0),
    ).collect(engine="streaming")
    certainty = matched.select(
        target_node_id="node_id",
        record_id="record_id",
        data_source=pl.col("data_source").cast(pl.String),
        kind=pl.when(pl.col("clean").is_not_null())
        .then(pl.lit("cataloguer_marked"))
        .otherwise(pl.lit("ambiguous_identifier")),
        notation=pl.lit("?"),
        source=pl.lit("cataloguer"),
        confidence=pl.lit(None, dtype=pl.Float64),
        best_candidate=pl.lit(None, dtype=pl.String),
        detail=pl.when(pl.col("clean").is_not_null())
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(
            pl.format(
                "{} alternative identifiers recorded as one value, each marked uncertain; kept as recorded",
                pl.col("n_alts"),
            )
        ),
    ).collect(engine="streaming")
    n_alt = certainty.height - proposals.height
    report["identifier_certainty"] = {
        "candidate_values": vals.height,
        "marked_values": parsed.height,
        "proposals": proposals.height,
        "ambiguous_nodes": n_alt,
    }
    log(f"identifiers: {proposals.height:,} marked reference numbers cleaned, {n_alt:,} alternative sets qualified")
    return proposals, certainty


def compile_notation(base: pl.LazyFrame, touched: pl.DataFrame, report: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cataloguer-recorded uncertainty on structured non-date fields"""
    excluded = DATE_FIELDS + FREE_TEXT + PROTECTED + IDENTIFIERS + MEASUREMENT + MONETARY
    elig = (
        base.filter(
            ~pl.col("field_type").is_in(excluded) & pl.col("base_value").is_not_null() & ~pl.col("contains_url")
        )
        .join(touched.lazy(), on="node_id", how="anti")
        .filter(pl.col("base_value").str.contains(PREFILTER))
    )
    vals = elig.select(pl.col("base_value").alias("value")).unique().collect(engine="streaming")
    log(f"notation: {vals.height:,} distinct candidate values")

    hits = [(v, m.clean, m.notation) for v in vals["value"] if (m := parse_certainty(v))]
    parsed = pl.DataFrame(hits, schema={"value": pl.String, "clean": pl.String, "notation": pl.String}, orient="row")
    matched = elig.join(parsed.lazy(), left_on="base_value", right_on="value")

    proposals = matched.select(
        node_id="node_id",
        expect_value="base_value",
        new_value="clean",
        component=pl.lit("certainty_notation"),
        tier=pl.lit(1, dtype=pl.Int32),
        confidence=pl.lit(1.0),
    ).collect(engine="streaming")
    certainty = matched.select(
        target_node_id="node_id",
        record_id="record_id",
        data_source=pl.col("data_source").cast(pl.String),
        kind=pl.lit("cataloguer_marked"),
        notation="notation",
        source=pl.lit("cataloguer"),
        confidence=pl.lit(None, dtype=pl.Float64),
        best_candidate=pl.lit(None, dtype=pl.String),
        detail=pl.lit(None, dtype=pl.String),
    ).collect(engine="streaming")
    report["certainty_notation"] = {
        "candidate_values": vals.height,
        "marked_values": parsed.height,
        "proposals": proposals.height,
        "certainty": certainty.height,
    }
    return proposals, certainty


def compile_encoding_damage(base: pl.LazyFrame, report: dict) -> pl.DataFrame:
    """A value carrying U+FFFD replacement characters is source-side encoding damage"""
    was_damaged = pl.col("value").str.contains("�", literal=True).fill_null(False)
    still = pl.col("base_value").str.contains("�", literal=True).fill_null(False)
    rows = (
        base.filter(was_damaged | still)
        .select(
            target_node_id="node_id",
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            kind=pl.when(still).then(pl.lit("encoding_damage")).otherwise(pl.lit("encoding_repair")),
            notation=pl.lit(None, dtype=pl.String),
            source=pl.lit("tier0"),
            confidence=pl.lit(None, dtype=pl.Float64),
            best_candidate=pl.lit(None, dtype=pl.String),
            detail=pl.when(still)
            .then(
                pl.lit(
                    "value carries U+FFFD replacement characters; the original text is unrecoverable from the export"
                )
            )
            .otherwise(
                pl.lit(
                    "U+FFFD damage repaired against a unique "
                    "corpus/authority attestation "
                    "(build_mojibake_repairs); the recorded "
                    "form survives in as_recorded"
                )
            ),
        )
        .collect(engine="streaming")
    )
    n_rep = int((rows["kind"] == "encoding_repair").sum())
    report["encoding_damage"] = {"annotated_nodes": rows.height - n_rep, "repaired_nodes": n_rep}
    log(f"encoding damage: {rows.height - n_rep:,} nodes annotated, {n_rep:,} repaired")
    return rows


def merge_proposals(base: pl.LazyFrame, proposals: pl.DataFrame, report: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One winner per node by tier then confidence; everything else logged"""
    checked = (
        proposals.lazy()
        .join(base.select("node_id", "base_value", "field_type"), on="node_id", how="inner")
        .collect(engine="streaming")
    )

    protected = checked.filter(pl.col("field_type").is_in(PROTECTED))
    stale = checked.filter(
        ~pl.col("field_type").is_in(PROTECTED)
        & pl.col("expect_value").is_not_null()
        & (pl.col("expect_value") != pl.col("base_value"))
    )
    valid = checked.filter(
        ~pl.col("field_type").is_in(PROTECTED)
        & (pl.col("expect_value").is_null() | (pl.col("expect_value") == pl.col("base_value")))
    ).filter(pl.col("new_value") != pl.col("base_value"))

    ranked = valid.sort(["tier", "confidence"], descending=[False, True], nulls_last=True)
    winners = ranked.unique(subset=["node_id"], keep="first")
    losers = ranked.join(winners.select("node_id", "component"), on=["node_id", "component"], how="anti")

    conflicts = pl.concat(
        [
            stale.select(
                node_id="node_id", component="component", reason=pl.lit("stale_snapshot"), detail="expect_value"
            ),
            protected.select(
                node_id="node_id", component="component", reason=pl.lit("protected_field"), detail="new_value"
            ),
            losers.select(
                node_id="node_id", component="component", reason=pl.lit("lost_precedence"), detail="new_value"
            ),
        ]
    )
    report["merge"] = {
        "proposals": proposals.height,
        "applied": winners.height,
        "stale_snapshot": stale.height,
        "protected_field": protected.height,
        "lost_precedence": losers.height,
    }
    return winners.select("node_id", "new_value", "component"), conflicts


def certainty_nodes(cert: pl.DataFrame, targets: pl.LazyFrame) -> pl.LazyFrame:
    """Inline every certainty annotation as a wrmthorne/certainty group node with scalar children"""
    with_depth = (
        cert.lazy()
        .join(targets.select(pl.col("node_id").alias("target_node_id"), "depth"), on="target_node_id", how="inner")
        .with_columns(ordinal=pl.int_range(pl.len()).over("target_node_id"))
        .with_columns(gid=bin16("target_node_id", "kind", "ordinal", pl.lit("certainty")))
    )

    groups = with_depth.select(
        record_id="record_id",
        data_source="data_source",
        node_id="gid",
        parent_id="target_node_id",
        depth=(pl.col("depth") + 1).cast(pl.UInt8),
        label=pl.lit("Certainty"),
        path=pl.lit(None, dtype=pl.String),
        field_type=pl.lit("wrmthorne/certainty"),
        value=pl.lit(None, dtype=pl.String),
        as_recorded=pl.lit(None, dtype=pl.String),
        component="source",
    )

    children = (
        with_depth.with_columns(
            pl.col("confidence").round(4).cast(pl.String).alias("confidence"), comp=pl.col("source")
        )
        .select("record_id", "data_source", "gid", "depth", "comp", *CERT_PARTS.keys())
        .unpivot(
            index=["record_id", "data_source", "gid", "depth", "comp"], variable_name="part", value_name="part_value"
        )
        .filter(pl.col("part_value").is_not_null())
        .select(
            record_id="record_id",
            data_source="data_source",
            node_id=bin16("gid", "part", pl.lit("certainty_child")),
            parent_id="gid",
            depth=(pl.col("depth") + 2).cast(pl.UInt8),
            label=pl.col("part").str.replace("_", " ").str.to_titlecase(),
            path=pl.lit(None, dtype=pl.String),
            field_type=pl.col("part").replace_strict(CERT_PARTS),
            value="part_value",
            as_recorded=pl.lit(None, dtype=pl.String),
            component="comp",
        )
    )
    return pl.concat([groups, children])


@track_emissions(project_name="compile_records", output_dir=str(EMISSIONS_LOG_PATH), log_level="error")
def main() -> None:
    ap = argparse.ArgumentParser(description="Compile all sidecars into the normalised record set.")
    ap.add_argument("--data-source", action="append", help="restrict to institution(s), for subset test runs")
    ap.add_argument("--out", type=Path, default=COMPILED)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"inputs": {"raw": str(RAW_PATH), "sources": args.data_source}}
    t0 = time.time()

    tmp_base = args.out / "_base.tmp.parquet"
    log("building tier-0 base table…")
    base = build_base(tmp_base, args.data_source)
    n_base = base.select(pl.len()).collect(engine="streaming").item()
    log(f"base: {n_base:,} nodes")

    if PERIODS_CSV.exists():
        load_periods(PERIODS_CSV)
    (date_props, date_cert, period_nodes, date_semantic, date_verified, date_bounds, date_edtf) = compile_dates(
        base, args.out / "date_parse_cache.parquet", report
    )
    mpq_retypes, mpq_props = compile_measured_part_qualifiers(base, report)
    lifespan_props, death_nodes, death_retypes = compile_person_dates(base, report)
    retypes = pl.concat(
        [compile_associated_dates(base, report), mpq_retypes, compile_admin_concepts(base, report), death_retypes]
    )
    vocab_props, vocab_cert, vocab_nodes = compile_vocab(report)
    # The fallback carries its own component so reports separate them
    place_sidecars = [(PLACE_ANN, "places"), (PLACE_FALLBACK_ANN, "places_fallback")]
    place_frames = [
        compile_vocab(report, path, component, "ambiguous_place")
        for path, component in place_sidecars
        if path.exists()
    ]
    if place_frames:
        place_props, place_cert, place_nodes = (pl.concat([f[i] for f in place_frames]) for i in range(3))
    else:
        place_props, place_cert, place_nodes = (vocab_props.clear(), vocab_cert.clear(), vocab_nodes.clear())
    optional_auth = [PERSON_LINK_ANN, *(p for p, _ in place_sidecars)]
    auth_nodes = authority_nodes(base, [VOCAB_ANN] + [p for p in optional_auth if p.exists()], report)
    person_children, person_stale, person_nodes = compile_persons(base, report)
    add_nodes, strip_props, patch_cert, patch_conf = compile_patches(base, args.data_source, report)
    add_nodes = edtf_patch_dates(add_nodes, report)
    objnum_nodes = compile_object_numbers(base, report)
    (dim_props, dim_nodes, dim_cert, dim_touched, dim_targets, dim_conf) = compile_dimensions(
        base, args.out / "dimension_parse_cache.parquet", args.out, report
    )
    count_props, count_cert, count_touched = compile_counts(base, report)
    ident_props, ident_cert = compile_identifier_certainty(base, report)

    touched = pl.concat(
        [vocab_nodes, place_nodes, person_nodes, dim_touched, count_touched, strip_props.select("node_id")]
    ).unique()
    note_props, note_cert = compile_notation(base, touched, report)
    # These sets only tell compile_notation what earlier stages rewrote
    del touched, vocab_nodes, place_nodes, person_nodes, dim_touched, count_touched
    gc.collect()

    proposals = pl.concat(
        [
            date_props,
            lifespan_props,
            vocab_props,
            place_props,
            strip_props,
            note_props,
            dim_props,
            count_props,
            mpq_props,
            ident_props,
        ]
    )
    winners, conflicts = merge_proposals(base, proposals, report)
    conflicts = pl.concat([conflicts, person_stale, patch_conf, dim_conf])
    # every proposal is now a winner or a conflict
    del proposals, date_props, lifespan_props, vocab_props, place_props, strip_props
    del note_props, dim_props, count_props, mpq_props, ident_props
    gc.collect()
    log("proposal frames freed")

    # Targets restricted to compiled nodes, which matters on subset runs
    enc_cert = compile_encoding_damage(base, report)
    targets = pl.concat([base.select("node_id", "depth"), dim_targets.lazy()])
    cert = pl.concat(
        [date_cert, vocab_cert, place_cert, patch_cert, note_cert, dim_cert, count_cert, ident_cert, enc_cert]
    )
    cert = (
        cert.lazy()
        .join(targets.select(pl.col("node_id").alias("target_node_id")), on="target_node_id", how="semi")
        .collect(engine="streaming")
    )
    cert.write_parquet(args.out / "certainty_annotations.parquet")
    inline = certainty_nodes(cert, targets)
    del date_cert, vocab_cert, place_cert, patch_cert, note_cert, dim_cert, count_cert, ident_cert, enc_cert
    gc.collect()
    log("certainty frames freed")

    log("assembling compiled corpus…")
    # A value-stage component outranks the retype marker
    compiled = (
        base.join(winners.lazy(), on="node_id", how="left")
        .join(retypes.lazy(), on="node_id", how="left")
        .with_columns(
            final=pl.coalesce("new_value", "base_value"),
            component=pl.when(pl.col("new_value").is_not_null())
            .then(pl.col("component"))
            .when(pl.col("retype_field").is_not_null())
            .then(pl.col("retype_component"))
            # ne_missing: a placeholder nulled at tier 0 counts
            .when(pl.col("base_value").ne_missing(pl.col("value")))
            .then(pl.lit("tier0"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            field_type=pl.coalesce("retype_field", "field_type"),
            label=pl.coalesce("retype_label", "label"),
        )
        .with_columns(
            # ne_missing keeps the recorded form of a tier-0 nulled value
            as_recorded=pl.when(pl.col("final").ne_missing(pl.col("value")))
            .then(pl.col("value"))
            .otherwise(pl.lit(None, dtype=pl.String))
        )
        .drop("value", "base_value", "new_value", "contains_url", "retype_field", "retype_label", "retype_component")
        .rename({"final": "value"})
        .select(OUT_COLS)
    )

    # Sink as built: one plan peaks at tens of GB
    qualified_ids = cert.lazy().select(node_id="target_node_id").unique().collect(engine="streaming")
    verified_ids = date_verified.lazy().select("node_id").unique().collect(engine="streaming")
    semantic_ids = date_semantic.lazy().select("node_id").collect(engine="streaming")
    enriched_ids = (
        pl.concat(
            [
                period_nodes.lazy().select(node_id="parent_id"),
                date_bounds.select(node_id="parent_id"),
                date_edtf.select(node_id="parent_id"),
                auth_nodes.filter(pl.col("field_type") == AUTH_FIELD).select(node_id="parent_id"),
            ]
        )
        .unique()
        .drop_nulls()
        .collect(engine="streaming")
    )
    n_cert = cert.height
    del cert, date_verified, date_semantic
    gc.collect()
    log("flag id-sets taken")

    parts: list[Path] = []

    def emit(i: int, frame: pl.LazyFrame) -> None:
        path = args.out / f"_new{i}.tmp.parquet"
        screen_generated(source_only(frame).select(OUT_COLS)).sink_parquet(path)
        parts.append(path)
        gc.collect()
        log(f"  part {i} written")

    emit(0, person_children)
    del person_children
    emit(1, add_nodes.lazy())
    del add_nodes
    emit(2, objnum_nodes.lazy())
    del objnum_nodes
    emit(3, dim_nodes)
    del dim_nodes
    emit(4, inline)
    del inline
    emit(5, period_nodes.lazy())
    del period_nodes
    emit(6, death_nodes.lazy())
    del death_nodes
    emit(7, date_bounds)
    del date_bounds
    emit(8, auth_nodes)
    del auth_nodes
    emit(9, date_edtf)
    del date_edtf
    gc.collect()
    log("parts freed")

    # Read the damage off the sunk parts, keeping nothing resident
    damaged = (
        pl.scan_parquet(parts)
        .filter(pl.col("value").str.contains("�", literal=True))
        .select("record_id", "data_source", "node_id", "depth")
        .collect(engine="streaming")
    )
    if damaged.height:
        child_cert = damaged.select(
            target_node_id="node_id",
            record_id="record_id",
            data_source=pl.col("data_source").cast(pl.String),
            kind=pl.lit("encoding_damage"),
            notation=pl.lit(None, dtype=pl.String),
            source=pl.lit("decomposition"),
            confidence=pl.lit(None, dtype=pl.Float64),
            best_candidate=pl.lit(None, dtype=pl.String),
            detail=pl.lit(
                "value carries U+FFFD replacement characters inherited from the node it was decomposed from"
            ),
        )
        pl.concat([pl.read_parquet(args.out / "certainty_annotations.parquet"), child_cert]).write_parquet(
            args.out / "certainty_annotations.parquet"
        )
        emit(len(parts), certainty_nodes(child_cert, damaged.select("node_id", "depth").lazy()))
        qualified_ids = pl.concat([qualified_ids, child_cert.select(node_id="target_node_id")]).unique()
        n_cert += child_cert.height
    report["encoding_damage"]["decomposed_children"] = damaged.height
    log(f"encoding damage: {damaged.height:,} decomposition children annotated")
    del damaged
    gc.collect()
    new_nodes = pl.scan_parquet(parts)

    out_path = args.out / "mds-normalised.parquet"

    # Fold the trust signal on, from every routed queue
    routed_sidecars = [VOCAB_ANN, PERSON_ANN, PLACE_ANN, PLACE_FALLBACK_ANN]
    deferred_ids = pl.concat(
        [
            (
                pl.scan_parquet(path)
                .filter((pl.col("status") == "deferred") & ~pl.col("field_type").is_in(NON_VOCAB_FIELDS))
                .select("node_id")
            )
            for path in routed_sidecars
            if path.exists()
        ]
        # Date knowledge-state markers are routed rather than untouched
        + [semantic_ids.lazy()]
    ).unique()
    disp = pl.Enum(DISPOSITION)
    # One union to disk: four simultaneous joins were OOM-killed
    flags = (
        pl.concat(
            [
                qualified_ids.lazy().select("node_id", _flag=pl.lit("qualified")),
                deferred_ids.select("node_id", _flag=pl.lit("deferred")),
                enriched_ids.lazy().select("node_id", _flag=pl.lit("enriched")),
                verified_ids.lazy().select("node_id", _flag=pl.lit("verified")),
            ]
        )
        .with_columns(
            _rank=pl.col("_flag").replace_strict(
                {"qualified": 0, "deferred": 2, "enriched": 3, "verified": 4}, return_dtype=pl.Int8
            )
        )
        .sort("_rank")
        .unique(subset=["node_id"], keep="first")
        .select("node_id", "_flag")
    )
    flags_path = args.out / "_flags.tmp.parquet"
    flags.sink_parquet(flags_path)

    # The assembly runs last, once the flags are on disk
    del targets, qualified_ids, deferred_ids, verified_ids, enriched_ids, semantic_ids, flags
    gc.collect()
    log("stage frames freed")
    compiled.sink_parquet(args.out / "_compiled.tmp.parquet")

    final = (
        pl.scan_parquet([args.out / "_compiled.tmp.parquet", *parts])
        .join(pl.scan_parquet(flags_path), on="node_id", how="left")
        .with_columns(
            disposition=pl.when(pl.col("_flag") == "qualified")
            .then(pl.lit("qualified"))
            .when(pl.col("component").is_not_null())
            .then(pl.lit("applied"))
            .when(pl.col("_flag").is_not_null())
            .then(pl.col("_flag"))
            .otherwise(pl.lit("untouched"))
            .cast(disp)
        )
        .select(*OUT_COLS, "disposition")
    )
    final.sink_parquet(out_path)

    conflicts.write_parquet(args.out / "compile_conflicts.parquet")

    out = pl.scan_parquet(out_path)
    n_out = out.select(pl.len()).collect(engine="streaming").item()
    n_new = new_nodes.select(pl.len()).collect(engine="streaming").item()

    # Occurrences per (institution, field, owning stage, disposition)
    census = (
        out.group_by("data_source", "field_type", "component", "disposition")
        .agg(occurrences=pl.len())
        .sort("data_source", "field_type", "component", "disposition")
        .collect(engine="streaming")
    )
    census.write_parquet(args.out / "coverage_census.parquet")
    disp_totals = {d: int(census.filter(pl.col("disposition") == d)["occurrences"].sum()) for d in DISPOSITION}
    report["disposition"] = disp_totals

    checks = {
        "rows_in": n_base,
        "rows_new": n_new,
        "rows_out": n_out,
        "row_identity_holds": n_out == n_base + n_new,
        "changed_nodes": out.filter(pl.col("as_recorded").is_not_null())
        .select(pl.len())
        .collect(engine="streaming")
        .item(),
        "protected_changed": out.filter(pl.col("field_type").is_in(PROTECTED) & pl.col("as_recorded").is_not_null())
        .select(pl.len())
        .collect(engine="streaming")
        .item(),
        "certainty_annotations": n_cert,
        "conflicts": conflicts.height,
        "disposition_sum_holds": sum(disp_totals.values()) == n_out,
    }
    report["verify"] = checks
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    (args.out / "compile_report.json").write_text(json.dumps(report, indent=2))
    log(json.dumps(checks, indent=2))
    log(f"disposition: {disp_totals}")
    if not checks["row_identity_holds"]:
        raise RuntimeError("output row count mismatch")
    if checks["protected_changed"] != 0:
        raise RuntimeError("protected field was modified")
    if not checks["disposition_sum_holds"]:
        raise RuntimeError("disposition does not partition the rows")

    if not args.keep_tmp:
        tmp_base.unlink()
        (args.out / "_compiled.tmp.parquet").unlink()
        flags_path.unlink()
        for path in parts:
            path.unlink()
    log(f"done → {out_path}")


if __name__ == "__main__":
    main()
