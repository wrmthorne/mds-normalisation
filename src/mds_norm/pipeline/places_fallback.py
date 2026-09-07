from __future__ import annotations

import json
import time
from collections.abc import Callable

import polars as pl
from codecarbon import EmissionsTracker

from mds_norm.paths import EMISSIONS_LOG, PLACE_ANNOTATIONS, PLACES_OUT, VOCAB_INDEXES, VOCABS
from mds_norm.pipeline.places_pipeline import parse_segments, route_values
from mds_norm.utils.atomise import norm_term

VOCAB_PATH = VOCABS
INDEX_DIR = VOCAB_INDEXES
OUT_DIR = PLACES_OUT
PLACE_ANN = PLACE_ANNOTATIONS
EMISSIONS_LOG_PATH = EMISSIONS_LOG

COMPONENT = "places_fallback"
TIER = 3  # after the TGN cascade, which is tier 2
CONSUMED = ("place_no_match", "sub_settlement")

# Below the TGN hierarchy rung: neither gazetteer is review-calibrated
CONFIDENCE = {"os_open_names": 0.9, "geonames": 0.85}
RESOLVED_BY_FACTOR = {"context_full": 1.0, "unique": 0.95, "prominent": 0.9}

POP_MIN = 5_000  # a population prior only speaks when the place is sizeable
POP_RATIO = 10  # …and dominates the runner-up by this factor
# Every OS record is in Great Britain, satisfying these terms
GB_CONTEXT = {"uk", "u.k", "united kingdom", "great britain", "britain", "gb", "england", "scotland", "wales"}
# Roads and vegetation excluded: those matches are usually coincidence
GEO_CLASSES = {"A", "P", "L", "T", "S", "H"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cached(name: str, build: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    path = INDEX_DIR / f"{name}.parquet"
    if not path.exists():
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        frame = build()
        frame.write_parquet(path)
        log(f"{name}: {frame['subject'].n_unique():,} places, {len(frame):,} names")
    return pl.read_parquet(path)


def load_queue() -> pl.LazyFrame:
    """Occurrence rows the TGN cascade deferred to a fallback authority"""
    return (
        pl.scan_parquet(PLACE_ANN)
        .filter(pl.col("defer_reason").is_in(CONSUMED))
        .with_columns(norm=norm_term(pl.col("atom")))
    )


def distinct_norms(queue: pl.LazyFrame) -> pl.DataFrame:
    return (
        queue.group_by("norm")
        .agg(atom=pl.col("atom").first(), count=pl.len(), n_institutions=pl.col("data_source").n_unique())
        .collect(engine="streaming")
        .filter(pl.col("norm") != "")
    )


OS_COLUMNS = [
    "ID",
    "NAMES_URI",
    "NAME1",
    "NAME1_LANG",
    "NAME2",
    "NAME2_LANG",
    "TYPE",
    "LOCAL_TYPE",
    "GEOMETRY_X",
    "GEOMETRY_Y",
    "MOST_DETAIL_VIEW_RES",
    "LEAST_DETAIL_VIEW_RES",
    "MBR_XMIN",
    "MBR_YMIN",
    "MBR_XMAX",
    "MBR_YMAX",
    "POSTCODE_DISTRICT",
    "POSTCODE_DISTRICT_URI",
    "POPULATED_PLACE",
    "POPULATED_PLACE_URI",
    "POPULATED_PLACE_TYPE",
    "DISTRICT_BOROUGH",
    "DISTRICT_BOROUGH_URI",
    "DISTRICT_BOROUGH_TYPE",
    "COUNTY_UNITARY",
    "COUNTY_UNITARY_URI",
    "COUNTY_UNITARY_TYPE",
    "REGION",
    "REGION_URI",
    "COUNTRY",
    "COUNTRY_URI",
    "RELATED_SPATIAL_OBJECT",
    "SAME_AS_DBPEDIA",
    "SAME_AS_GEONAMES",
]
OS_CONTEXT_COLUMNS = ["POPULATED_PLACE", "DISTRICT_BOROUGH", "COUNTY_UNITARY", "REGION", "COUNTRY"]


def build_os_index() -> pl.DataFrame:
    """(norm, subject, term, local_type, context) from the 819 grid-square CSVs"""
    data = VOCAB_PATH / "os_open_names" / "Data"
    frame = pl.scan_csv(data / "*.csv", has_header=False, new_columns=OS_COLUMNS, infer_schema_length=0).select(
        "ID", "NAME1", "NAME2", "LOCAL_TYPE", *OS_CONTEXT_COLUMNS
    )
    named = frame.unpivot(
        index=["ID", "LOCAL_TYPE", *OS_CONTEXT_COLUMNS],
        on=["NAME1", "NAME2"],
        variable_name="which",
        value_name="term",
    ).filter(pl.col("term").is_not_null() & (pl.col("term") != ""))
    return (
        named
        # NAMES_URI drops the 'osgb' prefix, so store the subject URI-ready
        .select(
            subject=pl.col("ID").str.strip_prefix("osgb"),
            term="term",
            local_type="LOCAL_TYPE",
            norm=norm_term(pl.col("term")),
            context=pl.concat_list([norm_term(pl.col(c).fill_null("")) for c in OS_CONTEXT_COLUMNS]),
        )
        .filter(pl.col("norm") != "")
        .with_columns(context=pl.col("context").list.set_difference(pl.lit([""], dtype=pl.List(pl.String))))
        .unique(["subject", "norm"])
        .collect(engine="streaming")
    )


GEO_COLUMNS = [
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "admin1_code",
    "admin2_code",
    "admin3_code",
    "admin4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date",
]


def _geonames_admin_names() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    base = VOCAB_PATH / "geonames"
    country = (
        pl.scan_csv(
            base / "countryInfo.txt", separator="\t", comment_prefix="#", has_header=False, infer_schema_length=0
        )
        .select(country_code=pl.col("column_1"), country_name=norm_term(pl.col("column_5")))
        .collect(engine="streaming")
    )
    admin1 = (
        pl.scan_csv(
            base / "admin1CodesASCII.txt", separator="\t", has_header=False, infer_schema_length=0, quote_char=None
        )
        .select(admin1_key=pl.col("column_1"), admin1_name=norm_term(pl.col("column_2")))
        .collect(engine="streaming")
    )
    admin2 = (
        pl.scan_csv(base / "admin2Codes.txt", separator="\t", has_header=False, infer_schema_length=0, quote_char=None)
        .select(admin2_key=pl.col("column_1"), admin2_name=norm_term(pl.col("column_2")))
        .collect(engine="streaming")
    )
    return country, admin1, admin2


def build_geonames_index(wanted: pl.DataFrame) -> pl.DataFrame:
    """(norm, subject, term, feature_class, population, context), restricted to the queue's norms"""
    base = VOCAB_PATH / "geonames"
    keep = wanted.select("norm").unique()
    country, admin1, admin2 = _geonames_admin_names()

    records = (
        pl.scan_csv(
            base / "allCountries.txt",
            separator="\t",
            has_header=False,
            new_columns=GEO_COLUMNS,
            infer_schema_length=0,
            quote_char=None,
        )
        .filter(pl.col("feature_class").is_in(sorted(GEO_CLASSES)))
        .select(
            "geonameid",
            "name",
            "asciiname",
            "feature_class",
            "country_code",
            "admin1_code",
            "admin2_code",
            "population",
        )
    )

    primary = (
        records.unpivot(
            index=["geonameid", "feature_class", "country_code", "admin1_code", "admin2_code", "population"],
            on=["name", "asciiname"],
            variable_name="which",
            value_name="term",
        )
        .filter(pl.col("term").is_not_null() & (pl.col("term") != ""))
        .with_columns(norm=norm_term(pl.col("term")))
        .join(keep.lazy(), on="norm", how="semi")
    )

    alt = (
        pl.scan_csv(
            base / "alternateNamesV2.txt",
            separator="\t",
            has_header=False,
            new_columns=[
                "altid",
                "geonameid",
                "isolanguage",
                "term",
                "is_preferred",
                "is_short",
                "is_colloquial",
                "is_historic",
                "from",
                "to",
            ],
            infer_schema_length=0,
            quote_char=None,
        )
        # link-type "languages" (link, wkdt, post…) are not names
        .filter(pl.col("isolanguage").is_null() | pl.col("isolanguage").is_in(["en", "cy", "gd", "ga", "abbr"]))
        .select("geonameid", "term")
        .filter(pl.col("term").is_not_null() & (pl.col("term") != ""))
        .with_columns(norm=norm_term(pl.col("term")))
        .join(keep.lazy(), on="norm", how="semi")
        .join(
            records.select("geonameid", "feature_class", "country_code", "admin1_code", "admin2_code", "population"),
            on="geonameid",
            how="inner",
        )
    )

    both = (
        pl.concat([primary.drop("which"), alt], how="diagonal")
        .with_columns(
            admin1_key=pl.col("country_code") + "." + pl.col("admin1_code"),
            admin2_key=(pl.col("country_code") + "." + pl.col("admin1_code") + "." + pl.col("admin2_code")),
        )
        .join(country.lazy(), on="country_code", how="left")
        .join(admin1.lazy(), on="admin1_key", how="left")
        .join(admin2.lazy(), on="admin2_key", how="left")
    )

    return (
        both.select(
            subject="geonameid",
            term="term",
            feature_class="feature_class",
            population=pl.col("population").cast(pl.Int64, strict=False).fill_null(0),
            norm="norm",
            context=pl.concat_list(
                pl.col("country_name").fill_null(""),
                pl.col("admin1_name").fill_null(""),
                pl.col("admin2_name").fill_null(""),
            ),
        )
        .with_columns(context=pl.col("context").list.set_difference(pl.lit([""], dtype=pl.List(pl.String))))
        .unique(["subject", "norm"])
        .collect(engine="streaming")
    )


def head_and_context(routed: pl.DataFrame) -> pl.DataFrame:
    """One row per (norm, orientation): the head to match and the context it must satisfy"""
    parsed, malformed = parse_segments(routed)
    simple = routed.filter(pl.col("route") == "simple").select(
        "norm",
        orientation=pl.lit("simple"),
        head_norm=pl.col("norm"),
        context=pl.lit([], dtype=pl.List(pl.String)),
        qualifier=pl.lit(None, dtype=pl.String),
    )
    return pl.concat([parsed, simple], how="vertical"), malformed


def resolve_against(index: pl.DataFrame, targets: pl.DataFrame, vocab: str, prominence: bool) -> pl.DataFrame:
    """The shared rung: match the head, then require the stated context"""
    cand = targets.join(index, left_on="head_norm", right_on="norm", how="inner")
    if not cand.height:
        return cand.clear().select(
            "norm",
            "orientation",
            "qualifier",
            subject=pl.lit(None, dtype=pl.String),
            matched_term=pl.lit(None, dtype=pl.String),
            n_candidates=pl.lit(None, dtype=pl.UInt32),
            resolved_by=pl.lit(None, dtype=pl.String),
            vocab=pl.lit(vocab),
        )

    gb = pl.lit(sorted(GB_CONTEXT), dtype=pl.List(pl.String))
    known = pl.col("known_context")
    satisfied = known.list.set_union(gb) if vocab == "os_open_names" else known
    cand = cand.rename({"context_right": "known_context"}).with_columns(
        # every context segment stated must sit in the candidate's
        context_ok=pl.col("context").list.set_difference(satisfied).list.len() == 0,
        stated=pl.col("context").list.len() > 0,
    )

    prom = pl.col("population") if prominence else pl.lit(0, dtype=pl.Int64)
    ranked = (
        cand.with_columns(prom=prom)
        .sort("prom", descending=True)
        .group_by("norm", "orientation", maintain_order=True)
        .agg(
            qualifier=pl.col("qualifier").first(),
            stated=pl.col("stated").first(),
            n_candidates=pl.col("subject").n_unique().cast(pl.UInt32),
            n_full=pl.col("context_ok").sum(),
            full_subject=pl.col("subject").filter("context_ok").first(),
            full_term=pl.col("term").filter("context_ok").first(),
            top_subject=pl.col("subject").first(),
            top_term=pl.col("term").first(),
            prom0=pl.col("prom").first(),
            prom1=pl.col("prom").sort(descending=True).slice(1, 1).first().fill_null(0),
        )
    )

    return ranked.select(
        "norm",
        "orientation",
        "qualifier",
        "n_candidates",
        subject=pl.when(pl.col("n_full") > 0).then(pl.col("full_subject")).otherwise(pl.col("top_subject")),
        matched_term=pl.when(pl.col("n_full") > 0).then(pl.col("full_term")).otherwise(pl.col("top_term")),
        # Test `stated` first: no context satisfies the context test vacuously
        resolved_by=pl.when(pl.col("stated") & (pl.col("n_full") == 1))
        .then(pl.lit("context_full"))
        .when(pl.col("stated"))
        .then(pl.lit(None, dtype=pl.String))
        .when(pl.col("n_candidates") == 1)
        .then(pl.lit("unique"))
        .when(prominence & (pl.col("prom0") >= POP_MIN) & (pl.col("prom0") >= POP_RATIO * (pl.col("prom1") + 1)))
        .then(pl.lit("prominent"))
        .otherwise(pl.lit(None, dtype=pl.String)),
        vocab=pl.lit(vocab),
    )


ORIENTATION_RANK = {"first": 0, "paren": 0, "last": 1, "simple": 0}
VOCAB_RANK = {"os_open_names": 0, "geonames": 1}


def best_per_norm(resolutions: pl.DataFrame) -> pl.DataFrame:
    """One decision per norm, resolved beating flagged and OS beating GeoNames"""
    return (
        resolutions.with_columns(
            resolved=pl.col("resolved_by").is_not_null(),
            v_rank=pl.col("vocab").replace_strict(VOCAB_RANK, return_dtype=pl.Int8),
            o_rank=pl.col("orientation").replace_strict(ORIENTATION_RANK, return_dtype=pl.Int8),
        )
        .sort(["resolved", "v_rank", "o_rank"], descending=[True, False, False])
        .unique("norm", keep="first", maintain_order=True)
        .drop("resolved", "v_rank", "o_rank")
    )


def assemble(routed: pl.DataFrame, malformed: pl.DataFrame, best: pl.DataFrame) -> pl.DataFrame:
    return (
        routed.join(best, on="norm", how="left")
        .join(malformed.with_columns(_m=pl.lit(True)), on="norm", how="left")
        .with_columns(
            sub_component=pl.when(pl.col("subject").is_not_null())
            .then(pl.col("vocab"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            status=pl.when(pl.col("resolved_by").is_not_null())
            .then(pl.lit("resolved"))
            .when(pl.col("subject").is_not_null())
            .then(pl.lit("flagged"))
            .otherwise(pl.lit("deferred")),
        )
        .with_columns(
            defer_reason=pl.when(pl.col("status") != "deferred")
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("_m").fill_null(False))
            .then(pl.lit("place_residue"))
            .otherwise(pl.lit("gazetteer_no_match")),
            confidence=pl.when(pl.col("status") == "resolved").then(
                pl.col("vocab").replace_strict(CONFIDENCE, return_dtype=pl.Float64, default=None)
                * pl.col("resolved_by").replace_strict(RESOLVED_BY_FACTOR, return_dtype=pl.Float64, default=1.0)
            ),
        )
        .drop("_m")
    )


def write_sidecar(queue: pl.LazyFrame, decisions: pl.DataFrame) -> int:
    ann = (
        queue.select(
            "record_id",
            "node_id",
            "data_source",
            "field_type",
            "value",
            "group",
            "atom",
            "span_start",
            "span_end",
            "norm",
        )
        .join(
            decisions.select(
                "norm",
                "vocab",
                "subject",
                "matched_term",
                "n_candidates",
                "resolved_by",
                "sub_component",
                "qualifier",
                "status",
                "defer_reason",
                "confidence",
            ).lazy(),
            on="norm",
            how="left",
        )
        .with_columns(
            score=pl.lit(None, dtype=pl.Float64),
            component=pl.lit(COMPONENT),
            tier=pl.when(pl.col("status").is_in(["resolved", "flagged"])).then(pl.lit(TIER, dtype=pl.Int32)),
        )
        .drop("norm")
    )
    out = OUT_DIR / "place_fallback_annotations.parquet"
    ann.sink_parquet(out)
    return pl.scan_parquet(out).select(pl.len()).collect().item()


def review_sample(decisions: pl.DataFrame, n_per_stratum: int = 20) -> pl.DataFrame:
    return (
        decisions.filter(pl.col("subject").is_not_null())
        .with_columns(stratum=pl.col("vocab") + "/" + pl.col("resolved_by").fill_null("flagged"))
        .sort("count", descending=True)
        .group_by("stratum")
        .head(n_per_stratum)
        .select(
            "stratum",
            "atom",
            "matched_term",
            "subject",
            "vocab",
            "resolved_by",
            "n_candidates",
            "count",
            "n_institutions",
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMISSIONS_LOG_PATH.mkdir(parents=True, exist_ok=True)
    with EmissionsTracker(project_name="places_fallback", output_dir=str(EMISSIONS_LOG_PATH), log_level="error"):
        queue = load_queue()
        distinct = distinct_norms(queue)
        log(f"queue: {len(distinct):,} distinct norms")

        routed = route_values(distinct)
        targets, malformed = head_and_context(routed)
        log(f"{len(targets):,} (norm, orientation) targets, {len(malformed):,} malformed")

        os_index = cached("os_open_names", build_os_index)
        geo_index = cached("geonames", lambda: build_geonames_index(targets.select(norm=pl.col("head_norm")).unique()))
        log(f"indexes: OS {len(os_index):,} names, GeoNames {len(geo_index):,} names")

        resolutions = pl.concat(
            [
                resolve_against(os_index, targets, "os_open_names", prominence=False),
                resolve_against(geo_index, targets, "geonames", prominence=True),
            ],
            how="diagonal",
        )
        best = best_per_norm(resolutions)
        decisions = assemble(routed, malformed, best)
        decisions.write_parquet(OUT_DIR / "place_fallback_decisions.parquet")

        rows = write_sidecar(queue, decisions)
        review_sample(decisions).write_csv(OUT_DIR / "place_fallback_review_sample.csv")

    by_status = decisions.group_by("status").agg(norms=pl.len(), occurrences=pl.col("count").sum())
    by_rung = (
        decisions.filter(pl.col("subject").is_not_null())
        .group_by("vocab", "resolved_by")
        .agg(norms=pl.len(), occurrences=pl.col("count").sum())
    )
    report = {
        "queue_norms": len(distinct),
        "queue_occurrences": int(distinct["count"].sum()),
        "annotation_rows": rows,
        "by_status": by_status.to_dicts(),
        "by_rung": by_rung.sort("occurrences", descending=True).to_dicts(),
    }
    (OUT_DIR / "places_fallback_report.json").write_text(json.dumps(report, indent=2))
    log(json.dumps(report["by_status"]))
    for row in report["by_rung"]:
        log(f"  {row['vocab']}/{row['resolved_by']}: {row['occurrences']:,} occ")


if __name__ == "__main__":
    main()
