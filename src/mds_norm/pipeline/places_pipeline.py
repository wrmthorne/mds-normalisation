from __future__ import annotations

import json
import re
import time

import polars as pl
from codecarbon import EmissionsTracker

from mds_norm.paths import EMISSIONS_LOG, PLACES_OUT, VOCAB_ANNOTATIONS, VOCAB_INDEXES
from mds_norm.pipeline.vocab_indexes import tgn_ancestors
from mds_norm.utils.atomise import norm_term

INDEX_DIR = VOCAB_INDEXES
VOCAB_ANN = VOCAB_ANNOTATIONS
OUT_DIR = PLACES_OUT
EMISSIONS_LOG_PATH = EMISSIONS_LOG

COMPONENT = "places"
TIER = 2
KIND_PRIORITY = {"prefLabelGVP": 0, "prefLabel": 1, "altLabel": 2}
# the cascade's prominence rung thresholds, unchanged
PROMINENCE_MIN = 10
PROMINENCE_RATIO = 3
MAX_SEGMENTS = 6

# Hierarchy outranks bare exact-prominence: the context is corroboration
CONFIDENCE = {"hierarchy": 0.95, "paren_hierarchy": 0.9, "abbrev": 0.9}
RESOLVED_BY_FACTOR = {
    "context_full": 1.0,
    "context_full_kind": 0.95,
    "context_full_prominent": 0.92,
    "abbrev_unique": 1.0,
    "abbrev_prominent": 0.92,
}

# Hierarchy separators besides the comma, normalised before segmentation
SEPARATOR_RX = r"\s*[>:]\s*"
# A variant counts only if it exact-matches the TGN index
ABBREV_WORD = {"st": "saint"}

# Marker-plus-place compounds: they assert something and must persist
PLACE_SEMANTIC_MARKERS = {
    "unattributed place",
    "place unknown",
    "place not known",
    "place not recorded",
    "unknown place",
    "no known place",
    "place not stated",
    "unlocated",
    "findspot unknown",
    "provenance unknown",
}
# Mixed or qualified content the deterministic parser must not touch
RESIDUE_RX = re.compile(
    r"[\d\[\]{}?]|\b(near|nr|probably|possibly|perhaps|presumably|formerly|"
    r"vicinity|between|off|or)\b",
    re.IGNORECASE,
)
# Level suffixes label deferral queues only; no resolution uses them
STREET_RX = re.compile(
    r"\b(street|st|road|rd|lane|avenue|ave|terrace|crescent|row|court|close|"
    r"gate|quay|wharf|embankment|mews|gardens|walk|parade|drive|grove|place|square)$"
)
BUILDING_RX = re.compile(
    r"\b(hall|house|hospital|church|chapel|cathedral|abbey|priory|castle|tower|"
    r"mill|farm|school|college|university|station|inn|hotel|manor|lodge|barracks|"
    r"fort|palace|museum|gallery|library|works|factory|colliery|pit|quarry|mine|"
    r"cemetery|churchyard|park|pier|bridge|dock|docks|shipyard|brewery|pottery)$"
)
SITE_RX = re.compile(
    r"\b(cave|barrow|camp|hillfort|henge|tumulus|earthwork|villa|excavation|"
    r"site|kiln|midden|cairn|crannog|broch)$"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_queue() -> pl.LazyFrame:
    """Occurrence rows deferred to this pipeline, with the decision key `norm`"""
    return (
        pl.scan_parquet(VOCAB_ANN)
        .filter(pl.col("defer_reason") == "place_pipeline")
        .with_columns(norm=norm_term(pl.col("atom")))
    )


def distinct_norms(queue: pl.LazyFrame) -> pl.DataFrame:
    return (
        queue.group_by("norm")
        .agg(atom=pl.col("atom").first(), count=pl.len(), n_institutions=pl.col("data_source").n_unique())
        .collect(engine="streaming")
    )


def level_of(norm_col: pl.Expr) -> pl.Expr:
    return (
        pl.when(norm_col.str.contains(STREET_RX.pattern))
        .then(pl.lit("street"))
        .when(norm_col.str.contains(BUILDING_RX.pattern))
        .then(pl.lit("building"))
        .when(norm_col.str.contains(SITE_RX.pattern))
        .then(pl.lit("site"))
        .otherwise(pl.lit("area"))
    )


PAREN = r"^(.+?) ?\((.+)\)$"


def route_values(distinct: pl.DataFrame) -> pl.DataFrame:
    """semantic_marker / residue / hierarchical / simple, plus the level"""
    return distinct.with_columns(
        route=pl.when(pl.col("norm").is_in(sorted(PLACE_SEMANTIC_MARKERS)))
        .then(pl.lit("semantic_marker"))
        .when(pl.col("norm").str.contains(RESIDUE_RX.pattern))
        .then(pl.lit("residue"))
        .when(pl.col("norm").str.contains(r"[,>:]") | pl.col("norm").str.contains(PAREN))
        .then(pl.lit("hierarchical"))
        .otherwise(pl.lit("simple")),
        level=level_of(pl.col("norm")),
    )


def parse_segments(routed: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One row per (norm, orientation): head_norm plus context segment norms"""
    hier = routed.filter(pl.col("route") == "hierarchical").with_columns(
        sep_norm=pl.col("norm").str.replace_all(SEPARATOR_RX, ", "),
        sep_atom=pl.col("atom").str.replace_all(SEPARATOR_RX, ", "),
    )
    empty = pl.DataFrame(
        schema={
            "norm": pl.String,
            "orientation": pl.String,
            "head_norm": pl.String,
            "context": pl.List(pl.String),
            "qualifier": pl.String,
        }
    )

    comma = hier.filter(pl.col("sep_norm").str.contains(",", literal=True)).with_columns(
        segs=pl.col("sep_norm").str.split(",").list.eval(pl.element().str.strip_chars()),
        raw_segs=pl.col("sep_atom").str.split(",").list.eval(pl.element().str.strip_chars()),
    )
    bad = (
        (pl.col("segs").list.len() > MAX_SEGMENTS)
        | (pl.col("segs").list.eval(pl.element() == "").list.any())
        | (pl.col("segs").list.len() != pl.col("raw_segs").list.len())
    )
    malformed = comma.filter(bad).select("norm")
    comma = comma.filter(~bad)

    if comma.height:  # several list ops mistype on an empty frame
        # qualifiers via string ops: list.join-after-slice mistypes its output
        first = comma.select(
            "norm",
            orientation=pl.lit("first"),
            head_norm=pl.col("segs").list.first(),
            context=pl.col("segs").list.slice(1),
            qualifier=pl.col("sep_atom").str.replace(r"^[^,]*,\s*", ""),
        )
        last = comma.select(
            "norm",
            orientation=pl.lit("last"),
            head_norm=pl.col("segs").list.last(),
            context=pl.col("segs").list.slice(0, pl.col("segs").list.len() - 1),
            qualifier=pl.col("sep_atom").str.replace(r"\s*,[^,]*$", ""),
        )
    else:
        first = last = empty

    paren = (
        hier.filter(~pl.col("sep_norm").str.contains(",", literal=True))
        .with_columns(head=norm_term(pl.col("norm").str.extract(PAREN, 1)), qual=pl.col("norm").str.extract(PAREN, 2))
        .filter((pl.col("head") != "") & pl.col("qual").is_not_null())
        .select(
            "norm",
            orientation=pl.lit("paren"),
            head_norm="head",
            context=pl.concat_list(norm_term(pl.col("qual"))),
            qualifier=pl.col("atom").str.extract(PAREN, 2),
        )
    )
    if not paren.height:
        paren = empty

    return pl.concat([first, last, paren]), malformed


def abbrev_variants(norm: str) -> list[str]:
    """Deterministic spelling variants, most-conservative first, anchored on the vocabulary"""
    variants: list[str] = []

    def push(v: str) -> None:
        if v and v != norm and v not in variants:
            variants.append(v)

    undotted = re.sub(r"\s+", " ", norm.replace(".", " ")).strip()
    push(undotted)
    for base in (norm, undotted):
        toks = base.split(" ")
        if any(t in ABBREV_WORD for t in toks):
            push(" ".join(ABBREV_WORD.get(t, t) for t in toks))
    return variants


def _lookup(index: pl.LazyFrame, norms: pl.DataFrame) -> pl.DataFrame:
    """Exact candidates per norm: best matching term per subject"""
    return (
        index.join(norms.select("norm").lazy(), on="norm", how="semi")
        .with_columns(
            kind_p=pl.col("kind").replace_strict(KIND_PRIORITY, return_dtype=pl.Int8),
            lang_p=(~pl.col("lang").str.starts_with("en")).cast(pl.Int8).fill_null(1),
        )
        .group_by("norm", "subject")
        .agg(kind_p=pl.col("kind_p").min(), matched_term=pl.col("term").sort_by(["kind_p", "lang_p"]).first())
        .collect(engine="streaming")
    )


def variant_pairs(norms: pl.DataFrame) -> pl.DataFrame:
    """(norm, vnorm, prio) rows for every abbreviation variant"""
    return (
        norms.select("norm")
        .unique()
        .with_columns(vnorm=pl.col("norm").map_elements(abbrev_variants, return_dtype=pl.List(pl.String)))
        .explode("vnorm")
        .drop_nulls("vnorm")
        .with_row_index("prio")
    )


def head_candidates(index: pl.LazyFrame, head_norms: pl.DataFrame) -> pl.DataFrame:
    """Direct exact candidates plus abbreviation-variant candidates per head norm"""
    heads = head_norms.rename({"head_norm": "norm"})
    direct = _lookup(index, heads)
    pairs = variant_pairs(heads)
    via = (
        pairs.join(_lookup(index, pairs.select(norm="vnorm").unique()).rename({"norm": "vnorm"}), on="vnorm")
        .sort("prio")
        .select("norm", "subject", "kind_p", "matched_term")
    )
    return (
        pl.concat([direct, via])
        .unique(["norm", "subject"], keep="first", maintain_order=True)
        .rename({"norm": "head_norm"})
    )


def context_matches(
    index: pl.LazyFrame, ancestors: pl.LazyFrame, candidates: pl.DataFrame, ctx_norms: pl.DataFrame
) -> pl.DataFrame:
    """(subject, ctx_norm) pairs where some ancestor of subject carries the context norm as a term"""
    anc_terms = (
        index.join(ctx_norms.lazy(), on="norm", how="semi").select(ancestor="subject", ctx_norm="norm").unique()
    )
    return (
        ancestors.join(candidates.select("subject").unique().lazy(), on="subject", how="semi")
        .join(anc_terms, on="ancestor")
        .select("subject", "ctx_norm")
        .unique()
        .collect(engine="streaming")
    )


def resolve_hierarchical(
    parsed: pl.DataFrame, candidates: pl.DataFrame, cand_ctx: pl.DataFrame, prominence: pl.LazyFrame
) -> pl.DataFrame:
    """Per norm: the best (subject, term) with how it was settled"""
    scored = (
        parsed.with_columns(n_ctx=pl.col("context").list.len())
        .join(candidates, on="head_norm")
        .explode("context")
        .join(
            cand_ctx.rename({"ctx_norm": "context"}).with_columns(hit=pl.lit(True)),
            on=["subject", "context"],
            how="left",
        )
        .group_by("norm", "orientation", "qualifier", "subject", "kind_p", "matched_term", "n_ctx")
        .agg(n_matched=pl.col("hit").fill_null(False).sum())
        .join(prominence.collect(engine="streaming"), on="subject", how="left")
        .with_columns(pl.col("prominence").fill_null(0), full=pl.col("n_matched") == pl.col("n_ctx"))
    )

    # keep the winning orientation per norm: full-context first, narrow-first tie-break
    orient_rank = (
        scored.group_by("norm", "orientation")
        .agg(any_full=pl.col("full").any(), best_matched=pl.col("n_matched").max())
        .sort(["any_full", "best_matched", pl.col("orientation") == "first"], descending=True)
        .unique("norm", keep="first", maintain_order=True)
        .select("norm", "orientation")
    )
    scored = scored.join(orient_rank, on=["norm", "orientation"])

    full = scored.filter("full")
    stats = full.group_by("norm").agg(n_full=pl.col("subject").n_unique(), min_kind=pl.col("kind_p").min())
    best = (
        full.join(stats, on="norm")
        .filter(pl.col("kind_p") == pl.col("min_kind"))
        .sort(["kind_p", "prominence", "subject"], descending=[False, True, False])
        .group_by("norm", maintain_order=True)
        .agg(
            subject=pl.col("subject").first(),
            matched_term=pl.col("matched_term").first(),
            qualifier=pl.col("qualifier").first(),
            orientation=pl.col("orientation").first(),
            n_candidates=pl.col("n_full").first().cast(pl.UInt32),
            n_best=pl.col("subject").n_unique(),
            prom_top=pl.col("prominence").sort(descending=True).head(2),
        )
        .with_columns(
            prom0=pl.col("prom_top").list.get(0, null_on_oob=True).fill_null(0),
            prom1=pl.col("prom_top").list.get(1, null_on_oob=True).fill_null(0),
        )
        .with_columns(
            resolved_by=pl.when(pl.col("n_candidates") == 1)
            .then(pl.lit("context_full"))
            .when(pl.col("n_best") == 1)
            .then(pl.lit("context_full_kind"))
            .when((pl.col("prom0") >= PROMINENCE_MIN) & (pl.col("prom0") >= PROMINENCE_RATIO * (pl.col("prom1") + 1)))
            .then(pl.lit("context_full_prominent"))
        )
        .drop("n_best", "prom_top", "prom0", "prom1")
    )

    partial = (
        scored.join(best.select("norm"), on="norm", how="anti")
        .filter(pl.col("n_matched") > 0)
        .sort(["n_matched", "kind_p", "prominence", "subject"], descending=[True, False, True, False])
        .group_by("norm", maintain_order=True)
        .agg(
            subject=pl.col("subject").first(),
            matched_term=pl.col("matched_term").first(),
            qualifier=pl.col("qualifier").first(),
            orientation=pl.col("orientation").first(),
            n_candidates=pl.col("subject").n_unique().cast(pl.UInt32),
        )
        .with_columns(resolved_by=pl.lit(None, dtype=pl.String))
    )

    # Head matched, no context: flag narrow-first and paren readings
    no_ctx = (
        scored.join(best.select("norm"), on="norm", how="anti")
        .join(partial.select("norm"), on="norm", how="anti")
        .filter(pl.col("orientation").is_in(["first", "paren"]))
        .sort(["kind_p", "prominence", "subject"], descending=[False, True, False])
        .group_by("norm", maintain_order=True)
        .agg(
            subject=pl.col("subject").first(),
            matched_term=pl.col("matched_term").first(),
            qualifier=pl.col("qualifier").first(),
            orientation=pl.col("orientation").first(),
            n_candidates=pl.col("subject").n_unique().cast(pl.UInt32),
        )
        .with_columns(resolved_by=pl.lit(None, dtype=pl.String))
    )

    return pl.concat(
        [
            best.select(
                "norm", "subject", "matched_term", "qualifier", "orientation", "n_candidates", "resolved_by"
            ).with_columns(
                ctx_state=pl.when(pl.col("resolved_by").is_not_null())
                .then(pl.lit("full"))
                .otherwise(pl.lit("full_unsettled"))
            ),
            partial.select(
                "norm", "subject", "matched_term", "qualifier", "orientation", "n_candidates", "resolved_by"
            ).with_columns(ctx_state=pl.lit("partial")),
            no_ctx.select(
                "norm", "subject", "matched_term", "qualifier", "orientation", "n_candidates", "resolved_by"
            ).with_columns(ctx_state=pl.lit("none")),
        ]
    )


def resolve_simple(routed: pl.DataFrame, index: pl.LazyFrame, prominence: pl.LazyFrame) -> pl.DataFrame:
    """Abbreviation-variant rung for single-segment values"""
    pairs = variant_pairs(routed.filter(pl.col("route") == "simple"))
    cand = (
        pairs.join(_lookup(index, pairs.select(norm="vnorm").unique()).rename({"norm": "vnorm"}), on="vnorm")
        .sort("prio")
        .unique(["norm", "subject"], keep="first", maintain_order=True)
        .join(prominence.collect(engine="streaming"), on="subject", how="left")
        .with_columns(pl.col("prominence").fill_null(0))
    )
    if not cand.height:  # several list ops mistype on an empty frame
        return pl.DataFrame(
            schema={
                "norm": pl.String,
                "subject": pl.String,
                "matched_term": pl.String,
                "qualifier": pl.String,
                "orientation": pl.String,
                "n_candidates": pl.UInt32,
                "resolved_by": pl.String,
                "ctx_state": pl.String,
            }
        )
    return (
        cand.sort(["kind_p", "prominence", "subject"], descending=[False, True, False])
        .group_by("norm", maintain_order=True)
        .agg(
            subject=pl.col("subject").first(),
            matched_term=pl.col("matched_term").first(),
            n_candidates=pl.col("subject").n_unique().cast(pl.UInt32),
            prom_top=pl.col("prominence").sort(descending=True).head(2),
        )
        .with_columns(
            prom0=pl.col("prom_top").list.get(0, null_on_oob=True).fill_null(0),
            prom1=pl.col("prom_top").list.get(1, null_on_oob=True).fill_null(0),
        )
        .with_columns(
            resolved_by=pl.when(pl.col("n_candidates") == 1)
            .then(pl.lit("abbrev_unique"))
            .when((pl.col("prom0") >= PROMINENCE_MIN) & (pl.col("prom0") >= PROMINENCE_RATIO * (pl.col("prom1") + 1)))
            .then(pl.lit("abbrev_prominent")),
            qualifier=pl.lit(None, dtype=pl.String),
            orientation=pl.lit("simple"),
            ctx_state=pl.lit("abbrev"),
        )
        .select(
            "norm", "subject", "matched_term", "qualifier", "orientation", "n_candidates", "resolved_by", "ctx_state"
        )
    )


def assemble_decisions(
    routed: pl.DataFrame, parsed: pl.DataFrame, malformed: pl.DataFrame, resolutions: pl.DataFrame
) -> pl.DataFrame:
    """Fold router and cascade results into one decision row per norm"""
    head_first = (
        parsed.filter(pl.col("orientation").is_in(["first", "paren"]))
        .unique("norm", keep="first", maintain_order=True)
        .select("norm", "head_norm")
    )
    return (
        routed.join(head_first, on="norm", how="left")
        .with_columns(level=level_of(pl.coalesce("head_norm", "norm")))
        .drop("head_norm")
        .join(resolutions, on="norm", how="left")
        .join(malformed.with_columns(_m=pl.lit(True)), on="norm", how="left")
        .with_columns(
            sub_component=pl.when(pl.col("subject").is_null())
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("orientation") == "paren")
            .then(pl.lit("paren_hierarchy"))
            .when(pl.col("orientation") == "simple")
            .then(pl.lit("abbrev"))
            .otherwise(pl.lit("hierarchy")),
            status=pl.when(pl.col("resolved_by").is_not_null())
            .then(pl.lit("resolved"))
            .when(pl.col("subject").is_not_null())
            .then(pl.lit("flagged"))
            .otherwise(pl.lit("deferred")),
        )
        .with_columns(
            defer_reason=pl.when(pl.col("status") != "deferred")
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("route") == "semantic_marker")
            .then(pl.lit("semantic_marker"))
            .when((pl.col("route") == "residue") | pl.col("_m").fill_null(False))
            .then(pl.lit("place_residue"))
            .when(pl.col("level").is_in(["street", "building", "site"]))
            .then(pl.lit("sub_settlement"))
            .otherwise(pl.lit("place_no_match")),
            confidence=pl.when(pl.col("status") == "resolved").then(
                pl.col("sub_component").replace_strict(CONFIDENCE, return_dtype=pl.Float64, default=None)
                * pl.col("resolved_by").replace_strict(RESOLVED_BY_FACTOR, return_dtype=pl.Float64, default=1.0)
            ),
            vocab=pl.when(pl.col("subject").is_not_null()).then(pl.lit("tgn")),
        )
        .drop("_m", "orientation")
    )


def write_sidecar(queue: pl.LazyFrame, decisions: pl.DataFrame) -> int:
    """Join decisions back onto every occurrence row, vocab_annotations schema"""
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
    ann.sink_parquet(OUT_DIR / "place_annotations.parquet")
    return pl.scan_parquet(OUT_DIR / "place_annotations.parquet").select(pl.len()).collect(engine="streaming").item()


def review_sample(decisions: pl.DataFrame, n_per_stratum: int = 20) -> pl.DataFrame:
    """Head plus random per (status, resolved_by/ctx_state) stratum"""
    judged = decisions.filter(pl.col("status").is_in(["resolved", "flagged"])).with_columns(
        stratum=pl.coalesce("resolved_by", "ctx_state")
    )
    parts = []
    for (_stratum,), grp in judged.group_by("stratum"):
        parts.append(
            pl.concat(
                [
                    grp.sort("count", descending=True).head(n_per_stratum),
                    grp.sample(min(n_per_stratum, grp.height), seed=0),
                ]
            ).unique("norm", maintain_order=True)
        )
    return pl.concat(parts).select(
        "atom", "stratum", "status", "matched_term", "qualifier", "subject", "n_candidates", "count"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMISSIONS_LOG_PATH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: dict = {}

    with EmissionsTracker(project_name="places_pipeline", output_dir=str(EMISSIONS_LOG_PATH), log_level="error"):
        index = pl.scan_parquet(INDEX_DIR / "tgn.parquet")
        prominence = pl.scan_parquet(INDEX_DIR / "tgn_children.parquet")
        ancestors = tgn_ancestors()

        queue = load_queue()
        distinct = distinct_norms(queue)
        log(f"queue: {distinct.height:,} distinct norms / {distinct['count'].sum():,} occurrences")
        report["queue"] = {"distinct": distinct.height, "occurrences": int(distinct["count"].sum())}

        routed = route_values(distinct)
        report["routes"] = {
            r: {"distinct": int(d), "occurrences": int(o)}
            for r, d, o in routed.group_by("route").agg(pl.len(), pl.col("count").sum()).iter_rows()
        }
        log(f"routes: {report['routes']}")

        parsed, malformed = parse_segments(routed)
        heads = parsed.select("head_norm").unique()
        ctx = parsed.explode("context").select(norm="context").filter(pl.col("norm") != "").unique()
        candidates = head_candidates(index, heads)
        log(
            f"parsed: {parsed.select(pl.col('norm').n_unique()).item():,} "
            f"hierarchical norms; {candidates.height:,} head candidates; "
            f"{ctx.height:,} distinct context norms"
        )

        cand_ctx = context_matches(index, ancestors, candidates, ctx)
        resolutions = pl.concat(
            [resolve_hierarchical(parsed, candidates, cand_ctx, prominence), resolve_simple(routed, index, prominence)]
        )
        decisions = assemble_decisions(routed, parsed, malformed, resolutions)
        decisions.write_parquet(OUT_DIR / "place_value_decisions.parquet")

        outcome = (
            decisions.group_by("status", "resolved_by", "ctx_state", "defer_reason")
            .agg(distinct=pl.len(), occurrences=pl.col("count").sum())
            .sort("occurrences", descending=True)
        )
        report["outcomes"] = [
            {k: (int(v) if isinstance(v, int) else v) for k, v in row.items()} for row in outcome.iter_rows(named=True)
        ]
        for row in outcome.iter_rows(named=True):
            log(f"  {row}")

        n_ann = write_sidecar(queue, decisions)
        log(f"{n_ann:,} annotation rows → {OUT_DIR / 'place_annotations.parquet'}")

        review = review_sample(decisions)
        review.write_csv(OUT_DIR / "place_review_sample.csv")

        occ = decisions["count"].sum()
        res = decisions.filter(pl.col("status") == "resolved")["count"].sum()
        flg = decisions.filter(pl.col("status") == "flagged")["count"].sum()
        report["coverage"] = {
            "resolved_occ_share": round(res / occ, 4),
            "flagged_occ_share": round(flg / occ, 4),
            "annotation_rows": n_ann,
        }
        report["elapsed_seconds"] = round(time.time() - t0, 1)
        (OUT_DIR / "places_report.json").write_text(json.dumps(report, indent=2))
        log(f"resolved {res / occ:.1%} / flagged {flg / occ:.1%} of queue occurrences in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
