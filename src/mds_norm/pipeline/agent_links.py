from __future__ import annotations

from pathlib import Path

import polars as pl
from codecarbon import EmissionsTracker

from mds_norm.paths import EMISSIONS_LOG, PERSON_ANNOTATIONS, PERSON_LINKS
from mds_norm.pipeline.build_parent_chains import best_labels
from mds_norm.pipeline.vocab_indexes import (
    ISNI_EXPORTS,
    KIND_PRIORITY,
    indexes,
    isni_wikidata,
    log,
    ulan_facets,
    ulan_wikidata,
)
from mds_norm.utils.atomise import norm_term

# `people` is a collective; deferred types have no matchable name
FACET_FOR = {"person": "person", "organisation": "corporate"}
SIDECAR_COLUMNS = [
    "record_id",
    "node_id",
    "data_source",
    "field_type",
    "value",
    "span_start",
    "span_end",
    "vocab",
    "subject",
    "matched_term",
    "authority_label",
    "n_candidates",
    "status",
    "defer_reason",
]


def ulan_candidates() -> pl.LazyFrame:
    """ULAN name forms keyed by norm and facet, carrying the subject count that decides ambiguity"""
    best = ["kind_p", "lang_p", "term"]
    return (
        indexes()["ulan"]
        .join(ulan_facets(), on="subject")
        .with_columns(
            kind_p=pl.col("kind").replace_strict(KIND_PRIORITY, return_dtype=pl.Int8),
            lang_p=(~pl.col("lang").str.starts_with("en")).cast(pl.Int8).fill_null(1),
        )
        .group_by("norm", "facet")
        .agg(
            n_candidates=pl.col("subject").n_unique().cast(pl.UInt32),
            subject=pl.col("subject").sort_by(best).first(),
            matched_term=pl.col("term").sort_by(best).first(),
        )
    )


def ulan_preferred() -> pl.LazyFrame:
    """The canonical surface per ULAN subject: prefLabelGVP over prefLabel, English first"""
    return best_labels(indexes()["ulan"]).lazy().rename({"label": "authority_label"})


def isni_bridge() -> pl.LazyFrame:
    """ISNIs reachable from a ULAN subject through a shared Wikidata entity, without matching names"""
    isni = pl.concat([isni_wikidata(entity_type) for entity_type in ISNI_EXPORTS])
    return (
        ulan_wikidata()
        .join(isni, on="qid")
        .group_by("subject")
        .agg(isni=pl.col("isni").unique())
        # two ISNIs for one subject: neither can be asserted
        .filter(pl.col("isni").list.len() == 1)
        .select("subject", isni=pl.col("isni").list.first())
    )


def link_agents(ann_path: Path = PERSON_ANNOTATIONS) -> pl.LazyFrame:
    """A ULAN subject per parsed agent whose name form belongs to exactly one, plus its ISNI"""
    ulan = ulan_links(ann_path)
    isni = (
        ulan.filter(pl.col("status") == "resolved")
        .join(isni_bridge(), on="subject")
        .with_columns(vocab=pl.lit("isni"), subject=pl.col("isni"))
        .select(SIDECAR_COLUMNS)
    )
    return pl.concat([ulan, isni])


def ulan_links(ann_path: Path = PERSON_ANNOTATIONS) -> pl.LazyFrame:
    unique_hit = pl.col("n_candidates") == 1
    return (
        pl.scan_parquet(ann_path)
        .filter((pl.col("status") == "resolved") & pl.col("entity_type").is_in(list(FACET_FOR)))
        .with_columns(
            norm=norm_term(pl.col("value")),
            facet=pl.col("entity_type").replace_strict(FACET_FOR, return_dtype=pl.String),
        )
        .join(ulan_candidates(), on=["norm", "facet"], how="inner")
        .join(ulan_preferred(), on="subject", how="left")
        .select(
            "record_id",
            "node_id",
            "data_source",
            "field_type",
            "value",
            "span_start",
            "span_end",
            "n_candidates",
            vocab=pl.lit("ulan"),
            # A name shared by several ULAN agents cannot be resolved
            subject=pl.when(unique_hit).then(pl.col("subject")),
            matched_term=pl.when(unique_hit).then(pl.col("matched_term")),
            authority_label=pl.when(unique_hit).then(pl.col("authority_label")),
            status=pl.when(unique_hit).then(pl.lit("resolved")).otherwise(pl.lit("deferred")),
            defer_reason=pl.when(~unique_hit).then(pl.lit("ambiguous_authority")),
        )
        .select(SIDECAR_COLUMNS)
    )


def main() -> None:
    EMISSIONS_LOG.mkdir(parents=True, exist_ok=True)
    with EmissionsTracker(project_name="agent_links", output_dir=str(EMISSIONS_LOG), log_level="error"):
        links = link_agents().collect(engine="streaming")
    links.write_parquet(PERSON_LINKS)
    log(f"agent links: {len(links):,} authority references → {PERSON_LINKS}")
    for vocab, rows in sorted(links.partition_by("vocab", as_dict=True).items()):
        resolved = rows.filter(pl.col("status") == "resolved")
        log(
            f"  {vocab[0]}: {len(resolved):,} resolved over {resolved['subject'].n_unique():,} subjects, "
            f"{len(rows) - len(resolved):,} deferred as ambiguous"
        )


if __name__ == "__main__":
    main()
