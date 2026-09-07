from __future__ import annotations

import time

import polars as pl
from codecarbon import EmissionsTracker
from mds_data_model.introspection import agent_fields, fields_referencing
from mds_data_model.models.object import Object
from mds_data_model.models.organisation import Organisation
from mds_data_model.models.people import People
from mds_data_model.models.person import Person

from mds_norm.parsers.parse_person import NAME_FIELDS, parse_mononym_organisation, parse_person
from mds_norm.paths import EMISSIONS_LOG, FIELD_STATS, PERSON_ANNOTATIONS, PERSON_DECISIONS, PERSONS_OUT
from mds_norm.utils.atomise import PLACEHOLDER_MARKERS, SEMANTIC_MARKERS

COMPONENT = "persons"
TIER = 2

AGENT_FIELDS = list(agent_fields())
# Organisation-only fields: the field itself supplies the entity type
ORG_ONLY_FIELDS = [
    f"spectrum/{f}"
    for f in set(fields_referencing(Object, (Organisation,)))
    - set(fields_referencing(Object, (Person,)))
    - set(fields_referencing(Object, (People,)))
]

ORG_RX = (
    r"(?i)\b(ltd|limited|inc|plc|llp|gmbh|co|company|corporation|bros|museum|gallery|"
    r"society|association|university|college|school|council|committee|board|department|"
    r"ministry|institute|institution|library|archives?|studio|works|factory|press|"
    r"publishers?|railways?|regiment|church|club|band|orchestra|trust|foundation|guild|"
    r"union|agency|bank|hospital|pottery|potteries|manufactory)\b|& ?(co\b|sons?\b)"
)
PEOPLE_RX = r"(?i)\b(family|et al|and others|brothers|sisters|and (his|her|their) \w+)\b"
QUALIFIER_RX = (
    r"(?i)\b(attributed( to)?|after|school of|circle of|follower of|manner of|style of|"
    r"workshop of|studio of|possibly|probably|unknown|unidentified|anonymous|anon)\b"
)

# The organisation rungs sit lowest because neither verifies a name
CONFIDENCE = {
    "inverted": 0.95,
    "inverted_tail": 0.95,
    "slashed": 0.95,
    "angle_tail": 0.95,
    "crf": 0.85,
    "crf_corporation": 0.85,
    "crf_and_organisation": 0.8,
    "mononym_organisation": 0.8,
    "router": 0.8,
}
FIELD_KEYED = ["entity_type", "display", "sub_component", "confidence"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def agent_cells(field_stats: pl.LazyFrame) -> pl.LazyFrame:
    return (
        field_stats.filter(pl.col("field_type").is_in(AGENT_FIELDS) & pl.col("value").is_not_null())
        .select("record_id", "node_id", "data_source", "field_type", "value")
        .with_columns(pl.col("value").str.strip_chars().str.replace_all(r"\s+", " "))
        .filter(pl.col("value") != "")
    )


def route(cells: pl.LazyFrame) -> pl.DataFrame:
    """One row per distinct agent value, bucketed by the shape of the string"""
    distinct = (
        cells.group_by("value")
        .agg(count=pl.len(), n_institutions=pl.col("data_source").n_unique())
        .sort("count", descending=True)
        .collect(engine="streaming")
    )
    # Whole-value markers carry no entity; both defer, for different reasons
    marker = pl.col("value").str.to_lowercase().str.strip_chars().str.strip_chars_end(" .")
    return distinct.with_columns(
        route=pl.when(marker.is_in(sorted(PLACEHOLDER_MARKERS)))
        .then(pl.lit("placeholder"))
        .when(marker.is_in(sorted(SEMANTIC_MARKERS)))
        .then(pl.lit("knowledge_state"))
        .when(pl.col("value").str.contains(r"[\d()\[\]]") | pl.col("value").str.contains(QUALIFIER_RX))
        .then(pl.lit("residue"))
        .when(pl.col("value").str.contains(";"))
        .then(pl.lit("multiple"))
        .when(pl.col("value").str.contains(ORG_RX))
        .then(pl.lit("organisation"))
        .when(pl.col("value").str.contains(PEOPLE_RX))
        .then(pl.lit("people"))
        .otherwise(pl.lit("person"))
    )


def parse_person_route(routed: pl.DataFrame) -> pl.DataFrame:
    values = routed.filter(pl.col("route") == "person")["value"].to_list()
    parsed = [parse_person(v) for v in values]
    return pl.DataFrame(
        [
            {
                "value": value,
                "entity_type": entity,
                "sub_component": sub,
                "defer_reason": reason,
                **dict.fromkeys([*NAME_FIELDS, "display"]),
                **(name or {}),
            }
            for value, (name, entity, sub, reason) in zip(values, parsed, strict=True)
        ],
        schema_overrides=dict.fromkeys(
            ["value", "entity_type", "sub_component", "defer_reason", *NAME_FIELDS, "display"], pl.String
        ),
    )


def other_routes(routed: pl.DataFrame) -> pl.DataFrame:
    resolved = pl.col("route").is_in(["organisation", "people"])
    return routed.filter(pl.col("route") != "person").select(
        "value",
        entity_type=pl.col("route"),
        display=pl.when(resolved).then(pl.col("value")),
        defer_reason=pl.when(resolved).then(pl.lit(None)).otherwise(pl.col("route")),
        sub_component=pl.lit("router"),
    )


def value_decisions(routed: pl.DataFrame) -> pl.DataFrame:
    """One decision per distinct value, whichever rung reached it"""
    parsed = pl.col("defer_reason").is_null()
    return (
        pl.concat([parse_person_route(routed), other_routes(routed)], how="diagonal")
        .join(routed.select("value", "count"), on="value")
        .with_columns(
            component=pl.lit(COMPONENT),
            tier=pl.lit(TIER),
            status=pl.when(parsed).then(pl.lit("resolved")).otherwise(pl.lit("deferred")),
            confidence=pl.when(parsed).then(
                pl.col("sub_component").replace_strict(CONFIDENCE, return_dtype=pl.Float64)
            ),
        )
    )


def mononym_decisions(cells: pl.LazyFrame, decisions: pl.DataFrame) -> pl.DataFrame:
    """The one rung that cannot be keyed by value alone: a lone name the field itself types"""
    return (
        cells.filter(pl.col("field_type").is_in(ORG_ONLY_FIELDS))
        .select("value", "field_type")
        .unique()
        .join(decisions.lazy().filter(pl.col("defer_reason") == "no_surname").select("value"), on="value", how="semi")
        .collect(engine="streaming")
        .filter(
            pl.col("value").map_elements(lambda v: parse_mononym_organisation(v) is not None, return_dtype=pl.Boolean)
        )
        .with_columns(
            entity_type_field=pl.lit("organisation"),
            display_field=pl.col("value").str.strip_chars(),
            sub_component_field=pl.lit("mononym_organisation"),
            confidence_field=pl.lit(CONFIDENCE["mononym_organisation"]),
        )
    )


def annotations(cells: pl.LazyFrame, decisions: pl.DataFrame, mononyms: pl.DataFrame) -> pl.DataFrame:
    """Join decisions back by value onto every (record_id, node_id) occurrence"""
    resolved_here = pl.col("sub_component_field").is_not_null()
    return (
        cells.join(decisions.drop("count").lazy(), on="value", how="left")
        # a field-keyed decision overrides the value-level one
        .join(mononyms.lazy(), on=["value", "field_type"], how="left")
        .with_columns(**{c: pl.coalesce(f"{c}_field", c) for c in FIELD_KEYED}, resolved_here=resolved_here)
        .with_columns(
            defer_reason=pl.when("resolved_here")
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.col("defer_reason")),
            status=pl.when("resolved_here").then(pl.lit("resolved")).otherwise(pl.col("status")),
        )
        .drop("resolved_here", *[f"{c}_field" for c in FIELD_KEYED])
        .with_columns(span_start=pl.lit(0, dtype=pl.UInt32), span_end=pl.col("value").str.len_chars())
        .collect(engine="streaming")
    )


def main() -> None:
    PERSONS_OUT.mkdir(parents=True, exist_ok=True)
    EMISSIONS_LOG.mkdir(parents=True, exist_ok=True)
    with EmissionsTracker(project_name="persons", output_dir=str(EMISSIONS_LOG), log_level="error"):
        cells = agent_cells(pl.scan_parquet(FIELD_STATS))
        routed = route(cells)
        log(f"{len(routed):,} distinct values / {routed['count'].sum():,} occurrences")
        decisions = value_decisions(routed)
        decisions.write_parquet(PERSON_DECISIONS)
        log(f"{len(decisions):,} value decisions → {PERSON_DECISIONS}")
        mononyms = mononym_decisions(cells, decisions)
        log(f"{len(mononyms):,} (value, organisation-only field) pairs read as a mononym")
        ann = annotations(cells, decisions, mononyms)
    ann.write_parquet(PERSON_ANNOTATIONS)
    resolved = ann.filter(pl.col("status") == "resolved")
    log(f"{len(ann):,} annotation rows ({len(resolved) / len(ann):.1%} resolved) → {PERSON_ANNOTATIONS}")


if __name__ == "__main__":
    main()
