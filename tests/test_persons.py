import polars as pl
import pytest

from mds_norm.pipeline import persons as p

AGENT_FIELD = "spectrum/object_production_person"


def cells(*rows):
    return pl.LazyFrame(rows, schema=["record_id", "node_id", "data_source", "field_type", "value"], orient="row")


def routed(*values):
    rows = [(f"r{i}", f"n{i}", "aberdeen", AGENT_FIELD, v) for i, v in enumerate(values)]
    return p.route(cells(*rows))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Dürer, Albrecht", "person"),
        ("Harland and Wolff Ltd", "organisation"),
        ("Doret family", "people"),
        ("attributed to Turner", "residue"),
        ("Smith, John (1820-1890)", "residue"),
        ("Smith, John; Jones, Mary", "multiple"),
        # a marker screens before the qualifier check
        ("unknown", "knowledge_state"),
    ],
)
def test_the_router_buckets_by_the_shape_of_the_string(value, expected):
    assert routed(value)["route"].to_list() == [expected]


def test_markers_never_reach_a_parser():
    frame = routed("n/a", "Not known").sort("value")
    assert set(frame["route"]) == {"placeholder", "knowledge_state"}


def test_a_parsed_name_resolves_with_its_rung_confidence():
    row = p.value_decisions(routed("Dürer, Albrecht")).row(0, named=True)
    assert (row["status"], row["entity_type"], row["surname"]) == ("resolved", "person", "Dürer")
    assert row["confidence"] == p.CONFIDENCE[row["sub_component"]]


def test_a_deferred_value_carries_its_reason_and_no_confidence():
    row = p.value_decisions(routed("Smith, John; Jones, Mary")).row(0, named=True)
    assert (row["status"], row["defer_reason"]) == ("deferred", "multiple")
    assert row["confidence"] is None


def test_the_router_route_resolves_organisations_without_parsing():
    row = p.value_decisions(routed("Harland and Wolff Ltd")).row(0, named=True)
    assert (row["status"], row["entity_type"], row["sub_component"]) == ("resolved", "organisation", "router")
    assert row["display"] == "Harland and Wolff Ltd"


def test_decisions_fan_out_to_every_occurrence():
    rows = cells(
        ("r0", "n0", "aberdeen", AGENT_FIELD, "Dürer, Albrecht"), ("r1", "n1", "leeds", AGENT_FIELD, "Dürer, Albrecht")
    )
    decisions = p.value_decisions(p.route(rows))
    ann = p.annotations(rows, decisions, p.mononym_decisions(rows, decisions))
    assert len(ann) == 2
    assert ann["surname"].to_list() == ["Dürer", "Dürer"]
    # the parsers consume the cell: the span covers everything
    assert ann["span_start"].to_list() == [0, 0]
    assert ann["span_end"].to_list() == [15, 15]


def test_an_organisation_only_field_types_a_lone_name():
    org_field = f"spectrum/{p.ORG_ONLY_FIELDS[0].split('/')[-1]}"
    rows = cells(("r0", "n0", "aberdeen", org_field, "Wolseley"), ("r1", "n1", "leeds", AGENT_FIELD, "Wolseley"))
    decisions = p.value_decisions(p.route(rows))
    assert decisions.row(0, named=True)["defer_reason"] == "no_surname"

    ann = p.annotations(rows, decisions, p.mononym_decisions(rows, decisions)).sort("node_id")
    org, person_field = ann.row(0, named=True), ann.row(1, named=True)
    # a field-keyed decision overrides only within its field
    assert (org["status"], org["entity_type"]) == ("resolved", "organisation")
    assert org["sub_component"] == "mononym_organisation"
    assert person_field["status"] == "deferred"


def test_whitespace_is_normalised_before_values_are_counted():
    rows = cells(("r0", "n0", "aberdeen", AGENT_FIELD, "  Dürer,   Albrecht "), ("r1", "n1", "leeds", AGENT_FIELD, ""))
    assert p.agent_cells(rows).collect()["value"].to_list() == ["Dürer, Albrecht"]
