import polars as pl

from mds_norm.evaluation.validate_records import check, record_errors, summarise


def node(field_type, value, node_id=None, parent_id=None, depth=0, pos=0):
    return {
        "record_id": "r1",
        "data_source": "Test Museum",
        "node_id": node_id or field_type.encode(),
        "parent_id": parent_id,
        "depth": depth,
        "source_array_pos": pos,
        "field_type": field_type,
        "value": value,
    }


VALID = [
    node("spectrum/object_number", "1986.01"),
    node("spectrum/object_name", "chair", pos=1),
    node("ciim/license", None, node_id=b"l0", pos=2),
    node("ciim/value", "CC BY", node_id=b"l1", parent_id=b"l0", depth=1),
    node("ciim/license_url", "https://creativecommons.org/licenses/by/4.0/", node_id=b"l2", parent_id=b"l0", depth=1),
]


def errors(rows):
    return {(e["field"], e["error"]) for e in record_errors(rows)}


def test_a_minimal_record_validates():
    assert record_errors(VALID) == []


def test_required_field_absent():
    assert errors(VALID[1:]) == {("object_number", "missing")}


def test_unparsed_date_fails_the_iso_pattern():
    rows = [*VALID, node("spectrum/object_production_date", "c late 1940s", pos=2)]
    assert errors(rows) == {("object_production_date", "string_pattern_mismatch")}


def test_edtf_beyond_the_model_pattern_is_reported():
    rows = [*VALID, node("spectrum/associated_date", "-2399/-1799", pos=2)]
    assert errors(rows) == {("associated_date", "string_pattern_mismatch")}


def test_repeated_value_in_a_single_valued_field():
    rows = [
        *VALID,
        node("spectrum/accession_date", "1997", node_id=b"a1", pos=2),
        node("spectrum/accession_date", "1998", node_id=b"a2", pos=3),
    ]
    assert errors(rows) == {("accession_date", "multiple_values")}


def test_location_names_the_nested_slot():
    rows = [
        *VALID,
        node("spectrum/object_production_date", None, node_id=b"d0", pos=2),
        node("spectrum/date_earliest_single", "1870", node_id=b"d1", parent_id=b"d0", depth=1),
        node("spectrum/date_earliest_single", "1871", node_id=b"d2", parent_id=b"d0", depth=1, pos=1),
    ]
    (violation,) = record_errors(rows)
    assert violation["error"] == "multiple_values"
    assert violation["location"] == "object_production_date.date_earliest_single"


def test_field_outside_the_model():
    rows = [*VALID, node("spectrum/not_a_spectrum_field", "x", pos=2)]
    assert errors(rows) == {("not_a_spectrum_field", "unknown_field")}


def test_check_counts_one_verdict_per_record():
    rows = pl.DataFrame([*VALID, node("spectrum/object_name", "stool", node_id=b"n2", pos=2)])
    violations, verdicts = check(rows)
    assert violations.is_empty()
    assert verdicts.to_dicts() == [{"record_id": "r1", "data_source": "Test Museum", "valid": True}]


def test_summarise_counts_records_per_field_and_error():
    violations, _ = check(pl.DataFrame(VALID[1:]))
    assert summarise(violations).to_dicts() == [
        {"field": "object_number", "error": "missing", "records": 1, "locations": ["object_number"], "examples": []}
    ]
