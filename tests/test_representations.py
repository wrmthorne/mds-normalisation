import pytest

from experiments.extraction_common import parse_ops_diff, parse_ops_fields


def test_diff_parses_lines():
    content = (
        "+ material: oak (from description)\nsome commentary\n+ object_production_date: c. 1850 (from history.note)\n"
    )
    assert parse_ops_diff(content) == [
        {"op": "add", "field": "material", "value": "oak", "source_field": "description"},
        {"op": "add", "field": "object_production_date", "value": "c. 1850", "source_field": "history.note"},
    ]


def test_diff_empty_response_is_valid_empty():
    assert parse_ops_diff("nothing to extract") == []
    assert parse_ops_diff("") == []


def test_diff_malformed_findings_defer():
    # emitted "+" findings but none in-format: unparseable, not silently empty
    assert parse_ops_diff("+ material oak") is None


def test_diff_none_defers():
    assert parse_ops_diff(None) is None


def test_fields_parses_lists_and_scalars():
    ops = parse_ops_fields('{"material": ["oak", "brass"], "dimension": "10cm"}')
    assert [(o["field"], o["value"]) for o in ops] == [
        ("material", "oak"),
        ("material", "brass"),
        ("dimension", "10cm"),
    ]
    assert all(o["source_field"] is None for o in ops)


def test_fields_empty_object_is_valid_empty():
    assert parse_ops_fields("{}") == []


def test_fields_skips_non_string_values():
    assert parse_ops_fields('{"material": [1, "oak"], "n": 3}') == [
        {"op": "add", "field": "material", "value": "oak", "source_field": None}
    ]


@pytest.mark.parametrize("content", [None, "not json", "[1,2]", '"str"'])
def test_fields_junk_defers(content):
    assert parse_ops_fields(content) is None
