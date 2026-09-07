import pytest

from mds_norm.utils.patches import FREE_TEXT, _source_key, parse_ops, validate_op, verify_value

FREE_TEXT_FIELD = sorted(FREE_TEXT)[0]  # any model free-text field will do


def req(**over):
    base = {
        "allowed": {"material": "spectrum/material", "object_production_date": "spectrum/object_production_date"},
        "nodes": {
            "description": [(b"n1", "spectrum/description", "A carved Oak chest, made around 1850.")],
            "note": [(b"n2", "spectrum/note", "Donated by the estate.")],
        },
    }
    return base | over


def op(**over):
    base = {"op": "add", "field": "material", "value": "Oak", "source_field": "description"}
    return base | over


def test_parse_ops_extracts_embedded_json():
    assert parse_ops('noise {"ops": [{"op": "add"}]} noise') == [{"op": "add"}]


@pytest.mark.parametrize("content", [None, "no json", '{"ops": 3}', '{"x": 1}'])
def test_parse_ops_junk_is_none(content):
    assert parse_ops(content) is None


def test_accepts_verbatim_copy_with_span():
    parsed, reason = validate_op(req(), op())
    assert reason is None
    assert parsed["field"] == "spectrum/material"
    assert parsed["value"] == "Oak"
    assert (parsed["span_start"], parsed["span_end"]) == (9, 12)
    assert parsed["source_field"] == "spectrum/description"


def test_recovers_source_exact_casing():
    parsed, reason = validate_op(req(), op(value="oak"))
    assert reason is None
    assert parsed["value"] == "Oak"  # span recovered from the source node


def test_rejects_paraphrase():
    assert validate_op(req(), op(value="oaken wood")) == (None, "copy_check_failed")


def test_rejects_unlisted_field():
    assert validate_op(req(), op(field="colour")) == (None, "field_not_admitted")


def test_rejects_bad_op_kind():
    assert validate_op(req(), op(op="replace")) == (None, "bad_op")
    assert validate_op(req(), "not an op") == (None, "not_an_object")


def test_rejects_empty_value():
    assert validate_op(req(), op(value="  ")) == (None, "empty_value")


def test_hallucinated_source_falls_back_to_whole_record():
    parsed, reason = validate_op(req(), op(source_field="no.such[0].key"))
    assert reason is None
    assert parsed["source_field"] == "spectrum/description"


def test_rejects_whole_cell_extraction_from_free_text():
    text = "Entire free text cell."
    r = req(nodes={"free": [(b"n3", FREE_TEXT_FIELD, text)]})
    assert validate_op(r, op(value=text, source_field="free")) == (None, "whole_cell_extraction")


def test_rejects_a_source_already_typed_as_the_destination_kind():
    r = req(nodes={"associated_date": [(b"n4", "spectrum/associated_date", "1850")]})
    assert validate_op(r, op(field="object_production_date", value="1850", source_field="associated_date")) == (
        None,
        "same_kind_source",
    )


def test_a_differently_typed_source_still_passes():
    r = req(nodes={"technique": [(b"n5", "spectrum/technique", "carved oak")]})
    parsed, reason = validate_op(r, op(source_field="technique", value="oak"))
    assert reason is None
    assert parsed["source_field"] == "spectrum/technique"


def test_tier1_verifier_gates_the_destination():
    parsed, reason = validate_op(req(), op(field="object_production_date", value="Oak"))
    assert (parsed, reason) == (None, "date_parse_failed")
    parsed, reason = validate_op(req(), op(field="object_production_date", value="1850"))
    assert reason is None
    assert parsed["value"] == "1850"


def test_verify_material_rules():
    assert verify_value("spectrum/material", "oak") is None
    assert verify_value("spectrum/material", "oak 2") == "material_has_digits"
    assert verify_value("spectrum/material", "a b c d e f g") == "material_too_long"


def test_verify_unknown_field():
    assert verify_value("spectrum/whatever", "x") == "unknown_target_field"


@pytest.mark.parametrize(
    ("src", "key"),
    [
        ("description", "description"),
        ("text[0].value", "text"),
        ("spectrum/material", "material"),
        ("record.note[2]", "note"),
    ],
)
def test_source_key_normalisation(src, key):
    assert _source_key(src) == key
