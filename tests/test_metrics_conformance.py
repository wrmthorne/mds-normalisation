import polars as pl
import pytest

from mds_norm.metrics import conformance
from mds_norm.metrics.common import numeric_field_names, positive_int_field_names


def compute(rows):
    # an optional 5th element is the parent_id
    rows = [r if len(r) == 5 else (*r, None) for r in rows]
    base = pl.LazyFrame(
        rows,
        schema={
            "record_id": pl.String,
            "data_source": pl.String,
            "field_type": pl.String,
            "value": pl.String,
            "parent_id": pl.Binary,
        },
        orient="row",
    )
    return conformance.compute(base).sort("record_id")


def test_model_field_sets_come_from_annotations():
    assert positive_int_field_names() == {"spectrum/age", "spectrum/number_of_objects"}
    assert "spectrum/dimension_value" in numeric_field_names()
    # the shared generic alias is never a numeric slot
    assert "spectrum/value" not in numeric_field_names()
    assert not any(f.startswith("wrmthorne/") for f in numeric_field_names())


def test_release_namespace_invisible_to_checks():
    out = compute(
        [
            ("r1", "M", "spectrum/title", "A carved oak chest"),
            # identical scalar children of two certainty groups: not a duplicate
            ("r1", "M", "wrmthorne/kind", "ambiguous_homograph"),
            ("r1", "M", "wrmthorne/kind", "ambiguous_homograph"),
        ]
    )
    row = out.row(0, named=True)
    assert row["pass_schema_field"] is True  # wrmthorne is not "unknown"
    assert row["pass_duplicate_stmt"] is True


def test_duplicate_stmt_scoped_to_parent_group():
    # repeats across sibling groups are not duplicates
    out = compute(
        [
            ("r1", "M", "spectrum/title", "Oak chest"),
            ("r1", "M", "spectrum/dimension_measurement_unit", "cm", b"g1"),
            ("r1", "M", "spectrum/dimension_measurement_unit", "cm", b"g2"),
            ("r2", "M", "spectrum/title", "Bronze bowl"),
            ("r2", "M", "spectrum/dimension_measurement_unit", "cm", b"g3"),
            ("r2", "M", "spectrum/dimension_measurement_unit", "cm", b"g3"),
            # depth-0 repeats still fail (both share a null parent)
            ("r3", "M", "spectrum/material", "oak"),
            ("r3", "M", "spectrum/material", "oak"),
        ]
    )
    dup = dict(zip(out["record_id"], out["pass_duplicate_stmt"], strict=True))
    assert dup["r1"] is True
    assert dup["r2"] is False
    assert dup["r3"] is False


def test_html_check_ignores_coded_enumerations():
    # coded '<NN> ' prefixes are recording practice, not markup
    out = compute(
        [
            ("r1", "M", "spectrum/object_name_type", "<01> Title"),
            ("r2", "M", "spectrum/brief_description", "<p>Oak chest</p>"),
        ]
    )
    html = dict(zip(out["record_id"], out["pass_html"], strict=True))
    assert html["r1"] is True
    assert html["r2"] is False


def test_html_check_ignores_bracketed_content():
    # a tag has to name an HTML element
    out = compute(
        [
            ("r1", "M", "spectrum/object_name", "print <b/w>"),
            ("r2", "M", "spectrum/object_name", "print <sepia>"),
            ("r3", "M", "spectrum/object_name", "<lutelike chordophones with long neck: plucked>"),
            ("r4", "M", "spectrum/brief_description", '<DIV STYLE="text-align:Justify;">x'),
            ("r5", "M", "spectrum/brief_description", "Oak chest<br/>painted"),
        ]
    )
    html = dict(zip(out["record_id"], out["pass_html"], strict=True))
    assert [html[r] for r in ("r1", "r2", "r3")] == [True, True, True]
    assert [html[r] for r in ("r4", "r5")] == [False, False]


def test_model_typed_slots():
    out = compute(
        [
            ("r1", "M", "spectrum/title", "Iron key"),
            ("r1", "M", "spectrum/number_of_objects", "1 bag"),  # not an integer
            ("r1", "M", "spectrum/dimension_value", "68 x 51"),  # not a number
            ("r2", "M", "spectrum/title", "Bronze bowl"),
            ("r2", "M", "spectrum/number_of_objects", "2"),
            ("r2", "M", "spectrum/dimension_value", "8.5"),
            ("r3", "M", "spectrum/number_of_objects", "0"),  # Gt(0) violated
            ("r4", "M", "spectrum/title", "Untyped record"),
        ]
    )
    typed = dict(zip(out["record_id"], out["pass_model_typed"], strict=True))
    assert typed["r1"] == 0.0
    assert typed["r2"] == 1.0
    assert typed["r3"] == 0.0
    assert typed["r4"] is None  # no typed slots: layer contributes nothing


def test_conformance_folds_typed_layer():
    out = compute([("r1", "M", "spectrum/title", "Iron key"), ("r1", "M", "spectrum/number_of_objects", "2")])
    assert out.row(0, named=True)["conformance"] == pytest.approx(1.0)


def test_uri_only_free_text():
    # any free-text field holding only a URL fails
    out = compute(
        [
            ("r1", "M", "spectrum/text", "https://example.org/object/1"),
            ("r1", "M", "spectrum/title", "Roman coin"),
            ("r2", "M", "spectrum/brief_description", "https://example.org/2"),
            ("r3", "M", "spectrum/title", "A described coin"),
            # a URL outside free text is not this defect
            ("r4", "M", "spectrum/text_reference_number", "https://example.org/3"),
        ]
    )
    uri = dict(zip(out["record_id"], out["pass_uri_only_text"], strict=True))
    assert uri["r1"] is False
    assert uri["r2"] is False
    assert uri["r3"] is True
    assert uri["r4"] is None
