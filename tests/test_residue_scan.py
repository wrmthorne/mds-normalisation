import polars as pl
import pytest

from mds_norm.evaluation import residue_scan as rs


@pytest.mark.parametrize(
    ("value", "is_edtf"),
    [
        # the shapes the date stage actually publishes
        ("1885", True),
        ("-0118", True),
        ("1885-03", True),
        ("1885-03-14", True),
        ("1885~", True),
        ("1885?", True),
        ("1601/1850", True),
        ("1601~/1850~", True),
        ("19XX", True),
        # an unspecified digit, not only a whole decade
        ("198X", True),
        ("193X", True),
        ("187X~", True),
        # intervals of year-months, qualified or not
        ("1994-04/1994-10", True),
        ("1934-04?/1935-03?", True),
        ("[1885, 1886]", True),
        ("../1850", True),
        ("1850/..", True),
        # residue: what the review counts
        ("April 1993", False),
        ("Roman", False),
        ("16/07/2008", False),
        ("1/1983", False),
        ("AD 43 - 418", False),
        ("n/k", False),
        ("1997/--/--", False),
        ("31/12/79", False),
        # an unpadded month/day is *not* EDTF
        ("1938-12-3", False),
        ("885", False),
    ],
)
def test_edtf_shape(value, is_edtf):
    assert pl.select(pl.lit(value).str.contains(rs.EDTF_RE)).item() is is_edtf


def test_summarise_shape():
    rows = pl.LazyFrame(
        {
            "field_type": ["spectrum/material"] * 3 + ["spectrum/condition"],
            "value": ["wood;", "wood;", "oak;", "good:"],
            "data_source": ["M", "M", "N", "M"],
        }
    )
    out = rs._summarise(rows, "dangling_delimiter", "note text")
    assert out.columns == [
        "family",
        "field_type",
        "occurrences",
        "distinct_values",
        "institutions",
        "routed",
        "examples",
        "note",
    ]
    top = out.row(0, named=True)
    assert (top["field_type"], top["occurrences"], top["distinct_values"], top["institutions"]) == (
        "spectrum/material",
        3,
        2,
        2,
    )
    # examples are ordered by occurrence, most frequent first
    assert top["examples"][0] == "wood;"
    # a skipped family's empty frame matches the schema
    assert rs._EMPTY.columns == out.columns


def test_summarise_splits_routed_from_untouched():
    # a queued value differs from an unexamined one
    rows = pl.LazyFrame(
        {
            "field_type": ["spectrum/object_production_date"] * 3,
            "value": ["Roman", "Mesolithic", "16/07/2008"],
            "data_source": ["M", "M", "M"],
            "disposition": ["deferred", "qualified", "untouched"],
        }
    )
    out = rs._summarise(rows, "dates_non_edtf", "note")
    assert out.row(0, named=True)["occurrences"] == 3
    assert out.row(0, named=True)["routed"] == 2
