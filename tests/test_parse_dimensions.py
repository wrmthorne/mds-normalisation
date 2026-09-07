import pytest
from case_loader import load_cases

from mds_norm.parsers.parse_dimensions import parse_dimensions


def rows(raw):
    r = parse_dimensions(raw)
    if r is None:
        return None
    return [
        (
            m["dimension_type"],
            m["dimension_value"],
            m["dimension_measurement_unit"],
            m["dimension_value_qualifier"],
            m["dimension_measured_part"],
        )
        for m in r["measurements"]
    ]


@pytest.mark.parametrize(("raw", "expected"), load_cases("parse_dimensions"))
def test_parse_dimensions(raw, expected):
    assert rows(raw) == expected


def test_unparsed_tail_is_residue():
    r = parse_dimensions("overall: 35 mm x 35 mm; 1 slide")
    assert r["status"] == "partial", r
    assert r["residue"] == ["1 slide"], r
    assert all(m["dimension_measured_part"] == "overall" for m in r["measurements"]), r


def test_parenthetical_part_and_ambiguous_bare_cross():
    r = parse_dimensions("5 x 5cm (mount)")
    assert all(m["dimension_measured_part"] == "mount" for m in r["measurements"]), r
    assert all(m["type_ambiguous"] for m in r["measurements"]), "bare crosses must flag the h x w convention"


def test_explicit_keywords_are_not_ambiguous():
    r = parse_dimensions("140mm height x 90mm width")
    assert not any(m["type_ambiguous"] for m in r["measurements"]), "explicit keywords are not ambiguous"
    assert [m["axis"] for m in r["measurements"]] == [0, 1], r


def test_prime_unit_override():
    # corpus convention: a prime is feet; Royal Armouries writes inches
    default = parse_dimensions("2.98' (75.7mm)")
    assert (
        default["measurements"][0]["dimension_value"],
        default["measurements"][0]["dimension_measurement_unit"],
    ) == (2.98, "ft")
    ra = parse_dimensions("2.98' (75.7mm)", prime_unit="in")
    assert (ra["measurements"][0]["dimension_value"], ra["measurements"][0]["dimension_measurement_unit"]) == (
        2.98,
        "in",
    )
    # double prime is inches under either reading
    assert parse_dimensions('6"', prime_unit="in")["measurements"][0]["dimension_measurement_unit"] == "in"
