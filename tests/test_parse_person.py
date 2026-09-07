import pytest
from case_loader import load_cases

from mds_norm.parsers.parse_person import parse_person, read_tail


def row(raw, *, org_field=False):
    name, entity, sub, reason = parse_person(raw, org_field=org_field)
    name = name or {}
    return (entity, sub, reason, *(name.get(f) for f in ("prefix", "given", "middle", "surname", "suffix")))


@pytest.mark.parametrize(("raw", "expected"), load_cases("parse_person"))
def test_parse_person(raw, expected):
    assert row(raw) == expected


def test_mononym_needs_an_organisation_field():
    # the Person model has no whole-name slot
    assert row("Wolseley", org_field=True)[:2] == ("organisation", "mononym_organisation")
    assert row("Wolseley")[2] == "no_surname"


def test_mononym_shape_keeps_the_initialism_queue_out():
    # 'DCM'/'RER' resolve to nobody without knowing whose initials they are
    for initialism in ("DCM", "RER", "S", "MJS"):
        assert row(initialism, org_field=True)[2] is not None


def test_mononym_refuses_punctuated_and_short_forms():
    for value in ("Mint:", "A", "1873"):
        assert row(value, org_field=True)[2] is not None


@pytest.mark.parametrize(
    ("segment", "expected"),
    [
        ("Dr", ("prefix", "Dr")),
        ("Sgt", ("prefix", "Sgt")),
        ("Lord Provost", ("prefix", "Lord Provost")),
        ("OBE", ("suffix", "OBE")),
        ("F.S.I.A.", ("suffix", "F.S.I.A.")),
        ("the elder", ("suffix", "the elder")),
        ("Esq", ("suffix", "Esq")),
        # not name parts: address, role, forename, collection
        ("Suffolk", None),
        ("London", None),
        ("Former owner", None),
        ("P. Maurice", None),
        ("", None),
    ],
)
def test_read_tail_is_closed_sets_only(segment, expected):
    assert read_tail(segment) == expected


def test_display_is_the_reordered_parts():
    name, *_ = parse_person("White, Francis Buchanan, Dr")
    assert name["display"] == "Dr Francis Buchanan White"
