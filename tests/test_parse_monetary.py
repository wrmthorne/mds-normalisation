import pytest
from case_loader import load_cases

from mds_norm.parsers.parse_monetary import parse_monetary


def row(raw):
    a = parse_monetary(raw)
    if a is None:
        return None
    return a.currency_system, a.normalised_pence, a.uncertain, a.context


@pytest.mark.parametrize(("raw", "expected"), load_cases("parse_monetary"))
def test_parse_monetary(raw, expected):
    assert row(raw) == expected
