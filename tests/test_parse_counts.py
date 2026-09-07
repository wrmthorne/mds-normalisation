import pytest
from case_loader import load_cases

from mds_norm.parsers.parse_counts import parse_count


def rows(raw):
    parts = parse_count(raw)
    if parts is None:
        return None
    return [(p.count, p.qualifier, p.noun, p.range_lo, p.range_hi) for p in parts]


@pytest.mark.parametrize(("raw", "expected"), load_cases("parse_counts"))
def test_parse_count(raw, expected):
    assert rows(raw) == expected
