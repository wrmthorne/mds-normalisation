import re

import pytest
from case_loader import load_cases

from mds_norm.parsers.parse_certainty import MAX_MARKED_LEN, PREFILTER, CertaintyMatch, parse_certainty

PRE = re.compile(PREFILTER)

OVER_MAX_LEN = [
    "what is this object?" + " padding" * 12,
    "Is this a teapot? Unclear. And why?" * 3,  # long prose
]


@pytest.mark.parametrize(("raw", "expected"), load_cases("parse_certainty"))
def test_parse_certainty(raw, expected):
    got = parse_certainty(raw)
    if expected is None:
        assert got is None
        return
    assert got == CertaintyMatch(clean=expected[0], notation=expected[1])
    assert PRE.search(raw), "prefilter must admit every parseable value"


@pytest.mark.parametrize("value", OVER_MAX_LEN)
def test_over_max_marked_len(value):
    assert len(value) > MAX_MARKED_LEN, "case no longer crosses the boundary"
    assert parse_certainty(value) is None


def test_interior_question_mark_untouched():
    # '?' replacing a lost character is not stripped
    assert parse_certainty("Caf? Royal") is None


def test_trailing_q_beats_mid_value_brackets():
    got = parse_certainty("silver [hallmarked] spoon?")
    assert got == CertaintyMatch(clean="silver [hallmarked] spoon", notation="?")
