from pathlib import Path

import pytest
from case_loader import load_cases

from mds_norm.parsers.parse_dates import Conventions, build_period_index, complete_year, load_periods, parse_date

PERIODS_CSV = Path(__file__).parent.parent / "data" / "vocabularies" / "periods.csv"


def row(raw):
    r = parse_date(raw)
    if r is None:
        return None
    return (r["value_edtf"], r["date_earliest_single"], r["date_latest"], r["date_earliest_single_certainty"])


@pytest.mark.parametrize(("raw", "expected"), load_cases("parse_dates"))
def test_parse_dates(raw, expected):
    assert row(raw) == expected


def test_bare_month_day_without_year_refuses():
    # a lost year must not surface as an XXXX/-00 phantom
    assert parse_date("3 October") is None


@pytest.mark.parametrize(
    ("raw", "edtf"),
    [
        # a zero day degrades to month precision
        ("1900.07.00", "1900-07"),
        ("00/11/1934", "1934-11"),
        ("1863.09.00", "1863-09"),
        # zeros in every slot still leave only the year
        ("1900.00.00", "1900"),
        ("00/00/1934", "1934"),
        # a real day is untouched
        ("1983.12.24", "1983-12-24"),
    ],
)
def test_zero_null_day(raw, edtf):
    assert parse_date(raw, Conventions(dm_order="DM", zero_null=True))["value_edtf"] == edtf


@pytest.fixture
def periods():
    load_periods(PERIODS_CSV)
    yield
    build_period_index({})  # leave the module period-free for the other tests


@pytest.mark.parametrize(
    ("raw", "period", "edtf"),
    [
        # a pure period label yields no dates
        ("Victorian", "Victorian", None),
        ("post Medieval", "Post-Medieval", None),
        ("IRON AGE", "Iron Age", None),
        # a period beside a parenthesised span keeps the span reading
        ("Victorian period (1837 - 1901)", "Victorian", "1837/1901"),
        ("Medieval (c.1070-c.1500)", "Medieval", "1070~/1500~"),
        # the lexicon's years are never written for a bare label
        ("Georgian (1714-1830)", "Georgian", "1714/1830"),
        # a compound period never matches its substring
        ("Romano-British", "Romano-British", None),
        # plain dates are unaffected
        ("c. 1850", None, "1850~"),
    ],
)
def test_periods(periods, raw, period, edtf):
    r = parse_date(raw)
    assert r is not None
    assert r["date_period"] == period
    assert r["value_edtf"] == edtf


def test_no_period_index_no_period():
    assert parse_date("Victorian") is None


@pytest.mark.parametrize(
    ("raw", "ceiling", "edtf"),
    [
        # the accession year bounds which century a two-digit year means
        ("31/12/79", 1995, "1979-12-31"),
        ("28.9.83", 1990, "1983-09-28"),
        ("1/1/98", 2005, "1998-01-01"),
        # the completion never postdates the accession: 1998 is impossible here
        ("1/1/98", 1995, "1898-01-01"),
        # no accession year, no century to choose
        ("31/12/79", None, None),
        # a four-digit year is never touched
        ("31/12/1979", 1995, "1979-12-31"),
        ("31/12/1979", None, "1979-12-31"),
    ],
)
def test_two_digit_year_completion(raw, ceiling, edtf):
    r = parse_date(raw, Conventions(dm_order="DM", century_ceiling=ceiling))
    assert (r["value_edtf"] if r else None) == edtf


@pytest.mark.parametrize(
    ("tok", "ceiling", "expected"),
    [
        ("79", 1995, "1979"),
        ("98", 1995, "1898"),
        ("05", 2005, "2005"),
        ("06", 2005, "1906"),
        ("1979", None, "1979"),  # already complete
        ("79", None, None),  # no ceiling to choose by
    ],
)
def test_complete_year(tok, ceiling, expected):
    assert complete_year(tok, Conventions(century_ceiling=ceiling)) == expected
