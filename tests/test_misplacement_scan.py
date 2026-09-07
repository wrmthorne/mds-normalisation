import polars as pl
import pytest

from mds_norm.pipeline.misplacement_scan import MIN_HITS, consensus_home, score_pairs

PRODUCTION_DATE = "spectrum/object_production_date"
ASSOCIATED_DATE = "spectrum/associated_date"


def usage_frame(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame([(*row, 1) for row in rows], schema=["data_source", "field_type", "norm", "n"], orient="row")


def spread(value: str, field: str, institutions: list[str]) -> list[tuple[str, str, str]]:
    return [(inst, field, value) for inst in institutions]


HOME_SIX = ["inst1", "inst2", "inst3", "inst4", "inst5", "inst6"]


def test_the_odd_institution_out_is_the_only_one_flagged():
    usage = usage_frame(spread("1920s", PRODUCTION_DATE, HOME_SIX) + spread("1920s", ASSOCIATED_DATE, ["odd"]))
    hits = consensus_home(usage)
    assert hits["data_source"].to_list() == ["odd"]
    assert hits["home_field"].to_list() == [PRODUCTION_DATE]


def test_an_institution_never_judges_itself_against_its_own_usage():
    # an institution's own usage never supports its own home
    usage = usage_frame(spread("1920s", PRODUCTION_DATE, HOME_SIX) + spread("1920s", ASSOCIATED_DATE, ["inst1"]))
    hits = consensus_home(usage)
    assert hits.filter(pl.col("data_source") == "inst1")["used_field"].to_list() == [ASSOCIATED_DATE]


def test_too_few_institutions_agree_on_a_home():
    usage = usage_frame(spread("1920s", PRODUCTION_DATE, HOME_SIX[:4]) + spread("1920s", ASSOCIATED_DATE, ["odd"]))
    assert consensus_home(usage).is_empty()


def test_a_value_both_fields_hold_has_no_home():
    usage = usage_frame(
        spread("1920s", PRODUCTION_DATE, HOME_SIX[:5]) + spread("1920s", ASSOCIATED_DATE, ["odd", "inst7", "inst8"])
    )
    assert consensus_home(usage).is_empty()


def hits_frame(institution: str, n: int, home_present: bool) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "data_source": [institution] * n,
            "field_type": ["spectrum/material"] * n,
            "home_field": ["spectrum/dimension"] * n,
            "record_id": [f"{institution}-{i}" for i in range(n)],
            "value": [f"{i} cm" for i in range(n)],
            "detector": ["typed"] * n,
            "home_present": [home_present] * n,
        }
    )


def denominator_frame(rows: list[tuple[str, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["data_source", "field_type", "occ"], orient="row").with_columns(
        pl.col("occ").cast(pl.UInt32)
    )


def test_an_institution_alone_in_its_habit_is_flagged():
    denominators = denominator_frame([("odd", "spectrum/material", 1000), ("peer", "spectrum/material", 100_000)])
    pairs = score_pairs(hits_frame("odd", 200, home_present=False), denominators)
    row = pairs.row(0, named=True)
    assert row["share"] == pytest.approx(0.2)
    assert row["lift"] > 1000
    assert row["peers"] == 0
    assert row["home_occ"] == 0  # the institution has no dimension field at all
    assert row["mapping_issue"]


def test_a_populated_home_field_means_the_institution_is_drawing_a_distinction():
    denominators = denominator_frame(
        [
            ("odd", "spectrum/material", 1000),
            ("odd", "spectrum/dimension", 5000),
            ("peer", "spectrum/material", 100_000),
        ]
    )
    pairs = score_pairs(hits_frame("odd", 200, home_present=True), denominators)
    row = pairs.row(0, named=True)
    assert row["home_absent"] == 0
    assert not row["mapping_issue"]


def test_a_habit_the_whole_corpus_shares_is_not_a_mapping_error():
    hits = pl.concat(
        [hits_frame("odd", 200, home_present=False)] + [hits_frame(f"peer{i}", 200, False) for i in range(5)]
    )
    denominators = denominator_frame(
        [("odd", "spectrum/material", 1000)] + [(f"peer{i}", "spectrum/material", 1000) for i in range(5)]
    )
    pairs = score_pairs(hits, denominators)
    assert pairs["peers"].to_list() == [5] * 6
    assert not pairs["mapping_issue"].any()


def test_pairs_below_the_mass_floor_are_dropped():
    denominators = denominator_frame([("odd", "spectrum/material", 100)])
    assert score_pairs(hits_frame("odd", MIN_HITS - 1, home_present=False), denominators).is_empty()
