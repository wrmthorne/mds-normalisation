import polars as pl
import pytest

from mds_norm.pipeline.accession_schemes import (
    AGREE_TOLERANCE,
    MIN_DATED,
    _restart_rate,
    slot_is_year,
    structural,
    verdicts,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1949", "1950", "1951", "2001"], True),
        # four digits outside the calendar window: a sequence
        (["4901", "5012", "6511", "7003"], False),
        # mostly short runs: a sequence
        (["12", "7", "104", "1950"], False),
        ([], False),
    ],
)
def test_slot_is_year(values, expected):
    assert slot_is_year(values) is expected


def test_a_sequence_that_restarts_each_year_scores_high():
    years = [1990, 1990, 1991, 1991, 1992]
    tails = [1, 40, 2, 12, 3]
    assert _restart_rate({"years": years, "tails": tails}) == 1.0


def test_a_sequence_that_runs_on_scores_zero():
    assert _restart_rate({"years": [1990, 1991, 1992], "tails": [10, 20, 30]}) == 0.0


def test_one_year_gives_no_restart_evidence():
    assert _restart_rate({"years": [1990], "tails": [10]}) == 0.0


def frame(rows: list[dict]) -> pl.DataFrame:
    # typed integer columns even where no year is recorded
    return pl.DataFrame(
        rows, schema={"data_source": pl.String, "recorded_diff": pl.Int32, "production_diff": pl.Int32}
    )


def test_a_scheme_agreeing_with_recorded_years_is_confirmed():
    diffs = frame([{"data_source": "A", "recorded_diff": 0, "production_diff": 50} for _ in range(MIN_DATED + 5)])
    out = verdicts(diffs, structural_frame(), coverage_frame())
    assert out.row(by_predicate=pl.col("data_source") == "A", named=True)["verdict"] == "confirmed"


def test_a_scheme_disagreeing_is_rejected_on_that_evidence():
    diffs = frame([{"data_source": "A", "recorded_diff": 30 + i, "production_diff": 50} for i in range(MIN_DATED + 5)])
    out = verdicts(diffs, structural_frame(), coverage_frame())
    row = out.row(by_predicate=pl.col("data_source") == "A", named=True)
    assert row["verdict"] == "rejected"
    assert row["evidence"] == "dated records"


def test_a_year_either_side_still_agrees():
    diffs = frame(
        [{"data_source": "A", "recorded_diff": AGREE_TOLERANCE, "production_diff": 10} for _ in range(MIN_DATED + 5)]
    )
    out = verdicts(diffs, structural_frame(), coverage_frame())
    assert out.row(by_predicate=pl.col("data_source") == "A", named=True)["verdict"] == "confirmed"


def test_with_no_dated_records_the_structure_decides():
    diffs = frame([{"data_source": "A", "recorded_diff": None, "production_diff": None}])
    out = verdicts(diffs, structural_frame(), coverage_frame())
    row = out.row(by_predicate=pl.col("data_source") == "A", named=True)
    assert row["verdict"] == "plausible"
    assert row["evidence"] == "structure"


def test_a_year_that_never_varies_is_no_year():
    diffs = frame([{"data_source": "A", "recorded_diff": None, "production_diff": None}])
    out = verdicts(diffs, structural_frame(n_years=1), coverage_frame())
    assert out.row(by_predicate=pl.col("data_source") == "A", named=True)["verdict"] == "rejected"


def test_numbers_that_mostly_carry_no_year_slot_are_rejected():
    diffs = frame([{"data_source": "A", "recorded_diff": None, "production_diff": None}])
    out = verdicts(diffs, structural_frame(), coverage_frame(year_coverage=0.01))
    assert out.row(by_predicate=pl.col("data_source") == "A", named=True)["verdict"] == "rejected"


def structural_frame(n_years: int = 30) -> pl.DataFrame:
    return pl.DataFrame(
        {"data_source": ["A"], "n_records": [500], "n_years": [n_years], "year_span": [n_years], "restart_rate": [0.5]}
    )


def coverage_frame(year_coverage: float = 0.9) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "data_source": ["A"],
            "n_records": [500],
            "year_coverage": [year_coverage],
            "inspected_share": [1.0],
            "n_year_patterns": [1],
        }
    )


def test_structural_reads_the_sequence_beside_the_year():
    objnum = pl.DataFrame(
        {
            "record_id": ["1", "2", "3"],
            "data_source": ["A", "A", "A"],
            "year": [1990, 1991, 1992],
            "value": ["1990.40", "1991.2", "1992.7"],
        }
    )
    out = structural(objnum)
    assert out["n_years"].item() == 3
    assert out["restart_rate"].item() > 0
