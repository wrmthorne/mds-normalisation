import numpy as np
import polars as pl

from mds_norm.pipeline.practice_boundaries import (
    BOUNDARY_SIDE_YEARS,
    composition_test,
    field_families,
    field_tests,
    shared_boundaries,
)


def test_a_clean_step_passes_as_abrupt():
    rates = np.array([0.1] * 15 + [0.8] * 15)
    tests = [t for t in field_tests(rates) if t["passed"]]
    assert [t["split"] for t in tests] == [15]
    assert tests[0]["abrupt"]
    assert tests[0]["change"] > 0


def test_a_ramp_passes_but_is_gradual():
    rates = np.linspace(0.05, 0.95, 30)
    tests = [t for t in field_tests(rates) if t["passed"]]
    assert tests
    assert not any(t["abrupt"] for t in tests)


def test_a_flat_series_passes_nothing():
    rates = 0.5 + 0.02 * np.sin(np.arange(30))
    assert not any(t["passed"] for t in field_tests(rates))


def test_a_pass_needs_readable_years_either_side():
    rates = np.array([0.1] * 3 + [0.9] * 27)
    assert all(t["split"] >= BOUNDARY_SIDE_YEARS for t in field_tests(rates))


def test_fields_filled_on_the_same_records_form_one_family():
    records = [f"r{i}" for i in range(40)]
    present = pl.DataFrame(
        {
            "record_id": records[:20] + records[:20] + records[20:],
            "field_type": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
        }
    )
    years = pl.DataFrame({"record_id": records, "data_source": "u", "year": 2000})
    family = dict(field_families(present, years).select("field_type", "family").rows())
    assert family["a"] == family["b"] != family["c"]


def test_shared_boundaries_need_more_groups_than_chance():
    readable = np.arange(1950, 2020)
    # eight groups fall together in 2000; two others step at unrelated years
    steps = pl.DataFrame(
        {
            "year": [2000] * 8 + [1971, 1985],
            "family": list(range(8)) + [8, 9],
            "direction": ["down"] * 8 + ["up", "up"],
        }
    )
    shared, min_groups, chance = shared_boundaries(steps, readable)
    assert shared == [2000]
    assert min_groups <= 8
    assert chance < 0.05


def test_a_change_in_what_was_collected_does_not_survive_its_object_names():
    # before 2000 the unit took coins, which never carry the field; from 2000 it took prints, which always do
    years = [1995] * 20 + [2005] * 20
    records = pl.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(40)],
            "data_source": "u",
            "year": years,
            "name": ["coin"] * 20 + ["print"] * 20,
        }
    )
    present = pl.DataFrame({"record_id": [f"r{i}" for i in range(20, 40)], "field_type": "f"})
    mark = pl.DataFrame({"data_source": "u", "field_type": "f", "year": 2000, "from_year": 1995, "to_year": 2005})
    assert not composition_test(mark, records, present)["survives"][0]
    # the same rise within one object name survives
    within = records.with_columns(name=pl.lit("print"))
    assert composition_test(mark, within, present)["survives"][0]
