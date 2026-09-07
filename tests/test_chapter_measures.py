import polars as pl
import pytest

from mds_norm.metrics import conformance, thinness
from mds_norm.pipeline.consistency_induction import TAU_SWEEP, head_floor_sweep
from mds_norm.pipeline.uncertainty_census import STYLE_MIN_RATE, styles
from mds_norm.pipeline.vocabulary_fragmentation import fragmentation


def base_frame(rows: list[dict]) -> pl.LazyFrame:
    return pl.DataFrame(
        rows,
        schema={
            "record_id": pl.String,
            "data_source": pl.String,
            "field_type": pl.String,
            "value": pl.String,
            "node_id": pl.String,
            "parent_id": pl.String,
            "depth": pl.Int32,
        },
    ).lazy()


def record(rid: str, values: dict[str, str]) -> list[dict]:
    return [
        {
            "record_id": rid,
            "data_source": "A",
            "field_type": field,
            "value": value,
            "node_id": f"{rid}:{i}",
            "parent_id": None,
            "depth": 0,
        }
        for i, (field, value) in enumerate(values.items())
    ]


def test_thinness_falls_as_a_record_holds_more_in_its_structured_fields():
    rows = record("full", {"spectrum/title": "a" * 400, "spectrum/object_name": "b" * 400}) + record(
        "thin", {"spectrum/title": "c" * 10}
    )
    out = thinness.compute(base_frame(rows)).sort("record_id")
    scores = dict(zip(out["record_id"], out["thinness"], strict=True))
    assert scores["thin"] > scores["full"]


def test_a_record_whose_content_is_all_prose_scores_as_thin_as_an_empty_one():
    rows = record("prose", {"spectrum/brief_description": "d" * 400}) + record(
        "structured", {"spectrum/object_name": "e" * 400}
    )
    out = thinness.compute(base_frame(rows))
    scores = dict(zip(out["record_id"], out["thinness"], strict=True))
    assert scores["prose"] == pytest.approx(1.0)
    assert scores["structured"] < scores["prose"]


def masked_frame() -> pl.DataFrame:
    # patterns holding 7, 2 and 1 of 10 values
    patterns = ["d{4}"] * 7 + ["d{2}"] * 2 + ["s{3}"]
    return pl.DataFrame(
        {
            "record_id": [str(i) for i in range(10)],
            "data_source": ["A"] * 10,
            "field_type": ["spectrum/object_production_date"] * 10,
            "merged_pattern": patterns,
        }
    )


def test_a_higher_head_floor_calls_more_values_divergent():
    sweep = head_floor_sweep(masked_frame()).sort("tau")
    shares = dict(zip(sweep["tau"], sweep["divergent_share"], strict=True))
    assert sorted(shares) == sorted(TAU_SWEEP)
    assert shares[0.02] <= shares[0.05] <= shares[0.10]


def test_at_the_lowest_floor_every_pattern_is_a_head_pattern():
    sweep = head_floor_sweep(masked_frame())
    assert sweep.filter(pl.col("tau") == 0.02)["divergent_share"].item() == pytest.approx(0.0)


def rates(question: float, brackets: float, lexical: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "data_source": ["A"],
            "n_values": [1000],
            "question": [question],
            "brackets": [brackets],
            "lexical": [lexical],
        }
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((0.0, 0.0, 0.0), "no marker"),
        ((0.1, 0.0, 0.0), "question"),
        ((0.1, 0.0, 0.1), "question + lexical"),
        ((0.1, 0.1, 0.1), "question + brackets + lexical"),
        # below the floor a marker is not style
        ((STYLE_MIN_RATE / 2, 0.0, 0.1), "lexical"),
    ],
)
def test_a_style_is_the_set_of_markers_an_institution_uses(values, expected):
    assert styles(rates(*values))["style"].item() == expected


def test_one_concept_written_many_ways_fragments_more_than_one_written_once():
    tight = pl.DataFrame({"data_source": ["A"] * 2, "norm": ["oak", "pine"], "occ": [500, 500]})
    loose = pl.DataFrame(
        {"data_source": ["B"] * 6, "norm": [f"oak{i}" for i in range(6)], "occ": [200, 200, 200, 200, 100, 100]}
    )
    scores = pl.concat([fragmentation(tight, "material"), fragmentation(loose, "material")])
    by_source = dict(zip(scores["data_source"], scores["fragmentation"], strict=True))
    assert by_source["B"] > by_source["A"]


def test_the_working_vocabulary_is_the_terms_that_carry_most_of_the_field():
    atoms = pl.DataFrame({"data_source": ["A"] * 4, "norm": ["a", "b", "c", "d"], "occ": [800, 100, 50, 50]})
    assert fragmentation(atoms, "material")["k_80"].item() == 1


def test_the_breakdown_reports_only_the_checks_that_ran():
    rows = record("r1", {"spectrum/title": "a title", "spectrum/brief_description": "a description of the object"})
    base = base_frame(rows)
    scored = conformance.compute(base)
    per_check, _ = conformance.failure_breakdown(base, scored)
    assert per_check["applicable"].min() > 0
    assert "wellformed" not in per_check["check"].to_list()
