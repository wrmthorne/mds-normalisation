import polars as pl
import pytest

from mds_norm.metrics import completeness

CORE = [
    ("spectrum/object_number", "1"),
    ("spectrum/object_name", "chest"),
    ("spectrum/brief_description", "an oak chest"),
]


def base_from(records, data_source="M"):
    rows = [(rid, data_source, ft, v) for rid, fields in records.items() for ft, v in CORE + fields]
    return pl.LazyFrame(
        rows,
        schema={"record_id": pl.String, "data_source": pl.String, "field_type": pl.String, "value": pl.String},
        orient="row",
    )


def multi_base(**per_institution):
    return pl.concat([base_from(records, data_source=ds) for ds, records in per_institution.items()])


WEIGHTS = pl.DataFrame(
    {"field_type": ["spectrum/associated_date", "spectrum/material"], "data_source": ["M", "M"], "w": [0.7, 0.5]}
)


def scores(records):
    out = completeness.compute(base_from(records), weights=WEIGHTS)
    return dict(zip(out["record_id"], out["completeness"], strict=True))


def test_retyped_field_takes_source_weight():
    # a retyped field falls back to the source's weight
    c = scores(
        {
            "r1": [("spectrum/object_production_date", "1900"), ("spectrum/material", "oak")],
            "r2": [("spectrum/associated_date", "1900"), ("spectrum/material", "oak")],
        }
    )
    assert c["r1"] == pytest.approx(1.0)
    assert c["r2"] == pytest.approx(1.0)


def test_source_and_destination_count_once():
    c = scores({"r1": [("spectrum/associated_date", "1900"), ("spectrum/object_production_date", "1900")]})
    assert c["r1"] == pytest.approx(0.7 / 1.2)


def test_unweighted_field_without_retype_source_earns_nothing():
    c = scores({"r1": [("spectrum/condition", "good")]})
    assert c["r1"] == pytest.approx(0.0)


def test_retyped_field_inherits_larger_source_weight_over_own_row():
    # a fill-neutral retype must not read as a loss
    weights = pl.DataFrame(
        {
            "field_type": ["spectrum/associated_date", "spectrum/object_production_date", "spectrum/material"],
            "data_source": ["M", "M", "M"],
            "w": [0.7, 0.2, 0.5],
        }
    )
    c = completeness.compute(
        base_from(
            {
                "r1": [("spectrum/object_production_date", "1900")],
                "r2": [("spectrum/associated_date", "1900"), ("spectrum/object_production_date", "1900")],
            }
        ),
        weights=weights,
    )
    scored = dict(zip(c["record_id"], c["completeness"], strict=True))
    assert scored["r1"] == pytest.approx(0.7 / 1.4)
    assert scored["r2"] == pytest.approx(0.9 / 1.4)


def w_of(weights, field, inst):
    row = weights.filter((pl.col("field_type") == field) & (pl.col("data_source") == inst))
    return row["w"][0] if row.height else None


def test_zero_fill_institution_gets_global_field_row():
    # ignoring a field adopted elsewhere is penalised
    base = multi_base(
        A={"a1": [("spectrum/material", "oak")], "a2": [("spectrum/material", "ash")]},
        B={"b1": [("spectrum/condition", "good")], "b2": [("spectrum/condition", "poor")]},
    )
    w = completeness.field_weights(base)
    assert w_of(w, "spectrum/material", "B") == pytest.approx(0.5)
    assert w_of(w, "spectrum/material", "A") == pytest.approx(1.0)
    # the penalty reaches the scores too
    c = completeness.compute(base, weights=w)
    scored = dict(
        zip(zip(c["record_id"], c["data_source"].cast(pl.String), strict=True), c["completeness"], strict=True)
    )
    assert scored[("b1", "B")] == pytest.approx(1.0 / 1.5)


def test_global_fill_is_mean_over_all_institutions():
    # G averages over all institutions, not only adopters
    base = multi_base(
        A={"a1": [("spectrum/material", "oak")]},
        B={"b1": [("spectrum/condition", "good")]},
        C={"c1": [("spectrum/condition", "fair")]},
    )
    w = completeness.field_weights(base)
    assert w_of(w, "spectrum/material", "B") == pytest.approx(1 / 3)


def test_niche_field_stays_local():
    # a field below the threshold stays applicable at its adopter
    a = {f"a{i}": [("spectrum/material", "oak")] if i < 3 else [("spectrum/condition", "good")] for i in range(20)}
    b = {f"b{i}": [("spectrum/condition", "good")] for i in range(20)}
    w = completeness.field_weights(multi_base(A=a, B=b))
    assert w_of(w, "spectrum/material", "A") == pytest.approx(0.15)
    assert w_of(w, "spectrum/material", "B") is None


def test_release_channel_carries_no_weight_and_earns_nothing():
    base = multi_base(
        A={"a1": [("spectrum/material", "oak"), ("wrmthorne/kind", "exact")], "a2": [("wrmthorne/kind", "exact")]}
    )
    w = completeness.field_weights(base)
    assert w.filter(pl.col("field_type").str.starts_with("wrmthorne/")).height == 0
    c = completeness.compute(base, weights=w)
    scored = dict(zip(c["record_id"], c["completeness"], strict=True))
    assert scored["a2"] == pytest.approx(0.0)
