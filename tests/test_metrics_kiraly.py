import math

import polars as pl
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mds_norm.metrics import kiraly


def frame(rows):
    return pl.LazyFrame(
        rows,
        schema={"record_id": pl.String, "data_source": pl.String, "field_type": pl.String, "value": pl.String},
        orient="row",
    )


@pytest.fixture
def base():
    return frame(
        [
            ("r1", "A", "spectrum/object_number", "1"),
            ("r1", "A", "spectrum/object_name", "chair"),
            ("r1", "A", "spectrum/material", "oak"),
            ("r2", "A", "spectrum/object_number", "2"),
            ("r2", "A", "spectrum/object_name", "chair"),
            ("r3", "B", "spectrum/object_number", "3"),
            ("r3", "B", "spectrum/object_name", "stool"),
        ]
    )


def test_completeness_denominator_is_the_whole_schema(base):
    out = kiraly.completeness(base).sort("record_id")
    n_fields = len(kiraly.schema_fields())
    assert out["k_completeness_ratio"].to_list() == pytest.approx([3 / n_fields, 2 / n_fields, 2 / n_fields])


def test_importance_weight_is_one_coefficient_per_field(base):
    weights = kiraly.importance_weights(base)
    assert weights.height == weights["field_type"].n_unique()
    # object_number is universal; material fills half of A only
    by_field = dict(zip(weights["field_type"], weights["w"], strict=True))
    assert by_field["spectrum/object_number"] == pytest.approx(1.0)
    assert by_field["spectrum/material"] == pytest.approx(0.25)


def test_information_content_is_zero_for_a_universal_value(base):
    freq = kiraly.value_frequencies(base)
    ic = dict(zip(kiraly.information_content(freq)["value"], kiraly.information_content(freq)["ic"], strict=True))
    # 'chair' occurs twice in three values
    assert ic["chair"] == pytest.approx(1.0 - math.log(2) / math.log(3))
    assert ic["stool"] == pytest.approx(1.0)


def test_conformance_averages_over_categorical_values_only():
    base = frame(
        [
            ("r1", "A", "spectrum/object_name", "chair"),
            ("r1", "A", "spectrum/brief_description", "a chair"),
            ("r2", "A", "spectrum/object_name", "chair"),
        ]
    )
    out = kiraly.conformance_to_expectations(base).sort("record_id")
    # free text is excluded; only object_name scores
    assert out["n_categorical_values"].to_list() == [1, 1]
    assert out["k_information_content"].to_list() == pytest.approx([0.0, 0.0])


def test_coherence_undefined_below_two_textual_fields():
    base = frame(
        [
            ("r1", "A", "spectrum/brief_description", "carved oak dining chair"),
            ("r1", "A", "spectrum/comments", "carved oak dining chair"),
            ("r2", "A", "spectrum/brief_description", "only one textual field"),
        ]
    )
    out = kiraly.coherence(base, min_df=1)
    assert out["record_id"].to_list() == ["r1"]
    assert out["k_coherence"].item() == pytest.approx(0.0, abs=1e-9)


@pytest.fixture
def textual():
    return frame(
        [
            ("r1", "A", "spectrum/brief_description", "Carved oak dining chair with turned legs"),
            ("r1", "A", "spectrum/comments", "oak chair, one of a set of six"),
            ("r1", "A", "spectrum/description", "The seat is rush and the frame oak"),
            ("r2", "A", "spectrum/brief_description", "Silver teapot"),
            ("r2", "A", "spectrum/comments", "Hallmarked London 1790"),
            ("r3", "B", "spectrum/brief_description", "Pair of brass candlesticks"),
            ("r3", "B", "spectrum/comments", "brass candlesticks pair"),
            ("r3", "B", "spectrum/description", "One candlestick is dented"),
            ("r4", "B", "spectrum/brief_description", "single field only"),
        ]
    )


def test_coherence_matches_sklearn_tfidf(textual):
    docs = kiraly.documents(textual).collect().sort("record_id", "field_type")
    matrix = TfidfVectorizer(min_df=1, sublinear_tf=True).fit_transform(docs["text"].to_list())
    expected = {}
    for rid in docs["record_id"].unique():
        idx = (docs["record_id"] == rid).to_numpy().nonzero()[0]
        if len(idx) < 2:
            continue
        sim = cosine_similarity(matrix[idx])
        pairs = [sim[i, j] for i in range(len(idx)) for j in range(i + 1, len(idx))]
        expected[rid] = 1 - sum(pairs) / len(pairs)

    out = kiraly.coherence(textual, min_df=1).sort("record_id")
    assert out["record_id"].to_list() == sorted(expected)
    assert out["k_coherence"].to_list() == pytest.approx([expected[r] for r in sorted(expected)])
    assert out["data_source"].to_list() == ["A", "A", "B"]


def test_coherence_chunks_and_external_idf_agree(textual):
    idf = kiraly.idf_weights(textual, min_df=1)
    chunked_idf = kiraly.idf_weights(textual, min_df=1, n_chunks=3)
    assert chunked_idf.drop("token_id").sort("token").equals(idf.drop("token_id").sort("token"))
    whole = kiraly.coherence(textual, idf=idf).sort("record_id")
    chunked = kiraly.coherence(textual, idf=idf, n_chunks=3).sort("record_id")
    assert chunked["record_id"].to_list() == whole["record_id"].to_list()
    assert chunked["k_coherence"].to_list() == pytest.approx(whole["k_coherence"].to_list())


def test_idf_weights_drop_rare_tokens_and_cap_vocabulary(textual):
    idf = kiraly.idf_weights(textual, min_df=2)
    assert set(idf["token"]) == {"oak", "chair", "brass", "candlesticks", "pair", "of"}
    assert kiraly.idf_weights(textual, min_df=2, max_features=3).height == 3


def test_field_frequency_profile_ranks_and_accumulates(base):
    profile = kiraly.field_frequency_profile(base).filter(pl.col("data_source") == "A")
    assert profile["rank"].to_list() == [1, 2, 3]
    assert profile["cum_share"].max() == pytest.approx(1.0)
    summary = kiraly.pareto_summary(profile)
    assert summary["n_fields"].item() == 3


def test_field_vocabulary_reports_both_readings(base):
    vocab = kiraly.field_vocabulary(base).filter(pl.col("field_type") == "spectrum/object_name")
    assert vocab["n_distinct"].item() == 2
    assert vocab["singleton_rate"].item() == pytest.approx(0.5)
    assert vocab["k_80"].item() == 2
