import polars as pl
import pytest

from mds_norm.pipeline.divergence_tests import MIN_N, corpus_slots, correct, digit_test, letter_test


def test_a_cell_matching_the_corpus_has_a_small_effect():
    values = [str(v) for v in range(1, 32)] * 4
    effect, _ = digit_test(values, values)
    assert effect == pytest.approx(0.0)


def test_a_cell_concentrated_on_one_value_diverges():
    corpus = [str(v) for v in range(1, 32)] * 4
    effect, p = digit_test(["1"] * MIN_N * 2, corpus)
    assert effect > 0.9
    assert p < 0.01


def test_a_slot_below_the_evidence_floor_is_not_tested():
    corpus = [str(v) for v in range(1, 32)] * 4
    assert digit_test(["1"] * (MIN_N - 1), corpus) is None
    assert letter_test(["a"] * (MIN_N - 1), ["a", "b"] * MIN_N) is None


def test_a_letter_slot_using_its_own_vocabulary_diverges():
    effect, p = letter_test(["jan"] * MIN_N * 2, (["jan", "feb", "mar", "apr"] * MIN_N))
    assert 0 < effect <= 1
    assert p < 0.01


def test_the_corpus_distribution_pools_every_institution():
    cells = pl.DataFrame(
        {
            "data_source": ["A", "B"],
            "merged_pattern": ["d{4}", "d{4}"],
            "slot_idx": [0, 0],
            "slot_kind": ["d", "d"],
            "values": [["1900"], ["2000"]],
        }
    )
    pooled = corpus_slots(cells)
    assert sorted(pooled[("d{4}", 0)]) == ["1900", "2000"]


def test_correction_marks_only_the_cells_that_survive_it():
    tested = pl.DataFrame({"family": ["date"] * 3, "p": [1e-9, 0.04, 0.9], "effect": [0.9, 0.2, 0.01]})
    out = correct(tested)
    assert out["significant"].to_list() == [True, False, False]
    # the ranking is by effect size, not by p
    assert out["effect"].to_list() == [0.9, 0.2, 0.01]
