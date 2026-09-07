import re

import polars as pl
import pytest

from mds_norm.pipeline.accession_schemes import year_at_slot
from mds_norm.pipeline.consistency_induction import induce_lookup
from mds_norm.pipeline.pattern_exports import MIN_SLOT_SUPPORT, SLOT_RUNS, pattern_dist, patterns_merged, slot_values
from mds_norm.utils.masking import MASK, signature_of

OBJECT_NUMBERS = ["NWHCM : 1949.123.4", "NWHCM : 1950.7.11", "TWCMS : G1234", "ARM.1987.2", "1974-56/a"]


def masked(value: str) -> str:
    return pl.DataFrame({"value": [value]}).select(MASK).item()


def signature(value: str) -> str:
    return signature_of(masked(value))


@pytest.mark.parametrize("value", OBJECT_NUMBERS)
def test_source_pattern_matches_the_signature_fingerprints_computes(value):
    """_slot_lookup joins on (field_type, source_pattern), so the two must agree character for character"""
    induced = induce_lookup(pl.LazyFrame({"merge_group": ["__object_number__"], "value": [value]}), keep_features=True)
    written = induced["pattern"].str.replace_all(r"\{1\}", "").item()
    assert written == signature(value)


@pytest.mark.parametrize("value", OBJECT_NUMBERS)
def test_slot_index_agrees_with_the_year_walk(value):
    """accession_years reads slot_idx from the export and walks the mask itself to fetch the value"""
    slots = re.findall(SLOT_RUNS, value)
    for idx, slot in enumerate(slots):
        walked = year_at_slot({"value": value, "masked": masked(value), "year_slot_idx": idx})
        expected = int(slot) if slot.isdigit() and len(slot) == 4 else None
        assert walked == expected


def test_a_line_break_survives_the_run_split():
    """A dimension written over three lines must keep its separators, or it induces the shape of one line"""
    value = "Length\r\nBreadth\r\nWeight"
    induced = induce_lookup(
        pl.LazyFrame({"merge_group": ["spectrum/dimension"], "value": [value]}), keep_features=True
    )
    assert induced["pattern"].item() == "s{6}\r\ns{7}\r\ns{6}"
    assert signature(value) == induced["pattern"].item()


def frame(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "data_source": ["Test Museum"] * n,
            "field_type": ["spectrum/object_number"] * n,
            "value": [f"AB.{1900 + i % 100}.{i}" for i in range(n)],
            "source_pattern": ["s{2}.d{4}.d"] * n,
            "merged_pattern": ["s{2}.d{4}.d{1,3}"] * n,
        }
    )


def test_a_slot_below_the_evidence_floor_is_not_written():
    assert slot_values(frame(MIN_SLOT_SUPPORT - 1)).height == 0
    assert slot_values(frame(MIN_SLOT_SUPPORT)).height > 0


def test_slot_kinds_separate_letters_from_digits():
    slots = slot_values(frame(MIN_SLOT_SUPPORT))
    assert set(slots["slot_kind"]) == {"s", "d"}
    letters = slots.filter(pl.col("slot_kind") == "s")
    assert letters["slot_idx"].to_list() == [0]
    assert set(letters["values"].explode()) == {"AB"}


def test_pattern_probabilities_sum_to_one_per_institution():
    dist = pattern_dist(pl.concat([frame(10), frame(5).with_columns(merged_pattern=pl.lit("d{4}"))]))
    assert dist["prob"].sum() == pytest.approx(1.0)


def test_members_carry_the_field_type_and_source_pattern_of_each_absorbed_pattern():
    merged = patterns_merged(frame(3))
    assert merged.height == 1
    assert merged["members"].item() == (
        '[{"field_type": "spectrum/object_number", "pattern": "s{2}.d{4}.d", "count": 3}]'
    )
