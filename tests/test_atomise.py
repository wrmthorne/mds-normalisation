import pytest

from mds_norm.utils import atomise as atomise_mod
from mds_norm.utils.atomise import (
    NULL_MARKERS,
    PLACEHOLDER_MARKERS,
    SEMANTIC_MARKERS,
    atomise,
    morph_variants,
    parse_llm_atoms,
    split_pattern,
)


@pytest.mark.parametrize(
    ("norm", "expected"),
    [
        # the motivating case: the technique noun is generated
        ("engraved", ["engraveing", "engraving"]),
        ("gilded", ["gildeing", "gilding"]),
        # e-restored form comes first: "dyeing" before "dying"
        ("glazed", ["glazeing", "glazing"]),
        # multi-word atoms vary one token at a time
        ("hand painted", ["hand painteing", "hand painting"]),
        # too short, or not a participle shape: nothing generated
        ("red", []),
        ("engraving", []),
        ("tweed", []),  # -eed is not a participle suffix
    ],
)
def test_morph_variants(norm, expected):
    assert morph_variants(norm) == expected


def test_marker_partition():
    # semantic markers carry meaning, so placeholders must exclude them
    assert set() == SEMANTIC_MARKERS & PLACEHOLDER_MARKERS
    assert NULL_MARKERS == PLACEHOLDER_MARKERS | SEMANTIC_MARKERS
    assert {"unknown", "not known", "unspecified", "unidentified"} <= SEMANTIC_MARKERS
    assert {"-", "n/a", "none", "tbc"} <= PLACEHOLDER_MARKERS


def test_base_separators_always_apply():
    atoms = atomise("wood; metal|glass", None)
    assert [a["atom"] for a in atoms] == ["wood", "metal", "glass"]


def test_candidate_separators_only_when_accepted():
    assert [a["atom"] for a in atomise("oak & pine", None)] == ["oak & pine"]
    assert [a["atom"] for a in atomise("oak & pine", ["ampersand"])] == ["oak", "pine"]


def test_spans_index_the_original_string():
    value = " wood ;  metal "
    for a in atomise(value, None):
        assert value[a["span_start"] : a["span_end"]] == a["atom"]


def test_comma_not_split_between_digits():
    atoms = atomise("1,200 coins, silver", ["comma"])
    assert [a["atom"] for a in atoms] == ["1,200 coins", "silver"]


def test_separator_only_value_yields_nothing():
    assert atomise(";;;", None) == []


def test_split_pattern_cached_and_sorted():
    assert split_pattern(["comma", "and"]) is split_pattern(["and", "comma"])


def test_parse_valid_array():
    out = parse_llm_atoms("oak and pine", '["oak", "pine"]')
    assert [(d["sub_atom"], d["sub_start"], d["sub_end"]) for d in out] == [("oak", 0, 3), ("pine", 8, 12)]


def test_parse_recovers_source_casing():
    out = parse_llm_atoms("Oak and Pine", '["oak", "pine"]')
    assert [d["sub_atom"] for d in out] == ["Oak", "Pine"]


def test_parse_rejects_rewrites():
    assert parse_llm_atoms("oak and pine", '["oak wood", "pine"]') == [
        {"sub_atom": "pine", "sub_start": 8, "sub_end": 12}
    ]


def test_parse_rejects_whole_value_echo():
    assert parse_llm_atoms("oak and pine", '["oak and pine"]') is None


def test_parse_dedupes_case_insensitively():
    out = parse_llm_atoms("oak, Oak, oak", '["oak", "Oak"]')
    assert len(out) == 1


def test_parse_enforces_token_cap():
    long_atom = "one two three four five six seven"
    assert parse_llm_atoms(long_atom + " x", f'["{long_atom}"]') is None


def test_parse_cursor_prefers_ordered_matches():
    out = parse_llm_atoms("tin box, tin", '["box", "tin"]')
    # "tin" matched after the cursor (position 9), not at 0
    assert out[1] == {"sub_atom": "tin", "sub_start": 9, "sub_end": 12}


def test_parse_extracts_first_array_from_wrapping():
    # lenient bracket extraction: a wrapped array still parses
    out = parse_llm_atoms("oak and pine", '{"atoms": ["oak"]}')
    assert [d["sub_atom"] for d in out] == ["oak"]


@pytest.mark.parametrize("completion", ["no json here", "{}", "[]", '["", "  "]', ""])
def test_parse_junk_returns_none(completion):
    assert parse_llm_atoms("oak and pine", completion) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("engraving on paper", ("engraving", "paper", "on paper")),
        ("woodblock print on paper", ("woodblock print", "paper", "on paper")),
        ("printed in black", ("printed", "black", "in black")),
        ("oil on canvas", ("oil", "canvas", "on canvas")),
        ("watercolour with bodycolour", ("watercolour", "bodycolour", "with bodycolour")),
        # leftmost preposition only; an unattested tail refuses the value
        ("ink on paper on board", ("ink", "paper on board", "on paper on board")),
        # nothing to split, or a prose or date side
        ("on paper", None),
        ("engraving", None),
        ("printed in 1880", None),
        ("a very long technique name on some other long support here", None),
    ],
)
def test_compound_head(value, expected):
    assert atomise_mod.compound_head(value) == expected


def test_compound_fields_are_the_out_of_field_tail_ones():
    # material excluded: its tail is a second material
    assert {"spectrum/technique", "spectrum/inscription_method"} == atomise_mod.COMPOUND_FIELDS
    assert "spectrum/material" not in atomise_mod.COMPOUND_FIELDS
