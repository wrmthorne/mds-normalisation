import polars as pl
import pytest

from mds_norm.pipeline import vocab_alignment as va
from mds_norm.pipeline import vocab_rerank as vr
from mds_norm.pipeline.vocab_indexes import CONCEPT_GROUPS
from mds_norm.utils.atomise import GROUP_DESC

MATERIAL = "aat+fish_building_materials"


def atoms(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    """(institution, value, group) triples shaped like the cascade's atom table"""
    return pl.DataFrame(
        [
            {
                "group": group,
                "data_source": inst,
                "value": value,
                "atom": value,
                "norm": value.lower(),
                "count": 10,
                "split_ok": True,
                "compound_ok": False,
                "atom_route": "cascade",
                "span_start": 0,
                "span_end": len(value),
            }
            for inst, value, group in rows
        ]
    )


def test_exclusive_evidence_outranks_the_base_vocabulary():
    prediction = {"vocabs": ["aat", "bm_materials"], "specific_evidence": {"bm_materials": {}}}
    assert va.house_vocabs(prediction, MATERIAL) == ["bm_materials"]


def test_a_group_vocabulary_is_never_a_house_vocabulary():
    # aat is the group's own target, not a house list
    assert va.house_vocabs({"vocabs": ["aat"]}, MATERIAL) == []


def test_an_unbuilt_vocabulary_is_ignored():
    assert va.house_vocabs({"vocabs": ["something_unpublished"]}, MATERIAL) == []


def test_the_detection_credits_at_least_one_institution_a_house_list():
    ranks = va.house_ranks()
    assert len(ranks)
    assert set(ranks["vocab"]) <= set(va.HOUSE_BUILDERS)


def test_the_house_list_wins_and_the_pooled_match_survives_as_the_crosswalk():
    inst = va.house_ranks().filter(pl.col("vocab") == "bm_materials")["data_source"][0]
    frame = atoms([(inst, "Limestone", MATERIAL), ("Museum Without A House List", "Limestone", MATERIAL)])
    decisions = va.assemble(frame, va.pooled_ladder(frame), va.house_tier(frame), *va.no_llm_tier())

    flagged, pooled = (
        decisions.filter(pl.col("data_source") == inst).row(0, named=True),
        decisions.filter(pl.col("data_source") != inst).row(0, named=True),
    )
    assert flagged["vocab"] == "bm_materials"
    assert flagged["sub_component"] == "house_exact"
    assert flagged["xref_vocab"] == pooled["vocab"]
    assert flagged["xref_subject"] == pooled["subject"]
    # no house list means no crosswalk
    assert pooled["sub_component"] == "exact"
    assert pooled["xref_vocab"] is None


def test_the_house_overlay_takes_the_corpus_enum_for_institutions():
    """The corpus keys institutions as an Enum and `house_ranks` as strings; the tier joins both"""
    inst = va.house_ranks().filter(pl.col("vocab") == "bm_materials")["data_source"][0]
    frame = atoms([(inst, "Limestone", MATERIAL)]).with_columns(
        pl.col("data_source").cast(pl.Enum([inst, "Museum Without A House List"]))
    )
    decisions = va.assemble(frame, va.pooled_ladder(frame), va.house_tier(frame), *va.no_llm_tier())
    assert decisions.row(0, named=True)["vocab"] == "bm_materials"
    assert decisions.schema["data_source"] == frame.schema["data_source"]


def test_an_atom_missing_from_the_house_list_falls_back_to_the_pooled_vocabularies():
    inst = va.house_ranks().filter(pl.col("vocab") == "bm_materials")["data_source"][0]
    frame = atoms([(inst, "polyurethane foam", MATERIAL)])
    decisions = va.assemble(frame, va.pooled_ladder(frame), va.house_tier(frame), *va.no_llm_tier())
    row = decisions.row(0, named=True)
    assert row["vocab"] == "aat"
    assert row["sub_component"] == "exact"
    assert row["xref_vocab"] is None  # a pooled hit is the link itself, not a crosswalk


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("3", 3),
        ("Answer: 1", 1),
        ("The value matches option 2.", 2),
        ("0", None),  # the reject option
        ("7", None),  # out of range
        ("no idea", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_choice(completion, expected):
    assert vr.parse_choice(completion, 5) == expected


def test_render_options_numbers_from_one_and_tolerates_a_missing_gloss():
    assert vr.render_options(["tin", "oak"], ["a metal", None]) == "1. tin — a metal\n2. oak"


def test_every_concept_group_can_be_described_to_the_model():
    assert set(GROUP_DESC) >= CONCEPT_GROUPS


def test_the_queue_skips_prose_dates_and_untyped_atoms():
    pending = pl.DataFrame(
        {
            "group": ["aat", "aat", "periodo", MATERIAL, "tgn"],
            "atom": ["a", "moulded glass", "1875", "x" * 120, "Sheffield"],
            "norm": ["a", "moulded glass", "1875", "x" * 120, "sheffield"],
            "count": [1, 5, 3, 2, 9],
        }
    )
    assert set(vr.rerank_queue(pending)["norm"]) == {"moulded glass"}


def test_the_queue_caps_the_singleton_tail_but_keeps_the_atom_that_crosses_the_cap():
    tail = [f"tail term {i}" for i in range(400)]
    pending = pl.DataFrame(
        {
            "group": ["aat"] * (len(tail) + 1),
            "atom": ["common term", *tail],
            "norm": ["common term", *tail],
            "count": [10_000, *([1] * len(tail))],
        }
    )
    queued = set(vr.rerank_queue(pending)["norm"])
    assert "common term" in queued
    assert len(queued) < len(tail)


def test_the_splitter_hands_on_what_it_could_not_place():
    pending = pl.DataFrame(
        {
            "group": ["aat", "aat"],
            "atom": ["oil on canvas", "sheet steele"],
            "norm": ["oil on canvas", "sheet steele"],
            "count": [4, 2],
        }
    )
    split = pl.DataFrame(
        {
            "group": ["aat", "aat"],
            "atom": ["oil on canvas", "oil on canvas"],
            "sub_atom": ["oil", "canvas"],
            "sub_start": [0, 8],
            "sub_end": [3, 14],
            "sub_norm": ["oil", "canvas"],
        }
    )
    lookup = pl.DataFrame({"group": ["aat"], "norm": ["oil"], "subject": ["300015050"]})
    handed_on = va.rerank_pending(pending, split, lookup)
    # the split parent goes; unmatched sub-atoms are queued
    assert set(handed_on["norm"]) == {"canvas", "sheet steele"}


def place_index(rows: list[tuple[str, str, str]]) -> pl.LazyFrame:
    """(subject, term, kind) candidates for one norm, shaped like a group index"""
    return pl.DataFrame(
        [
            {
                "subject": subject,
                "term": term,
                "lang": "en",
                "kind": kind,
                "norm": term.lower(),
                "vocab": "tgn",
                "vocab_priority": 0,
            }
            for subject, term, kind in rows
        ]
    ).lazy()


def prior(rows: list[tuple[str, int]]) -> pl.LazyFrame:
    return pl.DataFrame(
        [{"subject": s, "preference": p, "pref_tiebreak": 0} for s, p in rows],
        schema_overrides={"preference": pl.Int8, "pref_tiebreak": pl.Int32},
    ).lazy()


def sizes(rows: list[tuple[str, int]]) -> pl.LazyFrame:
    return pl.DataFrame(
        [{"subject": s, "prominence": n} for s, n in rows], schema_overrides={"prominence": pl.UInt32}
    ).lazy()


# Getty prefers a Mississippi township for 'Perthshire'
PERTHSHIRE = [("2057207", "Perthshire", "prefLabelGVP"), ("7019152", "Perthshire", "altLabel")]


def test_the_kind_tier_alone_reads_perthshire_as_the_mississippi_township():
    got = va.resolve_norms(place_index(PERTHSHIRE), None)
    assert got["subject"].to_list() == ["2057207"]
    assert got["resolved_by"].to_list() == ["kind_tier"]


def test_the_prior_settles_perthshire_on_the_scottish_county_ahead_of_the_kind_tier():
    got = va.resolve_norms(place_index(PERTHSHIRE), None, preference=prior([("2057207", 3), ("7019152", 1)]))
    assert got["subject"].to_list() == ["7019152"]
    assert got["resolved_by"].to_list() == ["spatial"]


def test_a_major_place_outranks_a_minor_local_one_carrying_the_same_name():
    # a local dock named Canada must not win
    index = place_index([("1000005", "Canada", "prefLabelGVP"), ("7011781", "Canada", "altLabel")])
    got = va.resolve_norms(
        index, None, prominence=sizes([("1000005", 13)]), preference=prior([("1000005", 0), ("7011781", 1)])
    )
    assert got["subject"].to_list() == ["1000005"]
    # the size guard is credited before the prior
    assert got["resolved_by"].to_list() == ["prominent"]
