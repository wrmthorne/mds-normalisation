import polars as pl
import pytest

from mds_norm.pipeline import places_fallback as pf


def index(rows):
    """(subject, term, context[, population]) -> the shape resolve_against reads"""
    return pl.DataFrame(
        [{"subject": s, "term": t, "norm": t.lower(), "context": c, "population": p} for s, t, c, p in rows],
        schema={
            "subject": pl.String,
            "term": pl.String,
            "norm": pl.String,
            "context": pl.List(pl.String),
            "population": pl.Int64,
        },
    )


def targets(rows):
    """(norm, head_norm, context) -> the shape head_and_context produces"""
    return pl.DataFrame(
        [
            {"norm": n, "orientation": "first" if c else "simple", "head_norm": h, "context": c, "qualifier": None}
            for n, h, c in rows
        ],
        schema={
            "norm": pl.String,
            "orientation": pl.String,
            "head_norm": pl.String,
            "context": pl.List(pl.String),
            "qualifier": pl.String,
        },
    )


BOSTONS = index(
    [
        ("1", "Boston", ["united states", "massachusetts"], 654776),
        ("2", "Boston", ["united kingdom", "england", "lincolnshire"], 41340),
        ("3", "Boston", ["united kingdom", "england", "west yorkshire"], 0),
    ]
)


def resolve(idx, tgts, vocab="geonames", prominence=True):
    return pf.resolve_against(idx, tgts, vocab, prominence=prominence)


def test_stated_context_selects_the_candidate_that_carries_it():
    out = resolve(BOSTONS, targets([("boston, lincolnshire", "boston", ["lincolnshire"])]))
    assert out["resolved_by"].to_list() == ["context_full"]
    assert out["subject"].to_list() == ["2"]


def test_context_the_gazetteer_does_not_carry_flags_rather_than_guesses():
    out = resolve(BOSTONS, targets([("boston, cornwall", "boston", ["cornwall"])]))
    assert out["resolved_by"].to_list() == [None]  # flagged, best candidate kept
    assert out["subject"].to_list() == ["1"]  # most prominent as the best


def test_several_full_context_matches_flag():
    twins = index([("1", "Newport", ["wales", "gwent"], 10), ("2", "Newport", ["wales", "pembrokeshire"], 10)])
    out = resolve(twins, targets([("newport, wales", "newport", ["wales"])]))
    assert out["resolved_by"].to_list() == [None]


def test_a_value_with_no_context_resolves_only_when_unique():
    unique = index([("9", "Bradnop", ["england", "staffordshire"], 0)])
    assert resolve(unique, targets([("bradnop", "bradnop", [])]))["resolved_by"].to_list() == ["unique"]
    # …and never when the gazetteer offers a choice
    assert resolve(BOSTONS, targets([("boston", "boston", [])]))["resolved_by"].to_list() == ["prominent"]


def test_population_prior_needs_size_and_dominance():
    close = index([("1", "Springfield", ["us", "illinois"], 116000), ("2", "Springfield", ["us", "missouri"], 169000)])
    assert resolve(close, targets([("springfield", "springfield", [])]))["resolved_by"].to_list() == [None]
    small = index([("1", "Tinytown", ["us", "iowa"], 300), ("2", "Tinytown", ["us", "ohio"], 5)])
    assert resolve(small, targets([("tinytown", "tinytown", [])]))["resolved_by"].to_list() == [None]


def test_os_has_no_population_prior_so_an_ambiguous_gb_name_defers():
    lanes = index([("1", "Cuckoo Lane", ["hampshire", "england"], 0), ("2", "Cuckoo Lane", ["devon", "england"], 0)])
    out = resolve(lanes, targets([("cuckoo lane", "cuckoo lane", [])]), vocab="os_open_names", prominence=False)
    assert out["resolved_by"].to_list() == [None]


def test_os_records_satisfy_the_country_context_by_construction():
    # every OS record is in Great Britain
    holborn = index([("1", "Holborn", ["london", "greater london", "england"], 0)])
    out = resolve(holborn, targets([("holborn, uk", "holborn", ["uk"])]), vocab="os_open_names", prominence=False)
    assert out["resolved_by"].to_list() == ["context_full"]
    # …but GeoNames is global, so no free pass
    out = resolve(holborn, targets([("holborn, uk", "holborn", ["uk"])]))
    assert out["resolved_by"].to_list() == [None]


def test_no_candidate_at_all_returns_the_empty_contract():
    out = resolve(BOSTONS, targets([("nowhere", "nowhere", [])]))
    assert out.height == 0
    assert set(out.columns) >= {"norm", "subject", "resolved_by", "vocab"}


def test_os_is_preferred_where_both_authorities_resolve():
    both = pl.DataFrame(
        [
            {
                "norm": "bradnop",
                "orientation": "simple",
                "qualifier": None,
                "n_candidates": 1,
                "subject": "geo",
                "matched_term": "Bradnop",
                "resolved_by": "unique",
                "vocab": "geonames",
            },
            {
                "norm": "bradnop",
                "orientation": "simple",
                "qualifier": None,
                "n_candidates": 1,
                "subject": "os",
                "matched_term": "Bradnop",
                "resolved_by": "unique",
                "vocab": "os_open_names",
            },
        ]
    )
    assert pf.best_per_norm(both)["vocab"].to_list() == ["os_open_names"]


def test_a_resolved_reading_beats_a_flagged_one_from_either_authority():
    mixed = pl.DataFrame(
        [
            {
                "norm": "x",
                "orientation": "first",
                "qualifier": None,
                "n_candidates": 4,
                "subject": "os",
                "matched_term": "X",
                "resolved_by": None,
                "vocab": "os_open_names",
            },
            {
                "norm": "x",
                "orientation": "first",
                "qualifier": None,
                "n_candidates": 1,
                "subject": "geo",
                "matched_term": "X",
                "resolved_by": "context_full",
                "vocab": "geonames",
            },
        ]
    )
    best = pf.best_per_norm(mixed)
    assert best["vocab"].to_list() == ["geonames"]
    assert best["resolved_by"].to_list() == ["context_full"]


@pytest.mark.parametrize("reason", pf.CONSUMED)
def test_only_the_two_no_match_queues_are_consumed(reason):
    # a gazetteer must not get what the vocabulary refused
    assert reason in ("place_no_match", "sub_settlement")
    assert "place_residue" not in pf.CONSUMED
    assert "semantic_marker" not in pf.CONSUMED
