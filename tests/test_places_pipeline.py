from typing import ClassVar

import polars as pl
import pytest

import mds_norm.pipeline.apply_place_verdicts as apv
import mds_norm.pipeline.places_pipeline as pp

# Toy TGN: namesakes, a unique county, a prominence pair
TOY_INDEX = pl.DataFrame(
    [
        ("1", "London", "en", "prefLabelGVP"),
        ("1", "Greater London", "en", "altLabel"),
        ("2", "London", "en", "prefLabelGVP"),
        ("3", "England", "en", "prefLabelGVP"),
        ("3", "UK England", "en", "altLabel"),
        ("4", "United Kingdom", "en", "prefLabelGVP"),
        ("4", "UK", "en", "altLabel"),
        ("5", "Yeovil", "en", "prefLabelGVP"),
        ("6", "Somerset", "en", "prefLabelGVP"),
        ("7", "Ontario", "en", "prefLabelGVP"),
        ("8", "Canada", "en", "prefLabelGVP"),
        ("9", "England", "en", "prefLabel"),  # namesake hamlet
        ("10", "Nottinghamshire", "en", "prefLabelGVP"),
        ("11", "Saint Asaph", "en", "prefLabelGVP"),
    ],
    schema=["subject", "term", "lang", "kind"],
    orient="row",
).with_columns(norm=pp.norm_term(pl.col("term")))

# child → parent for the toy hierarchy
PARENT = {"1": "3", "2": "7", "3": "4", "5": "6", "6": "3", "9": "8", "10": "3", "11": "3"}


def _closure(parent: dict[str, str]) -> list[tuple[str, str]]:
    pairs = []
    for subject in parent:
        anc = parent.get(subject)
        while anc is not None:
            pairs.append((subject, anc))
            anc = parent.get(anc)
    return pairs


TOY_ANCESTORS = pl.DataFrame(_closure(PARENT), schema=["subject", "ancestor"], orient="row")

TOY_PROMINENCE = pl.DataFrame(
    {"subject": ["3", "4"], "prominence": [40, 4]}, schema_overrides={"prominence": pl.UInt32}
)


def make_distinct(atoms: list[str]) -> pl.DataFrame:
    return (
        pl.DataFrame({"atom": atoms})
        .with_columns(
            norm=pp.norm_term(pl.col("atom")),
            count=pl.lit(1, dtype=pl.UInt32),
            n_institutions=pl.lit(1, dtype=pl.UInt32),
        )
        .select("norm", "atom", "count", "n_institutions")
    )


def run_pipeline(atoms: list[str]) -> pl.DataFrame:
    routed = pp.route_values(make_distinct(atoms))
    parsed, malformed = pp.parse_segments(routed)
    candidates = pp.head_candidates(TOY_INDEX.lazy(), parsed.select("head_norm").unique())
    ctx = parsed.explode("context").select(norm="context").filter(pl.col("norm") != "").unique()
    cand_ctx = pp.context_matches(TOY_INDEX.lazy(), TOY_ANCESTORS.lazy(), candidates, ctx)
    resolutions = pl.concat(
        [
            pp.resolve_hierarchical(parsed, candidates, cand_ctx, TOY_PROMINENCE.lazy()),
            pp.resolve_simple(routed, TOY_INDEX.lazy(), TOY_PROMINENCE.lazy()),
        ]
    )
    return pp.assemble_decisions(routed, parsed, malformed, resolutions)


def decision(atoms: list[str], atom: str) -> dict:
    df = run_pipeline(atoms)
    return df.filter(pl.col("atom") == atom).row(0, named=True)


class TestAncestorClosure:
    def test_toy_closure_transitive(self):
        anc = TOY_ANCESTORS.filter(pl.col("subject") == "1")["ancestor"].to_list()
        assert set(anc) == {"3", "4"}

    def test_yeovil_reaches_country(self):
        anc = TOY_ANCESTORS.filter(pl.col("subject") == "5")["ancestor"].to_list()
        assert set(anc) == {"6", "3", "4"}


class TestRouter:
    def test_routes(self):
        routed = pp.route_values(
            make_distinct(
                [
                    "Unattributed place",
                    "near Yeovil",
                    "TQ 1234 5678",
                    "[London]",
                    "London, England",
                    "London (England)",
                    "Tryfan",
                ]
            )
        )
        got = dict(zip(routed["atom"], routed["route"], strict=True))
        assert got["Unattributed place"] == "semantic_marker"
        assert got["near Yeovil"] == "residue"
        assert got["TQ 1234 5678"] == "residue"
        assert got["[London]"] == "residue"
        assert got["London, England"] == "hierarchical"
        assert got["London (England)"] == "hierarchical"
        assert got["Tryfan"] == "simple"

    def test_levels(self):
        routed = pp.route_values(make_distinct(["Brooke Street", "Keswick Hall", "La Cotte Cave", "West Wales"]))
        got = dict(zip(routed["atom"], routed["level"], strict=True))
        assert got["Brooke Street"] == "street"
        assert got["Keswick Hall"] == "building"
        assert got["La Cotte Cave"] == "site"
        assert got["West Wales"] == "area"


class TestHierarchicalResolution:
    def test_narrow_first_full_context(self):
        row = decision(["London, England"], "London, England")
        assert row["status"] == "resolved"
        assert row["subject"] == "1"
        assert row["resolved_by"] == "context_full"
        assert row["matched_term"] == "London"
        assert row["qualifier"] == "England"
        assert row["confidence"] == pytest.approx(0.95)

    def test_broad_first_chain(self):
        row = decision(["UK, England, Nottinghamshire"], "UK, England, Nottinghamshire")
        assert row["status"] == "resolved"
        assert row["subject"] == "10"
        assert row["resolved_by"] == "context_full"
        assert row["qualifier"] == "UK, England"

    def test_context_disambiguates_homograph(self):
        ont = decision(["London, Ontario"], "London, Ontario")
        assert (ont["status"], ont["subject"]) == ("resolved", "2")
        eng = decision(["London, United Kingdom"], "London, United Kingdom")
        assert (eng["status"], eng["subject"]) == ("resolved", "1")

    def test_paren_qualifier_is_context(self):
        row = decision(["London (England)"], "London (England)")
        assert row["status"] == "resolved"
        assert row["subject"] == "1"
        assert row["sub_component"] == "paren_hierarchy"
        assert row["confidence"] == pytest.approx(0.9)

    def test_alt_label_context_matches(self):
        # 'UK' is an altLabel, not a hand-kept expansion
        row = decision(["Yeovil, UK"], "Yeovil, UK")
        assert (row["status"], row["subject"]) == ("resolved", "5")

    def test_wrong_context_flags_best_candidate(self):
        row = decision(["London, Canada"], "London, Canada")
        assert row["status"] == "flagged"
        assert row["resolved_by"] is None
        assert row["subject"] is not None  # best candidate attached

    def test_unmatched_head_defers_by_level(self):
        row = decision(["Brooke Street, Holborn"], "Brooke Street, Holborn")
        assert row["status"] == "deferred"
        assert row["defer_reason"] == "sub_settlement"
        misspelt = decision(["Colchster, England"], "Colchster, England")
        assert misspelt["defer_reason"] == "place_no_match"

    def test_prominence_settles_full_context_tie(self):
        # 'England, United Kingdom' resolves via context, not prominence
        row = decision(["England, United Kingdom"], "England, United Kingdom")
        assert row["status"] == "resolved"
        assert row["subject"] == "3"

    def test_malformed_comma_value_defers_as_residue(self):
        row = decision(["a, b, c, d, e, f, g, h"], "a, b, c, d, e, f, g, h")
        assert row["status"] == "deferred"
        assert row["defer_reason"] == "place_residue"


class TestSeparatorConventions:
    def test_gt_separator(self):
        row = decision(["UK > England > Nottinghamshire"], "UK > England > Nottinghamshire")
        assert (row["status"], row["subject"]) == ("resolved", "10")
        assert row["qualifier"] == "UK, England"

    def test_colon_separator(self):
        row = decision(["United Kingdom: England"], "United Kingdom: England")
        assert (row["status"], row["subject"]) == ("resolved", "3")


class TestAbbrevVariants:
    def test_variant_generation(self):
        assert pp.abbrev_variants("st asaph") == ["saint asaph"]
        assert pp.abbrev_variants("st. asaph") == ["st asaph", "saint asaph"]
        assert pp.abbrev_variants("yeovil") == []

    def test_simple_st_resolves_unique(self):
        row = decision(["St Asaph"], "St Asaph")
        assert row["status"] == "resolved"
        assert row["subject"] == "11"
        assert row["resolved_by"] == "abbrev_unique"
        assert row["matched_term"] == "Saint Asaph"
        assert row["sub_component"] == "abbrev"
        assert row["confidence"] == pytest.approx(0.9)

    def test_hierarchical_st_head_uses_context(self):
        row = decision(["St Asaph, England"], "St Asaph, England")
        assert (row["status"], row["subject"]) == ("resolved", "11")
        assert row["resolved_by"] == "context_full"


class TestSimpleRoute:
    def test_simple_defers_no_retry(self):
        row = decision(["Tryfan"], "Tryfan")
        assert row["status"] == "deferred"
        assert row["defer_reason"] == "place_no_match"

    def test_street_defers_to_sub_settlement(self):
        row = decision(["Brooke Street"], "Brooke Street")
        assert row["defer_reason"] == "sub_settlement"

    def test_semantic_marker_persists(self):
        row = decision(["Unattributed place"], "Unattributed place")
        assert row["status"] == "deferred"
        assert row["defer_reason"] == "semantic_marker"


class TestApplyVerdicts:
    """The rung verdicts the review sample measured, folded back into the decisions"""

    DECISIONS = pl.DataFrame(
        [
            ("os unique", "os_open_names", "unique", "resolved", 0.855, 100),
            ("geonames unique", "geonames", "unique", "resolved", 0.8075, 200),
            ("geonames prominent", "geonames", "prominent", "resolved", 0.765, 50),
            ("geonames flagged", "geonames", None, "flagged", None, 30),
        ],
        schema=["norm", "vocab", "resolved_by", "status", "confidence", "count"],
        orient="row",
    )
    VERDICTS: ClassVar[dict[str, dict]] = {
        "fallback/os_open_names/unique": {"action": "keep", "precision": 1.0},
        "fallback/geonames/unique": {"action": "demote", "precision": 0.75},
        "fallback/geonames/prominent": {"action": "demote", "precision": 0.65},
        "fallback/geonames/flagged": {"action": "promote", "precision": 0.9},
    }

    def applied(self) -> dict[str, dict]:
        out = apv.apply_verdicts(self.DECISIONS, apv.FALLBACK_RUNG, self.VERDICTS)
        return {row["norm"]: row for row in out.iter_rows(named=True)}

    def test_a_demoted_rung_flags_its_values_and_drops_their_confidence(self):
        row = self.applied()["geonames unique"]
        assert (row["status"], row["confidence"]) == ("flagged", None)

    def test_the_verdict_keys_on_the_gazetteer_as_well_as_the_rung(self):
        # `unique` is demoted on GeoNames, kept on OS
        row = self.applied()["os unique"]
        assert (row["status"], row["confidence"]) == ("resolved", 0.855)

    def test_a_promotion_is_left_for_the_release_decision(self):
        row = self.applied()["geonames flagged"]
        assert (row["status"], row["confidence"]) == ("flagged", None)

    def test_applying_twice_changes_nothing_further(self):
        once = apv.apply_verdicts(self.DECISIONS, apv.FALLBACK_RUNG, self.VERDICTS)
        twice = apv.apply_verdicts(once, apv.FALLBACK_RUNG, self.VERDICTS)
        assert once.equals(twice)

    def test_an_unscored_rung_is_left_as_the_cascade_left_it(self):
        decisions = pl.DataFrame(
            [("abbrev unique", "abbrev_unique", "abbrev", "resolved", 0.9, 10)],
            schema=["norm", "resolved_by", "ctx_state", "status", "confidence", "count"],
            orient="row",
        )
        row = apv.apply_verdicts(decisions, apv.TGN_RUNG, {}).row(0, named=True)
        assert (row["status"], row["confidence"]) == ("resolved", 0.9)
