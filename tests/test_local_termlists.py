import json

import polars as pl
import pytest

import mds_norm.pipeline.build_local_termlists as blt
from mds_norm.paths import VOCABS

SEEDS = sorted(p.stem for p in (VOCABS / "local").glob("*.json"))


def test_every_seed_is_registered_in_the_field_map():
    registered = json.loads((VOCABS / "field_vocab_map.json").read_text())
    for seed in SEEDS:
        vocab = json.loads((VOCABS / "local" / f"{seed}.json").read_text())["vocab"]
        for field in blt.seed_fields(seed):
            assert field in registered, f"{seed} serves {field}, which is unregistered"
            assert vocab in registered[field]


@pytest.mark.parametrize("seed", SEEDS)
def test_index_schema(seed):
    index = blt.load_termlist_index(seed)
    assert index.columns == ["subject", "term", "lang", "kind", "norm"]
    assert set(index["kind"].unique()) <= {"prefLabel", "altLabel"}
    assert (index["norm"] != "").all()


@pytest.mark.parametrize("seed", SEEDS)
def test_one_pref_label_per_subject(seed):
    index = blt.load_termlist_index(seed)
    prefs = index.filter(pl.col("kind") == "prefLabel")
    assert prefs["subject"].n_unique() == len(prefs)
    # every subject has a prefLabel (alts never orphaned)
    assert set(index["subject"]) == set(prefs["subject"])


@pytest.mark.parametrize("seed", SEEDS)
def test_norms_are_unambiguous(seed):
    index = blt.load_termlist_index(seed)
    assert not index["norm"].is_duplicated().any()


@pytest.mark.parametrize("seed", SEEDS)
def test_every_term_carries_a_source(seed):
    # the tag records a term's distance from the corpus
    seed_json = json.loads((VOCABS / "local" / f"{seed}.json").read_text())
    allowed = {"harvest", "harvest+spectrum", "spectrum-form", "harvest-decoded"}
    for term in seed_json["terms"]:
        assert term.get("source", "harvest") in allowed, term


def test_shared_seeds_declare_the_fields_they_serve():
    assert blt.seed_fields("persons_association") == ["persons_association", "organisations_association"]
    assert blt.seed_fields("association_event") == ["date_association", "place_association"]
    # a single-field seed needs no declaration
    assert blt.seed_fields("condition") == ["condition"]


def test_homograph_in_seed_rejected(tmp_path, monkeypatch):
    seed = {"field": "bad", "terms": [{"term": "maker", "alt": ["made by"]}, {"term": "producer", "alt": ["made by"]}]}
    (tmp_path / "bad.json").write_text(json.dumps(seed))
    monkeypatch.setattr(blt, "LOCAL_VOCAB_PATH", tmp_path)
    with pytest.raises(ValueError, match="homograph"):
        blt.load_termlist_index("bad")
