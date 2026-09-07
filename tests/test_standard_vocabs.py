import polars as pl
import pytest

from mds_norm.paths import VOCABS
from mds_norm.pipeline import build_standard_vocabs as bsv


def index(vocab: str) -> pl.DataFrame:
    if not (VOCABS / vocab).exists():
        pytest.skip(f"{vocab} not fetched (see fetch_vocabs.sh)")
    return bsv.build(vocab)


@pytest.mark.parametrize("vocab", sorted(bsv.BUILDERS))
def test_index_schema(vocab):
    frame = index(vocab)
    assert frame.columns == ["subject", "term", "lang", "kind", "norm"]
    assert set(frame["kind"].unique()) <= {"prefLabel", "altLabel"}
    assert (frame["norm"] != "").all()
    assert len(frame) > 0


@pytest.mark.parametrize("vocab", sorted(bsv.BUILDERS))
def test_one_pref_label_per_subject_and_language(vocab):
    frame = index(vocab)
    prefs = frame.filter(pl.col("kind") == "prefLabel")
    # SKOS allows one preferred label per language
    assert len(prefs.unique(subset=["subject", "lang"])) == len(prefs)
    # every altLabel's subject also has a preferred term
    assert set(frame["subject"]) == set(prefs["subject"])


def test_gbif_rank_carries_the_corpus_ranks():
    norms = set(index("gbif_rank")["norm"])
    assert {"kingdom", "phylum", "class", "order", "family", "genus", "species"} <= norms


def test_gbif_type_status_carries_the_specimen_terms():
    norms = set(index("gbif_type_status")["norm"])
    assert {"holotype", "paratype", "syntype", "lectotype", "paralectotype"} <= norms


def test_dcmi_type_indexes_label_and_class_name():
    frame = index("dcmi_type")
    still = frame.filter(pl.col("subject") == "StillImage")
    assert {"still image", "stillimage"} <= set(still["norm"])


def test_iana_indexes_the_registered_name_beside_the_full_type():
    frame = index("iana_media_types")
    jpeg = frame.filter(pl.col("subject") == "image/jpeg")
    assert {"image/jpeg", "jpeg"} <= set(jpeg["norm"])
    assert jpeg.filter(pl.col("kind") == "prefLabel")["term"].to_list() == ["image/jpeg"]


def test_loc_relators_reads_terms_uf_and_deprecated_forms():
    frame = index("loc_relators")
    assert frame.filter(pl.col("norm") == "manufacturer")["subject"].to_list() == ["mfr"]
    # a UF cross-reference indexes under the term it defers to
    assert frame.filter(pl.col("norm") == "recipient")["subject"].to_list() == ["rcp"]
    # ... as does a deprecated term's USE replacement
    assert frame.filter(pl.col("norm") == "graphic technician")["subject"].to_list() == ["art"]


def test_fish_monument_types_maps_non_preferred_terms_to_their_preferred_id():
    frame = index("fish_monument_types")
    assert {"hillfort", "midden", "colliery", "long barrow"} <= set(frame["norm"])
    # 'Byre' is non-preferred: indexed under a preferred subject
    byre = frame.filter(pl.col("norm") == "byre")
    assert byre["kind"].to_list() == ["altLabel"]
    assert byre["subject"][0] in set(frame.filter(pl.col("kind") == "prefLabel")["subject"])
