import polars as pl
import pytest

from mds_norm.pipeline import agent_links as al

# two Turners share a name; Wedgwood is person and firm
ULAN = [
    ("500011051", "Turner, Joseph Mallord William", None, "prefLabelGVP"),
    ("500011051", "J. M. W. Turner", "en", "altLabel"),
    ("500030481", "Turner, Joseph Mallord William", None, "prefLabel"),
    ("500042976", "Basan", "en", "prefLabelGVP"),
    ("500123456", "Wedgwood", "en", "prefLabelGVP"),
    ("500999001", "Doulton & Co.", "en", "prefLabelGVP"),
    ("500999001", "Doulton and Company", "en", "altLabel"),
]
FACETS = {
    "500011051": "person",
    "500030481": "person",
    "500042976": "person",
    "500123456": "person",
    "500999001": "corporate",
}


@pytest.fixture
def ulan(monkeypatch):
    index = pl.DataFrame(ULAN, schema=["subject", "term", "lang", "kind"], orient="row").with_columns(
        norm=pl.col("term").str.to_lowercase()
    )
    facets = pl.DataFrame({"subject": list(FACETS), "facet": list(FACETS.values())})
    # Basan alone has a Wikidata entity; one ISNI matches
    ulan_qids = pl.DataFrame({"subject": ["500042976"], "qid": ["Q123"]})
    isni_qids = pl.DataFrame({"isni": ["0000000121234567"], "qid": ["Q123"]})
    monkeypatch.setattr(al, "indexes", lambda: {"ulan": index.lazy()})
    monkeypatch.setattr(al, "ulan_facets", facets.lazy)
    monkeypatch.setattr(al, "ulan_wikidata", ulan_qids.lazy)
    monkeypatch.setattr(al, "isni_wikidata", lambda entity_type: isni_qids.lazy())
    monkeypatch.setattr(al, "ISNI_EXPORTS", {"person": "ISNI_persons.rdf.gz"})


def annotations(tmp_path, *rows):
    frame = pl.DataFrame(
        [(f"r{i}", f"n{i}", "aberdeen", "spectrum/object_production_person", *row) for i, row in enumerate(rows)],
        schema=["record_id", "node_id", "data_source", "field_type", "value", "entity_type", "status"],
        orient="row",
    ).with_columns(span_start=pl.lit(0, dtype=pl.UInt32), span_end=pl.col("value").str.len_chars())
    path = tmp_path / "person_annotations.parquet"
    frame.write_parquet(path)
    return path


def link(tmp_path, *rows, vocab="ulan"):
    return al.link_agents(annotations(tmp_path, *rows)).collect().filter(pl.col("vocab") == vocab)


def test_a_name_form_held_by_one_agent_resolves(tmp_path, ulan):
    row = link(tmp_path, ("Basan", "person", "resolved")).row(0, named=True)
    assert (row["subject"], row["status"], row["n_candidates"]) == ("500042976", "resolved", 1)
    assert row["vocab"] == "ulan"
    assert row["matched_term"] == "Basan"


def test_a_name_form_shared_by_two_agents_defers(tmp_path, ulan):
    row = link(tmp_path, ("Turner, Joseph Mallord William", "person", "resolved")).row(0, named=True)
    assert (row["status"], row["defer_reason"], row["n_candidates"]) == ("deferred", "ambiguous_authority", 2)
    assert row["subject"] is None
    assert row["matched_term"] is None


def test_the_facet_keeps_an_organisation_off_a_person_record(tmp_path, ulan):
    linked = link(tmp_path, ("Wedgwood", "organisation", "resolved"), ("Wedgwood", "person", "resolved"))
    assert linked["value"].to_list() == ["Wedgwood"]
    assert linked.row(0, named=True)["subject"] == "500123456"


def test_an_organisation_matches_the_corporate_facet(tmp_path, ulan):
    row = link(tmp_path, ("Doulton and Company", "organisation", "resolved")).row(0, named=True)
    assert (row["subject"], row["status"]) == ("500999001", "resolved")
    # matched_term reports the surface that matched
    assert row["matched_term"] == "Doulton and Company"


def test_unparsed_and_collective_agents_are_left_alone(tmp_path, ulan):
    linked = link(
        tmp_path, ("Basan", "person", "deferred"), ("Basan", "people", "resolved"), ("Basan", "residue", "deferred")
    )
    assert len(linked) == 0


def test_the_isni_bridge_adds_an_identifier_without_matching_a_name(tmp_path, ulan):
    row = link(tmp_path, ("Basan", "person", "resolved"), vocab="isni").row(0, named=True)
    assert (row["subject"], row["status"]) == ("0000000121234567", "resolved")
    # ISNI publishes no preferred form; the surface comes from ULAN
    assert row["authority_label"] == "Basan"


def test_a_ulan_subject_off_the_bridge_gets_no_isni(tmp_path, ulan):
    assert len(link(tmp_path, ("Doulton and Company", "organisation", "resolved"), vocab="isni")) == 0


def test_the_sidecar_carries_what_compile_reads(tmp_path, ulan):
    linked = link(tmp_path, ("Basan", "person", "resolved"))
    assert linked.columns == al.SIDECAR_COLUMNS
    # compile reads the authority link off these five columns
    assert {"node_id", "record_id", "data_source", "vocab", "subject"} <= set(linked.columns)
    assert linked["span_start"].to_list() == [0]
    assert linked["span_end"].to_list() == [5]
