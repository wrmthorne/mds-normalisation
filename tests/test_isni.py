import gzip

import polars as pl
import pytest

from mds_norm.pipeline import vocab_indexes as vi

BLOCK = """<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" \
xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:schema="http://schema.org/" \
xmlns:madsrdf="http://www.loc.gov/mads/rdf/v1#" xmlns:owl="http://www.w3.org/2002/07/owl#" \
xmlns:foaf="http://xmlns.com/foaf/0.1/">{body}</rdf:RDF>"""

LIVE = """<rdf:Description rdf:about="https://isni.org/isni/0000000000000095">\
<rdf:type rdf:resource="http://schema.org/Person"/><rdfs:label>ISNI 0000 0000 0000 0095</rdfs:label>\
<schema:alternateName>Léon Gernez </schema:alternateName><schema:alternateName>Gernez, Léon</schema:alternateName>\
<schema:birthDate>1875</schema:birthDate><schema:deathDate>1937</schema:deathDate>\
<owl:sameAs rdf:resource="http://www.wikidata.org/entity/Q3270990"/>\
<madsrdf:isIdentifiedByAuthority rdf:resource="http://data.bnf.fr/ark:/12148/cb10199651x"/></rdf:Description>\
<rdf:Description rdf:about="https://isni.org/isni/0000000000000095/about">\
<foaf:primaryTopic rdf:resource="https://isni.org/isni/0000000000000095"/></rdf:Description>"""

DEPRECATED = """<rdf:Description rdf:about="https://isni.org/isni/0000000000000108">\
<owl:deprecated>true</owl:deprecated><rdfs:comment>Deprecated ISNI</rdfs:comment>\
<rdfs:seeAlso rdf:resource="https://isni.org/isni/0000000000000095"/></rdf:Description>"""


def write_export(tmp_path, *bodies, separators=True):
    """The export as ISNI publishes it: a 0x1E between every pair of elements, gzipped"""
    blocks = "\n".join(BLOCK.format(body=b) for b in bodies)
    xml = f'<?xml version="1.0" encoding="UTF-8" ?>\n<catalog>\n{blocks}\n</catalog>\n'
    if separators:
        xml = xml.replace("><", ">\x1e<")
    path = tmp_path / "ISNI_persons.rdf.gz"
    path.write_bytes(gzip.compress(xml.encode()))
    return path


def records(tmp_path, *bodies, **kwargs):
    out = tmp_path / "records.parquet"
    vi.sink_isni(write_export(tmp_path, *bodies, **kwargs), out)
    return pl.read_parquet(out)


def test_reads_names_dates_and_links(tmp_path):
    row = records(tmp_path, LIVE).row(0, named=True)
    assert row["isni"] == "0000000000000095"
    assert row["names"] == ["Léon Gernez ", "Gernez, Léon"]
    assert (row["begin_date"], row["end_date"]) == ("1875", "1937")
    assert row["same_as"] == ["http://www.wikidata.org/entity/Q3270990"]
    assert row["authorities"] == ["http://data.bnf.fr/ark:/12148/cb10199651x"]
    assert row["replaced_by"] is None


def test_deprecated_record_keeps_only_its_replacement(tmp_path):
    frame = records(tmp_path, LIVE, DEPRECATED)
    assert len(frame) == 2
    dead = frame.filter(pl.col("replaced_by").is_not_null()).row(0, named=True)
    assert dead["isni"] == "0000000000000108"
    assert dead["replaced_by"] == "0000000000000095"
    assert dead["names"] == []


def test_parses_without_the_separators(tmp_path):
    assert len(records(tmp_path, LIVE, separators=False)) == 1


def test_separator_inside_a_text_node_is_refused(tmp_path):
    body = LIVE.replace("Gernez, Léon", "Gernez,\x1e Léon")
    with pytest.raises(ValueError, match="text node"):
        records(tmp_path, body)


def test_separator_check_spans_chunk_boundaries(tmp_path):
    with gzip.open(write_export(tmp_path, LIVE), "rb") as raw:
        source = vi.StripSeparators(raw)
        assert b"\x1e" not in b"".join(iter(lambda: source.read(64), b""))


def test_index_is_term_index_shaped_over_live_records(tmp_path, monkeypatch):
    monkeypatch.setattr(vi, "VOCAB_INDEXES", tmp_path)
    monkeypatch.setattr(vi, "VOCABS", tmp_path)
    (tmp_path / "isni").mkdir()
    export = write_export(tmp_path, LIVE, DEPRECATED)
    export.rename(tmp_path / "isni" / vi.ISNI_EXPORTS["person"])

    index = vi.isni_index("person").collect()
    assert index.columns == ["subject", "term", "lang", "kind", "norm"]
    assert index["subject"].unique().to_list() == ["0000000000000095"]
    assert set(index["norm"]) == {"léon gernez", "gernez, léon"}
    assert set(index["kind"]) == {"altLabel"}
