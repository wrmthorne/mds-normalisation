import polars as pl
import pytest
from ingest import NODE_SCHEMA

from mds_norm import tables
from mds_norm.pipeline import compile_records as c

NODES = [
    ("r1", "Coventry", b"n0", None, 0, 1, "Material", "Mat", "spectrum/material", "Oak", None),
    ("r1", "Coventry", b"n1", None, 0, 0, "Title", None, "spectrum/title", "Humber", '{"confidence": 0.9}'),
    ("r1", "Coventry", b"n2", b"n1", 1, 0, "Text", None, "spectrum/text", "deep", None),
    ("r2", "Coventry", b"n3", None, 0, 0, "Title", None, "spectrum/title", "Olympia", None),
]


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "nodes.parquet"
    pl.DataFrame(NODES, schema=NODE_SCHEMA, orient="row").write_parquet(path)
    return path


def test_generated_nodes_carry_the_source_only_columns_empty():
    part = pl.LazyFrame({"node_id": [b"g0"], "field_type": ["wrmthorne/kind"], "value": ["circa"]})
    out = tables.source_only(part).collect()
    assert out["source_array_pos"].to_list() == [None]
    assert out["extra"].to_list() == [None]
    assert out.schema["source_array_pos"] == pl.UInt16


def test_siblings_are_ordered_by_their_position_in_the_export(table):
    """Row order put Material first; its position in the source `units` array puts it second"""
    records = tables.record_nodes(["r1", "r2"], table)
    assert [n["value"] for n in records["r1"]["nodes"]] == ["Humber", "Oak", "deep"]
    assert records["r2"]["data_source"] == "Coventry"


def test_the_release_is_the_node_table_plus_the_compile_columns():
    assert list(NODE_SCHEMA) == c.BASE_COLS
    assert c.OUT_COLS[len(c.BASE_COLS) :] == ["as_recorded", "component"]
