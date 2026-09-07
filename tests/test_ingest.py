import json

import polars as pl
import pytest
from ingest import ID_BYTES, admin_columns, flatten_batch, jsonl_batches, record_columns

DEEPEST = {"label": "Sex", "type": "spectrum/sex", "value": "deepest"}
DEEP = {"label": "Text", "type": "spectrum/text", "value": "deep", "units": [DEEPEST]}
LATEST = {"label": "Date - Latest", "type": "spectrum/date_latest", "value": "1914", "units": [DEEP]}
LICENCE_URL = {"label": "License Url", "type": "ciim/license_url", "value": "https://example.invalid/by-nc/4.0/"}

UNITS = [
    {"label": "License", "type": "ciim/license", "value": "CC BY-NC", "units": [LICENCE_URL]},
    {"label": "Material", "path": "Mat", "type": "spectrum/material", "value": "Oak"},
    {"label": "Material", "path": "Mat", "type": "spectrum/material", "value": "Iron"},
    {"label": "Material", "path": "Mat", "type": "spectrum/material", "value": "Oak"},
    {"label": "Object Production Date", "type": "spectrum/object_production_date", "units": [LATEST]},
]

DATA_SOURCE = {
    "code": "Q00000000-01",
    "name": "Example Museum / Generic XML 01",
    "organisation": "Example Museum",
    "group": "Q00000000",
}


def record(uuid: str, units: list[dict]) -> dict:
    return {
        "@document": {"type": "ciim/object", "units": units},
        "@admin": {
            "processed": 1700000001000,
            "sequence": 1,
            "uid": "00000000-0000-3000-a000-0000000000ad",
            "added": 1700000000000,
            "stream": "Q00000000-01",
            "id": "1",
            "source": "Q00000000-01",
            "uuid": uuid,
            "data_source": DATA_SOURCE,
        },
    }


RECORDS = [record("00000000-0000-3000-a000-000000000001", UNITS), record("00000000-0000-3000-a000-000000000002", [])]


@pytest.fixture
def export(tmp_path):
    path = tmp_path / "export.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    return path


@pytest.fixture
def ingested(export):
    batch = pl.from_arrow(next(jsonl_batches(export, 100)))
    return flatten_batch(record_columns(batch)), admin_columns(batch)


def rebuild(nodes: pl.DataFrame, record_id: str, parent: bytes | None) -> list[dict]:
    under = pl.col("parent_id").is_null() if parent is None else pl.col("parent_id") == parent
    kids = nodes.filter((pl.col("record_id") == record_id) & under)
    out = []
    for k in sorted(kids.iter_rows(named=True), key=lambda r: r["source_array_pos"]):
        unit = {"label": k["label"], "type": k["field_type"]}
        if k["path"] is not None:
            unit["path"] = k["path"]
        children = rebuild(nodes, record_id, k["node_id"])
        if children:
            unit["units"] = children
        if k["value"] is not None:
            unit["value"] = k["value"]
        out.append(unit)
    return out


def test_units_rebuild_exactly(ingested):
    nodes, _ = ingested
    assert rebuild(nodes, RECORDS[0]["@admin"]["uuid"], None) == UNITS
    assert rebuild(nodes, RECORDS[1]["@admin"]["uuid"], None) == []


def test_admin_keeps_every_field(ingested):
    _, admin = ingested
    row = admin.filter(pl.col("record_id") == RECORDS[0]["@admin"]["uuid"]).to_dicts()[0]
    src = RECORDS[0]["@admin"]
    assert row["uid"] == src["uid"]
    assert row["source_record_id"] == src["id"]
    assert (row["source"], row["stream"]) == (src["source"], src["stream"])
    assert {k: row[k] for k in DATA_SOURCE} == DATA_SOURCE
    assert row["sequence"] == src["sequence"]
    assert row["document_type"] == RECORDS[0]["@document"]["type"]
    assert int(row["added"].timestamp() * 1000) == src["added"]
    assert int(row["processed"].timestamp() * 1000) == src["processed"]


def test_repeated_siblings_keep_their_order(ingested):
    """Three `Material` siblings, two sharing a value, are separable only by position"""
    nodes, _ = ingested
    mats = nodes.filter(pl.col("field_type") == "spectrum/material").sort("source_array_pos")
    assert mats["value"].to_list() == ["Oak", "Iron", "Oak"]
    assert mats["source_array_pos"].to_list() == [1, 2, 3]


def test_position_is_scoped_to_its_parent(ingested):
    """The index restarts inside each `units` array rather than running across the record"""
    nodes, _ = ingested
    assert nodes.filter(pl.col("depth") > 0)["source_array_pos"].unique().to_list() == [0]


def test_rows_stay_grouped_under_their_record(tmp_path):
    """Levels are exploded one at a time; the batch is written a record at a time"""
    path = tmp_path / "two.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (record("aaa", UNITS), record("bbb", UNITS))) + "\n")
    nodes = flatten_batch(record_columns(pl.from_arrow(next(jsonl_batches(path, 100)))))
    assert nodes["record_id"].to_list() == ["aaa"] * 9 + ["bbb"] * 9
    assert nodes.filter(pl.col("record_id") == "aaa")["depth"].to_list() == [0, 0, 0, 0, 0, 1, 1, 2, 3]


def test_deepest_level_is_reached(ingested):
    nodes, _ = ingested
    assert nodes.filter(pl.col("depth") == 3)["value"].to_list() == ["deepest"]


def test_node_ids_are_random_and_narrow(ingested):
    nodes, _ = ingested
    assert nodes["node_id"].bin.size().unique().to_list() == [ID_BYTES]
    assert nodes["node_id"].n_unique() == nodes.height
    assert nodes["node_id"].to_list() != sorted(nodes["node_id"].to_list())


def test_parents_link_to_real_nodes(ingested):
    nodes, _ = ingested
    parents = nodes["parent_id"].drop_nulls()
    assert parents.is_in(nodes["node_id"].implode()).all()
    assert nodes.filter(pl.col("depth") == 0)["parent_id"].is_null().all()


def ingest_one(tmp_path, rec: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    batch = pl.from_arrow(next(jsonl_batches(path, 100)))
    return flatten_batch(record_columns(batch)), admin_columns(batch)


def test_a_re_emitted_record_keeps_only_its_newest_copy(tmp_path):
    """A record catalogued mid-download arrives twice; the later copy is its current state"""
    uuid = "00000000-0000-3000-a000-000000000001"
    first, second = record(uuid, UNITS[:2]), record(uuid, UNITS)
    second["@admin"]["sequence"] = first["@admin"]["sequence"] + 1
    path = tmp_path / "resent.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (first, record("other", UNITS[:1]), second)) + "\n")

    batch = pl.from_arrow(next(jsonl_batches(path, 100)))
    nodes, admin = flatten_batch(record_columns(batch)), admin_columns(batch)
    assert admin["record_id"].to_list() == ["other", uuid]
    assert admin.filter(pl.col("record_id") == uuid)["sequence"].item() == second["@admin"]["sequence"]
    assert rebuild(nodes, uuid, None) == UNITS


def test_unknown_record_keys_are_kept(tmp_path):
    """A key the schema does not model is carried as JSON rather than dropped"""
    rec = record("00000000-0000-3000-a000-000000000001", UNITS)
    rec["@admin"]["deleted"] = True
    rec["@admin"]["data_source"] = DATA_SOURCE | {"region": "West Midlands"}
    rec["@document"]["schema"] = "v2"
    rec["@links"] = {"self": "https://example.invalid/1"}
    _, admin = ingest_one(tmp_path, rec)
    assert json.loads(admin["extra"][0]) == {
        "@links": {"self": "https://example.invalid/1"},
        "@admin.deleted": True,
        "@admin.data_source.region": "West Midlands",
        "@document.schema": "v2",
    }


def test_unknown_unit_keys_are_kept(tmp_path):
    rec = record("00000000-0000-3000-a000-000000000001", [UNITS[1] | {"confidence": 0.9}])
    nodes, _ = ingest_one(tmp_path, rec)
    assert json.loads(nodes["extra"][0]) == {"confidence": 0.9}
    assert nodes["value"].to_list() == ["Oak"]


def test_nesting_past_the_modelled_depth_is_kept(tmp_path):
    """The schema holds four levels; a fifth is kept verbatim on the deepest node"""
    fifth = {"label": "Sex", "type": "spectrum/sex", "value": "too deep"}
    deepest = DEEPEST | {"units": [fifth]}
    rec = record(
        "00000000-0000-3000-a000-000000000001",
        [{**UNITS[4], "units": [{**LATEST, "units": [{**DEEP, "units": [deepest]}]}]}],
    )
    nodes, _ = ingest_one(tmp_path, rec)
    assert nodes.height == 4
    assert json.loads(nodes.filter(pl.col("depth") == 3)["extra"][0]) == {"units": [fifth]}


def test_extra_is_null_when_nothing_spills(ingested):
    nodes, admin = ingested
    assert nodes["extra"].null_count() == nodes.height
    assert admin["extra"].null_count() == admin.height
