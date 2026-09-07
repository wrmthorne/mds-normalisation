import json

import polars as pl

from mds_norm.pipeline import extraction as ex
from mds_norm.utils.patches import PROMPT_ORDER, RECORD_FIRST, TASK_FIRST, TASK_TEXT_V2, build_request, render_prompt

ROWS = [
    {
        "node_id": b"n0",
        "parent_id": None,
        "depth": 0,
        "field_type": "spectrum/brief_description",
        "value": "A carved oak chest, made around 1850. Height 40cm.",
        "data_source": "Test Museum",
    },
    {
        "node_id": b"n1",
        "parent_id": None,
        "depth": 0,
        "field_type": "spectrum/object_name",
        "value": "chest",
        "data_source": "Test Museum",
    },
]


def request(missing=("date", "material", "dimension"), reloc=(), order=PROMPT_ORDER):
    return build_request(
        "r1", [dict(r) for r in ROWS], list(missing), list(reloc), task_text=TASK_TEXT_V2, order=order
    )


def test_record_first_puts_the_record_before_the_task():
    """The production layout: a record's single-task prompts differ only in their tail"""
    req = request()
    prompts = [p for _, p in ex.units_of(req, "single_task", RECORD_FIRST)]
    record = prompts[0][: prompts[0].index("Extract, from the record above:")]
    assert record.startswith("Record:\n")
    assert all(p.startswith(record) for p in prompts)
    assert all(p.endswith(TASK_TEXT_V2[u]) for (u, p) in ex.units_of(req, "single_task", RECORD_FIRST))


def test_task_first_layout_is_unchanged():
    """What the earlier extraction variants measured stays reachable, byte for byte"""
    req = request(order=TASK_FIRST)
    prompt = render_prompt([TASK_TEXT_V2["date"]], req["record_json"], TASK_FIRST)
    assert prompt.startswith("Tasks:\n" + TASK_TEXT_V2["date"] + "\n\nRecord:\n")
    # the two variants differ in order and nothing else
    other = render_prompt([TASK_TEXT_V2["date"]], req["record_json"], RECORD_FIRST)
    record = json.dumps(req["record_json"], indent=1, ensure_ascii=False)
    assert record in prompt
    assert record in other
    assert TASK_TEXT_V2["date"] in prompt
    assert TASK_TEXT_V2["date"] in other


def test_production_default_is_record_first():
    assert PROMPT_ORDER == RECORD_FIRST
    assert request()["prompt"].startswith("Record:\n")


def test_single_task_sends_one_prompt_per_task():
    units = ex.units_of(request(), "single_task")
    assert [u for u, _ in units] == ["date", "material", "dimension"]
    for unit, prompt in units:
        # each prompt carries its own task bullet and no other
        assert TASK_TEXT_V2[unit] in prompt
        assert sum(t in prompt for t in TASK_TEXT_V2.values()) == 1
        assert "carved oak chest" in prompt


def test_composed_sends_the_record_once():
    units = ex.units_of(request(), "composed")
    assert [u for u, _ in units] == ["composed"]
    assert all(t in units[0][1] for t in TASK_TEXT_V2.values())


def test_relocation_hits_get_their_own_unit():
    hit = {"node_id": b"n1", "field_type": "spectrum/object_name", "candidate": "40cm", "field": "spectrum/dimension"}
    units = ex.units_of(request(missing=["material"], reloc=[hit]), "single_task")
    assert [u for u, _ in units] == ["material", "relocate"]


def test_value_truncation_cuts_a_suffix_so_spans_stay_valid():
    rows = [dict(ROWS[0]) | {"value": "oak " * ex.MAX_VALUE_CHARS}]
    assert ex._truncate(rows) is True
    assert len(rows[0]["value"]) == ex.MAX_VALUE_CHARS
    assert rows[0]["value"].startswith("oak ")


def test_short_values_are_left_alone():
    rows = [dict(r) for r in ROWS]
    assert ex._truncate(rows) is False
    assert rows[0]["value"] == ROWS[0]["value"]


def test_clip_marks_an_overlong_prompt_and_keeps_its_head():
    prompt = "x" * (ex.MAX_PROMPT_CHARS + 500)
    clipped = ex._clip(prompt)
    assert clipped.startswith("x" * ex.MAX_PROMPT_CHARS)
    assert clipped.endswith("(record truncated)")
    assert ex._clip("short") == "short"


def rows_of(*replies):
    """One request per record carrying all three tasks, paired with the units that asked"""
    req = request()
    units = [(req, unit) for unit in req["units"][: len(replies)]]
    return ex.gate(units, list(replies), shard=0)


def reply(content, error=None):
    return {"content": content, "prompt_tokens": 10, "completion_tokens": 5, "error": error}


def test_accepted_op_is_tagged_and_confidence_stamped():
    ops = rows_of(
        reply(
            '{"ops": [{"op": "add", "field": "material",'
            ' "value": "oak", "source_field": "brief_description",'
            ' "rationale": "stated"}]}'
        )
    )
    row = ops.row(0, named=True)
    assert (row["status"], row["task"], row["field"]) == ("resolved", "extract_material", "spectrum/material")
    assert row["confidence"] == ex.LLM_CONFIDENCE
    assert row["value"] == "oak"
    assert row["rationale"] == "stated"
    assert (row["span_start"], row["span_end"]) == (9, 12)
    assert row["unit"] == "date"  # the unit that asked, whatever the op targets


def test_rejected_op_keeps_the_proposal_and_reason():
    ops = rows_of(
        reply('{"ops": [{"op": "add", "field": "material", "value": "walnut", "source_field": "brief_description"}]}')
    )
    row = ops.row(0, named=True)
    assert (row["status"], row["reason"]) == ("rejected", "copy_check_failed")
    assert row["value"] == "walnut"
    assert row["node_id"] is None


def test_unparseable_and_errored_replies_defer_separately():
    ops = rows_of(reply("sorry, no JSON"), reply(None, error="boom"))
    assert ops["status"].to_list() == ["deferred", "deferred"]
    assert ops["reason"].to_list() == ["unparseable_response", "llm_error"]


def test_dedupe_keeps_one_op_but_every_rejection():
    op = (
        '{"ops": [{"op": "add", "field": "material", "value": "oak",'
        ' "source_field": "brief_description"},'
        ' {"op": "add", "field": "material", "value": "walnut",'
        ' "source_field": "brief_description"}]}'
    )
    ops = rows_of(reply(op), reply(op))
    assert ops.filter(pl.col("status") == "resolved").height == 1
    assert ops.filter(pl.col("status") == "rejected").height == 2
    assert ops.schema == pl.Schema(ex.OPS_SCHEMA)


def test_gate_on_no_units_returns_the_empty_frame():
    empty = ex.gate([], [], shard=0)
    assert empty.height == 0
    assert empty.schema == pl.Schema(ex.OPS_SCHEMA)


def date_key(req):
    return ex.content_sha(req["record_json"], TASK_TEXT_V2["date"], "v2", PROMPT_ORDER)


def test_row_order_does_not_change_the_cache_key():
    """`reform` renders in row order, so a key over prompt bytes would miss on an unchanged record"""
    forward = request()
    reversed_rows = build_request(
        "r1", [dict(r) for r in reversed(ROWS)], ["date"], [], task_text=TASK_TEXT_V2, order=PROMPT_ORDER
    )
    # the rendered prompts genuinely differ: json.dumps keeps insertion order
    assert forward["prompt"] != reversed_rows["prompt"]
    assert date_key(forward) == date_key(reversed_rows)


def test_a_changed_value_changes_the_cache_key():
    edited = [dict(ROWS[0], value="A carved elm chest, made around 1850. Height 40cm."), dict(ROWS[1])]
    other = build_request("r1", edited, ["date"], [], task_text=TASK_TEXT_V2, order=PROMPT_ORDER)
    assert date_key(request()) != date_key(other)


def test_two_relocate_tasks_on_one_record_key_apart():
    """They share a record and a unit label, so keying on the unit would reuse one reply for both"""
    hits = [
        {"node_id": b"n1", "field_type": "spectrum/object_name", "candidate": "1850", "field": "spectrum/date"},
        {"node_id": b"n1", "field_type": "spectrum/object_name", "candidate": "40cm", "field": "spectrum/dimension"},
    ]
    req = request(missing=(), reloc=hits)
    triples = ex.units_with_tasks(req, "single_task", PROMPT_ORDER)
    assert [u for u, _, _ in triples] == ["relocate", "relocate"]
    keys = {ex.content_sha(req["record_json"], task, "v2", PROMPT_ORDER) for _, task, _ in triples}
    assert len(keys) == 2
