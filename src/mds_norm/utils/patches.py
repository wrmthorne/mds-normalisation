from __future__ import annotations

import json
import re

import polars as pl
from mds_data_model.introspection import all_free_text_fields, date_fields, measurement_fields, monetary_fields

from mds_norm.parsers.parse_dates import parse_date
from mds_norm.parsers.parse_dimensions import parse_dimensions
from mds_norm.parsers.parse_monetary import parse_monetary

PROD_DATE = "spectrum/object_production_date"

DATE_DEST_BY_SOURCE = {
    "spectrum/acquisition_note": "spectrum/acquisition_date",
    "spectrum/field_collection_note": "spectrum/field_collection_date",
}
PRICE_DEST = "spectrum/object_purchase_price"
MONETARY_FIELDS = set(monetary_fields())
MEASUREMENT_GROUPS = list(measurement_fields())
# the whole-cell guard covers every free-text field, not just Object's
FREE_TEXT = {n for names in all_free_text_fields().values() for n in names}

# a source of the target's own kind is refused
SAME_KIND = (
    dict.fromkeys(date_fields(), "date")
    | dict.fromkeys(measurement_fields(), "measurement")
    | dict.fromkeys(monetary_fields(), "monetary")
)

DEST = (
    pl.when(pl.col("group") == "__date__")
    .then(pl.col("field_type").replace_strict(DATE_DEST_BY_SOURCE, default=PROD_DATE))
    .when(pl.col("group") == "__price__")
    .then(pl.lit(PRICE_DEST))
    .otherwise(pl.col("group"))  # measurement groups are the field type
)

MATERIAL_MAX_WORDS = 6


def verify_value(field: str, value: str) -> str | None:
    """Tier-1 verification of a proposed value. Returns a reject reason or None"""
    if field == PROD_DATE or field.endswith("_date"):
        return None if parse_date(value) else "date_parse_failed"
    if field in MEASUREMENT_GROUPS:
        parsed = parse_dimensions(value)
        return None if parsed and parsed["measurements"] else "dimension_parse_failed"
    if field == "spectrum/material":
        if re.search(r"\d", value):
            return "material_has_digits"
        if len(value.split()) > MATERIAL_MAX_WORDS:
            return "material_too_long"
        return None
    if field in MONETARY_FIELDS:
        # the £-s-d grammar is the real gate
        return None if parse_monetary(value) else "price_parse_failed"
    return "unknown_target_field"


# FIXME: Why limiting to just production date? Why also not allowing for extraction of people, locations, orgs, etc.
TARGET_PRESENCE = {
    "date": [PROD_DATE, "spectrum/date_earliest_single", "spectrum/date_latest"],
    "material": ["spectrum/material"],
    "dimension": ["spectrum/dimension", "spectrum/dimension_value"],
}
MIN_TEXT_CHARS = 200


# TODO: give the model the data model's fields and descriptions, and let it extract optimistically
SYSTEM = (
    "You extract catalogue data from UK museum object records. You only locate and copy text that is already "
    "in the record; you never compose, rephrase, or normalise values.\n"
    "\n"
    "Respond with a single JSON object:\n"
    '{"ops": [{"op": "add"|"move", "field": "<target field>", "value": "<verbatim contiguous substring>", '
    '"source_field": "<record key the value was copied from>", "rationale": "<max 10 words>"}]}\n'
    "\n"
    "Rules:\n"
    "- value must be copied character-for-character from the text of source_field.\n"
    "- Only use target fields named in the tasks.\n"
    "- If a task finds nothing, emit no op for it; an empty ops list is valid.\n"
    "- Output only the JSON object, no markdown."
)

TASK_TEXT = {
    "date": "- add object_production_date: the date the object itself was made or produced,"
    " if stated (not acquisition, donation, or association dates).",
    "material": "- add material: each material the object is made of, one op per material"
    " (a material is at most a few words).",
    "dimension": "- add dimension: the object's stated measurements, copied as written"
    " including units (one op per measurement expression).",
}
FIELD_FOR_TASK = {"date": PROD_DATE, "material": "spectrum/material", "dimension": "spectrum/dimension"}
TASK_OF_FIELD = {v: f"extract_{k}" for k, v in FIELD_FOR_TASK.items()}

# The variant chosen for production; v1 above is unchanged

SYSTEM_V2 = (
    "You extract catalogue data from UK museum object records. You only locate and copy text that is already "
    "in the record; you never compose, rephrase, or normalise values.\n"
    "\n"
    "Respond with a single JSON object:\n"
    '{"ops": [{"op": "add"|"move", "field": "<target field>", "value": "<verbatim contiguous substring>", '
    '"source_field": "<record key the value was copied from>", "rationale": "<max 10 words>"}]}\n'
    "\n"
    "Rules:\n"
    "- value must be copied character-for-character from the text of source_field.\n"
    "- Only use target fields named in the tasks.\n"
    "- Extract only facts stated about the catalogued object itself. Record text often describes other things:"
    " general background essays, the class of objects this one belongs to, an original that this object "
    "reproduces (the brass beneath a rubbing, the map on a modern CD), the subject it depicts, or an "
    "album/container it is kept in. Facts about those are not facts about this object — emit nothing for them.\n"
    "- Many records state nothing for a task; an empty ops list is a correct and common answer. Never lower "
    "the bar to produce an op.\n"
    "- Output only the JSON object, no markdown."
)

TASK_TEXT_V2 = {
    "date": "- add object_production_date: the date this object itself was made or"
    " produced, if stated (not acquisition, donation, or association dates;"
    " not a date the object commemorates, depicts, or carries in its"
    " content; not the date of an original it reproduces). Keep an"
    " uncertainty qualifier with its date: copy 'about 1955' or"
    " '1843 (circa.)', not the bare year.",
    "material": "- add material: each material this object is made of, one op per"
    " material. Copy the bare material name, dropping colour, pattern"
    " and form words ('pale quilted silk' -> 'silk', 'bronze coloured"
    " metal' -> 'metal', 'black enamel' -> 'enamel'); keep a compound"
    " name whole only when the compound is itself the material"
    " ('machine lace', 'silk satin', 'cotton canvas')."
    " Techniques and processes (woodcut, stitch names), implements,"
    " colours, places, names of the object or its parts (shell, CD,"
    " newspaper, band, boning, border), and contents are not"
    " materials unless the record states what they are made of.",
    "dimension": "- add dimension: this object's stated measurements, copied as"
    " written including units (one op per measurement expression)."
    " Where the record states the same measurement twice, copy the"
    " unit-bearing form. A scale ratio (1:24) or a size designation"
    " is not a measurement.",
}

CONTRACTS = {"v1": (SYSTEM, TASK_TEXT), "v2": (SYSTEM_V2, TASK_TEXT_V2)}


def strip_prefix(ft: str) -> str:
    return ft.split("/", 1)[-1]


def _append(d: dict, key: str, val: object) -> None:
    if key in d:
        d[key] = d[key] + [val] if isinstance(d[key], list) else [d[key], val]
    else:
        d[key] = val


def reform(rows: list[dict]) -> dict:
    """Flat rows for one record -> nested dict, repeated fields as lists"""
    kids: dict = {}
    for r in rows:
        if r["parent_id"] is not None:
            kids.setdefault(r["parent_id"], []).append(r)

    def build(r: dict) -> dict | str | None:
        ch = kids.get(r["node_id"], [])
        if not ch:
            return r["value"]
        obj = {"value": r["value"]} if r["value"] is not None else {}
        for c in ch:
            if (cv := build(c)) is not None:
                _append(obj, strip_prefix(c["field_type"]), cv)
        return obj or None

    out: dict = {}
    for r in rows:
        if r["depth"] == 0 and (v := build(r)) is not None:
            _append(out, strip_prefix(r["field_type"]), v)
    return out


# Two layouts; the variants measured task_first, production uses record_first

TASK_FIRST, RECORD_FIRST = "task_first", "record_first"
PROMPT_ORDER = RECORD_FIRST


def render_prompt(tasks: list[str], record_json: dict, order: str = TASK_FIRST) -> str:
    record = json.dumps(record_json, indent=1, ensure_ascii=False)
    if order == RECORD_FIRST:
        return "Record:\n" + record + "\n\nExtract, from the record above:\n" + "\n".join(tasks)
    return "Tasks:\n" + "\n".join(tasks) + "\n\nRecord:\n" + record


def build_request(
    record_id: str,
    rows: list[dict],
    missing: list[str] = (),
    reloc: list[dict] = (),
    task_text: dict[str, str] = TASK_TEXT,
    order: str = TASK_FIRST,
) -> dict | None:
    """Assemble one extraction request from a record's flat rows, None when there is nothing to ask"""
    nodes_by_field: dict[str, list] = {}
    for r in rows:
        if r["value"] is not None:
            nodes_by_field.setdefault(strip_prefix(r["field_type"]), []).append(
                (r["node_id"], r["field_type"], r["value"])
            )
    tasks, units, allowed = [], [], {}

    for k in missing:
        tasks.append(task_text[k])
        units.append(k)
        allowed[strip_prefix(FIELD_FOR_TASK[k])] = FIELD_FOR_TASK[k]
    for hit in reloc:
        src = strip_prefix(hit["field_type"])
        tasks.append(
            f"- move: the {src} field contains misplaced structured content:"
            f' "{hit["candidate"]}". If it does not belong in {src}, emit a move op'
            f" with the misplaced portion to {strip_prefix(hit['field'])}."
        )
        units.append("relocate")
        allowed[strip_prefix(hit["field"])] = hit["field"]
    if not tasks:
        return None

    record_json = {"data_source": rows[0]["data_source"], **reform(rows)}
    return {
        "record_id": record_id,
        "data_source": rows[0]["data_source"],
        "prompt": render_prompt(tasks, record_json, order),
        "tasks": tasks,
        "units": units,
        "record_json": record_json,
        "allowed": allowed,
        "nodes": nodes_by_field,
    }


_JSON = re.compile(r"{.*}", re.DOTALL)
LLM_CONFIDENCE = 0.7


def parse_ops(content: str | None) -> list | None:
    if content is None or (m := _JSON.search(content)) is None:
        return None
    try:
        ops = json.loads(m.group()).get("ops")
    except json.JSONDecodeError:
        return None
    return ops if isinstance(ops, list) else None


def _source_key(src: str) -> str:
    """Normalise a source_field reference ("text[0].value", "spectrum/x") to a key"""
    parts = [re.sub(r"\[\d+]$", "", p) for p in re.split(r"[./]", src)]
    parts = [p for p in parts if p and p not in ("value", "spectrum")]
    return parts[-1] if parts else "value"


def validate_op(req: dict, op: dict) -> tuple[dict | None, str | None]:
    if not isinstance(op, dict):
        return None, "not_an_object"
    kind, field, value, src = (op.get(k) for k in ("op", "field", "value", "source_field"))
    if kind not in ("add", "move"):
        return None, "bad_op"
    if not isinstance(field, str) or (full_field := req["allowed"].get(_source_key(field))) is None:
        return None, "field_not_admitted"
    if not isinstance(value, str) or not value.strip():
        return None, "empty_value"
    candidates = req["nodes"].get(_source_key(src)) if isinstance(src, str) else None
    if not candidates:
        # named key missing: fall back to a whole-record search
        candidates = [n for lst in req["nodes"].values() for n in lst]
    for candidate in candidates:
        if (start := candidate[2].find(value)) >= 0:
            break
    else:
        for candidate in candidates:
            if (start := candidate[2].lower().find(value.lower())) >= 0:
                value = candidate[2][start : start + len(value)]  # recover the source-exact span
                break
        else:
            return None, "copy_check_failed"
    node_id, source_field, source = candidate
    if (source_kind := SAME_KIND.get(source_field)) is not None and source_kind == SAME_KIND.get(full_field):
        return None, "same_kind_source"
    if source_field in FREE_TEXT and len(value) == len(source):
        return None, "whole_cell_extraction"
    if (reason := verify_value(full_field, value)) is not None:
        return None, reason
    return {
        "node_id": node_id,
        "source_field": source_field,
        "op": kind,
        "field": full_field,
        "value": value,
        "span_start": start,
        "span_end": start + len(value),
    }, None
