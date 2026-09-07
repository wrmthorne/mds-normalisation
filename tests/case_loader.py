import json
from pathlib import Path

import pytest

CASES_DIR = Path(__file__).parent / "cases"

_CASE_KEYS = {"raw", "expected", "note"}


def _record(expected, fields, defaults, raw):
    unknown = sorted(set(expected) - set(fields))
    if unknown:
        raise ValueError(f"case {raw!r}: unknown field(s) {unknown}, expected some of {fields}")
    return tuple(expected.get(f, defaults.get(f)) for f in fields)


def _expected(expected, fields, defaults, raw):
    if expected is None:
        return None
    if isinstance(expected, list):
        return [_record(e, fields, defaults, raw) for e in expected]
    return _record(expected, fields, defaults, raw)


def load_cases(name):
    """Parametrisation for tests/cases/<name>.json, as (raw, expected) params"""
    data = json.loads((CASES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    fields, defaults = data["fields"], data.get("defaults", {})
    groups = data.get("groups") or [{"cases": data["cases"]}]

    params = []
    for group in groups:
        for case in group["cases"]:
            raw = case["raw"]
            unknown = sorted(set(case) - _CASE_KEYS)
            if unknown:
                raise ValueError(f"case {raw!r}: unknown key(s) {unknown}")
            params.append(pytest.param(raw, _expected(case["expected"], fields, defaults, raw), id=raw or "<empty>"))
    return params
