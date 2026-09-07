from __future__ import annotations

import re

import polars as pl

# Two classes, letter and digit, shared by every reading
MASK = pl.col("value").str.replace_all(r"[A-Za-z]", "s").str.replace_all(r"[0-9]", "d")

# Maximal runs of one class; [\s\S] because values span lines
RUNS = r"d+|s+|[\s\S]"

_RUN_RE = re.compile(f"({RUNS})")


def signature(runs: list[str]) -> str:
    """A masked value's run-length signature: d{4}-d{2} for 1954-03"""
    return "".join(
        f"{r[0]}{{{len(r)}}}" if r[0] in "ds" and len(r) > 1 else (r[0] if r[0] in "ds" else r) for r in runs
    )


def signature_of(masked: str) -> str:
    return signature(_RUN_RE.findall(masked))
