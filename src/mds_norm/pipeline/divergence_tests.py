from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import false_discovery_control

from mds_norm.paths import PATTERNS_OUT
from mds_norm.pipeline.pattern_exports import FAMILIES

OUT_PATH = PATTERNS_OUT / "slot_divergence.parquet"

# Fewer values than this on a side says nothing
MIN_N = 30

# Each side is truncated; the pooled corpus needs more
CELL_CAP, CORPUS_CAP = 10_000, 100_000

# A chi-square needs two categories to compare
MIN_TERMS = 2

# Benjamini-Hochberg false-positive share among significant cells
FDR = 0.05


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cells(family: str) -> pl.DataFrame:
    return pl.read_parquet(PATTERNS_OUT / f"{family}_institutional_slot_values.parquet").with_columns(
        family=pl.lit(family)
    )


def corpus_slots(cells: pl.DataFrame) -> dict[tuple[str, int], list[str]]:
    """Every institution's values pooled per (merged pattern, slot), which is what a cell is compared against"""
    pooled = cells.group_by("merged_pattern", "slot_idx").agg(values=pl.col("values").list.explode().head(CORPUS_CAP))
    return {(r["merged_pattern"], r["slot_idx"]): r["values"] for r in pooled.iter_rows(named=True)}


def digit_test(cell: list[str], corpus: list[str]) -> tuple[float, float] | None:
    """Two-sample KS: does the cell's numbers differ from the corpus's, and by how far"""
    a = np.array([int(v) for v in cell if v.isdigit()], dtype=np.int64)
    b = np.array([int(v) for v in corpus if v.isdigit()], dtype=np.int64)
    if len(a) < MIN_N or len(b) < MIN_N:
        return None
    result = stats.ks_2samp(a, b)
    return float(result.statistic), float(result.pvalue)


def letter_test(cell: list[str], corpus: list[str]) -> tuple[float, float] | None:
    """Chi-square on term counts, with Cramer's V as the effect size"""
    if len(cell) < MIN_N or len(corpus) < MIN_N:
        return None
    a, b = Counter(cell), Counter(corpus)
    terms = sorted(a.keys() | b.keys())
    if len(terms) < MIN_TERMS:
        return None
    table = np.array([[a[t] for t in terms], [b[t] for t in terms]], dtype=np.int64)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < MIN_TERMS:
        return None
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.sum()
    return float(np.sqrt(chi2 / n)), float(p)


def test_cells(family: str) -> pl.DataFrame:
    """Every cell of one family tested against the corpus distribution for its own pattern and slot"""
    frame = cells(family)
    corpus = corpus_slots(frame)
    rows = []
    for r in frame.iter_rows(named=True):
        pooled = corpus[(r["merged_pattern"], r["slot_idx"])]
        values = r["values"][:CELL_CAP]
        # Pooling keeps the comparison institution-against-corpus, not against everyone else
        test = digit_test(values, pooled) if r["slot_kind"] == "d" else letter_test(values, pooled)
        if test is None:
            continue
        effect, p = test
        rows.append(
            {
                "family": family,
                "data_source": r["data_source"],
                "merged_pattern": r["merged_pattern"],
                "slot_idx": r["slot_idx"],
                "slot_kind": r["slot_kind"],
                "test": "ks" if r["slot_kind"] == "d" else "chi2",
                "effect": effect,
                "p": p,
                "n_cell": len(values),
                "n_corpus": len(pooled),
            }
        )
    return pl.DataFrame(rows)


def correct(tested: pl.DataFrame) -> pl.DataFrame:
    """Benjamini-Hochberg over every cell tested, then rank by effect size"""
    if tested.is_empty():
        return tested
    q = false_discovery_control(tested["p"].to_numpy(), method="bh")
    return (
        tested.with_columns(q=pl.Series(q))
        .with_columns(significant=pl.col("q") <= FDR)
        .sort("effect", descending=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="How far each institution's slot values diverge from the corpus.")
    ap.add_argument("--family", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES))
    args = ap.parse_args()

    tested = []
    for family in args.family:
        frame = test_cells(family)
        log(f"{family}: {frame.height:,} cells tested")
        tested.append(frame)

    out = correct(pl.concat(tested))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH)
    log(f"{out.height:,} cells, {out['significant'].sum():,} significant at {FDR} → {OUT_PATH}")
    with pl.Config(tbl_rows=15, fmt_str_lengths=40):
        print(out.head(15).select("family", "data_source", "merged_pattern", "slot_idx", "test", "effect", "n_cell"))


if __name__ == "__main__":
    main()
