from __future__ import annotations

import argparse
import time
from itertools import combinations

import polars as pl

from mds_norm.paths import VOCAB_INSTITUTIONAL
from mds_norm.pipeline.institutional_vocab_detect import FIELD_CANDIDATES, atom_universe

OUT_PATH = VOCAB_INSTITUTIONAL / "fragmentation.parquet"
CONFLATION_PATH = VOCAB_INSTITUTIONAL / "conflation.parquet"

# The working vocabulary covers this share of occurrences
COVERAGE = 0.80

# Below this many occurrences the field says nothing
MIN_OCC = 200


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fragmentation(atoms: pl.DataFrame, field: str) -> pl.DataFrame:
    """Per institution: how many ways it writes this field's concepts, and how few of them carry most of it"""
    ranked = (
        atoms.sort("occ", descending=True)
        .with_columns(cum=pl.col("occ").cum_sum().over("data_source"), total=pl.col("occ").sum().over("data_source"))
        .with_columns(cum_share=pl.col("cum") / pl.col("total"))
    )
    return (
        ranked.group_by("data_source")
        .agg(
            n_terms=pl.len(),
            n_occurrences=pl.col("occ").sum(),
            singleton_rate=(pl.col("occ") == 1).mean(),
            # k_80: terms, largest first, covering four fifths
            k_80=(pl.col("cum_share") < COVERAGE).sum() + 1,
        )
        .filter(pl.col("n_occurrences") >= MIN_OCC)
        .with_columns(
            # A raw distinct-to-total ratio falls as a field grows
            fragmentation=pl.col("n_terms").log() / pl.col("n_occurrences").log(),
            raw_ratio=pl.col("n_terms") / pl.col("n_occurrences"),
            k80_share=pl.col("k_80") / pl.col("n_terms"),
            field=pl.lit(field),
        )
        .sort("fragmentation", descending=True)
    )


def conflation(universes: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One concept spread across two fields: the share of the smaller vocabulary the larger also holds"""
    rows = []
    for a, b in combinations(sorted(universes), 2):
        left = universes[a].group_by("data_source").agg(terms=pl.col("norm"))
        right = universes[b].group_by("data_source").agg(terms=pl.col("norm"))
        both = left.join(right, on="data_source", suffix="_b")
        for r in both.iter_rows(named=True):
            va, vb = set(r["terms"]), set(r["terms_b"])
            if min(len(va), len(vb)) == 0:
                continue
            rows.append(
                {
                    "data_source": r["data_source"],
                    "field_a": a,
                    "field_b": b,
                    "n_a": len(va),
                    "n_b": len(vb),
                    "shared": len(va & vb),
                    "overlap": len(va & vb) / min(len(va), len(vb)),
                }
            )
    return pl.DataFrame(rows).sort("overlap", descending=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="How many ways each institution writes a controlled field's terms.")
    ap.add_argument("--field", nargs="+", default=sorted(FIELD_CANDIDATES), choices=sorted(FIELD_CANDIDATES))
    args = ap.parse_args()

    universes = {field: atom_universe(field) for field in args.field}
    frames = [fragmentation(atoms, field) for field, atoms in universes.items()]
    out = pl.concat(frames)
    VOCAB_INSTITUTIONAL.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH)
    log(f"{out.height} (institution, field) pairs → {OUT_PATH}")
    print(
        out.group_by("field").agg(
            institutions=pl.len(),
            fragmentation=pl.col("fragmentation").median(),
            k_80=pl.col("k_80").median(),
            k80_share=pl.col("k80_share").median(),
            singleton_rate=pl.col("singleton_rate").median(),
        )
    )

    if len(universes) > 1:
        pairs = conflation(universes)
        pairs.write_parquet(CONFLATION_PATH)
        log(f"{pairs.height} field pairs per institution → {CONFLATION_PATH}")
        print(
            pairs.group_by("field_a", "field_b")
            .agg(institutions=pl.len(), overlap=pl.col("overlap").median())
            .sort("overlap", descending=True)
        )


if __name__ == "__main__":
    main()
