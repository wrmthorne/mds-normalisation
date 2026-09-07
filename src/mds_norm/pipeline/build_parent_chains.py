from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from mds_norm.paths import VOCAB_INDEXES, VOCABS

VOCAB_PATH = VOCABS
INDEX_DIR = VOCAB_INDEXES

KIND_PRIORITY = {"prefLabelGVP": 0, "prefLabel": 1, "altLabel": 2}
BROADER = "<http://vocab.getty.edu/ontology#broaderPreferred>"
LEVELS = 3  # two or three levels is enough to disambiguate


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def best_labels(index: pl.LazyFrame) -> pl.DataFrame:
    """One display label per subject: prefLabelGVP > prefLabel > altLabel, English first"""
    return (
        index.with_columns(
            kind_p=pl.col("kind").replace_strict(KIND_PRIORITY, return_dtype=pl.Int8),
            lang_p=(~pl.col("lang").str.starts_with("en")).cast(pl.Int8).fill_null(1),
        )
        .sort("kind_p", "lang_p", "term")
        .unique("subject", keep="first")
        .select("subject", label=pl.col("term"))
        .collect(engine="streaming")
    )


def broader_preferred(nt_path: Path, voc: str) -> pl.DataFrame:
    """Direct <voc/child> broaderPreferred <voc/parent> pairs"""
    pat = (
        rf"^<http://vocab\.getty\.edu/{voc}/(\d+)> "
        + r"<http://vocab\.getty\.edu/ontology#broaderPreferred> "
        + rf"<http://vocab\.getty\.edu/{voc}/(\d+)> \.$"
    )
    return (
        pl.scan_csv(nt_path, separator="\x00", has_header=False, new_columns=["line"], quote_char=None)
        .filter(pl.col("line").str.contains("#broaderPreferred>", literal=True))
        .select(subject=pl.col("line").str.extract(pat, 1), parent=pl.col("line").str.extract(pat, 2))
        .drop_nulls()
        .unique("subject")  # broaderPreferred is single-valued; belt and braces
        .collect(engine="streaming")
    )


def parent_chains(voc: str, rels_nt: Path, index: pl.LazyFrame) -> pl.DataFrame:
    log(f"{voc}: scanning {rels_nt.name}")
    up = broader_preferred(rels_nt, voc)
    log(f"{voc}: {up.height:,} broaderPreferred pairs")
    labels = best_labels(index)
    log(f"{voc}: {labels.height:,} subject labels")

    chain = up
    hops = []
    for lvl in range(LEVELS):
        hops.append(
            chain.join(labels, left_on="parent", right_on="subject", how="left").select(
                "subject", pl.col("label").alias(f"p{lvl}")
            )
        )
        chain = chain.join(up, left_on="parent", right_on="subject", how="inner").select(
            "subject", parent=pl.col("parent_right")
        )
        if not chain.height:
            break

    merged = hops[0]
    for h in hops[1:]:
        merged = merged.join(h, on="subject", how="left")
    cols = [c for c in merged.columns if c.startswith("p")]
    return merged.select(
        "subject", parents=pl.concat_str([pl.col(c) for c in cols], separator=" ← ", ignore_nulls=True)
    )


def _year(bound: dict | None, keys: tuple[str, ...]) -> int | None:
    if not isinstance(bound, dict):
        return None
    inside = bound.get("in")
    if isinstance(inside, dict):
        for k in keys:
            v = inside.get(k)
            if v not in (None, ""):
                try:
                    return int(v)
                except ValueError:
                    return None
    return None


def _authority_title(authority: dict) -> str | None:
    src = authority.get("source", {})
    return (
        src.get("title")
        or (src.get("partOf") or {}).get("title")
        or " & ".join(c.get("name", "") for c in src.get("creators", []))
        or None
    )


def periodo_bounds() -> pl.DataFrame:
    data = json.loads((VOCAB_PATH / "periodo/periodo-dataset.json").read_text())
    rows = []
    for authority in data["authorities"].values():
        title = _authority_title(authority)
        for pid, period in authority.get("periods", {}).items():
            rows.append(
                (
                    pid,
                    _year(period.get("start"), ("year", "earliestYear", "latestYear")),
                    _year(period.get("stop"), ("year", "latestYear", "earliestYear")),
                    title,
                )
            )
    # Int64: PeriodO covers geological time (start years around -2.45e9)
    return pl.DataFrame(
        rows,
        schema={"subject": pl.String, "start_year": pl.Int64, "stop_year": pl.Int64, "authority": pl.String},
        orient="row",
    )


def main() -> None:
    for voc in ("aat", "tgn"):
        out = INDEX_DIR / f"{voc}_parents.parquet"
        rels = VOCAB_PATH / f"{voc}/{voc.upper()}Out_HierarchicalRels.nt"
        chains = parent_chains(voc, rels, pl.scan_parquet(INDEX_DIR / f"{voc}.parquet"))
        chains.write_parquet(out)
        log(f"{voc}: {chains.height:,} parent chains → {out}")
    bounds = periodo_bounds()
    bounds.write_parquet(INDEX_DIR / "periodo_bounds.parquet")
    log(f"periodo: {bounds.height:,} period bounds → {INDEX_DIR / 'periodo_bounds.parquet'}")


if __name__ == "__main__":
    main()
