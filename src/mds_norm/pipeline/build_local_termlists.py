from __future__ import annotations

import argparse
import json
import time

import polars as pl
from codecarbon import EmissionsTracker

from mds_norm.paths import EMISSIONS_LOG, FIELD_STATS, VOCAB_LOCAL, VOCABS
from mds_norm.utils.atomise import norm_term

LOCAL_VOCAB_PATH = VOCABS / "local"
OUT_DIR = VOCAB_LOCAL
EMISSIONS_LOG_PATH = EMISSIONS_LOG

RESIDUE_MIN_OCC = 100  # residue worklist floor; below it the tail is parked


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_fields(name: str) -> list[str]:
    """The SPECTRUM field(s) served by a seed"""
    seed = json.loads((LOCAL_VOCAB_PATH / f"{name}.json").read_text())
    return seed.get("fields", [seed.get("field", name)])


def load_termlist_index(field: str) -> pl.DataFrame:
    """Seed JSON -> (subject, term, lang, kind, norm), the authority-index schema"""
    seed = json.loads((LOCAL_VOCAB_PATH / f"{field}.json").read_text())
    rows = []
    for entry in seed["terms"]:
        subject = f"{field}/{entry['term'].replace(' ', '_')}"
        rows.append((subject, entry["term"], "en", "prefLabel"))
        rows.extend((subject, alt, "en", "altLabel") for alt in entry.get("alt", []))
    index = (
        pl.DataFrame(rows, schema=["subject", "term", "lang", "kind"], orient="row")
        .with_columns(norm=norm_term(pl.col("term")))
        .filter(pl.col("norm") != "")
        .unique()
    )
    dupes = index.filter(index["norm"].is_duplicated())
    if len(dupes):
        raise ValueError(
            f"{field}: one norm maps to several subjects — homograph in the seed list itself:\n{dupes.sort('norm')}"
        )
    return index


def harvest(field: str) -> pl.DataFrame:
    fields = [f"spectrum/{f}" for f in seed_fields(field)]
    return (
        pl.scan_parquet(FIELD_STATS)
        .filter(pl.col("field_type").is_in(fields) & pl.col("value").is_not_null())
        .with_columns(pl.col("value").str.strip_chars().str.replace_all(r"\s+", " "))
        .filter(pl.col("value") != "")
        .with_columns(norm=norm_term(pl.col("value")))
        .group_by("norm")
        .agg(occ=pl.len(), institutions=pl.col("data_source").n_unique(), example=pl.col("value").first())
        .sort("occ", descending=True)
        .collect(engine="streaming")
    )


def report(field: str, refresh_harvest: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    harvest_path = OUT_DIR / f"{field}_harvest.parquet"
    if refresh_harvest or not harvest_path.exists():
        log(f"{field}: harvesting distinct values from the field census")
        with EmissionsTracker(project_name="local_termlists", output_dir=str(EMISSIONS_LOG_PATH), log_level="error"):
            harvest(field).write_parquet(harvest_path)
    harvested = pl.read_parquet(harvest_path)

    index = load_termlist_index(field)
    matched = harvested.join(index.select("norm", "subject").unique(), on="norm", how="left")
    total = matched["occ"].sum()
    hit = matched.filter(pl.col("subject").is_not_null())["occ"].sum()
    n_terms = index["subject"].n_unique()
    log(f"{field}: {n_terms} canonical terms, {len(index)} surface forms")
    log(
        f"{field}: exact-norm coverage {hit:,}/{total:,} occurrences ({hit / total:.1%}), "
        f"{matched['subject'].is_not_null().sum():,}/{len(matched):,} distinct norms"
    )

    residue = matched.filter(pl.col("subject").is_null() & (pl.col("occ") >= RESIDUE_MIN_OCC)).select(
        "norm", "occ", "institutions", "example"
    )
    residue.write_parquet(OUT_DIR / f"{field}_residue.parquet")
    log(
        f"{field}: residue worklist — {len(residue)} norms ≥ {RESIDUE_MIN_OCC} occ "
        f"({residue['occ'].sum():,} occurrences) -> {OUT_DIR / f'{field}_residue.parquet'}"
    )
    with pl.Config(tbl_rows=25, fmt_str_lengths=60):
        print(residue.head(25))


def main() -> None:
    ap = argparse.ArgumentParser(description="Local harmonised termlists for the SPECTRUM termlist fields.")
    ap.add_argument(
        "--field", nargs="+", default=["persons_association"], help="termlist field(s), without the spectrum/ prefix"
    )
    ap.add_argument(
        "--refresh-harvest", action="store_true", help="rescan the field census instead of reusing the cached harvest"
    )
    args = ap.parse_args()
    for field in args.field:
        report(field, refresh_harvest=args.refresh_harvest)


if __name__ == "__main__":
    main()
