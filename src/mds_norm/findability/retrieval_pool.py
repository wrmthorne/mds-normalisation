from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import polars as pl

from mds_norm.findability.retrieval_benchmark import Index, Predicate, Query, fold
from mds_norm.paths import GOLD_RETRIEVAL, RAW_RECORDS, RETRIEVAL_OUT
from mds_norm.tables import record_nodes

RETRIEVAL = GOLD_RETRIEVAL
SAMPLES = RETRIEVAL / "samples"
RAW = RAW_RECORDS

POOL_DEPTH = 25  # top-k per corpus pooled for judgment


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def judgment_id(query_id: str, record_id: str) -> str:
    h = hashlib.sha1(f"{query_id}\x1f{record_id}".encode(), usedforsecurity=False).hexdigest()
    return f"x8j-{h[:12]}"


GIST_SEP = " · "
MAX_TERM_TOKENS = 6  # a longer segment is a catalogue sentence, not a query
MIN_WORD_CHARS = 3  # shorter words are not terms a searcher would type
DECADE = 10


def _segments(value: str) -> list[str]:
    return [s.strip() for s in value.split(GIST_SEP) if s.strip()]


def _words(segment: str) -> list[str]:
    """The words of a segment a searcher would plausibly type: no qualifiers, no identifiers"""
    segment = re.sub(r"\([^)]*\)", " ", segment)  # parenthetical qualifiers
    return [w for w in fold(segment).split() if len(w) >= MIN_WORD_CHARS and not any(c.isdigit() for c in w)]


def _term(gist: dict, family: str, head_of_comma_list: bool = False) -> str | None:
    """The first gist segment of a family that reads as a searchable term, else the first one truncated"""
    value = gist.get(family)
    if not value:
        return None
    candidates = []
    for segment in _segments(value):
        head = segment.split(",")[0] if head_of_comma_list else segment
        words = _words(head)
        if not words:
            continue
        if len(words) <= MAX_TERM_TOKENS:
            return " ".join(words)
        candidates.append(words)
    # ask for opening words rather than nothing
    return " ".join(candidates[0][:MAX_TERM_TOKENS]) if candidates else None


def _decade_window(lo: int, hi: int) -> tuple[int, int]:
    """The seed's own bounds widened to whole decades, so the date predicate is a period and not a key"""
    return (lo // DECADE) * DECADE, ((hi // DECADE) + 1) * DECADE - 1


def seed_queries() -> list[Query]:
    """One query per seed, built from the descriptive values of the seed's own record"""
    seeds = [json.loads(line) for line in (RETRIEVAL / "seeds.jsonl").read_text().splitlines() if line.strip()]
    raw_dates = RETRIEVAL_OUT / "raw_dates.parquet"
    bounds = dict(pl.read_parquet(raw_dates).select("record_id", pl.struct("year_lo", "year_hi")).iter_rows())

    out = []
    for s in seeds:
        gist = s.get("gist", {})
        preds: list[Predicate] = []
        obj = _term(gist, "object")
        if obj is None:
            continue  # nothing descriptive to ask for
        preds.append(Predicate("object", term=obj))
        place = _term(gist, "place", head_of_comma_list=True)
        if place:
            preds.append(Predicate("place", term=place))
        b = bounds.get(s["record_id"])
        if b is not None:
            lo, hi = _decade_window(b["year_lo"], b["year_hi"])
            preds.append(Predicate("date", lo=lo, hi=hi))
        out.append(Query(query_id=f"auto-{s['id']}", seed_id=s["id"], text=describe(preds), predicates=preds))
    return out


def describe(preds: list[Predicate]) -> str:
    """The query as a reader of the judging queue sees it"""
    parts = []
    for p in preds:
        if p.is_date:
            parts.append(f"date {p.lo}–{p.hi}")
        else:
            parts.append(f"{p.family} {p.term!r}")
    return "; ".join(parts)


def load_queries(path: Path) -> list[Query]:
    """Fold the formulation journal to its latest formulated entry per query_id"""
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            latest[e["query_id"]] = e
    return [Query.from_dict(e) for e in latest.values() if e.get("status") == "formulated" and e.get("predicates")]


def pool(queries: list[Query]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run every query against both corpora and pool the ranked lists"""
    raw, comp = Index("raw"), Index("compiled")
    ranked_rows, prov_rows = [], []
    for i, q in enumerate(queries, 1):
        hr = raw.run(q, k=POOL_DEPTH)
        hc = comp.run(q, k=POOL_DEPTH)
        for corpus, hits in (("raw", hr), ("compiled", hc)):
            for rank, h in enumerate(hits):
                ranked_rows.append(
                    {
                        "query_id": q.query_id,
                        "corpus": corpus,
                        "rank": rank,
                        "record_id": h["record_id"],
                        "n_matched": h["n_matched"],
                    }
                )
        rrank = {h["record_id"]: r for r, h in enumerate(hr)}
        crank = {h["record_id"]: r for r, h in enumerate(hc)}
        prov_rows.extend(
            {
                "query_id": q.query_id,
                "seed_id": q.seed_id,
                "query_class": q.query_class,
                "record_id": rid,
                "raw_rank": rrank.get(rid),
                "compiled_rank": crank.get(rid),
                "retrieved_by": ("both" if rid in rrank and rid in crank else "raw" if rid in rrank else "compiled"),
                "is_seed_record": None,  # filled by caller (needs seed→record map)
            }
            for rid in dict.fromkeys([*rrank, *crank])  # union, stable order
        )
        if i % 25 == 0 or i == len(queries):
            log(f"        pooled {i}/{len(queries)} queries")
    return pl.DataFrame(ranked_rows), pl.DataFrame(prov_rows)


def reconstruct(record_ids: list[str]) -> dict[str, dict]:
    """Raw substructure of every pooled record, for the judgment UI"""
    log(f"reconstructing {len(record_ids):,} pooled records")
    return record_nodes(record_ids, RAW)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    SAMPLES.mkdir(parents=True, exist_ok=True)

    qpath = RETRIEVAL / "queries.jsonl"
    if qpath.exists():
        queries = load_queries(qpath)
        log(f"{len(queries)} formulated queries loaded from {qpath.name}")
    else:
        queries = seed_queries()
        log(f"{len(queries)} queries derived from the seed records")

    ranked, prov = pool(queries)
    # mark seed records so known-item success is readable
    seed_map = {q.seed_id: q.query_id for q in queries}
    seeds = {
        json.loads(line)["id"]: json.loads(line)["record_id"]
        for line in (RETRIEVAL / "seeds.jsonl").read_text().splitlines()
    }
    seed_rec = {qid: seeds.get(sid) for sid, qid in seed_map.items()}
    prov = prov.with_columns(
        (pl.col("record_id") == pl.col("query_id").replace_strict(seed_rec, default=None)).alias("is_seed_record")
    )

    ranked.write_parquet(RETRIEVAL / "ranked_lists.parquet")
    prov.write_parquet(RETRIEVAL / "pool_provenance.parquet")
    log(f"ranked lists: {ranked.height:,} rows → ranked_lists.parquet")
    log(
        f"pool: {prov.height:,} (query, record) pairs, "
        f"{prov['record_id'].n_unique():,} distinct records "
        f"→ pool_provenance.parquet"
    )

    records = reconstruct(prov["record_id"].unique().to_list())
    with (RETRIEVAL / "judgment_records.jsonl").open("w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # blind items, shuffled so a query's hits are not adjacent
    qtext = {q.query_id: q.text for q in queries}
    items = (
        prov.select("query_id", "query_class", "record_id")
        .with_columns(
            pl.struct("query_id", "record_id")
            .map_elements(lambda s: judgment_id(s["query_id"], s["record_id"]), pl.String)
            .alias("id"),
            pl.col("query_id").replace_strict(qtext, default="").alias("query_text"),
        )
        .sort(pl.col("id").hash())
    )
    with (SAMPLES / "retrieval_judgments.jsonl").open("w", encoding="utf-8") as f:
        for it in items.iter_rows(named=True):
            f.write(
                json.dumps(
                    {
                        "id": it["id"],
                        "task": "retrieval_judgments",
                        "query_id": it["query_id"],
                        "query_text": it["query_text"],
                        "query_class": it["query_class"],
                        "record_id": it["record_id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    log(f"wrote {items.height:,} blind judgment items → retrieval_judgments.jsonl")


if __name__ == "__main__":
    main()
