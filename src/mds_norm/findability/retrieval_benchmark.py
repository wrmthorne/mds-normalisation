from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from mds_norm.paths import COMPILED as COMPILED_DIR
from mds_norm.paths import PLACES_OUT, RAW_RECORDS, RETRIEVAL_OUT, VOCAB_INDEXES

RAW = RAW_RECORDS
COMPILED = COMPILED_DIR / "mds-normalised.parquet"
PLACES = PLACES_OUT / "place_value_decisions.parquet"
TGN = VOCAB_INDEXES / "tgn.parquet"
TGN_ANC = VOCAB_INDEXES / "tgn_ancestors.parquet"
OUT = RETRIEVAL_OUT

# source fields unioned into one searchable bag per family
FAMILY_FIELDS: dict[str, list[str]] = {
    "object": ["spectrum/object_name", "spectrum/title", "spectrum/object_component_name"],
    "material": ["spectrum/material"],
    "technique": ["spectrum/technique"],
    "person": [
        "spectrum/object_production_person",
        "spectrum/persons_surname",
        "spectrum/persons_forenames",
        "spectrum/persons_association",
        "spectrum/associated_person",
        "spectrum/content_person",
        "spectrum/object_production_organisation",
        "spectrum/organisations_association",
    ],
    "place": [
        "spectrum/object_production_place",
        "spectrum/place_association",
        "spectrum/field_collection_place",
        "spectrum/associated_place",
        "spectrum/content_place",
        "spectrum/ownership_place",
    ],
    # date is indexed as text; typed bounds are computed separately
    "date": ["spectrum/object_production_date", "spectrum/date_period", "spectrum/date_text"],
}
TEXT_FAMILIES = list(FAMILY_FIELDS)
# numeric fields giving a record its typed production-date bound
DATE_BOUND_FIELDS = ["spectrum/date_earliest_single", "spectrum/date_latest"]

_FOLD = re.compile(r"[^0-9a-z]+")
_YEAR = re.compile(r"-?\d{3,4}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fold(s: str) -> str:
    """The single normalisation applied identically to both corpora"""
    return _FOLD.sub(" ", s.casefold()).strip()


MIN_TOKEN = 3  # shorter tokens prefix-match too much to constrain anything
MAX_YEAR = 2026  # a parsed year beyond the present is a mis-parse


def query_tokens(term: str) -> list[str]:
    """The folded tokens a text predicate requires, dropping stop-length noise"""
    return [t for t in fold(term).split() if len(t) >= MIN_TOKEN]


def _fold_expr(col: str = "value") -> pl.Expr:
    return pl.col(col).cast(pl.String).str.to_lowercase().str.replace_all(r"[^0-9a-z]+", " ").str.strip_chars()


def place_expansion() -> pl.DataFrame:
    """The extra tokens a TGN-aware search should see for a resolved place"""
    dec = (
        pl.read_parquet(PLACES, columns=["matched_term", "subject", "status"])
        .filter((pl.col("status") == "resolved") & pl.col("matched_term").is_not_null())
        .with_columns(pl.col("matched_term").map_elements(fold, pl.String).alias("mt"))
        .select("mt", "subject")
        .unique()
    )
    if dec.is_empty():
        return pl.DataFrame({"mt": [], "extra": []}, schema={"mt": pl.String, "extra": pl.String})
    anc = pl.read_parquet(TGN_ANC)  # subject -> ancestor
    labels = (
        pl.read_parquet(TGN, columns=["subject", "norm", "kind"])
        .filter(pl.col("kind").str.starts_with("prefLabel"))
        .select(pl.col("subject").alias("sid"), pl.col("norm").alias("label"))
        .unique()
    )
    # ancestor subjects of each resolved term → their folded labels
    return (
        dec.join(anc, on="subject", how="inner")
        .select("mt", pl.col("ancestor").alias("sid"))
        .join(labels, on="sid", how="inner")
        .group_by("mt")
        .agg(pl.col("label").unique().str.join(" ").alias("extra"))
    )


def _text_index(corpus: pl.LazyFrame, expand_places: bool) -> pl.DataFrame:
    """One (record_id, data_source, family, bag) row per record and family"""
    field_to_family = {f: fam for fam, fs in FAMILY_FIELDS.items() for f in fs}
    nodes = (
        corpus.filter(pl.col("field_type").is_in(list(field_to_family)) & pl.col("value").is_not_null())
        .select(
            "record_id",
            "data_source",
            pl.col("field_type").replace_strict(field_to_family).alias("family"),
            _fold_expr().alias("tok"),
        )
        .filter(pl.col("tok").str.len_chars() > 0)
    )
    if expand_places:
        exp = place_expansion()
        log(f"        place expansion: {exp.height:,} resolved canonical terms")
        # append ancestor tokens to resolved place nodes
        nodes = (
            nodes.join(exp.lazy(), left_on="tok", right_on="mt", how="left")
            .with_columns(
                pl.when((pl.col("family") == "place") & pl.col("extra").is_not_null())
                .then(pl.col("tok") + " " + pl.col("extra"))
                .otherwise(pl.col("tok"))
                .alias("tok")
            )
            .drop("extra")
        )
    return (
        nodes.group_by("record_id", "data_source", "family")
        .agg(pl.col("tok").unique().str.join(" ").alias("bag"))
        .collect(engine="streaming")
    )


def _date_index(corpus: pl.LazyFrame) -> pl.DataFrame:
    """Typed production-date bounds per record"""
    return (
        corpus.filter(pl.col("field_type").is_in(DATE_BOUND_FIELDS) & pl.col("value").is_not_null())
        .with_columns(pl.col("value").str.extract(r"(-?\d{3,4})", 1).cast(pl.Int64, strict=False).alias("yr"))
        .filter(pl.col("yr").is_not_null() & (pl.col("yr").abs() <= MAX_YEAR))
        .group_by("record_id")
        .agg(pl.col("yr").min().alias("year_lo"), pl.col("yr").max().alias("year_hi"))
        .collect(engine="streaming")
    )


def build_index(corpus_path: Path, name: str, expand_places: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"building {name} index from {corpus_path.name}")
    corpus = pl.scan_parquet(corpus_path)
    text = _text_index(corpus, expand_places)
    text.write_parquet(OUT / f"{name}_text.parquet")
    log(f"        text: {text.height:,} record×family bags → {name}_text.parquet")
    dates = _date_index(corpus)
    dates.write_parquet(OUT / f"{name}_dates.parquet")
    log(f"        dates: {dates.height:,} records with a typed bound → {name}_dates.parquet")


@dataclass
class Predicate:
    family: str
    term: str | None = None  # text families
    lo: int | None = None  # date family
    hi: int | None = None

    @property
    def is_date(self) -> bool:
        return self.family == "date"


@dataclass
class Query:
    query_id: str
    seed_id: str
    text: str
    predicates: list[Predicate] = field(default_factory=list)

    @property
    def query_class(self) -> str:
        return "+".join(sorted({p.family for p in self.predicates}))

    @classmethod
    def from_dict(cls, d: dict) -> Query:
        return cls(
            query_id=d["query_id"],
            seed_id=d.get("seed_id", ""),
            text=d.get("text", ""),
            predicates=[Predicate(**p) for p in d["predicates"]],
        )


class Index:
    """A loaded per-corpus index, ready to answer queries"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.text = pl.read_parquet(OUT / f"{name}_text.parquet")
        self.dates = pl.read_parquet(OUT / f"{name}_dates.parquet")
        self._by_family = {fam: g for (fam,), g in self.text.group_by("family")}
        self._src = self.text.select("record_id", "data_source").unique()

    def _match_text(self, pred: Predicate) -> pl.Series:
        g = self._by_family.get(pred.family)
        tokens = query_tokens(pred.term or "")
        if not tokens or g is None:
            return pl.Series("record_id", [], dtype=pl.String)
        # every token must appear as a word-prefix; bags are unordered
        matched = g
        for tok in tokens:
            matched = matched.filter(pl.col("bag").str.contains(r"\b" + re.escape(tok)))
        return matched["record_id"]

    def _match_date(self, pred: Predicate) -> pl.Series:
        lo = pred.lo if pred.lo is not None else -10_000
        hi = pred.hi if pred.hi is not None else 10_000
        # typed: parsed-bound interval overlap
        typed = self.dates.filter((pl.col("year_lo") <= hi) & (pl.col("year_hi") >= lo))["record_id"]
        # literal-year fallback, only for records with no typed bound
        date_bags = self._by_family.get("date")
        if date_bags is None:
            return typed
        untyped = date_bags.join(self.dates.select("record_id"), on="record_id", how="anti").with_columns(
            pl.col("bag").str.extract_all(r"-?\d{3,4}").alias("yrs")
        )
        strmatch = (
            untyped.explode("yrs")
            .with_columns(pl.col("yrs").cast(pl.Int64, strict=False))
            .filter(pl.col("yrs").is_between(lo, hi))["record_id"]
            .unique()
        )
        return pl.concat([typed, strmatch]).unique()

    def run(self, query: Query, k: int = 25) -> list[dict]:
        """Rank records by number of predicates satisfied and return the top k"""
        hits = [(self._match_date(p) if p.is_date else self._match_text(p)) for p in query.predicates]
        hits = [h for h in hits if h.len()]
        if not hits:
            return []
        scored = (
            pl.concat(hits)
            .value_counts()
            .rename({"count": "n_matched"})
            .join(self._src, on="record_id", how="left")
            .sort(["n_matched", "record_id"], descending=[True, False])
            .head(k)
        )
        return [
            {"record_id": r, "data_source": ds, "n_matched": n, "corpus": self.name}
            for r, ds, n in scored.select("record_id", "data_source", "n_matched").iter_rows()
        ]


SMOKE = [
    Query(
        "smoke-1",
        "",
        "ceramics, England, 1800-1850",
        [Predicate("object", term="pot"), Predicate("place", term="england"), Predicate("date", lo=1800, hi=1850)],
    ),
    Query("smoke-2", "", "photographs (any date)", [Predicate("object", term="photograph")]),
    Query("smoke-3", "", "oak furniture", [Predicate("object", term="chair"), Predicate("material", term="oak")]),
]


def smoke() -> None:
    raw, comp = Index("raw"), Index("compiled")
    for q in SMOKE:
        hr, hc = raw.run(q), comp.run(q)
        print(f"\n[{q.query_id}] class={q.query_class!r}  {q.text!r}")
        print(f"    raw:      {len(hr)} hits, top n_matched={hr[0]['n_matched'] if hr else 0}")
        print(f"    compiled: {len(hc)} hits, top n_matched={hc[0]['n_matched'] if hc else 0}")
        pool = {h["record_id"] for h in hr} | {h["record_id"] for h in hc}
        print(f"    pooled union: {len(pool)} records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build both indices")
    ap.add_argument("--smoke", action="store_true", help="run synthetic queries")
    args = ap.parse_args()
    if args.build:
        build_index(RAW, "raw", expand_places=False)
        build_index(COMPILED, "compiled", expand_places=True)
    if args.smoke:
        smoke()
    if not (args.build or args.smoke):
        ap.error("nothing to do: pass --build and/or --smoke")


if __name__ == "__main__":
    main()
