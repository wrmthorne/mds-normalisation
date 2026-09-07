from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

from mds_norm.paths import FIELD_STATS, VOCAB_DECISIONS, VOCAB_INDEXES, VOCAB_INSTITUTIONAL, VOCABS
from mds_norm.utils.atomise import morph_variants, norm_term

VOCAB_PATH = VOCABS
INDEX_DIR = VOCAB_INDEXES
DECISIONS = VOCAB_DECISIONS
OUT_DIR = VOCAB_INSTITUTIONAL
MAP_OUT = VOCABS / "institution_vocab_map.json"
TABLE_OUT = VOCABS / "institution_vocab_table.parquet"

# field_vocab_map targets plus the house lists institutions catalogue from
FIELD_CANDIDATES = {
    "material": ["aat", "fish_building_materials", "bm_materials"],
    "date_period": ["periodo", "he_periods"],
    "associated_concept": ["aat", "shic", "fish_subjects"],
    "content_concept": ["aat", "shic", "fish_subjects"],
}

MIN_OCC_TOTAL = 500  # below this an institution's field sample can't support a verdict
PREDICT_MIN_COV = 0.5  # predicted base vocabulary must cover at least half the occurrences
CONFIRM_COV = 0.7  # …and is "confirmed" at this coverage
FREE_TEXT_MAX_COV = 0.3  # …and below this the field reads as free text
# Exclusive terms only, with a share floor against sheer volume
SPECIFIC_MIN_TYPES = 20
SPECIFIC_MIN_OCC = 200
SPECIFIC_MIN_SHARE = 0.001
FUZZY_CUTOFF = 92.0  # indel ratio floor for the near-miss (typo) class
FUZZY_MIN_OCC = 5  # fuzzy classification floor against large vocabularies
FUZZY_LARGE_VOCAB = 10_000  # past this many terms every norm finds a neighbour
EXTENSION_MIN_OCC = 50  # a frequent, term-shaped residue norm is an extension
EXTENSION_MAX_TOKENS = 3  # …and term-shaped means no more words than this
TERM_MAX_TOKENS, TERM_MAX_CHARS = 4, 40  # past either, a residue norm reads as prose
EXCLUSIVE_EXAMPLES = 10  # terms carried per (institution, list) as evidence


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ## BM Materials Thesaurus index

# Per-letter HTML; anchor numbers are unique and become subject ids

BM_PREF = re.compile(r'<A NAME="(\d+)"></A><B>([^<]+)</B>')
BM_ALT = re.compile(r'<I>([^<]+)</I>\s*see\s*<B>[^<]+</B><A HREF="mathes\w\.htm#(\d+)"')


def build_bm_materials() -> pl.DataFrame:
    src = VOCAB_PATH / "bm_materials"
    rows = []
    for page in sorted(src.glob("mathes*.htm")):
        text = re.sub(r"\s+", " ", page.read_text(encoding="cp1252"))
        for subject, term in BM_PREF.findall(text):
            rows.append((f"bm/{subject}", term.strip(), "en", "prefLabel"))
    for page in sorted(src.glob("maindex*.htm")):
        text = re.sub(r"\s+", " ", page.read_text(encoding="cp1252"))
        for term, subject in BM_ALT.findall(text):
            rows.append((f"bm/{subject}", term.strip(), "en", "altLabel"))
    return (
        pl.DataFrame(rows, schema=["subject", "term", "lang", "kind"], orient="row")
        .with_columns(norm=norm_term(pl.col("term")))
        .filter(pl.col("norm") != "")
        .unique()
    )


# ## Historic England Periods list

# Preferred name, comma-separated altLabels, and the short entry code


def build_he_periods() -> pl.DataFrame:
    import csv

    src = next((VOCAB_PATH / "he_periods").glob("HE_Periods*.csv"))
    rows = []
    with src.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            subject = f"he/{r['PERIOD_UID']}"
            rows.append((subject, r["NAME"].strip(), "en", "prefLabel"))
            rows.extend((subject, alt.strip(), "en", "altLabel") for alt in r["altLabel"].split(",") if alt.strip())
            if r["SHORT_NAME"].strip():
                rows.append((subject, r["SHORT_NAME"].strip(), "en", "code"))
    return (
        pl.DataFrame(rows, schema=["subject", "term", "lang", "kind"], orient="row")
        .with_columns(norm=norm_term(pl.col("term")))
        .filter(pl.col("norm") != "")
        .unique()
    )


# ## SHIC (Social History & Industrial Classification)

# Hand-extracted from a PDF: one indented `code label` line each

SHIC_LINE = re.compile(r"^\s*(\d+(?:\.\d+)?)(?:\s+-(\d+))?\s+(.+?)\s*$")
SHIC_GENERIC = {"general", "other"}


def build_shic() -> pl.DataFrame:
    rows = []
    for line in (VOCAB_PATH / "SHIC.txt").read_text().splitlines():
        m = SHIC_LINE.match(line)
        if not m:
            continue
        code, range_end, label = m.groups()
        codes = [code] + ([f"{code.split('.')[0]}.{range_end}"] if range_end else [])
        subject = f"shic/{code}" + (f"-{range_end}" if range_end else "")
        if label.lower() not in SHIC_GENERIC:
            rows.append((subject, label, "en", "prefLabel"))
        for c in codes:
            if "." in c:
                rows.append((subject, c, "en", "code"))
            rows.append((subject, f"{c} {label}", "en", "codeLabel"))
    return (
        pl.DataFrame(rows, schema=["subject", "term", "lang", "kind"], orient="row")
        .with_columns(norm=norm_term(pl.col("term")))
        .filter(pl.col("norm") != "")
        .unique()
    )


# ## FISH Heritage Subjects & Themes

# heritagedata.org scheme 595, fetched as concepts JSON; prefLabels only


def build_fish_subjects() -> pl.DataFrame:
    data = json.loads((VOCAB_PATH / "fish_subjects" / "fish_subjects_concepts.json").read_text())
    rows = [
        (c["uri"].rstrip("/").rsplit("/", 1)[-1], c["label"].strip(), c.get("label lang"), "prefLabel") for c in data
    ]
    return (
        pl.DataFrame(rows, schema=["subject", "term", "lang", "kind"], orient="row")
        .with_columns(norm=norm_term(pl.col("term")))
        .filter(pl.col("norm") != "")
        .unique()
    )


BUILDERS = {
    "bm_materials": build_bm_materials,
    "he_periods": build_he_periods,
    "shic": build_shic,
    "fish_subjects": build_fish_subjects,
}


def load_index(vocab: str) -> pl.DataFrame:
    cache = INDEX_DIR / f"{vocab}.parquet"
    if not cache.exists():
        if vocab not in BUILDERS:
            raise FileNotFoundError(f"{cache} — build it in mds_norm.pipeline.vocab_indexes first")
        index = BUILDERS[vocab]()
        index.write_parquet(cache)
        log(f"{vocab}: {index['subject'].n_unique()} subjects, {len(index)} surface forms -> {cache}")
    return pl.read_parquet(cache)


# ## Detection


def field_value_counts(field: str) -> pl.DataFrame:
    """Per-institution occurrence counts of this field's values, from field_stats"""
    return (
        pl.scan_parquet(FIELD_STATS)
        .filter((pl.col("field_type") == f"spectrum/{field}") & pl.col("value").is_not_null())
        .select(pl.col("data_source").cast(pl.String), pl.col("value").str.strip_chars().str.replace_all(r"\s+", " "))
        .filter(pl.col("value") != "")
        .group_by("data_source", "value")
        .agg(count=pl.len().cast(pl.UInt32))
        .collect(engine="streaming")
    )


def atom_universe(field: str) -> pl.DataFrame:
    field_map = json.loads((VOCAB_PATH / "field_vocab_map.json").read_text())
    group = "+".join(field_map[field])
    atoms = (
        pl.read_parquet(DECISIONS)
        .filter((pl.col("group") == group) & (pl.col("atom_route") == "cascade"))
        .with_columns(pl.col("data_source").cast(pl.String))
    )
    if atoms.is_empty():
        raise ValueError(f"no cascade atoms for group {group!r} in {DECISIONS}")
    pooled = [f for f, v in field_map.items() if "+".join(v) == group]
    if len(pooled) > 1:
        # The decisions parquet pools fields, so weights are re-derived here
        atoms = atoms.drop("count").join(field_value_counts(field), on=["data_source", "value"], how="inner")
        log(f"{field}: group {group!r} pools {len(pooled)} fields — reweighted to field-only counts")
    return (
        atoms.filter(pl.col("norm").is_not_null() & (pl.col("norm") != ""))
        .group_by("data_source", "norm")
        .agg(
            occ=pl.col("count").sum(),
            example=pl.col("atom").first(),
            nb4_vocab=pl.col("vocab").drop_nulls().first(),
            nb4_status=pl.col("status").first(),
        )
    )


def membership(per_inst: pl.DataFrame, indexes: dict[str, pl.DataFrame]) -> pl.DataFrame:
    norms = per_inst.select("norm").unique()
    for vocab, index in indexes.items():
        norms = norms.with_columns(pl.col("norm").is_in(index["norm"].implode()).alias(vocab))
    flags = [pl.col(v) for v in indexes]
    return norms.with_columns(n_vocabs=pl.sum_horizontal(flags))


def coverage_table(per_inst: pl.DataFrame, flagged: pl.DataFrame, vocabs: list[str]) -> pl.DataFrame:
    df = per_inst.join(flagged, on="norm")
    aggs = [pl.col("occ").sum().alias("occ_total"), pl.len().alias("n_norms")]
    for v in vocabs:
        aggs += [
            (pl.col("occ") * pl.col(v)).sum().alias(f"occ_{v}"),
            pl.col(v).sum().alias(f"types_{v}"),
            (pl.col("occ") * (pl.col(v) & (pl.col("n_vocabs") == 1))).sum().alias(f"excl_{v}"),
            (pl.col(v) & (pl.col("n_vocabs") == 1)).sum().alias(f"excltypes_{v}"),
            # Filter the sort key too, or the group lengths disagree
            pl.col("example")
            .filter(pl.col(v) & (pl.col("n_vocabs") == 1))
            .sort_by(pl.col("occ").filter(pl.col(v) & (pl.col("n_vocabs") == 1)), descending=True)
            .head(EXCLUSIVE_EXAMPLES)
            .alias(f"exclterms_{v}"),
        ]
    aggs.append((pl.col("occ") * (pl.col("n_vocabs") == 0)).sum().alias("occ_unmatched"))
    table = df.group_by("data_source").agg(aggs)
    return table.with_columns(
        [(pl.col(f"occ_{v}") / pl.col("occ_total")).alias(f"cov_{v}") for v in vocabs]
        # Coverage by term as well as by occurrence
        + [(pl.col(f"types_{v}") / pl.col("n_norms")).alias(f"covtypes_{v}") for v in vocabs]
        + [(pl.col(f"excl_{v}") / pl.col("occ_total")).alias(f"exclshare_{v}") for v in vocabs]
    ).sort("occ_total", descending=True)


def predict(row: dict, vocabs: list[str]) -> dict:
    """Two separable claims per institution: the base vocabulary and the specific house list"""
    if row["occ_total"] < MIN_OCC_TOTAL:
        return {"vocabs": [], "verdict": "insufficient_data"}
    base = max(vocabs, key=lambda v: row[f"cov_{v}"])
    specific = [
        v
        for v in vocabs
        if v != base
        and row[f"excltypes_{v}"] >= SPECIFIC_MIN_TYPES
        and row[f"excl_{v}"] >= SPECIFIC_MIN_OCC
        and row[f"exclshare_{v}"] >= SPECIFIC_MIN_SHARE
    ]
    if row[f"cov_{base}"] < PREDICT_MIN_COV and not specific:
        verdict = "free_text" if row[f"cov_{base}"] < FREE_TEXT_MAX_COV else "unclear"
        return {
            "vocabs": [],
            "verdict": verdict,
            "coverage": {base: round(row[f"cov_{base}"], 4)},
            "occ_total": row["occ_total"],
        }
    predicted = [base, *specific]
    verdict = "confirmed" if row[f"cov_{base}"] >= CONFIRM_COV else "partial"
    out = {
        "vocabs": predicted,
        "verdict": verdict,
        "coverage": {v: round(row[f"cov_{v}"], 4) for v in predicted},
        "occ_total": row["occ_total"],
    }
    if specific:
        out["specific_evidence"] = {
            v: {
                "exclusive_types": row[f"excltypes_{v}"],
                "exclusive_occ": row[f"excl_{v}"],
                "terms": row[f"exclterms_{v}"],
            }
            for v in specific
        }
    return out


# ## Residue classification

# One label per unmatched norm, tested in the order listed

PAREN_QUALIFIER = re.compile(r"\s*\([^()]*\)$")


def plural_variants(norm: str) -> list[str]:
    out = []
    if norm.endswith("ies"):
        out.append(norm[:-3] + "y")
    if norm.endswith("es"):
        out.append(norm[:-2])
    if norm.endswith("s"):
        out.append(norm[:-1])
    out += [norm + "s", norm + "es"]
    return out


def classify_residue(residue: pl.DataFrame, predicted: list[str], indexes: dict[str, pl.DataFrame]) -> pl.DataFrame:
    vocab_norms = {v: set(ix["norm"].to_list()) for v, ix in indexes.items()}
    target = set().union(*(vocab_norms[v] for v in predicted))
    target_arr = list(target)
    fuzzy_ok = residue.filter(pl.col("occ") >= (FUZZY_MIN_OCC if len(target_arr) > FUZZY_LARGE_VOCAB else 1))
    scores: dict[str, tuple[str, float]] = {}
    queries = fuzzy_ok["norm"].to_list()
    for start in range(0, len(queries), 512):
        chunk = queries[start : start + 512]
        mat = cdist(chunk, target_arr, scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF, dtype=np.uint8, workers=-1)
        best = mat.argmax(axis=1)
        for i, (q, j) in enumerate(zip(chunk, best, strict=True)):
            if mat[i, j] >= FUZZY_CUTOFF:
                scores[q] = (target_arr[j], float(mat[i, j]))

    def label(norm: str, occ: int) -> tuple[str, str | None, float | None]:
        for other, norms in vocab_norms.items():
            if other not in predicted and norm in norms:
                return f"other:{other}", None, None
        stripped = PAREN_QUALIFIER.sub("", norm)
        if stripped != norm and stripped in target:
            return "qualifier_variant", stripped, None
        for variant in plural_variants(norm) + morph_variants(norm):
            if variant in target:
                return "variant", variant, None
        if norm.endswith(")") and "(" not in norm:
            base = norm.rstrip(") ")
            if base in target:
                return "split_artifact", base, None
        if norm in scores:
            nearest, score = scores[norm]
            return "near_miss", nearest, score
        tokens = norm.count(" ") + 1
        if occ >= EXTENSION_MIN_OCC and tokens <= EXTENSION_MAX_TOKENS and not any(c.isdigit() for c in norm):
            return "extension_candidate", None, None
        if tokens > TERM_MAX_TOKENS or len(norm) > TERM_MAX_CHARS or any(c.isdigit() for c in norm):
            return "free_text", None, None
        return "low_freq_tail", None, None

    labelled = [label(n, o) for n, o in zip(residue["norm"], residue["occ"], strict=True)]
    return residue.with_columns(
        residue_class=pl.Series([lab[0] for lab in labelled]),
        nearest_term=pl.Series([lab[1] for lab in labelled], dtype=pl.String),
        nearest_score=pl.Series([lab[2] for lab in labelled], dtype=pl.Float64),
    )


def write_map_table(mapping: dict) -> None:
    """The (field, institution, list) table the cascade consults before matching anything"""
    rows = [
        {
            "data_source": inst,
            "field": field,
            "vocab": vocab,
            "verdict": pred["verdict"],
            "coverage": (pred.get("coverage") or {}).get(vocab),
            "occ_total": pred.get("occ_total"),
            "exclusive_terms": ((pred.get("specific_evidence") or {}).get(vocab) or {}).get("terms"),
        }
        for inst, fields in mapping.items()
        for field, pred in fields.items()
        for vocab in (pred.get("vocabs") or [None])
    ]
    pl.DataFrame(
        rows,
        schema={
            "data_source": pl.String,
            "field": pl.String,
            "vocab": pl.String,
            "verdict": pl.String,
            "coverage": pl.Float64,
            "occ_total": pl.Int64,
            "exclusive_terms": pl.List(pl.String),
        },
    ).sort("field", "data_source").write_parquet(TABLE_OUT)


def run_field(field: str) -> None:
    vocabs = FIELD_CANDIDATES[field]
    indexes = {v: load_index(v) for v in vocabs}
    for v, ix in indexes.items():
        log(f"{v}: {ix['norm'].n_unique():,} distinct norms")
    per_inst = atom_universe(field)
    log(
        f"{field}: {len(per_inst):,} distinct (institution, norm) atoms, "
        f"{per_inst['occ'].sum():,} occurrences, "
        f"{per_inst['data_source'].n_unique()} institutions"
    )

    flagged = membership(per_inst, indexes)
    table = coverage_table(per_inst, flagged, vocabs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(OUT_DIR / f"{field}_vocab_coverage.parquet")

    predictions = {row["data_source"]: predict(row, vocabs) for row in table.iter_rows(named=True)}
    existing = json.loads(MAP_OUT.read_text()) if MAP_OUT.exists() else {}
    for inst, pred in predictions.items():
        existing.setdefault(inst, {})[field] = pred
    MAP_OUT.write_text(json.dumps(existing, indent=1, ensure_ascii=False) + "\n")
    write_map_table(existing)
    log(f"{field}: predictions -> {MAP_OUT} and {TABLE_OUT}")

    with pl.Config(tbl_rows=30, tbl_cols=20, fmt_str_lengths=44):
        cols = ["data_source", "occ_total"] + [f"cov_{v}" for v in vocabs] + [f"exclshare_{v}" for v in vocabs]
        print(table.select(cols).head(30))

    # Residue for institutions with a predicted (partial or confirmed) vocabulary
    joined = per_inst.join(flagged, on="norm")
    residues = []
    for inst, pred in predictions.items():
        if not pred["vocabs"]:
            continue
        matched_any = pl.any_horizontal([pl.col(v) for v in pred["vocabs"]])
        residue = (
            joined.filter((pl.col("data_source") == inst) & ~matched_any)
            .select("data_source", "norm", "occ", "example", "nb4_vocab", "nb4_status")
            .sort("occ", descending=True)
        )
        if residue.is_empty():
            continue
        residues.append(
            classify_residue(residue, pred["vocabs"], indexes).with_columns(
                predicted_vocab=pl.lit("+".join(pred["vocabs"]))
            )
        )
    if residues:
        all_residue = pl.concat(residues)
        all_residue.write_parquet(OUT_DIR / f"{field}_vocab_residue.parquet")
        summary = (
            all_residue.group_by("predicted_vocab", "residue_class")
            .agg(n=pl.len(), occ=pl.col("occ").sum())
            .sort(["predicted_vocab", "occ"], descending=[False, True])
        )
        log(f"{field}: residue classes across {all_residue['data_source'].n_unique()} predicted institutions")
        print(summary)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect which institutions catalogue a field from which published vocabulary."
    )
    ap.add_argument("--field", nargs="+", default=["material"], choices=sorted(FIELD_CANDIDATES))
    args = ap.parse_args()
    for field in args.field:
        run_field(field)


if __name__ == "__main__":
    main()
