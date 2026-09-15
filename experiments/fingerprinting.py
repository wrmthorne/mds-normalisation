from __future__ import annotations

import argparse
import re

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from experiments import record_sample
from experiments.harness import log
from mds_norm import paths
from mds_norm.metrics import completeness, conformance, kiraly, thinness
from mds_norm.metrics.common import ADMIN, RELEASE

SEED = record_sample.SEED
OUT = paths.EXP_OUT / "fingerprinting"
DIVERGENCE = paths.PATTERNS_OUT / "slot_divergence.parquet"
CONSISTENCY_FIELD = paths.CONSISTENCY_OUT / "consistency_field_raw.parquet"

# The three fields nearly every institution populates
CORE = ["spectrum/object_number", "spectrum/object_name", "spectrum/brief_description"]

# Candidate representations; the textual ones become tf-idf documents
NAMES = {
    "occ": "occupancy",
    "mor": "morphology",
    "fmo": "field-tagged morphology",
    "lex": "lexical",
    "qual": "quality vector",
}
# The quality vector reads the whole record, ignoring dropout

TEXT_KEYS = ("occ", "mor", "fmo", "lex")
MAX_FEATURES = {"occ": 400, "mor": 5000, "fmo": 20000, "lex": 30000}
# Each candidate alone, the two coarse views, the three field views, and those with the quality vector. Value
# morphology is the field-tagged distribution summed over fields, so it is left out of the larger combinations
COMBOS = [(k,) for k in NAMES] + [("occ", "mor"), ("occ", "fmo", "lex"), ("occ", "fmo", "lex", "qual")]

# The measures the quality vector stacks, in order
QUALITY = ["completeness", "thinness", "decomposition_rate", "conformance", "consistency", "k_completeness_weighted"]

LEX_VOCAB = 20000
PER_SIDE, MIN_SIDE = 400, 200
MIN_SAMPLE_RECORDS = record_sample.MIN_RECORDS
WITHIN_REPS = 3
FOLDS = 5
# Share of each record's populated fields withheld at test time
DROPOUT = [0.0, 0.25, 0.5, 0.75, 0.9]
# Records per half in the sample-efficiency sweep, and how many times each size is drawn
EFFICIENCY_SIZES = [1, 2, 5, 10, 25, 50, 100, 150]
EFFICIENCY_DRAWS = 20
# Minimum records a side for the institution-level tests
FP_MIN_SIDE = 150

# A field earns its own run only where widely used
VARIANT_FIELD_MIN_INSTITUTIONS = 20
VARIANT_FIELD_MIN_OCCURRENCES = 10_000
# Past this length a value's shape is unique to it
MAX_SHAPE_CHARS = 64

RUN = re.compile(r"(s+|d+|.)")
MASK = pl.col("value").str.replace_all(r"[A-Za-z]", "s").str.replace_all(r"[0-9]", "d").str.slice(0, MAX_SHAPE_CHARS)


def signature(mask: str) -> str:
    return "".join(f"{m[0]}{{{len(m)}}}" if len(m) > 1 else m for m in RUN.findall(mask))


def corpus_rows(all_institutions: bool) -> pl.DataFrame:
    """The balanced sample, optionally with every record of the institutions below its record floor"""
    rows = record_sample.sample_rows()
    if not all_institutions:
        return rows
    counts = record_sample.institution_counts()
    small = counts.filter(pl.col("n_records") < MIN_SAMPLE_RECORDS)["data_source"].to_list()
    extra = record_sample.cached(
        "sample_rows_small",
        lambda: (
            pl.scan_parquet(paths.RAW_RECORDS).filter(pl.col("data_source").is_in(small)).collect(engine="streaming")
        ),
    )
    log(
        f"adding {extra['data_source'].n_unique()} institutions below {MIN_SAMPLE_RECORDS} records "
        f"({extra['record_id'].n_unique():,} records)"
    )
    return pl.concat([rows, extra.select(rows.columns)])


def prepare(rows: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Populated descriptive statements, and the short subset carrying a masked signature"""
    populated = rows.filter(pl.col("value").is_not_null() & ~(ADMIN | RELEASE))
    short = populated.filter(pl.col("value").str.len_chars().is_between(1, MAX_SHAPE_CHARS)).with_columns(
        MASK.alias("mask")
    )
    sigs = (
        short.select("mask")
        .unique()
        # Spaces are escaped so a shape stays one feature
        .with_columns(
            pl.col("mask").map_elements(signature, return_dtype=pl.String).str.replace_all(" ", "␣").alias("sig")
        )
    )
    return populated, short.join(sigs, on="mask", how="left")


def documents(populated: pl.DataFrame, short: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    """One row per record with a document for each textual representation"""
    return (
        keys.join(
            populated.group_by("record_id").agg(pl.col("field_type").unique().sort().str.join(" ").alias("occ")),
            on="record_id",
            how="left",
        )
        .join(
            short.group_by("record_id").agg(pl.col("sig").sort().str.join(" ").alias("mor")),
            on="record_id",
            how="left",
        )
        .join(
            short.with_columns(fs=pl.col("field_type").cast(pl.String) + "‖" + pl.col("sig"))
            .group_by("record_id")
            .agg(pl.col("fs").sort().str.join(" ").alias("fmo")),
            on="record_id",
            how="left",
        )
        # Lexical reads every value, prose included
        .join(
            populated.group_by("record_id").agg(pl.col("value").str.join(" ").alias("lex")), on="record_id", how="left"
        )
        .with_columns(pl.col("occ", "mor", "fmo", "lex").fill_null(""))
        # unique() does not preserve order, so the folds would otherwise differ between runs
        .sort("record_id")
    )


def quality_vectors(keys: pl.DataFrame) -> np.ndarray:
    """The quality measures per record, medians filled in where a measure is undefined"""
    metrics = record_sample.record_metrics().select("record_id", *QUALITY)
    joined = keys.select("record_id").join(metrics, on="record_id", how="left")
    return np.column_stack([joined[m].fill_null(joined[m].median() or 0.0).fill_nan(0.0).to_numpy() for m in QUALITY])


def features(populated: pl.DataFrame, short: pl.DataFrame, keys: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    docs = documents(populated, short, keys)
    return docs, quality_vectors(docs)


def frozen_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """The corpus-wide weight tables the quality measures read, frozen across dropout"""
    weights = pl.read_parquet(paths.METRICS_OUT / "field_weights.parquet")
    propensity = pl.read_parquet(paths.METRICS_OUT / "decomposition_propensity.parquet")
    importance = pl.read_parquet(paths.EXP_OUT / "kiraly" / "importance_weights.parquet")
    return weights, propensity, importance


def density_norm(populated: pl.DataFrame) -> pl.DataFrame:
    """Per-institution 90th-percentile of total characters, from complete records, held fixed under dropout"""
    counts = record_sample.institution_counts()
    stats = thinness._char_stats(record_sample.typed(populated, counts, wide=True))
    return stats.group_by("data_source").agg(
        pl.col("c_total").quantile(thinness.DENSITY_PERCENTILE).alias("c_total_q90")
    )


def recompute_quality(
    base: pl.DataFrame,
    keys: pl.DataFrame,
    frozen: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
    dens: pl.DataFrame,
    verdicts: pl.DataFrame,
) -> np.ndarray:
    """The six quality measures recomputed on the fields present in `base`, aligned to keys' record order"""
    weights, propensity, importance = frozen
    counts = record_sample.institution_counts()
    typed = record_sample.typed(base, counts, wide=True)
    ours = (
        completeness.compute(typed, weights=weights)
        .join(
            thinness.compute(typed, propensity=propensity, density_norm=dens),
            on=["record_id", "data_source"],
            how="full",
            coalesce=True,
        )
        .join(
            conformance.compute(typed).select("record_id", "data_source", "conformance"),
            on=["record_id", "data_source"],
            how="full",
            coalesce=True,
        )
        .join(
            kiraly.completeness(typed, weights=importance).select(
                "record_id", "data_source", "k_completeness_weighted"
            ),
            on=["record_id", "data_source"],
            how="full",
            coalesce=True,
        )
    )
    # consistency over the record's surviving fields only, from the frozen per-field verdicts
    surviving = base.select("record_id", "field_type").unique()
    consistency = (
        verdicts.join(surviving, on=["record_id", "field_type"], how="semi")
        .group_by("record_id")
        .agg(pl.col("n_conventional_ok").sum(), pl.col("n_conventional_applicable").sum())
        .with_columns((pl.col("n_conventional_ok") / pl.col("n_conventional_applicable")).alias("consistency"))
        .select("record_id", "consistency")
    )
    # documents() emits one row per record sorted by record_id; align the vector to that order
    joined = (
        keys.select("record_id")
        .unique()
        .sort("record_id")
        .join(ours, on="record_id", how="left")
        .join(consistency, on="record_id", how="left")
    )
    return np.column_stack([joined[m].fill_null(joined[m].median() or 0.0).fill_nan(0.0).to_numpy() for m in QUALITY])


def quality_context(
    populated: pl.DataFrame, keys: pl.DataFrame
) -> tuple[tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame], pl.DataFrame, pl.DataFrame]:
    """The frozen tables, density normaliser and consistency verdicts recompute_quality needs"""
    frozen = frozen_tables()
    dens = density_norm(populated)
    verdicts = pl.read_parquet(CONSISTENCY_FIELD).join(keys.select("record_id"), on="record_id", how="semi")
    return frozen, dens, verdicts


def solver(combo: tuple[str, ...]) -> bool:
    """Whether to solve in the dual, which turns on the shape of the block rather than on preference"""
    return combo != ("qual",)


def fit(
    docs: dict[str, list[str]], qual: np.ndarray, y: np.ndarray, combos: list[tuple[str, ...]] | None = None
) -> tuple[dict, dict]:
    vecs, blocks = {}, {}
    for key in TEXT_KEYS:
        analyzer = str.split if key in ("occ", "mor", "fmo") else "word"
        vecs[key] = TfidfVectorizer(analyzer=analyzer, max_features=MAX_FEATURES[key], sublinear_tf=True, min_df=3)
        blocks[key] = vecs[key].fit_transform(docs[key])
    vecs["qual"] = StandardScaler()
    blocks["qual"] = csr_matrix(vecs["qual"].fit_transform(qual))
    models = {
        combo: LinearSVC(random_state=SEED, dual=solver(combo)).fit(hstack([blocks[k] for k in combo]).tocsr(), y)
        for combo in (combos or COMBOS)
    }
    return vecs, models


def transform(vecs: dict, docs: dict[str, list[str]], qual: np.ndarray) -> dict:
    blocks = {k: vecs[k].transform(docs[k]) for k in TEXT_KEYS}
    blocks["qual"] = csr_matrix(vecs["qual"].transform(qual))
    return blocks


def evaluate(vecs: dict, models: dict, docs: dict[str, list[str]], qual: np.ndarray, y: np.ndarray) -> dict:
    blocks = transform(vecs, docs, qual)
    return {
        combo: float(f1_score(y, models[combo].predict(hstack([blocks[k] for k in combo]).tocsr()), average="macro"))
        for combo in models
    }


def as_docs(feats: pl.DataFrame) -> dict[str, list[str]]:
    return {k: feats[k].to_list() for k in TEXT_KEYS}


def combo_name(combo: tuple[str, ...]) -> str:
    return " + ".join(NAMES[k] for k in combo)


def show(frame: pl.DataFrame, sort: str) -> None:
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=3):
        print(frame.sort(sort, descending=True))


def write(frame: pl.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT / f"{name}.parquet")
    log(f"→ {OUT / f'{name}.parquet'}")


def regime_sides() -> tuple[pl.DataFrame, list[str]]:
    """Records from each institution's earliest and latest practice stratum, capped per side"""
    strata = pl.read_parquet(paths.INSTITUTIONAL / "practice_strata.parquet")
    accession = pl.read_parquet(paths.INSTITUTIONAL / "accession_years.parquet")
    multi = (
        strata.group_by("data_source").agg(pl.len().alias("n_strata")).filter(pl.col("n_strata") > 1)["data_source"]
    )
    edges = (
        strata.filter(pl.col("data_source").is_in(multi.implode()))
        .sort("data_source", "stratum_idx")
        .group_by("data_source", maintain_order=True)
        .agg(pl.col("stratum_idx").first().alias("earliest"), pl.col("stratum_idx").last().alias("latest"))
    )
    span = strata.select(
        "data_source", "stratum_idx", pl.int_ranges("start_year", pl.col("end_year") + 1).alias("year")
    ).explode("year")
    sides = (
        accession.join(span, left_on=["data_source", "accession_year"], right_on=["data_source", "year"])
        .join(edges, on="data_source")
        .sort("record_id")
        .with_columns(
            pl.when(pl.col("stratum_idx") == pl.col("earliest"))
            .then(pl.lit("early"))
            .when(pl.col("stratum_idx") == pl.col("latest"))
            .then(pl.lit("late"))
            .otherwise(None)
            .alias("side")
        )
        .filter(pl.col("side").is_not_null())
        .sample(fraction=1.0, shuffle=True, seed=SEED)
        .with_columns(pl.int_range(pl.len()).over("data_source", "side").alias("pos"))
        .filter(pl.col("pos") < PER_SIDE)
    )
    sized = sides.group_by("data_source", "side").len().pivot(on="side", index="data_source", values="len")
    keep = sized.filter((pl.col("early") >= MIN_SIDE) & (pl.col("late") >= MIN_SIDE))["data_source"].to_list()
    total = record_sample.institution_counts().height
    log(
        f"regime population: {total} institutions in the corpus -> {accession['data_source'].n_unique()} with "
        f"accession years -> {strata['data_source'].n_unique()} with detected practice strata -> {len(multi)} "
        f"multi-stratum -> {len(keep)} with >= {MIN_SIDE} records in both the earliest and latest stratum"
    )
    return sides.filter(pl.col("data_source").is_in(keep)).select("record_id", "data_source", "side"), sorted(keep)


def regime_frames() -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    sides, insts = regime_sides()
    rows = record_sample.cached(
        "regime_rows",
        lambda: (
            pl.scan_parquet(paths.RAW_RECORDS)
            .join(sides.lazy(), on=["record_id", "data_source"], how="semi")
            .collect(engine="streaming")
        ),
    )
    populated, short = prepare(rows)
    return (
        populated.join(sides, on=["record_id", "data_source"]),
        short.join(sides, on=["record_id", "data_source"]),
        insts,
    )


def variant_regime() -> None:
    """Train on the earliest stratum and predict the latest, against a matched within-regime control"""
    populated, short, insts = regime_frames()
    log(
        f"{len(insts)} institutions, {populated['record_id'].n_unique():,} records, "
        f"{populated.height:,} statements across the earliest and latest practice stratum"
    )
    keys = populated.select("record_id", "data_source", "side").unique()
    rows = []

    # The third scope drops object_number, close to a name badge
    scopes = [("all fields", None), ("common core", CORE), ("core, no object_number", CORE[1:])]
    for scope, restrict in scopes:
        pop = populated if restrict is None else populated.filter(pl.col("field_type").is_in(restrict))
        sht = short if restrict is None else short.filter(pl.col("field_type").is_in(restrict))
        feats, qual = features(pop, sht, keys)
        y = feats["data_source"].cast(pl.String).to_numpy()
        side = feats["side"].to_numpy()
        docs = as_docs(feats)

        for train_side, test_side in [("early", "late"), ("late", "early")]:
            tr, te = np.flatnonzero(side == train_side), np.flatnonzero(side == test_side)
            vecs, models = fit({k: [docs[k][i] for i in tr] for k in TEXT_KEYS}, qual[tr], y[tr])
            scored = evaluate(vecs, models, {k: [docs[k][i] for i in te] for k in TEXT_KEYS}, qual[te], y[te])
            for combo, f1 in scored.items():
                rows.append(
                    {
                        "scope": scope,
                        "condition": f"{train_side} -> {test_side}",
                        "features": combo_name(combo),
                        "macro_f1": f1,
                        "sd": 0.0,
                    }
                )

        # Matched control: both halves drawn across the regime boundary
        control: dict[tuple[str, ...], list[float]] = {}
        rng = np.random.default_rng(SEED)
        strata = np.array([f"{a}|{b}" for a, b in zip(y, side, strict=True)])
        for _ in range(WITHIN_REPS):
            half = np.zeros(len(y), dtype=bool)
            for group in np.unique(strata):
                sel = np.flatnonzero(strata == group)
                rng.shuffle(sel)
                half[sel[: len(sel) // 2]] = True
            tr, te = np.flatnonzero(half), np.flatnonzero(~half)
            vecs, models = fit({k: [docs[k][i] for i in tr] for k in TEXT_KEYS}, qual[tr], y[tr])
            scored = evaluate(vecs, models, {k: [docs[k][i] for i in te] for k in TEXT_KEYS}, qual[te], y[te])
            for combo, f1 in scored.items():
                control.setdefault(combo, []).append(f1)
        for combo, vals in control.items():
            rows.append(
                {
                    "scope": scope,
                    "condition": "within (mixed halves)",
                    "features": combo_name(combo),
                    "macro_f1": float(np.mean(vals)),
                    "sd": float(np.std(vals)),
                }
            )

    results = pl.DataFrame(rows)
    write(results, "regime_attribution")
    wide = results.pivot(on="condition", index=["scope", "features"], values="macro_f1").with_columns(
        drop=pl.col("within (mixed halves)") - (pl.col("early -> late") + pl.col("late -> early")) / 2
    )
    for scope, _ in scopes:
        print(f"\n{scope}: macro-F1, chance = {1 / len(insts):.3f}")
        show(wide.filter(pl.col("scope") == scope).drop("scope"), "within (mixed halves)")


def profiles(populated: pl.DataFrame, short: pl.DataFrame, split: pl.DataFrame, insts: list[str]) -> dict:
    """A distribution per (institution, side) for each representation, plus the quality vector's means"""
    idx = {s: i for i, s in enumerate(insts)}
    lex = (
        populated.join(split, on=["record_id", "data_source"])
        .select("data_source", "side", token=pl.col("value").str.to_lowercase().str.extract_all(r"[a-z0-9]{2,}"))
        .explode("token")
        .drop_nulls()
    )
    pop, sht = (frame.join(split, on=["record_id", "data_source"]) for frame in (populated, short))
    sht = sht.with_columns(fs=pl.col("field_type").cast(pl.String) + "‖" + pl.col("sig"))
    frames = {"occ": (pop, "field_type"), "mor": (sht, "sig"), "fmo": (sht, "fs"), "lex": (lex, "token")}
    vocabs = {
        "occ": sorted(set(pop["field_type"].cast(pl.String))),
        "mor": sht.group_by("sig").len().sort("len", descending=True).head(2000)["sig"].to_list(),
        "fmo": sht.group_by("fs").len().sort("len", descending=True).head(5000)["fs"].to_list(),
        "lex": lex.group_by("token").len().sort("len", descending=True).head(LEX_VOCAB)["token"].to_list(),
    }

    out = {}
    for key, (frame, col) in frames.items():
        vidx = {v: i for i, v in enumerate(vocabs[key])}
        early, late = (np.zeros((len(insts), len(vocabs[key]))) for _ in range(2))
        counts = (
            frame.with_columns(pl.col(col).cast(pl.String))
            .filter(pl.col(col).is_in(vocabs[key]))
            .group_by("data_source", "side", col)
            .len(name="c")
        )
        for r in counts.iter_rows(named=True):
            if r["data_source"] in idx:
                (early if r["side"] == "early" else late)[idx[r["data_source"]], vidx[r[col]]] = r["c"]
        out[key] = tuple((m + 1e-9) / (m + 1e-9).sum(1, keepdims=True) for m in (early, late))

    # The quality vector is averaged per side, not normalised
    metrics = record_sample.record_metrics().select("record_id", *QUALITY)
    qual = (
        split.join(metrics, on="record_id", how="left")
        .group_by("data_source", "side")
        .agg([pl.col(m).mean() for m in QUALITY])
    )
    early, late = (np.zeros((len(insts), len(QUALITY))) for _ in range(2))
    for r in qual.iter_rows(named=True):
        if r["data_source"] in idx:
            (early if r["side"] == "early" else late)[idx[r["data_source"]]] = [r[m] or 0.0 for m in QUALITY]
    scaler = StandardScaler().fit(np.vstack([early, late]))
    out["qual"] = (scaler.transform(early), scaler.transform(late))
    return out


def match(profiles: dict, combo: tuple[str, ...]) -> dict:
    """Re-identification: is an institution's nearest profile on the other side its own?"""
    if combo == ("qual",) or ("qual" in combo and len(combo) == 1):
        a, b = profiles["qual"]
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    else:
        keys = [k for k in combo if k != "qual"]
        a = np.hstack([profiles[k][0] / len(keys) for k in keys])
        b = np.hstack([profiles[k][1] / len(keys) for k in keys])
        d = np.array([[jensenshannon(a[i], b[j], base=2) for j in range(len(b))] for i in range(len(a))])
    rank = (d < np.diag(d)[:, None]).sum(1) + 1
    return {
        "top1": float((d.argmin(1) == np.arange(len(d))).mean()),
        "mean_true_rank": float(rank.mean()),
        "mean_self_distance": float(np.mean(np.diag(d))),
        "mean_other_distance": float(d[~np.eye(len(d), dtype=bool)].mean()),
    }


def variant_fingerprint() -> None:
    """Institution-level re-identification across the regime boundary"""
    populated, short, insts = regime_frames()
    split = populated.select("record_id", "data_source", "side").unique()
    prof = profiles(populated, short, split, insts)
    rows = [{"features": combo_name(c)} | match(prof, c) for c in COMBOS]
    results = pl.DataFrame(rows)
    write(results, "regime_fingerprint")
    print(f"\ninstitution-level cross-regime match, {len(insts)} institutions, chance = {1 / len(insts):.3f}")
    show(results, "top1")


def variant_efficiency() -> None:
    """How many records a fingerprint needs before it re-identifies its institution from a held-out half"""
    populated, short = prepare(corpus_rows(False))
    keys = populated.select("record_id", "data_source").unique().sort("record_id")
    rng = np.random.default_rng(SEED)
    halves = keys.with_columns(
        side=pl.Series([rng.integers(0, 2) for _ in range(keys.height)]).replace_strict({0: "early", 1: "late"})
    )
    sized = halves.group_by("data_source", "side").len().pivot(on="side", index="data_source", values="len")
    insts = sorted(
        sized.filter((pl.col("early") >= FP_MIN_SIDE) & (pl.col("late") >= FP_MIN_SIDE))["data_source"].to_list()
    )
    log(f"sample efficiency over {len(insts)} institutions with >= {FP_MIN_SIDE} records a side")

    # The halves stay fixed across draws, so the qualifying institutions do too and only the draw varies
    rows = []
    for n in EFFICIENCY_SIZES:
        for draw in range(EFFICIENCY_DRAWS):
            drawn = (
                halves.filter(pl.col("data_source").is_in(insts))
                .sample(fraction=1.0, shuffle=True, seed=SEED + 1000 * draw + n)
                .with_columns(pos=pl.int_range(pl.len()).over("data_source", "side"))
                .filter(pl.col("pos") < n)
                .select("record_id", "data_source", "side")
            )
            prof = profiles(populated, short, drawn, insts)
            rows.extend(
                {"records_per_side": n, "draw": draw, "n_institutions": len(insts), "features": combo_name(combo)}
                | match(prof, combo)
                for combo in COMBOS
            )
        log(f"  {n} records a side done")

    results = pl.DataFrame(rows)
    write(results, "sample_efficiency")
    means = results.group_by("records_per_side", "features").agg(pl.col("top1").mean())
    show(means.pivot(on="records_per_side", index="features", values="top1"), "features")


def variant_partial(all_institutions: bool = False) -> None:
    """Accuracy on full records, under a growing share of withheld fields, and on the common core"""
    populated, short = prepare(corpus_rows(all_institutions))
    keys = populated.select("record_id", "data_source").unique()
    # One field ordering per record, so dropout levels nest
    order = (
        populated.select("record_id", "field_type")
        .unique()
        .with_columns(pl.struct("record_id", "field_type").hash(seed=SEED).alias("h"))
        .with_columns(
            (pl.col("h").rank("ordinal").over("record_id") - 1).alias("rank"), pl.len().over("record_id").alias("k")
        )
        .with_columns((pl.col("rank") / pl.col("k")).alias("cut"))
        .select("record_id", "field_type", "cut")
    )
    populated = populated.join(order, on=["record_id", "field_type"], how="left")
    short = short.join(order, on=["record_id", "field_type"], how="left")

    conditions: list[tuple[str, pl.Expr]] = [(f"drop {p:.0%}", pl.col("cut") < 1 - p) for p in DROPOUT]
    conditions.append(("core only", pl.col("field_type").is_in(CORE)))
    # The quality vector is recomputed on each condition's surviving fields, so it too sees the loss
    frozen, dens, verdicts = quality_context(populated, keys)

    def variant(keep: pl.Expr) -> tuple[pl.DataFrame, np.ndarray]:
        kept_pop, kept_short = populated.filter(keep), short.filter(keep)
        return documents(kept_pop, kept_short, keys), recompute_quality(kept_pop, keys, frozen, dens, verdicts)

    variants = {name: variant(keep) for name, keep in conditions}
    full, qual = variants["drop 0%"]
    y = full["data_source"].cast(pl.String).to_numpy()
    log(f"partial-record variant: {len(y):,} records, {len(set(y))} institutions, chance = {1 / len(set(y)):.3f}")
    for name, (frame, _) in variants.items():
        empty = (frame["occ"] == "").sum()
        log(f"  {name:10} mean fields kept {frame['occ'].str.split(' ').list.len().mean():5.2f}; {empty} empty")

    docs = {name: as_docs(frame) for name, (frame, _) in variants.items()}
    scores: dict[tuple[str, str], list[float]] = {}
    for fold, (tr, te) in enumerate(
        StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)
    ):
        vecs, models = fit({k: [docs["drop 0%"][k][i] for i in tr] for k in TEXT_KEYS}, qual[tr], y[tr])
        for name in docs:
            got = evaluate(
                vecs, models, {k: [docs[name][k][i] for i in te] for k in TEXT_KEYS}, variants[name][1][te], y[te]
            )
            for combo, f1 in got.items():
                scores.setdefault((name, combo_name(combo)), []).append(f1)
        log(f"  fold {fold + 1}/{FOLDS} done")

    results = pl.DataFrame(
        [
            {"condition": name, "features": feat, "macro_f1": float(np.mean(v)), "sd": float(np.std(v))}
            for (name, feat), v in scores.items()
        ]
    )
    write(results, f"partial_records{'_123' if all_institutions else ''}")
    print("\nmacro-F1 with a share of each test record's fields withheld (trained on complete records)")
    show(results.pivot(on="condition", index="features", values="macro_f1"), "drop 0%")


def predictions(model: LinearSVC, blocks: dict, combo: tuple[str, ...], record_id: np.ndarray) -> pl.DataFrame:
    """Each record's predicted institution and the gap to the runner-up, so failures can be read back"""
    scores = model.decision_function(hstack([blocks[k] for k in combo]).tocsr())
    top = np.argsort(-scores, axis=1)[:, :2]
    rows = np.arange(len(record_id))
    return pl.DataFrame(
        {
            "record_id": record_id,
            "predicted": model.classes_[top[:, 0]],
            "runner_up": model.classes_[top[:, 1]],
            "margin": scores[rows, top[:, 0]] - scores[rows, top[:, 1]],
        }
    )


def variant_small() -> None:
    """Attribution over all 123 institutions, including those the record floor drops"""
    counts = record_sample.institution_counts()
    populated, short = prepare(corpus_rows(True))
    keys = populated.select("record_id", "data_source").unique()

    frozen, dens, verdicts = quality_context(populated, keys)
    rows_out, per_class, guesses = [], [], []
    for scope, restrict in [("all fields", None), ("common core", CORE)]:
        pop = populated if restrict is None else populated.filter(pl.col("field_type").is_in(restrict))
        sht = short if restrict is None else short.filter(pl.col("field_type").is_in(restrict))
        feats = documents(pop, sht, keys)
        qual = recompute_quality(pop, keys, frozen, dens, verdicts)
        y = feats["data_source"].cast(pl.String).to_numpy()
        rid = feats["record_id"].cast(pl.String).to_numpy()
        docs = as_docs(feats)
        scores: dict[tuple[str, ...], list[float]] = {}
        by_class: dict[tuple[str, str], list[float]] = {}
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y):
            vecs, models = fit({k: [docs[k][i] for i in tr] for k in TEXT_KEYS}, qual[tr], y[tr])
            blocks = transform(vecs, {k: [docs[k][i] for i in te] for k in TEXT_KEYS}, qual[te])
            for combo in COMBOS:
                guess = predictions(models[combo], blocks, combo, rid[te])
                pred = guess["predicted"].to_numpy()
                scores.setdefault(combo, []).append(float(f1_score(y[te], pred, average="macro")))
                labels = sorted(set(y))
                for inst, f1 in zip(labels, f1_score(y[te], pred, average=None, labels=labels), strict=True):
                    by_class.setdefault((combo_name(combo), inst), []).append(float(f1))
                guesses.append(
                    guess.with_columns(
                        scope=pl.lit(scope), features=pl.lit(combo_name(combo)), data_source=pl.Series(y[te])
                    )
                )
        for combo, vals in scores.items():
            rows_out.append(
                {
                    "scope": scope,
                    "features": combo_name(combo),
                    "macro_f1": float(np.mean(vals)),
                    "sd": float(np.std(vals)),
                }
            )
        per_class.extend(
            {"scope": scope, "features": feat, "data_source": inst, "f1": float(np.mean(v))}
            for (feat, inst), v in by_class.items()
        )

    results = pl.DataFrame(rows_out)
    write(results, "all_institutions")
    guessed = pl.concat(guesses).select(
        "scope", "features", "record_id", "data_source", "predicted", "runner_up", "margin"
    )
    write(guessed, "record_predictions")
    classes = (
        pl.DataFrame(per_class)
        .join(counts, on="data_source")
        .with_columns(
            band=pl.when(pl.col("n_records") < MIN_SAMPLE_RECORDS)
            .then(pl.lit("< 300 records"))
            .otherwise(pl.lit(">= 300 records"))
        )
    )
    write(classes, "per_institution_f1")
    n_inst = classes["data_source"].n_unique()
    print(f"\nmacro-F1 over all {n_inst} institutions, chance = {1 / n_inst:.4f}")
    show(results.pivot(on="scope", index="features", values="macro_f1"), "all fields")

    for feature in ("lexical", combo_name(COMBOS[-1])):
        wrong = guessed.filter(
            (pl.col("scope") == "common core")
            & (pl.col("features") == feature)
            & (pl.col("predicted") != pl.col("data_source"))
        )
        print(f"\ncommon core, {feature}: {wrong.height:,} misattributed records, most confused pairs")
        show(wrong.group_by("data_source", "predicted").len().sort("len", descending=True).head(15), "len")


def single_field(short: pl.DataFrame, populated: pl.DataFrame, keys: pl.DataFrame, field: str) -> list[dict]:
    """Attribution from one field alone, under each representation that can read it"""
    pop = populated.filter(pl.col("field_type") == field)
    sht = short.filter(pl.col("field_type") == field)
    if pop.is_empty():
        return []
    feats, qual = features(pop, sht, keys)
    y = feats["data_source"].cast(pl.String).to_numpy()
    docs = as_docs(feats)
    # One field reads only two ways: shapes or words
    wanted = [("mor",), ("lex",)]
    rows = []
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y):
        vecs, models = fit({k: [docs[k][i] for i in tr] for k in TEXT_KEYS}, qual[tr], y[tr], combos=wanted)
        blocks = transform(vecs, {k: [docs[k][i] for i in te] for k in TEXT_KEYS}, qual[te])
        for combo in wanted:
            pred = models[combo].predict(hstack([blocks[k] for k in combo]).tocsr())
            rows.append(
                {
                    "field": field,
                    "features": combo_name(combo),
                    "macro_f1": float(f1_score(y[te], pred, average="macro")),
                    "n_records": len(y),
                }
            )
    return rows


def variant_fields() -> None:
    """Each field's attribution on its own, with object name as the control, and the core without object number"""
    populated, short = prepare(corpus_rows(False))
    keys = populated.select("record_id", "data_source").unique()
    counts = populated.group_by("field_type").agg(pl.col("data_source").n_unique().alias("k"), pl.len().alias("n"))
    fields = (
        counts.filter((pl.col("k") >= VARIANT_FIELD_MIN_INSTITUTIONS) & (pl.col("n") >= VARIANT_FIELD_MIN_OCCURRENCES))
        .sort("n", descending=True)
        .head(20)["field_type"]
        .cast(pl.String)
        .to_list()
    )
    # Object name is the control: masking removes its distinctiveness
    for field in CORE:
        if field not in fields:
            fields.append(field)
    log(f"single-field attribution over {len(fields)} fields")

    rows = []
    for field in fields:
        rows.extend(single_field(short, populated, keys, field))
        log(f"  {field} done")
    per_field = (
        pl.DataFrame(rows)
        .group_by("field", "features")
        .agg(macro_f1=pl.col("macro_f1").mean(), sd=pl.col("macro_f1").std(), n_records=pl.col("n_records").first())
    )
    write(per_field, "single_field")
    show(per_field.pivot(on="features", index="field", values="macro_f1"), "field")

    # The core with and without object_number, testing the numbering scheme
    frozen, dens, verdicts = quality_context(populated, keys)
    scope_rows = []
    for scope, restrict in [("common core", CORE), ("core, no object_number", CORE[1:])]:
        pop = populated.filter(pl.col("field_type").is_in(restrict))
        sht = short.filter(pl.col("field_type").is_in(restrict))
        feats = documents(pop, sht, keys)
        qual = recompute_quality(pop, keys, frozen, dens, verdicts)
        y = feats["data_source"].cast(pl.String).to_numpy()
        docs = as_docs(feats)
        scores: dict[tuple[str, ...], list[float]] = {}
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y):
            vecs, models = fit({k: [docs[k][i] for i in tr] for k in TEXT_KEYS}, qual[tr], y[tr])
            got = evaluate(vecs, models, {k: [docs[k][i] for i in te] for k in TEXT_KEYS}, qual[te], y[te])
            for combo, f1 in got.items():
                scores.setdefault(combo, []).append(f1)
        scope_rows.extend(
            {"scope": scope, "features": combo_name(c), "macro_f1": float(np.mean(v)), "sd": float(np.std(v))}
            for c, v in scores.items()
        )
    scopes = pl.DataFrame(scope_rows)
    write(scopes, "core_scopes")
    show(scopes.pivot(on="scope", index="features", values="macro_f1"), "features")


def variant_reliance() -> None:
    """Per (institution, field) classifier weight, set against the model-free divergence ranking"""
    populated, short = prepare(corpus_rows(False))
    keys = populated.select("record_id", "data_source").unique()
    feats, _ = features(populated, short, keys)
    y = feats["data_source"].cast(pl.String).to_numpy()
    docs = feats["fmo"].to_list()

    vec = TfidfVectorizer(analyzer=str.split, max_features=MAX_FEATURES["fmo"], sublinear_tf=True, min_df=3)
    x = vec.fit_transform(docs)
    model = LinearSVC(random_state=SEED, dual=True).fit(x, y)
    terms = np.array(vec.get_feature_names_out())
    fields = np.array([t.split("‖")[0] for t in terms])
    patterns = np.array([t.split("‖")[-1] for t in terms])

    rows = []
    for i, inst in enumerate(model.classes_):
        w = model.coef_[i]
        top = np.argsort(-w)[:20]
        rows.extend(
            {
                "data_source": str(inst),
                "field_type": str(fields[j]),
                "merged_pattern": str(patterns[j]),
                "weight": float(w[j]),
            }
            for j in top
        )
    reliance = pl.DataFrame(rows)
    write(reliance, "per_institution_reliance")

    # The same cells by classifier weight and by divergence
    if DIVERGENCE.exists():
        divergence = (
            pl.read_parquet(DIVERGENCE)
            .group_by("data_source", "merged_pattern")
            .agg(effect=pl.col("effect").max(), significant=pl.col("significant").any())
        )
        joined = reliance.join(divergence, on=["data_source", "merged_pattern"], how="inner")
        write(joined, "reliance_against_divergence")
        rho, r = spearmanr(joined["weight"], joined["effect"])[0], pearsonr(joined["weight"], joined["effect"])[0]
        write(pl.DataFrame({"cells": [joined.height], "spearman": [rho], "pearson": [r]}), "reliance_correlation")
        log(f"{joined.height} cells carry a classifier weight and a divergence effect; rho = {rho:.2f}, r = {r:.2f}")
    show(reliance.group_by("field_type").agg(weight=pl.col("weight").mean(), n=pl.len()), "weight")


VARIANTS = {
    "regime": variant_regime,
    "fingerprint": variant_fingerprint,
    "efficiency": variant_efficiency,
    "partial": variant_partial,
    "small": variant_small,
    "fields": variant_fields,
    "reliance": variant_reliance,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Which representation of a record identifies its institution.")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    ap.add_argument("--all-institutions", action="store_true", help="include the institutions below the record floor")
    args = ap.parse_args()
    for variant in args.variant or sorted(VARIANTS):
        print(f"\n{'=' * 30} {variant}")
        if variant == "partial":
            variant_partial(args.all_institutions)
        else:
            VARIANTS[variant]()


if __name__ == "__main__":
    main()
