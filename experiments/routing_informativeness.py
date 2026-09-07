from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from experiments.harness import log
from mds_norm import paths
from mds_norm.metrics.common import ADMIN

EXP = "routing_informativeness"
OUT = paths.EXP_OUT / EXP
COMPILED = paths.COMPILED / "mds-normalised.parquet"
PATCHES = paths.ANALYSIS_OUTPUT / "record_fixes" / "record_patches.parquet"
VOCAB_DECISIONS = paths.ANALYSIS_OUTPUT / "vocabularies" / "vocab_value_decisions.parquet"
SEED = 20260710

SAMPLE_ONE_IN = 100  # ~1.5M values, enough for the smallest rung
MAX_LEVELS = 240  # a categorical the histogram learner can bin
STRING_TIER = 2

# Which tier each stage sits at; split stages resolve below
COMPONENT_TIER = {
    "tier0": 0,
    "dates": 1,
    "dimensions": 1,
    "counts": 1,
    "certainty_notation": 1,
    "identifier_certainty": 1,
    "person_dates": 1,
    "associated_dates": 1,
    "object_number_transfer": 1,
    "admin_note_routing": 1,
    "cataloguer": 1,
    "places": 2,
    "places_fallback": 2,
    "agents": 2,
    "authorities": 2,
}
SPLIT_COMPONENTS = {"vocab_alignment", "record_fixes"}

# What the pipeline knows before routing, grouped by source
GROUPS = {
    "field": ["field_type", "depth"],
    "morphology": ["n_chars", "n_tokens", "digit_share", "alpha_share", "punct_share", "n_delims", "shape"],
    "institution": ["data_source"],
    "priors": ["dm_order", "eq_means_range", "zero_placeholder", "prime_unit", "n_strata"],
    "record": ["record_fields", "record_chars"],
}
CATEGORICAL = {"field_type", "data_source", "shape", "dm_order", "prime_unit"}


def sample_values() -> pl.DataFrame:
    """A deterministic one-in-N sample of the populated Spectrum values the pipeline was given"""
    raw = pl.scan_parquet(paths.RAW_RECORDS).filter(~ADMIN & pl.col("value").is_not_null())
    picked = raw.filter(pl.col("node_id").hash(seed=SEED) % SAMPLE_ONE_IN == 0).select(
        "node_id", "record_id", "data_source", "field_type", "depth", "value"
    )
    out = picked.collect(engine="streaming")
    log(f"sampled {out.height:,} values (one in {SAMPLE_ONE_IN})")
    return out


def record_context(record_ids: pl.Series) -> pl.DataFrame:
    """How much the record around a value holds, which routing could read without touching the value"""
    return (
        pl.scan_parquet(paths.RAW_RECORDS)
        .filter(~ADMIN & pl.col("value").is_not_null())
        .join(pl.LazyFrame({"record_id": record_ids}), on="record_id", how="semi")
        .group_by("record_id")
        .agg(
            record_fields=pl.col("field_type").n_unique().cast(pl.Int32),
            record_chars=pl.col("value").str.len_chars().sum().cast(pl.Int64),
        )
        .collect(engine="streaming")
    )


def outcomes(node_ids: pl.Series) -> pl.DataFrame:
    """The stage and disposition each sampled value ended up with"""
    return (
        pl.scan_parquet(COMPILED)
        .join(pl.LazyFrame({"node_id": node_ids}), on="node_id", how="semi")
        .select("node_id", "component", "disposition")
        .collect(engine="streaming")
    )


def tier_of(frame: pl.DataFrame) -> pl.DataFrame:
    """Label each value with whether a stage decided it, and which rung of the ladder rewrote it"""
    # vocab_alignment runs both a string cascade and a model rung
    vocab = (
        pl.scan_parquet(VOCAB_DECISIONS)
        .filter(pl.col("status") == "resolved")
        .group_by("data_source", "value")
        .agg(vocab_tier=pl.col("tier").max())
        .collect(engine="streaming")
        .with_columns(pl.col("data_source").cast(pl.String))
    )
    # record_fixes spans mechanical repair and model extraction
    patches = (
        pl.scan_parquet(PATCHES)
        .filter(pl.col("status") == "resolved")
        .group_by("node_id")
        .agg(fix_llm=(pl.col("sub_component") == "llm").any())
        .collect(engine="streaming")
    )
    return (
        frame.join(vocab, on=["data_source", "value"], how="left")
        .join(patches, on="node_id", how="left")
        .with_columns(
            decided=(pl.col("disposition") != "untouched").cast(pl.Int8),
            tier=pl.when(pl.col("component") == "vocab_alignment")
            .then(pl.col("vocab_tier").cast(pl.Int32))
            .when(pl.col("component") == "record_fixes")
            .then(pl.when(pl.col("fix_llm").fill_null(False)).then(5).otherwise(1).cast(pl.Int32))
            .otherwise(pl.col("component").replace_strict(COMPONENT_TIER, default=None, return_dtype=pl.Int32)),
        )
        .with_columns(
            route=pl.when(pl.col("tier").is_null())
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("tier") <= 1)
            .then(pl.lit("deterministic"))
            .when(pl.col("tier") == STRING_TIER)
            .then(pl.lit("string"))
            .otherwise(pl.lit("model"))
        )
    )


_SHAPE = [
    (r"^\d+$", "digits"),
    (r"^[A-Za-z]+$", "letters"),
    (r"^[\d\s./-]+$", "numeric_punct"),
    (r"^[A-Za-z\s]+$", "words"),
    (r"[A-Za-z]", "mixed"),
]


def features(frame: pl.DataFrame) -> pl.DataFrame:
    """The signals available before any stage has touched the value"""
    conventions = pl.read_parquet(paths.INSTITUTIONAL / "date_conventions.parquet").select(
        "data_source", "dm_order", "eq_means_range", "zero_placeholder"
    )
    units = pl.read_parquet(paths.INSTITUTIONAL / "unit_conventions.parquet").select("data_source", "prime_unit")
    strata = (
        pl.read_parquet(paths.INSTITUTIONAL / "practice_strata.parquet")
        .group_by("data_source")
        .agg(n_strata=pl.len().cast(pl.Int32))
    )
    shape = pl.lit("other", dtype=pl.String)
    for pattern, name in reversed(_SHAPE):
        shape = pl.when(pl.col("value").str.contains(pattern)).then(pl.lit(name)).otherwise(shape)
    return (
        frame.with_columns(
            n_chars=pl.col("value").str.len_chars().cast(pl.Int32),
            n_tokens=pl.col("value").str.split(" ").list.len().cast(pl.Int32),
            digit_share=pl.col("value").str.count_matches(r"\d") / pl.col("value").str.len_chars(),
            alpha_share=pl.col("value").str.count_matches(r"[A-Za-z]") / pl.col("value").str.len_chars(),
            punct_share=pl.col("value").str.count_matches(r"[^\w\s]") / pl.col("value").str.len_chars(),
            n_delims=pl.col("value").str.count_matches(r"[;,|/]").cast(pl.Int32),
            shape=shape,
            depth=pl.col("depth").cast(pl.Int32),
        )
        .join(conventions, on="data_source", how="left")
        .join(units, on="data_source", how="left")
        .join(strata, on="data_source", how="left")
        .with_columns(
            dm_order=pl.col("dm_order").fill_null("none"),
            prime_unit=pl.col("prime_unit").fill_null("none"),
            eq_means_range=pl.col("eq_means_range").fill_null(False).cast(pl.Int8),
            zero_placeholder=pl.col("zero_placeholder").fill_null(False).cast(pl.Int8),
            n_strata=pl.col("n_strata").fill_null(0),
        )
    )


def encode(frame: pl.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[int]]:
    """Numeric matrix plus the indices a histogram learner should treat as categorical"""
    cols, cat_idx = [], []
    for i, name in enumerate(columns):
        series = frame[name]
        if name in CATEGORICAL:
            top = series.value_counts(sort=True).head(MAX_LEVELS)[name].to_list()
            codes = {level: j for j, level in enumerate(top)}
            cols.append(np.array([codes.get(v, len(codes)) for v in series.to_list()], dtype=float))
            cat_idx.append(i)
        else:
            cols.append(series.cast(pl.Float64).fill_null(-1.0).to_numpy())
    return np.column_stack(cols), cat_idx


def fit_score(frame: pl.DataFrame, columns: list[str], y: np.ndarray) -> dict:
    """Macro F1 of predicting the rung that resolved a value from the given signals"""
    x, cat_idx = encode(frame, columns)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)
    model = HistGradientBoostingClassifier(
        categorical_features=cat_idx, random_state=SEED, max_iter=200, early_stopping=True
    )
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    labels = sorted(set(y_te.tolist()))
    return {
        "accuracy": round(float((pred == y_te).mean()), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "per_class_f1": [round(float(v), 4) for v in f1_score(y_te, pred, average=None, labels=labels)],
        "test_support": [int((y_te == c).sum()) for c in labels],
    }


def ablate(frame: pl.DataFrame, y: np.ndarray, label: str) -> dict:
    """Fit on every signal, then price each group by what its removal costs"""
    every = [c for cols in GROUPS.values() for c in cols]
    full = fit_score(frame, every, y)
    majority = float(np.bincount(y).max() / len(y))
    log(f"{label}: accuracy {full['accuracy']}, macro F1 {full['macro_f1']} (majority {majority:.4f})")

    per_group = {}
    for name, cols in GROUPS.items():
        alone = fit_score(frame, cols, y)
        without = fit_score(frame, [c for c in every if c not in cols], y)
        per_group[name] = {
            "columns": cols,
            "alone": alone,
            "without": without,
            "unique_contribution_f1": round(full["macro_f1"] - without["macro_f1"], 4),
        }
        log(
            f"  {name}: alone F1 {alone['macro_f1']}, without {without['macro_f1']}, "
            f"unique {full['macro_f1'] - without['macro_f1']:+.4f}"
        )
    return {"n": len(y), "majority_class": round(majority, 4), "all_signals": full, "per_group": per_group}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = sample_values()
    log("joining the outcome each value reached")
    resolved = sample.join(outcomes(sample["node_id"]), on="node_id", how="inner")
    log(f"  {resolved.height:,} of the sample survive into the release")
    log("record context")
    resolved = resolved.join(record_context(resolved["record_id"]), on="record_id", how="left")
    frame = features(tier_of(resolved))

    log(f"\n{frame.group_by('disposition').len(name='n').sort('n', descending=True)}")
    log(f"{frame.group_by('route').len(name='n').sort('n', descending=True)}")

    log("\n— will a stage decide this value at all?")
    decided = ablate(frame, frame["decided"].to_numpy().astype(int), "decided vs untouched")

    rewritten = frame.filter(pl.col("route").is_not_null())
    codes = {r: i for i, r in enumerate(sorted(set(rewritten["route"].to_list())))}
    log(f"\n— which rung rewrote it? ({rewritten.height:,} values a stage rewrote)")
    which = ablate(rewritten, np.array([codes[r] for r in rewritten["route"].to_list()]), "rung that rewrote")
    which["classes"] = sorted(codes, key=codes.get)
    signals = which["all_signals"]
    log(f"  classes {which['classes']}: F1 {signals['per_class_f1']} on {signals['test_support']}")

    report = {
        "n_values": frame.height,
        "sample_one_in": SAMPLE_ONE_IN,
        "disposition_counts": {str(d): int(n) for d, n in frame.group_by("disposition").len(name="n").iter_rows()},
        "route_counts": {str(r): int(n) for r, n in frame.group_by("route").len(name="n").iter_rows()},
        "decided_at_all": decided,
        "which_rung": which,
    }
    (OUT / "routing_informativeness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    every = [c for cols in GROUPS.values() for c in cols]
    frame.select("node_id", "decided", "route", *every).write_parquet(OUT / "routing_sample.parquet")
    log(f"→ {OUT / 'routing_informativeness.json'}")


if __name__ == "__main__":
    main()
