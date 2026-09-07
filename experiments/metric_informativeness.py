from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split

from experiments import record_sample
from experiments.harness import log
from mds_norm import paths

EXP = "metric_informativeness"
OUT = paths.EXP_OUT / EXP
SEED = record_sample.SEED  # the seed the metrics sample was drawn under

# The reported measures, with Kiraly's weighted completeness as the baseline
METRICS = ["completeness", "thinness", "decomposition_rate", "conformance", "consistency", "k_completeness_weighted"]
N_BINS = 20  # fixed-width bins for the concentration reading
MIN_SCORABLE = 30  # below this a per-measure AUC says nothing
MIN_PAIRS = 3
N_CLASSES = 2
FOREST = {"n_estimators": 300, "n_jobs": -1, "random_state": SEED}


def spread(frame: pl.DataFrame) -> list[dict]:
    """How much of its own range each measure actually uses, and on how many records it is defined"""
    n = frame.height
    rows = []
    for m in METRICS:
        col = frame[m].drop_nulls()
        lo, hi = col.min(), col.max()
        # A near-constant measure separates nothing, however wide its range
        counts = np.histogram(col.to_numpy(), bins=N_BINS, range=(lo, hi))[0]
        p = counts[counts > 0] / counts.sum()
        rows.append(
            {
                "metric": m,
                "defined_on": col.len(),
                "defined_share": round(col.len() / n, 4),
                "median": round(float(col.median()), 4),
                "iqr": round(float(col.quantile(0.75) - col.quantile(0.25)), 4),
                "sd": round(float(col.std()), 4),
                "modal_bin_share": round(float(counts.max() / counts.sum()), 4),
                "bin_entropy": round(float(-(p * np.log(p)).sum() / np.log(N_BINS)), 4),
            }
        )
    return rows


def between_institution(frame: pl.DataFrame) -> dict[str, float]:
    """Share of each measure's variance that lies between institutions rather than within them"""
    out = {}
    for m in METRICS:
        sub = frame.select("data_source", m).drop_nulls()
        grand = sub[m].mean()
        groups = sub.group_by("data_source").agg(n=pl.len(), mean=pl.col(m).mean())
        between = float((groups["n"] * (groups["mean"] - grand) ** 2).sum())
        total = float(((sub[m] - grand) ** 2).sum())
        out[m] = round(between / total, 4) if total else 0.0
    return out


def design(frame: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Feature matrix with medians filled in and a column recording where a measure was undefined"""
    cols, names = [], []
    for m in METRICS:
        series = frame[m]
        cols.append(series.fill_null(series.median()).to_numpy())
        names.append(m)
        if series.null_count():
            # Undefined is itself a fact about the record
            cols.append(series.is_null().cast(pl.Int8).to_numpy())
            names.append(f"{m}__undefined")
    return np.column_stack(cols), names


def redundancy(x: np.ndarray, names: list[str]) -> dict[str, float]:
    """How well the remaining measures reproduce each one; a measure the others predict adds nothing"""
    out = {}
    for m in METRICS:
        keep = [i for i, n in enumerate(names) if n != m and not n.startswith(f"{m}__")]
        target = names.index(m)
        x_tr, x_te, y_tr, y_te = train_test_split(x[:, keep], x[:, target], test_size=0.3, random_state=SEED)
        model = RandomForestRegressor(**FOREST).fit(x_tr, y_tr)
        out[m] = round(float(r2_score(y_te, model.predict(x_te))), 4)
        log(f"  redundancy {m}: R2 {out[m]}")
    return out


def attribution(x: np.ndarray, names: list[str], y: np.ndarray) -> dict:
    """Institution attribution from the measures alone, and what each measure contributes to it"""
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)
    full = RandomForestClassifier(**FOREST).fit(x_tr, y_tr)
    pred = full.predict(x_te)
    base = {
        "accuracy": round(float((pred == y_te).mean()), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
    }
    log(f"  all measures: accuracy {base['accuracy']}, macro F1 {base['macro_f1']}")
    majority = float((y_te == np.bincount(y_tr).argmax()).mean())

    per_metric = {}
    for m in METRICS:
        idx = [i for i, n in enumerate(names) if n == m or n.startswith(f"{m}__")]
        keep = [i for i in range(len(names)) if i not in idx]

        alone = RandomForestClassifier(**FOREST).fit(x_tr[:, idx], y_tr)
        without = RandomForestClassifier(**FOREST).fit(x_tr[:, keep], y_tr)
        a = float((alone.predict(x_te[:, idx]) == y_te).mean())
        w = float((without.predict(x_te[:, keep]) == y_te).mean())
        per_metric[m] = {
            "alone_accuracy": round(a, 4),
            "without_accuracy": round(w, 4),
            "unique_contribution": round(base["accuracy"] - w, 4),
        }
        log(f"  {m}: alone {a:.4f}, without {w:.4f}, unique {base['accuracy'] - w:+.4f}")
    return {"all_measures": base, "majority_class": round(majority, 4), "per_metric": per_metric}


DELTA_AXES = ["completeness", "thinness", "decomposition", "conformance", "consistency"]
QUALITY_DELTAS = paths.ANALYSIS_OUTPUT / "evaluation" / "quality_deltas_per_institution.parquet"


def delta_redundancy() -> dict:
    """Whether the before/after axes move together, which is the question redundancy in levels raises"""
    frame = pl.read_parquet(QUALITY_DELTAS)
    out = {"n_institutions": frame.height, "deltas": {}, "levels": {}}
    for kind, suffix in (("deltas", "_delta"), ("levels", "_before")):
        for i, a in enumerate(DELTA_AXES):
            for b in DELTA_AXES[i + 1 :]:
                x, y = frame[f"{a}{suffix}"].to_numpy(), frame[f"{b}{suffix}"].to_numpy()
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > MIN_PAIRS:
                    out[kind][f"{a}|{b}"] = round(float(spearmanr(x[mask], y[mask]).statistic), 4)
    strongest = max(out["deltas"].items(), key=lambda kv: abs(kv[1]))
    log(f"  strongest delta pair: {strongest[0]} at {strongest[1]}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = record_sample.record_metrics()
    log(f"{frame.height:,} records over {frame['data_source'].n_unique()} institutions")

    log("spread and definedness")
    spread_rows = spread(frame)

    log("between-institution variance")
    eta = between_institution(frame)

    x, names = design(frame)
    y = frame["data_source"].to_physical().to_numpy() if frame["data_source"].dtype == pl.Categorical else None
    if y is None:
        codes = {s: i for i, s in enumerate(sorted(set(frame["data_source"].to_list())))}
        y = np.array([codes[s] for s in frame["data_source"].to_list()])

    log("redundancy against the other measures")
    red = redundancy(x, names)

    log("institution attribution")
    attr = attribution(x, names, y)

    log("do the evaluation's before/after axes move together?")
    report = {
        "n_records": frame.height,
        "n_institutions": int(frame["data_source"].n_unique()),
        "spread": spread_rows,
        "between_institution_variance_share": eta,
        "redundancy_r2": red,
        "attribution": attr,
        "evaluation_axis_correlation": delta_redundancy(),
    }
    (OUT / "metric_informativeness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    table = pl.DataFrame(spread_rows).join(
        pl.DataFrame(
            {
                "metric": METRICS,
                "between_institution": [eta[m] for m in METRICS],
                "redundancy_r2": [red[m] for m in METRICS],
                "alone_accuracy": [attr["per_metric"][m]["alone_accuracy"] for m in METRICS],
                "unique_contribution": [attr["per_metric"][m]["unique_contribution"] for m in METRICS],
            }
        ),
        on="metric",
    )
    table.write_parquet(OUT / "metric_informativeness.parquet")
    log(f"\n{table}")
    log(f"→ {OUT / 'metric_informativeness.json'}")


if __name__ == "__main__":
    main()
