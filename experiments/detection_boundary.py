from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from experiments.harness import log
from experiments.metric_informativeness import METRICS, design
from mds_norm import paths
from mds_norm.metrics import completeness as m_completeness
from mds_norm.metrics import conformance as m_conformance
from mds_norm.metrics import kiraly as m_kiraly
from mds_norm.metrics import thinness as m_thinness

EXP = "detection_boundary"
OUT = paths.EXP_OUT / EXP
SEED = 20260806
MIN_SCORABLE = 30  # below this a per-measure AUC says nothing
N_CLASSES = 2
FOREST = {"n_estimators": 300, "n_jobs": -1, "random_state": SEED}


RECALL_SAMPLE = paths.GOLD / "samples" / "recall.jsonl"
RECALL_LABELS = paths.GOLD_LABELS / "recall.jsonl"
KIRALY_OUT = paths.EXP_OUT / "kiraly"


def recall_targets() -> pl.DataFrame:
    """The audited records and whether a reader found anything wrong with them"""
    sample = {
        json.loads(line)["id"]: json.loads(line)["record_id"]
        for line in RECALL_SAMPLE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    latest = {}
    for line in RECALL_LABELS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["id"]] = row
    rows = [
        {
            "record_id": sample[i],
            "has_problem": bool((row.get("gold") or {}).get("classes")),
            "classes": (row.get("gold") or {}).get("classes") or [],
        }
        for i, row in latest.items()
        if row.get("status") == "labelled" and i in sample
    ]
    return pl.DataFrame(rows)


def metrics_for(record_ids: list[str]) -> pl.DataFrame:
    """Every per-record measure, computed for an arbitrary set of records against the frozen corpus tables"""
    weights = pl.read_parquet(paths.METRICS_OUT / "field_weights.parquet")
    propensity = pl.read_parquet(paths.METRICS_OUT / "decomposition_propensity.parquet")
    importance = pl.read_parquet(KIRALY_OUT / "importance_weights.parquet")

    corpus = pl.scan_parquet(paths.RAW_RECORDS)
    rows = corpus.filter(pl.col("record_id").is_in(record_ids)).collect(engine="streaming")
    log(f"  {rows.height:,} statements over {rows['record_id'].n_unique():,} audited records")

    fields = sorted(
        set(rows["field_type"].cast(pl.String)) | set(weights["field_type"]) | set(propensity["field_type"])
    )
    # The Enum must cover institutions the frozen weights also name
    sources = sorted(set(rows["data_source"].cast(pl.String)) | set(weights["data_source"]))
    base = rows.lazy().with_columns(
        pl.col("field_type").cast(pl.String).cast(pl.Enum(fields)),
        pl.col("data_source").cast(pl.String).cast(pl.Enum(sources)),
    )

    log("  corpus pass for the value frequencies these records need")
    # The corpus carries field_type as a plain string
    frequencies = m_kiraly.value_frequencies(
        corpus, restrict_to=base.with_columns(pl.col("field_type").cast(pl.String))
    )

    ours = (
        m_completeness.compute(base, weights=weights)
        .join(
            m_thinness.compute(base, propensity=propensity), on=["record_id", "data_source"], how="full", coalesce=True
        )
        .join(
            m_conformance.compute(base).select("record_id", "data_source", "conformance"),
            on=["record_id", "data_source"],
            how="full",
            coalesce=True,
        )
        .join(
            pl.scan_parquet(paths.CONSISTENCY_OUT / "consistency_record_raw.parquet")
            .filter(pl.col("record_id").is_in(record_ids))
            .select("record_id", "consistency")
            .collect(engine="streaming"),
            on="record_id",
            how="left",
        )
    )
    idf = pl.read_parquet(KIRALY_OUT / "idf.parquet")
    theirs = (
        m_kiraly.completeness(base, weights=importance)
        .join(
            m_kiraly.conformance_to_expectations(base, frequencies=frequencies),
            on=["record_id", "data_source"],
            how="full",
            coalesce=True,
        )
        .join(m_kiraly.coherence(base, idf=idf).drop("data_source"), on="record_id", how="left")
    )
    return ours.join(theirs, on=["record_id", "data_source"], how="full", coalesce=True).with_columns(
        pl.col("data_source").cast(pl.String)
    )


def detection(frame: pl.DataFrame) -> dict:
    """Whether each measure separates the records the audit faulted from the ones it passed"""
    y = frame["has_problem"].to_numpy().astype(int)
    per_metric = {}
    for m in METRICS:
        sub = frame.select(m, "has_problem").drop_nulls()
        yy = sub["has_problem"].to_numpy().astype(int)
        if sub.height < MIN_SCORABLE or len(set(yy)) < N_CLASSES:
            per_metric[m] = {"n": sub.height, "auc": None}
            continue
        auc = float(roc_auc_score(yy, sub[m].to_numpy()))
        # Direction is not part of the claim, only separation
        per_metric[m] = {"n": sub.height, "auc": round(auc, 4), "auc_directionless": round(max(auc, 1 - auc), 4)}
        log(f"  {m}: n={sub.height}, AUC {auc:.4f}")

    x, _ = design(frame)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def cv_auc(features: np.ndarray) -> tuple[float, float]:
        scores = cross_val_score(RandomForestClassifier(**FOREST), features, y, cv=folds, scoring="roc_auc")
        return float(scores.mean()), float(scores.std())

    joint, joint_sd = cv_auc(x)
    log(f"  all measures together: AUC {joint:.4f} (sd {joint_sd:.4f})")

    # Control for institution identity, whose fault rates differ
    codes = {s: i for i, s in enumerate(sorted(set(frame["data_source"].to_list())))}
    onehot = np.zeros((frame.height, len(codes)))
    onehot[np.arange(frame.height), [codes[s] for s in frame["data_source"].to_list()]] = 1
    inst, inst_sd = cv_auc(onehot)
    log(f"  institution identity alone: AUC {inst:.4f} (sd {inst_sd:.4f}) over {len(codes)} institutions")

    both, both_sd = cv_auc(np.hstack([x, onehot]))
    log(f"  institution identity plus the measures: AUC {both:.4f} (sd {both_sd:.4f})")

    size = frame["n_categorical_values"].fill_null(0).to_numpy().astype(float)
    size_auc = float(roc_auc_score(y, size))
    log(f"  record size alone: AUC {size_auc:.4f}")

    return {
        "n_records": frame.height,
        "n_faulted": int(y.sum()),
        "base_rate": round(float(y.mean()), 4),
        "n_institutions": len(codes),
        "per_metric": per_metric,
        "all_measures_auc": round(joint, 4),
        "all_measures_auc_sd": round(joint_sd, 4),
        "institution_identity_auc": round(inst, 4),
        "institution_identity_auc_sd": round(inst_sd, 4),
        "institution_plus_measures_auc": round(both, 4),
        "measures_over_institution": round(both - inst, 4),
        "record_size_auc": round(size_auc, 4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = recall_targets()
    log(f"{targets.height} audited records, {int(targets['has_problem'].sum())} faulted")
    audited = metrics_for(targets["record_id"].to_list()).join(targets, on="record_id", how="inner")
    audited.write_parquet(OUT / "audited_record_metrics.parquet")
    report = detection(audited)
    (OUT / "detection_boundary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"→ {OUT / 'detection_boundary.json'}")


if __name__ == "__main__":
    main()
