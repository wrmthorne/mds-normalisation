from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from codecarbon import EmissionsTracker

from mds_norm import paths

ROOT = paths.ROOT
EXP_OUT = paths.EXP_OUT
EMISSIONS = EXP_OUT / "emissions_logs"
SAMPLES = paths.GOLD_SAMPLES
CASES = paths.GOLD_CASES

SEED = 20260710  # the seed every gold draw uses


@dataclass(frozen=True)
class Variant:
    """One experimental configuration; `model=None` means no endpoint is involved"""

    name: str
    model: str | None = None
    base_url: str = "http://localhost:30000/v1"
    concurrency: int = 100
    decoding: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    notes: str = ""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


@contextmanager
def measure(exp: str, variant: str) -> Iterator[dict]:
    """Codecarbon around one configuration run"""
    EMISSIONS.mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(
        project_name=f"exp:{exp}:{variant}", output_dir=str(EMISSIONS), log_level="error", tracking_mode="machine"
    )
    cost: dict = {}
    tracker.start()
    try:
        yield cost
    finally:
        emissions_kg = tracker.stop() or 0.0
        data = tracker.final_emissions_data
        cost["energy_wh"] = data.energy_consumed * 1e3
        cost["co2_g"] = emissions_kg * 1e3
        cost["duration_s"] = data.duration


def write_run(exp: str, variant: Variant, predictions: pl.DataFrame, metrics: dict) -> Path:
    """Persist a variant's predictions and journal the run configuration"""
    out = EXP_OUT / exp
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / f"{variant.name.replace(':', '_')}_predictions.parquet"
    predictions.write_parquet(pred_path)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exp": exp,
        "variant": variant.name,
        "model": variant.model,
        "base_url": variant.base_url if variant.model else None,
        "decoding": variant.decoding,
        "params": variant.params,
        "seed": SEED,
        "git": git_sha(),
        "n_predictions": predictions.height,
        **metrics,
    }
    with (out / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log(f"{exp}/{variant.name}: {predictions.height} predictions → {pred_path.name}; run journalled")
    return pred_path


def load_predictions(exp: str, variant_name: str) -> pl.DataFrame | None:
    p = EXP_OUT / exp / f"{variant_name.replace(':', '_')}_predictions.parquet"
    return pl.read_parquet(p) if p.exists() else None


def sample_items(task: str, limit: int | None = None) -> list[dict]:
    """The frozen gold sample the runners predict over (drawn by sample_gold.py)"""
    p = SAMPLES / f"{task}.jsonl"
    if not p.exists():
        raise SystemExit(f"no sample at {p} — draw the {task} sample first")
    items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    return items[:limit] if limit else items


def gold_cases(task: str) -> dict[str, dict]:
    """Labelled gold keyed by item id; unlabelled items are simply absent"""
    p = CASES / f"gold_{task}.json"
    if not p.exists():
        raise SystemExit(f"no gold at {p} — label the sample and export its cases first")
    table = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for case in table["cases"]:
        if "id" not in case:
            raise SystemExit(f"gold_{task}.json cases carry no `id` — re-export them")
        out[case["id"]] = case
    return out


def gold_records() -> dict[str, dict]:
    """The exemplar records captured beside the samples, keyed by record id"""
    p = paths.GOLD_CONTEXT / "records.jsonl"
    if not p.exists():
        return {}
    records = (json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line)
    return {r["record_id"]: r for r in records}


def ht_weight(item: dict) -> float | None:
    """Horvitz-Thompson occurrence weight, defined on the core stratum only"""
    if item["stratum"] == "census":
        return float(item["n_occ"])
    if item["stratum"] == "core" and item.get("incl_prob"):
        return item["n_occ"] / item["incl_prob"]
    return None
