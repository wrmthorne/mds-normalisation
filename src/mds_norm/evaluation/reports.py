from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from mds_norm.paths import (
    COMPILED,
    EMISSIONS_LOG,
    EVAL_OUT,
    EXP_OUT,
    FIELD_STATS,
    GOLD_FRAMES,
    LLM_RESPONSES,
    METRICS_OUT,
    RAW_RECORDS,
    RECORD_FIXES_OUT,
    ROOT,
    VOCAB_ANNOTATIONS,
    VOCAB_DECISIONS,
)

EMISSIONS_LOG_PATH = EMISSIONS_LOG


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# per-field and corpus-totals coverage from the compile-produced census

CENSUS = COMPILED / "coverage_census.parquet"

# a field is owned when enough occurrences left `untouched`
OWNED_MIN_SHARE = 0.05


def cmd_coverage(_args: argparse.Namespace) -> None:
    cen = pl.read_parquet(CENSUS)
    # `verified` and `enriched` are coverage without a value change
    worked = pl.col("disposition").is_in(["applied", "qualified", "deferred"])

    per_field = (
        cen.group_by("field_type")
        .agg(
            occurrences=pl.col("occurrences").sum(),
            institutions=pl.col("data_source").n_unique(),
            applied=pl.col("occurrences").filter(pl.col("disposition") == "applied").sum(),
            qualified=pl.col("occurrences").filter(pl.col("disposition") == "qualified").sum(),
            deferred=pl.col("occurrences").filter(pl.col("disposition") == "deferred").sum(),
            enriched=pl.col("occurrences").filter(pl.col("disposition") == "enriched").sum(),
            verified=pl.col("occurrences").filter(pl.col("disposition") == "verified").sum(),
            untouched=pl.col("occurrences").filter(pl.col("disposition") == "untouched").sum(),
        )
        .with_columns(
            worked_share=(pl.col("applied") + pl.col("qualified") + pl.col("deferred")) / pl.col("occurrences")
        )
    )

    owner = (
        cen.filter(worked)
        .with_columns(component=pl.col("component").fill_null("unattributed"))
        .group_by("field_type", "component")
        .agg(comp_occ=pl.col("occurrences").sum())
        .sort("comp_occ", descending=True)
        .group_by("field_type", maintain_order=True)
        .agg(owning_stage=pl.col("component").first(), owning_stage_occ=pl.col("comp_occ").first())
    )

    table = (
        per_field.join(owner, on="field_type", how="left")
        .with_columns(
            owning_stage=pl.when(pl.col("worked_share") >= OWNED_MIN_SHARE)
            .then(pl.col("owning_stage"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            owning_stage_share=pl.when(pl.col("worked_share") >= OWNED_MIN_SHARE)
            .then(pl.col("owning_stage_occ") / pl.col("occurrences"))
            .otherwise(pl.lit(None, dtype=pl.Float64)),
        )
        .drop("owning_stage_occ")
        .sort("occurrences", descending=True)
    )
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    table.write_parquet(EVAL_OUT / "coverage_table.parquet")

    corpus_occ = int(cen["occurrences"].sum())
    owned = table.filter(pl.col("owning_stage").is_not_null())
    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "source": str(CENSUS.relative_to(ROOT)),
        "owned_min_share": OWNED_MIN_SHARE,
        "corpus_occurrences": corpus_occ,
        "fields": table.height,
        "institutions": int(cen["data_source"].n_unique()),
        "dispositions": {
            d: int(cen.filter(pl.col("disposition") == d)["occurrences"].sum())
            for d in ("applied", "qualified", "deferred", "enriched", "verified", "untouched")
        },
        "component_worked_occ": {
            r["component"]: int(r["occ"])
            for r in cen.filter(worked)
            .with_columns(component=pl.col("component").fill_null("unattributed"))
            .group_by("component")
            .agg(occ=pl.col("occurrences").sum())
            .sort("occ", descending=True)
            .iter_rows(named=True)
        },
        "value_level_coverage": {
            "fields": owned.height,
            "occ": int(owned["occurrences"].sum()),
            "share": round(int(owned["occurrences"].sum()) / corpus_occ, 4),
        },
    }
    # raw-corpus view: same fields, occurrence mass from field_stats
    raw = pl.scan_parquet(FIELD_STATS).group_by("field_type").agg(occ=pl.len()).collect(engine="streaming")
    raw_occ = int(raw["occ"].sum())
    raw_owned = int(raw.join(owned.select("field_type"), on="field_type", how="semi")["occ"].sum())
    summary["value_level_coverage_raw_corpus"] = {
        "corpus_occurrences": raw_occ,
        "occ": raw_owned,
        "share": round(raw_owned / raw_occ, 4),
    }
    (EVAL_OUT / "coverage_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["dispositions"], indent=2))
    print(
        f"value-level coverage: {summary['value_level_coverage']['fields']} fields, "
        f"{summary['value_level_coverage']['share']:.1%} of occurrence mass"
    )
    print(f"wrote {EVAL_OUT / 'coverage_table.parquet'} and coverage_summary.json")


# raw and compiled measurements need one fixed ruler

RAW_PATH = RAW_RECORDS
WEIGHTS_OUT_DIR = METRICS_OUT


def cmd_freeze_weights(_args: argparse.Namespace) -> None:
    import subprocess

    from codecarbon import track_emissions

    from mds_norm.metrics import completeness, load_base, thinness

    @track_emissions(project_name="freeze_metric_weights", output_dir=str(EMISSIONS_LOG_PATH), log_level="error")
    def _run() -> None:
        base = load_base(RAW_PATH)

        weights = (
            completeness.field_weights(base)
            .with_columns(pl.col("field_type").cast(pl.String), pl.col("data_source").cast(pl.String))
            .sort("data_source", "field_type")
        )
        propensity = (
            thinness.decomposition_propensity(base)
            .with_columns(pl.col("field_type").cast(pl.String))
            .sort("field_type")
        )

        WEIGHTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        weights.write_parquet(WEIGHTS_OUT_DIR / "field_weights.parquet")
        propensity.write_parquet(WEIGHTS_OUT_DIR / "decomposition_propensity.parquet")

        try:
            git = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            git = None
        manifest = {
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "raw_corpus": str(RAW_PATH),
            "git": git,
            "field_weights": {
                "rows": weights.height,
                "institutions": weights["data_source"].n_unique(),
                "fields": weights["field_type"].n_unique(),
            },
            "decomposition_propensity": {"rows": propensity.height},
        }
        (WEIGHTS_OUT_DIR / "freeze_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

        log(
            f"field weights: {weights.height:,} (institution, field) rows "
            f"({weights['data_source'].n_unique()} institutions, "
            f"{weights['field_type'].n_unique()} fields)"
        )
        log(f"propensity: {propensity.height:,} fields ({propensity.filter(pl.col('p_f') > 0).height} with p_f > 0)")
        log(f"frozen → {WEIGHTS_OUT_DIR}")

    _run()


# both stages pre-date the emissions log, so each is re-run

DATE_CACHE = COMPILED / "date_parse_cache.parquet"
TIER01_OUT = EVAL_OUT / "tier01_rerun.json"


def _flatten_verbose(pat: str) -> str:
    """Ftfy's re.VERBOSE badness regex flattened so the polars engine can run it natively"""
    out, in_class, esc, in_comment = [], False, False, False
    for ch in pat:
        if in_comment:
            if ch == "\n":
                in_comment = False
        elif esc:
            out.append(ch)
            esc = False
        elif ch == "\\":
            out.append(ch)
            esc = True
        elif in_class:
            out.append(ch)
            if ch == "]":
                in_class = False
        elif ch == "[":
            in_class = True
            out.append(ch)
        elif ch == "#":
            in_comment = True
        elif not ch.isspace():
            out.append(ch)
    return "".join(out)


def cmd_tier01_energy(_args: argparse.Namespace) -> None:
    import tempfile

    import ftfy
    from codecarbon import EmissionsTracker

    from mds_norm.parsers.parse_dates import Conventions, load_periods, parse_date
    from mds_norm.pipeline.compile_records import PERIODS_CSV, PROTECTED, _dehtml
    from mds_norm.pipeline.compile_records import RAW_PATH as COMPILE_RAW_PATH

    mojibake_regex = _flatten_verbose(ftfy.badness.BADNESS_RE.pattern)

    def fix_mojibake(s: pl.Series) -> pl.Series:
        return pl.Series([ftfy.fix_text(v) for v in s], dtype=pl.String())

    def _summary(tracker: EmissionsTracker, seconds: float) -> dict:
        d = tracker.final_emissions_data
        return {
            "seconds": round(seconds, 1),
            "energy_wh": round(d.energy_consumed * 1e3, 3),
            "emissions_g": round(d.emissions * 1e3, 3),
        }

    def run_tier0(scratch: Path) -> dict:
        """Trim, mojibake fix and HTML scrub over every spectrum/ node, streamed to a scratch sink"""
        t0 = time.time()
        with EmissionsTracker(
            project_name="tier0_standardise", output_dir=str(EMISSIONS_LOG_PATH), log_level="error"
        ) as tracker:
            base = (
                pl.scan_parquet(COMPILE_RAW_PATH)
                .filter(pl.col("field_type").str.starts_with("spectrum/") & pl.col("value").is_not_null())
                .with_columns(pl.col("value").str.strip_chars())
                .filter(pl.col("value") != "")
                .with_columns(contains_mojibake=pl.col("value").str.contains(mojibake_regex))
            )
            fixed = base.filter(pl.col("contains_mojibake")).select(
                "node_id",
                value_fixed=pl.col("value").map_batches(fix_mojibake, return_dtype=pl.String(), is_elementwise=True),
            )
            (
                base.join(fixed, on="node_id", how="left")
                .with_columns(value=pl.coalesce("value_fixed", "value"))
                .with_columns(
                    value=pl.when(pl.col("field_type").is_in(PROTECTED))
                    .then(pl.col("value"))
                    .otherwise(_dehtml(pl.col("value")))
                )
                .select("node_id", "value")
                .sink_parquet(scratch)
            )
        nodes = pl.scan_parquet(scratch).select(pl.len()).collect().item()
        scratch.unlink()
        out = {"nodes": nodes, **_summary(tracker, time.time() - t0)}
        print(f"tier0_standardise: {nodes:,} nodes, {out['seconds']}s, {out['energy_wh']} Wh")
        return out

    def run_tier1() -> dict:
        """parse_date over the production cache's distinct (value, conventions) tuples"""
        load_periods(PERIODS_CSV)
        tuples = pl.read_parquet(DATE_CACHE)
        t0 = time.time()
        parsed = 0
        with EmissionsTracker(
            project_name="tier1_date_parse", output_dir=str(EMISSIONS_LOG_PATH), log_level="error"
        ) as tracker:
            for v, o, e, z in tuples.select("value", "dm_order", "eq", "z0").iter_rows():
                try:
                    p = parse_date(v, Conventions(dm_order=o or None, eq_range=e, zero_null=z))
                except Exception:
                    p = None
                parsed += p is not None
        cached_parse = int(tuples["edtf"].is_not_null().sum())
        out = {
            "distinct_tuples": tuples.height,
            "parsed": parsed,
            "cache_edtf_nonnull": cached_parse,
            **_summary(tracker, time.time() - t0),
        }
        print(
            f"tier1_date_parse: {tuples.height:,} tuples, {parsed:,} parse "
            f"(cache edtf non-null {cached_parse:,}), {out['seconds']}s, "
            f"{out['energy_wh']} Wh"
        )
        return out

    scratch = Path(tempfile.gettempdir()) / "tier0_rerun_scratch.parquet"
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "note": "single production-shaped pass per stage",
        "tier0_standardise": run_tier0(scratch),
        "tier1_date_parse": run_tier1(),
    }
    TIER01_OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {TIER01_OUT}")


# measured subset beside the full-queue projection, read from artefacts

EMISSIONS_CSV = EMISSIONS_LOG_PATH / "emissions.csv"
EXTRACTION_RUN = RECORD_FIXES_OUT / "extraction_report.json"
EXTRACTION_FRAME = GOLD_FRAMES / "extraction.parquet"
EXTRACTION_RESULTS = EXP_OUT / "extraction_results.parquet"
ATOMISER_RESULTS = EXP_OUT / "atomiser_variants" / "results.parquet"
VOCAB_ANN = VOCAB_ANNOTATIONS
TERMLIST_PRICING = ROOT / "experiments" / "termlist_pricing" / "persons_association.json"
LLM_QUEUE_OUT = EVAL_OUT / "llm_queue_projections.json"


def project_energy(project: str, emissions: pl.DataFrame) -> float:
    return float(emissions.filter(pl.col("project_name") == project)["energy_consumed"].sum() * 1e3)


def cmd_llm_queue(_args: argparse.Namespace) -> None:
    emissions = pl.read_csv(EMISSIONS_CSV)

    # measured rate once the queue has run, else projected
    scaling = (
        pl.read_parquet(EXTRACTION_RESULTS)
        .filter(pl.col("variant") == "gpt-oss-20b:single_task_v2")
        .row(0, named=True)
    )
    scaling_configuration = {
        "variant": scaling["variant"],
        "wh_per_record": round(scaling["wh_per_record"], 5),
        "projected_wh": round(scaling["projected_queue_wh"], 1),
        "projection_records": pl.read_parquet(EXTRACTION_FRAME).height,
        "source": "analysis_output/experiments/extraction_results.parquet",
    }

    if EXTRACTION_RUN.exists():
        run = json.loads(EXTRACTION_RUN.read_text())
        queue, done = run["queue"], run["run"]
        record_fixes = {
            "tier": 5,
            "queue_fully_processed": queue["shards_done"] >= queue["shards"],
            "measured": {
                "records": done.get("records"),
                "requests": done.get("requests"),
                "accepted_ops": run["ops"]["llm_resolved"],
                "energy_wh": done.get("energy_wh"),
                "wh_per_record": run["unit_cost"]["wh_per_record"],
                "configuration": run["configuration"]["scale"] + "/" + run["configuration"]["contract"],
                "note": "measured over the queue this run processed "
                f"({queue['shards_done']}/{queue['shards']} shards); "
                "energy is this stage's shards only",
                "source": str(EXTRACTION_RUN.relative_to(ROOT)),
            },
            "full_queue": {
                "records": queue["records"],
                "requests": queue["requests"],
                "source": "data/extraction/queue.parquet",
            },
            "projected_wh": round(run["unit_cost"]["wh_per_record"] * queue["records"], 1),
            "scaling_configuration": scaling_configuration,
        }
    else:
        subset_records = pl.read_parquet(LLM_RESPONSES).height
        rf_wh = project_energy("record_fixes_llm", emissions)
        queue_records = pl.read_parquet(EXTRACTION_FRAME).height
        rf_rate = rf_wh / subset_records
        record_fixes = {
            "tier": 5,
            "measured_subset": {
                "records": subset_records,
                "energy_wh": round(rf_wh, 1),
                "wh_per_record": round(rf_rate, 5),
                "note": "energy is every logged production run (phase-1 "
                "configuration, incl. re-runs), so the rate is an upper "
                "bound on a clean pass",
            },
            "full_queue": {"records": queue_records, "source": str(EXTRACTION_FRAME.relative_to(ROOT))},
            "projected_wh": round(rf_rate * queue_records, 1),
            "scaling_configuration": scaling_configuration,
        }

        # the atomiser's rungs are the llm_* sub-components
    splitter = pl.col("sub_component").str.starts_with("llm")
    t4_values = pl.scan_parquet(VOCAB_DECISIONS).filter(splitter).select(pl.len()).collect().item()
    t4_occ = pl.scan_parquet(VOCAB_ANN).filter(splitter).select(pl.len()).collect(engine="streaming").item()
    rerank = pl.col("sub_component") == "rerank"
    rerank_values = pl.scan_parquet(VOCAB_DECISIONS).filter(rerank).select(pl.len()).collect().item()
    rerank_occ = pl.scan_parquet(VOCAB_ANN).filter(rerank).select(pl.len()).collect(engine="streaming").item()
    atomiser = pl.read_parquet(ATOMISER_RESULTS)
    atomiser_rows = {r["variant"]: r for r in atomiser.iter_rows(named=True)}
    vocab_atomiser = {
        "tier": 4,
        "queue_fully_processed": True,
        "measured": {
            "distinct_values": t4_values,
            "annotated_occ": t4_occ,
            "energy_wh": round(project_energy("vocab_atomise_llm", emissions), 1),
            "note": "the production cascade ran its whole admission queue "
            "(2026-07-17), so the projection equals the "
            "measurement; energy is every logged run",
        },
        "atomiser_counterfactuals_projected_wh": {
            v: {
                "projected_queue_wh": round(atomiser_rows[v]["projected_queue_wh"], 1),
                "projection_values": atomiser_rows[v]["projection_values"],
            }
            for v in ("deterministic", "gpt-oss-20b", "gpt-oss-20b:everywhere")
            if v in atomiser_rows
        },
    }

    vocab_rerank = {
        "tier": 4,
        "queue_fully_processed": True,
        "measured": {
            "distinct_values": rerank_values,
            "annotated_occ": rerank_occ,
            "energy_wh": round(project_energy("vocab_rerank_llm", emissions), 1),
            "note": "top-k retrieval then one selection call per queued atom; energy is every logged run",
        },
    }

    pricing = json.loads(TERMLIST_PRICING.read_text())
    rung = pricing["rungs"]["llm_atomiser"]
    admissible = pricing["target_restatement"]["admissible_termlist_occ"]
    wh_per_1k = rung["wh_per_1k_occ"]
    termlist = {
        "tier": 4,
        "measured_subset": {
            "queue_atoms": pricing["llm_queue"]["atoms"],
            "queue_occ": pricing["llm_queue"]["occ"],
            "resolved_occ": rung["new_occ"],
            "energy_wh": round(rung["energy_kwh"] * 1e3, 3),
            "wh_per_1k_occ_resolved": wh_per_1k,
            "source": str(TERMLIST_PRICING.relative_to(ROOT)),
        },
        "full_queue": {
            "admissible_termlist_occ": admissible,
            "note": "upper bound: assumes the future fields' whole admissible "
            "occurrence mass reaches the rung, though the "
            "deterministic rungs resolved 86.4% for the pilot field "
            "and the rung is omittable by decision (2026-07-17)",
        },
        "projected_wh_upper_bound": round(wh_per_1k * admissible / 1e3, 1),
    }

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "definition": "measured subset (units, Wh, Wh/unit) beside the "
        "full-queue projection Wh/unit x queue size, per LLM "
        "stage; production energy from the production emissions "
        "log, experiment rates from the experiment artefacts",
        "record_fixes_llm": record_fixes,
        "vocab_atomise_llm": vocab_atomiser,
        "vocab_rerank_llm": vocab_rerank,
        "termlist_llm_rung": termlist,
    }
    LLM_QUEUE_OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {LLM_QUEUE_OUT}")


# every check, its failure rate, and the fields driving it


def cmd_conformance_failures(_args: argparse.Namespace) -> None:
    from mds_norm.metrics import conformance, load_base

    out_dir = EVAL_OUT / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_base(RAW_RECORDS)
    scored = conformance.compute(base)
    per_check, per_field = conformance.failure_breakdown(base, scored)
    per_check.write_parquet(out_dir / "conformance_checks.parquet")
    per_field.write_parquet(out_dir / "conformance_fields.parquet")
    log(f"{per_check.height} checks scored over {scored.height:,} records → {out_dir}")
    with pl.Config(tbl_rows=25):
        print(per_check)
        print(per_field.head(15))


COMMANDS = {
    "coverage": (cmd_coverage, "persist the census/coverage table"),
    "conformance-failures": (cmd_conformance_failures, "per-check conformance failure counts"),
    "freeze-weights": (cmd_freeze_weights, "freeze the raw-corpus metric weights"),
    "tier01-energy": (cmd_tier01_energy, "measure tier 0 + tier-1 date-parse energy"),
    "llm-queue": (cmd_llm_queue, "project full-queue LLM stage costs"),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluation reports, one subcommand each.", prog="python -m mds_norm.evaluation.reports"
    )
    sub = ap.add_subparsers(dest="command", required=True)
    for name, (fn, help_text) in COMMANDS.items():
        sub.add_parser(name, help=help_text).set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
