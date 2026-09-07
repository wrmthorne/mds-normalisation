from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass

import polars as pl

from experiments.harness import log
from mds_norm.paths import EXP_OUT, ROOT

COMPARISONS = EXP_OUT / "method_comparisons.parquet"
TEX = EXP_OUT / "method_comparisons.tex"
TERMLIST_PRICING = ROOT / "experiments" / "termlist_pricing" / "persons_association.json"
BAR = 0.95  # the precision bar every wiring decision is measured against
ALPHA = 0.05
P_FLOOR = 0.001  # below this a p-value is quoted as an inequality
THOUSAND = 1000


@dataclass
class Row:
    variant: str
    shipped: bool
    verdict: str
    n: int | None = None
    quality_axis: str | None = None
    quality: float | None = None
    quality_lo: float | None = None
    quality_hi: float | None = None
    interval: str | None = None
    quality2_axis: str | None = None
    quality2: float | None = None
    cost_axis: str | None = None
    cost: float | None = None
    # Variants measured under different conditions are blocked, not ranked
    group: str | None = None


@dataclass
class Decision:
    slug: str
    label: str
    source: str
    collect: Callable[[], Iterator[Row]]
    note: str = ""


DECISIONS: list[Decision] = []


def decision(slug: str, label: str, source: str, note: str = "") -> Callable[[Callable], Callable]:
    def register(fn: Callable[[], Iterator[Row]]) -> Callable[[], Iterator[Row]]:
        DECISIONS.append(Decision(slug, label, source, fn, note))
        return fn

    return register


def bar_verdict(precision: float | None, n: int | None = None) -> str:
    if precision is None:
        return "not measured"
    if precision >= BAR:
        return f"clears bar ($n{{=}}{n}$)" if n is not None else "clears bar"
    return "below bar"


@decision(
    "atomiser",
    "Compound-value atomiser (tier 4): a small model against the deterministic split",
    "atomiser_variants/results.parquet",
    "Quality is occurrence-weighted exact match over the whole value and $F_1$ over the atoms it "
    "yields; \\texttt{everywhere} is the counterfactual pole, every value sent whole to the model "
    "with no split, guards or eligibility filter.",
)
def atomiser() -> Iterator[Row]:
    df = pl.read_parquet(EXP_OUT / "atomiser_variants" / "results.parquet").sort("ht_exact_match", descending=True)
    for r in df.iter_rows(named=True):
        shipped = r["variant"] == "deterministic"
        yield Row(
            variant=r["variant"],
            shipped=shipped,
            verdict="shipped" if shipped else ("cost pole" if "everywhere" in r["variant"] else "rejected"),
            n=r["n_scored"],
            quality_axis="Exact (occ.-wtd)",
            quality=r["ht_exact_match"],
            quality2_axis="Atom $F_1$",
            quality2=r["atom_f1"],
            cost_axis="Wh/1k values",
            cost=r["wh_per_1k_values"],
        )


def _extraction_variants(exp: str, shipped_variant: str, reference: set[str]) -> Iterator[Row]:
    df = (
        pl.read_parquet(EXP_OUT / "extraction_results.parquet")
        .filter(pl.col("exp") == exp)
        .sort("precision_strict", descending=True)
    )
    for r in df.iter_rows(named=True):
        shipped = r["variant"] == shipped_variant
        yield Row(
            variant=r["variant"],
            shipped=shipped,
            verdict="shipped" if shipped else ("cost pole" if r["variant"] in reference else "rejected"),
            n=r["n_items"],
            quality_axis="P (strict)",
            quality=r["precision_strict"],
            quality2_axis="R (strict)",
            quality2=r["recall_strict"],
            cost_axis="Wh/record",
            cost=r["wh_per_record"],
        )


@decision(
    "extraction_model",
    "Record-level extraction (tier 5): model and prompt scale",
    "extraction_results.parquet",
    "Strict scoring requires the value and its destination field to match gold; \\texttt{mechanical} "
    "is the probe scan's operations replayed on the same records with no model at all.",
)
def extraction_model() -> Iterator[Row]:
    return _extraction_variants("extraction_variants", "gpt-oss-20b:single_task_v2", {"mechanical"})


@decision(
    "representation",
    "Edit representation: how a proposed change is written",
    "extraction_results.parquet",
    "Same records, model and gate throughout, so what varies is the format the model writes its edit in.",
)
def representation() -> Iterator[Row]:
    return _extraction_variants("representation_variants", "rep:json_patch", set())


@decision(
    "extraction_tuning",
    "Extraction prompt tuning: task contract and record rendering",
    "extraction_tuning/results.parquet",
    "Replicated variants are reported as the mean over runs with the spread as $\\pm$ one standard "
    "deviation; the five repeats of the unchanged incumbent set the noise floor at $\\mathrm{sd}=0.019$ "
    "$F_1$, so a single-run gap smaller than that decides nothing. \\texttt{dimension\\_parser} "
    "is the no-model pole for the dimension task. The last round re-ran its own controls because "
    "the tier-1 repairs moved the gate's recall ceiling from 0.84/0.79 to 0.97/0.91, so its scores "
    "compare within the block and not across it.",
)
def extraction_tuning() -> Iterator[Row]:
    repaired = {"incumbent_pf", "v3_dimension_pf", "v3_dim_freetext", "dimension_parser"}
    df = (
        pl.read_parquet(EXP_OUT / "extraction_tuning" / "results.parquet")
        .with_columns(base=pl.col("variant").str.replace(r"#\d+$", ""))
        .group_by("base")
        .agg(
            runs=pl.len(),
            f1=pl.col("f1_strict").mean(),
            f1_sd=pl.col("f1_strict").std(),
            recall=pl.col("recall_strict").mean(),
            wh=pl.col("wh_per_record").mean(),
        )
        .with_columns(block=pl.col("base").is_in(repaired))
        .sort("block", "f1", descending=[False, True])
    )
    for r in df.iter_rows(named=True):
        shipped = r["base"] == "incumbent"
        sd = r["f1_sd"]
        yield Row(
            variant=r["base"],
            shipped=shipped,
            verdict="shipped" if shipped else ("no-model pole" if r["base"] == "dimension_parser" else "rejected"),
            group="Against the repaired parsers" if r["block"] else "Against the released parsers",
            n=r["runs"],
            quality_axis="$F_1$ (strict)",
            quality=r["f1"],
            quality_lo=None if sd is None else r["f1"] - sd,
            quality_hi=None if sd is None else r["f1"] + sd,
            interval=None if sd is None else "sd",
            quality2_axis="R (strict)",
            quality2=r["recall"],
            cost_axis="Wh/record",
            cost=r["wh"],
        )


@decision(
    "extraction_native",
    "Extraction-native models against the incumbent",
    "nuextract_trial/results.parquet",
    "Models trained for schema-constrained extraction, scored on the same gold as the incumbent. "
    "Their operations were never pooled for judgement, so their strict precision is a lower bound "
    "and the incumbent's is unaffected.",
)
def extraction_native() -> Iterator[Row]:
    df = pl.read_parquet(EXP_OUT / "nuextract_trial" / "results.parquet").sort("f1_strict", descending=True)
    for r in df.iter_rows(named=True):
        shipped = r["variant"] == "gpt-oss-20b:single_task_v2_record_first"
        yield Row(
            variant=r["variant"],
            shipped=shipped,
            verdict="shipped" if shipped else "rejected",
            n=r["n_items"],
            quality_axis="$F_1$ (strict)",
            quality=r["f1_strict"],
            quality2_axis="R (strict)",
            quality2=r["recall_strict"],
            cost_axis="Wh/record",
            cost=r["wh_per_record"],
        )


@decision(
    "homograph_rungs",
    "Homograph disambiguation: precision of each rung of the ladder",
    "homograph/rung_precision.json, homograph/rung_verdicts.json",
    "\\emph{Declined} is the share of items where the rung's own candidate set held no right answer. "
    "A rung whose interval excludes the 0.95 bar publishes its picks as qualified rather than applied.",
)
def homograph_rungs() -> Iterator[Row]:
    rungs = json.loads((EXP_OUT / "homograph" / "rung_precision.json").read_text())["rungs"]
    verdicts = json.loads((EXP_OUT / "homograph" / "rung_verdicts.json").read_text())
    for name, r in sorted(rungs.items(), key=lambda kv: -kv[1]["precision"]):
        action = verdicts[name]["action"]
        yield Row(
            variant=name,
            shipped=action == "keep",
            verdict={"keep": "applied", "keep_flagged": "qualified", "demote": "demoted to qualified"}[action],
            n=r["n_decided"],
            quality_axis="Precision",
            quality=r["precision"],
            quality_lo=r["wilson_lo"],
            quality_hi=r["wilson_hi"],
            interval="wilson",
            quality2_axis="Declined",
            quality2=r["none_rate"],
        )


def _guard_frontier(sweep: list[dict], baseline_n: int) -> list[dict]:
    """The shipped threshold, the best precision at half volume, and the sweep's maximum"""
    half = [s for s in sweep if s["n_admitted"] >= baseline_n / 2]
    picks = [sweep[0], max(half, key=lambda s: s["precision"]), max(sweep, key=lambda s: s["precision"])]
    seen: dict[float, dict] = {}
    for p in picks:
        seen.setdefault(p["threshold"], p)
    return sorted(seen.values(), key=lambda s: -s["n_admitted"])


@decision(
    "homograph_guards",
    "Homograph guards: can a threshold rescue the rungs the calibration demoted?",
    "homograph/guard_sweep.json",
    "The acceptance-score sweep collapsed to its frontier: the shipped threshold, the best precision "
    "retaining at least half the admitted items, and the sweep's maximum. \\emph{Lost} counts correct "
    "links the threshold would drop. The length floors are non-binding on both rungs --- every sampled "
    "fuzzy atom is already at least 7 characters and every semantic atom at least 8 --- so no length "
    "row appears.",
)
def homograph_guards() -> Iterator[Row]:
    guards = json.loads((EXP_OUT / "homograph" / "guard_sweep.json").read_text())["guards"]
    for name, g in guards.items():
        for point in _guard_frontier(g["score_sweep"], g["n"]):
            shipped = point is g["score_sweep"][0]
            yield Row(
                variant=f"{name}, score $\\geq$ {point['threshold']:.2f}",
                shipped=shipped,
                verdict="shipped" if shipped else bar_verdict(point["precision"], point["n_admitted"]),
                n=point["n_admitted"],
                quality_axis="Precision",
                quality=point["precision"],
                quality_lo=point["wilson_lo"],
                quality_hi=point["wilson_hi"],
                interval="wilson",
                quality2_axis="Lost",
                quality2=point["dropped_correct"],
            )


@decision(
    "homograph_verifier",
    "Homograph verifier: a model reading the atom against its candidate concepts",
    "homograph/verifier_precision{,_medium}.json",
    "\\emph{Accuracy} is the verifier deciding on its own; \\emph{confirm} is the share correct among "
    "the items where it endorses the pipeline's existing pick, which is the number a verification gate "
    "would have to clear.",
)
def homograph_verifier() -> Iterator[Row]:
    for effort, name in (("", "gpt-oss-20b, reasoning low"), ("_medium", "gpt-oss-20b, reasoning medium")):
        o = json.loads((EXP_OUT / "homograph" / f"verifier_precision{effort}.json").read_text())["overall"]
        yield Row(
            variant=name,
            shipped=False,
            verdict="not wired",
            n=o["n"],
            quality_axis="Accuracy",
            quality=o["verifier_accuracy"],
            quality_lo=o["verifier_wilson"][0],
            quality_hi=o["verifier_wilson"][1],
            interval="wilson",
            quality2_axis="Confirm P",
            quality2=o["confirm_precision"],
        )


@decision(
    "rerank_selector",
    "Vocabulary rerank (tier 3): cosine selection against a model reading the candidates",
    "rerank/cosine_rerank_comparison.json, rerank/cosine_sweep_rerank.json, "
    "rerank/context_variants_rerank_{strict,qualified}.json",
    "Scored on 231 queue-drawn items. Cosine at its shipped thresholds asserts a link for one item in "
    "231, so the comparison that decides anything is at matched volume, taken from the "
    "(accept, margin) grid at the assertion count the model reaches. \\emph{Link recall} is the share "
    "of gold links the rung finds.",
)
def rerank_selector() -> Iterator[Row]:
    base = json.loads((EXP_OUT / "rerank" / "cosine_rerank_comparison.json").read_text())
    sweep = json.loads((EXP_OUT / "rerank" / "cosine_sweep_rerank.json").read_text())
    variants = {
        regime: json.loads((EXP_OUT / "rerank" / f"context_variants_rerank_{regime}.json").read_text())["summary"][
            "none"
        ]
        for regime in ("strict", "qualified")
    }
    matched_n = variants["qualified"]["n_asserted_mean"]
    matched = min(sweep, key=lambda s: abs(s["n_asserted"] - matched_n))

    yield Row(
        variant="cosine, shipped thresholds",
        shipped=False,
        verdict="superseded",
        n=base["rung"]["n_asserted"],
        quality_axis="Precision",
        quality=base["rung"]["precision_lower"],
        quality_lo=base["rung"]["precision_lower_wilson"][0],
        quality_hi=base["rung"]["precision_lower_wilson"][1],
        interval="wilson",
        quality2_axis="Link recall",
        quality2=base["rung"]["link_recall"],
        cost_axis="Wh/item",
        cost=base["wh_per_item"],
    )
    yield Row(
        variant=f"cosine, accept {matched['accept']:g}/margin {matched['margin']:g}",
        shipped=False,
        verdict="matched-volume comparison",
        n=matched["n_asserted"],
        quality_axis="Precision",
        quality=matched["precision_lower"],
        quality_lo=matched["precision_lower_wilson"][0],
        quality_hi=matched["precision_lower_wilson"][1],
        interval="wilson",
        quality2_axis="Link recall",
        quality2=matched["link_recall"],
        cost_axis="Wh/item",
        cost=base["wh_per_item"],
    )
    for regime, s in variants.items():
        shipped = regime == "qualified"
        yield Row(
            variant=f"gpt-oss-20b, {regime} prompt",
            shipped=shipped,
            verdict="shipped, published qualified" if shipped else "rejected",
            n=round(s["n_asserted_mean"]),
            quality_axis="Precision",
            quality=s["precision_mean"],
            quality2_axis="Link recall",
            quality2=s["link_recall_mean"],
            cost_axis="Wh/item",
            cost=s["wh_per_item_mean"],
        )


@decision(
    "rerank_context",
    "Record context in the rerank prompt",
    "rerank/context_variants_rerank_qualified.json",
    "Five replicates per context against the same 231 items. \\emph{Agreement} is with the gold "
    "decision over every item, asserted or rejected, quoted as the mean $\\pm$ one standard "
    "deviation; the verdict reports the paired sign test against the no-context baseline.",
)
def rerank_context() -> Iterator[Row]:
    summary = json.loads((EXP_OUT / "rerank" / "context_variants_rerank_qualified.json").read_text())["summary"]
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["agreement_mean"]):
        vs = s["vs_none"]
        if vs is None:
            verdict = "shipped"
        elif vs["p_value"] < ALPHA:
            p = "$p<0.001$" if vs["p_value"] < P_FLOOR else f"$p={vs['p_value']:.3f}$"
            verdict = f"worse ({p})"
        else:
            verdict = f"no gain ($p={vs['p_value']:.2f}$)"
        yield Row(
            variant=name,
            shipped=name == "none",
            verdict=verdict,
            n=s["n_runs"],
            quality_axis="Agreement",
            quality=s["agreement_mean"],
            quality_lo=s["agreement_mean"] - s["agreement_sd"],
            quality_hi=s["agreement_mean"] + s["agreement_sd"],
            interval="sd",
            quality2_axis="Precision",
            quality2=s["precision_mean"],
            cost_axis="Wh/item",
            cost=s["wh_per_item_mean"],
        )


@decision(
    "ner_backend",
    "Off-the-shelf entity recognition over free-text cells",
    "ner/ner_scores.json",
    "263 cells, 1{,}076 gold spans, eight types. \\emph{Strict} needs identical offsets and type, "
    "which is the bar a patch-producing stage must clear; \\emph{relaxed} needs any overlap and the "
    "same type; \\emph{untyped} ignores type and is the detection ceiling. spaCy's tag set cannot "
    "express \\emph{material} or \\emph{technique}, a structural zero rather than a tuning failure. "
    "$n$ is the variant's predicted-span count, the denominator of its precision.",
)
def ner_backend() -> Iterator[Row]:
    variants = json.loads((EXP_OUT / "ner" / "ner_scores.json").read_text())["variants"]
    for variant, a in variants.items():
        for regime in ("strict", "relaxed", "untyped"):
            r = a["regimes"][regime]
            yield Row(
                variant=f"{variant}, {regime}",
                shipped=False,
                verdict="not wired",
                n=r["tp"] + r["fp"],
                quality_axis="Precision",
                quality=r["precision"],
                quality_lo=r["precision_wilson"][0],
                quality_hi=r["precision_wilson"][1],
                interval="wilson",
                quality2_axis="Recall",
                quality2=r["recall"],
            )


@decision(
    "candidate_router",
    "The candidate router: a model choosing an extracted value's destination field",
    "routing/routing_precision.json",
    "Scored against the placement gold, so the router and the pipeline are read on the same items. "
    "The baseline is what the pipeline's context-free destination rule achieves on those items; the "
    "router must beat it \\emph{and} clear the 0.95 bar to be wired.",
)
def candidate_router() -> Iterator[Row]:
    block = json.loads((EXP_OUT / "routing" / "routing_precision.json").read_text())["all"]
    verdicts = {"insufficient": "worse than pipeline", "beats_pipeline": "beats pipeline, below bar"}
    for family, f in sorted(block["families"].items(), key=lambda kv: -kv[1]["router_accuracy"]):
        yield Row(
            variant=family,
            shipped=False,
            verdict=verdicts.get(f["verdict"], f["verdict"]),
            n=f["n_scored"],
            quality_axis="Router accuracy",
            quality=f["router_accuracy"],
            quality_lo=f["router_wilson"][0],
            quality_hi=f["router_wilson"][1],
            interval="wilson",
            quality2_axis="Pipeline",
            quality2=f["pipeline_baseline"],
        )
    o = block["overall"]
    yield Row(
        variant="overall",
        shipped=False,
        verdict="not wired",
        n=o["n_scored"],
        quality_axis="Router accuracy",
        quality=o["router_accuracy"],
        quality_lo=o["router_wilson"][0],
        quality_hi=o["router_wilson"][1],
        interval="wilson",
        quality2_axis="Pipeline",
        quality2=o["pipeline_baseline"],
    )


@decision(
    "placement",
    "Destination precision of applied changes, by field",
    "placement/placement_precision.json",
    "\\emph{Precision} is the share of placements whose destination field is right; "
    "\\emph{misrouting} is the share filed somewhere the value does not belong. What the compiler "
    "then applies or qualifies follows its destination rule rather than this table, and is "
    "described in \\S\\ref{sec:tier5run}.",
)
def placement() -> Iterator[Row]:
    p = json.loads((EXP_OUT / "placement" / "placement_precision.json").read_text())
    rows = sorted(p["per_destination"].items(), key=lambda kv: -kv[1]["precision"])
    for dest, d in [*rows, ("overall", p["overall"])]:
        yield Row(
            variant=dest.removeprefix("spectrum/"),
            shipped=False,
            verdict=bar_verdict(d["precision"]),
            n=d["n_placement"],
            quality_axis="Precision",
            quality=d["precision"],
            quality_lo=d["wilson_lo"],
            quality_hi=d["wilson_hi"],
            interval="wilson",
            quality2_axis="Misrouting",
            quality2=d["misrouting"],
        )


@decision(
    "convention_regimes",
    "Convention-prior regimes for the tier-1 date parse",
    "fingerprint_routing/regimes_predictions.parquet, fingerprint_routing/regime_precision.json",
    "The whole date-field population, so the volume axes need no labels. Richer priors convert "
    "qualified occurrences into resolved ones and never the reverse: every disagreement between "
    "R1 and R2 is a promotion, and no row emits a different EDTF value.",
)
def convention_regimes() -> Iterator[Row]:
    regimes = {
        "r0": ("corpus-global defaults", "baseline"),
        "r1": ("per institution", "shipped"),
        "r2": ("per (institution, stratum)", "not wired: confidence only"),
    }
    df = (
        pl.read_parquet(EXP_OUT / "fingerprint_routing" / "regimes_predictions.parquet")
        .group_by("regime", "outcome")
        .agg(occ=pl.col("occ").sum())
        .pivot("outcome", index="regime", values="occ")
        .sort("regime")
    )
    for r in df.iter_rows(named=True):
        label, verdict = regimes[r["regime"]]
        yield Row(
            variant=label,
            shipped=r["regime"] == "r1",
            verdict=verdict,
            quality_axis="Qualified occ.",
            quality=r["qualified"],
            quality2_axis="Resolved occ.",
            quality2=r["resolved"],
        )


@decision(
    "validation_gate",
    "The validation gate's opportunity cost, by rejection reason",
    "rejected_ops/gate_trade.json",
    "Of the operations the gate rejected, \\emph{forgone} were correct fixes it killed and "
    "\\emph{invented} were values with nothing behind them. The core stratum is the unweighted "
    "sample; the reason strata oversample their own cause.",
)
def validation_gate() -> Iterator[Row]:
    strata = json.loads((EXP_OUT / "rejected_ops" / "gate_trade.json").read_text())["strata"]
    core = strata.pop("core")
    for name, s in [("core", core), *sorted(strata.items(), key=lambda kv: -kv[1]["forgone_fix_rate"])]:
        yield Row(
            variant=name.removeprefix("issue:"),
            shipped=False,
            verdict="gate kept",
            n=s["n"],
            quality_axis="Forgone",
            quality=s["forgone_fix_rate"],
            quality_lo=s["forgone_fix_wilson"][0],
            quality_hi=s["forgone_fix_wilson"][1],
            interval="wilson",
            quality2_axis="Invented",
            quality2=s["invented_value_rate"],
        )


@decision(
    "termlist_rungs",
    "Termlist cascade: what each rung adds on the pilot field",
    "experiments/termlist_pricing/persons_association.json",
    "The \\texttt{persons\\_association} pilot, a 185-term seed run through the whole cascade scoped "
    "to one field. \\emph{Gained} counts occurrences the rung resolved that no cheaper rung had; a "
    "rung gaining nothing on this field is priced but omittable.",
)
def termlist_rungs() -> Iterator[Row]:
    pricing = json.loads(TERMLIST_PRICING.read_text())
    for name, r in pricing["rungs"].items():
        gained = r["new_occ"]
        yield Row(
            variant=name,
            shipped=name != "llm_atomiser",
            verdict="shipped" if name != "llm_atomiser" else "priced, declined",
            quality_axis="Gained occ.",
            quality=gained,
            quality2_axis="New norms",
            quality2=r["new_norms"],
            cost_axis="Wh/1k occ.",
            cost=r["wh_per_1k_occ"],
        )


SCHEMA = {
    "decision": pl.String,
    "decision_label": pl.String,
    "variant": pl.String,
    "shipped": pl.Boolean,
    "verdict": pl.String,
    "n": pl.Int64,
    "quality_axis": pl.String,
    "quality": pl.Float64,
    "quality_lo": pl.Float64,
    "quality_hi": pl.Float64,
    "interval": pl.String,
    "quality2_axis": pl.String,
    "quality2": pl.Float64,
    "cost_axis": pl.String,
    "cost": pl.Float64,
    "group": pl.String,
    "source": pl.String,
}

# Count axes render whole rather than as rates
COUNT_AXES = frozenset({"Lost", "New norms", "Gained occ.", "Qualified occ.", "Resolved occ."})


def build() -> pl.DataFrame:
    rows = []
    for d in DECISIONS:
        collected = list(d.collect())
        if not collected:
            raise RuntimeError(f"{d.slug} produced no rows")
        rows += [{"decision": d.slug, "decision_label": d.label, "source": d.source, **asdict(r)} for r in collected]
    return pl.DataFrame(rows, schema=SCHEMA)


def tex_escape(text: str) -> str:
    for old, new in (("_", r"\_"), ("#", r"\#"), ("&", r"\&"), ("%", r"\%"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(old, new)
    return text


def fmt_count(value: float | None) -> str:
    if value is None:
        return "---"
    return f"{value:,.0f}".replace(",", "{,}") if abs(value) >= THOUSAND else f"{value:.0f}"


def fmt_rate(value: float | None) -> str:
    return "---" if value is None else f"{value:.3f}"


def fmt_axis(value: float | None, axis: str | None) -> str:
    return fmt_count(value) if axis in COUNT_AXES else fmt_rate(value)


def fmt_quality(row: dict) -> str:
    value = fmt_axis(row["quality"], row["quality_axis"])
    if row["quality"] is None or row["quality_lo"] is None:
        return value
    if row["interval"] == "sd":
        return f"{value} $\\pm$ {row['quality_hi'] - row['quality']:.3f}"
    return f"{value} ({row['quality_lo']:.3f}--{row['quality_hi']:.3f})"


def fmt_cost(value: float | None) -> str:
    return "---" if value is None else f"{value:.4g}"


def render(dec: Decision, frame: pl.DataFrame) -> str:
    rows = frame.to_dicts()
    has_n = any(r["n"] is not None for r in rows)
    has_q2 = any(r["quality2"] is not None for r in rows)
    has_cost = any(r["cost"] is not None for r in rows)
    quality_axis = next(r["quality_axis"] for r in rows if r["quality_axis"])
    quality_align = "r" if quality_axis in COUNT_AXES else "l"
    quoted = [r["interval"] == "wilson" for r in rows if r["quality"] is not None]
    if any(quoted):
        quality_axis += " (95\\% CI)" if all(quoted) else " (95\\% CI where quoted)"

    spec = "l" + ("r" if has_n else "") + quality_align + ("r" if has_q2 else "") + ("r" if has_cost else "") + "l"
    header = ["Variant", *(["$n$"] if has_n else []), quality_axis]
    if has_q2:
        header.append(next(r["quality2_axis"] for r in rows if r["quality2_axis"]))
    if has_cost:
        header.append(next(r["cost_axis"] for r in rows if r["cost_axis"]))
    header.append("Verdict")

    body = []
    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            if body:
                body.append(r"    \midrule")
            body.append(f"    \\multicolumn{{{len(header)}}}{{l}}{{\\emph{{{group}}}}} \\\\")
        variant = tex_escape(r["variant"]) if "$" not in r["variant"] else r["variant"]
        cells = [f"\\textbf{{{variant}}}" if r["shipped"] else variant]
        if has_n:
            cells.append(fmt_count(r["n"]))
        cells.append(fmt_quality(r))
        if has_q2:
            cells.append(fmt_axis(r["quality2"], r["quality2_axis"]))
        if has_cost:
            cells.append(fmt_cost(r["cost"]))
        cells.append(r["verdict"])
        body.append("    " + " & ".join(cells) + r" \\")

    note = f" {dec.note}" if dec.note else ""
    bold = " Bold marks what the release ships." if any(r["shipped"] for r in rows) else ""
    stop = "" if dec.label.endswith(("?", "!")) else "."
    return "\n".join(
        [
            r"\begin{table}",
            f"  \\caption{{{dec.label}{stop}{note}{bold} Source: \\texttt{{{tex_escape(dec.source)}}}.}}",
            f"  \\label{{tab:ml-{dec.slug.replace('_', '-')}}}",
            r"  \small",
            f"  \\begin{{tabular}}{{{spec}}}",
            r"    \toprule",
            "    " + " & ".join(header) + r" \\",
            r"    \midrule",
            *body,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    comparisons = build()
    COMPARISONS.parent.mkdir(parents=True, exist_ok=True)
    comparisons.write_parquet(COMPARISONS)

    TEX.parent.mkdir(parents=True, exist_ok=True)
    tables = [render(d, comparisons.filter(pl.col("decision") == d.slug)) for d in DECISIONS]
    TEX.write_text(
        "% Generated by experiments/method_comparisons.py; edit that script, not this file.\n\n" + "\n".join(tables)
    )

    shipped = comparisons.filter(pl.col("shipped")).height
    log(f"{comparisons.height} variants over {comparisons['decision'].n_unique()} decisions, {shipped} shipped")
    log(f"wrote {COMPARISONS}")
    log(f"wrote {TEX}")


if __name__ == "__main__":
    main()
