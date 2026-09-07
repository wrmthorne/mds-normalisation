from __future__ import annotations

import argparse
import asyncio
import json
import re

import polars as pl
from mds_data_model.introspection import date_fields

from experiments.harness import log
from mds_norm.parsers.parse_dates import Conventions, parse_date
from mds_norm.paths import EXP_OUT, FIELD_STATS, GOLD, GOLD_LABELS, INSTITUTIONAL, VOCAB_DECISIONS
from mds_norm.pipeline import extraction
from mds_norm.pipeline.vocab_alignment import SEPARATORS
from mds_norm.utils.atomise import BASE_SEPARATORS, separator_literal

EXP = "routing_signals"
OUT = EXP_OUT / EXP
QUEUE = extraction.QUEUE
RECALL_SAMPLE = GOLD / "samples" / "recall.jsonl"
RECALL_LABELS = GOLD_LABELS / "recall.jsonl"
DATE_CONVENTIONS = INSTITUTIONAL / "date_conventions.parquet"

# Values sampled per institution when the date conventions are scored
DATE_SAMPLE = 20_000

# The probe size, and the admitted records' accepted-operation rate
PROBE_RECORDS = 2_000
QUEUE_YIELD = 0.301
SEED = 20260904

# Share of scored institutions a corpus-wide separator needs
CORPUS_WIDE_MIN_SHARE = 0.5


def separator_effect() -> pl.DataFrame:
    """What splitting on each institution's own separators buys over one corpus-wide set"""
    # The cascade writes the institution as an enum
    scores = pl.read_parquet(SEPARATORS).with_columns(pl.col("data_source").cast(pl.String))
    accepted = scores.filter("is_separator")
    log(f"{accepted.height} accepted (group, institution, separator) triples over {scores.height} scored")

    # The counterfactual: one separator set every institution would get
    corpus_wide = (
        scores.group_by("group", "separator")
        .agg(accepted=pl.col("is_separator").sum(), scored=pl.len())
        .with_columns(share=pl.col("accepted") / pl.col("scored"))
        .filter(pl.col("share") >= CORPUS_WIDE_MIN_SHARE)
        .select("group", "separator")
    )
    log(f"corpus-wide set: {corpus_wide.height} (group, separator) pairs")

    decisions = pl.read_parquet(
        VOCAB_DECISIONS, columns=["group", "data_source", "value", "count", "atom", "norm", "status"]
    ).with_columns(pl.col("data_source").cast(pl.String))
    # A base separator would have split this value anyway
    induced = {separator_literal(sep) for sep in scores["separator"].unique()} - set(BASE_SEPARATORS)
    induced_rx = "|".join(re.escape(f" {lit} " if lit.isalpha() else lit) for lit in sorted(induced))
    base_rx = "|".join(re.escape(sep) for sep in BASE_SEPARATORS)
    split_values = decisions.filter(pl.col("value") != pl.col("atom")).with_columns(
        induced_split=pl.col("value").str.contains(induced_rx) & ~pl.col("value").str.contains(base_rx)
    )

    per_institution = (
        split_values.group_by("group", "data_source")
        .agg(
            occurrences=pl.col("count").sum(),
            induced_occurrences=pl.col("count").filter("induced_split").sum(),
            resolved=pl.col("count").filter(pl.col("status") == "resolved").sum(),
            resolved_induced=pl.col("count").filter(pl.col("induced_split") & (pl.col("status") == "resolved")).sum(),
        )
        .join(accepted.with_columns(learned=pl.lit(True)), on=["group", "data_source"], how="left")
        .join(corpus_wide.with_columns(corpus_wide=pl.lit(True)), on=["group", "separator"], how="left")
        .with_columns(pl.col("learned").fill_null(False), pl.col("corpus_wide").fill_null(False))
    )
    return per_institution.with_columns(
        resolve_rate=pl.col("resolved") / pl.col("occurrences"),
        induced_resolve_rate=pl.col("resolved_induced") / pl.col("induced_occurrences"),
    )


def date_convention_effect() -> pl.DataFrame:
    """Dates left ambiguous under one corpus-wide reading against each institution's own convention"""
    conventions = pl.read_parquet(DATE_CONVENTIONS)
    orders = dict(zip(conventions["data_source"], conventions["dm_order"], strict=True))
    values = (
        pl.scan_parquet(FIELD_STATS)
        .filter(pl.col("field_type").is_in(sorted(date_fields())) & pl.col("value").is_not_null())
        .select(pl.col("data_source").cast(pl.String), "value")
        .unique()
        .collect(engine="streaming")
        # Only an all-numeric triple can be ambiguous
        .filter(pl.col("value").str.contains(r"^\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*$"))
    )
    log(f"{values.height:,} distinct values carry a day-month-year shape")

    rows = []
    for source, frame in values.group_by("data_source"):
        institution = source[0] if isinstance(source, tuple) else source
        sample = frame["value"].head(DATE_SAMPLE).to_list()
        order = orders.get(str(institution))
        corpus = [parse_date(v) for v in sample]
        own = [parse_date(v, Conventions(dm_order=order)) for v in sample]
        # These all parse either way; the convention settles which reading
        rows.append(
            {
                "data_source": str(institution),
                "n_values": len(sample),
                "dm_order": order,
                "ambiguous_corpus_wide": sum(1 for p in corpus if p and p["dm_ambiguous"]),
                "ambiguous_own_convention": sum(1 for p in own if p and p["dm_ambiguous"]),
                "reading_changed": sum(
                    1 for a, b in zip(corpus, own, strict=True) if a and b and a["value_edtf"] != b["value_edtf"]
                ),
            }
        )
    return pl.DataFrame(rows).with_columns(
        settled=(pl.col("ambiguous_corpus_wide") - pl.col("ambiguous_own_convention")) / pl.col("n_values"),
        changed=pl.col("reading_changed") / pl.col("n_values"),
    )


def admission_effect() -> pl.DataFrame:
    """What the admission rule left out: the audited records it never sent to the model, and what they held"""
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
    audited = pl.DataFrame(
        [
            {"record_id": sample[i], "classes": (row.get("gold") or {}).get("classes") or []}
            for i, row in latest.items()
            if row.get("status") == "labelled" and i in sample
        ]
    )
    queue = pl.read_parquet(QUEUE, columns=["record_id"]).unique().with_columns(admitted=pl.lit(True))
    joined = audited.join(queue, on="record_id", how="left").with_columns(pl.col("admitted").fill_null(False))
    return (
        joined.explode("classes")
        .group_by("admitted", "classes")
        .agg(n=pl.len())
        .join(joined.group_by("admitted").agg(records=pl.len()), on="admitted")
        .with_columns(rate=pl.col("n") / pl.col("records"))
        .sort("admitted", "n", descending=[False, True])
    )


async def admission_probe(n_records: int, model: str, base_url: str, concurrency: int) -> pl.DataFrame:
    """Ask the model for the records admission left out, and count what the gate would accept"""
    from experiments.extraction_common import load_requests, single_task_prompts
    from experiments.harness import measure
    from mds_norm.pipeline.extraction import DECODING, MIN_TEXT_CHARS, record_targets
    from mds_norm.utils.inference import Inference
    from mds_norm.utils.patches import SYSTEM_V2, parse_ops, validate_op

    targets = record_targets().filter(pl.col("n_missing") > 0)
    left_out = targets.filter(pl.col("text_chars") < MIN_TEXT_CHARS)
    log(
        f"{targets.height:,} records miss a target; {left_out.height:,} of them sit below the "
        f"{MIN_TEXT_CHARS}-character text floor the admission rule applies"
    )
    drawn = left_out.sample(min(n_records, left_out.height), shuffle=True, seed=SEED)
    items = [
        {
            "id": row["record_id"],
            "record_id": row["record_id"],
            "issues": [f"missing_{k}" for k in ("date", "material", "dimension") if not row[f"has_{k}"]],
        }
        for row in drawn.iter_rows(named=True)
    ]
    reqs = load_requests(items)
    units = [(req, task, prompt) for req in reqs for task, prompt in single_task_prompts(req)]
    log(f"{len(reqs):,} records render a prompt, {len(units):,} requests, one task each")

    inf = Inference(model=model, base_url=base_url, concurrency=concurrency, timeout=600.0)
    with measure(EXP, "admission_probe") as cost:
        replies = await inf.generate([u[2] for u in units], system=SYSTEM_V2, usage=True, progress=True, **DECODING)
    log(f"probe cost: {cost.get('energy_wh', 0):.1f} Wh")

    rows = []
    for (req, task, _), reply in zip(units, replies, strict=True):
        ops = parse_ops(reply["content"]) or []
        if not ops:
            rows.append({"record_id": req["record_id"], "task": task, "status": "nothing", "reason": None})
            continue
        for op in ops:
            applied, reason = validate_op(req, op)
            rows.append(
                {
                    "record_id": req["record_id"],
                    "task": task,
                    "status": "accepted" if applied else "rejected",
                    "reason": reason,
                }
            )
    return pl.DataFrame(
        rows, schema={"record_id": pl.String, "task": pl.String, "status": pl.String, "reason": pl.String}
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="What each institutional signal buys the routing that uses it.")
    ap.add_argument("--signal", nargs="+", default=["separators", "dates", "admission"])
    ap.add_argument("--probe-records", type=int, default=PROBE_RECORDS)
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--base-url", default="http://localhost:30000/v1")
    ap.add_argument("--concurrency", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if "separators" in args.signal:
        sep = separator_effect()
        sep.write_parquet(OUT / "separator_effect.parquet")
        log(f"separators → {OUT / 'separator_effect.parquet'}")
        print(
            sep.group_by("learned").agg(
                institutions=pl.len(),
                occurrences=pl.col("occurrences").sum(),
                resolve_rate=pl.col("resolve_rate").median(),
            )
        )

    if "dates" in args.signal:
        dates = date_convention_effect()
        dates.write_parquet(OUT / "date_convention_effect.parquet")
        log(f"date conventions → {OUT / 'date_convention_effect.parquet'}")
        print(
            dates.group_by(pl.col("dm_order").is_not_null().alias("has_convention")).agg(
                institutions=pl.len(),
                values=pl.col("n_values").sum(),
                ambiguous=pl.col("ambiguous_corpus_wide").sum(),
                settled=pl.col("ambiguous_corpus_wide").sum() - pl.col("ambiguous_own_convention").sum(),
                changed=pl.col("reading_changed").sum(),
            )
        )

    if "admission" in args.signal:
        adm = admission_effect()
        adm.write_parquet(OUT / "admission_effect.parquet")
        log(f"admission → {OUT / 'admission_effect.parquet'}")
        print(adm)

    if "probe" in args.signal:
        ops = asyncio.run(admission_probe(args.probe_records, args.model, args.base_url, args.concurrency))
        ops.write_parquet(OUT / "admission_probe.parquet")
        per_record = ops.group_by("record_id").agg(accepted=(pl.col("status") == "accepted").any())
        log(
            f"{per_record.height:,} records probed; {per_record['accepted'].mean():.3f} yield an accepted "
            f"operation, against {QUEUE_YIELD:.3f} over the admitted queue → {OUT / 'admission_probe.parquet'}"
        )
        print(ops.group_by("status", "reason").agg(n=pl.len()).sort("n", descending=True))


if __name__ == "__main__":
    main()
