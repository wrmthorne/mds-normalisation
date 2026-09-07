from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl

from mds_norm.paths import COMPILED, EVAL_OUT, ROOT

RESIDUE_OUT = EVAL_OUT / "residue"
TOP_EXAMPLES = 15
"""Distinct values kept per family — enough to characterise it, not a dump"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_EMPTY = pl.DataFrame(
    schema={
        "family": pl.String,
        "field_type": pl.String,
        "occurrences": pl.UInt32,
        "distinct_values": pl.UInt32,
        "institutions": pl.UInt32,
        "routed": pl.UInt32,
        "examples": pl.List(pl.String),
        "note": pl.String,
    }
)


def _populated(compiled: Path) -> pl.LazyFrame:
    """Populated source-channel nodes, excluding the `wrmthorne/` release channel"""
    return pl.scan_parquet(compiled).filter(
        pl.col("value").is_not_null() & ~pl.col("field_type").str.starts_with("wrmthorne/")
    )


def _summarise(rows: pl.LazyFrame, family: str, note: str) -> pl.DataFrame:
    """Collapse a family's matching rows to (family, field, occ, distinct, examples)"""
    has_disp = "disposition" in rows.collect_schema().names()
    routed = (
        pl.col("disposition").cast(pl.String).is_in(["deferred", "qualified"])
        if has_disp
        else pl.lit(None, dtype=pl.Boolean)
    )
    per_field = (
        rows.group_by("field_type")
        .agg(
            occurrences=pl.len(),
            distinct_values=pl.col("value").n_unique(),
            institutions=pl.col("data_source").n_unique(),
            routed=routed.sum(),
        )
        .sort("occurrences", descending=True)
        .collect(engine="streaming")
    )
    examples = (
        rows.group_by("field_type", "value")
        .agg(occ=pl.len())
        .sort("occ", descending=True)
        .group_by("field_type", maintain_order=True)
        .agg(examples=pl.col("value").head(3))
        .collect(engine="streaming")
    )
    return (
        per_field.join(examples, on="field_type", how="left")
        .with_columns(family=pl.lit(family), note=pl.lit(note))
        .select("family", "field_type", "occurrences", "distinct_values", "institutions", "routed", "examples", "note")
    )


# shape only: did the date stage leave the value alone
_EDTF_DATE = r"-?[0-9X]{4}(-[0-9X]{2}(-[0-9X]{2})?)?[~?%]{0,2}"
EDTF_RE = (
    rf"^({_EDTF_DATE}"  # a date
    rf"|{_EDTF_DATE}/{_EDTF_DATE}"  # closed interval
    rf"|\.\./{_EDTF_DATE}|{_EDTF_DATE}/\.\."  # open-ended interval
    r"|\[[^]]+]|\{[^}]+})$"
)  # one-of / all-of set


def scan_dates(compiled: Path) -> pl.DataFrame:
    from mds_norm.pipeline.compile_records import DATE_FIELDS

    rows = _populated(compiled).filter(
        pl.col("field_type").is_in(DATE_FIELDS) & ~pl.col("value").str.contains(EDTF_RE)
    )
    return _summarise(rows, "dates_non_edtf", "value in a date field that is not in EDTF form")


def scan_placeholders(compiled: Path) -> pl.DataFrame:
    """Whole-value placeholders that survived tier 0"""
    from mds_norm.pipeline.compile_records import TIER0_PLACEHOLDERS

    norm = pl.col("value").str.strip_chars().str.to_lowercase()
    rows = _populated(compiled).filter(norm.is_in(sorted(TIER0_PLACEHOLDERS)))
    return _summarise(rows, "placeholder_whole_value", "recording-practice placeholder still published as content")


def scan_delimiters(compiled: Path) -> pl.DataFrame:
    """Values still carrying a dangling separator, outside the fields the strip exempts"""
    from mds_norm.pipeline.compile_records import NO_EDIT

    rows = _populated(compiled).filter(
        ~pl.col("field_type").is_in(NO_EDIT) & pl.col("value").str.contains(r"^\s*[,;|:]|[,;|:]\s*$")
    )
    return _summarise(rows, "dangling_delimiter", "leading or trailing separator with nothing on the other side")


def scan_numeric(compiled: Path) -> pl.DataFrame:
    """Non-numeric content in slots the model types as numbers"""
    from mds_norm.metrics.common import numeric_field_names, positive_int_field_names

    numeric = set(numeric_field_names()) | set(positive_int_field_names())
    rows = _populated(compiled).filter(
        pl.col("field_type").is_in(sorted(numeric))
        & ~pl.col("value").str.strip_chars().str.contains(r"^-?\d+(\.\d+)?$")
    )
    return _summarise(rows, "numeric_slot_non_numeric", "model types the slot as a number; the value is not one")


def scan_markup(compiled: Path) -> pl.DataFrame:
    """Real markup and encoding damage surviving into the release"""
    from mds_norm.utils.markup import HTML_ENTITY_RE, HTML_TAG_RE

    rows = _populated(compiled).filter(
        pl.col("value").str.contains(HTML_TAG_RE)
        | pl.col("value").str.contains(HTML_ENTITY_RE)
        | pl.col("value").str.contains("�", literal=True)
    )
    return _summarise(rows, "markup_or_encoding_damage", "HTML markup, entity references, or U+FFFD damage")


def scan_field_name_echo(compiled: Path) -> pl.DataFrame:
    """A value that just names its own field"""
    from mds_norm.pipeline.compile_records import _field_name_echo

    rows = _populated(compiled).filter(_field_name_echo(pl.col("value")))
    return _summarise(rows, "field_name_echo", "value is its own column header")


def scan_untouched_mass(compiled: Path) -> pl.DataFrame:
    """Where the untouched mass sits, per field, largest first"""
    if "disposition" not in pl.scan_parquet(compiled).collect_schema().names():
        log("  untouched: corpus has no `disposition` column — skipped")
        return _EMPTY
    rows = _populated(compiled).filter(pl.col("disposition") == "untouched")
    return _summarise(rows, "untouched", "no stage read, changed, qualified or enriched the value")


FAMILIES = {
    "dates": scan_dates,
    "placeholders": scan_placeholders,
    "delimiters": scan_delimiters,
    "numeric": scan_numeric,
    "markup": scan_markup,
    "field-name-echo": scan_field_name_echo,
    "untouched": scan_untouched_mass,
}


def dimension_residue(out_dir: Path) -> dict:
    path = out_dir / "dimension_residue.parquet"
    if not path.exists():
        return {}
    d = pl.read_parquet(path)
    by_reason = d.group_by("reason").agg(pairs=pl.len()).sort("pairs", descending=True)
    return {
        "deferred_pairs": d.height,
        "by_reason": {r["reason"]: int(r["pairs"]) for r in by_reason.iter_rows(named=True)},
    }


def dispositions(out_dir: Path) -> dict:
    path = out_dir / "coverage_census.parquet"
    if not path.exists():
        return {}
    cen = pl.read_parquet(path)
    return {
        d: int(cen.filter(pl.col("disposition") == d)["occurrences"].sum())
        for d in sorted(cen["disposition"].unique().to_list())
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scan the compiled corpus for content the pipeline did not normalise.",
        prog="python -m mds_norm.evaluation.residue_scan",
    )
    ap.add_argument(
        "--compiled", type=Path, default=COMPILED / "mds-normalised.parquet", help="compiled corpus to scan"
    )
    ap.add_argument(
        "--family", action="append", choices=sorted(FAMILIES), help="scan only these families (default: all)"
    )
    ap.add_argument("--out", type=Path, default=RESIDUE_OUT)
    args = ap.parse_args()

    if not args.compiled.exists():
        raise SystemExit(f"no compiled corpus at {args.compiled}")
    args.out.mkdir(parents=True, exist_ok=True)
    chosen = args.family or sorted(FAMILIES)

    frames = []
    for name in chosen:
        log(f"scanning {name}…")
        frame = FAMILIES[name](args.compiled)
        frame.write_parquet(args.out / f"{name}.parquet")
        frames.append(frame)
        log(f"  {name}: {int(frame['occurrences'].sum()):,} occurrences over {frame.height} fields")

    table = pl.concat(frames).sort("occurrences", descending=True) if frames else pl.DataFrame()
    if table.height:
        table.write_parquet(args.out / "residue_scan.parquet")

    out_dir = args.compiled.parent
    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "compiled": str(args.compiled.relative_to(ROOT)) if args.compiled.is_relative_to(ROOT) else str(args.compiled),
        "families": {name: int(frame["occurrences"].sum()) for name, frame in zip(chosen, frames, strict=True)},
        "dimension_residue": dimension_residue(out_dir),
        "dispositions": dispositions(out_dir),
    }
    (args.out / "residue_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
