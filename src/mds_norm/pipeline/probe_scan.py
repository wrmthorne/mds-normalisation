from __future__ import annotations

import re
import time

import polars as pl
from codecarbon import EmissionsTracker
from mds_data_model.introspection import free_text_fields, measurement_fields, vocab_fields

from mds_norm.parsers.parse_dates import parse_date
from mds_norm.parsers.parse_dimensions import parse_dimensions
from mds_norm.parsers.parse_monetary import parse_monetary
from mds_norm.paths import ANALYSIS_OUTPUT, EMISSIONS_LOG, FIELD_STATS, PROBE_CANDIDATES, PROBE_CANDIDATES_RAW
from mds_norm.pipeline.consistency_induction import PATTERN_FIELDS

LEXICON_CANDIDATES = ANALYSIS_OUTPUT / "lexicon_candidates.json"
LEXICON = ANALYSIS_OUTPUT / "lexicon.json"
REVIEW_SAMPLE = ANALYSIS_OUTPUT / "probe_review_sample.parquet"
CANDIDATE_PATTERNS = ANALYSIS_OUTPUT / "probe_candidate_patterns.parquet"
PROSE_COMPOSITION = ANALYSIS_OUTPUT / "prose_composition.parquet"

MEASUREMENT_GROUPS = list(measurement_fields())  # measurements are not collapsed during induction

TEXT_FIELDS = pl.col("field_type").is_in(free_text_fields())
TARGET_FIELDS = (TEXT_FIELDS | pl.col("field_type").is_in(vocab_fields())) & ~PATTERN_FIELDS
# Shorter values cannot carry a digit and a literal
MIN_PROBE_CHARS = 3
PREFILTER = (pl.col("digit_chars") > 0) & (pl.col("char_count") >= MIN_PROBE_CHARS)

# Token lexicon: frequent in its group, rare in prose
MIN_TOKEN_COUNT, MIN_LIFT, MAX_TOKENS = 50, 20.0, 40
PROSE_SAMPLE_N = 300_000
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}

# Probe compilation floors and the head-mass cutoff
MIN_SLOTS, MIN_DIGIT_SLOTS = 2, 1
PROBE_COVERAGE = 0.75
ANCHORS = set("£$€×:")
# Currency words the induced shapes cannot reach (no digit-adjacent literal)
MANUAL_PROBES = {"__price__": [r"\d{1,4}(?:[./]\d{1,2})? ?(?:guineas|gns|shillings|pence|pounds)"]}

# Diagnostic only: above this share the lexicon is failing
MAX_PROSE_HIT_RATE = 0.01
REVIEW_N = 8
# Dates from 2000 are usually accession numbers, so flagged
SUSPECT_RECENT_KEY = 20_000_101

# Few joining words plus high span coverage means a list
JOIN_SHARE_MAX, MATCH_SHARE_MIN = 0.05, 0.5
# too short for a sentence, so joining words say nothing
COMPOSITION_MIN_TOKENS = 5

_TOKEN_RE = re.compile(r"([dsS])(?:{(\d+)(?:,(\d+))?})?|(.)")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_lexicon(field_stats: pl.LazyFrame) -> pl.DataFrame:
    """Letter-run tokens each group's structured values use and prose does not"""
    inducible = field_stats.filter(pl.col("merged_pattern").is_not_null())
    group_tok = (
        inducible.select("merge_group", tok=pl.col("value").str.to_lowercase().str.extract_all(r"[a-z]+"))
        .explode("tok", empty_as_null=True)
        .drop_nulls("tok")
        .group_by("merge_group", "tok")
        .len("n")
        .with_columns(rate=pl.col("n") / pl.col("n").sum().over("merge_group"))
        .collect(engine="streaming")
    )
    prose_tok = (
        field_stats.filter(TEXT_FIELDS & pl.col("value").is_not_null())
        .head(PROSE_SAMPLE_N)
        .select(tok=pl.col("value").str.to_lowercase().str.extract_all(r"[a-z]+"))
        .explode("tok", empty_as_null=True)
        .drop_nulls("tok")
        .group_by("tok")
        .len("pn")
        .collect(engine="streaming")
        .with_columns(prate=(pl.col("pn") + 1) / pl.col("pn").sum())
    )
    return (
        group_tok.join(prose_tok.select("tok", "prate"), on="tok", how="left")
        .with_columns(prate=pl.col("prate").fill_null(prose_tok["prate"].min()))
        .with_columns(lift=pl.col("rate") / pl.col("prate"))
        .filter((pl.col("n") >= MIN_TOKEN_COUNT) & (pl.col("lift") >= MIN_LIFT))
        .sort("n", descending=True)
        .group_by("merge_group", maintain_order=True)
        .head(MAX_TOKENS)
        # roman numerals in dates are volume/part numbers, not month names
        .filter(~(pl.col("merge_group").eq("__date__") & pl.col("tok").is_in(ROMAN)))
    )


def lexicon_alternations() -> dict[str, str]:
    """The curated lexicon as one regex alternation per group"""
    return {
        row["merge_group"]: "|".join(re.escape(t) for t in sorted(row["tok"], key=len, reverse=True))
        for row in pl.read_json(LEXICON).iter_rows(named=True)
    }


def to_probe(signature: str, lex_alt: str) -> str | None:
    """Compile an induced merged pattern into a regex, or refuse it"""
    tokens = []
    for m in _TOKEN_RE.finditer(signature):
        kind, lo, hi, literal = m.groups()
        if kind == "d":
            lo = int(lo) if lo else 1
            hi = int(hi) if hi else lo
            tokens.append(("class", rf"\d{{{lo}}}" if lo == hi else rf"\d{{{lo},{hi}}}"))
        elif kind:
            if not lex_alt:
                return None
            tokens.append(("lexical", rf"(?:{lex_alt})"))
        else:
            tokens.append(("literal", re.escape(literal)))
    has_lexical = any(k == "lexical" for k, _ in tokens)
    has_anchor = any(k == "literal" and t.lstrip("\\") in ANCHORS for k, t in tokens)
    if not has_lexical and not has_anchor:
        return None
    parts = (
        ([r"\b"] if tokens[0][0] != "literal" else [])
        + [t for _, t in tokens]
        + ([r"\b"] if tokens[-1][0] != "literal" else [])
    )
    rx = "".join(parts)
    try:
        re.compile(rx)
    except re.error:
        return None
    return rx


def compile_probes(field_stats: pl.LazyFrame, lex_alt: dict[str, str]) -> pl.DataFrame:
    """Head-mass merged patterns compiled to regexes, plus the manual currency probes"""
    inducible = field_stats.filter(pl.col("merged_pattern").is_not_null())
    skeleton = pl.col("merged_pattern").str.replace_all(r"\{[^}]*\}", "")
    n_slots = skeleton.str.count_matches("[dsS]")
    floors = (n_slots >= MIN_SLOTS) & (skeleton.str.count_matches("d") >= MIN_DIGIT_SLOTS)

    probes = (
        inducible.group_by("merge_group", "merged_pattern")
        .len("count")
        .filter(floors)
        .sort("count", descending=True)
        .with_columns(
            prev_cum=(pl.col("count").cum_sum().over("merge_group") - pl.col("count"))
            / pl.col("count").sum().over("merge_group")
        )
        .filter(pl.col("prev_cum") < PROBE_COVERAGE)
        .collect(engine="streaming")
        .with_columns(
            regex=pl.struct("merge_group", "merged_pattern").map_elements(
                lambda r: to_probe(r["merged_pattern"], lex_alt.get(r["merge_group"], "")), return_dtype=pl.String
            )
        )
        .drop_nulls("regex")
    )
    manual = pl.DataFrame(
        {
            "merge_group": [g for g, rxs in MANUAL_PROBES.items() for _ in rxs],
            "merged_pattern": "manual",
            "count": 0,
            "prev_cum": 0.0,
            "regex": [rx for rxs in MANUAL_PROBES.values() for rx in rxs],
        }
    )
    return pl.concat([probes, manual], how="vertical_relaxed")


def group_alternations(probes: pl.DataFrame) -> dict[str, str]:
    return {
        group: "(?i)(?:" + "|".join(sorted(sub["regex"].unique(), key=len, reverse=True)) + ")"
        for (group,), sub in probes.group_by("merge_group")
    }


def scan_targets(field_stats: pl.LazyFrame, group_alt: dict[str, str]) -> None:
    """Every probe match in the target fields, one row per span"""
    base = field_stats.filter(TARGET_FIELDS & PREFILTER)
    pl.concat(
        [
            base.with_columns(candidate=pl.col("value").str.extract_all(alt), group=pl.lit(group))
            .filter(pl.col("candidate").list.len() > 0)
            .explode("candidate", empty_as_null=True)
            .select("node_id", "record_id", "data_source", "field_type", "group", "candidate")
            for group, alt in group_alt.items()
        ]
    ).sink_parquet(PROBE_CANDIDATES_RAW)


def _edtf_key(iso: str, end: bool) -> int:
    """Sortable integer for an ISO point; missing components open to the bound's side"""
    negative = iso.startswith("-")
    year, *rest = iso.lstrip("-").split("-")
    month = int(rest[0]) if rest else (12 if end else 1)
    day = int(rest[1]) if len(rest) > 1 else (31 if end else 1)
    return (-1 if negative else 1) * int(year) * 10_000 + month * 100 + day


def date_bounds(value: str) -> tuple[int, int] | None:
    parsed = parse_date(value)
    if parsed is None or parsed["date_earliest_single"] is None:
        return None
    lo = parsed["date_earliest_single"]
    hi = parsed["date_latest"] or lo
    return _edtf_key(lo, end=False), _edtf_key(hi, end=True)


def verify_measurement(value: str) -> bool:
    parsed = parse_dimensions(value)
    return parsed is not None and bool(parsed["measurements"])


def verify_amount(value: str) -> bool:
    """Positive-number gate for the measurement groups with no tier-1 parser"""
    m = _NUM.search(value)
    return m is not None and float(m.group().replace(",", ".")) > 0


VERIFY = {"__price__": lambda v: parse_monetary(v) is not None, "spectrum/dimension": verify_measurement}
VERIFY |= {g: VERIFY.get(g, verify_amount) for g in MEASUREMENT_GROUPS}


def accepts(group: str, value: str) -> bool:
    """The group's own tier-1 verifier. Dates go through the bounds parser"""
    return date_bounds(value) is not None if group == "__date__" else VERIFY[group](value)


def verify_candidates(raw: pl.LazyFrame) -> pl.DataFrame:
    """Per distinct (group, candidate): does the group's own parser accept it?"""
    distinct = raw.select("group", "candidate").unique().collect(engine="streaming")
    rows = []
    for group, candidate in distinct.iter_rows():
        if group == "__date__":
            bounds = date_bounds(candidate)
            rows.append((group, candidate, bounds is not None, *(bounds or (None, None))))
        else:
            rows.append((group, candidate, accepts(group, candidate), None, None))
    verified = pl.DataFrame(
        rows,
        schema={"group": pl.String, "candidate": pl.String, "ok": pl.Boolean, "lo": pl.Int64, "hi": pl.Int64},
        orient="row",
    )
    log(f"verified {verified['ok'].sum():,} / {len(distinct):,} distinct candidates")
    return verified.filter(pl.col("ok")).drop("ok")


def date_status(field_stats: pl.LazyFrame, raw: pl.LazyFrame, verified: pl.DataFrame) -> pl.LazyFrame:
    """Interval comparison against the record's own parsed date values"""
    inducible = field_stats.filter(pl.col("merged_pattern").is_not_null())
    record_values = (
        inducible.filter(pl.col("merge_group") == "__date__")
        .select("record_id", "value")
        .unique()
        .collect(engine="streaming")
    )
    bounds = {v: b for v in record_values["value"].unique() if (b := date_bounds(v)) is not None}

    record_dates = (
        record_values.with_columns(
            lo=pl.col("value").map_elements(lambda v: (bounds.get(v) or (None,))[0], return_dtype=pl.Int64),
            hi=pl.col("value").map_elements(lambda v: (bounds.get(v) or (None, None))[1], return_dtype=pl.Int64),
        )
        .drop_nulls("lo")
        .select("record_id", slo="lo", shi="hi")
    )
    has_date_field = (
        inducible.filter(pl.col("merge_group") == "__date__")
        .select("record_id")
        .unique()
        .with_columns(has_date_field=pl.lit(True))
    )

    return (
        raw.filter(pl.col("group") == "__date__")
        .join(verified.filter(pl.col("group") == "__date__").lazy(), on=["group", "candidate"])
        .join(record_dates.lazy(), on="record_id", how="left")
        .join(has_date_field, on="record_id", how="left")
        .with_columns(
            echo=(pl.col("lo") <= pl.col("slo")) & (pl.col("hi") >= pl.col("shi")),
            refine=(pl.col("lo") >= pl.col("slo"))
            & (pl.col("hi") <= pl.col("shi"))
            & ~((pl.col("lo") == pl.col("slo")) & (pl.col("hi") == pl.col("shi"))),
        )
        .group_by("node_id", "record_id", "data_source", "field_type", "group", "candidate")
        .agg(
            n_structured=pl.col("slo").drop_nulls().len(),
            any_echo=pl.col("echo").any(),
            any_refine=pl.col("refine").any(),
            has_date_field=pl.col("has_date_field").first().fill_null(False),
            lo=pl.col("lo").first(),
        )
        .with_columns(
            status=pl.when(pl.col("n_structured") == 0)
            .then(pl.when(pl.col("has_date_field")).then(pl.lit("unparsed_field")).otherwise(pl.lit("novel")))
            .when(pl.col("any_echo"))
            .then(pl.lit("echo"))
            .when(pl.col("any_refine"))
            .then(pl.lit("refine"))
            .otherwise(pl.lit("additional")),
            suspect_recent=pl.col("lo") >= SUSPECT_RECENT_KEY,
        )
        .select("node_id", "record_id", "data_source", "field_type", "group", "candidate", "status", "suspect_recent")
    )


def value_status(field_stats: pl.LazyFrame, raw: pl.LazyFrame, verified: pl.DataFrame) -> pl.LazyFrame:
    """String comparison for dimensions and prices, where interval semantics do not apply"""

    def norm(col: pl.Expr) -> pl.Expr:
        return col.str.to_lowercase().str.replace_all(r"\s+", "")

    record_values = (
        field_stats.filter(pl.col("merged_pattern").is_not_null() & (pl.col("merge_group") != "__date__"))
        .select("record_id", group="merge_group", vnorm=norm(pl.col("value")))
        .unique()
    )
    return (
        raw.filter(pl.col("group") != "__date__")
        .join(verified.lazy().select("group", "candidate"), on=["group", "candidate"])
        .with_columns(cnorm=norm(pl.col("candidate")))
        .join(
            record_values.group_by("record_id", "group").agg(vnorms=pl.col("vnorm")),
            on=["record_id", "group"],
            how="left",
        )
        .with_columns(
            status=pl.when(pl.col("vnorms").is_null())
            .then(pl.lit("novel"))
            .when(pl.col("vnorms").list.contains(pl.col("cnorm")))
            .then(pl.lit("echo"))
            .otherwise(pl.lit("refine"))
        )
        .select("node_id", "record_id", "data_source", "field_type", "group", "candidate", "status")
    )


def prose_hit_report(field_stats: pl.LazyFrame, probes: pl.DataFrame) -> None:
    """Which probes still fire on prose, and how much of that the parser accepts"""
    prose = (
        field_stats.filter(TEXT_FIELDS & pl.col("value").is_not_null() & (pl.col("digit_chars") > 0))
        .select("value")
        .head(PROSE_SAMPLE_N)
        .collect(engine="streaming")
    )
    hit = prose.select(
        [
            pl.col("value").str.contains("(?i)" + rx).mean().alias(f"{g}::{mp}")
            for g, mp, rx in probes.select("merge_group", "merged_pattern", "regex").iter_rows()
        ]
    ).row(0, named=True)

    noisy = {k: v for k, v in hit.items() if v > MAX_PROSE_HIT_RATE}
    log(f"{len(noisy)} noisy probes" if noisy else "no probe exceeds the prose gate")
    for name, rate in sorted(noisy.items(), key=lambda kv: -kv[1]):
        group, merged = name.split("::")
        rx = (
            "(?i)" + probes.filter((pl.col("merge_group") == group) & (pl.col("merged_pattern") == merged))["regex"][0]
        )
        matched = (
            prose.select(hit=pl.col("value").str.extract_all(rx))["hit"]
            .explode(empty_as_null=True)
            .drop_nulls()
            .unique()
        )
        ok = sum(1 for v in matched if accepts(group, v))
        log(f"  {name}  prose_rate={rate:.3f}  parse_rate={ok / len(matched):.2f}")


def candidate_patterns(verified: pl.DataFrame, probes: pl.DataFrame) -> pl.DataFrame:
    """The merged pattern each verified candidate matches, so a span can be reported by its shape"""
    pending = verified.select("group", "candidate").unique()
    matched = []
    for row in probes.sort("merge_group").iter_rows(named=True):
        if pending.is_empty():
            break
        rx = "(?i)^(?:" + row["regex"] + ")$"
        hit = pending.filter(
            (pl.col("group") == row["merge_group"]) & pl.col("candidate").str.contains(rx)
        ).with_columns(merged_pattern=pl.lit(row["merged_pattern"]))
        if hit.is_empty():
            continue
        matched.append(hit)
        pending = pending.join(hit.select("group", "candidate"), on=["group", "candidate"], how="anti")
    log(f"{sum(m.height for m in matched):,} candidates carry a pattern; {pending.height:,} matched none whole")
    return (
        pl.concat(matched)
        if matched
        else pl.DataFrame(schema={"group": pl.String, "candidate": pl.String, "merged_pattern": pl.String})
    )


def prose_composition(field_stats: pl.LazyFrame, candidates: pl.LazyFrame) -> pl.DataFrame:
    """How much of each matched value is span rather than sentence"""
    spans = (
        candidates.group_by("node_id", "field_type", "data_source")
        .agg(n_spans=pl.len(), matched_chars=pl.col("candidate").str.len_chars().sum())
        .collect(engine="streaming")
    )
    values = (
        field_stats.filter(TARGET_FIELDS & pl.col("value").is_not_null())
        .select("node_id", "char_count", "token_count", "stop_count")
        .collect(engine="streaming")
    )
    return (
        spans.join(values, on="node_id", how="left")
        .with_columns(
            match_share=(pl.col("matched_chars") / pl.col("char_count")).clip(0.0, 1.0),
            join_share=pl.when(pl.col("token_count") >= COMPOSITION_MIN_TOKENS)
            .then(pl.col("stop_count") / pl.col("token_count"))
            .otherwise(None),
        )
        .with_columns(is_list=(pl.col("join_share") <= JOIN_SHARE_MAX) & (pl.col("match_share") >= MATCH_SHARE_MIN))
    )


def review_sample(candidates: pl.LazyFrame) -> pl.DataFrame:
    return (
        candidates.filter(pl.col("status") != "echo")
        .group_by("group", "status")
        .agg(sample=pl.struct("field_type", "candidate", "record_id").shuffle(seed=0).head(REVIEW_N))
        .collect(engine="streaming")
    )


def main() -> None:
    EMISSIONS_LOG.mkdir(parents=True, exist_ok=True)
    field_stats = pl.scan_parquet(FIELD_STATS)

    lexicon = build_lexicon(field_stats)
    lexicon.group_by("merge_group").agg(pl.col("tok")).write_json(LEXICON_CANDIDATES)
    log(f"{len(lexicon):,} lexicon candidates → {LEXICON_CANDIDATES}")
    if not LEXICON.exists():
        raise SystemExit(f"{LEXICON} is missing — curate {LEXICON_CANDIDATES.name} into it before compiling probes")

    probes = compile_probes(field_stats, lexicon_alternations())
    group_alt = group_alternations(probes)
    log(f"{len(probes)} probes over {len(group_alt)} groups")

    with EmissionsTracker(project_name="probe_scan_targets", output_dir=str(EMISSIONS_LOG), log_level="error"):
        scan_targets(field_stats, group_alt)
    raw = pl.scan_parquet(PROBE_CANDIDATES_RAW)
    log(f"{raw.select(pl.len()).collect(engine='streaming').item():,} raw candidate spans")

    verified = verify_candidates(raw)
    with EmissionsTracker(project_name="probe_status_dates", output_dir=str(EMISSIONS_LOG), log_level="error"):
        dates = date_status(field_stats, raw, verified)
    with EmissionsTracker(project_name="probe_status_dims_prices", output_dir=str(EMISSIONS_LOG), log_level="error"):
        others = value_status(field_stats, raw, verified)
        pl.concat([dates, others], how="diagonal").sink_parquet(PROBE_CANDIDATES)

    candidates = pl.scan_parquet(PROBE_CANDIDATES)
    log(f"{candidates.select(pl.len()).collect(engine='streaming').item():,} verified spans → {PROBE_CANDIDATES}")

    candidate_patterns(verified, probes).write_parquet(CANDIDATE_PATTERNS)
    log(f"candidate patterns → {CANDIDATE_PATTERNS}")
    composition = prose_composition(field_stats, candidates)
    composition.write_parquet(PROSE_COMPOSITION)
    log(
        f"{composition.height:,} matched values scored; {composition['is_list'].sum():,} read as a list of "
        f"structured content rather than prose → {PROSE_COMPOSITION}"
    )
    prose_hit_report(field_stats, probes)
    review_sample(candidates).write_parquet(REVIEW_SAMPLE)
    log(f"review sample → {REVIEW_SAMPLE}")


if __name__ == "__main__":
    main()
