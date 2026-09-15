from __future__ import annotations

import argparse
import time

import polars as pl

from mds_norm.parsers.parse_certainty import LEXICAL_MARKERS
from mds_norm.paths import FIELD_STATS, INSTITUTIONAL
from mds_norm.pipeline.probe_scan import COMPOSITION_MIN_TOKENS, JOIN_SHARE_MAX

OUT_PATH = INSTITUTIONAL / "uncertainty_marking.parquet"
STYLES_PATH = INSTITUTIONAL / "uncertainty_styles.parquet"
QUESTION_PATH = INSTITUTIONAL / "question_mark_uses.parquet"
QUESTION_JOIN_PATH = INSTITUTIONAL / "question_mark_join_share.parquet"

# The three ways a cataloguer marks doubt inside a value
MARKERS = {
    "question": r"\?",
    "brackets": r"^\s*\[[^\[\]]+\]\s*$",
    "lexical": r"(?i:\b(?:" + "|".join([*LEXICAL_MARKERS, r"circa", r"ca?\.\s*\d"]) + r")\b)",
}

# A marker becomes style once this share carries it
STYLE_MIN_RATE = 0.001

# The question mark's count favours precision: prose and URLs disqualify
URLISH = r"https?://|www\.|\?[\w.%+-]+=|&[\w.%+-]+="
# a trailing question mark marks the value; mid-sentence punctuates
TRAILING_Q = r"[^?]\?+\s*$|^\s*\?+[^?]"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def marked_values() -> pl.DataFrame:
    """Every populated descriptive value, flagged for each marker it carries"""
    return (
        pl.scan_parquet(FIELD_STATS)
        .filter(pl.col("value").is_not_null() & pl.col("field_type").str.starts_with("spectrum/"))
        .select(
            pl.col("data_source").cast(pl.String),
            "field_type",
            "value",
            "token_count",
            "stop_count",
            **{name: pl.col("value").str.contains(rx) for name, rx in MARKERS.items()},
        )
        .with_columns(
            urlish=pl.col("value").str.contains(URLISH),
            trailing_q=pl.col("value").str.contains(TRAILING_Q),
            join_share=pl.when(pl.col("token_count") >= COMPOSITION_MIN_TOKENS)
            .then(pl.col("stop_count") / pl.col("token_count"))
            .otherwise(0.0),
        )
        # doubt only outside URLs and sentences, at an edge
        .with_columns(
            question_doubt=pl.col("question")
            & ~pl.col("urlish")
            & pl.col("trailing_q")
            & (pl.col("join_share") <= JOIN_SHARE_MAX)
        )
        .collect(engine="streaming")
    )


def _question_use(values: pl.DataFrame) -> pl.DataFrame:
    """Every question-mark value labelled with the use its shape implies"""
    return values.filter("question").with_columns(
        use=pl.when(pl.col("urlish"))
        .then(pl.lit("link or query string"))
        .when(pl.col("join_share") > JOIN_SHARE_MAX)
        .then(pl.lit("a sentence asking something"))
        .when(~pl.col("trailing_q"))
        .then(pl.lit("mid-value punctuation"))
        .otherwise(pl.lit("doubt about the value"))
    )


def question_uses(values: pl.DataFrame) -> pl.DataFrame:
    """How the question marks divide between doubt, a sentence, a link, and a mark inside running text"""
    return (
        _question_use(values)
        .group_by("use")
        .agg(n=pl.len(), fields=pl.col("field_type").n_unique(), institutions=pl.col("data_source").n_unique())
        .sort("n", descending=True)
    )


def question_join_share(values: pl.DataFrame, bins: int = 40) -> pl.DataFrame:
    """The joining-word share of every question-mark value, binned, one series per use"""
    return (
        _question_use(values)
        .with_columns(bin=(pl.col("join_share") * bins).floor().cast(pl.Int32).clip(0, bins - 1) / bins)
        .group_by("use", "bin")
        .agg(n=pl.len())
        .sort("use", "bin")
    )


def per_institution(values: pl.DataFrame) -> pl.DataFrame:
    """Each institution's rate of use for each marker"""
    return (
        values.group_by("data_source")
        .agg(
            n_values=pl.len(),
            question=pl.col("question_doubt").mean(),
            brackets=pl.col("brackets").mean(),
            lexical=pl.col("lexical").mean(),
        )
        .sort("data_source")
    )


def styles(rates: pl.DataFrame) -> pl.DataFrame:
    """Institutions grouped by which markers they use, none being a style of its own"""
    used = rates.with_columns(
        [(pl.col(m) >= STYLE_MIN_RATE).alias(f"uses_{m}") for m in ("question", "brackets", "lexical")]
    )
    return used.with_columns(
        style=pl.when(~pl.col("uses_question") & ~pl.col("uses_brackets") & ~pl.col("uses_lexical"))
        .then(pl.lit("no marker"))
        .otherwise(
            pl.concat_str(
                [
                    pl.when(pl.col(f"uses_{m}")).then(pl.lit(m)).otherwise(pl.lit(""))
                    for m in ("question", "brackets", "lexical")
                ],
                separator=" ",
            )
            .str.strip_chars()
            .str.replace_all(r"\s+", " + ")
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="How each institution marks uncertainty inside its values.")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    values = marked_values()
    log(f"{values.height:,} values over {values['data_source'].n_unique()} institutions")

    uses = question_uses(values)
    uses.write_parquet(QUESTION_PATH)
    log(f"question-mark uses → {QUESTION_PATH}")
    print(uses)

    shares = question_join_share(values)
    shares.write_parquet(QUESTION_JOIN_PATH)
    log(f"question-mark joining-word shares → {QUESTION_JOIN_PATH}")

    rates = per_institution(values)
    styled = styles(rates)
    INSTITUTIONAL.mkdir(parents=True, exist_ok=True)
    styled.write_parquet(args.out)
    counts = styled.group_by("style").agg(institutions=pl.len()).sort("institutions", descending=True)
    counts.write_parquet(STYLES_PATH)
    log(f"{styled.height} institutions → {args.out}; styles → {STYLES_PATH}")
    print(counts)


if __name__ == "__main__":
    main()
