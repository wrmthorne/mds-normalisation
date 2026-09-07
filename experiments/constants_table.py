from __future__ import annotations

import importlib
import json

import polars as pl

from experiments.harness import log
from mds_norm import paths

OUT = paths.EXP_OUT / "constants"

CONSTANTS: list[tuple[str, str, str, str]] = [
    # module, attribute, what it decides, what would replace it
    (
        "mds_norm.pipeline.consistency_induction",
        "TAU",
        "a pattern is one of an institution's usual forms once it holds this share of its values",
        "labelled values called usual or unusual by a cataloguer of that institution",
    ),
    (
        "mds_norm.pipeline.consistency_induction",
        "WIDE_MIN",
        "digit runs this long or longer read as wide in a date's width signature",
        "the width at which a run stops being a day or month, read off the slot distributions",
    ),
    (
        "mds_norm.pipeline.consistency_induction",
        "SPLIT_THRESHOLD",
        "slot medians diverging by this much split a merged pattern in two",
        "labelled pairs of patterns judged to be the same shape or different",
    ),
    (
        "mds_norm.pipeline.pattern_exports",
        "MIN_SLOT_SUPPORT",
        "a slot below this many values is not written for the role tests",
        "the sample size at which a slot role stops changing when resampled",
    ),
    (
        "mds_norm.pipeline.divergence_tests",
        "MIN_N",
        "a divergence test needs this many values on each side",
        "the power calculation for the effect sizes actually reported",
    ),
    (
        "mds_norm.pipeline.divergence_tests",
        "FDR",
        "expected share of false positives among the cells called significant",
        "nothing: this is a stated tolerance rather than an estimate",
    ),
    (
        "mds_norm.pipeline.institutional_priors",
        "GT12_FRAC",
        "a slot exceeding 12 this often is a day rather than a month",
        "labelled dates whose day-month order a cataloguer confirms",
    ),
    (
        "mds_norm.pipeline.institutional_priors",
        "GT31_FRAC",
        "a slot exceeding 31 this often is a year fragment",
        "the same labelled dates",
    ),
    (
        "mds_norm.pipeline.institutional_priors",
        "ZERO_FRAC_MIN",
        "a slot filled with zero this often is a placeholder",
        "institutions asked what a zero month means in their catalogue",
    ),
    (
        "mds_norm.metrics.completeness",
        "APPLICABILITY",
        "a field counts against a record once its institution or the corpus fills it this often",
        "the fields each institution says apply to its collection",
    ),
    (
        "mds_norm.metrics.thinness",
        "DENSITY_PERCENTILE",
        "the institution's record against which another record's content volume is measured",
        "nothing: any high percentile is a stand-in for a full record",
    ),
    (
        "mds_norm.metrics.thinness",
        "SKEW_MIN_CHARS",
        "below this many characters a record's prose share is not defined",
        "the length at which the share stops moving with length",
    ),
    (
        "mds_norm.metrics.conformance",
        "TITLE_REUSE",
        "a title shared by this many records in one institution counts as systematically reused",
        "reused titles a cataloguer confirms are a defect rather than a series",
    ),
    (
        "mds_norm.metrics.conformance",
        "SHORT_DESC",
        "a description shorter than this is too short",
        "descriptions judged adequate or not by a reader",
    ),
    (
        "mds_norm.metrics.conformance",
        "LONG_TITLE",
        "a title longer than this is a description in the wrong field",
        "the same reader judgements",
    ),
    (
        "mds_norm.pipeline.accession_schemes",
        "MIN_DATED",
        "dated records needed before a numbering scheme is judged on agreement rather than structure",
        "the sample size at which the agreement rate stops moving",
    ),
    (
        "mds_norm.pipeline.accession_schemes",
        "AGREE_CONFIRM",
        "agreement with recorded accession years that confirms a scheme",
        "schemes an institution confirms or denies directly",
    ),
    (
        "mds_norm.pipeline.accession_schemes",
        "AGREE_TOLERANCE",
        "years either side of the recorded year that still count as agreement",
        "how long each institution takes to record what it accessions",
    ),
    (
        "mds_norm.pipeline.accession_schemes",
        "IMPOSSIBLE_MAX",
        "records produced after their own accession year that reject a scheme",
        "the rate of genuinely wrong production dates, which this cannot separate",
    ),
    (
        "mds_norm.pipeline.institutional_fingerprints",
        "MIN_RECORDS",
        "records an institution needs before its history is segmented",
        "the size at which a detected boundary stops depending on the sample",
    ),
    (
        "mds_norm.pipeline.institutional_fingerprints",
        "EFF_PROP",
        "the coverage step a field must take for a regime boundary to be kept",
        "changes of practice an institution can date from its own records",
    ),
    (
        "mds_norm.pipeline.institutional_fingerprints",
        "MIN_STRATUM_RECORDS",
        "records each side of a boundary, so no stratum is a sliver",
        "the size at which a stratum's own conventions become readable",
    ),
    (
        "mds_norm.pipeline.institutional_fingerprints",
        "CUT_DIST",
        "the distance at which the institution dendrogram is cut into clusters",
        "institutions known to share a cataloguing practice",
    ),
    (
        "mds_norm.pipeline.probe_scan",
        "PROBE_COVERAGE",
        "share of a group's values its probes must cover",
        "the coverage at which the probes stop finding new spans in prose",
    ),
    (
        "mds_norm.pipeline.probe_scan",
        "JOIN_SHARE_MAX",
        "joining-word share below which a value reads as a list rather than prose",
        "values a reader calls a list or a description",
    ),
    (
        "mds_norm.pipeline.probe_scan",
        "MATCH_SHARE_MIN",
        "share of a value inside matched spans for the same reading",
        "the same reader judgements",
    ),
    (
        "mds_norm.pipeline.vocab_alignment",
        "ATTEST_THRESHOLD",
        "share of split parts that must occur alone before a separator is accepted",
        "values a reader judges to be one term or several",
    ),
    (
        "mds_norm.pipeline.vocab_alignment",
        "MIN_SUPPORT_RECORDS",
        "records containing a separator before it can be accepted for an institution",
        "the support at which the acceptance stops changing",
    ),
    (
        "mds_norm.pipeline.institutional_vocab_detect",
        "CONFIRM_COV",
        "coverage of a published list that confirms an institution catalogues from it",
        "institutions asked which list they use",
    ),
    (
        "mds_norm.pipeline.institutional_vocab_detect",
        "SPECIFIC_MIN_TYPES",
        "terms exclusive to a small list before it is credited",
        "the same answers",
    ),
    (
        "mds_norm.pipeline.uncertainty_census",
        "STYLE_MIN_RATE",
        "how often a marker must appear before it counts as part of an institution's style",
        "cataloguers asked which marks they use",
    ),
    (
        "mds_norm.pipeline.extraction",
        "MIN_TEXT_CHARS",
        "free text a record needs before the model is asked to read it",
        "the yield curve of the model against record length",
    ),
    (
        "experiments.method_comparisons",
        "BAR",
        "precision a component must reach before its proposals are applied rather than qualified",
        "nothing: this is the project's stated tolerance for a wrong change",
    ),
]


def table() -> pl.DataFrame:
    rows = []
    for module_name, attribute, decides, replacement in CONSTANTS:
        module = importlib.import_module(module_name)
        rows.append(
            {
                "constant": f"{module_name.split('.')[-1]}.{attribute}",
                "value": str(getattr(module, attribute)),
                "decides": decides,
                "would_be_replaced_by": replacement,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = table()
    frame.write_parquet(OUT / "hand_set_constants.parquet")
    (OUT / "hand_set_constants.json").write_text(json.dumps(frame.to_dicts(), indent=1), encoding="utf-8")
    log(f"{frame.height} hand-set constants → {OUT}")
    with pl.Config(tbl_rows=60, fmt_str_lengths=60, tbl_width_chars=200):
        print(frame)


if __name__ == "__main__":
    main()
