from __future__ import annotations

from pathlib import Path


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"no pyproject.toml above {here}; cannot locate the repository root")


ROOT = _find_root()

DATA = ROOT / "data"
RAW_RECORDS = DATA / "mds-flat-records.parquet"
# Everything in @admin, one row per record, keyed by record_id
RECORD_ADMIN = DATA / "mds-record-admin.parquet"
COMPILED = DATA / "compiled"
RECORD_INDEX = COMPILED / "record_index.parquet"
INSTITUTIONAL = DATA / "institutional"
VOCABS = DATA / "vocabularies"
# vendored data the corpus cannot supply
REFERENCE = DATA / "reference"
MAPPING_MUSEUMS = REFERENCE / "MappingMuseumsData2021_09_30.csv"
INSTITUTION_REGISTRY = REFERENCE / "institution_registry.parquet"
INSTITUTION_PSEUDONYMS = REFERENCE / "institution_pseudonyms.json"

GOLD = DATA / "gold"
GOLD_FRAMES = GOLD / "frames"
GOLD_SAMPLES = GOLD / "samples"
GOLD_CONTEXT = GOLD / "context"
GOLD_LABELS = GOLD / "labels"
GOLD_CASES = GOLD / "cases"
GOLD_RETRIEVAL = GOLD / "retrieval"

# named per stage; a moved output changes one line
ANALYSIS_OUTPUT = DATA / "analysis_output"

FIELD_STATS = ANALYSIS_OUTPUT / "field_stats.parquet"
PROBE_CANDIDATES = ANALYSIS_OUTPUT / "probe_candidates.parquet"
PROBE_CANDIDATES_RAW = ANALYSIS_OUTPUT / "probe_candidates_raw.parquet"
MOJIBAKE_REPAIRS = ANALYSIS_OUTPUT / "mojibake_repairs.parquet"

MISPLACEMENT_OUT = ANALYSIS_OUTPUT / "misplacement"
MISPLACEMENT_PAIRS = MISPLACEMENT_OUT / "field_pairs.parquet"
MISPLACEMENT_HITS = MISPLACEMENT_OUT / "hits.parquet"
MISPLACEMENT_EXAMPLES = MISPLACEMENT_OUT / "examples.parquet"

EMISSIONS_LOG = ANALYSIS_OUTPUT / "emissions_logs"
EMISSIONS_CSV = EMISSIONS_LOG / "emissions.csv"

VOCAB_OUT = ANALYSIS_OUTPUT / "vocabularies"
VOCAB_INDEXES = VOCAB_OUT / "indexes"
VOCAB_LOCAL = VOCAB_OUT / "local"
VOCAB_INSTITUTIONAL = VOCAB_OUT / "institutional"
VOCAB_DECISIONS = VOCAB_OUT / "vocab_value_decisions.parquet"
VOCAB_ANNOTATIONS = VOCAB_OUT / "vocab_annotations.parquet"

PERSONS_OUT = ANALYSIS_OUTPUT / "persons"
PERSON_DECISIONS = PERSONS_OUT / "person_value_decisions.parquet"
PERSON_ANNOTATIONS = PERSONS_OUT / "person_annotations.parquet"
PERSON_LINKS = PERSONS_OUT / "person_authority_links.parquet"

PLACES_OUT = ANALYSIS_OUTPUT / "places"
PLACE_ANNOTATIONS = PLACES_OUT / "place_annotations.parquet"
PLACE_FALLBACK_ANNOTATIONS = PLACES_OUT / "place_fallback_annotations.parquet"

RECORD_FIXES_OUT = ANALYSIS_OUTPUT / "record_fixes"
RECORD_PATCHES = RECORD_FIXES_OUT / "record_patches.parquet"
LLM_RESPONSES = RECORD_FIXES_OUT / "llm_responses.parquet"

CONSISTENCY_OUT = ANALYSIS_OUTPUT / "consistency"
PATTERNS_OUT = ANALYSIS_OUTPUT / "patterns"
METRICS_OUT = ANALYSIS_OUTPUT / "metrics"

EVAL_OUT = ANALYSIS_OUTPUT / "evaluation"
RETRIEVAL_OUT = EVAL_OUT / "retrieval"
EXP_OUT = ANALYSIS_OUTPUT / "experiments"
