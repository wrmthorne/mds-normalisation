# mds_norm

The library and pipeline, installed editable by `uv sync` alongside `experiments/`. Everything imports as `mds_norm.*`; there are no `sys.path` tricks in the repository.

| Subpackage | Contents |
|---|---|
| `paths.py` | Every file location in the repository. Modules anchor here rather than computing paths from `__file__`. |
| `tables.py` | What the node table holds beyond what a stage derives: `record_nodes` fetches a record's substructure in document order, `source_only` gives a generated node the export-only columns, empty. |
| `parsers/` | Deterministic value parsers: dates to EDTF, dimensions, counts, money including £sd, cataloguer uncertainty notation, and agent names. Pure functions, regression-tested from `tests/cases/`. |
| `metrics/` | The four quality axes — completeness, thinness, conformance, consistency — plus the shared loader. Each is a pure `compute(base)`; all I/O belongs to the caller. |
| `utils/` | Atomisation and term normalisation, the async client for the local model endpoint, patch primitives and the validation gate, the remote energy agent, and the markup definition. |
| `pipeline/` | The corpus-building stages, in run order: `field_census`, `probe_scan`, `vocab_indexes` and `vocab_alignment`, `places_pipeline` and `places_fallback`, `persons` and `agent_links`, `extraction`, then `compile_records`. Plus the supporting builders (vocabularies, termlists, mojibake repairs, parent chains, institutional priors and fingerprints). |
| `evaluation/` | `quality_deltas` (the axes before and after, on frozen weights), `reports` (coverage, weight freezing, tier-0/1 energy, queue projections), `residue_scan`, and `validate_records` (records rebuilt from the node table and validated against the `mds_data_model` pydantic schema). |
| `findability/` | The retrieval benchmark: `retrieval_benchmark` builds the symmetric indexes and answers queries, `retrieval_pool` pools results for blind judging. |

## Running a stage

Every stage is a module with a `main()`:

```bash
uv run python -m mds_norm.pipeline.compile_records --help
uv run python -m mds_norm.evaluation.reports --help
```

Corpus-scale stages run under the memory cap:

```bash
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 \
    .venv/bin/python3 -m mds_norm.pipeline.compile_records
```

`extraction` is the only long-running stage and the only one that talks to a model endpoint, so it splits into commands and journals each completed shard — stopping it costs at most the shard in flight:

```bash
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 \
    .venv/bin/python3 -m mds_norm.pipeline.extraction prepare
uv run python -m mds_norm.pipeline.extraction run        # needs the endpoint
uv run python -m mds_norm.pipeline.extraction status
uv run python -m mds_norm.pipeline.extraction finalise
```

## Conventions

- No module mutates a source value. Stages emit sidecars and the compiler applies them.
- Artefacts land under `data/` or `data/analysis_output/`, always through `paths`, never cwd-relative.
- Heavy or optional dependencies (torch, sentence-transformers, codecarbon) are imported inside the functions that need them, so importing a module stays cheap.
- Modules carry no header docstring. A function docstring is one line naming what it returns, with no closing full stop. A comment explains a *why* the code cannot show, in ten words or fewer, and never points at a document outside the repository.
