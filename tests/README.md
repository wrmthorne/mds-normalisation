# tests

`uv sync --group dev && uv run pytest`. The suite is fast (no corpus access —
metrics tests build tiny frames inline) and must stay green after every
refactor.

## Parser regression cases

The parser tests are data-driven: each parser has a JSON case table in
`cases/` (`parse_dates.json`, `parse_dimensions.json`, …) loaded by
`case_loader.py`. A case gives a `raw` value and the `expected` record keyed
by field name — omitted fields read as the parser default, and `null` means
the parser must refuse. Tables can be flat (`cases: [...]`) or grouped
(`groups: [...]`), each group carrying the corpus observation that motivated
it. **Add a case by editing the JSON; no test code changes are needed.**

Gold labels frozen as cases land in `data/gold/cases/` in the same shape, so
labelled gold can graduate into regression cases.

## Layout

- `test_parse_*.py` — one per parser, driven by the case tables.
- `test_atomise.py`, `test_patches.py` — the `mds_norm.utils` modules.
- `test_metrics_*.py` — the metric axes over hand-built micro-frames.
- `test_compile_records.py`, `test_places_pipeline.py`,
  `test_institutional_priors.py`, `test_local_termlists.py` — pipeline-stage
  units.
- Imports are plain package imports (`from mds_norm.parsers...`,
  `from experiments...`); the only pytest path entry is `tests/` itself, for
  `case_loader`.
