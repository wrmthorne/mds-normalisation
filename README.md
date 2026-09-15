# MDS Normalisation

This repo provides a pipeline to normalise (make data in the same fields express their values in the same way) the Museum Data Service corpus: 166 million metadata nodes from 123 UK institutions (data collected on 6th April 2026). Every value is routed to the cheapest sufficient method: first, deterministic parsers; then, string and embedding matching; and finally, small language models only for what is left. Each stage writes to a saved external parquet (sidecar), rather than mutating any values in-place. Only when the corpus is compiled from the raw data are those changes applied, so every change is reversible and carries its provenance.

The data model for this work is documented in a separate [Github repository](https://github.com/wrmthorne/mds-data-model).

Common functionality (evaluation, findability, metrics, parsers, pipeline, utils, plotting) is written into a python module, built in this repository. Notebooks covering the main concepts of the analysis and the pipeline are given in [notebooks/](notebooks/).

## 1. Setup

> All instructions expect the use of systemd linux. Code _may_ run on Windows, but it is untested and unsupported.

uv is the preferred package manager ([install instructions](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
uv sync        # dependencies plus mds_norm and experiments, installed editable
uv run pytest  # needs no corpus data
```

## 2. Dataset preparation

The pipeline reads two files. `data/mds-flat-records.parquet` is a node table with one row per metadata assertion (`record_id`, `node_id`, `parent_id`, `depth`, `source_array_pos`, `label`, `path`, `field_type`, `value`, `extra`), and `data/mds-record-admin.parquet` holds the rest of each record's `@admin` block, one row per record, joined on `record_id`. Together they carry everything the export sends: 20,000 records rebuilt from the pair match the source JSON exactly. Anything a future export adds that the schema does not model arrives as JSON in `extra` rather than being dropped, and where the export re-emits a record that changed mid-download, the newest copy is the one kept. The compiled corpus is the same table with the compile's columns appended. Building them from the Museum Data Service takes two steps.

1. Download the records - MDS serves its data through a resume-token API. The [mds-exporter](https://github.com/wrmthorne/mds-exporter) library is recommended to run the export. More information can be found in the project's README.

```bash
uvx mds-exporter --token YOUR_MDS_TOKEN --compress --output mds-records.jsonl.zst
```

2. Ingest - The export is one JSON record per line, nested up to four levels. `scripts/ingest.py` reads it and writes both tables:

```bash
uv run scripts/ingest.py mds-records.jsonl.zst
```

That reads the compressed export directly. If you also want the records in their original nested shape, write it first and ingest from there instead — the flat output is identical either way:

```bash
uv run python scripts/ingest.py mds-records.jsonl.zst data/mds-records.parquet --form nested
uv run python scripts/ingest.py data/mds-records.parquet
```

### 2a. Downloading supplementary data

The vocabularies the alignment stage matches against can all be downloaded from one script. This download can take a long time:

```bash
chmod +x ./data/vocabularies/fetch_vocabs.sh
./data/vocabularies/fetch_vocabs.sh
```

Also download the mapping museums data:

```bash
wget https://museweb.dcs.bbk.ac.uk/static/pdf/MappingMuseumsData2021_09_30.csv -O ./data/reference/MappingMuseumsData2021_09_30.csv
```

## 3. Running the pipeline

The node table, built by the ingest described under [Exporting the corpus](#exporting-the-corpus-from-mds), and the authority releases fetched by the [vocabulary script](#downloading-supplementary-data), must exist before running anything.

Every stage is a module with a `main()` and a `--help`. Each writes a the proposed changes to a separate file under `data/analysis_output/` and mutates nothing. Run in this order, and apply the memory cap shown under [Compiling the dataset](#compiling-the-dataset) where scripts scan the whole corpus.

### 3a. Model endpoints

Steps 5 and 9 are the only ones that call a model; the rest are CPU and disk. Both expect an OpenAI-compatible server on localhost, and match on the served model name:

| Stage                                 | Port  | Served model name | Approx. VRAM |
|---------------------------------------|-------|-------------------|--------------|
| step 5, `vocab_alignment` tier 4      | 30001 | `LFM2.5-350M`     | 1-4GB or CPU |
| step 5, `vocab_alignment` rerank rung | 30000 | `gpt-oss-20b`     | 17-20GB      |
| step 9, `extraction run`              | 30000 | `gpt-oss-20b`     | 17-20GB      |

1. Tier 0 and the census — character repair, Unicode normalisation, whitespace trimming and pattern induction over every node. Needed for all subsequent stages.

```bash
uv run python -m mds_norm.pipeline.field_census
```

2. Authority indexes — the term indexes the alignment cascade matches against, the TGN parent chains that anchor place disambiguation, and the corpus-derived local termlists.

```bash
uv run python -m mds_norm.pipeline.vocab_indexes
uv run python -m mds_norm.pipeline.build_parent_chains
uv run python -m mds_norm.pipeline.build_local_termlists
```

3. Institutional conventions — the per-family pattern artefacts induced from the census, then the per-institution date and unit conventions and the practice strata the routing reads.

```bash
uv run python -m mds_norm.pipeline.pattern_exports
uv run python -m mds_norm.pipeline.accession_schemes
uv run python -m mds_norm.pipeline.institutional_priors
uv run python -m mds_norm.pipeline.institutional_fingerprints
uv run python -m mds_norm.pipeline.practice_boundaries      # tested boundaries in field use by acquisition year
```

Without the first, the others cannot run, and the compile falls back to empty conventions rather than failing. `accession_schemes` decides which institutions' object numbers carry an accession year, which the practice strata are segmented on. `--case-comparison` on `pattern_exports` writes the pattern counts under both maskings and exits, without touching the shipped artefacts.

The readings the analysis chapter takes from the same artefacts, each writing to `analysis_output/` and needed by no later stage:

```bash
uv run python -m mds_norm.pipeline.divergence_tests            # how far each institution's slots diverge
uv run python -m mds_norm.pipeline.uncertainty_census          # how each institution marks doubt
uv run python -m mds_norm.pipeline.vocabulary_fragmentation    # spelling variety and working vocabularies
uv run python -m mds_norm.evaluation.remediation               # where remediation would pay
```

4. Character repairs — replacement candidates for values carrying U+FFFD, attested against the corpus's own undamaged tokens and the authority indexes.

```bash
uv run python -m mds_norm.pipeline.build_mojibake_repairs
```

5. Vocabulary alignment — the exact and fuzzy cascade, then a retrieval-plus-model rerank rung. `institutional_vocab_detect` reads the first pass to work out which institutions catalogue a field from a published list of their own, so the cascade runs again with those lists read ahead of the field's pooled targets. `apply_homograph_verdicts` then applies the labelled rung calibration as a join, which is why it needs no third run.

> Expects `LiquidAI/LFM2.5-350M` to be running at `localhost:30001` and `openai/gpt-oss-20b` at `localhost:30000`

```bash
uv run python -m mds_norm.pipeline.vocab_alignment    # --skip-llm to run tier 4 off
uv run python -m mds_norm.pipeline.institutional_vocab_detect --field material
uv run python -m mds_norm.pipeline.vocab_alignment
uv run python -m mds_norm.pipeline.apply_homograph_verdicts
```

6. Places — TGN first, anchored on the hierarchy, then the queues TGN cannot answer against OS Open Names and GeoNames. `apply_place_verdicts` folds in the labelled rung calibration, keyed by gazetteer as well as rung, and rewrites both sidecars.

```bash
uv run python -m mds_norm.pipeline.places_pipeline
uv run python -m mds_norm.pipeline.places_fallback
uv run python -m mds_norm.pipeline.apply_place_verdicts
```

7. Agent names — `persons` parses each agent value into its name components, then `agent_links` matches the parsed names against the ULAN and ISNI indexes and defers the ambiguous ones. Both write to `data/analysis_output/persons/`. Skipping them costs the compile its person sidecars and nothing else.

```bash
uv run python -m mds_norm.pipeline.persons
uv run python -m mds_norm.pipeline.agent_links
```

8. Probe scan — structured content sitting in free text, every candidate span verified by the group's own tier-1 parser.

```bash
uv run python -m mds_norm.pipeline.probe_scan
```

Reading the same candidates per institution rather than per value finds columns an institution has mapped onto the wrong field. Diagnostic only — it writes no sidecar.

```bash
uv run python -m mds_norm.pipeline.misplacement_scan
```

9. Record-level extraction — the only stage that talks to a model endpoint and the only long one. It journals each completed shard, so stopping it costs at most the shard in flight, and `status` prints the projected remaining time and energy.

> Expects `openai/gpt-oss-20b` to be running at `localhost:30000`

```bash
uv run python -m mds_norm.pipeline.extraction prepare
uv run python -m mds_norm.pipeline.extraction run
uv run python -m mds_norm.pipeline.extraction status
uv run python -m mds_norm.pipeline.extraction finalise
```

`cache` keys the replies already on disk by the record content that earned them, so `run --reuse` calls the endpoint only for records whose values or task list changed. The key ignores the order rows arrive in, so node ids may churn without costing a call.

```bash
uv run python -m mds_norm.pipeline.extraction prepare
uv run python -m mds_norm.pipeline.extraction cache
uv run python -m mds_norm.pipeline.extraction run --reuse
```

Replies recorded before the key existed carry no hash, and the rows they were rendered from are gone. `--from-snapshot` re-keys them against the current queue, keeping a reply only where the record still renders a prompt of the same length.

Serving the model from another machine leaves its energy unmeasured, so `run` refuses a non-local `--base-url` unless the agent is running beside the endpoint. Start it on the serving host, then point the run at it:

```bash
python -m mds_norm.utils.energy serve --country-iso-code GBR             # on the serving host, port 8770
python -m mds_norm.utils.energy check http://<serving-host>:8770         # what it can measure

uv run python -m mds_norm.pipeline.extraction run \
    --base-url http://<serving-host>:30000/v1 --energy-url http://<serving-host>:8770
```

`--allow-unmetered` runs without the agent, recording client-side energy only, so the serving host's GPU draw is missing from the totals.

## Compiling the dataset

The compiler merges every sidecar into the compiled corpus. It is deterministic and re-runnable, and it is the only stage that writes values.

```bash
# Run one of these commands, not both
uv run python -m mds_norm.pipeline.compile_records

# (Optional) guard compile against OOM errors
# Replace 48G with <=75% of physical memory - only tested with 48G
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 \
    .venv/bin/python3 -m mds_norm.pipeline.compile_records
```

`--cc0` writes `mds-normalised_CC0.parquet` beside the full output, holding only the nodes of records whose licence unit says CC0.


## Brief overview

Published figures are frozen artefacts. Metric weights live in `data/analysis_output/metrics/` and are never re-estimated when measuring the compiled corpus; evaluation summaries and experiment results sit beside them under `data/analysis_output/`, and the figures are built from those files.

Every stage emits rows with one of four dispositions:

- `resolved`: a confident normalisation, applied by the compiler.
- `rejected`: a candidate was considered and refused; nothing changes.
- `deferred`: this stage could not decide, and the row names the more expensive stage that should consume it.
- `flagged`: the value passes through unchanged with an explicit uncertainty annotation.

Escalation between methods is the movement of deferred rows from one stage's output into the next stage's input. The compiler applies the decisions stored in the tables in `data/analysis_output/`, keeps the original in `as_recorded` wherever a value changed, and logs conflicts where two stages propose for the same node.
