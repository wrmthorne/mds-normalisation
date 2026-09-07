# Method experiments

Controlled comparisons over frozen gold sets, mostly of the language-model rungs: which model and prompt scale to use for extraction, whether a model beats the deterministic atomiser, which edit representation to send, and whether conditioning on institutional convention is worth wiring.

Everything shares `harness.py` — fixed decoding per variant (`variants.py`), codecarbon around every configuration, seeds and git state journalled per run. Experiment emissions go to their own log, separate from the production one, so a rerun never lands in the pipeline's cost table.

Models are served one at a time, so each runner is invoked once per served model. All runners take `--limit N` for a smoke test against a live endpoint.

## Run sheet

The gold sets ship already labelled under `data/gold/`: the sampling frames, the drawn samples, the labels, and the frozen cases the harness reads. The tools that drew and judged them are not part of this distribution, so the run sheet starts at the comparisons themselves.

Order matters in one place: the extraction and representation gold is judged over the *pooled* outputs of every variant, so those predictions must exist before that gold can be judged.

**1. Atomiser variants.** The deterministic baseline needs no endpoint; then one run per model:

```bash
uv run python -m experiments.atomiser_variants --variant deterministic
uv run python -m experiments.atomiser_variants --variant lfm2.5-350m
uv run python -m experiments.atomiser_variants --variant qwen3-1.7b
uv run python -m experiments.atomiser_variants --variant gpt-oss-20b
uv run python -m experiments.atomiser_variants --variant gpt-oss-20b:everywhere
uv run python -m experiments.atomiser_variants --score
```

**2. Extraction and representation predictions**, then pool them into judging candidates:

```bash
uv run python -m experiments.extraction_variants --variant gpt-oss-20b:single_task_v2
uv run python -m experiments.extraction_variants --variant qwen3-1.7b:composed
uv run python -m experiments.extraction_variants --variant mechanical     # no endpoint
uv run python -m experiments.representation_variants --variant rep:json_patch
uv run python -m experiments.representation_variants --variant rep:diff
uv run python -m experiments.representation_variants --variant rep:fields_only
uv run python -m experiments.pool_ops
```

`variants.py` lists the full variant set; the calls above are the shape.

**3. Score against the pooled gold.** Judging the pool means accepting or rejecting each candidate and adding whatever every variant missed; with the shipped labels in place, score directly:

```bash
uv run python -m experiments.score_extraction
```

**Convention-prior routing** needs no endpoint and no labels for its volume axes, but it scans the raw corpus:

```bash
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 \
    uv run python -m experiments.fingerprint_routing
uv run python -m experiments.regime_precision
```

**Which representation identifies an institution** runs the fingerprinting variants. Each writes to `analysis_output/experiments/fingerprinting/`; the regime variants need `practice_strata.parquet`, so export the strata first:

```bash
uv run python -m mds_norm.pipeline.institutional_fingerprints --only strata
systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 \
    uv run python -m experiments.fingerprinting
```

**The comparison table** consolidates every variant of every comparison here into one long-form table, written as both parquet and LaTeX. It reads artefacts only, so rerun it after any rescoring:

```bash
uv run python -m experiments.method_comparisons
```