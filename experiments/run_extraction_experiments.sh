#!/bin/bash


uv run python -m experiments.atomiser_variants --variant gpt-oss-20b
uv run python -m experiments.atomiser_variants --variant gpt-oss-20b:everywhere
uv run python -m experiments.atomiser_variants --score

uv run python -m experiments.extraction_variants --variant gpt-oss-20b:composed
uv run python -m experiments.extraction_variants --variant gpt-oss-20b:single_task
uv run python -m experiments.extraction_variants --variant gpt-oss-20b:composed_v2
uv run python -m experiments.extraction_variants --variant gpt-oss-20b:single_task_v2
uv run python -m experiments.score_extraction

uv run python -m experiments.representation_variants --variant rep:json_patch
uv run python -m experiments.representation_variants --variant rep:diff
uv run python -m experiments.representation_variants --variant rep:fields_only
uv run python -m experiments.pool_ops