from experiments.harness import Variant

BASE_URL = "http://localhost:30000/v1"

# thinking off for the small chat models, low for gpt-oss
_NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _by_name(variants: list[Variant]) -> dict[str, Variant]:
    return {v.name: v for v in variants}


ATOMISER_VARIANTS = _by_name(
    [
        Variant(
            "deterministic",
            notes="expanded fast path: replays the per-(group, institution) "
            "attested separators (frames/atomiser_separators.parquet)",
        ),
        Variant(
            "lfm2.5-350m",
            model="LFM2.5-350M",
            base_url=BASE_URL,
            concurrency=256,
            decoding={"temperature": 0.1, "extra_body": {"top_k": 50, "repetition_penalty": 1.05}},
            notes="production configuration (cascade tier 4)",
        ),
        Variant(
            "lfm2.5-8b-a1b",
            model="LFM2.5-8B-A1B",
            base_url=BASE_URL,
            concurrency=256,
            decoding={"temperature": 0.1, "extra_body": {"top_k": 50, "repetition_penalty": 1.05}},
            notes="production configuration (cascade tier 4)",
        ),
        Variant("qwen3-1.7b", model="Qwen3-1.7B", base_url=BASE_URL, concurrency=256, decoding={"temperature": 0}),
        Variant(
            "gpt-oss-20b",
            model="gpt-oss-20b",
            base_url=BASE_URL,
            concurrency=200,
            decoding={"temperature": 0.2, "top_p": 1.0, "reasoning_effort": "low"},
        ),
        Variant(
            "gpt-oss-20b:everywhere",
            model="gpt-oss-20b",
            base_url=BASE_URL,
            concurrency=200,
            decoding={"temperature": 0.2, "top_p": 1.0, "reasoning_effort": "low"},
            params={"composition": "everywhere"},
            notes="expensive-everywhere variant: the LLM splits every "
            "gold value whole; no deterministic fast path, no guards",
        ),
    ]
)


_EXTRACTION_DECODING = {
    "gpt-oss-20b": {"temperature": 0.2, "top_p": 1.0, "reasoning_effort": "low"},
    "Qwen3-1.7B": {"temperature": 0.0, **_NOTHINK},
}

EXTRACTION_VARIANTS = _by_name(
    [
        Variant(
            f"{model.lower()}:{scale}",
            model=model,
            base_url=BASE_URL,
            concurrency=100 if model == "gpt-oss-20b" else 256,
            decoding=_EXTRACTION_DECODING[model],
            params={"prompt_scale": scale, "representation": "json_patch"},
            notes="production configuration (extraction tier)"
            if model == "gpt-oss-20b" and scale == "composed"
            else "",
        )
        for model in ("gpt-oss-20b", "Qwen3-1.7B")
        for scale in ("composed", "single_task")
    ]
    + [
        Variant(
            f"gpt-oss-20b:{scale}_v2",
            model="gpt-oss-20b",
            base_url=BASE_URL,
            concurrency=100,
            decoding=_EXTRACTION_DECODING["gpt-oss-20b"],
            params={"prompt_scale": scale, "representation": "json_patch", "prompt_rev": "v2"},
            notes="task contract v2 from the extraction FP analysis: bare "
            "material heads, object-aboutness rule, non-material and "
            "non-production-date exclusions",
        )
        for scale in ("composed", "single_task")
    ]
    + [
        Variant(
            "mechanical",
            notes="mechanical-op baseline: replays the probe-scan "
            "refine/novel ops on the gold records, restricted to each "
            "item's missing tasks; no LLM. Materials have no probe group, "
            "so material tasks go structurally unanswered",
        )
    ]
)


REPRESENTATION_VARIANTS = _by_name(
    [
        Variant(
            f"rep:{rep}",
            model="gpt-oss-20b",
            base_url=BASE_URL,
            concurrency=100,
            decoding=_EXTRACTION_DECODING["gpt-oss-20b"],
            params={"prompt_scale": "composed", "representation": rep},
        )
        for rep in ("json_patch", "diff", "fields_only")
    ]
)
