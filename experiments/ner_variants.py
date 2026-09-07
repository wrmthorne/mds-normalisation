from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mds_norm.paths import EXP_OUT, GOLD_SAMPLES

SAMPLE = GOLD_SAMPLES / "ner.jsonl"
OUT = EXP_OUT / "ner"

# The eight types the NER codebook defines
TYPES = ["person", "organisation", "place", "date", "material", "technique", "object_type", "measurement"]

# OntoNotes to codebook; unmapped tags are dropped, never forced
SPACY_MAP = {
    "PERSON": "person",
    "ORG": "organisation",
    "GPE": "place",
    "LOC": "place",
    "FAC": "place",
    "DATE": "date",
    "EVENT": "date",
    "QUANTITY": "measurement",
    "PRODUCT": "object_type",
    "WORK_OF_ART": "object_type",
}

# GLiNER2 is prompted with readable label names
GLINER_LABELS = {
    "person": "person",
    "organisation": "organisation",
    "place": "place",
    "date": "date",
    "material": "material",
    "technique": "technique",
    "object type": "object_type",
    "measurement": "measurement",
}

WINDOW_WORDS = 350
BATCH_SIZE = 32


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_items() -> list[dict]:
    if not SAMPLE.exists():
        raise SystemExit(f"no sample at {SAMPLE} — draw the NER sample first")
    return [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_predictions(variant: str, rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{variant}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"{variant}: {len(rows)} spans over {len({r['id'] for r in rows})} items → {path}")
    return path


def run_spacy(items: list[dict], model: str = "en_core_web_sm") -> list[dict]:
    import spacy

    nlp = spacy.load(model, exclude=["lemmatizer", "attribute_ruler", "tagger", "parser"])
    rows = []
    texts = [it["value"] for it in items]
    for it, doc in zip(items, nlp.pipe(texts, batch_size=64), strict=True):
        for ent in doc.ents:
            etype = SPACY_MAP.get(ent.label_)
            if etype is None:
                continue
            rows.append(
                {
                    "id": it["id"],
                    "variant": model,
                    "type": etype,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "raw_label": ent.label_,
                }
            )
    return rows


def _windows(text: str) -> list[tuple[int, str]]:
    """(char offset, chunk) windows of at most WINDOW_WORDS words, so offsets stay mappable"""
    words, spans, pos = text.split(), [], 0
    if len(words) <= WINDOW_WORDS:
        return [(0, text)]
    for w in words:  # locate each word in the original string, in order
        i = text.index(w, pos)
        spans.append((i, i + len(w)))
        pos = i + len(w)
    out = []
    for i in range(0, len(words), WINDOW_WORDS):
        chunk = spans[i : i + WINDOW_WORDS]
        out.append((chunk[0][0], text[chunk[0][0] : chunk[-1][1]]))
    return out


def run_gliner2(items: list[dict], model: str = "fastino/gliner2-base-v1", threshold: float = 0.5) -> list[dict]:
    import torch
    from gliner2 import GLiNER2

    m = GLiNER2.from_pretrained(model).cuda().eval().half()
    labels = list(GLINER_LABELS)

    jobs = [(it["id"], off, chunk) for it in items for off, chunk in _windows(it["value"])]
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i][2]))
    texts = [jobs[i][2] for i in order]
    log(f"gliner2: {len(jobs)} windows over {len(items)} items")

    with torch.inference_mode():
        results = m.batch_extract_entities(
            texts, labels, batch_size=BATCH_SIZE, threshold=threshold, include_spans=True
        )

    rows = []
    for pos, res in zip(order, results, strict=True):
        item_id, off, chunk = jobs[pos]
        for label, hits in (res.get("entities") or {}).items():
            etype = GLINER_LABELS.get(label)
            if etype is None:
                continue
            for hit in hits:
                # include_spans gives offsets; bare strings fall back to search
                if isinstance(hit, dict):
                    text, s = hit.get("text", ""), hit.get("start")
                    e = hit.get("end", (s + len(text)) if s is not None else None)
                else:
                    text = hit
                    s = chunk.find(text)
                    e = s + len(text) if s >= 0 else None
                if s is None or e is None or s < 0:
                    continue
                rows.append(
                    {
                        "id": item_id,
                        "variant": "gliner2-base-v1",
                        "type": etype,
                        "start": off + s,
                        "end": off + e,
                        "text": text,
                        "raw_label": label,
                    }
                )
    return rows


VARIANTS = {"spacy_sm": run_spacy, "gliner2": run_gliner2}


def pool(items: list[dict]) -> list[dict]:
    """Union the variants' spans onto each item, deduplicated on (type, start, end)"""
    pools: dict[str, dict[tuple, dict]] = {}
    n_variants = 0
    for variant in VARIANTS:
        path = OUT / f"{variant}.jsonl"
        if not path.exists():
            log(f"pool: no predictions for {variant}, skipping")
            continue
        n_variants += 1
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            key = (r["type"], r["start"], r["end"])
            entry = pools.setdefault(r["id"], {}).setdefault(
                key, {"type": r["type"], "start": r["start"], "end": r["end"], "text": r["text"], "variants": []}
            )
            if variant not in entry["variants"]:
                entry["variants"].append(variant)
    if not n_variants:
        raise SystemExit(f"no variant predictions under {OUT} — run `ner_variants.py run` first")

    n_with = 0
    for it in items:
        pool_i = pools.get(it["id"], {})
        it["candidates"] = sorted(pool_i.values(), key=lambda c: (c["start"], c["type"]))
        n_with += bool(it["candidates"])
    log(
        f"pooled {sum(len(p) for p in pools.values())} distinct spans from "
        f"{n_variants} variants; {n_with}/{len(items)} items carry candidates"
    )
    return items


def write_sample(items: list[dict]) -> None:
    tmp = SAMPLE.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(SAMPLE)
    log(f"rewrote {SAMPLE}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "pool"])
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="variants to run (default: all)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    items = load_items()
    if args.mode == "run":
        for variant in args.variant or sorted(VARIANTS):
            fn = VARIANTS[variant]
            rows = fn(items, threshold=args.threshold) if variant == "gliner2" else fn(items)
            write_predictions(variant, rows)
    else:
        write_sample(pool(items))


if __name__ == "__main__":
    main()
