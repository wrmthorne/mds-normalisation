from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable

import polars as pl

from mds_norm.paths import FIELD_STATS, MOJIBAKE_REPAIRS, VOCAB_INDEXES

INDEX_DIR = VOCAB_INDEXES
OUT = MOJIBAKE_REPAIRS

TOKEN_RX = r"[^\s]*�[^\s]*"
# Decodable mojibake that is really a false friend
REF_BLOCKLIST = {"itís"}
# a reference token: letters plus the usual name joiners
REF_TOKEN_RX = r"[\w'’\-\.]*[^\x00-\x7F][\w'’\-\.]*"
STRIP_PUNCT = ",.;:()[]\"'“”‘’!?"
# Leading or trailing characters used to index reference tokens
ANCHOR_LEN = 2
# surviving text needed either side of the damage
MIN_EDGE_RESIDUE = 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def damaged_values() -> pl.DataFrame:
    return (
        pl.scan_parquet(FIELD_STATS)
        .filter(pl.col("value").str.contains("�", literal=True))
        .group_by("value")
        .agg(occ=pl.len())
        .collect(engine="streaming")
    )


def reference_tokens() -> set[str]:
    """Distinct non-ASCII-bearing tokens from undamaged corpus values and the authority indexes"""
    corpus = (
        pl.scan_parquet(FIELD_STATS)
        .filter(pl.col("value").str.contains(r"[^\x00-\x7F]") & ~pl.col("value").str.contains("�", literal=True))
        .select(pl.col("value"))
        .unique()
        .select(tok=pl.col("value").str.extract_all(REF_TOKEN_RX))
        .explode("tok")
        .drop_nulls()
        .unique()
        .collect(engine="streaming")
    )
    toks = set(corpus["tok"].to_list())
    for path in sorted(INDEX_DIR.glob("*.parquet")):
        lf = pl.scan_parquet(path)
        if "term" not in lf.collect_schema().names():
            continue
        terms = (
            lf.filter(pl.col("term").str.contains(r"[^\x00-\x7F]"))
            .select(tok=pl.col("term").str.extract_all(REF_TOKEN_RX))
            .explode("tok")
            .drop_nulls()
            .unique()
            .collect(engine="streaming")
        )
        toks |= set(terms["tok"].to_list())
    return {t.strip(STRIP_PUNCT) for t in toks if t.strip(STRIP_PUNCT)} - REF_BLOCKLIST


def build_matcher(tokens: set[str]) -> Callable[[str], list[str]]:
    """Index reference tokens by 2-char prefix and suffix for wildcard lookup"""
    by_prefix, by_suffix = defaultdict(list), defaultdict(list)
    for t in tokens:
        by_prefix[t[:ANCHOR_LEN]].append(t)
        by_suffix[t[-ANCHOR_LEN:]].append(t)

    def candidates(core: str) -> list[str]:
        pre = core.split("�", 1)[0]
        suf = core.rsplit("�", 1)[-1]
        if len(pre) >= ANCHOR_LEN:
            pool = by_prefix.get(pre[:ANCHOR_LEN], [])
        elif len(suf) >= ANCHOR_LEN:
            pool = by_suffix.get(suf[-ANCHOR_LEN:], [])
        else:
            return []  # nothing to anchor on — stays damaged
        if any(c.isdigit() for c in core):
            return []  # '180�' lost a symbol (°), not a letter
        # Each � is one lost byte; letters only
        parts = []
        for piece in re.split("(�+)", core):
            if piece and set(piece) == {"�"}:
                n = len(piece)
                parts.append(r"([^\W\d_])" if n == 1 else rf"([^\W\d_]{{1,{n}}}?)")
            else:
                parts.append(re.escape(piece))
        rx = re.compile("^" + "".join(parts) + "$")
        out = []
        for t in pool:
            m = rx.match(t)
            if m and all(g and not g.isascii() for g in m.groups()):
                out.append(t)
        return sorted(set(out))

    return candidates


def main() -> None:
    dmg = damaged_values()
    log(f"{dmg.height:,} distinct damaged values ({int(dmg['occ'].sum()):,} occ)")
    refs = reference_tokens()
    log(f"{len(refs):,} reference tokens (non-ASCII, corpus + authorities)")
    candidates = build_matcher(refs)

    token_repair: dict[str, str | None] = {}
    quotes = set("\"'“”‘’`")

    def repair_token(tok: str, quote_adjacent: bool) -> str | None:
        core = tok.strip(STRIP_PUNCT)
        # An edge � needs solid residue and no quote nearby
        if (core.startswith("�") or core.endswith("�")) and (
            quote_adjacent or len(core.replace("�", "")) <= MIN_EDGE_RESIDUE
        ):
            return None
        if tok not in token_repair:
            lead = tok[: len(tok) - len(tok.lstrip(STRIP_PUNCT))]
            trail = tok[len(tok.rstrip(STRIP_PUNCT)) :]
            cands = candidates(core) if core else []
            token_repair[tok] = lead + cands[0] + trail if len(cands) == 1 else None
        return token_repair[tok]

    rows = []
    for value, occ in dmg.iter_rows():
        repairs = []
        for m in re.finditer(TOKEN_RX, value):
            prev_c = value[m.start() - 1] if m.start() else ""
            next_c = value[m.end()] if m.end() < len(value) else ""
            near_quote = bool((set(prev_c) | set(next_c) | set(m.group())) & quotes)
            repairs.append((m.group(), repair_token(m.group(), near_quote)))
        if repairs and all(r is not None for _, r in repairs):
            repaired = value
            for t, r in repairs:
                repaired = repaired.replace(t, r)
            rows.append((value, repaired, occ))

    out = pl.DataFrame(rows, schema={"value": pl.String, "repaired": pl.String, "occ": pl.UInt32}, orient="row")
    out.write_parquet(OUT)
    out.sort("occ", descending=True).write_csv(OUT.with_suffix(".csv"))
    n_occ = int(out["occ"].sum()) if out.height else 0
    log(f"repaired {out.height:,}/{dmg.height:,} distinct values ({n_occ:,}/{int(dmg['occ'].sum()):,} occ) -> {OUT}")


if __name__ == "__main__":
    main()
