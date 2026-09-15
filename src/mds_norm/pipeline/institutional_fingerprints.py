from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from codecarbon import track_emissions
from mds_data_model.introspection import all_free_text_fields
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import jensenshannon, squareform

from mds_norm.metrics import kiraly
from mds_norm.paths import EMISSIONS_LOG, INSTITUTIONAL, PATTERNS_OUT, RAW_RECORDS
from mds_norm.pipeline.accession_schemes import field_years
from mds_norm.utils.atomise import NULL_MARKERS
from mds_norm.utils.masking import MASK, signature_of

RAW_PATH = RAW_RECORDS
PATTERNS = PATTERNS_OUT
OUT_DIR = INSTITUTIONAL
EMISSIONS_LOG_PATH = EMISSIONS_LOG

# Regime boundary: an accession year where field coverage steps
MIN_RECORDS, MIN_YEARS = 500, 8
MIN_N, ALPHA, MAX_DEPTH = 150, 0.01, 6
EFF_PROP = 0.30
ADOPT_LO, ADOPT_HI = 0.05, 0.98
# Fewer year buckets than this leaves no split point
MIN_SPLIT_YEARS = 2
# Benjamini-Hochberg level for candidate shifts, above per-test ALPHA
BH_ALPHA = 0.05

# Steps merge within +-1yr; strata need records either side
TOL, MIN_STRATUM_RECORDS = 1, 200

# JS clustering: average-linkage dendrogram cut at a fixed distance
CUT_DIST, MIN_CLUSTER = 0.4, 3
# Member names a cluster's log line spells out
CLUSTER_LOG_MEMBERS = 6

# Null-marker candidate floors
NULL_SHARE_MIN, NULL_OCC_MIN, NULL_REUSE_FIELDS, NULL_REUSE_OCC = 0.05, 50, 3, 10

# Past this length style is just length; halves normalised separately
FP_ALPHABET, FP_MAX_VALUE_CHARS = 2000, 64
FP_KINDS = ("occupancy", "morphology", "combined")
# Floor keeping an unseen feature's Jensen-Shannon term finite
FP_SMOOTHING = 1e-9


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def accession_years() -> pl.DataFrame:
    """One accession year per dated record: the accession date where recorded, else the acquisition date"""
    # Years read from object numbers are not used: they are inferred, and the strata must rest on recorded dates
    sources = [
        ("accession_date", field_years("spectrum/accession_date")),
        ("acquisition_date", field_years("spectrum/acquisition_date")),
    ]
    pri = pl.when(pl.col("year_source") == "accession_date").then(0).otherwise(1)
    return (
        pl.concat([df.with_columns(year_source=pl.lit(src)) for src, df in sources])
        .with_columns(_pri=pri)
        .group_by("record_id")
        .agg(
            pl.col("data_source").first(),
            accession_year=pl.col("year").sort_by("_pri").first(),
            year_source=pl.col("year_source").sort_by("_pri").first(),
        )
    )


# A side scorer also takes the half it reports on
type SegmentScorer = Callable[[int, int, int], float]
type SideScorer = Callable[[int, int, int, str], float]


def _prefix(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(a, dtype=float))])


def _records_in(yr_n: dict[int, int], a: int, b: int) -> int:
    """Records accessioned between two years, both ends included"""
    return sum(v for y, v in yr_n.items() if a <= y <= b)


def binseg(totals: np.ndarray, eff_fn: SegmentScorer, p_fn: SegmentScorer, min_eff: float) -> list[dict]:
    """Binary segmentation over per-year totals"""
    cum = _prefix(totals)
    events = []

    def rec(lo: int, hi: int, depth: int) -> None:
        if depth >= MAX_DEPTH or hi - lo < MIN_SPLIT_YEARS:
            return
        best_s, best_e = None, min_eff
        for s in range(lo + 1, hi):
            if cum[s] - cum[lo] < MIN_N or cum[hi] - cum[s] < MIN_N:
                continue
            e = eff_fn(lo, s, hi)
            if e > best_e:
                best_e, best_s = e, s
        if best_s is not None and (p := p_fn(lo, best_s, hi)) < ALPHA:
            events.append(
                {
                    "split_idx": best_s,
                    "lo": lo,
                    "hi": hi,
                    "effect": float(best_e),
                    "p": float(p),
                    "n_before": int(cum[best_s] - cum[lo]),
                    "n_after": int(cum[hi] - cum[best_s]),
                }
            )
            rec(lo, best_s, depth + 1)
            rec(best_s, hi, depth + 1)

    rec(0, len(totals), 0)
    return events


def prop_scorers(hits: np.ndarray, totals: np.ndarray) -> tuple[SegmentScorer, SegmentScorer, SideScorer]:
    """|delta coverage| effect + two-proportion z p-value"""
    ck, cn = _prefix(hits), _prefix(totals)

    def parts(lo: int, s: int, hi: int) -> tuple[float, float, float, float]:
        return ck[s] - ck[lo], cn[s] - cn[lo], ck[hi] - ck[s], cn[hi] - cn[s]

    def eff(lo: int, s: int, hi: int) -> float:
        kl, nl, kr, nr = parts(lo, s, hi)
        return abs(kr / nr - kl / nl) if nl >= 1 and nr >= 1 else -1.0

    def pval(lo: int, s: int, hi: int) -> float:
        kl, nl, kr, nr = parts(lo, s, hi)
        pp = (kl + kr) / (nl + nr)
        se = np.sqrt(pp * (1 - pp) * (1 / nl + 1 / nr))
        if se == 0:
            return 1.0
        return float(2 * stats.norm.sf(abs((kr / nr - kl / nl) / se)))

    def prop(lo: int, s: int, hi: int, side: str) -> float:
        kl, nl, kr, nr = parts(lo, s, hi)
        return kl / nl if side == "L" else kr / nr

    return eff, pval, prop


def consensus(rows: list[dict]) -> list[dict]:
    """Cluster one institution's steps by year into consensus events, weighted by effect x log10(min side)"""
    rows = sorted(rows, key=lambda e: e["split_year"])
    clusters, cur = [], None
    for e in rows:
        w = e["effect"] * np.log10(max(10, min(e["n_before"], e["n_after"])))
        if cur is None or e["split_year"] - cur["last"] > TOL:
            cur = {"wsum": 0.0, "wyear": 0.0, "sigs": [], "last": None}
            clusters.append(cur)
        cur["wsum"] += w
        cur["wyear"] += w * e["split_year"]
        cur["last"] = e["split_year"]
        cur["sigs"].append((e["signal"], e["effect"]))
    out = []
    for c in clusters:
        top = "; ".join(f"{s.split('/')[-1]}({e:.2f})" for s, e in sorted(c["sigs"], key=lambda x: -x[1])[:5])
        out.append(
            {
                "event_year": round(c["wyear"] / c["wsum"]),
                "evidence": round(c["wsum"], 2),
                "n_fields": len(c["sigs"]),
                "top_fields": top,
            }
        )
    return sorted(out, key=lambda x: -x["evidence"])


def export_strata() -> None:
    rec_year = accession_years()
    rec_year.write_parquet(OUT_DIR / "accession_years.parquet")
    log(
        f"records assigned an accession year: {rec_year.height:,} "
        f"({dict(rec_year.group_by('year_source').len().iter_rows())})"
    )

    nodes = (
        pl.scan_parquet(RAW_PATH)
        .filter(pl.col("value").is_not_null() & (pl.col("value").str.strip_chars().str.len_chars() > 0))
        .join(rec_year.lazy().select("record_id", "accession_year"), on="record_id")
    )

    richness = (
        nodes.group_by("record_id")
        .agg(
            pl.col("data_source").cast(pl.String).first(),
            pl.col("accession_year").first(),
            n_fields=pl.len(),
            n_distinct_fields=pl.col("field_type").n_unique(),
            total_chars=pl.col("value").str.len_chars().sum(),
        )
        .collect(engine="streaming")
    )
    field_year = (
        nodes.group_by(pl.col("data_source").cast(pl.String), "accession_year", "field_type")
        .agg(n_with_field=pl.col("record_id").n_unique())
        .collect(engine="streaming")
    )
    log(f"richness rows: {richness.height:,}; field-year: {field_year.height:,}")
    # the adoption evidence itself, for per-institution coverage heatmaps
    field_year.write_parquet(OUT_DIR / "field_coverage_by_year.parquet")

    cohort = (
        richness.group_by("data_source", "accession_year")
        .agg(pl.len().alias("n"))
        .sort("data_source", "accession_year")
    )

    inst_summary = cohort.group_by("data_source").agg(
        records=pl.col("n").sum(),
        year_buckets=pl.len(),
        y0=pl.col("accession_year").min(),
        y1=pl.col("accession_year").max(),
    )
    analyse = (
        inst_summary.filter((pl.col("records") >= MIN_RECORDS) & (pl.col("year_buckets") >= MIN_YEARS))
        .sort("records", descending=True)["data_source"]
        .to_list()
    )
    log(f"{len(analyse)} institutions meet the {MIN_RECORDS}/{MIN_YEARS} gate")

    adopt_fields = (
        field_year.group_by("data_source", "field_type")
        .agg(k=pl.col("n_with_field").sum())
        .join(inst_summary.select("data_source", "records"), on="data_source")
        .filter((pl.col("k") / pl.col("records")).is_between(ADOPT_LO, ADOPT_HI))
    )

    events: list[dict] = []
    for inst in analyse:
        coh = cohort.filter(pl.col("data_source") == inst).sort("accession_year")
        years = coh["accession_year"].to_numpy()
        yidx = {int(y): i for i, y in enumerate(years)}
        n_year = coh["n"].to_numpy().astype(float)

        cov_inst = field_year.filter(pl.col("data_source") == inst)
        for f in adopt_fields.filter(pl.col("data_source") == inst)["field_type"]:
            k_year = np.zeros(len(years))
            for r in cov_inst.filter(pl.col("field_type") == f).iter_rows(named=True):
                k_year[yidx[int(r["accession_year"])]] = r["n_with_field"]
            eff, pval, prop = prop_scorers(k_year, n_year)
            for e in binseg(n_year, eff, pval, EFF_PROP):
                s = e["split_idx"]
                events.append(
                    {
                        "data_source": inst,
                        "signal": f,
                        "split_year": int(years[s]),
                        "effect": e["effect"],
                        "p": e["p"],
                        "before": str(round(prop(e["lo"], s, e["hi"], "L"), 3)),
                        "after": str(round(prop(e["lo"], s, e["hi"], "R"), 3)),
                        "n_before": e["n_before"],
                        "n_after": e["n_after"],
                    }
                )

    ev = pl.DataFrame(events).sort("p")
    log(f"raw candidate shifts: {ev.height:,}")
    p = ev["p"].to_numpy()
    p_bh = np.minimum.accumulate((p * ev.height / np.arange(1, ev.height + 1))[::-1])[::-1].clip(max=1.0)
    ev = (
        ev.with_columns(p_bh=pl.Series(p_bh))
        .filter(pl.col("p_bh") < BH_ALPHA)
        .with_columns(
            direction=pl.when(
                pl.col("after").cast(pl.Float64, strict=False) >= pl.col("before").cast(pl.Float64, strict=False)
            )
            .then(pl.lit("increase"))
            .otherwise(pl.lit("decrease"))
        )
    )
    ev.write_parquet(OUT_DIR / "practice_shift_events.parquet")
    log(f"significant shifts: {ev.height:,} across {ev['data_source'].n_unique()} institutions")

    consensus_rows, strata_rows = [], []
    for inst in ev["data_source"].unique():
        cevs = consensus(ev.filter(pl.col("data_source") == inst).to_dicts())
        consensus_rows += [{"data_source": inst, **c} for c in cevs]

        info = inst_summary.filter(pl.col("data_source") == inst)
        y0, y1 = int(info["y0"][0]), int(info["y1"][0])
        coh = cohort.filter(pl.col("data_source") == inst)
        yr_n = dict(zip(coh["accession_year"], coh["n"], strict=True))
        # Greedy by evidence, and only where both segments stay substantive
        accepted: list[int] = []
        for c in cevs:
            b = c["event_year"]
            if not (y0 < b <= y1):
                continue
            left = max([y0] + [x for x in accepted if x < b])
            right = min([y1 + 1] + [x for x in accepted if x > b])
            if (
                _records_in(yr_n, left, b - 1) >= MIN_STRATUM_RECORDS
                and _records_in(yr_n, b, right - 1) >= MIN_STRATUM_RECORDS
            ):
                accepted.append(b)
        edges = [y0, *sorted(accepted), y1 + 1]
        rich_i = richness.filter(pl.col("data_source") == inst)
        fy_i = field_year.filter(pl.col("data_source") == inst)
        for idx, (a, b) in enumerate(pairwise(edges)):
            seg = rich_i.filter(pl.col("accession_year").is_between(a, b - 1))
            if seg.height == 0:
                continue
            top_fields = (
                fy_i.filter(pl.col("accession_year").is_between(a, b - 1))
                .group_by("field_type")
                .agg(k=pl.col("n_with_field").sum())
                .sort("k", descending=True)
                .head(6)["field_type"]
            )
            strata_rows.append(
                {
                    "data_source": inst,
                    "stratum_idx": idx,
                    "start_year": a,
                    "end_year": b - 1,
                    "n_records": seg.height,
                    "mean_distinct_fields": round(seg["n_distinct_fields"].mean(), 2),
                    "mean_total_chars": round(seg["total_chars"].mean(), 1),
                    "top_fields": ", ".join(f.split("/")[-1] for f in top_fields),
                }
            )

    (
        pl.DataFrame(consensus_rows)
        .sort("data_source", "event_year")
        .write_parquet(OUT_DIR / "practice_consensus_events.parquet")
    )
    strata = pl.DataFrame(strata_rows).sort("data_source", "start_year")
    strata.write_parquet(OUT_DIR / "practice_strata.parquet")
    log(
        f"{len(consensus_rows)} consensus events; {strata.height} strata across "
        f"{strata['data_source'].n_unique()} institutions"
    )


def export_clusters() -> None:
    """Pairwise JS distance over combined merged-pattern distributions"""
    suffix = "_institutional_pattern_dist.parquet"
    families = sorted(p.name[: -len(suffix)] for p in PATTERNS.glob(f"*{suffix}"))
    log(f"pattern families: {families}")
    dist = pl.concat(
        [
            pl.read_parquet(PATTERNS / f"{f}{suffix}").with_columns(
                merged_pattern=pl.lit(f"{f}:") + pl.col("merged_pattern")
            )
            for f in families
        ]
    )
    dist = dist.with_columns(prob=pl.col("prob") / pl.col("prob").sum().over("data_source"))

    sources = sorted(dist["data_source"].unique())
    patterns = {p: j for j, p in enumerate(dist["merged_pattern"].unique().sort())}
    prob_matrix = np.zeros((len(sources), len(patterns)))
    for r in dist.iter_rows(named=True):
        prob_matrix[sources.index(r["data_source"]), patterns[r["merged_pattern"]]] = r["prob"]

    n = len(sources)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = 0.5 * (prob_matrix[i] + prob_matrix[j])
            with np.errstate(divide="ignore", invalid="ignore"):
                kl_pm = np.nansum(np.where(prob_matrix[i] > 0, prob_matrix[i] * np.log2(prob_matrix[i] / m), 0.0))
                kl_qm = np.nansum(np.where(prob_matrix[j] > 0, prob_matrix[j] * np.log2(prob_matrix[j] / m), 0.0))
            distances[i, j] = distances[j, i] = float(np.sqrt(max(0.0, 0.5 * (kl_pm + kl_qm))))

    linkage_matrix = linkage(squareform(distances, checks=False), method="average")
    log(f"top merge heights: {[round(h, 3) for h in sorted(linkage_matrix[:, 2].tolist(), reverse=True)[:8]]}")
    labels = fcluster(linkage_matrix, t=CUT_DIST, criterion="distance")
    sizes = Counter(int(lab) for lab in labels)
    big = {lab: rank + 1 for rank, (lab, cnt) in enumerate(sizes.most_common()) if cnt >= MIN_CLUSTER}
    # leaf order written beside the labels, for heatmap ordering
    order = {int(s): rank for rank, s in enumerate(leaves_list(linkage_matrix))}
    clusters = pl.DataFrame(
        {
            "data_source": sources,
            "cluster": [big.get(int(lab)) for lab in labels],
            "leaf_order": [order[i] for i in range(len(sources))],
        }
    )
    clusters.write_parquet(OUT_DIR / "js_clusters.parquet")

    pairs = [(sources[i], sources[j], distances[i, j]) for i in range(n) for j in range(i + 1, n)]
    (
        pl.DataFrame(pairs, schema=["source_a", "source_b", "js_distance"], orient="row").write_parquet(
            OUT_DIR / "js_distances.parquet"
        )
    )
    log(
        f"{len(sources)} institutions; "
        f"{clusters['cluster'].drop_nulls().n_unique()} clusters of >= {MIN_CLUSTER} "
        f"({clusters['cluster'].is_null().sum()} unclustered)"
    )
    for cid in sorted(big.values()):
        members = clusters.filter(pl.col("cluster") == cid)
        log(
            f"  cluster {cid} (n={members.height}): "
            + "; ".join(members["data_source"].head(CLUSTER_LOG_MEMBERS))
            + (" …" if members.height > CLUSTER_LOG_MEMBERS else "")
        )


def _fingerprint_counts(rows: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-institution occurrence counts over field types, and over masked value shapes"""
    descriptive = rows.filter(
        pl.col("value").is_not_null()
        & ~pl.col("field_type").str.starts_with("ciim/")
        & ~pl.col("field_type").str.starts_with("wrmthorne/")
    )
    occupancy = descriptive.group_by("data_source", "field_type").len(name="c").collect(engine="streaming")
    masked = (
        descriptive.filter(pl.col("value").str.len_chars().is_between(1, FP_MAX_VALUE_CHARS))
        .select("data_source", MASK.alias("mask"))
        .collect(engine="streaming")
    )
    shapes = (
        masked.select("mask")
        .unique()
        .with_columns(pl.col("mask").map_elements(signature_of, return_dtype=pl.String).alias("shape"))
    )
    morphology = masked.join(shapes, on="mask", how="left").group_by("data_source", "shape").len(name="c")
    return occupancy, morphology


def _distribution(counts: pl.DataFrame, key: str, sources: list[str], vocab: list[str]) -> np.ndarray:
    """Row-normalised institution-by-vocabulary matrix, aligned to the given orders"""
    sidx, vidx = {s: i for i, s in enumerate(sources)}, {v: i for i, v in enumerate(vocab)}
    matrix = np.zeros((len(sources), len(vocab)))
    for row in counts.filter(pl.col(key).is_in(vocab)).iter_rows(named=True):
        matrix[sidx[row["data_source"]], vidx[row[key]]] = row["c"]
    matrix += FP_SMOOTHING
    return matrix / matrix.sum(axis=1, keepdims=True)


def _js_matrix(dist: np.ndarray) -> np.ndarray:
    n = len(dist)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = float(jensenshannon(dist[i], dist[j], base=2))
    return out


def export_fingerprints(path: Path = RAW_PATH) -> None:
    """Per-institution fingerprints and their nearest neighbours under each of the three kinds"""
    occupancy, morphology = _fingerprint_counts(pl.scan_parquet(path))
    sources = sorted(set(occupancy["data_source"]) & set(morphology["data_source"]))
    field_vocab = sorted(set(occupancy["field_type"]))
    shape_vocab = (
        morphology.group_by("shape")
        .agg(pl.col("c").sum())
        .sort("c", descending=True)
        .head(FP_ALPHABET)["shape"]
        .to_list()
    )
    log(f"fingerprints over {len(sources)} institutions: {len(field_vocab)} fields, {len(shape_vocab)} value shapes")

    occ = _distribution(occupancy, "field_type", sources, field_vocab)
    mor = _distribution(morphology, "shape", sources, shape_vocab)
    # Halving keeps a distribution and weights the two halves equally
    dists = {"occupancy": occ, "morphology": mor, "combined": np.hstack([occ / 2, mor / 2])}

    frames = []
    for kind, dist in dists.items():
        d = _js_matrix(dist)
        np.fill_diagonal(d, np.inf)
        order = d.argsort(axis=1)
        frames.append(
            pl.DataFrame(
                {
                    "kind": [kind] * (len(sources) * len(sources[1:])),
                    "data_source": [s for s in sources for _ in sources[1:]],
                    "neighbour": [sources[j] for i in range(len(sources)) for j in order[i][: len(sources) - 1]],
                    "rank": [r + 1 for _ in sources for r in range(len(sources) - 1)],
                    "js_distance": [float(d[i][j]) for i in range(len(sources)) for j in order[i][: len(sources) - 1]],
                }
            )
        )
    neighbours = pl.concat(frames)
    neighbours.write_parquet(OUT_DIR / "fingerprint_neighbours.parquet")

    vocab = {"occupancy": field_vocab, "morphology": shape_vocab}
    pl.concat(
        [
            pl.DataFrame(
                {
                    "kind": kind,
                    "data_source": [s for s in sources for _ in vocab[kind]],
                    "feature": vocab[kind] * len(sources),
                    "prob": dists[kind].reshape(-1).tolist(),
                }
            )
            for kind in ("occupancy", "morphology")
        ]
    ).filter(pl.col("prob") > FP_SMOOTHING).write_parquet(OUT_DIR / "fingerprints.parquet")

    for kind in FP_KINDS:
        near = neighbours.filter((pl.col("kind") == kind) & (pl.col("rank") == 1))
        log(f"  {kind}: median nearest-neighbour distance {near['js_distance'].median():.3f}")


def export_null_markers() -> None:
    """Per-(institution, field) disguised-null candidates"""
    free_text = {n for names in all_free_text_fields().values() for n in names}
    vals = (
        pl.scan_parquet(RAW_PATH)
        .filter(
            pl.col("value").is_not_null()
            & ~pl.col("field_type").is_in(list(free_text))
            & (pl.col("field_type") != "spectrum/object_name")
        )
        .select(
            pl.col("data_source").cast(pl.String),
            "field_type",
            value_norm=pl.col("value").str.strip_chars().str.to_lowercase(),
        )
        .filter(pl.col("value_norm") != "")
        .group_by("data_source", "field_type", "value_norm")
        .agg(occ=pl.len())
        .collect(engine="streaming")
    )

    vals = vals.with_columns(
        share=pl.col("occ") / pl.col("occ").sum().over("data_source", "field_type"),
        n_fields_reused=(
            pl.col("field_type").filter(pl.col("occ") >= NULL_REUSE_OCC).n_unique().over("data_source", "value_norm")
        ),
        seed_match=pl.col("value_norm").is_in(sorted(NULL_MARKERS)),
    )
    cand = vals.filter(
        pl.col("seed_match")
        | (
            (pl.col("share") >= NULL_SHARE_MIN)
            & (pl.col("occ") >= NULL_OCC_MIN)
            & (pl.col("n_fields_reused") >= NULL_REUSE_FIELDS)
        )
    ).sort("data_source", "field_type", "occ", descending=[False, False, True])

    # Placeholders should score near zero on Ochoa-Duval information content
    corpus = pl.scan_parquet(RAW_PATH).with_columns(
        pl.col("value").str.strip_chars().str.to_lowercase().alias("value")
    )
    ic = kiraly.information_content(
        kiraly.value_frequencies(corpus, restrict_to=cand.lazy().select("field_type", value=pl.col("value_norm")))
    )
    cand = cand.join(ic, left_on=["field_type", "value_norm"], right_on=["field_type", "value"], how="left")
    cand.write_parquet(OUT_DIR / "null_marker_candidates.parquet")
    log(f"  median information content of a candidate marker: {cand['ic'].median():.4f}")
    log(
        f"null-marker candidates: {cand.height:,} (institution, field, value) rows "
        f"({cand.filter(~pl.col('seed_match')).height:,} beyond the seed list) "
        f"across {cand['data_source'].n_unique()} institutions"
    )


@track_emissions(project_name="institutional_fingerprints", output_dir=str(EMISSIONS_LOG_PATH), log_level="error")
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Institutional practice strata, cluster distances and null-marker candidates."
    )
    ap.add_argument(
        "--only",
        choices=["strata", "clusters", "nulls", "fingerprints"],
        action="append",
        help="run a subset of the exports",
    )
    args = ap.parse_args()
    which = args.only or ["strata", "clusters", "nulls", "fingerprints"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "clusters" in which:
        export_clusters()
    if "fingerprints" in which:
        export_fingerprints()
    if "nulls" in which:
        export_null_markers()
    if "strata" in which:
        export_strata()


if __name__ == "__main__":
    main()
