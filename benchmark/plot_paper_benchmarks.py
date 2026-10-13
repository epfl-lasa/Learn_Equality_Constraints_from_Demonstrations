from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Iterable

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_METHODS = ["dataaug", "oncl", "vae", "ecomann"]
DEFAULT_METRICS = [
    "proj_manifold_dist",
    "pred_precision",
    "train_seconds",
    "gt_to_learned_mean",
    "learned_to_gt_mean",
    "bidirectional_chamfer",
]

DEFAULT_BAR_YMAX = 0.5

LOWER_IS_BETTER_METRICS = {
    "proj_manifold_dist",
    "gt_to_learned_mean",
    "learned_to_gt_mean",
    "bidirectional_chamfer",
    "train_seconds",
}

HIGHER_IS_BETTER_METRICS = {
    "pred_precision",
}


def _split_csv(text: str | None, default: Iterable[str] | None = None) -> list[str]:
    if text is None or str(text).strip() == "":
        return list(default) if default is not None else []
    return [s.strip() for s in str(text).split(",") if s.strip()]


def _resolve_bench_dir(bench: str) -> str:
    b = str(bench).strip()
    if not b:
        raise ValueError("--bench is required")
    if os.path.isabs(b):
        return b
    p1 = os.path.abspath(os.path.join(_PROJECT_ROOT, "outputs", "bench", b))
    if os.path.isdir(p1):
        return p1
    p2 = os.path.abspath(os.path.join(_PROJECT_ROOT, b))
    if os.path.isdir(p2):
        return p2
    return os.path.abspath(b)


def _to_float(v: object, default: float = np.nan) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return x


def _load_from_per_case_csv(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "dataset": str(r.get("dataset", "")).strip(),
                    "method": str(r.get("method", "")).strip(),
                    "seed": str(r.get("seed", "")).strip() or "0",
                    "bidirectional_chamfer": _to_float(r.get("bidirectional_chamfer", "")),
                    "proj_manifold_dist": _to_float(r.get("proj_manifold_dist", "")),
                    "pred_precision": _to_float(r.get("pred_precision", "")),
                    "train_seconds": _to_float(r.get("train_seconds", "")),
                    "gt_to_learned_mean": _to_float(r.get("gt_to_learned_mean", "")),
                    "learned_to_gt_mean": _to_float(r.get("learned_to_gt_mean", "")),
                }
            )
    return rows


def _load_from_per_run_jsonl(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            metrics = r.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            rows.append(
                {
                    "dataset": str(r.get("dataset", "")).strip(),
                    "method": str(r.get("method", "")).strip(),
                    "seed": str(r.get("seed", "")).strip() or "0",
                    "bidirectional_chamfer": _to_float(metrics.get("bidirectional_chamfer", r.get("bidirectional_chamfer", ""))),
                    "proj_manifold_dist": _to_float(metrics.get("proj_manifold_dist", r.get("proj_manifold_dist", ""))),
                    "pred_precision": _to_float(metrics.get("pred_precision", r.get("pred_precision", ""))),
                    "train_seconds": _to_float(metrics.get("train_seconds", r.get("train_seconds", ""))),
                    "gt_to_learned_mean": _to_float(metrics.get("gt_to_learned_mean", r.get("gt_to_learned_mean", ""))),
                    "learned_to_gt_mean": _to_float(metrics.get("learned_to_gt_mean", r.get("learned_to_gt_mean", ""))),
                }
            )
    return rows


def _load_from_eval_jsons(bench_dir: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_dir in sorted(glob.glob(os.path.join(bench_dir, "*"))):
        if not os.path.isdir(method_dir):
            continue
        method = os.path.basename(method_dir)
        for p in sorted(glob.glob(os.path.join(method_dir, "**", "*_eval.json"), recursive=True)):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            parent = os.path.basename(os.path.dirname(p))
            if os.path.abspath(os.path.dirname(p)) != os.path.abspath(method_dir):
                dataset = parent
                stem = os.path.basename(p)[: -len("_eval.json")]
                seed = "last"
                marker = f"_{method}_seed"
                if marker in stem:
                    seed = stem.rsplit(marker, 1)[1]
                else:
                    marker = "_seed"
                    if marker in stem:
                        seed = stem.rsplit(marker, 1)[1]
            else:
                dataset = os.path.basename(p)
                if dataset.endswith("_eval.json"):
                    dataset = dataset[: -len("_eval.json")]
                if dataset.endswith(f"_{method}"):
                    dataset = dataset[: -(len(method) + 1)]
                dataset = dataset.replace("_oncl", "")
                seed = "last"
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "bidirectional_chamfer": _to_float(d.get("bidirectional_chamfer", "")),
                    "proj_manifold_dist": _to_float(d.get("proj_manifold_dist", "")),
                    "pred_precision": _to_float(d.get("pred_precision", "")),
                    "train_seconds": _to_float(d.get("train_seconds", "")),
                    "gt_to_learned_mean": _to_float(d.get("gt_to_learned_mean", "")),
                    "learned_to_gt_mean": _to_float(d.get("learned_to_gt_mean", "")),
                }
            )
    return rows


def _load_rows(bench_dir: str) -> tuple[list[dict[str, object]], str]:
    per_run = os.path.join(bench_dir, "per_run_metrics.jsonl")
    if os.path.isfile(per_run):
        return _load_from_per_run_jsonl(per_run), "per_run_metrics.jsonl"
    per_case = os.path.join(bench_dir, "per_case_metrics.csv")
    if os.path.isfile(per_case):
        return _load_from_per_case_csv(per_case), "per_case_metrics.csv"
    return _load_from_eval_jsons(bench_dir), "*_eval.json"


def _build_samples(
    rows: list[dict[str, object]],
    datasets: list[str],
    methods: list[str],
    metric: str,
    fill_missing_zero: bool,
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {d: {m: [] for m in methods} for d in datasets}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        m = str(r.get("method", "")).strip()
        if d not in out or m not in out[d]:
            continue
        v = _to_float(r.get(metric, np.nan))
        if np.isfinite(v):
            out[d][m].append(float(v))
    if fill_missing_zero:
        for d in datasets:
            for m in methods:
                if len(out[d][m]) == 0:
                    out[d][m] = [0.0]
    return out


def _compute_mean_std(samples: dict[str, dict[str, list[float]]], datasets: list[str], methods: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((len(methods), len(datasets)), dtype=np.float64)
    stds = np.zeros((len(methods), len(datasets)), dtype=np.float64)
    for mi, m in enumerate(methods):
        for di, d in enumerate(datasets):
            vals = np.asarray(samples[d][m], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                means[mi, di] = 0.0
                stds[mi, di] = 0.0
            else:
                means[mi, di] = float(np.mean(vals))
                stds[mi, di] = float(np.std(vals, ddof=0))
    return means, stds


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, df: float) -> float:
    if not np.isfinite(t) or df <= 0:
        return np.nan
    x = df / (df + t * t)
    ib = _regularized_incomplete_beta(0.5 * df, 0.5, x)
    if t >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def _paired_t_pvalue_good_direction(diff_good_direction: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(diff_good_direction, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return np.nan, np.nan
    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        mean = float(np.mean(diff))
        if mean > 0:
            return math.inf, 0.0
        if mean < 0:
            return -math.inf, 1.0
        return 0.0, 0.5
    stat = float(np.mean(diff) / (sd / math.sqrt(diff.size)))
    p_value = 1.0 - _student_t_cdf(stat, diff.size - 1)
    return stat, float(min(max(p_value, 0.0), 1.0))


def _sig_star(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (not np.isfinite(p_values[i]), p_values[i]))
    adjusted = [np.nan] * n
    running = 0.0
    for k, idx in enumerate(order):
        p = float(p_values[idx])
        if not np.isfinite(p):
            adjusted[idx] = np.nan
            continue
        running = max(running, (n - k) * p)
        adjusted[idx] = min(1.0, running)
    return adjusted


def _average_tie_ranks(values: list[float], *, lower_is_better: bool) -> list[float]:
    indexed = [(i, float(v)) for i, v in enumerate(values)]
    indexed.sort(key=lambda iv: iv[1] if lower_is_better else -iv[1])
    ranks = [np.nan] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _exact_wilcoxon_signed_rank_one_sided(diff_good_direction: np.ndarray) -> tuple[float, float]:
    """Exact one-sided Wilcoxon signed-rank test for H1: paired differences > 0."""
    diff = np.asarray(diff_good_direction, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    diff = diff[np.abs(diff) > 1e-12]
    if diff.size == 0:
        return 0.0, 1.0
    abs_ranks = np.asarray(_average_tie_ranks([float(x) for x in np.abs(diff)], lower_is_better=True), dtype=np.float64)
    observed = float(np.sum(abs_ranks[diff > 0]))
    total = 1 << int(diff.size)
    ge = 0
    for mask in range(total):
        w_plus = 0.0
        for i in range(int(diff.size)):
            if (mask >> i) & 1:
                w_plus += float(abs_ranks[i])
        if w_plus >= observed - 1e-12:
            ge += 1
    return observed, float(ge / total)


def _compute_cross_dataset_rank_comparison(
    *,
    rows: list[dict[str, object]],
    datasets: list[str],
    methods: list[str],
    baselines: list[str],
    metric: str,
    alpha: float,
) -> dict[str, object]:
    if metric not in LOWER_IS_BETTER_METRICS and metric not in HIGHER_IS_BETTER_METRICS:
        return {}
    lower_is_better = metric in LOWER_IS_BETTER_METRICS
    means: dict[tuple[str, str], float] = {}
    ranks: dict[tuple[str, str], float] = {}
    used_datasets: list[str] = []

    for d in datasets:
        ds_means: list[float] = []
        ds_methods: list[str] = []
        for m in methods:
            vals = _vals_from_rows(rows, d, m, metric)
            if vals.size == 0:
                continue
            mu = float(np.mean(vals))
            if not np.isfinite(mu):
                continue
            means[(d, m)] = mu
            ds_means.append(mu)
            ds_methods.append(m)
        if len(ds_methods) != len(methods):
            continue
        ds_ranks = _average_tie_ranks(ds_means, lower_is_better=lower_is_better)
        for m, rank in zip(ds_methods, ds_ranks):
            ranks[(d, m)] = float(rank)
        used_datasets.append(d)

    average_ranks: dict[str, float] = {}
    for m in methods:
        vals = [ranks[(d, m)] for d in used_datasets if (d, m) in ranks]
        average_ranks[m] = float(np.mean(vals)) if vals else np.nan

    comparisons: list[dict[str, object]] = []
    raw_p: list[float] = []
    for b in baselines:
        if b == "oncl" or b not in methods:
            continue
        diffs = []
        for d in used_datasets:
            if (d, "oncl") not in ranks or (d, b) not in ranks:
                continue
            diffs.append(float(ranks[(d, b)] - ranks[(d, "oncl")]))
        diff_arr = np.asarray(diffs, dtype=np.float64)
        w_stat, p_value = _exact_wilcoxon_signed_rank_one_sided(diff_arr)
        raw_p.append(p_value)
        comparisons.append(
            {
                "baseline": b,
                "n_datasets": int(diff_arr.size),
                "oncl_average_rank": average_ranks.get("oncl", np.nan),
                "baseline_average_rank": average_ranks.get(b, np.nan),
                "mean_rank_diff_good_direction": float(np.mean(diff_arr)) if diff_arr.size else np.nan,
                "wilcoxon_w_plus": float(w_stat),
                "p_value": float(p_value),
            }
        )
    adjusted = _holm_adjust(raw_p)
    for r, p_adj in zip(comparisons, adjusted):
        r["holm_p_value"] = float(p_adj)
        r["star"] = _sig_star(float(p_adj)) if np.isfinite(float(p_adj)) and float(p_adj) < alpha else ""
        r["significant"] = bool(r["star"])

    return {
        "datasets": used_datasets,
        "methods": methods,
        "means": means,
        "ranks": ranks,
        "average_ranks": average_ranks,
        "comparisons": comparisons,
    }


def _save_rank_comparison_outputs(
    *,
    outdir: str,
    metric_stem: str,
    result: dict[str, object],
) -> tuple[str, str, str]:
    os.makedirs(outdir, exist_ok=True)
    datasets = list(result.get("datasets", []))
    methods = list(result.get("methods", []))
    ranks = result.get("ranks", {})
    avg_ranks = result.get("average_ranks", {})
    comparisons = list(result.get("comparisons", []))

    rank_csv = os.path.join(outdir, f"{metric_stem}_cross_dataset_ranks.csv")
    with open(rank_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "Dataset"] + [_to_display_method(m) for m in methods])
        for d in datasets:
            w.writerow(
                [d, _to_display_dataset(d)]
                + [f"{float(ranks[(d, m)]):.6g}" if (d, m) in ranks else "" for m in methods]
            )

    comparison_csv = os.path.join(outdir, f"{metric_stem}_cross_dataset_rank_comparison.csv")
    with open(comparison_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "baseline",
            "Baseline",
            "n_datasets",
            "oncl_average_rank",
            "baseline_average_rank",
            "mean_rank_diff_good_direction",
            "wilcoxon_w_plus",
            "p_value",
            "holm_p_value",
            "star",
            "significant",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in comparisons:
            row = dict(r)
            row["Baseline"] = _to_display_method(str(row.get("baseline", "")))
            w.writerow({k: row.get(k, "") for k in fieldnames})

    md_path = os.path.join(outdir, f"{metric_stem}_cross_dataset_rank_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("## Average ranks\n\n")
        f.write("| Method | Average rank | n datasets |\n")
        f.write("|---|---:|---:|\n")
        for m in sorted(methods, key=lambda x: float(avg_ranks.get(x, np.inf))):
            f.write(f"| {_to_display_method(m)} | {float(avg_ranks.get(m, np.nan)):.3f} | {len(datasets)} |\n")
        f.write("\n## ONCL rank comparisons\n\n")
        f.write("| Comparison | n datasets | ONCL avg rank | Baseline avg rank | Mean rank diff | W+ | p | Holm p | sig |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in comparisons:
            star = str(r.get("star", "")) if str(r.get("star", "")) else "n.s."
            f.write(
                f"| ONCL vs {_to_display_method(str(r['baseline']))} | "
                f"{int(r['n_datasets'])} | "
                f"{float(r['oncl_average_rank']):.3f} | "
                f"{float(r['baseline_average_rank']):.3f} | "
                f"{float(r['mean_rank_diff_good_direction']):+.3f} | "
                f"{float(r['wilcoxon_w_plus']):.3g} | "
                f"{float(r['p_value']):.4g} | "
                f"{float(r['holm_p_value']):.4g} | "
                f"{star} |\n"
            )
        f.write(
            "\nPositive mean rank difference means ONCL has a better, lower rank. "
            "P-values are one-sided exact Wilcoxon signed-rank tests across datasets; "
            "Holm p-values correct the ONCL-vs-baseline comparisons.\n"
        )

    return rank_csv, comparison_csv, md_path


def _paired_arrays_by_seed(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    method_a: str,
    method_b: str,
    metric: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    vals_a: dict[str, float] = {}
    vals_b: dict[str, float] = {}
    for r in rows:
        if str(r.get("dataset", "")).strip() != str(dataset):
            continue
        seed = str(r.get("seed", "")).strip()
        method = str(r.get("method", "")).strip()
        v = _to_float(r.get(metric, np.nan))
        if not seed or not np.isfinite(v):
            continue
        if method == method_a:
            vals_a[seed] = float(v)
        elif method == method_b:
            vals_b[seed] = float(v)
    common = sorted(set(vals_a) & set(vals_b), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
    return common, np.asarray([vals_a[s] for s in common], dtype=np.float64), np.asarray([vals_b[s] for s in common], dtype=np.float64)


def _compute_oncl_paired_t_significance(
    *,
    rows: list[dict[str, object]],
    datasets: list[str],
    baselines: list[str],
    metric: str,
    alpha: float,
) -> dict[tuple[str, str], dict[str, object]]:
    out: dict[tuple[str, str], dict[str, object]] = {}
    if metric not in LOWER_IS_BETTER_METRICS and metric not in HIGHER_IS_BETTER_METRICS:
        return out
    for d in datasets:
        for b in baselines:
            seeds, oncl_vals, base_vals = _paired_arrays_by_seed(
                rows,
                dataset=d,
                method_a="oncl",
                method_b=b,
                metric=metric,
            )
            if oncl_vals.size < 2 or base_vals.size < 2:
                continue
            if metric in LOWER_IS_BETTER_METRICS:
                diff = base_vals - oncl_vals
            else:
                diff = oncl_vals - base_vals
            t_stat, p_value = _paired_t_pvalue_good_direction(diff)
            star = _sig_star(p_value) if float(np.mean(diff)) > 0 and p_value < alpha else ""
            out[(d, b)] = {
                "dataset": d,
                "baseline": b,
                "n": int(len(seeds)),
                "oncl_mean": float(np.mean(oncl_vals)),
                "baseline_mean": float(np.mean(base_vals)),
                "mean_diff_good_direction": float(np.mean(diff)),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "star": star,
                "significant": bool(star),
            }
    return out


def _save_significance_csv(path: str, sig: dict[tuple[str, str], dict[str, object]]) -> None:
    if not sig:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "dataset",
        "Dataset",
        "baseline",
        "Baseline",
        "n",
        "oncl_mean",
        "baseline_mean",
        "mean_diff_good_direction",
        "t_stat",
        "p_value",
        "star",
        "significant",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d, b in sorted(sig.keys()):
            r = sig[(d, b)]
            row = dict(r)
            row["Dataset"] = _to_display_dataset(d)
            row["Baseline"] = _to_display_method(b)
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _save_significance_md(path: str, sig: dict[tuple[str, str], dict[str, object]]) -> None:
    if not sig:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["Dataset", "Baseline", "n", "ONCL", "Baseline mean", "Delta", "t", "p", "sig"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for d, b in sorted(sig.keys()):
            r = sig[(d, b)]
            f.write(
                "| "
                + " | ".join(
                    [
                        _to_display_dataset(d),
                        _to_display_method(b),
                        str(r["n"]),
                        f"{float(r['oncl_mean']):.4f}",
                        f"{float(r['baseline_mean']):.4f}",
                        f"{float(r['mean_diff_good_direction']):+.4f}",
                        f"{float(r['t_stat']):.4g}",
                        f"{float(r['p_value']):.4g}",
                        str(r["star"]) if str(r["star"]) else "n.s.",
                    ]
                )
                + " |\n"
            )


def _plot_grouped_bar_with_error(
    out_path: str,
    title: str,
    ylabel: str,
    datasets: list[str],
    methods: list[str],
    samples: dict[str, dict[str, list[float]]],
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
    means, stds = _compute_mean_std(samples, datasets=datasets, methods=methods)

    n_ds = len(datasets)
    n_m = max(1, len(methods))
    fig_w = max(12, 1.1 * n_ds)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))

    x = np.arange(n_ds, dtype=np.float32)
    width = 0.8 / n_m

    for mi, m in enumerate(methods):
        pos = x + (mi - (n_m - 1) / 2.0) * width
        ax.bar(pos, means[mi], width=width * 0.90, color=color_map[m], alpha=0.72, edgecolor=color_map[m], linewidth=1.0, label=m if mi == 0 else None, zorder=2)
        ax.errorbar(pos, means[mi], yerr=stds[mi], fmt="none", ecolor="#1F2937", elinewidth=1.2, capsize=2.8, capthick=1.0, alpha=0.95, zorder=3)

    for i in range(n_ds - 1):
        ax.axvline(x=float(i) + 0.5, color="#6B7280", linestyle="-", linewidth=1.2, alpha=0.5, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_xlim(-0.6, float(n_ds) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    handles = [plt.Line2D([0], [0], color=color_map[m], lw=6, alpha=0.7, label=m) for m in methods]
    ax.legend(handles=handles, frameon=False, ncol=min(4, len(methods)), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_matrix_csv(path: str, datasets: list[str], methods: list[str], samples: dict[str, dict[str, list[float]]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "n", "mean", "median", "std", "min", "max"])
        for d in datasets:
            for m in methods:
                vals = np.asarray(samples[d][m], dtype=np.float64)
                if vals.size == 0:
                    continue
                w.writerow([d, m, int(vals.size), float(np.mean(vals)), float(np.median(vals)), float(np.std(vals, ddof=0)), float(np.min(vals)), float(np.max(vals))])


def _normalize_per_dataset(
    rows: list[dict[str, object]],
    *,
    datasets: list[str],
    methods: list[str],
    metric: str,
    fill_missing_zero: bool,
) -> dict[str, dict[str, list[float]]]:
    per_ds_vals: dict[str, list[float]] = {d: [] for d in datasets}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        if d not in per_ds_vals:
            continue
        v = _to_float(r.get(metric, np.nan))
        if np.isfinite(v):
            per_ds_vals[d].append(float(v))
    out: dict[str, dict[str, list[float]]] = {d: {m: [] for m in methods} for d in datasets}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        m = str(r.get("method", "")).strip()
        if d not in out or m not in out[d]:
            continue
        v = _to_float(r.get(metric, np.nan))
        if not np.isfinite(v):
            continue
        ds_pool = np.asarray(per_ds_vals[d], dtype=np.float64)
        if ds_pool.size == 0:
            nv = 0.0
        else:
            lo = float(np.min(ds_pool))
            hi = float(np.max(ds_pool))
            nv = 0.5 if hi <= lo + 1e-12 else (float(v) - lo) / (hi - lo)
        out[d][m].append(float(np.clip(nv, 0.0, 1.0)))
    if fill_missing_zero:
        for d in datasets:
            for m in methods:
                if len(out[d][m]) == 0:
                    out[d][m] = [0.0]
    return out


def _save_cross_dataset_csv(path: str, methods: list[str], datasets: list[str], norm_samples: dict[str, dict[str, list[float]]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_over_datasets", "std_over_datasets", "n_datasets"])
        for m in methods:
            ds_means: list[float] = []
            for d in datasets:
                vals = np.asarray(norm_samples[d][m], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                ds_means.append(float(np.mean(vals)) if vals.size > 0 else 0.0)
            arr = np.asarray(ds_means, dtype=np.float64)
            w.writerow([m, float(np.mean(arr)), float(np.std(arr, ddof=0)), int(arr.size)])


PAPER_DATASETS = [
    "2d_planar_arm_line_n2",
    "3d_planar_arm_line_n3_traj",
    "3d_spatial_arm_ellip_n3_traj",
    "3d_twosphere_traj",
    "3d_vz_2d_ellipse_traj",
    "linked_curves_3d",
    "multi_handle_4d",
    "6d_spatial_arm_up_n6_py_traj",
    "6d_surf_normal_traj",
    "12d_dual_arm_traj",
]

PAPER_DATASET_LABELS = {
    "RealKuka": "RealKuka",
    "2d_ellipse": "2DEllipse",
    "2d_planar_arm_line_n2": "2DPlanarArmLine",
    "2d_sparse_sine": "2DSineSparse",
    "3d_planar_arm_line_n3_traj": "3DPlanarArmLine",
    "3d_spatial_arm_ellip_n3_traj": "3DArmEllipse",
    "3d_twosphere_traj": "3DTwoSphere",
    "3d_vz_2d_ellipse_traj": "3DTwistEllipse",
    "3d_torus_surface_traj": "3DTorus",
    "6d_spatial_arm_up_n6_py_traj": "6DArmUp",
    "6d_surf_normal": "6DSurfNormal",
    "6d_surf_normal_traj": "6DSurfNormal",
    "12d_dual_arm_traj": "12DDualArm",
    "linked_curves_3d": "3DLinkedCurves",
    "multi_handle_4d": "4DMultiHandle",
}

METRIC_DISPLAY_LABELS = {
    "proj_manifold_dist": r"Distance Error $\epsilon_l$",
    "gt_to_learned_mean": r"Coverage Error $\epsilon_g$",
    "learned_to_gt_mean": r"Distance Error $\epsilon_l$",
    "bidirectional_chamfer": r"Bi-Chamfer $\epsilon_{\mathrm{bi}}$",
    "train_seconds": "Training Time",
}

METRIC_FILE_ALIASES = {
    "proj_manifold_dist": "distance_error",
    "gt_to_learned_mean": "coverage_error",
    "train_seconds": "training_time",
}

METRIC_INPUT_ALIASES = {
    "gt_to_learned": "gt_to_learned_mean",
    "learned_to_gt": "learned_to_gt_mean",
    "chamfer": "bidirectional_chamfer",
}

TRAJ_COMPARE_PAIRS = [
    # ("3d_planar_arm_line_n3", "3d_planar_arm_line_n3_traj", "3DPlanarArmLine"),
    ("3d_spatial_arm_ellip_n3", "3d_spatial_arm_ellip_n3_traj", "3DArmEllipse"),
    # ("3d_twosphere", "3d_twosphere_traj", "3DTwoSphere"),
    # ("3d_vz_2d_ellipse", "3d_vz_2d_ellipse_traj", "3DTwistedEliip"),
    ("6d_surf_normal", "6d_surf_normal_traj", "6DSurfNormal"),
]

TRAIN_TIME_TABLE_DATASETS = [
    "6d_surf_normal_traj",
    "3d_spatial_arm_ellip_n3_traj",
    "6d_spatial_arm_up_n6_py_traj",
]

CODIM_COMPARE_DATASETS = [
    "3d_spatial_arm_ellip_n3_traj",
    # "3d_vz_2d_ellipse_traj",
    "6d_surf_normal_traj",
    "6d_spatial_arm_up_n6_py_traj",
]

CODIM_HINTS = {
    "RealKuka": 4,
    "2d_ellipse": 1,
    "2d_planar_arm_line_n2": 1,
    "2d_sparse_sine": 1,
    "3d_planar_arm_line_n3": 1,
    "3d_planar_arm_line_n3_traj": 1,
    "3d_spatial_arm_ellip_n3": 2,
    "3d_spatial_arm_ellip_n3_traj": 2,
    "3d_twosphere": 1,
    "3d_twosphere_traj": 1,
    "3d_vz_2d_ellipse": 2,
    "3d_vz_2d_ellipse_traj": 2,
    "3d_torus_surface": 1,
    "3d_torus_surface_traj": 1,
    "6d_surf_normal": 2,
    "6d_surf_normal_traj": 2,
    "6d_spatial_arm_up_n6_py": 2,
    "6d_spatial_arm_up_n6_py_traj": 2,
    "12d_dual_arm_traj": 10,
    "linked_curves_3d": 2,
    "multi_handle_4d": 2,
}

METHOD_DISPLAY_LABELS = {
    "oncl": "ONCL (Ours)",
    "dataaug": "DataAug",
    "igr": "IGR",
    "vae": "VAE",
    "ecomann": "ECoMaNN",
}

METHOD_PRIORITY = ["oncl", "dataaug", "igr", "vae", "ecomann"]


def _to_display_dataset(dataset: str) -> str:
    return PAPER_DATASET_LABELS.get(str(dataset), str(dataset))


def _to_compact_display_dataset(dataset: str) -> str:
    label = _to_display_dataset(dataset)
    compact = {
        "3DArmEllipse": "3DArmEllip",
        "3DTwistEllipse": "3DTwistEllip",
        "3DPlanarArmLine": "3DPlanarArm",
        "2DPlanarArmLine": "2DPlanarArm",
        "6DSurfNormal": "6DSurfNormal",
        "12DDualArm": "12DDualArm",
    }
    return compact.get(label, label)


def _to_compact_display_dataset_multiline(dataset: str) -> str:
    label = _to_display_dataset(dataset)
    compact = {
        "3DArmEllipse": "3DArm\nEllip",
        "3DTwistEllipse": "3DTwist\nEllip",
        "6DArmUp": "6DArm\nUp",
        "6DSurfNormal": "6DSurf\nNormal",
    }
    return compact.get(label, _to_compact_display_dataset(dataset))


def _to_display_metric(metric: str) -> str:
    return METRIC_DISPLAY_LABELS.get(str(metric), str(metric))


def _to_metric_file_stem(metric: str) -> str:
    return METRIC_FILE_ALIASES.get(str(metric), str(metric))


def _normalize_metrics(metrics: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in metrics:
        mm = METRIC_INPUT_ALIASES.get(str(m), str(m))
        if mm in seen:
            continue
        seen.add(mm)
        out.append(mm)
    return out


def _to_display_method(method: str) -> str:
    return METHOD_DISPLAY_LABELS.get(str(method), str(method))


def _sort_methods_for_paper(methods: list[str]) -> list[str]:
    rank = {m: i for i, m in enumerate(METHOD_PRIORITY)}
    return sorted(methods, key=lambda m: (rank.get(m, 10_000), m))


def _remap_samples(
    samples: dict[str, dict[str, list[float]]],
    datasets: list[str],
) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    out: dict[str, dict[str, list[float]]] = {}
    display_datasets: list[str] = []
    for d in datasets:
        dd = _to_display_dataset(d)
        out[dd] = samples[d]
        display_datasets.append(dd)
    return out, display_datasets


def _remap_method_samples(
    samples: dict[str, dict[str, list[float]]],
    methods: list[str],
) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    out: dict[str, dict[str, list[float]]] = {}
    display_methods: list[str] = []
    for m in methods:
        dm = _to_display_method(m)
        display_methods.append(dm)
    for d, m_map in samples.items():
        out[d] = {}
        for m in methods:
            dm = _to_display_method(m)
            out[d][dm] = m_map.get(m, [])
    return out, display_methods


def _vals_from_rows(rows: list[dict[str, object]], dataset: str, method: str, metric: str) -> np.ndarray:
    vals: list[float] = []
    for r in rows:
        if str(r.get("dataset", "")).strip() != str(dataset):
            continue
        if str(r.get("method", "")).strip() != str(method):
            continue
        v = _to_float(r.get(metric, np.nan))
        if np.isfinite(v):
            vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def _plot_traj_overlay_compare(
    *,
    out_path: str,
    title: str,
    ylabel: str,
    rows: list[dict[str, object]],
    methods: list[str],
    metric: str,
) -> None:
    import matplotlib.pyplot as plt

    default_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: default_colors[i % len(default_colors)] for i, m in enumerate(methods)}
    # Keep the overlay figure visually aligned with the codim comparison plot:
    # ECoMaNN should use the same red as the IGR bar in distance_error_codim_bar.
    color_map["ecomann"] = "#E45756"
    n_groups = len(TRAJ_COMPARE_PAIRS)
    n_m = max(1, len(methods))
    # Narrower single-column subfigure size for side-by-side layout.
    fig, ax = plt.subplots(figsize=(2.45, 1.5))
    centers = np.arange(n_groups, dtype=np.float64)
    width = 0.8 / n_m

    for gi, (base_ds, traj_ds, _) in enumerate(TRAJ_COMPARE_PAIRS):
        group_center = centers[gi]
        for mi, m in enumerate(methods):
            x = group_center + (mi - (n_m - 1) / 2.0) * width
            bar_w = width * 0.86
            vals_traj = _vals_from_rows(rows, traj_ds, m, metric)
            vals_base = _vals_from_rows(rows, base_ds, m, metric)
            mean_traj = float(np.mean(vals_traj)) if vals_traj.size > 0 else np.nan
            std_traj = float(np.std(vals_traj, ddof=0)) if vals_traj.size > 0 else 0.0
            mean_base = float(np.mean(vals_base)) if vals_base.size > 0 else np.nan
            std_base = float(np.std(vals_base, ddof=0)) if vals_base.size > 0 else 0.0

            if np.isfinite(mean_traj):
                ax.bar(
                    x,
                    mean_traj,
                    width=bar_w,
                    color=color_map[m],
                    edgecolor=color_map[m],
                    linewidth=1.0,
                    alpha=0.82,
                    zorder=2,
                )
                ax.errorbar(
                    x,
                    mean_traj,
                    yerr=std_traj,
                    fmt="none",
                    ecolor=color_map[m],
                    elinewidth=0.9,
                    capsize=2.0,
                    capthick=0.9,
                    alpha=0.95,
                    zorder=5,
                )
            if np.isfinite(mean_base):
                ax.bar(
                    x,
                    mean_base,
                    width=bar_w,
                    color="none",
                    edgecolor="#6B7280",
                    linewidth=1.6,
                    hatch="///",
                    zorder=4,
                )
                ax.errorbar(
                    x,
                    mean_base,
                    yerr=std_base,
                    fmt="none",
                    ecolor="#6B7280",
                    elinewidth=0.85,
                    capsize=1.4,
                    capthick=0.85,
                    alpha=0.95,
                    zorder=6,
                )

    for i in range(n_groups - 1):
        ax.axvline(float(i) + 0.5, color="#6B7280", linestyle="-", linewidth=1.1, alpha=0.45, zorder=0)

    ax.set_xticks(centers)
    ax.set_xticklabels([_to_compact_display_dataset(traj_ds) for _, traj_ds, _ in TRAJ_COMPARE_PAIRS])
    ax.tick_params(axis="x", labelsize=7.4, pad=3)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.yaxis.label.set_size(10)
    ax.grid(axis="y", alpha=0.22, zorder=0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map[m], edgecolor=color_map[m], alpha=0.85, label=_to_display_method(m))
        for m in methods
    ] + [
        plt.Rectangle((0, 0), 1, 1, facecolor="#9CA3AF", edgecolor="#4B5563", alpha=0.75, label="Trajectory demonstrations"),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#4B5563", hatch="///", linewidth=1.4, label="Scatter data"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.4, -0.18),
        fontsize=6.2,
        columnspacing=0.9,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.21, right=0.99, top=0.93, bottom=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches=None, pad_inches=0.01)
    plt.close(fig)


def _plot_paper_disterror_grouped_bar(
    out_path: str,
    ylabel: str,
    datasets: list[str],
    methods: list[str],
    samples: dict[str, dict[str, list[float]]],
    ymax: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
    means, stds = _compute_mean_std(samples, datasets=datasets, methods=methods)

    n_ds = len(datasets)
    n_m = max(1, len(methods))
    x = np.arange(n_ds, dtype=np.float32)
    width = 0.82 / n_m
    # Paper double-column friendly size, close to traj-vs-scatter styling.
    fig, ax = plt.subplots(figsize=(7.1, 2.75))
    for mi, m in enumerate(methods):
        pos = x + (mi - (n_m - 1) / 2.0) * width
        ax.bar(
            pos,
            means[mi],
            width=width * 0.90,
            color=color_map[m],
            alpha=0.82,
            edgecolor=color_map[m],
            linewidth=1.0,
            zorder=2,
            label=m,
        )
        ax.errorbar(
            pos,
            means[mi],
            yerr=stds[mi],
            fmt="none",
            ecolor="#1F2937",
            elinewidth=1.0,
            capsize=2.0,
            capthick=1.0,
            zorder=3,
        )
        if ymax is not None and np.isfinite(float(ymax)):
            for x_i, mean_i in zip(pos, means[mi]):
                if np.isfinite(mean_i) and float(mean_i) > float(ymax):
                    ax.text(
                        float(x_i),
                        float(ymax) * 0.965,
                        f"{float(mean_i):.2f}",
                        ha="center",
                        va="top",
                        fontsize=6.7,
                        color="#111827",
                        rotation=90,
                        zorder=5,
                    )

    for i in range(n_ds - 1):
        ax.axvline(float(i) + 0.5, color="#6B7280", linestyle="-", linewidth=1.0, alpha=0.35, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=18, ha="right")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    if ymax is not None and np.isfinite(float(ymax)) and float(ymax) > 0.0:
        ax.set_ylim(0.0, float(ymax))
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.legend(frameon=False, ncol=min(4, len(methods)), loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=7)

    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0), pad=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_paper_cross_dataset_average(
    out_path: str,
    methods: list[str],
    datasets: list[str],
    norm_samples: dict[str, dict[str, list[float]]],
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    ds_method_mean = np.zeros((len(methods), len(datasets)), dtype=np.float64)
    for mi, m in enumerate(methods):
        for di, d in enumerate(datasets):
            vals = np.asarray(norm_samples[d][m], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            ds_method_mean[mi, di] = float(np.mean(vals)) if vals.size > 0 else 0.0

    method_mean = np.mean(ds_method_mean, axis=1)
    method_std = np.std(ds_method_mean, axis=1, ddof=0)

    x = np.arange(len(methods), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(2.75, 1.35))
    bar_colors = [color_map[m] for m in methods]
    ax.bar(
        x,
        method_mean,
        width=0.66,
        color=bar_colors,
        alpha=0.82,
        edgecolor=bar_colors,
        linewidth=1.0,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(r"Normalized DistError $\bar{\epsilon}_d$", fontsize=9)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    fig.tight_layout(pad=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_codim_gt1_oncl_compare(
    *,
    out_path: str,
    ylabel: str,
    rows: list[dict[str, object]],
    datasets: list[str],
    metric: str,
) -> None:
    import matplotlib.pyplot as plt

    methods = ["oncl", "igr"]
    seen_methods = {str(r.get("method", "")).strip() for r in rows}
    if any(m not in seen_methods for m in methods):
        return

    allowed = set(CODIM_COMPARE_DATASETS)
    ds_gt1 = [
        d for d in datasets
        if d in allowed and int(CODIM_HINTS.get(str(d), 1)) > 1
    ]
    if not ds_gt1:
        return

    labels = [_to_compact_display_dataset(d) for d in ds_gt1]
    centers = np.arange(len(ds_gt1), dtype=np.float64)
    width = 0.34
    colors = {"oncl": "#4C78A8", "igr": "#E45756"}

    fig, ax = plt.subplots(figsize=(2.45, 1.5))
    for mi, m in enumerate(methods):
        xs = centers + (-0.5 if mi == 0 else 0.5) * width
        vals = []
        errs = []
        for d in ds_gt1:
            arr = _vals_from_rows(rows, d, m, metric)
            vals.append(float(np.mean(arr)) if arr.size > 0 else np.nan)
            errs.append(float(np.std(arr, ddof=0)) if arr.size > 0 else np.nan)
        vals_np = np.asarray(vals, dtype=np.float64)
        errs_np = np.asarray(errs, dtype=np.float64)
        ok = np.isfinite(vals_np)
        if np.any(ok):
            ax.bar(
                xs[ok],
                vals_np[ok],
                width=width * 0.88,
                color=colors[m],
                edgecolor=colors[m],
                linewidth=1.0,
                alpha=0.85,
                label=_to_display_method(m),
                zorder=2,
            )
            ax.errorbar(
                xs[ok],
                vals_np[ok],
                yerr=errs_np[ok],
                fmt="none",
                ecolor="#1F2937",
                elinewidth=0.9,
                capsize=1.8,
                capthick=0.9,
                alpha=0.95,
                zorder=3,
            )

    for i in range(len(ds_gt1) - 1):
        ax.axvline(float(i) + 0.5, color="#6B7280", linestyle="-", linewidth=1.1, alpha=0.45, zorder=0)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.48, float(len(ds_gt1)) - 0.52)
    for label in ax.get_xticklabels():
        label.set_rotation(12)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.tick_params(axis="x", labelsize=6.8, pad=2)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=1)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=6.2, borderaxespad=0.0)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.96, bottom=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches=None, pad_inches=0.01)
    plt.close(fig)


def _write_training_time_table(
    *,
    outdir: str,
    rows: list[dict[str, object]],
    methods: list[str],
) -> None:
    ds_ids = list(TRAIN_TIME_TABLE_DATASETS)
    ds_labels = [_to_display_dataset(d) for d in ds_ids]
    mean_map: dict[tuple[str, str], float] = {}
    std_map: dict[tuple[str, str], float] = {}
    n_map: dict[tuple[str, str], int] = {}
    for d in ds_ids:
        for m in methods:
            arr = _vals_from_rows(rows, d, m, "train_seconds")
            arr = _remove_outliers_iqr(arr)
            n = int(arr.size)
            n_map[(d, m)] = n
            if n <= 0:
                mean_map[(d, m)] = float("nan")
                std_map[(d, m)] = float("nan")
            else:
                mean_map[(d, m)] = float(np.mean(arr))
                std_map[(d, m)] = float(np.std(arr, ddof=0))

    csv_path = os.path.join(outdir, "training_time_table.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["method"] + ds_labels
        w.writerow(header)
        for m in methods:
            row = [_to_display_method(m)]
            for d in ds_ids:
                mu = mean_map[(d, m)]
                sd = std_map[(d, m)]
                n = n_map[(d, m)]
                if n <= 0 or (not np.isfinite(mu)) or (not np.isfinite(sd)):
                    row.append("-")
                else:
                    row.append(f"{mu:.4f} ± {sd:.4f}")
            w.writerow(row)

    md_path = os.path.join(outdir, "training_time_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Method | " + " | ".join(ds_labels) + " |\n")
        f.write("|---" * (len(ds_labels) + 1) + "|\n")
        for m in methods:
            cells = []
            for d in ds_ids:
                mu = mean_map[(d, m)]
                sd = std_map[(d, m)]
                n = n_map[(d, m)]
                if n <= 0 or (not np.isfinite(mu)) or (not np.isfinite(sd)):
                    cells.append("-")
                else:
                    cells.append(f"{mu:.4f} ± {sd:.4f}")
            f.write("| " + _to_display_method(m) + " | " + " | ".join(cells) + " |\n")

    print(f"[saved] {csv_path}")
    print(f"[saved] {md_path}")


def main() -> None:
    def _parse_bool(s: str | bool) -> bool:
        if isinstance(s, bool):
            return s
        v = str(s).strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {s!r}")

    p = argparse.ArgumentParser(description="Paper-style benchmark grouped bar charts with display dataset names.")
    p.add_argument("--bench", type=str, default="paper_main_10env_15seed", help="Benchmark name (under outputs/bench) or absolute path.")
    p.add_argument("--methods", type=str, default="dataaug,oncl,vae,ecomann", help="Comma-separated methods. Default: dataaug,oncl,vae,ecomann")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset ids. Default: paper preset (2D non-traj + 3D/6D traj).")
    p.add_argument("--metrics", type=str, default="proj_manifold_dist,gt_to_learned", help="Comma-separated metrics to plot.")
    p.add_argument("--analysis-dir", type=str, default="analysis_bars_paper", help="Output dir under bench dir.")
    p.add_argument("--fill-missing-zero", action="store_true", default=True)
    p.add_argument("--no-fill-missing-zero", dest="fill_missing_zero", action="store_false")
    p.add_argument("--traj-overlay-compare", type=_parse_bool, default=True, help="Also plot traj vs non-traj overlay bars for selected dataset pairs.")
    p.add_argument("--traj-overlay-methods", type=str, default="oncl,ecomann", help="Comma-separated methods for traj/non-traj overlay plot only. Default: use --methods.")
    p.add_argument("--codim-gt1-compare", type=_parse_bool, default=True, help="Also plot codim>1 dataset comparison between oncl and igr.")
    p.add_argument("--significance", action="store_true", help="Write ONCL-vs-baseline one-sided paired t-test significance tables.")
    p.add_argument("--significance-baselines", type=str, default="dataaug,vae,ecomann", help="Baselines compared against ONCL when --significance is set.")
    p.add_argument("--significance-alpha", type=float, default=0.05, help="Alpha threshold for significance star annotations.")
    p.add_argument("--rank-comparison", action="store_true", help="Write cross-dataset average-rank and ONCL-vs-baseline Holm-corrected rank tests.")
    p.add_argument("--rank-comparison-baselines", type=str, default="dataaug,vae,ecomann", help="Baselines compared against ONCL for --rank-comparison.")
    p.add_argument(
        "--bar-ymax",
        type=float,
        default=DEFAULT_BAR_YMAX,
        help="Y-axis cap for grouped bar plots; off-scale bar means are labeled. Default: 0.5.",
    )
    args = p.parse_args()

    bench_dir = _resolve_bench_dir(args.bench)
    if not os.path.isdir(bench_dir):
        raise FileNotFoundError(f"benchmark directory not found: {bench_dir}")

    rows, source_name = _load_rows(bench_dir)
    if not rows:
        raise RuntimeError(f"no benchmark rows found under: {bench_dir}")

    methods = _split_csv(args.methods, DEFAULT_METHODS)
    metrics = _normalize_metrics(_split_csv(args.metrics, DEFAULT_METRICS))
    all_methods_in_rows = sorted({str(r.get("method", "")) for r in rows if str(r.get("method", ""))})
    if args.methods is None:
        methods = [m for m in DEFAULT_METHODS if m in all_methods_in_rows] + [m for m in all_methods_in_rows if m not in DEFAULT_METHODS]
    else:
        methods = [m for m in methods if m in all_methods_in_rows]
    methods = _sort_methods_for_paper(methods)
    overlay_methods = _split_csv(args.traj_overlay_methods, methods)
    overlay_methods = [m for m in overlay_methods if m in all_methods_in_rows]
    overlay_methods = _sort_methods_for_paper(overlay_methods)
    significance_baselines = _split_csv(args.significance_baselines, [])
    significance_baselines = [m for m in significance_baselines if m in all_methods_in_rows and m != "oncl"]
    significance_baselines = _sort_methods_for_paper(significance_baselines)
    rank_comparison_baselines = _split_csv(args.rank_comparison_baselines, [])
    rank_comparison_baselines = [m for m in rank_comparison_baselines if m in all_methods_in_rows and m != "oncl"]
    rank_comparison_baselines = _sort_methods_for_paper(rank_comparison_baselines)

    datasets = _split_csv(args.datasets, None)
    if not datasets:
        datasets = list(PAPER_DATASETS)

    outdir = os.path.join(bench_dir, args.analysis_dir)
    os.makedirs(outdir, exist_ok=True)

    for metric in metrics:
        metric_label = _to_display_metric(metric)
        metric_datasets = list(TRAIN_TIME_TABLE_DATASETS) if str(metric) == "train_seconds" else list(datasets)
        metric_rows = rows
        if str(metric) == "train_seconds":
            metric_rows = _rows_with_train_seconds_outliers_removed(rows, dataset_allow=set(metric_datasets))
        samples_raw = _build_samples(
            metric_rows,
            metric_datasets,
            methods,
            metric=metric,
            fill_missing_zero=bool(args.fill_missing_zero),
        )
        samples_ds, display_datasets = _remap_samples(samples_raw, metric_datasets)
        samples, display_methods = _remap_method_samples(samples_ds, methods)
        metric_stem = _to_metric_file_stem(metric)
        png_name = f"{metric_stem}_bar.png"
        csv_name = f"{metric_stem}_bar_stats.csv"
        out_png = os.path.join(outdir, png_name)
        out_csv = os.path.join(outdir, csv_name)
        if bool(args.significance) and "oncl" in methods and str(metric) == "proj_manifold_dist":
            sig_raw = _compute_oncl_paired_t_significance(
                rows=metric_rows,
                datasets=metric_datasets,
                baselines=significance_baselines,
                metric=metric,
                alpha=float(args.significance_alpha),
            )
            sig_csv = os.path.join(outdir, f"{metric_stem}_oncl_paired_t_significance.csv")
            sig_md = os.path.join(outdir, f"{metric_stem}_oncl_paired_t_significance.md")
            _save_significance_csv(sig_csv, sig_raw)
            _save_significance_md(sig_md, sig_raw)
            if sig_raw:
                print(f"[saved] {sig_csv}")
                print(f"[saved] {sig_md}")
        if bool(args.rank_comparison) and "oncl" in methods and str(metric) == "proj_manifold_dist":
            rank_result = _compute_cross_dataset_rank_comparison(
                rows=metric_rows,
                datasets=metric_datasets,
                methods=methods,
                baselines=rank_comparison_baselines,
                metric=metric,
                alpha=float(args.significance_alpha),
            )
            if rank_result:
                rank_csv, rank_cmp_csv, rank_md = _save_rank_comparison_outputs(
                    outdir=outdir,
                    metric_stem=metric_stem,
                    result=rank_result,
                )
                print(f"[saved] {rank_csv}")
                print(f"[saved] {rank_cmp_csv}")
                print(f"[saved] {rank_md}")
        if str(metric) in {
            "proj_manifold_dist",
            "gt_to_learned_mean",
            "learned_to_gt_mean",
            "bidirectional_chamfer",
            "train_seconds",
        }:
            _plot_paper_disterror_grouped_bar(
                out_path=out_png,
                ylabel=metric_label,
                datasets=display_datasets,
                methods=display_methods,
                samples=samples,
                ymax=args.bar_ymax,
            )
        else:
            _plot_grouped_bar_with_error(
                out_png,
                title=f"Paper Benchmark ({metric_label})",
                ylabel=metric_label,
                datasets=display_datasets,
                methods=display_methods,
                samples=samples,
            )
        _save_matrix_csv(out_csv, datasets=display_datasets, methods=display_methods, samples=samples)
        print(f"[saved] {out_png}")
        print(f"[saved] {out_csv}")

        if str(metric) == "train_seconds":
            continue

        norm_samples_raw = _normalize_per_dataset(
            metric_rows,
            datasets=metric_datasets,
            methods=methods,
            metric=metric,
            fill_missing_zero=bool(args.fill_missing_zero),
        )
        norm_samples_ds, norm_display_datasets = _remap_samples(norm_samples_raw, metric_datasets)
        norm_samples, norm_display_methods = _remap_method_samples(norm_samples_ds, methods)
        out_png_cross = os.path.join(outdir, f"{metric_stem}_normalized_bar.png")
        out_csv_cross = os.path.join(outdir, f"{metric_stem}_normalized_bar_stats.csv")
        _plot_paper_cross_dataset_average(
            out_png_cross,
            methods=norm_display_methods,
            datasets=norm_display_datasets,
            norm_samples=norm_samples,
        )
        _save_cross_dataset_csv(out_csv_cross, methods=norm_display_methods, datasets=norm_display_datasets, norm_samples=norm_samples)
        print(f"[saved] {out_png_cross}")
        print(f"[saved] {out_csv_cross}")

        if bool(args.traj_overlay_compare):
            out_overlay = os.path.join(outdir, f"{metric_stem}_traj_vs_scatter.png")
            _plot_traj_overlay_compare(
                out_path=out_overlay,
                title=f"Traj vs Non-traj Overlay ({metric_label})",
                ylabel=metric_label,
                rows=rows,
                methods=overlay_methods,
                metric=metric,
            )
            print(f"[saved] {out_overlay}")
        if bool(args.codim_gt1_compare):
            out_codim = os.path.join(outdir, f"{metric_stem}_codim_bar.png")
            _plot_codim_gt1_oncl_compare(
                out_path=out_codim,
                ylabel=metric_label,
                rows=rows,
                datasets=datasets,
                metric=metric,
            )
            if os.path.isfile(out_codim):
                print(f"[saved] {out_codim}")

    print(f"[info] bench={bench_dir}")
    print(f"[info] source={source_name}")
    print(f"[info] methods={','.join([_to_display_method(m) for m in methods])}")
    print(f"[info] datasets={','.join([_to_display_dataset(d) for d in datasets])}")
    _write_training_time_table(outdir=outdir, rows=rows, methods=methods)


def _remove_outliers_iqr(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return x
    q1 = float(np.percentile(x, 25.0))
    q3 = float(np.percentile(x, 75.0))
    iqr = q3 - q1
    if iqr <= 1e-12:
        return x
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    y = x[(x >= lo) & (x <= hi)]
    if y.size == 0:
        return x
    return y


def _rows_with_train_seconds_outliers_removed(
    rows: list[dict[str, object]],
    dataset_allow: set[str] | None = None,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for r in rows:
        d = str(r.get("dataset", "")).strip()
        m = str(r.get("method", "")).strip()
        grouped.setdefault((d, m), []).append(r)

    out: list[dict[str, object]] = []
    for (d, m), items in grouped.items():
        if dataset_allow is not None and d not in dataset_allow:
            out.extend(items)
            continue
        vals = []
        keep_idx = []
        for i, r in enumerate(items):
            v = _to_float(r.get("train_seconds", np.nan))
            if np.isfinite(v):
                vals.append(float(v))
                keep_idx.append(i)
        if len(vals) < 4:
            out.extend(items)
            continue
        arr = np.asarray(vals, dtype=np.float64)
        q1 = float(np.percentile(arr, 25.0))
        q3 = float(np.percentile(arr, 75.0))
        iqr = q3 - q1
        if iqr <= 1e-12:
            out.extend(items)
            continue
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        removed = 0
        for i, r in enumerate(items):
            v = _to_float(r.get("train_seconds", np.nan))
            if np.isfinite(v) and (v < lo or v > hi):
                removed += 1
                continue
            out.append(r)
        if removed > 0:
            print(f"[info][training_time] removed outliers for {d}/{m}: {removed}")
    return out


if __name__ == "__main__":
    main()
