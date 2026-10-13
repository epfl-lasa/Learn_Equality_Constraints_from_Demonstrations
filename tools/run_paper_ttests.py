from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import numpy as np

try:
    from scipy import stats
except Exception:  # scipy is optional; paired permutation tests need only numpy.
    stats = None

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmark import plot_paper_benchmarks as paper


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


def _independent_arrays(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    method_a: str,
    method_b: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    vals_a: list[float] = []
    vals_b: list[float] = []
    for r in rows:
        if str(r.get("dataset", "")).strip() != str(dataset):
            continue
        method = str(r.get("method", "")).strip()
        v = paper._to_float(r.get(metric, np.nan))
        if not np.isfinite(v):
            continue
        if method == str(method_a):
            vals_a.append(float(v))
        elif method == str(method_b):
            vals_b.append(float(v))
    return np.asarray(vals_a, dtype=np.float64), np.asarray(vals_b, dtype=np.float64)


def _paired_arrays(
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
        v = paper._to_float(r.get(metric, np.nan))
        if not seed or not np.isfinite(v):
            continue
        if method == str(method_a):
            vals_a[seed] = float(v)
        elif method == str(method_b):
            vals_b[seed] = float(v)
    common = sorted(set(vals_a) & set(vals_b), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))
    return (
        common,
        np.asarray([vals_a[s] for s in common], dtype=np.float64),
        np.asarray([vals_b[s] for s in common], dtype=np.float64),
    )


def _holm_bonferroni(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(np.asarray(pvals, dtype=np.float64))
    ranked = [float(pvals[i]) for i in order]
    adj_ranked = [0.0] * n
    running = 0.0
    for k, p in enumerate(ranked):
        adj = (n - k) * p
        running = max(running, adj)
        adj_ranked[k] = min(1.0, running)
    out = [0.0] * n
    for idx, adj in zip(order, adj_ranked):
        out[int(idx)] = float(adj)
    return out


def _cohens_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return np.nan
    nx = int(x.size)
    ny = int(y.size)
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 1e-12:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def _cohens_dz(diff_good_direction: np.ndarray) -> float:
    diff_good_direction = np.asarray(diff_good_direction, dtype=np.float64)
    if diff_good_direction.size < 2:
        return np.nan
    sd = float(np.std(diff_good_direction, ddof=1))
    if sd <= 1e-12:
        return np.nan
    return float(np.mean(diff_good_direction) / sd)


def _paired_good_direction_diff(metric: str, oncl_vals: np.ndarray, base_vals: np.ndarray) -> np.ndarray:
    if metric in LOWER_IS_BETTER_METRICS:
        return np.asarray(base_vals, dtype=np.float64) - np.asarray(oncl_vals, dtype=np.float64)
    if metric in HIGHER_IS_BETTER_METRICS:
        return np.asarray(oncl_vals, dtype=np.float64) - np.asarray(base_vals, dtype=np.float64)
    raise ValueError(f"unknown metric direction for paired test: {metric}")


def _paired_exact_permutation_pvalue(diff_good_direction: np.ndarray) -> tuple[float, float]:
    """One-sided paired sign-flip permutation test.

    The null is exchangeability of the paired method labels. Positive mean
    difference means ONCL is better according to the metric direction.
    """
    diff = np.asarray(diff_good_direction, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan, np.nan
    obs = float(np.mean(diff))
    n = int(diff.size)
    if n <= 20:
        extreme = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            total += 1
            stat = float(np.mean(diff * np.asarray(signs, dtype=np.float64)))
            if stat >= obs - 1e-15:
                extreme += 1
        return obs, float(extreme / total)
    rng = np.random.default_rng(0)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(200_000, n), replace=True)
    stats_mc = np.mean(signs * diff[None, :], axis=1)
    return obs, float(np.mean(stats_mc >= obs - 1e-15))


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


def _paired_t_pvalue(diff_good_direction: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(diff_good_direction, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return np.nan, np.nan
    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        stat = math.inf if float(np.mean(diff)) > 0 else -math.inf
        return stat, 0.0 if stat > 0 else 1.0
    stat = float(np.mean(diff) / (sd / math.sqrt(diff.size)))
    p_value = 1.0 - _student_t_cdf(stat, diff.size - 1)
    return stat, float(min(max(p_value, 0.0), 1.0))


def _metric_alternative(metric: str) -> str:
    m = str(metric)
    if m in LOWER_IS_BETTER_METRICS:
        return "less"
    if m in HIGHER_IS_BETTER_METRICS:
        return "greater"
    raise ValueError(f"unknown metric direction for t-test: {metric}")


def _mean_std_str(vals: np.ndarray) -> str:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return "-"
    return f"{np.mean(vals):.4f} ± {np.std(vals, ddof=0):.4f}"


def _save_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_md(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    cols = [
        "Dataset",
        "Baseline",
        "n_oncl",
        "n_base",
        "ONCL",
        "Baseline mean±std",
        "alternative",
        "stat",
        "p",
        "p_holm",
        "significant_0.05",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(r["Dataset"]),
                        str(r["Baseline"]),
                        str(r["n_oncl"]),
                        str(r["n_base"]),
                        str(r["ONCL"]),
                        str(r["Baseline mean±std"]),
                        str(r["alternative"]),
                        str(r["stat"]),
                        str(r["p"]),
                        str(r["p_holm"]),
                        str(r["significant_0.05"]),
                    ]
                )
                + " |\n"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Seed-level statistical tests for ONCL vs baselines on paper benchmark results.")
    p.add_argument("--bench", type=str, default="paper_mix_2d_3d6d_traj_vs_nontraj_7seed", help="Benchmark name or absolute path.")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset ids. Default: paper preset.")
    p.add_argument("--methods", type=str, default="dataaug,vae,ecomann", help="Baselines to compare against ONCL.")
    p.add_argument("--metrics", type=str, default="proj_manifold_dist,gt_to_learned", help="Comma-separated metrics.")
    p.add_argument("--analysis-dir", type=str, default="analysis_bars_paper", help="Output dir under bench dir.")
    p.add_argument(
        "--test",
        type=str,
        default="paired_permutation",
        choices=["paired_permutation", "paired_t", "welch"],
        help="Statistical test. Paired tests match ONCL and baseline by seed; paired_t needs no scipy.",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    args = p.parse_args()

    bench_dir = paper._resolve_bench_dir(args.bench)
    rows, _source = paper._load_rows(bench_dir)
    if not rows:
        raise RuntimeError(f"no benchmark rows found under: {bench_dir}")

    datasets = paper._split_csv(args.datasets, None)
    if not datasets:
        datasets = list(paper.PAPER_DATASETS)
    metrics = paper._normalize_metrics(paper._split_csv(args.metrics, paper.DEFAULT_METRICS))
    baselines = paper._sort_methods_for_paper(paper._split_csv(args.methods, []))

    all_methods = sorted({str(r.get("method", "")).strip() for r in rows if str(r.get("method", "")).strip()})
    baselines = [m for m in baselines if m in all_methods and m != "oncl"]
    if "oncl" not in all_methods:
        raise RuntimeError("ONCL results not found in benchmark rows")

    outdir = os.path.join(bench_dir, str(args.analysis_dir))
    os.makedirs(outdir, exist_ok=True)

    for metric in metrics:
        alternative = _metric_alternative(metric)
        raw_rows: list[dict[str, object]] = []
        holm_groups: list[float] = []

        for dataset in datasets:
            for baseline in baselines:
                if args.test == "welch":
                    if stats is None:
                        raise RuntimeError("scipy is required for --test welch; use --test paired_permutation instead")
                    oncl_vals, base_vals = _independent_arrays(
                        rows,
                        dataset=dataset,
                        method_a="oncl",
                        method_b=baseline,
                        metric=metric,
                    )
                    if oncl_vals.size < 2 or base_vals.size < 2:
                        continue
                    test = stats.ttest_ind(oncl_vals, base_vals, equal_var=False, alternative=alternative)
                    if metric in LOWER_IS_BETTER_METRICS:
                        # Positive diff means ONCL is better by this amount.
                        mean_diff = float(np.mean(base_vals) - np.mean(oncl_vals))
                        effect = _cohens_d_independent(base_vals, oncl_vals)
                    else:
                        mean_diff = float(np.mean(oncl_vals) - np.mean(base_vals))
                        effect = _cohens_d_independent(oncl_vals, base_vals)
                    stat_name = "t_stat"
                    stat_value = float(test.statistic)
                    p_value = float(test.pvalue)
                    test_type = "welch_independent"
                    n_oncl = int(oncl_vals.size)
                    n_base = int(base_vals.size)
                else:
                    seeds, oncl_vals, base_vals = _paired_arrays(
                        rows,
                        dataset=dataset,
                        method_a="oncl",
                        method_b=baseline,
                        metric=metric,
                    )
                    if oncl_vals.size < 2 or base_vals.size < 2:
                        continue
                    diff = _paired_good_direction_diff(metric, oncl_vals, base_vals)
                    if args.test == "paired_t":
                        stat_value, p_value = _paired_t_pvalue(diff)
                    else:
                        stat_value, p_value = _paired_exact_permutation_pvalue(diff)
                    mean_diff = float(np.mean(diff))
                    effect = _cohens_dz(diff)
                    stat_name = "t_stat" if args.test == "paired_t" else "mean_diff_stat"
                    test_type = "paired_t" if args.test == "paired_t" else "paired_exact_permutation"
                    n_oncl = int(oncl_vals.size)
                    n_base = int(base_vals.size)
                row = {
                    "dataset_id": dataset,
                    "Dataset": paper._to_display_dataset(dataset),
                    "baseline_id": baseline,
                    "Baseline": paper._to_display_method(baseline),
                    "metric": metric,
                    "metric_label": paper._to_display_metric(metric),
                    "n_oncl": n_oncl,
                    "n_base": n_base,
                    "oncl_mean": float(np.mean(oncl_vals)),
                    "oncl_std": float(np.std(oncl_vals, ddof=0)),
                    "baseline_mean": float(np.mean(base_vals)),
                    "baseline_std": float(np.std(base_vals, ddof=0)),
                    "mean_diff_good_direction": mean_diff,
                    "effect_size": effect,
                    "test_type": test_type,
                    "alternative": "ONCL < baseline" if alternative == "less" else "ONCL > baseline",
                    stat_name: stat_value,
                    "p_value": p_value,
                    "ONCL": _mean_std_str(oncl_vals),
                    "Baseline mean±std": _mean_std_str(base_vals),
                }
                raw_rows.append(row)
                holm_groups.append(float(p_value))

        adj = _holm_bonferroni(holm_groups)
        for row, p_adj in zip(raw_rows, adj):
            row["p_holm"] = float(p_adj)
            row["significant_0.05"] = bool(float(p_adj) < float(args.alpha))
            row["stat"] = f"{row.get('t_stat', row.get('mean_diff_stat', np.nan)):.4g}"
            row["p"] = f"{row['p_value']:.4g}"
            row["p_holm_fmt"] = f"{row['p_holm']:.4g}"

        csv_rows: list[dict[str, object]] = []
        md_rows: list[dict[str, object]] = []
        for row in raw_rows:
            csv_rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "Dataset": row["Dataset"],
                    "baseline_id": row["baseline_id"],
                    "Baseline": row["Baseline"],
                    "metric": row["metric"],
                    "metric_label": row["metric_label"],
                    "n_oncl": row["n_oncl"],
                    "n_base": row["n_base"],
                    "oncl_mean": row["oncl_mean"],
                    "oncl_std": row["oncl_std"],
                    "baseline_mean": row["baseline_mean"],
                    "baseline_std": row["baseline_std"],
                    "mean_diff_good_direction": row["mean_diff_good_direction"],
                    "effect_size": row["effect_size"],
                    "test_type": row["test_type"],
                    "alternative": row["alternative"],
                    "stat": row["stat"],
                    "p_value": row["p_value"],
                    "p_holm": row["p_holm"],
                    "significant_0.05": row["significant_0.05"],
                }
            )
            md_rows.append(
                {
                    "Dataset": row["Dataset"],
                    "Baseline": row["Baseline"],
                    "n_oncl": row["n_oncl"],
                    "n_base": row["n_base"],
                    "ONCL": row["ONCL"],
                    "Baseline mean±std": row["Baseline mean±std"],
                    "alternative": row["alternative"],
                    "stat": row["stat"],
                    "p": row["p"],
                    "p_holm": row["p_holm_fmt"],
                    "significant_0.05": "yes" if row["significant_0.05"] else "no",
                }
            )

        stem = paper._to_metric_file_stem(metric)
        suffix = {
            "welch": "welch_ttests",
            "paired_t": "paired_ttests",
            "paired_permutation": "paired_permutation_tests",
        }[args.test]
        csv_path = os.path.join(outdir, f"{stem}_{suffix}.csv")
        md_path = os.path.join(outdir, f"{stem}_{suffix}.md")
        _save_csv(csv_path, csv_rows)
        _save_md(md_path, md_rows)
        print(f"[saved] {csv_path}")
        print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
