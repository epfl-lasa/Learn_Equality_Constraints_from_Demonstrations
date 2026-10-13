#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Keep local W&B files under repo-root/wandb regardless of launch cwd.
os.environ["WANDB_DIR"] = os.path.join(_PROJECT_ROOT, "wandb")

from experiments.unified_experiment import VALID_METHODS, run_one

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

def _resolve_config_root(config_root: str) -> str:
    p = str(config_root).strip()
    if not p:
        p = "configs"
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    fallback = os.path.join(_PROJECT_ROOT, p)
    return fallback


def _resolve_outdir(outdir: str) -> str:
    p = str(outdir).strip()
    if not p:
        p = "outputs"
    if os.path.isabs(p):
        return p
    return os.path.join(_PROJECT_ROOT, p)


def _clear_directory(path: str) -> None:
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def _prepare_outdir(path: str, *, clearn_dir: bool, resume: bool, rewrite: bool) -> None:
    if os.path.exists(path) and not os.path.isdir(path):
        raise ValueError(f"outdir exists but is not a directory: {path}")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return
    has_existing = any(True for _ in os.scandir(path))
    if not has_existing:
        return
    if clearn_dir:
        print(f"[clearn_dir] outdir is non-empty: {path}")
        ans = input("[confirm] type 1 to clear this directory and continue: ").strip()
        if ans != "1":
            raise RuntimeError("clearn_dir cancelled by user (did not input 1).")
        print(f"[clearn_dir] clearing existing outdir: {path}")
        _clear_directory(path)
        return
    if resume or rewrite:
        mode = "rewrite" if rewrite else "resume"
        print(f"[{mode}] using existing outdir for incremental runs: {path}")
        return
    raise ValueError(
        "outdir already exists and is not empty; refusing to mix old/new results. "
        "Use --resume for incremental runs, --rewrite to replace selected runs, "
        "--clearn_dir (or -clearn_dir) to clear it first, "
        "or choose a new --outdir."
    )


def _infer_dataset_from_plot_basename(basename_no_ext: str, datasets: list[str]) -> str | None:
    # Disambiguate prefixes like "..._py" vs "..._py_traj" by longest-prefix match.
    matches = [ds for ds in datasets if basename_no_ext.startswith(f"{ds}_")]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return str(matches[0])


def _log_dataset_plots_to_wandb(
    *,
    outdir: str,
    method: str,
    dataset: str,
    all_datasets: list[str],
    step: int | None = None,
) -> None:
    method_dir = os.path.join(outdir, method)
    if not os.path.isdir(method_dir):
        return
    patterns = [
        os.path.join(method_dir, f"{dataset}*.png"),
        os.path.join(method_dir, f"{dataset}*.jpg"),
        os.path.join(method_dir, f"{dataset}*.jpeg"),
    ]
    plot_paths: list[str] = []
    for pat in patterns:
        plot_paths.extend(glob.glob(pat))
    if not plot_paths:
        return
    for p in sorted(set(plot_paths)):
        key = os.path.splitext(os.path.basename(p))[0]
        owner_ds = _infer_dataset_from_plot_basename(key, all_datasets)
        if owner_ds != str(dataset):
            continue
        try:
            payload = {f"{dataset}/{method}/plot/{key}": wandb.Image(p)}
            if step is None:
                wandb.log(payload)
            else:
                wandb.log(payload, step=int(step))
        except Exception as e:
            print(f"[warn] wandb image log failed: {p} ({e})")


def _sanitize_wandb_value(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _sanitize_wandb_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_wandb_value(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _flatten_dict_for_wandb(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}__{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict_for_wandb(v, key))
        else:
            out[key] = _sanitize_wandb_value(v)
    return out


def _sync_resolved_config_to_wandb(
    *,
    wb_run: Any,
    method: str,
    dataset: str,
    seed: int | None,
    resolved_config: dict[str, Any],
    loaded_paths: list[str],
) -> None:
    prefix = f"resolved__{method}__{dataset}__seed{seed if seed is not None else 'none'}"
    payload = _flatten_dict_for_wandb(dict(resolved_config), f"{prefix}__cfg")
    payload[f"{prefix}__seed"] = seed
    payload[f"{prefix}__loaded_config_paths"] = [str(p) for p in loaded_paths]
    payload[f"{prefix}__loaded_layers_count"] = int(len(loaded_paths))
    try:
        wb_run.config.update(payload, allow_val_change=True)
    except Exception as e:
        print(f"[warn] wandb config sync failed for {method}/{dataset}/seed={seed}: {e}")


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _override_keys(overrides: list[str]) -> set[str]:
    keys: set[str] = set()
    for expr in overrides:
        if "=" not in str(expr):
            continue
        k = str(expr).split("=", 1)[0].strip()
        if k:
            keys.add(k)
    return keys


def _with_default_non_gif_overrides(overrides: list[str]) -> list[str]:
    out = list(overrides)
    keys = _override_keys(out)
    if "plan_save_gif" not in keys:
        out.append("plan_save_gif=false")
    if "plan_pybullet_render" not in keys:
        out.append("plan_pybullet_render=false")
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run benchmark suite over methods and datasets")
    p.add_argument("--methods", default="oncl", help="comma-separated methods")
    p.add_argument("--datasets", default="6d_surf_normal", help="comma-separated datasets")
    p.add_argument("--seeds", default="10", help="comma-separated seeds; empty means no seed override")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("-clearn_dir", "--clearn_dir", action="store_true", help="if outdir is non-empty, ask once for confirmation (type 1) then clear it")
    p.add_argument("-resume", "--resume", action="store_true", help="incrementally append to an existing outdir and skip completed runs")
    p.add_argument("--rewrite", action="store_true", help="reuse an existing outdir but rerun and replace matching method/dataset/seed runs")
    p.add_argument("--config-root", default="configs")
    p.add_argument("--override", action="append", default=[], help="dotted key=value override")

    p.add_argument("--wandb-enable", action="store_true")
    p.add_argument("--wandb-project", default="LearnEqConstraints")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-run-name", default="")
    return p


def _metric_keys(results: list[dict[str, Any]]) -> list[str]:
    keys = set()
    for r in results:
        keys.update(r.get("metrics", {}).keys())
    return sorted(keys)


def _write_per_case_csv(path: str, results: list[dict[str, Any]]) -> None:
    metric_cols = _metric_keys(results)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "dataset", "seed", *metric_cols])
        for r in results:
            row = [r.get("method"), r.get("dataset"), r.get("seed")]
            m = r.get("metrics", {})
            row.extend([m.get(k, "") for k in metric_cols])
            w.writerow(row)


def _write_leaderboard_csv(path: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        grouped[str(r["method"])].append(r)

    metric_cols = _metric_keys(results)
    rows: list[dict[str, Any]] = []
    for method, items in grouped.items():
        agg: dict[str, Any] = {"method": method, "n_cases": len(items)}
        for k in metric_cols:
            vals = [it.get("metrics", {}).get(k) for it in items]
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            agg[k] = (sum(vals) / len(vals)) if vals else ""
        rows.append(agg)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_cases", *metric_cols])
        for row in sorted(rows, key=lambda x: str(x["method"])):
            w.writerow([row["method"], row["n_cases"], *[row.get(k, "") for k in metric_cols]])
    return rows


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _seed_key(v: Any) -> str:
    return str(v).strip()


def _run_key(method: Any, dataset: Any, seed: Any) -> tuple[str, str, str]:
    return (str(method).strip(), str(dataset).strip(), _seed_key(seed))


def _load_existing_results(outdir: str) -> list[dict[str, Any]]:
    per_run = os.path.join(outdir, "per_run_metrics.jsonl")
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    if os.path.isfile(per_run):
        with open(per_run, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                k = _run_key(row.get("method", ""), row.get("dataset", ""), row.get("seed", ""))
                by_key[k] = row
        return list(by_key.values())

    summary = os.path.join(outdir, "summary_metrics.json")
    if not os.path.isfile(summary):
        return []
    try:
        with open(summary, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    for row in data:
        if not isinstance(row, dict):
            continue
        k = _run_key(row.get("method", ""), row.get("dataset", ""), row.get("seed", ""))
        by_key[k] = {
            "timestamp": row.get("timestamp", ""),
            "method": row.get("method", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "metrics": row.get("metrics", {}),
            "config": row.get("config", {}),
            "loaded_config_paths": row.get("loaded_config_paths", []),
        }
    return list(by_key.values())


def _print_progress_snapshot(
    *,
    methods: list[str],
    datasets: list[str],
    seed_values: list[int | None],
    done_per_method: dict[str, int],
    completed_keys: set[tuple[str, str, str]],
    expected_keys: set[tuple[str, str, str]],
) -> None:
    # Method-level progress across all datasets/seeds.
    method_parts: list[str] = []
    total_per_method = max(1, len(datasets) * len(seed_values))
    for m in methods:
        method_parts.append(f"{m} {int(done_per_method.get(m, 0))}/{total_per_method}")
    print("[progress][methods] " + " | ".join(method_parts))

    # Dataset-level progress:
    # - runs: completed (method,dataset,seed) over total methods*seeds
    # - seeds: unique completed seeds over total seeds
    n_seed_total = max(1, len(seed_values))
    n_run_total_per_ds = max(1, len(methods) * len(seed_values))
    for ds in datasets:
        runs_done = 0
        seeds_done: set[str] = set()
        for m in methods:
            for s in seed_values:
                k = _run_key(m, ds, s)
                if k not in expected_keys:
                    continue
                if k in completed_keys:
                    runs_done += 1
                    seeds_done.add(_seed_key(s))
        print(
            f"[progress][dataset] {ds}: seeds {len(seeds_done)}/{n_seed_total} | runs {runs_done}/{n_run_total_per_ds}"
        )


def main() -> None:
    args = _build_parser().parse_args()
    config_root = _resolve_config_root(str(args.config_root))
    outdir = _resolve_outdir(str(args.outdir))
    methods = _split_csv(args.methods)
    datasets = _split_csv(args.datasets)
    seeds = _split_csv(args.seeds)

    if not methods:
        raise ValueError("no methods provided")
    bad = [m for m in methods if m not in VALID_METHODS]
    if bad:
        raise ValueError(f"unsupported methods: {bad}; valid={sorted(VALID_METHODS)}")
    if not datasets:
        raise ValueError("no datasets provided")

    seed_values: list[int | None]
    if seeds:
        seed_values = [int(s) for s in seeds]
    else:
        seed_values = [None]
    expected_keys: set[tuple[str, str, str]] = {
        _run_key(m, ds, s) for m in methods for ds in datasets for s in seed_values
    }

    if bool(args.clearn_dir) and (bool(args.resume) or bool(args.rewrite)):
        raise ValueError("--clearn_dir cannot be used together with --resume or --rewrite")
    _prepare_outdir(
        outdir,
        clearn_dir=bool(args.clearn_dir),
        resume=bool(args.resume),
        rewrite=bool(args.rewrite),
    )
    effective_overrides = _with_default_non_gif_overrides(args.override)
    partial_jsonl = os.path.join(outdir, "per_run_metrics.jsonl")
    partial_summary = os.path.join(outdir, "summary_metrics.partial.json")
    existing_rows: list[dict[str, Any]] = _load_existing_results(outdir) if (bool(args.resume) or bool(args.rewrite)) else []
    completed_keys: set[tuple[str, str, str]] = {
        _run_key(r.get("method", ""), r.get("dataset", ""), r.get("seed", "")) for r in existing_rows
    }
    if existing_rows:
        mode = "rewrite" if bool(args.rewrite) else "resume"
        print(f"[{mode}] loaded existing runs: {len(existing_rows)}")

    wb_run = None
    if args.wandb_enable:
        if wandb is None:
            print("[warn] wandb requested but not available; continuing without wandb")
        else:
            wb_run = wandb.init(
                project=str(args.wandb_project),
                entity=(str(args.wandb_entity) or None),
                name=(str(args.wandb_run_name) or None),
                config={
                    "methods": methods,
                    "datasets": datasets,
                    "seeds": seed_values,
                    "config_root": config_root,
                    "override": args.override,
                    "resume": bool(args.resume),
                    "rewrite": bool(args.rewrite),
                },
            )

    all_results: list[dict[str, Any]] = []
    for r in existing_rows:
        all_results.append(
            {
                "timestamp": r.get("timestamp", ""),
                "method": r.get("method", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "metrics": r.get("metrics", {}) if isinstance(r.get("metrics", {}), dict) else {},
                "config": r.get("config", {}) if isinstance(r.get("config", {}), dict) else {},
                "loaded_config_paths": r.get("loaded_config_paths", []),
            }
        )
    step = 0
    total_per_method = {m: len(datasets) * len(seed_values) for m in methods}
    done_per_method = {m: 0 for m in methods}
    for method in methods:
        for dataset in datasets:
            for seed in seed_values:
                key = _run_key(method, dataset, seed)
                if key in completed_keys and not bool(args.rewrite):
                    print(f"[skip] method={method} dataset={dataset} seed={seed} already exists (resume)")
                    done_per_method[method] += 1
                    _print_progress_snapshot(
                        methods=methods,
                        datasets=datasets,
                        seed_values=seed_values,
                        done_per_method=done_per_method,
                        completed_keys=completed_keys,
                        expected_keys=expected_keys,
                    )
                    continue
                if key in completed_keys and bool(args.rewrite):
                    print(f"[rewrite] method={method} dataset={dataset} seed={seed} already exists; rerunning and replacing")
                    all_results = [
                        r for r in all_results
                        if _run_key(r.get("method", ""), r.get("dataset", ""), r.get("seed", "")) != key
                    ]
                    completed_keys.discard(key)
                print(f"[run] method={method} dataset={dataset} seed={seed}")
                result, loaded = run_one(
                    method=method,
                    dataset=dataset,
                    out_root=outdir,
                    seed_override=seed,
                    config_root=config_root,
                    cli_overrides=effective_overrides,
                )
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                result["seed"] = seed
                result["timestamp"] = timestamp
                result["loaded_config_paths"] = loaded
                all_results.append(result)
                completed_keys.add(key)

                # Persist each finished run immediately so interrupted benchmarks
                # can still be recovered/aggregated.
                row = {
                    "timestamp": timestamp,
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "metrics": result.get("metrics", {}),
                    "config": result.get("config", {}),
                    "loaded_config_paths": loaded,
                }
                if bool(args.rewrite):
                    _write_jsonl(
                        partial_jsonl,
                        [
                            {
                                "timestamp": r.get("timestamp", ""),
                                "method": r.get("method", ""),
                                "dataset": r.get("dataset", ""),
                                "seed": r.get("seed", ""),
                                "metrics": r.get("metrics", {}),
                                "config": r.get("config", {}),
                                "loaded_config_paths": r.get("loaded_config_paths", []),
                            }
                            for r in all_results
                        ],
                    )
                else:
                    _append_jsonl(partial_jsonl, row)
                with open(partial_summary, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

                m = result["metrics"]
                print(
                    f"[done] method={method} dataset={dataset} seed={seed} "
                    f"proj_dist={m.get('proj_manifold_dist', float('nan')):.6f} "
                    f"recall={m.get('pred_recall', float('nan')):.6f} "
                    f"FPrate={m.get('pred_FPrate', float('nan')):.6f}"
                )
                done_per_method[method] += 1
                _print_progress_snapshot(
                    methods=methods,
                    datasets=datasets,
                    seed_values=seed_values,
                    done_per_method=done_per_method,
                    completed_keys=completed_keys,
                    expected_keys=expected_keys,
                )
                if wb_run is not None:
                    step += 1
                    _sync_resolved_config_to_wandb(
                        wb_run=wb_run,
                        method=str(method),
                        dataset=str(dataset),
                        seed=(int(seed) if seed is not None else None),
                        resolved_config=dict(result.get("config", {})),
                        loaded_paths=list(loaded),
                    )
                    wandb.log({f"{dataset}/{method}/{k}": v for k, v in m.items()}, step=step)
                    _log_dataset_plots_to_wandb(
                        outdir=outdir,
                        method=str(method),
                        dataset=str(dataset),
                        all_datasets=[str(d) for d in datasets],
                        step=step,
                    )

    summary_path = os.path.join(outdir, "summary_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    per_case_csv = os.path.join(outdir, "per_case_metrics.csv")
    leaderboard_csv = os.path.join(outdir, "leaderboard_mean_by_method.csv")
    _write_per_case_csv(per_case_csv, all_results)
    leaderboard = _write_leaderboard_csv(leaderboard_csv, all_results)

    print(f"[saved] {summary_path}")
    print(f"[saved] {partial_jsonl}")
    print(f"[saved] {partial_summary}")
    print(f"[saved] {per_case_csv}")
    print(f"[saved] {leaderboard_csv}")
    print("[leaderboard]", leaderboard)

    if wb_run is not None:
        for row in leaderboard:
            method = str(row["method"])
            payload = {f"avg/{method}/{k}": v for k, v in row.items() if k not in ("method", "n_cases") and isinstance(v, (int, float))}
            if payload:
                step += 1
                wandb.log(payload, step=step)
        wandb.finish()


if __name__ == "__main__":
    main()
