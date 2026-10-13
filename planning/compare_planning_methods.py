#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


DEFAULT_BENCH = "outputs/bench/paper"
DEFAULT_METHODS = "oncl,dataaug,ecomann,vae"


@dataclass(frozen=True)
class PlannerSpec:
    dataset_name: str
    tag: str
    module: str
    summary_file: str


def _repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_REPO_ROOT, path))


def _dataset_spec(name: str) -> PlannerSpec:
    key = str(name).strip().lower()
    if key in {"6d", "surfnormal", "6d_surfnormal", "6d_surf_normal_traj"}:
        return PlannerSpec(
            dataset_name="6d_surf_normal_traj",
            tag="6d_surfnormal",
            module="planning.plan_surf_normal",
            summary_file="surfnormal_planning_summary.json",
        )
    if key in {"12d", "dual", "dual_arm", "12d_dual_arm", "12d_dual_arm_traj"}:
        return PlannerSpec(
            dataset_name="12d_dual_arm_traj",
            tag="12d_dual_arm",
            module="planning.plan_dual_arm",
            summary_file="dual_arm_learned_planning_summary.json",
        )
    raise ValueError(f"unsupported dataset: {name}")


def _method_list(text: str) -> list[str]:
    out = [m.strip().lower() for m in str(text).split(",") if m.strip()]
    if not out:
        raise ValueError("empty --methods")
    return out


def _checkpoint_path(bench: str, dataset_name: str, method: str, seed: int = 0) -> str:
    return os.path.join(_repo_path(bench), method, dataset_name,
                        f"{dataset_name}_{method}_seed{seed}_model.pt")


def _forwarded_value(args: list[str], name: str, default: Any, cast: Any) -> Any:
    prefix = name + "="
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            return cast(args[i + 1])
        if arg.startswith(prefix):
            return cast(arg[len(prefix) :])
    return default


def _generate_fixed_pairs(
    *,
    spec: PlannerSpec,
    bench: str,
    out_path: str,
    n_pairs: int,
    planner_args: list[str],
    checkpoint_seed: int = 0,
) -> str:
    import torch

    seed = int(_forwarded_value(planner_args, "--seed", 91262 if spec.tag == "6d_surfnormal" else 0, int))
    ckpt_path = _checkpoint_path(bench, spec.dataset_name, "oncl", checkpoint_seed)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(seed)
    starts: list[np.ndarray] = []
    goals: list[np.ndarray] = []

    if spec.tag == "6d_surfnormal":
        from datasets.constraint_datasets import (
            sine_surface_affine_params,
            sine_surface_apply_affine_scalar,
            sine_surface_apply_affine_xy,
        )
        from planning import plan_surf_normal as ps

        pool, _grid = ps._build_data_pool(ckpt, seed=seed)
        surface_cfg = SimpleNamespace(**dict(ckpt.get("cfg", {})))
        center_raw = np.array(
            [
                float(_forwarded_value(planner_args, "--obstacle-cx", 0.0, float)),
                float(_forwarded_value(planner_args, "--obstacle-cy", -0.7, float)),
            ],
            dtype=np.float32,
        ).reshape(1, 2)
        center = sine_surface_apply_affine_xy(center_raw, surface_cfg)[0].astype(np.float32)
        radius = float(sine_surface_apply_affine_scalar(float(_forwarded_value(planner_args, "--obstacle-radius", 0.5, float)), surface_cfg))
        min_dist = float(sine_surface_apply_affine_scalar(float(_forwarded_value(planner_args, "--pair-min-dist", 0.8, float)), surface_cfg))
        max_dist = float(sine_surface_apply_affine_scalar(float(_forwarded_value(planner_args, "--pair-max-dist", 2.8, float)), surface_cfg))
        surface_scale, surface_offset = sine_surface_affine_params(surface_cfg)
        max_y = float(float(_forwarded_value(planner_args, "--pair-max-y", 1.2, float)) * float(surface_scale) + float(surface_offset[1]))
        cross_side_margin = float(sine_surface_apply_affine_scalar(float(_forwarded_value(planner_args, "--pair-cross-side-margin", 0.12, float)), surface_cfg))
        cross_y_band = float(sine_surface_apply_affine_scalar(float(_forwarded_value(planner_args, "--pair-cross-y-band", 1.35, float)), surface_cfg))
        diverse_min_dist = float(_forwarded_value(planner_args, "--pair-diverse-min-dist", 0.9, float))
        force_cross = bool(int(_forwarded_value(planner_args, "--pair-force-cross-obstacle", 1, int)))
        pair_tries = int(_forwarded_value(planner_args, "--pair-tries", 1600, int))
        cross_radius_scale = float(_forwarded_value(planner_args, "--pair-cross-radius-scale", 1.03, float))
        selected: list[np.ndarray] = []
        for _ in range(int(n_pairs)):
            avoid = np.asarray(selected, dtype=np.float32) if selected else None
            try:
                start, goal = ps._pick_pair(
                    rng=rng,
                    pool=pool,
                    center=center,
                    radius=radius * 1.05,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    max_y=max_y,
                    avoid_points=avoid,
                    diverse_min_dist=diverse_min_dist,
                    force_cross_obstacle=force_cross,
                    cross_radius_scale=cross_radius_scale,
                    cross_side_margin=cross_side_margin,
                    cross_y_band=cross_y_band,
                    tries=pair_tries,
                )
            except RuntimeError:
                start, goal = ps._pick_pair(
                    rng=rng,
                    pool=pool,
                    center=center,
                    radius=radius * 1.05,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    max_y=max_y,
                    avoid_points=avoid,
                    diverse_min_dist=diverse_min_dist,
                    force_cross_obstacle=False,
                    cross_radius_scale=cross_radius_scale,
                    cross_side_margin=cross_side_margin,
                    cross_y_band=cross_y_band,
                    tries=max(400, pair_tries // 2),
                )
            starts.append(start.astype(np.float32))
            goals.append(goal.astype(np.float32))
            selected.extend([start.astype(np.float32), goal.astype(np.float32)])
    else:
        from planning import plan_dual_arm as pd

        cfg, x_train, grid = pd._build_data_pool(ckpt, seed=seed)
        reference_mode = str(_forwarded_value(planner_args, "--reference-mode", "none", str))
        if reference_mode == "circle":
            pair_pool = pd._circle_pool(grid if len(grid) > 0 else x_train, x_train)
        else:
            pair_pool = pd._easy_pool(
                grid if len(grid) > 0 else x_train,
                x_train,
                center_keep_ratio=float(_forwarded_value(planner_args, "--easy-center-keep-ratio", 0.75, float)),
            )
        min_center = float(_forwarded_value(planner_args, "--pair-min-center-dist", 0.45, float))
        max_center = float(_forwarded_value(planner_args, "--pair-max-center-dist", 0.95, float))
        if reference_mode == "circle":
            min_center *= 0.88
            max_center *= 0.88
        pair_tries = int(_forwarded_value(planner_args, "--pair-tries", 2000, int))
        for _ in range(int(n_pairs)):
            start, goal = pd._pick_pair(
                pair_pool,
                rng,
                min_center_dist=min_center,
                max_center_dist=max_center,
                tries=pair_tries,
            )
            starts.append(start.astype(np.float32))
            goals.append(goal.astype(np.float32))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    starts_arr = np.stack(starts, axis=0).astype(np.float32)
    goals_arr = np.stack(goals, axis=0).astype(np.float32)
    np.savez_compressed(
        out_path,
        starts=starts_arr,
        goals=goals_arr,
        pairs=np.stack([starts_arr, goals_arr], axis=1).astype(np.float32),
        dataset=spec.dataset_name,
        seed=np.array([seed], dtype=np.int64),
        source_ckpt=ckpt_path,
    )
    print(f"[saved] fixed pairs: {out_path} ({len(starts_arr)} queries)")
    return out_path


def _run_streamed(cmd: list[str], *, cwd: str, log_path: str) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return int(proc.wait())


def _load_planning_time(summary_path: str) -> dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    n_trajs = int(summary.get("n_trajs", 0))
    per_path = [float(p.get("plan_seconds", 0.0)) for p in summary.get("paths", []) if "plan_seconds" in p]
    if per_path:
        mean_s = float(np.mean(per_path))
        std_s = float(np.std(per_path, ddof=1)) if len(per_path) > 1 else 0.0
        sem_s = float(std_s / np.sqrt(len(per_path))) if len(per_path) > 1 else 0.0
        total_s = float(np.sum(per_path))
        n_trajs = int(len(per_path))
    elif "mean_plan_seconds" in summary:
        mean_s = float(summary["mean_plan_seconds"])
        std_s = float(summary.get("std_plan_seconds", 0.0))
        sem_s = float(summary.get("sem_plan_seconds", 0.0))
        total_s = float(mean_s * max(0, n_trajs))
    else:
        mean_s = 0.0
        std_s = 0.0
        sem_s = 0.0
        total_s = 0.0
    return {
        "n_trajs": n_trajs,
        "mean_plan_seconds": mean_s,
        "std_plan_seconds": std_s,
        "sem_plan_seconds": sem_s,
        "total_plan_seconds": total_s,
        "per_query_plan_seconds": per_path,
        "summary": summary,
    }


def _write_outputs(rows: list[dict[str, Any]], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "planning_time_compare.csv")
    json_path = os.path.join(outdir, "planning_time_compare.json")
    fields = [
        "dataset",
        "method",
        "status",
        "returncode",
        "n_trajs",
        "mean_plan_seconds",
        "std_plan_seconds",
        "sem_plan_seconds",
        "total_plan_seconds",
        "pairs_file",
        "ckpt",
        "outdir",
        "summary_path",
        "log_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[saved] {csv_path}")
    print(f"[saved] {json_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same planner settings on multiple trained methods and "
            "summarize planning time. Extra unknown arguments are forwarded "
            "to the selected planner, e.g. --n-trajs 3 --opt-steps 500."
        )
    )
    parser.add_argument("--dataset", default="12d_dual_arm", help="6d_surfnormal or 12d_dual_arm.")
    parser.add_argument("--bench", default=DEFAULT_BENCH, help="Benchmark root containing method checkpoint dirs.")
    parser.add_argument("--methods", default=DEFAULT_METHODS, help="Comma-separated methods.")
    parser.add_argument("--checkpoint-seed", type=int, default=0, help="Training seed of the saved model; independent of the planning seed.")
    parser.add_argument("--outdir", default="", help="Comparison output root.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pairs-file", default="", help="Existing .npz starts/goals file passed to every method.")
    parser.add_argument("--fixed-pairs", type=int, default=0, help="Generate this many fixed start/goal pairs from ONCL and reuse them for every method.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child planner runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print child commands without running them.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed method.")
    args, planner_args = parser.parse_known_args(argv)

    spec = _dataset_spec(args.dataset)
    methods = _method_list(args.methods)
    outdir = _repo_path(args.outdir) if args.outdir else os.path.join(_repo_path(args.bench), "planning_compare", spec.tag)
    logs_dir = os.path.join(outdir, "logs")
    rows: list[dict[str, Any]] = []
    pairs_file = _repo_path(str(args.pairs_file)) if str(args.pairs_file).strip() else ""
    if not pairs_file and int(args.fixed_pairs) > 0:
        pairs_file = _generate_fixed_pairs(
            spec=spec,
            bench=str(args.bench),
            out_path=os.path.join(outdir, f"fixed_pairs_{int(args.fixed_pairs)}.npz"),
            n_pairs=int(args.fixed_pairs),
            planner_args=planner_args,
            checkpoint_seed=args.checkpoint_seed,
        )

    for method in methods:
        ckpt = _checkpoint_path(args.bench, spec.dataset_name, method, args.checkpoint_seed)
        method_outdir = os.path.join(outdir, method)
        summary_path = os.path.join(method_outdir, spec.summary_file)
        log_path = os.path.join(logs_dir, f"{method}.log")
        cmd = [
            str(args.python),
            "-m",
            spec.module,
            "--ckpt",
            ckpt,
            "--outdir",
            method_outdir,
            "--device",
            str(args.device),
        ]
        if pairs_file:
            cmd.extend(["--pairs-file", pairs_file])
        cmd.extend(planner_args)
        print(f"[method] {method}")
        print("$ " + " ".join(cmd))

        row: dict[str, Any] = {
            "dataset": spec.dataset_name,
            "method": method,
            "ckpt": ckpt,
            "outdir": method_outdir,
            "summary_path": summary_path,
            "log_path": log_path,
            "pairs_file": pairs_file,
        }
        if not os.path.exists(ckpt):
            row.update({"status": "missing_ckpt", "returncode": ""})
            rows.append(row)
            print(f"[skip] checkpoint not found: {ckpt}")
            if bool(args.stop_on_error):
                break
            continue
        if bool(args.dry_run):
            row.update({"status": "dry_run", "returncode": ""})
            rows.append(row)
            continue

        rc = _run_streamed(cmd, cwd=_REPO_ROOT, log_path=log_path)
        row["returncode"] = rc
        if rc == 0 and os.path.exists(summary_path):
            timing = _load_planning_time(summary_path)
            row.update(
                {
                    "status": "ok",
                    "n_trajs": timing["n_trajs"],
                    "mean_plan_seconds": timing["mean_plan_seconds"],
                    "std_plan_seconds": timing["std_plan_seconds"],
                    "sem_plan_seconds": timing["sem_plan_seconds"],
                    "total_plan_seconds": timing["total_plan_seconds"],
                }
            )
        else:
            row["status"] = "failed" if rc != 0 else "missing_summary"
        rows.append(row)
        _write_outputs(rows, outdir)
        if row["status"] != "ok" and bool(args.stop_on_error):
            break

    _write_outputs(rows, outdir)


if __name__ == "__main__":
    main()
