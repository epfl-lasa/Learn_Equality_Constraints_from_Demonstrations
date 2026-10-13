#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.eval_runner import run_eval_metrics
from experiments.dataset_resolve import resolve_dataset
from methods.utils.feature_normalizer import FeatureNormalizer
from methods.utils.projection import project_points_with_steps_numpy
from planning.checkpoint_loader import load_constraint_model_from_checkpoint


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summary_paths(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("runs/*/summary_*.json")))
            out.extend(sorted(p.glob("*/summary_*.json")))
        else:
            out.append(p)
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _projector_val(cfg: Any, key: str, default: float | int) -> float | int:
    proj = getattr(cfg, "projector", None)
    if isinstance(proj, dict) and key in proj:
        return proj[key]
    return default


def _generic_project_fn(cfg: Any):
    def _project(model, x0: np.ndarray, eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
        return project_points_with_steps_numpy(
            model,
            x0.astype(np.float32),
            device=str(getattr(cfg, "device", "cpu")),
            proj_steps=int(_projector_val(cfg, "steps", 100)),
            proj_alpha=float(_projector_val(cfg, "alpha", 0.3)),
            proj_min_steps=int(_projector_val(cfg, "min_steps", 0)),
            f_abs_stop=float(eps_stop),
        )

    return _project


def _feature_normalizer_from_ckpt(ckpt: dict[str, Any], in_dim: int) -> FeatureNormalizer:
    fd = ckpt.get("feature_normalizer", {}) or {}
    return FeatureNormalizer(
        center=np.asarray(fd.get("center", np.zeros((in_dim,), dtype=np.float32)), dtype=np.float32),
        scale=np.asarray(fd.get("scale", np.ones((in_dim,), dtype=np.float32)), dtype=np.float32),
        angle_dims=tuple(int(v) for v in fd.get("angle_dims", [])),
    )


def _evaluate_one(summary_path: Path, *, device: str) -> dict[str, Any]:
    summary = _load_summary(summary_path)
    method = str(summary["method"])
    dataset = str(summary["dataset"])
    cfg_dict = dict(summary.get("config", {}) or {})
    cfg_dict["device"] = str(device)
    cfg = SimpleNamespace(**cfg_dict)
    ds = resolve_dataset(dataset, cfg, optimize_ur5_train_only=True)
    x_train = np.asarray(ds["x_train"], dtype=np.float32)
    ckpt_path = str(summary["ckpt_path"])

    if method == "dataaug":
        eval_model, ckpt = load_constraint_model_from_checkpoint(ckpt_path, device=device, disable_feature_normalizer=False)
        base_model, _ = load_constraint_model_from_checkpoint(ckpt_path, device=device, disable_feature_normalizer=True)
        normalizer = _feature_normalizer_from_ckpt(ckpt, int(ckpt["in_dim"]))
        base_project = _generic_project_fn(cfg)

        def project_fn(_model, x0: np.ndarray, eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
            if normalizer.enabled:
                x0_norm = normalizer.transform(np.asarray(x0, dtype=np.float32))
                proj_norm, steps = base_project(base_model, x0_norm, eps_stop)
                return normalizer.inverse_transform(proj_norm).astype(np.float32), steps
            return base_project(_model, x0, eps_stop)

        model = eval_model
    elif method == "vae":
        model, _ckpt = load_constraint_model_from_checkpoint(ckpt_path, device=device)

        def project_fn(_model, x0: np.ndarray, _eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
            with torch.no_grad():
                xt = torch.from_numpy(x0.astype(np.float32)).to(device)
                x_proj = model.project_tensor(xt).detach().cpu().numpy().astype(np.float32)
            return x_proj, np.ones((len(x_proj),), dtype=np.float32)

    else:
        model, _ckpt = load_constraint_model_from_checkpoint(ckpt_path, device=device)
        project_fn = _generic_project_fn(cfg)

    metrics, _eval_cfg, _artifacts = run_eval_metrics(
        cfg=cfg,
        method_key=method,
        dataset_name=dataset,
        model=model,
        x_train=x_train,
        project_fn=project_fn,
        embed_fn=None,
        postprocess_fn=None,
    )
    out = {
        "summary_path": str(summary_path),
        "method": method,
        "dataset": dataset,
        "seed": int(cfg_dict.get("seed", -1)),
    }
    out.update(metrics)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "proj_manifold_dist",
        "real_iiwa_proj_position_error_mm_mean",
        "real_iiwa_proj_position_error_mm_median",
        "real_iiwa_proj_position_error_mm_p95",
        "real_iiwa_proj_orientation_error_deg_mean",
        "real_iiwa_proj_orientation_error_deg_median",
        "real_iiwa_proj_orientation_error_deg_p95",
        "proj_v_residual",
        "proj_steps",
    ]
    out: list[dict[str, Any]] = []
    for method in sorted({str(r["method"]) for r in rows}):
        group = [r for r in rows if str(r["method"]) == method]
        row: dict[str, Any] = {"method": method, "n": len(group)}
        for key in metrics:
            vals = np.asarray([float(r[key]) for r in group if key in r and np.isfinite(float(r[key]))], dtype=np.float64)
            row[f"{key}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate saved real-iiwa checkpoints with split position/orientation projection errors.")
    parser.add_argument("inputs", nargs="+", help="Summary json files or output directories containing runs/*/summary_*.json")
    parser.add_argument("--methods", default="oncl,dataaug,vae", help="Comma-separated method filter.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="outputs/analysis/real_iiwa_checkpoint_split_errors/per_seed.csv")
    parser.add_argument("--summary-out", default="outputs/analysis/real_iiwa_checkpoint_split_errors/summary.csv")
    args = parser.parse_args()

    methods = {m.strip() for m in str(args.methods).split(",") if m.strip()}
    rows: list[dict[str, Any]] = []
    for path in _summary_paths(list(args.inputs)):
        summary = _load_summary(path)
        if methods and str(summary.get("method")) not in methods:
            continue
        rows.append(_evaluate_one(path, device=str(args.device)))
        print(f"[done] {rows[-1]['method']} seed={rows[-1]['seed']} {path}")
    _write_csv(Path(args.out), rows)
    summary_rows = _mean_std(rows)
    _write_csv(Path(args.summary_out), summary_rows)
    print(f"[saved] {args.out}")
    print(f"[saved] {args.summary_out}")


if __name__ == "__main__":
    main()
