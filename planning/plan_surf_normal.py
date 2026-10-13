#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.dataset_resolve import resolve_dataset
from datasets.constraint_datasets import (
    sine_surface_affine_params,
    sine_surface_apply_affine_scalar,
    sine_surface_apply_affine_xy,
    sine_surface_z_and_normal_from_xy,
)
from planning.checkpoint_loader import load_constraint_model_from_checkpoint
from planning.planner import build_linear_path, plan_path
from methods.utils.projection import project_points_with_steps_numpy


DEFAULT_CKPT = (
    "outputs/bench/paper/oncl/6d_surf_normal_traj/"
    "6d_surf_normal_traj_oncl_seed0_model.pt"
)
DEFAULT_OUTDIR = (
    "outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/"
    "oncl/surfnormal_planning_paper"
)
DATASET_NAME = "6d_surf_normal_traj"


@dataclass
class PairCase:
    start: np.ndarray
    goal: np.ndarray
    waypoint: np.ndarray | None
    traj: np.ndarray
    plan_seconds: float
    min_obstacle_dist_xy: float


def _load_pairs_file(path: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(_resolve_path(path))
    if "starts" in data and "goals" in data:
        starts = np.asarray(data["starts"], dtype=np.float32)
        goals = np.asarray(data["goals"], dtype=np.float32)
    elif "pairs" in data:
        pairs = np.asarray(data["pairs"], dtype=np.float32)
        starts = pairs[:, 0, :]
        goals = pairs[:, 1, :]
    else:
        raise ValueError(f"pairs file must contain starts/goals or pairs arrays: {path}")
    if starts.ndim != 2 or goals.ndim != 2 or starts.shape != goals.shape or starts.shape[1] != int(dim):
        raise ValueError(f"expected starts/goals with shape (N,{dim}), got {starts.shape} and {goals.shape}")
    return starts.astype(np.float32), goals.astype(np.float32)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_repo_root(), path))


def _choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _rpy_zyx_to_local_z(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    z_x = cy * sp * cr + sy * sr
    z_y = sy * sp * cr - cy * sr
    z_z = cp * cr
    z = np.stack([z_x, z_y, z_z], axis=1).astype(np.float32)
    z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return z


def _workspace_surface_z_and_normal_from_xy(x: np.ndarray, y: np.ndarray, surface_cfg: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    return sine_surface_z_and_normal_from_xy(x, y, surface_cfg)


def _error_to_true_constraint(x: np.ndarray, surface_cfg: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    z_true, n_true = _workspace_surface_z_and_normal_from_xy(x[:, 0], x[:, 1], surface_cfg)
    pos_err = np.abs(x[:, 2] - z_true)
    z_axis = _rpy_zyx_to_local_z(x[:, 3], x[:, 4], x[:, 5])
    cosv = np.clip(np.sum(z_axis * n_true, axis=1), -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(cosv))
    return pos_err.astype(np.float32), ang_err_deg.astype(np.float32)


def _line_intersects_circle_xy(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float) -> bool:
    p = a[:2].astype(np.float64)
    q = b[:2].astype(np.float64)
    c = center.astype(np.float64)
    v = q - p
    vv = float(np.dot(v, v))
    if vv < 1e-12:
        return float(np.linalg.norm(p - c)) <= float(radius)
    t = float(np.dot(c - p, v) / vv)
    t = max(0.0, min(1.0, t))
    foot = p + t * v
    d = float(np.linalg.norm(foot - c))
    return d <= float(radius)


def _path_min_dist_to_circle_xy(path: np.ndarray, center: np.ndarray, radius: float) -> float:
    d = np.linalg.norm(path[:, :2] - center[None, :], axis=1) - float(radius)
    return float(np.min(d))


def _pick_pair(
    rng: np.random.Generator,
    pool: np.ndarray,
    center: np.ndarray,
    radius: float,
    min_dist: float,
    max_dist: float,
    max_y: float,
    avoid_points: np.ndarray | None,
    diverse_min_dist: float,
    force_cross_obstacle: bool,
    cross_radius_scale: float,
    cross_side_margin: float,
    cross_y_band: float,
    tries: int,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(max(1, int(tries))):
        i = int(rng.integers(0, len(pool)))
        j = int(rng.integers(0, len(pool)))
        if i == j:
            continue
        s = pool[i]
        g = pool[j]
        if float(s[1]) > float(max_y) or float(g[1]) > float(max_y):
            continue
        d = float(np.linalg.norm(s[:3] - g[:3]))
        if d < float(min_dist) or d > float(max_dist):
            continue
        # keep start/goal clear from obstacle center
        if float(np.linalg.norm(s[:2] - center)) <= radius or float(np.linalg.norm(g[:2] - center)) <= radius:
            continue
        if bool(force_cross_obstacle):
            # Force pair geometry to exhibit obstacle avoidance:
            # 1) straight segment intersects obstacle disk
            # 2) endpoints are on opposite sides in x around obstacle center
            # 3) endpoints remain near obstacle y-band for clearer visualization
            if not _line_intersects_circle_xy(s, g, center=center, radius=float(radius) * float(cross_radius_scale)):
                continue
            sx = float(s[0] - center[0])
            gx = float(g[0] - center[0])
            if sx * gx >= 0.0:
                continue
            if abs(sx) < float(cross_side_margin) or abs(gx) < float(cross_side_margin):
                continue
            if abs(float(s[1] - center[1])) > float(cross_y_band) or abs(float(g[1] - center[1])) > float(cross_y_band):
                continue
        if avoid_points is not None and len(avoid_points) > 0 and float(diverse_min_dist) > 0.0:
            d_s = np.linalg.norm(avoid_points - s[None, :], axis=1)
            d_g = np.linalg.norm(avoid_points - g[None, :], axis=1)
            if float(np.min(d_s)) < float(diverse_min_dist) or float(np.min(d_g)) < float(diverse_min_dist):
                continue
        return s.copy(), g.copy()
    raise RuntimeError("failed to sample valid start-goal pair")


def _pick_waypoint_around_circle(
    start: np.ndarray,
    goal: np.ndarray,
    pool: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    v = goal[:2] - start[:2]
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        v = np.array([1.0, 0.0], dtype=np.float32)
    else:
        v = (v / n).astype(np.float32)
    perp = np.array([-v[1], v[0]], dtype=np.float32)
    safe_r = float(radius) * 1.35
    c1 = center + safe_r * perp
    c2 = center - safe_r * perp
    pool_xy = pool[:, :2]
    d1 = np.linalg.norm(pool_xy - c1[None, :], axis=1)
    d2 = np.linalg.norm(pool_xy - c2[None, :], axis=1)
    for idx in np.argsort(np.minimum(d1, d2)):
        cand = pool[int(idx)]
        if float(np.linalg.norm(cand[:2] - center)) > float(radius):
            return cand.copy()
    return pool[int(np.argmin(np.linalg.norm(pool_xy - c1[None, :], axis=1)))].copy()


def _planner_cfg(
    *,
    device: str,
    opt_steps: int,
    opt_lr: float,
    lam_manifold: float,
    lam_len_joint: float,
    opt_lam_smooth: float,
    trust_scale: float,
    proj_steps: int,
    proj_alpha: float,
    proj_min_steps: int,
    obstacle_enable: bool,
    obstacle_center_xy: tuple[float, float],
    obstacle_radius: float,
    obstacle_margin: float,
    lam_obstacle: float,
) -> Any:
    return SimpleNamespace(
        device=device,
        planner={
            "opt_steps": int(opt_steps),
            "opt_lr": float(opt_lr),
            "lam_manifold": float(lam_manifold),
            "lam_len_joint": float(lam_len_joint),
            "opt_lam_smooth": float(opt_lam_smooth),
            "trust_scale": float(trust_scale),
            "obstacle_enable": bool(obstacle_enable),
            "obstacle_center_xy": [float(obstacle_center_xy[0]), float(obstacle_center_xy[1])],
            "obstacle_radius": float(obstacle_radius),
            "obstacle_margin": float(obstacle_margin),
            "lam_obstacle": float(lam_obstacle),
            "obstacle_exclude_endpoints": True,
        },
        projector={
            "steps": int(proj_steps),
            "alpha": float(proj_alpha),
            "min_steps": int(proj_min_steps),
        },
    )


def _piecewise_linear_path(
    x_start: np.ndarray,
    x_mid: np.ndarray,
    x_goal: np.ndarray,
    *,
    n_waypoints: int,
) -> np.ndarray:
    n = int(max(3, n_waypoints))
    n1 = max(2, n // 2 + 1)
    n2 = max(2, n - n1 + 1)
    p1 = build_linear_path(
        np.asarray(x_start, dtype=np.float32),
        np.asarray(x_mid, dtype=np.float32),
        n_waypoints=n1,
        periodic=False,
    )
    p2 = build_linear_path(
        np.asarray(x_mid, dtype=np.float32),
        np.asarray(x_goal, dtype=np.float32),
        n_waypoints=n2,
        periodic=False,
    )
    out = np.concatenate([p1[:-1], p2], axis=0).astype(np.float32)
    if len(out) != n:
        idx = np.linspace(0, len(out) - 1, num=n, dtype=np.int32)
        out = out[idx].astype(np.float32)
    out[0] = np.asarray(x_start, dtype=np.float32)
    out[-1] = np.asarray(x_goal, dtype=np.float32)
    out[:, 3:6] = _wrap_pi(out[:, 3:6])
    return out


def _point_project_path(
    model: nn.Module,
    init_path: np.ndarray,
    *,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    device: str,
    proj_steps: int,
    proj_alpha: float,
    proj_min_steps: int,
    f_abs_stop: float | None = None,
) -> np.ndarray:
    proj, _steps = project_points_with_steps_numpy(
        model,
        np.asarray(init_path, dtype=np.float32),
        device=str(device),
        proj_steps=int(proj_steps),
        proj_alpha=float(proj_alpha),
        proj_min_steps=int(proj_min_steps),
        f_abs_stop=(None if f_abs_stop is None else float(f_abs_stop)),
    )
    out = np.asarray(proj, dtype=np.float32)
    out[0] = np.asarray(x_start, dtype=np.float32)
    out[-1] = np.asarray(x_goal, dtype=np.float32)
    out[:, 3:6] = _wrap_pi(out[:, 3:6])
    return out


def _load_model(
    ckpt_path: str,
    device: str,
    *,
    disable_feature_normalizer: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    return load_constraint_model_from_checkpoint(
        ckpt_path,
        device=device,
        disable_feature_normalizer=bool(disable_feature_normalizer),
    )


def _build_data_pool(ckpt: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    cfg_d = dict(ckpt.get("cfg", {}))
    cfg_d["seed"] = int(seed)
    if "n_train" not in cfg_d:
        cfg_d["n_train"] = 1024
    if "traj_gene_n_grid" not in cfg_d:
        cfg_d["traj_gene_n_grid"] = max(4096, int(cfg_d["n_train"]))
    cfg = SimpleNamespace(**cfg_d)
    ds = resolve_dataset(DATASET_NAME, cfg, optimize_ur5_train_only=True)
    x_train = np.asarray(ckpt.get("x_train", ds["x_train"]), dtype=np.float32)
    grid = np.asarray(ds["grid"], dtype=np.float32)
    if x_train.shape[1] >= 6 and grid.shape[1] >= 6:
        return x_train, grid
    raise RuntimeError("dataset samples do not have 6D workspace-pose shape")


def _plan_with_obstacle(
    *,
    model: nn.Module,
    cfg: Any,
    pool: np.ndarray,
    rng: np.random.Generator,
    n_trajs: int,
    n_waypoints: int,
    planner_mode: str,
    center: np.ndarray,
    radius: float,
    min_dist: float,
    max_dist: float,
    max_y: float,
    diverse_min_dist: float,
    force_cross_obstacle: bool,
    cross_radius_scale: float,
    cross_side_margin: float,
    cross_y_band: float,
    pair_tries: int,
    selected_endpoints: list[np.ndarray] | None = None,
    fixed_pairs: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[PairCase]:
    cases: list[PairCase] = []
    if selected_endpoints is None:
        selected_endpoints = []
    planner_mode_norm = str(planner_mode).strip().lower()
    if planner_mode_norm not in {"traj_opt", "point_project"}:
        raise ValueError(f"unsupported planner_mode: {planner_mode}")
    if fixed_pairs is not None:
        starts, goals = fixed_pairs
        iterator = range(int(starts.shape[0]))
    else:
        starts = goals = None
        iterator = range(int(n_trajs))
    for pair_idx in iterator:
        avoid = np.asarray(selected_endpoints, dtype=np.float32) if len(selected_endpoints) > 0 else None
        if fixed_pairs is not None:
            start = np.asarray(starts[int(pair_idx)], dtype=np.float32).copy()
            goal = np.asarray(goals[int(pair_idx)], dtype=np.float32).copy()
        else:
            try:
                start, goal = _pick_pair(
                    rng=rng,
                    pool=pool,
                    center=center,
                    radius=radius * 1.05,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    max_y=max_y,
                    avoid_points=avoid,
                    diverse_min_dist=diverse_min_dist,
                    force_cross_obstacle=force_cross_obstacle,
                    cross_radius_scale=cross_radius_scale,
                    cross_side_margin=cross_side_margin,
                    cross_y_band=cross_y_band,
                    tries=pair_tries,
                )
            except RuntimeError:
                # Relax strict crossing constraints as fallback, to keep generation robust.
                start, goal = _pick_pair(
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
        t0 = time.time()
        waypoint = None
        if planner_mode_norm == "traj_opt":
            # Single-shot trajectory optimization to avoid sharp corner from segment stitching.
            traj = plan_path(
                model=model,
                x_start=start,
                x_goal=goal,
                cfg=cfg,
                planner_name="traj_opt",
                n_waypoints=int(n_waypoints),
                dataset_name=DATASET_NAME,
                periodic_joint=False,
            )
        else:
            safety_radius = float(radius) + float(cfg.planner.get("obstacle_margin", 0.0))
            intersects = _line_intersects_circle_xy(
                start,
                goal,
                center=center,
                radius=safety_radius,
            )
            if intersects:
                waypoint = _pick_waypoint_around_circle(
                    start,
                    goal,
                    pool=pool,
                    center=center,
                    radius=safety_radius,
                ).astype(np.float32)
                init = _piecewise_linear_path(
                    start,
                    waypoint,
                    goal,
                    n_waypoints=int(n_waypoints),
                )
            else:
                init = build_linear_path(
                    np.asarray(start, dtype=np.float32),
                    np.asarray(goal, dtype=np.float32),
                    n_waypoints=int(n_waypoints),
                    periodic=False,
                ).astype(np.float32)
            traj = _point_project_path(
                model,
                init,
                x_start=start,
                x_goal=goal,
                device=str(cfg.device),
                proj_steps=int(cfg.projector["steps"]),
                proj_alpha=float(cfg.projector["alpha"]),
                proj_min_steps=int(cfg.projector["min_steps"]),
                f_abs_stop=None,
            )
        plan_t = float(time.time() - t0)
        traj[:, 3:6] = _wrap_pi(traj[:, 3:6])
        min_d = _path_min_dist_to_circle_xy(traj, center=center, radius=radius)
        cases.append(
            PairCase(
                start=start.astype(np.float32),
                goal=goal.astype(np.float32),
                waypoint=(None if waypoint is None else waypoint.astype(np.float32)),
                traj=traj.astype(np.float32),
                plan_seconds=plan_t,
                min_obstacle_dist_xy=min_d,
            )
        )
        selected_endpoints.append(start.astype(np.float32))
        selected_endpoints.append(goal.astype(np.float32))
    return cases


def _plot_planning_paper(
    *,
    model: torch.nn.Module,
    cfg: Any,
    pool: np.ndarray,
    surface_points: int,
    surface_grid: int,
    surface_knn: int,
    surface_mask_percentile: float,
    surface_source: str,
    cases: list[PairCase],
    center: np.ndarray,
    radius: float,
    out_path: str,
    traj_labels: list[str] | None = None,
    traj_colors: list[str] | None = None,
    traj_linestyles: list[str] | None = None,
    surface_affine_scale: float = 1.0,
    surface_affine_offset: np.ndarray | None = None,
    global_paper_view: bool = False,
    surface_cfg: Any | None = None,
    orientation_arrow_length: float | None = None,
) -> None:
    aff_scale = float(surface_affine_scale)
    aff_offset = (
        np.asarray(surface_affine_offset, dtype=np.float32).reshape(3)
        if surface_affine_offset is not None
        else np.zeros((3,), dtype=np.float32)
    )
    env_scale, env_offset = sine_surface_affine_params(surface_cfg)
    g_res = int(max(24, surface_grid))
    if str(surface_source).lower() == "learned":
        n_surface = min(max(400, int(surface_points)), len(pool))
        idx = np.random.choice(len(pool), size=n_surface, replace=False)
        x0 = pool[idx].astype(np.float32).copy()
        # Jitter then project back: visualize learned manifold in workspace xyz.
        x0[:, :3] += np.random.normal(scale=0.12, size=(len(x0), 3)).astype(np.float32)
        x0[:, 3:6] = _wrap_pi(x0[:, 3:6] + np.random.normal(scale=0.25, size=(len(x0), 3)).astype(np.float32))
        x_proj, _ = project_points_with_steps_numpy(
            model,
            x0,
            device=str(cfg.device),
            proj_steps=int(cfg.projector.get("steps", 120)),
            proj_alpha=float(cfg.projector.get("alpha", 0.3)),
            proj_min_steps=int(cfg.projector.get("min_steps", 30)),
            f_abs_stop=None,
        )
        surf_xyz = x_proj[:, :3].astype(np.float32)
        sx = surf_xyz[:, 0]
        sy = surf_xyz[:, 1]
        sz_raw = surf_xyz[:, 2]

        # Build a continuous learned surface via local IDW interpolation on regular XY grid.
        x_lo, x_hi = float(np.percentile(sx, 1)), float(np.percentile(sx, 99))
        y_lo, y_hi = float(np.percentile(sy, 1)), float(np.percentile(sy, 99))
        gx = np.linspace(x_lo, x_hi, g_res).astype(np.float32)
        gy = np.linspace(y_lo, y_hi, g_res).astype(np.float32)
        gxx, gyy = np.meshgrid(gx, gy)
        gxy = np.stack([gxx.reshape(-1), gyy.reshape(-1)], axis=1).astype(np.float32)
        pxy = np.stack([sx, sy], axis=1).astype(np.float32)

        d2 = np.sum((gxy[:, None, :] - pxy[None, :, :]) ** 2, axis=2)
        k = int(max(4, min(int(surface_knn), len(surf_xyz))))
        knn_idx = np.argpartition(d2, kth=max(0, k - 1), axis=1)[:, :k]
        knn_d2 = np.take_along_axis(d2, knn_idx, axis=1)
        knn_z = sz_raw[knn_idx]
        w = 1.0 / (knn_d2 + 1e-6)
        z_idw = np.sum(w * knn_z, axis=1) / np.sum(w, axis=1)
        nn_dist = np.sqrt(np.min(knn_d2, axis=1))
        nn_thr = float(np.percentile(nn_dist, float(surface_mask_percentile)))
        z_grid = z_idw.reshape(gxx.shape)
        z_grid[nn_dist.reshape(gxx.shape) > nn_thr] = np.nan
        if abs(aff_scale - 1.0) > 1e-8 or float(np.linalg.norm(aff_offset)) > 1e-8:
            sx_t = aff_scale * gxx + aff_offset[0]
            sy_t = aff_scale * gyy + aff_offset[1]
            sz_t = aff_scale * z_grid + aff_offset[2]
        else:
            sx_t, sy_t, sz_t = gxx, gyy, z_grid
        surface_label = "Learned equality constraint"
    else:
        gx = (float(env_scale) * np.linspace(-2.0, 2.0, g_res).astype(np.float32) + float(env_offset[0])).astype(np.float32)
        gy = (float(env_scale) * np.linspace(-2.0, 2.0, g_res).astype(np.float32) + float(env_offset[1])).astype(np.float32)
        gxx, gyy = np.meshgrid(gx, gy)
        z_grid, _ = _workspace_surface_z_and_normal_from_xy(gxx.reshape(-1), gyy.reshape(-1), surface_cfg)
        z_grid = z_grid.reshape(gxx.shape)
        if abs(aff_scale - 1.0) > 1e-8 or float(np.linalg.norm(aff_offset)) > 1e-8:
            sx_t = aff_scale * gxx + aff_offset[0]
            sy_t = aff_scale * gyy + aff_offset[1]
            sz_t = aff_scale * z_grid + aff_offset[2]
        else:
            sx_t, sy_t, sz_t = gxx, gyy, z_grid
        surface_label = "True equality constraint"

    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    ):
        fig = plt.figure(figsize=(3.45, 2.9))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            sx_t,
            sy_t,
            sz_t,
            color="#22d3ee",
            alpha=0.30,
            linewidth=0.23,
            edgecolor=(0, 0, 0, 0.24),
            antialiased=True,
            shade=True,
            label=surface_label,
        )

        th = np.linspace(0.0, 2.0 * np.pi, 80)
        cx = center[0] + radius * np.cos(th)
        cy = center[1] + radius * np.sin(th)
        z_inter, _ = _workspace_surface_z_and_normal_from_xy(cx.astype(np.float32), cy.astype(np.float32), surface_cfg)
        cx_t = aff_scale * cx + aff_offset[0]
        cy_t = aff_scale * cy + aff_offset[1]
        z_inter_t = aff_scale * z_inter + aff_offset[2]
        # Use scale-proportional obstacle height in affine views to preserve apparent size.
        z_band = 0.18 * max(aff_scale, 1e-6)
        z_min = float(np.min(z_inter_t) - z_band)
        z_max = float(np.max(z_inter_t) + z_band)
        zz_c = np.linspace(z_min, z_max, 24)
        th_m, zz_m = np.meshgrid(th, zz_c)
        cx_m = aff_scale * (center[0] + radius * np.cos(th_m)) + aff_offset[0]
        cy_m = aff_scale * (center[1] + radius * np.sin(th_m)) + aff_offset[1]
        ax.plot_surface(
            cx_m,
            cy_m,
            zz_m,
            color="#f59e0b",
            alpha=0.26,
            linewidth=0.2,
            edgecolor=(0.40, 0.18, 0.0, 0.35),
            antialiased=True,
            shade=False,
            label="Known cylindrical obstacle",
        )
        # Intersection curve between obstacle cylinder and surface.
        ax.plot(
            cx_t,
            cy_t,
            z_inter_t,
            color="#7c2d12",
            linewidth=1.2,
            alpha=0.95,
        )

        colors = list(traj_colors) if traj_colors is not None else ["#b91c1c", "#2563eb", "#16a34a", "#ea580c"]
        linestyles = list(traj_linestyles) if traj_linestyles is not None else ["-"] * len(cases)
        traj_legend_handles: list[Line2D] = []
        for i, c in enumerate(cases):
            col = colors[i % len(colors)]
            ls = linestyles[i % len(linestyles)] if len(linestyles) > 0 else "-"
            tr = c.traj
            ax.plot(
                tr[:, 0],
                tr[:, 1],
                tr[:, 2],
                color=col,
                linewidth=1.1,
                linestyle=ls,
                alpha=0.95,
                label="_nolegend_",
            )
            if i < 3:
                traj_legend_handles.append(Line2D([0], [0], color=col, linewidth=1.5, linestyle=ls))
            ax.scatter(tr[0, 0], tr[0, 1], tr[0, 2], s=8, c=col, marker="o", alpha=0.95, label="_nolegend_")
            ax.scatter(tr[-1, 0], tr[-1, 1], tr[-1, 2], s=8, c=col, marker="s", alpha=0.95, label="_nolegend_")
            # Draw orientation arrows every ~10 points along trajectory.
            step = 10
            idx = np.arange(0, tr.shape[0], step, dtype=int)
            if len(idx) > 0 and idx[-1] != tr.shape[0] - 1:
                idx = np.concatenate([idx, np.array([tr.shape[0] - 1], dtype=int)])
            rpy = tr[idx, 3:6]
            dirs = _rpy_zyx_to_local_z(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            arrow_len = float(
                orientation_arrow_length
                if orientation_arrow_length is not None
                else max(0.08, 0.24 * aff_scale)
            )
            ax.quiver(
                tr[idx, 0],
                tr[idx, 1],
                tr[idx, 2],
                dirs[:, 0],
                dirs[:, 1],
                dirs[:, 2],
                length=arrow_len,
                normalize=True,
                color=col,
                linewidths=0.75,
                alpha=0.88,
                arrow_length_ratio=0.28,
            )

        ax.set_xlabel("x", fontsize=8, labelpad=1)
        ax.set_ylabel("y", fontsize=8, labelpad=1)
        ax.set_zlabel("z", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=7, pad=0)
        if bool(global_paper_view):
            x_lo = float(aff_scale * (float(env_scale) * (-2.0) + float(env_offset[0])) + aff_offset[0])
            x_hi = float(aff_scale * (float(env_scale) * (2.0) + float(env_offset[0])) + aff_offset[0])
            y_lo = float(aff_scale * (float(env_scale) * (-2.0) + float(env_offset[1])) + aff_offset[1])
            y_hi = float(aff_scale * (float(env_scale) * (2.0) + float(env_offset[1])) + aff_offset[1])
            z_lo = float(np.nanmin(sz_t))
            z_hi = float(np.nanmax(sz_t))
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_zlim(z_lo, z_hi)
            try:
                ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, z_hi - z_lo))
            except Exception:
                pass
        else:
            x_vals_surface = np.asarray(sx_t[np.isfinite(sz_t)], dtype=np.float32)
            y_vals_surface = np.asarray(sy_t[np.isfinite(sz_t)], dtype=np.float32)
            x_vals_traj = np.concatenate([c.traj[:, 0] for c in cases], axis=0).astype(np.float32)
            y_vals_traj = np.concatenate([c.traj[:, 1] for c in cases], axis=0).astype(np.float32)
            z_vals_surface = np.asarray(sz_t[np.isfinite(sz_t)], dtype=np.float32)
            z_vals_traj = np.concatenate([c.traj[:, 2] for c in cases], axis=0).astype(np.float32)
            x_all = np.concatenate([x_vals_surface, x_vals_traj, cx_t.astype(np.float32)], axis=0)
            y_all = np.concatenate([y_vals_surface, y_vals_traj, cy_t.astype(np.float32)], axis=0)
            z_all = np.concatenate([z_vals_surface, z_vals_traj, z_inter_t.astype(np.float32)], axis=0)
            x_rng = float(max(np.max(x_all) - np.min(x_all), 1e-3))
            y_rng = float(max(np.max(y_all) - np.min(y_all), 1e-3))
            z_rng = float(max(np.max(z_all) - np.min(z_all), 1e-3))
            x_pad = max(0.06, 0.12 * x_rng)
            y_pad = max(0.06, 0.12 * y_rng)
            z_pad = max(0.05, 0.18 * z_rng)
            ax.set_xlim(float(np.min(x_all) - x_pad), float(np.max(x_all) + x_pad))
            ax.set_ylim(float(np.min(y_all) - y_pad), float(np.max(y_all) + y_pad))
            ax.set_zlim(float(np.min(z_all) - z_pad), float(np.max(z_all) + 0.55 * z_pad))
            try:
                ax.set_box_aspect((x_rng + 2.0 * x_pad, y_rng + 2.0 * y_pad, z_rng + 1.55 * z_pad))
            except Exception:
                pass
        ax.view_init(elev=60, azim=-64)
        h_surface = Patch(facecolor="#22d3ee", edgecolor=(0, 0, 0, 0.24), alpha=0.30)
        h_obstacle = Patch(facecolor="#f59e0b", edgecolor=(0.40, 0.18, 0.0, 0.35), alpha=0.26)
        handles: list[Any] = [h_surface, h_obstacle]
        labels = [surface_label, "Known cylindrical obstacle"]
        if traj_labels is not None and len(traj_labels) > 0:
            for i, lab in enumerate(traj_labels[: len(cases)]):
                ls = linestyles[i % len(linestyles)] if len(linestyles) > 0 else "-"
                handles.append(Line2D([0], [0], color=colors[i % len(colors)], linewidth=1.5, linestyle=ls))
                labels.append(str(lab))
        elif len(traj_legend_handles) > 0:
            handles.append(tuple(traj_legend_handles))
            labels.append("Plan with learned constraint")
        ax.legend(
            handles,
            labels,
            loc="upper left",
            frameon=False,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
        )
        # Pull camera slightly closer for a tighter view.
        try:
            ax.dist = 8.5
        except Exception:
            pass
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.998)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


def _plot_error_distribution_paper(
    *,
    pos_err: np.ndarray,
    ang_err_deg: np.ndarray,
    out_path: str,
    pos_err_compare: np.ndarray | None = None,
    ang_err_deg_compare: np.ndarray | None = None,
    primary_label: str = "Primary",
    compare_label: str = "Compare",
) -> None:
    pos_arrays = [np.asarray(pos_err, dtype=np.float32).reshape(-1)]
    ang_arrays = [np.asarray(ang_err_deg, dtype=np.float32).reshape(-1)]
    if pos_err_compare is not None and ang_err_deg_compare is not None:
        pos_arrays.append(np.asarray(pos_err_compare, dtype=np.float32).reshape(-1))
        ang_arrays.append(np.asarray(ang_err_deg_compare, dtype=np.float32).reshape(-1))
    pos_cat = np.concatenate(pos_arrays, axis=0)
    ang_cat = np.concatenate(ang_arrays, axis=0)
    pos_cap = max(float(np.percentile(pos_cat, 99)), 1e-4)
    ang_cap = max(float(np.percentile(ang_cat, 99)), 1.0)
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    ):
        fig = plt.figure(figsize=(3.45, 1.9))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        bins_pos = np.linspace(0.0, pos_cap, 34)
        bins_ang = np.linspace(0.0, ang_cap, 34)
        ax1.hist(pos_err, bins=bins_pos, color="#4C78A8", alpha=0.82, label=primary_label)
        if pos_err_compare is not None:
            ax1.hist(pos_err_compare, bins=bins_pos, color="#F58518", alpha=0.62, label=compare_label)
        ax1.set_xlabel("Position Error")
        ax1.set_ylabel("Count")
        ax1.grid(alpha=0.22)
        if pos_err_compare is not None:
            ax1.legend(loc="best", fontsize=6.5)

        ax2.hist(ang_err_deg, bins=bins_ang, color="#E45756", alpha=0.82, label=primary_label)
        if ang_err_deg_compare is not None:
            ax2.hist(ang_err_deg_compare, bins=bins_ang, color="#72B7B2", alpha=0.62, label=compare_label)
        ax2.set_xlabel("Orientation Error (deg)")
        ax2.grid(alpha=0.22)
        if ang_err_deg_compare is not None:
            ax2.legend(loc="best", fontsize=6.5)

        fig.tight_layout(pad=0.25)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def _save_pointwise_csv(path: str, cases: list[PairCase], pos_err: np.ndarray, ang_err_deg: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "traj_id",
        "point_id",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "pos_err",
        "ori_err_deg",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        off = 0
        for ti, c in enumerate(cases):
            for pi in range(c.traj.shape[0]):
                row = c.traj[pi]
                w.writerow(
                    {
                        "traj_id": ti,
                        "point_id": pi,
                        "x": float(row[0]),
                        "y": float(row[1]),
                        "z": float(row[2]),
                        "roll": float(row[3]),
                        "pitch": float(row[4]),
                        "yaw": float(row[5]),
                        "pos_err": float(pos_err[off + pi]),
                        "ori_err_deg": float(ang_err_deg[off + pi]),
                    }
                )
            off += c.traj.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load trained ONCL model for 6D surfnormal and run obstacle-aware point-to-point planning."
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Checkpoint path.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    parser.add_argument("--disable-feature-normalizer", type=int, default=0, help="1 to bypass FeatureNormalizedModel for timing-only ablations.")
    parser.add_argument("--seed", type=int, default=91262, help="Random seed for planning pair sampling.")
    parser.add_argument("--n-trajs", type=int, default=3, help="How many planned trajectories to generate.")
    parser.add_argument("--pairs-file", default="", help="Optional .npz with fixed starts/goals arrays.")
    parser.add_argument("--n-waypoints", type=int, default=70, help="Waypoints per trajectory.")
    parser.add_argument("--pair-min-dist", type=float, default=0.8, help="Min start-goal distance in xyz.")
    parser.add_argument("--pair-max-dist", type=float, default=2.8, help="Max start-goal distance in xyz.")
    parser.add_argument("--pair-max-y", type=float, default=1.2, help="Require sampled start/goal y <= this value.")
    parser.add_argument("--pair-diverse-min-dist", type=float, default=0.9, help="Minimum distance between all sampled starts/goals to encourage diversity.")
    parser.add_argument("--pair-force-cross-obstacle", type=int, default=1, help="1 to force obstacle-crossing start-goal pairs, 0 to disable.")
    parser.add_argument("--pair-cross-radius-scale", type=float, default=1.03, help="Intersection radius multiplier for forced-cross pair sampling.")
    parser.add_argument("--pair-cross-side-margin", type=float, default=0.12, help="Minimum |x-center_x| for endpoints when forcing obstacle crossing.")
    parser.add_argument("--pair-cross-y-band", type=float, default=1.35, help="Maximum |y-center_y| for endpoints when forcing obstacle crossing.")
    parser.add_argument("--pair-tries", type=int, default=1600, help="Pair sampling retries.")

    parser.add_argument("--obstacle-cx", type=float, default=0.0, help="Obstacle center x in workspace.")
    parser.add_argument("--obstacle-cy", type=float, default=-0.7, help="Obstacle center y in workspace.")
    parser.add_argument("--obstacle-radius", type=float, default=0.5, help="Obstacle radius in xy plane.")
    parser.add_argument("--obstacle-margin", type=float, default=0.15, help="Safety margin added to obstacle radius in traj_opt.")
    parser.add_argument("--lam-obstacle", type=float, default=20.0, help="Weight of obstacle penalty in traj_opt.")

    parser.add_argument("--opt-steps", type=int, default=1240, help="traj_opt iterations.")
    parser.add_argument("--opt-lr", type=float, default=0.01, help="traj_opt learning rate.")
    parser.add_argument("--planner-mode", choices=["traj_opt", "point_project"], default="traj_opt", help="Planning mode.")
    parser.add_argument("--lam-manifold", type=float, default=1.0, help="Manifold loss weight.")
    parser.add_argument("--lam-len-joint", type=float, default=0.4, help="Path length loss weight.")
    parser.add_argument("--lam-smooth", type=float, default=0.2, help="Smoothness loss weight.")
    parser.add_argument("--trust-scale", type=float, default=0.8, help="Trust region scale.")
    parser.add_argument("--proj-steps", type=int, default=120, help="Projector steps.")
    parser.add_argument("--proj-alpha", type=float, default=0.3, help="Projector alpha.")
    parser.add_argument("--proj-min-steps", type=int, default=30, help="Projector min steps.")
    parser.add_argument("--surface-points", type=int, default=5000, help="Number of projected points used to build learned surface.")
    parser.add_argument("--surface-grid", type=int, default=120, help="Grid resolution per axis for learned surface rendering.")
    parser.add_argument("--surface-knn", type=int, default=40, help="KNN count for IDW interpolation on learned surface.")
    parser.add_argument("--surface-mask-percentile", type=float, default=93.0, help="Percentile threshold for masking unsupported grid areas.")
    parser.add_argument("--surface-source", choices=["true", "learned"], default="true", help="Surface source for rendering.")
    args = parser.parse_args()

    ckpt_path = _resolve_path(args.ckpt)
    outdir = _resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    device = _choose_device(str(args.device))
    model, ckpt = _load_model(
        ckpt_path,
        device=device,
        disable_feature_normalizer=bool(int(args.disable_feature_normalizer)),
    )
    surface_cfg = SimpleNamespace(**dict(ckpt.get("cfg", {})))
    fixed_pairs = _load_pairs_file(str(args.pairs_file), 6) if str(args.pairs_file).strip() else None
    x_train, pool = _build_data_pool(ckpt, seed=int(args.seed))
    _, surface_offset = sine_surface_affine_params(surface_cfg)
    center = sine_surface_apply_affine_xy(
        np.asarray([[float(args.obstacle_cx), float(args.obstacle_cy)]], dtype=np.float32),
        surface_cfg,
    )[0]
    radius = float(sine_surface_apply_affine_scalar(float(args.obstacle_radius), surface_cfg))
    obstacle_margin = float(sine_surface_apply_affine_scalar(float(args.obstacle_margin), surface_cfg))
    pair_min_dist = float(sine_surface_apply_affine_scalar(float(args.pair_min_dist), surface_cfg))
    pair_max_dist = float(sine_surface_apply_affine_scalar(float(args.pair_max_dist), surface_cfg))
    pair_max_y = float(float(args.pair_max_y) * float(sine_surface_affine_params(surface_cfg)[0]) + float(surface_offset[1]))
    pair_cross_side_margin = float(sine_surface_apply_affine_scalar(float(args.pair_cross_side_margin), surface_cfg))
    pair_cross_y_band = float(sine_surface_apply_affine_scalar(float(args.pair_cross_y_band), surface_cfg))
    cfg = _planner_cfg(
        device=device,
        opt_steps=int(args.opt_steps),
        opt_lr=float(args.opt_lr),
        lam_manifold=float(args.lam_manifold),
        lam_len_joint=float(args.lam_len_joint),
        opt_lam_smooth=float(args.lam_smooth),
        trust_scale=float(args.trust_scale),
        proj_steps=int(args.proj_steps),
        proj_alpha=float(args.proj_alpha),
        proj_min_steps=int(args.proj_min_steps),
        obstacle_enable=True,
        obstacle_center_xy=(float(center[0]), float(center[1])),
        obstacle_radius=float(radius),
        obstacle_margin=float(obstacle_margin),
        lam_obstacle=float(args.lam_obstacle),
    )
    cases = _plan_with_obstacle(
        model=model,
        cfg=cfg,
        pool=pool,
        rng=rng,
        n_trajs=int(args.n_trajs),
        n_waypoints=int(args.n_waypoints),
        planner_mode=str(args.planner_mode),
        center=center,
        radius=radius,
        min_dist=float(pair_min_dist),
        max_dist=float(pair_max_dist),
        max_y=float(pair_max_y),
        diverse_min_dist=float(args.pair_diverse_min_dist),
        force_cross_obstacle=bool(args.pair_force_cross_obstacle),
        cross_radius_scale=float(args.pair_cross_radius_scale),
        cross_side_margin=float(pair_cross_side_margin),
        cross_y_band=float(pair_cross_y_band),
        pair_tries=int(args.pair_tries),
        fixed_pairs=fixed_pairs,
    )

    pts_all = np.concatenate([c.traj for c in cases], axis=0).astype(np.float32)
    pos_err, ang_err_deg = _error_to_true_constraint(pts_all, surface_cfg)

    point_csv = os.path.join(outdir, "surfnormal_planning_pointwise_errors.csv")
    _save_pointwise_csv(point_csv, cases, pos_err, ang_err_deg)

    traj_fig = os.path.join(outdir, "surfnormal_obstacle_planning.png")
    _plot_planning_paper(
        model=model,
        cfg=cfg,
        pool=pool,
        surface_points=int(args.surface_points),
        surface_grid=int(args.surface_grid),
        surface_knn=int(args.surface_knn),
        surface_mask_percentile=float(args.surface_mask_percentile),
        surface_source=str(args.surface_source),
        cases=cases,
        center=center,
        radius=radius,
        out_path=traj_fig,
        surface_cfg=surface_cfg,
    )
    dist_fig = os.path.join(outdir, "6d_surf_normal_traj_oncl_planning_error_distribution_paper.png")
    _plot_error_distribution_paper(pos_err=pos_err, ang_err_deg=ang_err_deg, out_path=dist_fig)

    summary = {
        "task": "surfnormal_obstacle",
        "dataset": DATASET_NAME,
        "ckpt": ckpt_path,
        "n_trajs": int(len(cases)),
        "seed": int(args.seed),
        "pairs_file": str(args.pairs_file),
        "obstacle": {
            "type": "circle_xy",
            "center_xy": [float(center[0]), float(center[1])],
            "radius": float(radius),
        },
        "mean_position_error": float(np.mean(pos_err)),
        "std_position_error": float(np.std(pos_err)),
        "mean_orientation_error_deg": float(np.mean(ang_err_deg)),
        "std_orientation_error_deg": float(np.std(ang_err_deg)),
        "mean_plan_seconds": float(np.mean([c.plan_seconds for c in cases])),
        "std_plan_seconds": float(np.std([c.plan_seconds for c in cases], ddof=1)) if len(cases) > 1 else 0.0,
        "sem_plan_seconds": float(np.std([c.plan_seconds for c in cases], ddof=1) / math.sqrt(len(cases))) if len(cases) > 1 else 0.0,
        "paths": [
            {
                "index": int(i),
                "plan_seconds": float(c.plan_seconds),
                "start": c.start.astype(float).tolist(),
                "goal": c.goal.astype(float).tolist(),
                "min_obstacle_dist_xy": float(c.min_obstacle_dist_xy),
            }
            for i, c in enumerate(cases)
        ],
        "min_obstacle_dist_xy_per_traj": [float(c.min_obstacle_dist_xy) for c in cases],
        "outputs": {
            "pointwise_csv": point_csv,
            "planning_plot": traj_fig,
            "distribution_plot": dist_fig,
        },
    }
    summary_path = os.path.join(outdir, "surfnormal_planning_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        "[summary] "
        f"pos_mean={summary['mean_position_error']:.6f}, pos_std={summary['std_position_error']:.6f}, "
        f"ori_mean_deg={summary['mean_orientation_error_deg']:.4f}, ori_std_deg={summary['std_orientation_error_deg']:.4f}, "
        f"plan_time_mean={summary['mean_plan_seconds']:.3f}s"
    )
    print(f"[saved] {summary_path}")
    print(f"[saved] {traj_fig}")
    print(f"[saved] {dist_fig}")
    print(f"[saved] {point_csv}")


if __name__ == "__main__":
    main()
