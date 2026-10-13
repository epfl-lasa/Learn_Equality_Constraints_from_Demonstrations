#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.constraint_datasets import _dual_arm_pose_from_s_u
from evaluation.evaluator import (
    _dual_arm_curve_center_tnb_from_s,
    _dual_arm_pose_analytic_dist_embed,
    _dual_arm_pose_analytic_target_raw,
    _dual_arm_pose_params,
    _rpy_zyx_to_rotmat_batch,
    compute_eps_stop,
    resolve_eval_cfg,
)
from experiments.dataset_resolve import resolve_dataset
from planning.checkpoint_loader import load_constraint_model_from_checkpoint
from planning.planner import build_linear_path, plan_path
from methods.utils.projection import project_points_with_steps_numpy


DATASET_NAME = "12d_dual_arm_traj"
DEFAULT_CKPT = "outputs/bench/paper/oncl/12d_dual_arm_traj/12d_dual_arm_traj_oncl_seed0_model.pt"
DEFAULT_OUTDIR = "outputs/bench/dual_arm_learned_planning"


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


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_REPO_ROOT, path))


def _choose_device(device: str) -> str:
    if device != "auto":
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _load_constraint_model(
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


def _build_data_pool(ckpt: dict[str, Any], seed: int) -> tuple[Any, np.ndarray, np.ndarray]:
    cfg_d = dict(ckpt.get("cfg", {}))
    cfg_d["seed"] = int(seed)
    cfg_d.setdefault("n_train", 1000)
    cfg_d.setdefault("n_grid", cfg_d.get("traj_gene_n_grid", 8192))
    cfg = SimpleNamespace(**cfg_d)
    ds = resolve_dataset(str(ckpt.get("dataset", DATASET_NAME)), cfg)
    x_train = np.asarray(ckpt.get("x_train", ds["x_train"]), dtype=np.float32)
    grid = np.asarray(ds["grid"], dtype=np.float32)
    if x_train.ndim != 2 or x_train.shape[1] < 12:
        raise RuntimeError(f"expected 12D dual-arm samples, got x_train shape={x_train.shape}")
    return cfg, x_train[:, :12].astype(np.float32), grid[:, :12].astype(np.float32)


def _centers(x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float32)
    return (0.5 * (xx[:, 0:3] + xx[:, 6:9])).astype(np.float32)


def _estimate_s_u(x: np.ndarray, cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=np.float32)
    p = _dual_arm_pose_params(cfg)
    center_obs = _centers(xx)
    s_grid = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    center_grid, _, _, _ = _dual_arm_curve_center_tnb_from_s(s_grid, cfg)
    d2 = np.sum((center_obs[:, None, 0:2] - center_grid[None, :, 0:2]) ** 2, axis=2)
    s_star = s_grid[np.argmin(d2, axis=1)].astype(np.float32)
    u_star = np.clip(center_obs[:, 2] - p["z_base"], -p["z_half_range"], p["z_half_range"]).astype(np.float32)
    return s_star, u_star


def _task_init_path(start: np.ndarray, goal: np.ndarray, cfg: Any, n_waypoints: int) -> np.ndarray:
    p = _dual_arm_pose_params(cfg)
    s0, u0 = _estimate_s_u(start.reshape(1, -1), cfg)
    s1, u1 = _estimate_s_u(goal.reshape(1, -1), cfg)
    t = np.linspace(0.0, 1.0, int(n_waypoints), dtype=np.float32)
    s = ((1.0 - t) * float(s0[0]) + t * float(s1[0])).astype(np.float32)
    u = ((1.0 - t) * float(u0[0]) + t * float(u1[0])).astype(np.float32)
    path = _dual_arm_pose_from_s_u(
        s,
        u,
        grasp_span=p["grasp_span"],
        x_span=p["x_span"],
        y_amp=p["y_amp"],
        y_freq=p["y_freq"],
        z_base=p["z_base"],
        z_amp=p["z_amp"],
        z_freq=p["z_freq"],
        right_hand_opposite=False,
    ).astype(np.float32)
    path[0] = start.astype(np.float32)
    path[-1] = goal.astype(np.float32)
    path[:, 3:6] = _wrap_pi(path[:, 3:6])
    path[:, 9:12] = _wrap_pi(path[:, 9:12])
    return path


def _center_init_path(start: np.ndarray, goal: np.ndarray, cfg: Any, n_waypoints: int) -> np.ndarray:
    p = _dual_arm_pose_params(cfg)
    c0 = _centers(start.reshape(1, -1))[0].astype(np.float32)
    c1 = _centers(goal.reshape(1, -1))[0].astype(np.float32)
    t = np.linspace(0.0, 1.0, int(n_waypoints), dtype=np.float32).reshape(-1, 1)
    center = ((1.0 - t) * c0.reshape(1, 3) + t * c1.reshape(1, 3)).astype(np.float32)

    s_grid = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    center_grid, _, _, _ = _dual_arm_curve_center_tnb_from_s(s_grid, cfg)
    d2 = np.sum((center[:, None, 0:2] - center_grid[None, :, 0:2]) ** 2, axis=2)
    s_star = s_grid[np.argmin(d2, axis=1)].astype(np.float32)
    _curve_center, tang, y_axis, z_axis = _dual_arm_curve_center_tnb_from_s(s_star, cfg)

    R = np.stack([tang, y_axis, z_axis], axis=2).astype(np.float32)
    sy = -np.clip(R[:, 2, 0], -1.0, 1.0)
    pitch = np.arcsin(sy).astype(np.float32)
    cp = np.cos(pitch)
    roll = np.where(np.abs(cp) > 1e-8, np.arctan2(R[:, 2, 1], R[:, 2, 2]), 0.0).astype(np.float32)
    yaw = np.where(
        np.abs(cp) > 1e-8,
        np.arctan2(R[:, 1, 0], R[:, 0, 0]),
        np.arctan2(-R[:, 0, 1], R[:, 1, 1]),
    ).astype(np.float32)
    rpy = np.stack([roll, pitch, yaw], axis=1).astype(np.float32)

    offset = (0.5 * float(p["grasp_span"]) * tang).astype(np.float32)
    pose_1 = np.concatenate([center - offset, rpy], axis=1).astype(np.float32)
    pose_2 = np.concatenate([center + offset, rpy], axis=1).astype(np.float32)
    path = np.concatenate([pose_1, pose_2], axis=1).astype(np.float32)
    path[0] = start.astype(np.float32)
    path[-1] = goal.astype(np.float32)
    path[:, 3:6] = _wrap_pi(path[:, 3:6])
    path[:, 9:12] = _wrap_pi(path[:, 9:12])
    return path


def _normalize_rows(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(vv, axis=1, keepdims=True).astype(np.float32)
    out = vv / np.maximum(n, float(eps))
    bad = (n.reshape(-1) <= float(eps))
    if np.any(bad):
        out[bad] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return out.astype(np.float32)


def _interp_rpy_shortest(rpy0: np.ndarray, rpy1: np.ndarray, t: np.ndarray) -> np.ndarray:
    a0 = np.asarray(rpy0, dtype=np.float32).reshape(1, 3)
    a1 = np.asarray(rpy1, dtype=np.float32).reshape(1, 3)
    tt = np.asarray(t, dtype=np.float32).reshape(-1, 1)
    delta = _wrap_pi(a1 - a0).reshape(1, 3)
    return _wrap_pi(a0 + tt * delta).astype(np.float32)


def _blend_paths_shortest(path_a: np.ndarray, path_b: np.ndarray, blend: float) -> np.ndarray:
    a = np.asarray(path_a, dtype=np.float32)
    b = np.asarray(path_b, dtype=np.float32)
    w = float(np.clip(blend, 0.0, 1.0))
    if w <= 0.0:
        return a.astype(np.float32)
    if w >= 1.0:
        return b.astype(np.float32)
    out = ((1.0 - w) * a + w * b).astype(np.float32)
    for dims in ([3, 4, 5], [9, 10, 11]):
        aa = a[:, dims].astype(np.float32)
        bb = b[:, dims].astype(np.float32)
        delta = _wrap_pi(bb - aa).astype(np.float32)
        out[:, dims] = _wrap_pi(aa + w * delta).astype(np.float32)
    out[0] = a[0].astype(np.float32)
    out[-1] = a[-1].astype(np.float32)
    return out.astype(np.float32)


def _center_interp_init_path(start: np.ndarray, goal: np.ndarray, n_waypoints: int) -> np.ndarray:
    s = np.asarray(start, dtype=np.float32).reshape(12)
    g = np.asarray(goal, dtype=np.float32).reshape(12)
    c0 = (0.5 * (s[0:3] + s[6:9])).astype(np.float32)
    c1 = (0.5 * (g[0:3] + g[6:9])).astype(np.float32)
    d0 = (s[6:9] - s[0:3]).astype(np.float32)
    d1 = (g[6:9] - g[0:3]).astype(np.float32)
    l0 = float(np.linalg.norm(d0))
    l1 = float(np.linalg.norm(d1))
    dir0 = (d0 / max(l0, 1e-8)).astype(np.float32)
    dir1 = (d1 / max(l1, 1e-8)).astype(np.float32)

    t = np.linspace(0.0, 1.0, int(n_waypoints), dtype=np.float32).reshape(-1, 1)
    center = ((1.0 - t) * c0.reshape(1, 3) + t * c1.reshape(1, 3)).astype(np.float32)
    dirs = _normalize_rows((1.0 - t) * dir0.reshape(1, 3) + t * dir1.reshape(1, 3))
    span = ((1.0 - t[:, 0]) * l0 + t[:, 0] * l1).astype(np.float32)
    offset = (0.5 * span.reshape(-1, 1) * dirs).astype(np.float32)

    left_rpy = _interp_rpy_shortest(s[3:6], g[3:6], t[:, 0])
    right_rpy = _interp_rpy_shortest(s[9:12], g[9:12], t[:, 0])
    left = np.concatenate([center - offset, left_rpy], axis=1).astype(np.float32)
    right = np.concatenate([center + offset, right_rpy], axis=1).astype(np.float32)
    path = np.concatenate([left, right], axis=1).astype(np.float32)
    path[0] = s
    path[-1] = g
    path[:, 3:6] = _wrap_pi(path[:, 3:6])
    path[:, 9:12] = _wrap_pi(path[:, 9:12])
    return path


def _wrap_angle_scalar(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def _circle_center_reference(c0: np.ndarray, c1: np.ndarray, n_waypoints: int, *, bulge_ratio: float) -> np.ndarray:
    c0 = np.asarray(c0, dtype=np.float32).reshape(3)
    c1 = np.asarray(c1, dtype=np.float32).reshape(3)
    n = int(max(2, n_waypoints))
    yz0 = c0[1:3].astype(np.float32)
    yz1 = c1[1:3].astype(np.float32)
    dyz = (yz1 - yz0).astype(np.float32)
    chord = float(np.linalg.norm(dyz))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x_ref = ((1.0 - t) * float(c0[0]) + t * float(c1[0])).astype(np.float32)
    if chord <= 1e-6:
        yz = ((1.0 - t).reshape(-1, 1) * yz0.reshape(1, 2) + t.reshape(-1, 1) * yz1.reshape(1, 2)).astype(np.float32)
        center = np.stack([x_ref, yz[:, 0], yz[:, 1]], axis=1).astype(np.float32)
        center[0] = c0
        center[-1] = c1
        return center

    u = (dyz / chord).astype(np.float32)
    perp = np.asarray([-u[1], u[0]], dtype=np.float32)
    sag = float(max(1e-4, abs(float(bulge_ratio)) * chord))
    radius = float((chord * chord) / (8.0 * sag) + 0.5 * sag)
    midpoint = (0.5 * (yz0 + yz1)).astype(np.float32)
    center_yz = (midpoint + perp * float(radius - sag)).astype(np.float32)
    v0 = (yz0 - center_yz).astype(np.float32)
    v1 = (yz1 - center_yz).astype(np.float32)
    a0 = float(math.atan2(float(v0[1]), float(v0[0])))
    a1 = float(math.atan2(float(v1[1]), float(v1[0])))
    da = _wrap_angle_scalar(a1 - a0)
    ang = (a0 + t * da).astype(np.float32)
    yz = np.stack(
        [
            center_yz[0] + radius * np.cos(ang),
            center_yz[1] + radius * np.sin(ang),
        ],
        axis=1,
    ).astype(np.float32)
    center = np.stack([x_ref, yz[:, 0], yz[:, 1]], axis=1).astype(np.float32)
    center[0] = c0
    center[-1] = c1
    return center


def _apply_circle_reference(path: np.ndarray, *, bulge_ratio: float) -> np.ndarray:
    pp = np.asarray(path, dtype=np.float32).copy()
    center = _centers(pp)
    ref_center = _circle_center_reference(center[0], center[-1], len(pp), bulge_ratio=float(bulge_ratio))
    cur_center = center.astype(np.float32)
    delta = (ref_center - cur_center).astype(np.float32)
    pp[:, 0:3] = (pp[:, 0:3] + delta).astype(np.float32)
    pp[:, 6:9] = (pp[:, 6:9] + delta).astype(np.float32)
    pp[0, 0:3] = path[0, 0:3]
    pp[0, 6:9] = path[0, 6:9]
    pp[-1, 0:3] = path[-1, 0:3]
    pp[-1, 6:9] = path[-1, 6:9]
    pp[:, 3:6] = _wrap_pi(pp[:, 3:6])
    pp[:, 9:12] = _wrap_pi(pp[:, 9:12])
    return pp.astype(np.float32)


def _make_roundtrip_path(path: np.ndarray) -> np.ndarray:
    pp = np.asarray(path, dtype=np.float32)
    if len(pp) <= 1:
        return pp.astype(np.float32)
    back = pp[-2::-1].copy()
    out = np.concatenate([pp, back], axis=0).astype(np.float32)
    out[:, 3:6] = _wrap_pi(out[:, 3:6])
    out[:, 9:12] = _wrap_pi(out[:, 9:12])
    return out


def _make_circle_roundtrip_path(path: np.ndarray, *, bulge_ratio: float) -> np.ndarray:
    pp = np.asarray(path, dtype=np.float32)
    if len(pp) <= 1:
        return pp.astype(np.float32)
    forward = _apply_circle_reference(pp, bulge_ratio=float(abs(bulge_ratio)))
    backward_full = _apply_circle_reference(pp[::-1].copy(), bulge_ratio=-float(abs(bulge_ratio)))
    out = np.concatenate([forward, backward_full[1:]], axis=0).astype(np.float32)
    out[:, 3:6] = _wrap_pi(out[:, 3:6])
    out[:, 9:12] = _wrap_pi(out[:, 9:12])
    return out


def _resample_path_linear(path: np.ndarray, n_waypoints: int) -> np.ndarray:
    pp = np.asarray(path, dtype=np.float32)
    n_src = int(len(pp))
    n_dst = int(max(2, n_waypoints))
    if n_src == n_dst:
        return pp.astype(np.float32)
    t_src = np.linspace(0.0, 1.0, n_src, dtype=np.float32)
    t_dst = np.linspace(0.0, 1.0, n_dst, dtype=np.float32)
    out = np.zeros((n_dst, pp.shape[1]), dtype=np.float32)
    for d in range(pp.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, pp[:, d]).astype(np.float32)
    out[:, 3:6] = _wrap_pi(out[:, 3:6])
    out[:, 9:12] = _wrap_pi(out[:, 9:12])
    return out.astype(np.float32)


def _unwrap_angle_series(a: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    if aa.size == 0:
        return aa.astype(np.float32)
    out = np.zeros_like(aa, dtype=np.float32)
    out[0] = aa[0]
    for i in range(1, aa.shape[0]):
        out[i] = out[i - 1] + float(_wrap_pi(aa[i] - aa[i - 1]))
    return out.astype(np.float32)


def _resample_path_center_arclength(path: np.ndarray, n_waypoints: int) -> np.ndarray:
    pp = np.asarray(path, dtype=np.float32)
    n_dst = int(max(2, n_waypoints))
    if len(pp) == n_dst:
        # Still reparameterize in case the source has collapsed waypoint clusters.
        pass
    center = _centers(pp).astype(np.float32)
    ds = np.linalg.norm(np.diff(center, axis=0), axis=1).astype(np.float32)
    s = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(ds, dtype=np.float32)], axis=0)
    keep = np.ones(len(pp), dtype=bool)
    keep[1:] = np.diff(s) > 1e-9
    if int(np.count_nonzero(keep)) < 2 or float(s[-1]) <= 1e-9:
        return _resample_path_linear(pp, n_dst).astype(np.float32)
    src = pp[keep].astype(np.float32)
    s_src = s[keep].astype(np.float32)
    s_dst = np.linspace(0.0, float(s_src[-1]), n_dst, dtype=np.float32)
    out = np.zeros((n_dst, pp.shape[1]), dtype=np.float32)
    ang_dims = {3, 4, 5, 9, 10, 11}
    for d in range(pp.shape[1]):
        vals = src[:, d].astype(np.float32)
        if d in ang_dims:
            vals = _unwrap_angle_series(vals)
            out[:, d] = _wrap_pi(np.interp(s_dst, s_src, vals).astype(np.float32))
        else:
            out[:, d] = np.interp(s_dst, s_src, vals).astype(np.float32)
    out[0] = pp[0].astype(np.float32)
    out[-1] = pp[-1].astype(np.float32)
    out[:, 3:6] = _wrap_pi(out[:, 3:6])
    out[:, 9:12] = _wrap_pi(out[:, 9:12])
    return out.astype(np.float32)


def _endpoint_warp_path(template: np.ndarray, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    tmp = np.asarray(template, dtype=np.float32).copy()
    s = np.asarray(start, dtype=np.float32).reshape(12)
    g = np.asarray(goal, dtype=np.float32).reshape(12)
    n = int(len(tmp))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, 1)

    pos_dims = [0, 1, 2, 6, 7, 8]
    d0_pos = (s[pos_dims] - tmp[0, pos_dims]).astype(np.float32).reshape(1, -1)
    d1_pos = (g[pos_dims] - tmp[-1, pos_dims]).astype(np.float32).reshape(1, -1)
    tmp[:, pos_dims] = (tmp[:, pos_dims] + (1.0 - t) * d0_pos + t * d1_pos).astype(np.float32)

    ang_dims = [3, 4, 5, 9, 10, 11]
    d0_ang = _wrap_pi(s[ang_dims] - tmp[0, ang_dims]).astype(np.float32).reshape(1, -1)
    d1_ang = _wrap_pi(g[ang_dims] - tmp[-1, ang_dims]).astype(np.float32).reshape(1, -1)
    tmp[:, ang_dims] = _wrap_pi(tmp[:, ang_dims] + (1.0 - t) * d0_ang + t * d1_ang).astype(np.float32)

    tmp[0] = s
    tmp[-1] = g
    tmp[:, 3:6] = _wrap_pi(tmp[:, 3:6])
    tmp[:, 9:12] = _wrap_pi(tmp[:, 9:12])
    return tmp.astype(np.float32)


def _split_demo_blocks(x_train: np.ndarray, traj_len: int) -> list[np.ndarray]:
    xx = np.asarray(x_train, dtype=np.float32)
    t_len = int(max(2, traj_len))
    n_blk = int(len(xx) // t_len)
    blocks: list[np.ndarray] = []
    for k in range(n_blk):
        blk = xx[k * t_len : (k + 1) * t_len]
        if len(blk) == t_len:
            blocks.append(blk.astype(np.float32))
    if blocks:
        return blocks
    return [xx.astype(np.float32)] if len(xx) > 0 else []


def _nearest_demo_init_path(
    start: np.ndarray,
    goal: np.ndarray,
    x_train: np.ndarray,
    cfg: Any,
    n_waypoints: int,
    *,
    center_interp_blend: float = 0.0,
) -> np.ndarray:
    xx = np.asarray(x_train, dtype=np.float32)
    traj_len = int(getattr(cfg, "traj_len", max(8, min(len(xx), 100))))
    blocks = _split_demo_blocks(xx, traj_len)
    if not blocks:
        return build_linear_path(start, goal, n_waypoints=int(n_waypoints), periodic=False).astype(np.float32)

    s = np.asarray(start, dtype=np.float32).reshape(12)
    g = np.asarray(goal, dtype=np.float32).reshape(12)
    cs = _centers(s.reshape(1, -1))[0]
    cg = _centers(g.reshape(1, -1))[0]

    best_score = float("inf")
    best_template = None
    for blk in blocks:
        for cand in (blk, blk[::-1].copy()):
            c0 = _centers(cand[0:1])[0]
            c1 = _centers(cand[-1:])[0]
            pos_score = float(np.linalg.norm(c0 - cs) + np.linalg.norm(c1 - cg))
            ang_score = float(
                np.mean(np.abs(_wrap_pi(s[[3, 4, 5, 9, 10, 11]] - cand[0, [3, 4, 5, 9, 10, 11]])))
                + np.mean(np.abs(_wrap_pi(g[[3, 4, 5, 9, 10, 11]] - cand[-1, [3, 4, 5, 9, 10, 11]])))
            )
            score = pos_score + 0.10 * ang_score
            if score < best_score:
                best_score = score
                best_template = cand.astype(np.float32)

    assert best_template is not None
    tmp = _resample_path_linear(best_template, int(n_waypoints))
    demo_warp = _endpoint_warp_path(tmp, s, g)
    blend = float(np.clip(center_interp_blend, 0.0, 1.0))
    if blend <= 0.0:
        return demo_warp.astype(np.float32)
    center_interp = _center_interp_init_path(s, g, int(n_waypoints))
    return _blend_paths_shortest(demo_warp, center_interp, blend)


def _rotation_geodesic_deg(rpy_a: np.ndarray, rpy_b: np.ndarray) -> np.ndarray:
    ra = _rpy_zyx_to_rotmat_batch(rpy_a.astype(np.float32))
    rb = _rpy_zyx_to_rotmat_batch(rpy_b.astype(np.float32))
    r_rel = np.einsum("nij,njk->nik", np.transpose(ra, (0, 2, 1)), rb)
    tr = r_rel[:, 0, 0] + r_rel[:, 1, 1] + r_rel[:, 2, 2]
    return np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))).astype(np.float32)


def _model_residual(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        f = model(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device))
        if f.dim() == 1:
            f = f.unsqueeze(1)
        return torch.linalg.norm(f, dim=1).detach().cpu().numpy().astype(np.float32)


def _evaluate_path(model: nn.Module, path: np.ndarray, cfg: Any, device: str) -> dict[str, Any]:
    target = _dual_arm_pose_analytic_target_raw(path[:, :12], cfg)
    center = _centers(path)
    center_t = _centers(target)
    span = np.linalg.norm(path[:, 6:9] - path[:, 0:3], axis=1).astype(np.float32)
    p = _dual_arm_pose_params(cfg)
    out = {
        "left_pos_err_mean": float(np.mean(np.linalg.norm(path[:, 0:3] - target[:, 0:3], axis=1))),
        "left_pos_err_max": float(np.max(np.linalg.norm(path[:, 0:3] - target[:, 0:3], axis=1))),
        "right_pos_err_mean": float(np.mean(np.linalg.norm(path[:, 6:9] - target[:, 6:9], axis=1))),
        "right_pos_err_max": float(np.max(np.linalg.norm(path[:, 6:9] - target[:, 6:9], axis=1))),
        "left_ori_err_deg_mean": float(np.mean(_rotation_geodesic_deg(path[:, 3:6], target[:, 3:6]))),
        "left_ori_err_deg_max": float(np.max(_rotation_geodesic_deg(path[:, 3:6], target[:, 3:6]))),
        "right_ori_err_deg_mean": float(np.mean(_rotation_geodesic_deg(path[:, 9:12], target[:, 9:12]))),
        "right_ori_err_deg_max": float(np.max(_rotation_geodesic_deg(path[:, 9:12], target[:, 9:12]))),
        "center_err_mean": float(np.mean(np.linalg.norm(center - center_t, axis=1))),
        "center_err_max": float(np.max(np.linalg.norm(center - center_t, axis=1))),
        "span_err_mean": float(np.mean(np.abs(span - p["grasp_span"]))),
        "span_err_max": float(np.max(np.abs(span - p["grasp_span"]))),
        "analytic_vector_dist_mean": float(np.mean(_dual_arm_pose_analytic_dist_embed(path[:, :12], cfg))),
        "analytic_vector_dist_max": float(np.max(_dual_arm_pose_analytic_dist_embed(path[:, :12], cfg))),
        "model_residual_mean": float(np.mean(_model_residual(model, path[:, :12], device))),
        "model_residual_max": float(np.max(_model_residual(model, path[:, :12], device))),
        "center_path_length": float(np.sum(np.linalg.norm(np.diff(center, axis=0), axis=1))),
    }
    return out


def _path_error_arrays(model: nn.Module, path: np.ndarray, cfg: Any, device: str) -> dict[str, np.ndarray]:
    pp = np.asarray(path, dtype=np.float32)
    target = _dual_arm_pose_analytic_target_raw(pp[:, :12], cfg)
    left_pos = np.linalg.norm(pp[:, 0:3] - target[:, 0:3], axis=1).astype(np.float32)
    right_pos = np.linalg.norm(pp[:, 6:9] - target[:, 6:9], axis=1).astype(np.float32)
    left_ori = _rotation_geodesic_deg(pp[:, 3:6], target[:, 3:6])
    right_ori = _rotation_geodesic_deg(pp[:, 9:12], target[:, 9:12])
    p = _dual_arm_pose_params(cfg)
    span = np.linalg.norm(pp[:, 6:9] - pp[:, 0:3], axis=1).astype(np.float32)
    return {
        "mean_pos_err": (0.5 * (left_pos + right_pos)).astype(np.float32),
        "mean_ori_err_deg": (0.5 * (left_ori + right_ori)).astype(np.float32),
        "analytic_vector_dist": _dual_arm_pose_analytic_dist_embed(pp[:, :12], cfg).astype(np.float32),
        "model_residual": _model_residual(model, pp[:, :12], device).astype(np.float32),
        "span_err": np.abs(span - float(p["grasp_span"])).astype(np.float32),
        "center_err": np.linalg.norm(_centers(pp) - _centers(target), axis=1).astype(np.float32),
    }


def _concat_error_arrays(
    model: nn.Module,
    paths: list[np.ndarray],
    cfg: Any,
    device: str,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {}
    for path in paths:
        arrs = _path_error_arrays(model, path, cfg, device)
        for key, val in arrs.items():
            chunks.setdefault(key, []).append(np.asarray(val, dtype=np.float32).reshape(-1))
    return {key: np.concatenate(vals, axis=0).astype(np.float32) for key, vals in chunks.items() if vals}


def _stats(vals: np.ndarray) -> dict[str, float]:
    v = np.asarray(vals, dtype=np.float32).reshape(-1)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def _plot_error_distributions(
    *,
    model: nn.Module,
    init_paths: list[np.ndarray],
    planned_paths: list[np.ndarray],
    cfg: Any,
    device: str,
    out_path: str,
) -> dict[str, dict[str, dict[str, float]]]:
    init_err = _concat_error_arrays(model, init_paths, cfg, device)
    plan_err = _concat_error_arrays(model, planned_paths, cfg, device)
    specs = [
        ("mean_pos_err", "mean EE position error", "meter"),
        ("mean_ori_err_deg", "mean EE orientation error", "deg"),
        ("analytic_vector_dist", "analytic embedding L2 error", "vector L2"),
        ("model_residual", "learned constraint residual", "||f_theta(x)||"),
        ("center_err", "center manifold error", "meter"),
        ("span_err", "rigid-link span error", "meter"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.4))
    axes = axes.reshape(-1)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for ax, (key, title, xlabel) in zip(axes, specs):
        a = init_err.get(key, np.zeros((0,), dtype=np.float32))
        b = plan_err.get(key, np.zeros((0,), dtype=np.float32))
        vals = np.concatenate([a, b], axis=0) if len(a) and len(b) else (a if len(a) else b)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        cap = float(np.percentile(vals, 99))
        cap = max(cap, 1e-6)
        bins = np.linspace(0.0, cap, 50)
        if len(a):
            ax.hist(a, bins=bins, color="#64748b", alpha=0.68, label="init")
        if len(b):
            ax.hist(b, bins=bins, color="#16a34a", alpha=0.58, label="planned")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        summary[key] = {"init": _stats(a), "planned": _stats(b)}
    fig.suptitle("12D dual-arm planning error distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _pick_pair(
    pool: np.ndarray,
    rng: np.random.Generator,
    *,
    min_center_dist: float,
    max_center_dist: float,
    tries: int,
) -> tuple[np.ndarray, np.ndarray]:
    c = _centers(pool)
    n = int(len(pool))
    for _ in range(int(max(1, tries))):
        i, j = rng.integers(0, n, size=2)
        if int(i) == int(j):
            continue
        d = float(np.linalg.norm(c[int(i)] - c[int(j)]))
        if float(min_center_dist) <= d <= float(max_center_dist):
            return pool[int(i)].copy(), pool[int(j)].copy()
    # Fallback: choose the farthest candidate from a random anchor among a subset.
    i = int(rng.integers(0, n))
    idx = rng.choice(n, size=min(n, 512), replace=False)
    j = int(idx[np.argmax(np.linalg.norm(c[idx] - c[i], axis=1))])
    return pool[i].copy(), pool[j].copy()


def _easy_pool(
    pool: np.ndarray,
    x_train: np.ndarray,
    *,
    center_keep_ratio: float,
) -> np.ndarray:
    pp = np.asarray(pool, dtype=np.float32)
    xx = np.asarray(x_train, dtype=np.float32)
    if len(pp) == 0 or len(xx) == 0:
        return pp.astype(np.float32)
    keep = float(np.clip(center_keep_ratio, 0.05, 1.0))
    if keep >= 0.999:
        return pp.astype(np.float32)

    c_pool = _centers(pp[:, :12])
    c_train = _centers(xx[:, :12])
    lo_q = 0.5 * (1.0 - keep)
    hi_q = 1.0 - lo_q
    lo = np.quantile(c_train, lo_q, axis=0).astype(np.float32)
    hi = np.quantile(c_train, hi_q, axis=0).astype(np.float32)
    mask = np.all((c_pool >= lo.reshape(1, 3)) & (c_pool <= hi.reshape(1, 3)), axis=1)
    subset = pp[mask]
    if len(subset) >= max(32, min(len(pp), 256)):
        return subset.astype(np.float32)
    return pp.astype(np.float32)


def _circle_pool(pool: np.ndarray, x_train: np.ndarray) -> np.ndarray:
    pp = np.asarray(pool, dtype=np.float32)
    xx = np.asarray(x_train, dtype=np.float32)
    if len(pp) == 0 or len(xx) == 0:
        return pp.astype(np.float32)
    c_pool = _centers(pp[:, :12])
    c_train = _centers(xx[:, :12])

    # Prefer regions away from the flattest central band of the sine manifold.
    x_med = float(np.median(c_train[:, 0]))
    x_abs_train = np.abs(c_train[:, 0] - x_med).astype(np.float32)
    x_thr = float(np.quantile(x_abs_train, 0.45))
    y_abs_train = np.abs(c_train[:, 1]).astype(np.float32)
    y_thr = float(np.quantile(y_abs_train, 0.50))

    mask = (np.abs(c_pool[:, 0] - x_med) >= x_thr) | (np.abs(c_pool[:, 1]) >= y_thr)
    subset = pp[mask]
    if len(subset) >= max(64, min(len(pp), 384)):
        return subset.astype(np.float32)
    return pp.astype(np.float32)


def _planner_cfg(args: argparse.Namespace, device: str, task_cfg: Any) -> Any:
    p = _dual_arm_pose_params(task_cfg)
    z_lo = float(p["z_base"] - p["z_half_range"])
    z_hi = float(p["z_base"] + p["z_half_range"])
    traj_opt_objective = str(args.traj_opt_objective).strip().lower()
    minimal_traj_opt = traj_opt_objective in (
        "minimal",
        "minimal_zsmooth",
        "minimal_refinit",
        "minimal_refinit_zlinear",
    )
    minimal_with_zsmooth = traj_opt_objective == "minimal_zsmooth"
    minimal_with_refinit = traj_opt_objective == "minimal_refinit"
    minimal_with_refinit_zlinear = traj_opt_objective == "minimal_refinit_zlinear"
    planner = {
        "traj_opt_objective": str(args.traj_opt_objective),
        "opt_steps": int(args.opt_steps),
        "opt_lr": float(args.opt_lr),
        "lam_manifold": float(args.lam_manifold),
        "lam_len_joint": (0.0 if minimal_traj_opt else float(args.lam_len)),
        "opt_lam_smooth": (0.0 if minimal_traj_opt else float(args.lam_smooth)),
        "lam_ref_path": (
            float(args.lam_ref_path)
            if (minimal_with_refinit or minimal_with_refinit_zlinear)
            else (0.0 if minimal_traj_opt else float(args.lam_ref_path))
        ),
        "lam_center_z_ref": (
            float(args.lam_center_z_ref)
            if minimal_with_refinit_zlinear
            else (0.0 if minimal_traj_opt else float(args.lam_center_z_ref))
        ),
        "lam_center_z_smooth": (
            float(args.lam_center_z_smooth) if minimal_with_zsmooth else (0.0 if minimal_traj_opt else float(args.lam_center_z_smooth))
        ),
        "trust_scale": float(args.trust_scale),
        "obstacle_enable": False,
    }
    if bool(int(args.enforce_z_bounds)) and not minimal_traj_opt:
        # Only constrain the two TCP z coordinates. No xy/ribbon/analytic-shape
        # penalty is used, so the learned constraint still defines the manifold.
        planner.update({"bound_indices": [2, 8], "bound_lo": z_lo, "bound_hi": z_hi})
    return SimpleNamespace(
        device=str(device),
        planner=planner,
        projector={
            "steps": int(args.proj_steps),
            "alpha": float(args.proj_alpha),
            "min_steps": int(args.proj_min_steps),
        },
    )


def _point_project_path(
    model: nn.Module,
    init_path: np.ndarray,
    *,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    task_cfg: Any,
    device: str,
    proj_steps: int,
    proj_alpha: float,
    proj_min_steps: int,
    f_abs_stop: float | None,
    polish_iters: int,
    center_z_smooth: float,
    center_z_ref: float,
    reproj_steps: int,
    path_polish_steps: int,
    path_polish_lr: float,
    path_lam_manifold: float,
    path_lam_smooth: float,
    path_lam_len: float,
    path_lam_ref_init: float,
    path_lam_ref_proj: float,
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
    p = _dual_arm_pose_params(task_cfg)
    z_lo = float(p["z_base"] - p["z_half_range"])
    z_hi = float(p["z_base"] + p["z_half_range"])
    init_center_z = (0.5 * (init_path[:, 2] + init_path[:, 8])).astype(np.float32)
    polish_n = int(max(0, polish_iters))
    smooth_w = float(np.clip(center_z_smooth, 0.0, 1.0))
    ref_w = float(np.clip(center_z_ref, 0.0, 1.0))
    if polish_n > 0 and len(out) >= 3 and (smooth_w > 0.0 or ref_w > 0.0):
        reproj_n = int(max(0, reproj_steps))
        for _ in range(polish_n):
            center_z = (0.5 * (out[:, 2] + out[:, 8])).astype(np.float32)
            target_z = center_z.copy()
            local_avg = (0.5 * (center_z[:-2] + center_z[2:])).astype(np.float32)
            target_z[1:-1] = (
                (1.0 - smooth_w - ref_w) * center_z[1:-1]
                + smooth_w * local_avg
                + ref_w * init_center_z[1:-1]
            ).astype(np.float32)
            target_z[0] = float(center_z[0])
            target_z[-1] = float(center_z[-1])
            target_z = np.clip(target_z, z_lo, z_hi).astype(np.float32)
            dz = (target_z - center_z).astype(np.float32)
            out[:, 2] = (out[:, 2] + dz).astype(np.float32)
            out[:, 8] = (out[:, 8] + dz).astype(np.float32)
            out[:, 2] = np.clip(out[:, 2], z_lo, z_hi).astype(np.float32)
            out[:, 8] = np.clip(out[:, 8], z_lo, z_hi).astype(np.float32)
            if reproj_n > 0 and len(out) > 2:
                inner, _ = project_points_with_steps_numpy(
                    model,
                    np.asarray(out[1:-1], dtype=np.float32),
                    device=str(device),
                    proj_steps=reproj_n,
                    proj_alpha=float(proj_alpha),
                    proj_min_steps=min(int(proj_min_steps), reproj_n),
                    f_abs_stop=(None if f_abs_stop is None else float(f_abs_stop)),
                )
                out[1:-1] = np.asarray(inner, dtype=np.float32)
                out[1:-1, 2] = np.clip(out[1:-1, 2], z_lo, z_hi).astype(np.float32)
                out[1:-1, 8] = np.clip(out[1:-1, 8], z_lo, z_hi).astype(np.float32)
    polish_steps = int(max(0, path_polish_steps))
    if polish_steps > 0 and len(out) >= 3:
        q = torch.tensor(out.astype(np.float32), device=device, requires_grad=True)
        q_init = torch.tensor(np.asarray(init_path, dtype=np.float32), device=device)
        q_proj = torch.tensor(out.astype(np.float32), device=device)
        q0 = torch.tensor(np.asarray(x_start, dtype=np.float32), device=device)
        qT = torch.tensor(np.asarray(x_goal, dtype=np.float32), device=device)
        opt = torch.optim.Adam([q], lr=float(path_polish_lr))
        for _ in range(polish_steps):
            opt.zero_grad(set_to_none=True)
            f = model(q)
            if f.dim() == 1:
                f = f.unsqueeze(1)
            loss_man = (f ** 2).mean()
            v = q[1:] - q[:-1]
            loss_len = (v ** 2).mean() if len(v) else torch.tensor(0.0, device=q.device)
            if q.shape[0] >= 3:
                dv = v[1:] - v[:-1]
                loss_smooth = (dv ** 2).mean()
            else:
                loss_smooth = torch.tensor(0.0, device=q.device)
            loss_ref_init = ((q - q_init) ** 2).mean()
            loss_ref_proj = ((q - q_proj) ** 2).mean()
            loss = (
                float(path_lam_manifold) * loss_man
                + float(path_lam_smooth) * loss_smooth
                + float(path_lam_len) * loss_len
                + float(path_lam_ref_init) * loss_ref_init
                + float(path_lam_ref_proj) * loss_ref_proj
            )
            loss.backward()
            opt.step()
            with torch.no_grad():
                q[:, 2] = torch.clamp(q[:, 2], min=z_lo, max=z_hi)
                q[:, 8] = torch.clamp(q[:, 8], min=z_lo, max=z_hi)
                q[:, 3:6] = torch.remainder(q[:, 3:6] + np.pi, 2.0 * np.pi) - np.pi
                q[:, 9:12] = torch.remainder(q[:, 9:12] + np.pi, 2.0 * np.pi) - np.pi
                q[0] = q0
                q[-1] = qT
        out = q.detach().cpu().numpy().astype(np.float32)
    out[0] = np.asarray(x_start, dtype=np.float32)
    out[-1] = np.asarray(x_goal, dtype=np.float32)
    out[:, 3:6] = _wrap_pi(out[:, 3:6])
    out[:, 9:12] = _wrap_pi(out[:, 9:12])
    return out


def _plot_paths(paths: list[np.ndarray], out_path: str, cfg: Any, *, plot_arms: bool) -> None:
    fig = plt.figure(figsize=(16.2, 10.2))
    s = np.linspace(-1.0, 1.0, 240, dtype=np.float32)
    mid, _, _, _ = _dual_arm_curve_center_tnb_from_s(s, cfg)
    p = _dual_arm_pose_params(cfg)
    low = mid.copy()
    high = mid.copy()
    low[:, 2] = p["z_base"] - p["z_half_range"]
    high[:, 2] = p["z_base"] + p["z_half_range"]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    all_pts = [low, high, mid]
    centers: list[np.ndarray] = []
    for path in paths:
        center = _centers(path)
        centers.append(center)
        all_pts.append(center)
        if plot_arms:
            all_pts.extend([path[:, 0:3], path[:, 6:9]])
    pts = np.concatenate(all_pts, axis=0)
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    ctr = 0.5 * (mins + maxs)
    half = 0.56 * float(max(np.max(maxs - mins), 1e-3))
    xlim = (float(ctr[0] - half), float(ctr[0] + half))
    ylim = (float(ctr[1] - half), float(ctr[1] + half))
    zlim = (float(ctr[2] - half), float(ctr[2] + half))
    views = [
        (1, 24, -55, "Perspective"),
        (3, 90, -90, "Top-Down (3D)"),
        (4, 12, 0, "Front"),
        (5, 8, 90, "Side"),
    ]

    for subplot_idx, elev, azim, title in views:
        ax = fig.add_subplot(2, 3, subplot_idx, projection="3d")
        ax.plot(mid[:, 0], mid[:, 1], mid[:, 2], "--", color="#334155", lw=1.5, alpha=0.8, label="centerline")
        ax.plot(low[:, 0], low[:, 1], low[:, 2], color="#94a3b8", lw=1.0, alpha=0.55, label="z bounds")
        ax.plot(high[:, 0], high[:, 1], high[:, 2], color="#94a3b8", lw=1.0, alpha=0.55)
        for k, path in enumerate(paths):
            color = colors[k % len(colors)]
            center = centers[k]
            ax.plot(center[:, 0], center[:, 1], center[:, 2], color=color, lw=2.3, label=f"path_{k} center")
            ax.scatter(center[[0, -1], 0], center[[0, -1], 1], center[[0, -1], 2], color=color, s=32)
            if plot_arms:
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, lw=0.9, alpha=0.32)
                ax.plot(path[:, 6], path[:, 7], path[:, 8], color=color, lw=0.9, alpha=0.32)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        if subplot_idx == 1:
            ax.legend(loc="best", fontsize=8)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(mid[:, 0], mid[:, 1], "--", color="#334155", lw=1.5, alpha=0.85, label="centerline")
    for k, path in enumerate(paths):
        color = colors[k % len(colors)]
        center = centers[k]
        ax2.plot(center[:, 0], center[:, 1], color=color, lw=2.3, label=f"path_{k} center")
        ax2.scatter(center[[0, -1], 0], center[[0, -1], 1], color=color, s=28)
        if plot_arms:
            ax2.plot(path[:, 0], path[:, 1], color=color, lw=0.9, alpha=0.26)
            ax2.plot(path[:, 6], path[:, 7], color=color, lw=0.9, alpha=0.26)
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Top-Down (2D XY Projection)")
    ax2.grid(alpha=0.22)
    ax2.legend(loc="best", fontsize=8)

    ax_blank = fig.add_subplot(2, 3, 6)
    ax_blank.axis("off")
    legend_handles = [
        plt.Line2D([0], [0], color="#334155", lw=1.5, linestyle="--", label="centerline"),
        plt.Line2D([0], [0], color="#94a3b8", lw=1.0, label="z bounds"),
    ]
    for k in range(len(paths)):
        color = colors[k % len(colors)]
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2.3, label=f"path_{k} center"))
    ax_blank.legend(handles=legend_handles, loc="center left", fontsize=10, frameon=False, title="Legend")

    fig.suptitle("12D dual-arm learned-constraint planning")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan 12D dual-arm trajectories with a learned constraint field.")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--disable-feature-normalizer", type=int, default=0, help="1 to bypass FeatureNormalizedModel for timing-only ablations.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trajs", type=int, default=3)
    ap.add_argument("--pairs-file", default="", help="Optional .npz with fixed starts/goals arrays.")
    ap.add_argument("--n-waypoints", type=int, default=80)
    ap.add_argument("--planner-mode", choices=["traj_opt", "point_project", "init_only"], default="traj_opt")
    ap.add_argument(
        "--traj-opt-objective",
        choices=["minimal", "minimal_zsmooth", "minimal_refinit", "minimal_refinit_zlinear", "full"],
        default="minimal_refinit_zlinear",
        help="minimal: only learned-manifold loss; minimal_zsmooth: add only center-z second-difference smoothing; minimal_refinit: add only weak init-path trust; minimal_refinit_zlinear: weak init-path trust plus linear center-z reference from start to goal; full: include all trajectory regularizers.",
    )
    ap.add_argument("--init-mode", choices=["task", "nearest_demo", "linear"], default="nearest_demo")
    ap.add_argument("--nearest-demo-center-blend", type=float, default=0.0)
    ap.add_argument("--reference-mode", choices=["none", "circle"], default="none")
    ap.add_argument("--circle-bulge-ratio", type=float, default=0.35)
    ap.add_argument("--roundtrip", type=int, default=0, help="If 1, save/use path as start->goal->start.")
    ap.add_argument("--pair-min-center-dist", type=float, default=0.45)
    ap.add_argument("--pair-max-center-dist", type=float, default=0.95)
    ap.add_argument("--pair-tries", type=int, default=2000)
    ap.add_argument("--easy-center-keep-ratio", type=float, default=0.75)
    ap.add_argument("--opt-steps", type=int, default=3000)
    ap.add_argument("--opt-lr", type=float, default=0.001)
    ap.add_argument("--lam-manifold", type=float, default=1.0)
    ap.add_argument("--lam-len", type=float, default=0.03)
    ap.add_argument("--lam-smooth", type=float, default=0.10)
    ap.add_argument("--lam-ref-path", type=float, default=0.02)
    ap.add_argument("--lam-center-z-ref", type=float, default=0.01)
    ap.add_argument("--lam-center-z-smooth", type=float, default=2.0)
    ap.add_argument("--trust-scale", type=float, default=0.03)
    ap.add_argument(
        "--traj-resample-center-arclength",
        type=int,
        default=1,
        help="1 to reparameterize traj_opt output uniformly by center-path arclength, reducing waypoint collapse/stalls.",
    )
    ap.add_argument("--proj-steps", type=int, default=80)
    ap.add_argument("--proj-alpha", type=float, default=0.08)
    ap.add_argument("--proj-min-steps", type=int, default=20)
    ap.add_argument("--point-polish-iters", type=int, default=2)
    ap.add_argument("--point-center-z-smooth", type=float, default=0.25)
    ap.add_argument("--point-center-z-ref", type=float, default=0.15)
    ap.add_argument("--point-reproj-steps", type=int, default=10)
    ap.add_argument("--point-path-polish-steps", type=int, default=0)
    ap.add_argument("--point-path-polish-lr", type=float, default=0.01)
    ap.add_argument("--point-lam-manifold", type=float, default=1.0)
    ap.add_argument("--point-lam-smooth", type=float, default=6.0)
    ap.add_argument("--point-lam-len", type=float, default=0.4)
    ap.add_argument("--point-lam-ref-init", type=float, default=1.5)
    ap.add_argument("--point-lam-ref-proj", type=float, default=1.0)
    ap.add_argument("--enforce-z-bounds", type=int, default=1, help="Clamp only TCP z dims [2,8] to task z bounds during traj_opt.")
    ap.add_argument("--plot-arms", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ckpt_path = _resolve_path(str(args.ckpt))
    outdir = _resolve_path(str(args.outdir))
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = _choose_device(str(args.device))
    model, ckpt = _load_constraint_model(
        ckpt_path,
        device,
        disable_feature_normalizer=bool(int(args.disable_feature_normalizer)),
    )
    cfg, x_train, grid = _build_data_pool(ckpt, seed=int(args.seed))
    fixed_pairs = _load_pairs_file(str(args.pairs_file), 12) if str(args.pairs_file).strip() else None
    plan_cfg = _planner_cfg(args, device, cfg)
    eval_cfg = resolve_eval_cfg(
        cfg,
        method_key=str(ckpt.get("method", "")),
        dataset_name=str(ckpt.get("dataset", DATASET_NAME)),
    )
    eval_cfg.device = device
    proj_eps_stop = float(compute_eps_stop(model, x_train, eval_cfg))

    paths: list[np.ndarray] = []
    init_paths: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    selected_endpoints: list[np.ndarray] = []
    if str(args.reference_mode) == "circle":
        pair_pool = _circle_pool(grid if len(grid) > 0 else x_train, x_train)
    else:
        pair_pool = _easy_pool(
            grid if len(grid) > 0 else x_train,
            x_train,
            center_keep_ratio=float(args.easy_center_keep_ratio),
        )
    pair_min_center_dist = float(args.pair_min_center_dist)
    pair_max_center_dist = float(args.pair_max_center_dist)
    if str(args.reference_mode) == "circle":
        pair_min_center_dist *= 0.88
        pair_max_center_dist *= 0.88

    n_plan = int(fixed_pairs[0].shape[0]) if fixed_pairs is not None else int(args.n_trajs)
    for k in range(n_plan):
        if fixed_pairs is not None:
            start = np.asarray(fixed_pairs[0][int(k)], dtype=np.float32).copy()
            goal = np.asarray(fixed_pairs[1][int(k)], dtype=np.float32).copy()
        else:
            start, goal = _pick_pair(
                pair_pool,
                rng,
                min_center_dist=pair_min_center_dist,
                max_center_dist=pair_max_center_dist,
                tries=int(args.pair_tries),
            )
        selected_endpoints.extend([start, goal])
        if str(args.init_mode) == "task":
            init = _task_init_path(start, goal, cfg, int(args.n_waypoints))
        elif str(args.init_mode) == "center":
            init = _center_init_path(start, goal, cfg, int(args.n_waypoints))
        elif str(args.init_mode) == "center_interp":
            init = _center_interp_init_path(start, goal, int(args.n_waypoints))
        elif str(args.init_mode) == "nearest_demo":
            init = _nearest_demo_init_path(
                start,
                goal,
                x_train,
                cfg,
                int(args.n_waypoints),
                center_interp_blend=float(args.nearest_demo_center_blend),
            )
        else:
            init = build_linear_path(start, goal, n_waypoints=int(args.n_waypoints), periodic=False)
        if str(args.reference_mode) == "circle":
            if int(args.roundtrip) == 1:
                init = _make_circle_roundtrip_path(init, bulge_ratio=float(args.circle_bulge_ratio))
            else:
                init = _apply_circle_reference(init, bulge_ratio=float(args.circle_bulge_ratio))
        elif int(args.roundtrip) == 1:
            init = _make_roundtrip_path(init)
        init = init.astype(np.float32)
        init_paths.append(init.copy())
        t0 = time.time()
        if str(args.planner_mode) == "init_only":
            path = init.copy()
        elif str(args.planner_mode) == "point_project":
            path = _point_project_path(
                model,
                init,
                x_start=start,
                x_goal=goal,
                task_cfg=cfg,
                device=device,
                proj_steps=int(args.proj_steps),
                proj_alpha=float(args.proj_alpha),
                proj_min_steps=int(args.proj_min_steps),
                f_abs_stop=proj_eps_stop,
                polish_iters=int(args.point_polish_iters),
                center_z_smooth=float(args.point_center_z_smooth),
                center_z_ref=float(args.point_center_z_ref),
                reproj_steps=int(args.point_reproj_steps),
                path_polish_steps=int(args.point_path_polish_steps),
                path_polish_lr=float(args.point_path_polish_lr),
                path_lam_manifold=float(args.point_lam_manifold),
                path_lam_smooth=float(args.point_lam_smooth),
                path_lam_len=float(args.point_lam_len),
                path_lam_ref_init=float(args.point_lam_ref_init),
                path_lam_ref_proj=float(args.point_lam_ref_proj),
            )
        else:
            path = plan_path(
                model=model,
                x_start=start,
                x_goal=goal,
                cfg=plan_cfg,
                planner_name="traj_opt",
                n_waypoints=int(args.n_waypoints),
                dataset_name=str(ckpt.get("dataset", DATASET_NAME)),
                periodic_joint=False,
                init_path=init,
                keep_endpoints=True,
            ).astype(np.float32)
            if bool(int(args.traj_resample_center_arclength)):
                path = _resample_path_center_arclength(path, int(args.n_waypoints))
            path[:, 3:6] = _wrap_pi(path[:, 3:6])
            path[:, 9:12] = _wrap_pi(path[:, 9:12])
        if int(args.roundtrip) == 1:
            if str(args.reference_mode) == "circle":
                path = _make_circle_roundtrip_path(path, bulge_ratio=float(args.circle_bulge_ratio))
            else:
                path = _make_roundtrip_path(path)
        elapsed = float(time.time() - t0)
        metrics = _evaluate_path(model, path, cfg, device)
        metrics["plan_seconds"] = elapsed
        metrics["index"] = int(k)
        metrics["start"] = start.astype(float).tolist()
        metrics["goal"] = goal.astype(float).tolist()
        metrics["start_center"] = _centers(start.reshape(1, -1))[0].tolist()
        metrics["goal_center"] = _centers(goal.reshape(1, -1))[0].tolist()
        summaries.append(metrics)
        paths.append(path)
        print(
            f"[path {k}] vec={metrics['analytic_vector_dist_mean']:.5f}, "
            f"Lpos={metrics['left_pos_err_mean']:.4f}, Rpos={metrics['right_pos_err_mean']:.4f}, "
            f"Lori={metrics['left_ori_err_deg_mean']:.2f}deg, Rori={metrics['right_ori_err_deg_mean']:.2f}deg, "
            f"f={metrics['model_residual_mean']:.5f}, time={elapsed:.2f}s"
        )

    npz_path = os.path.join(outdir, "dual_arm_learned_plans.npz")
    np.savez_compressed(npz_path, **{f"path_{i}": p for i, p in enumerate(paths)})
    plan_seconds = [float(p.get("plan_seconds", 0.0)) for p in summaries]
    summary = {
        "ckpt": ckpt_path,
        "dataset": str(ckpt.get("dataset", DATASET_NAME)),
        "method": str(ckpt.get("method", "")),
        "seed": int(args.seed),
        "n_trajs": int(len(paths)),
        "pairs_file": str(args.pairs_file),
        "n_waypoints": int(args.n_waypoints),
        "planner_mode": str(args.planner_mode),
        "traj_opt_objective": str(args.traj_opt_objective),
        "init_mode": str(args.init_mode),
        "reference_mode": str(args.reference_mode),
        "circle_bulge_ratio": float(args.circle_bulge_ratio),
        "nearest_demo_center_blend": float(args.nearest_demo_center_blend),
        "point_project": {
            "proj_steps": int(args.proj_steps),
            "proj_alpha": float(args.proj_alpha),
            "proj_min_steps": int(args.proj_min_steps),
            "polish_iters": int(args.point_polish_iters),
            "center_z_smooth": float(args.point_center_z_smooth),
            "center_z_ref": float(args.point_center_z_ref),
            "reproj_steps": int(args.point_reproj_steps),
            "path_polish_steps": int(args.point_path_polish_steps),
            "path_polish_lr": float(args.point_path_polish_lr),
            "lam_manifold": float(args.point_lam_manifold),
            "lam_smooth": float(args.point_lam_smooth),
            "lam_len": float(args.point_lam_len),
            "lam_ref_init": float(args.point_lam_ref_init),
            "lam_ref_proj": float(args.point_lam_ref_proj),
        },
        "pair_min_center_dist_used": float(pair_min_center_dist),
        "pair_max_center_dist_used": float(pair_max_center_dist),
        "roundtrip": int(args.roundtrip),
        "traj_resample_center_arclength": int(args.traj_resample_center_arclength),
        "mean_plan_seconds": float(np.mean(plan_seconds)) if plan_seconds else 0.0,
        "std_plan_seconds": float(np.std(plan_seconds, ddof=1)) if len(plan_seconds) > 1 else 0.0,
        "sem_plan_seconds": float(np.std(plan_seconds, ddof=1) / math.sqrt(len(plan_seconds))) if len(plan_seconds) > 1 else 0.0,
        "planner": plan_cfg.planner,
        "paths": summaries,
    }
    error_plot_path = os.path.join(outdir, "dual_arm_learned_planning_error_distributions.png")
    summary["error_distributions"] = _plot_error_distributions(
        model=model,
        init_paths=init_paths,
        planned_paths=paths,
        cfg=cfg,
        device=device,
        out_path=error_plot_path,
    )
    summary_path = os.path.join(outdir, "dual_arm_learned_planning_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_path = os.path.join(outdir, "dual_arm_learned_planning_paths.png")
    _plot_paths(paths, plot_path, cfg, plot_arms=bool(int(args.plot_arms)))
    print(f"[saved] {summary_path}")
    print(f"[saved] {npz_path}")
    print(f"[saved] {plot_path}")
    print(f"[saved] {error_plot_path}")


if __name__ == "__main__":
    main()
