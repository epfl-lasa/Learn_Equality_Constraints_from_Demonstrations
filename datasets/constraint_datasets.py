#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, Tuple

import numpy as np
import torch

LIFT_3D_VZ_DEFAULTS = {
    "z_amp1": 0.35,
    "z_amp2": 0.20,
    "z_freq1": 1.5,
    "z_freq2": 1.2,
}

TRAJ_3D_CODIM1_BASES = {
    "3d_paraboloid",
    "3d_twosphere",
    "3d_saddle_surface",
    "3d_sphere_surface",
    "3d_torus_surface",
    "3d_double_torus_surface",
    "3d_double_torus_extra_handle_surface",
    "3d_genus3_tube_surface",
    "3d_faceted_sphere_surface",
    "3d_spiky_sphere_surface",
    "3d_planar_arm_line_n3",
    "3d_spatial_arm_plane_n3",
    "3d_spatial_arm_ellip_n3",
}

_SINE_SURFACE_AFFINE_SCALE = 0.11
_SINE_SURFACE_AFFINE_OFFSET_XYZ = np.asarray([0.40, -0.22, 1.10], dtype=np.float32)


@dataclass
class DatasetSpec:
    name: str
    dim: int
    latent_dim_default: int
    train_on_sampler: Callable[[int, np.random.Generator], np.ndarray]
    eval_on_sampler: Callable[[int, np.random.Generator], np.ndarray]
    eval_off_sampler: Callable[[int, np.random.Generator], np.ndarray]
    gt_distance_fn: Callable[[np.ndarray], np.ndarray]
    plot_bounds: Tuple[np.ndarray, np.ndarray]


def get_lift_3d_vz_params(cfg=None) -> dict[str, float]:
    p = dict(LIFT_3D_VZ_DEFAULTS)
    if cfg is None:
        return p
    for k in ("z_amp1", "z_amp2", "z_freq1", "z_freq2"):
        if hasattr(cfg, k):
            p[k] = float(getattr(cfg, k))
    return p


def lift_xy_to_3d_var(x2: np.ndarray, cfg=None) -> np.ndarray:
    p = get_lift_3d_vz_params(cfg)
    x = x2[:, 0:1].astype(np.float32)
    y = x2[:, 1:2].astype(np.float32)
    z = (
        float(p["z_amp1"]) * np.sin(float(p["z_freq1"]) * x)
        + float(p["z_amp2"]) * np.cos(float(p["z_freq2"]) * y)
    ).astype(np.float32)
    return np.concatenate([x2.astype(np.float32), z], axis=1)


def lift_xy_to_3d_zero(x2: np.ndarray) -> np.ndarray:
    z = np.zeros((x2.shape[0], 1), dtype=np.float32)
    return np.concatenate([x2.astype(np.float32), z], axis=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_spiral_on(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 4.0 * np.pi, size=n)
    a = 0.25
    x = np.cos(theta)
    y = np.sin(theta)
    z = a * theta - 2.0
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_spiral_off(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 4.0 * np.pi, size=n)
    a = 0.25
    x = np.cos(theta)
    y = np.sin(theta)
    z = a * theta - 2.0
    x += rng.uniform(0.4, 1.0, size=n) * rng.choice([-1, 1], size=n)
    y += rng.uniform(0.4, 1.0, size=n) * rng.choice([-1, 1], size=n)
    z += rng.uniform(0.4, 1.0, size=n) * rng.choice([-1, 1], size=n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_sphere_on(
    n: int,
    rng: np.random.Generator,
    radius: float = 1.0,
    center=(0.0, 0.0, 0.0),
) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    pts = radius * v + np.array(center, dtype=float)[None, :]
    return pts.astype(np.float32)


def sample_paraboloid_on(
    n: int,
    rng: np.random.Generator,
    xy_range: float = 1.2,
    z_scale: float = 1.0,
) -> np.ndarray:
    x = rng.uniform(-xy_range, xy_range, size=n)
    y = rng.uniform(-xy_range, xy_range, size=n)
    z = z_scale * (x ** 2 + y ** 2)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_two_sphere_outer_on(n: int, rng: np.random.Generator) -> np.ndarray:
    ca = np.array([-0.8, 0.0, 0.0])
    cb = np.array([+0.8, 0.0, 0.0])
    r = 1.0
    m = int(n * 2.2) + 64
    pts_a = sample_sphere_on(m, rng, radius=r, center=ca)
    pts_b = sample_sphere_on(m, rng, radius=r, center=cb)
    # Keep only the boundary of union(A, B): remove sphere patches that lie
    # strictly inside the other sphere.
    db_a = np.linalg.norm(pts_a - cb[None, :], axis=1)
    da_b = np.linalg.norm(pts_b - ca[None, :], axis=1)
    keep_a = db_a >= (r - 1e-6)
    keep_b = da_b >= (r - 1e-6)
    pts = np.concatenate([pts_a[keep_a], pts_b[keep_b]], axis=0)
    if pts.shape[0] < n:
        return pts.astype(np.float32)
    idx = rng.choice(pts.shape[0], size=n, replace=False)
    return pts[idx].astype(np.float32)


def sample_square_on(n: int, rng: np.random.Generator, half: float = 1.0) -> np.ndarray:
    edge = rng.integers(0, 4, size=n)
    u = rng.uniform(-half, half, size=n)
    x = np.empty(n)
    y = np.empty(n)
    m = edge == 0
    x[m], y[m] = u[m], half
    m = edge == 1
    x[m], y[m] = u[m], -half
    m = edge == 2
    x[m], y[m] = half, u[m]
    m = edge == 3
    x[m], y[m] = -half, u[m]
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_paraboloid_off(
    n: int,
    rng: np.random.Generator,
    xy_range: float = 1.2,
    z_max: float = 3.0,
) -> np.ndarray:
    x = rng.uniform(-xy_range, xy_range, size=n)
    y = rng.uniform(-xy_range, xy_range, size=n)
    z_surface = x ** 2 + y ** 2
    sign = rng.choice([-1.0, 1.0], size=n)
    delta = rng.uniform(0.35, 1.0, size=n)
    z = np.clip(z_surface + sign * delta, 0.0, z_max)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_two_sphere_outer_off(n: int, rng: np.random.Generator, max_iters: int = 50) -> np.ndarray:
    ca = np.array([-0.8, 0.0, 0.0])
    cb = np.array([+0.8, 0.0, 0.0])
    r = 1.0
    lo = np.array([-2.5, -2.0, -2.0])
    hi = np.array([+2.5, +2.0, +2.0])
    out = []
    for _ in range(max_iters):
        if len(out) >= n:
            break
        m = int((n - len(out)) * 1.8) + 32
        pts = rng.uniform(lo, hi, size=(m, 3))
        da = np.linalg.norm(pts - ca[None, :], axis=1)
        db = np.linalg.norm(pts - cb[None, :], axis=1)
        dist_to_boundary = np.minimum(np.abs(da - r), np.abs(db - r))
        sel = pts[dist_to_boundary > 0.25]
        out.append(sel)
        out = [np.concatenate(out, axis=0)[:n]]
    if len(out) < n:
        missing = n - len(out)
        out.append(rng.uniform(lo, hi, size=(missing, 3)))
        out = [np.concatenate(out, axis=0)[:n]]
    return out[0].astype(np.float32)


def _rotation_y(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _rotation_x(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def _rotation_z(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _double_torus_params() -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    major_r = 1.0
    tube_r = 0.34
    center_sep = 1.02
    ca = np.asarray([-center_sep, 0.0, 0.0], dtype=np.float32)
    cb = np.asarray([center_sep, 0.12, 0.02], dtype=np.float32)
    frame_a = np.eye(3, dtype=np.float32)
    frame_b = (_rotation_z(math.radians(-8.0)) @ _rotation_y(math.radians(38.0))).astype(np.float32)
    return major_r, tube_r, ca, cb, frame_a, frame_b


def _torus_surface_from_uv(
    u: np.ndarray,
    v: np.ndarray,
    *,
    center: np.ndarray,
    frame: np.ndarray,
    major_r: float,
    tube_r: float,
) -> np.ndarray:
    uu = np.asarray(u, dtype=np.float64).reshape(-1)
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    local = np.stack(
        [
            (major_r + tube_r * np.cos(vv)) * np.cos(uu),
            (major_r + tube_r * np.cos(vv)) * np.sin(uu),
            tube_r * np.sin(vv),
        ],
        axis=1,
    )
    pts = local @ np.asarray(frame, dtype=np.float64).T + np.asarray(center, dtype=np.float64)[None, :]
    return pts.astype(np.float32)


def _torus_sdf(
    points: np.ndarray,
    *,
    center: np.ndarray,
    frame: np.ndarray,
    major_r: float,
    tube_r: float,
) -> np.ndarray:
    q_world = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :]
    q = q_world @ np.asarray(frame, dtype=np.float64)
    rho = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2)
    tube_dist = np.sqrt((rho - float(major_r)) ** 2 + q[:, 2] ** 2)
    return (tube_dist - float(tube_r)).astype(np.float32)


def _double_torus_union_sdf(points: np.ndarray) -> np.ndarray:
    major_r, tube_r, ca, cb, frame_a, frame_b = _double_torus_params()
    da = _torus_sdf(points, center=ca, frame=frame_a, major_r=major_r, tube_r=tube_r)
    db = _torus_sdf(points, center=cb, frame=frame_b, major_r=major_r, tube_r=tube_r)
    return np.minimum(da, db).astype(np.float32)


def _torus_centerline_point(*, u: float, center: np.ndarray, frame: np.ndarray, major_r: float) -> np.ndarray:
    local = np.asarray(
        [float(major_r) * math.cos(float(u)), float(major_r) * math.sin(float(u)), 0.0],
        dtype=np.float32,
    )
    return (local @ np.asarray(frame, dtype=np.float32).T + np.asarray(center, dtype=np.float32)).astype(np.float32)


def _cubic_bezier_points(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    *,
    n: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float32)[:, None]
    return ((1.0 - t) ** 3 * p0 + 3.0 * (1.0 - t) ** 2 * t * p1 + 3.0 * (1.0 - t) * t ** 2 * p2 + t ** 3 * p3).astype(np.float32)


def _extra_handle_bridge_curve(n: int = 96) -> np.ndarray:
    major_r, _tube_r, ca, cb, frame_a, frame_b = _double_torus_params()
    p0 = _torus_centerline_point(u=-0.55 * math.pi, center=ca, frame=frame_a, major_r=major_r)
    p3 = _torus_centerline_point(u=-0.42 * math.pi, center=cb, frame=frame_b, major_r=major_r)
    p1 = np.asarray([-0.82, -1.72, 0.68], dtype=np.float32)
    p2 = np.asarray([0.86, -1.68, -0.52], dtype=np.float32)
    return _cubic_bezier_points(p0, p1, p2, p3, n=int(n))


def _polyline_tube_sdf(points: np.ndarray, curve: np.ndarray, *, radius: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    seg_a = np.asarray(curve[:-1], dtype=np.float32)
    seg_b = np.asarray(curve[1:], dtype=np.float32)
    seg_ab = seg_b - seg_a
    seg_len2 = np.sum(seg_ab * seg_ab, axis=1).clip(1e-8)
    out = np.full((len(pts),), np.inf, dtype=np.float32)
    seg_chunk = 80
    for s0 in range(0, len(seg_a), seg_chunk):
        a = seg_a[s0 : s0 + seg_chunk]
        ab = seg_ab[s0 : s0 + seg_chunk]
        l2 = seg_len2[s0 : s0 + seg_chunk]
        ap = pts[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / l2[None, :], 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d2 = np.sum((pts[:, None, :] - closest) ** 2, axis=2)
        out = np.minimum(out, np.sqrt(np.min(d2, axis=1)).astype(np.float32))
    return (out - float(radius)).astype(np.float32)


def _double_torus_extra_handle_sdf(points: np.ndarray, *, bridge_radius: float = 0.18) -> np.ndarray:
    base = _double_torus_union_sdf(points)
    bridge = _polyline_tube_sdf(points, _extra_handle_bridge_curve(), radius=float(bridge_radius))
    return np.minimum(base, bridge).astype(np.float32)


def _double_torus_outer_from_surface_points(
    points: np.ndarray,
    *,
    other_center: np.ndarray,
    other_frame: np.ndarray,
) -> np.ndarray:
    major_r, tube_r, _ca, _cb, _frame_a, _frame_b = _double_torus_params()
    other_sdf = _torus_sdf(points, center=other_center, frame=other_frame, major_r=major_r, tube_r=tube_r)
    return points[other_sdf >= -1e-6].astype(np.float32)


def _sample_double_torus_outer_candidates(
    n_per_torus: int,
    rng: np.random.Generator,
) -> np.ndarray:
    major_r, tube_r, ca, cb, frame_a, frame_b = _double_torus_params()
    u = rng.uniform(0.0, 2.0 * np.pi, size=max(1, int(n_per_torus)))
    v = rng.uniform(0.0, 2.0 * np.pi, size=max(1, int(n_per_torus)))
    pa = _torus_surface_from_uv(u, v, center=ca, frame=frame_a, major_r=major_r, tube_r=tube_r)
    pb = _torus_surface_from_uv(u, v, center=cb, frame=frame_b, major_r=major_r, tube_r=tube_r)
    return np.concatenate(
        [
            _double_torus_outer_from_surface_points(pa, other_center=cb, other_frame=frame_b),
            _double_torus_outer_from_surface_points(pb, other_center=ca, other_frame=frame_a),
        ],
        axis=0,
    ).astype(np.float32)


def sample_double_torus_outer_on(n: int, rng: np.random.Generator, max_rounds: int = 8) -> np.ndarray:
    out: list[np.ndarray] = []
    want = max(1, int(n))
    for round_i in range(max_rounds):
        cur = np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)
        if len(cur) >= want:
            break
        missing = want - len(cur)
        cand = _sample_double_torus_outer_candidates(max(64, int(1.3 * missing)), rng)
        out.append(cand)
        if round_i == max_rounds - 1:
            cur = np.concatenate(out, axis=0)
    pts = np.concatenate(out, axis=0) if out else _sample_double_torus_outer_candidates(want, rng)
    if len(pts) >= want:
        idx = rng.choice(len(pts), size=want, replace=False)
        return pts[idx].astype(np.float32)
    idx = rng.choice(len(pts), size=want, replace=True)
    return pts[idx].astype(np.float32)


def sample_double_torus_outer_off(n: int, rng: np.random.Generator, max_iters: int = 50) -> np.ndarray:
    lo = np.asarray([-2.8, -1.8, -1.15], dtype=np.float32)
    hi = np.asarray([2.8, 1.8, 1.15], dtype=np.float32)
    out = []
    for _ in range(max_iters):
        cur_len = int(sum(len(a) for a in out))
        if cur_len >= int(n):
            break
        m = int((int(n) - cur_len) * 1.8) + 64
        pts = rng.uniform(lo, hi, size=(m, 3)).astype(np.float32)
        sel = pts[np.abs(_double_torus_union_sdf(pts)) > 0.16]
        out.append(sel)
    pts = np.concatenate(out, axis=0) if out else rng.uniform(lo, hi, size=(int(n), 3)).astype(np.float32)
    if len(pts) < int(n):
        missing = int(n) - len(pts)
        pts = np.concatenate([pts, rng.uniform(lo, hi, size=(missing, 3)).astype(np.float32)], axis=0)
    return pts[: int(n)].astype(np.float32)


def sample_square_off(n: int, rng: np.random.Generator, half: float = 1.0, max_iters: int = 50) -> np.ndarray:
    out = []
    for _ in range(max_iters):
        if len(out) >= n:
            break
        m = int((n - len(out)) * 1.6) + 16
        pts = rng.uniform(-1.8 * half, 1.8 * half, size=(m, 2))
        linf = np.maximum(np.abs(pts[:, 0]), np.abs(pts[:, 1]))
        sel = pts[np.abs(linf - half) > 0.18 * half]
        out.append(sel)
        out = [np.concatenate(out, axis=0)[:n]]
    if len(out) < n:
        missing = n - len(out)
        out.append(rng.uniform(-1.8 * half, 1.8 * half, size=(missing, 2)))
        out = [np.concatenate(out, axis=0)[:n]]
    return out[0].astype(np.float32)


def sample_two_horizontal_lines_on(
    n: int,
    rng: np.random.Generator,
    *,
    x_half_width: float = 2.0,
    y_abs: float = 1.0,
) -> np.ndarray:
    x = rng.uniform(-float(x_half_width), float(x_half_width), size=int(n))
    y = float(y_abs) * rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=int(n))
    return np.stack([x, y], axis=1).astype(np.float32)


def sample_two_horizontal_lines_off(
    n: int,
    rng: np.random.Generator,
    *,
    x_half_width: float = 2.0,
    y_abs: float = 1.0,
    max_iters: int = 50,
) -> np.ndarray:
    lo = np.asarray([-float(x_half_width), -1.8 * float(y_abs)], dtype=np.float32)
    hi = np.asarray([float(x_half_width), 1.8 * float(y_abs)], dtype=np.float32)
    out = []
    for _ in range(max_iters):
        cur_len = int(sum(len(a) for a in out))
        if cur_len >= int(n):
            break
        m = int((int(n) - cur_len) * 1.6) + 16
        pts = rng.uniform(lo, hi, size=(m, 2)).astype(np.float32)
        sel = pts[np.abs(np.abs(pts[:, 1]) - float(y_abs)) > 0.18 * float(y_abs)]
        out.append(sel)
    pts = np.concatenate(out, axis=0) if out else rng.uniform(lo, hi, size=(int(n), 2)).astype(np.float32)
    if len(pts) < int(n):
        missing = int(n) - len(pts)
        pts = np.concatenate([pts, rng.uniform(lo, hi, size=(missing, 2)).astype(np.float32)], axis=0)
    return pts[: int(n)].astype(np.float32)


def gt_dist_spiral(points: np.ndarray, theta_samples: int = 2048) -> np.ndarray:
    theta = np.linspace(0.0, 4.0 * np.pi, num=theta_samples, dtype=np.float32)
    a = 0.25
    curve = np.stack([np.cos(theta), np.sin(theta), a * theta - 2.0], axis=1).astype(np.float32)
    d2 = np.sum((points[:, None, :] - curve[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def gt_dist_paraboloid(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.abs(z - (x ** 2 + y ** 2))


def gt_dist_two_sphere_outer(points: np.ndarray) -> np.ndarray:
    ca = np.array([-0.8, 0.0, 0.0])
    cb = np.array([+0.8, 0.0, 0.0])
    r = 1.0
    da = np.linalg.norm(points - ca[None, :], axis=1)
    db = np.linalg.norm(points - cb[None, :], axis=1)
    return np.minimum(np.abs(da - r), np.abs(db - r))


def gt_dist_double_torus_outer(points: np.ndarray) -> np.ndarray:
    return np.abs(_double_torus_union_sdf(points)).astype(np.float32)


def _warped_double_torus_x4(points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    centers = np.asarray(
        [
            [-1.72, 0.42, 0.24],
            [-1.02, -0.62, -0.28],
            [-0.28, 0.78, 0.10],
            [0.34, -0.72, 0.42],
            [0.94, 0.52, -0.20],
            [1.58, -0.18, 0.28],
            [1.90, 0.70, -0.32],
            [0.02, 0.10, -0.54],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([0.24, -0.20, 0.16, -0.18, 0.22, -0.15, 0.12, -0.10], dtype=np.float64)
    sigma = 0.78
    d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    rbf = np.exp(-0.5 * d2 / (sigma ** 2))
    w = rbf @ weights
    # A small global tilt avoids flat regions while keeping the lift single-valued and topology-preserving.
    w += 0.045 * pts[:, 0] - 0.030 * pts[:, 1] + 0.025 * pts[:, 2]
    return w.astype(np.float32)


def lift_double_torus_outer_to_4d(points_xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points_xyz, dtype=np.float32)
    w = _warped_double_torus_x4(xyz).reshape(-1, 1)
    return np.concatenate([xyz, w], axis=1).astype(np.float32)


def lift_double_torus_extra_handle_to_4d(points_xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points_xyz, dtype=np.float32)
    w = _warped_double_torus_x4(xyz).reshape(-1, 1)
    return np.concatenate([xyz, w], axis=1).astype(np.float32)


_GENUS3_TUBE_MESH_CACHE: dict[tuple[int, int, int, float], tuple[np.ndarray, np.ndarray]] = {}
_DOUBLE_TORUS_EXTRA_HANDLE_MESH_CACHE: dict[tuple[int, int, int, float], tuple[np.ndarray, np.ndarray]] = {}
_FACETED_SPHERE_MESH_CACHE: dict[tuple[int, int, float, float, float], tuple[np.ndarray, np.ndarray]] = {}
_SPIKY_SPHERE_MESH_CACHE: dict[tuple[int, int, float, float, float, int], tuple[np.ndarray, np.ndarray]] = {}


def _double_torus_extra_handle_mesh(
    *,
    nx: int = 112,
    ny: int = 96,
    nz: int = 88,
    bridge_radius: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    key = (int(nx), int(ny), int(nz), round(float(bridge_radius), 6))
    cached = _DOUBLE_TORUS_EXTRA_HANDLE_MESH_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from skimage.measure import marching_cubes  # type: ignore
    except Exception as e:
        raise RuntimeError("3d_double_torus_extra_handle_surface requires scikit-image for marching_cubes") from e

    xv = np.linspace(-2.55, 2.55, int(nx), dtype=np.float32)
    yv = np.linspace(-2.05, 1.75, int(ny), dtype=np.float32)
    zv = np.linspace(-1.35, 1.35, int(nz), dtype=np.float32)
    xg, yg, zg = np.meshgrid(xv, yv, zv, indexing="ij")
    pts = np.column_stack([xg.reshape(-1), yg.reshape(-1), zg.reshape(-1)]).astype(np.float32)

    field = np.empty((len(pts),), dtype=np.float32)
    chunk = 50000
    for start in range(0, len(pts), chunk):
        p = pts[start : start + chunk]
        field[start : start + chunk] = _double_torus_extra_handle_sdf(p, bridge_radius=float(bridge_radius))
    field = field.reshape((int(nx), int(ny), int(nz)))
    spacing = (float(xv[1] - xv[0]), float(yv[1] - yv[0]), float(zv[1] - zv[0]))
    verts, faces, _normals, _values = marching_cubes(field, level=0.0, spacing=spacing)
    verts[:, 0] += float(xv[0])
    verts[:, 1] += float(yv[0])
    verts[:, 2] += float(zv[0])
    out = (verts.astype(np.float32), faces.astype(np.int64))
    _DOUBLE_TORUS_EXTRA_HANDLE_MESH_CACHE[key] = out
    return out


def _unit_vec(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return (a / max(float(np.linalg.norm(a)), 1e-8)).astype(np.float32)


def _faceted_sphere_control_directions() -> np.ndarray:
    dirs = np.asarray(
        [
            [1.00, 0.00, 0.00],
            [-1.00, 0.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, -1.00, 0.00],
            [0.00, 0.00, 1.00],
            [0.00, 0.00, -1.00],
            [0.78, 0.42, 0.46],
            [-0.64, 0.70, 0.32],
            [0.35, -0.86, 0.38],
            [-0.82, -0.31, -0.48],
            [0.52, 0.64, -0.57],
            [-0.28, 0.38, -0.88],
            [0.91, -0.18, -0.36],
            [-0.45, -0.76, 0.47],
            [0.15, 0.96, -0.24],
            [-0.94, 0.12, 0.30],
        ],
        dtype=np.float32,
    )
    return (dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def _faceted_sphere_control_radii(*, bump_scale: float = 0.72) -> np.ndarray:
    offsets = np.asarray(
        [0.50, -0.32, 0.34, -0.42, 0.44, -0.30, 0.26, -0.46, -0.74, -0.35, 0.24, -0.40, -0.68, -0.28, 0.36, -0.34],
        dtype=np.float32,
    )
    return (1.0 + float(bump_scale) * offsets).astype(np.float32)


def faceted_sphere_control_points(*, bump_scale: float = 0.72) -> np.ndarray:
    dirs = _faceted_sphere_control_directions()
    radii = _faceted_sphere_control_radii(bump_scale=float(bump_scale))
    return (dirs * radii[:, None]).astype(np.float32)


def _faceted_sphere_radius(
    directions: np.ndarray,
    *,
    bump_scale: float = 0.72,
    sigma: float = 0.20,
    sharpness: float = 3.2,
) -> np.ndarray:
    d = np.asarray(directions, dtype=np.float32)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
    ctrl_dirs = _faceted_sphere_control_directions()
    ctrl_r = _faceted_sphere_control_radii(bump_scale=float(bump_scale))
    dots = np.clip(d @ ctrl_dirs.T, -1.0, 1.0)
    ang = np.arccos(dots)
    base_w = np.exp(-0.5 * (ang / max(float(sigma), 1e-6)) ** 2).astype(np.float32)
    w = np.power(base_w, max(float(sharpness), 1.0)).astype(np.float32)
    # Keep only local influences so neighboring control regions meet through
    # sharper triangular patches rather than one globally smooth radial blend.
    top_k = min(4, int(w.shape[1]))
    cutoff = np.partition(w, -top_k, axis=1)[:, -top_k][:, None]
    w = np.where(w >= cutoff, w, 0.0).astype(np.float32)
    nearest = np.max(base_w, axis=1)
    smooth = (w @ ctrl_r) / np.maximum(np.sum(w, axis=1), 1e-8)
    hard_idx = np.argmax(base_w, axis=1)
    hard = ctrl_r[hard_idx]
    blend = np.clip(0.56 + 0.28 * nearest, 0.56, 0.84).astype(np.float32)
    return ((1.0 - blend) * smooth + blend * hard).astype(np.float32)


def _faceted_sphere_mesh(
    *,
    n_lat: int = 18,
    n_lon: int = 36,
    bump_scale: float = 0.72,
    sigma: float = 0.20,
    sharpness: float = 3.2,
) -> tuple[np.ndarray, np.ndarray]:
    key = (int(n_lat), int(n_lon), round(float(bump_scale), 6), round(float(sigma), 6), round(float(sharpness), 6))
    cached = _FACETED_SPHERE_MESH_CACHE.get(key)
    if cached is not None:
        return cached

    n_lat = max(4, int(n_lat))
    n_lon = max(8, int(n_lon))
    dirs: list[np.ndarray] = [np.asarray([0.0, 0.0, 1.0], dtype=np.float32)]
    for i in range(1, n_lat):
        phi = math.pi * float(i) / float(n_lat)
        theta_offset = (0.5 * (i % 2)) * (2.0 * math.pi / float(n_lon))
        for j in range(n_lon):
            theta = 2.0 * math.pi * float(j) / float(n_lon) + theta_offset
            dirs.append(np.asarray([math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)], dtype=np.float32))
    dirs.append(np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    dirs_arr = np.asarray(dirs, dtype=np.float32)
    radii = _faceted_sphere_radius(
        dirs_arr,
        bump_scale=float(bump_scale),
        sigma=float(sigma),
        sharpness=float(sharpness),
    )
    verts = (dirs_arr * radii[:, None]).astype(np.float32)

    north = 0
    south = len(verts) - 1

    def vid(ring: int, col: int) -> int:
        return 1 + (ring - 1) * n_lon + (col % n_lon)

    faces: list[list[int]] = []
    for j in range(n_lon):
        faces.append([north, vid(1, j), vid(1, j + 1)])
    for ring in range(1, n_lat - 1):
        for j in range(n_lon):
            a = vid(ring, j)
            b = vid(ring, j + 1)
            c = vid(ring + 1, j)
            d = vid(ring + 1, j + 1)
            faces.append([a, c, b])
            faces.append([b, c, d])
    last_ring = n_lat - 1
    for j in range(n_lon):
        faces.append([vid(last_ring, j), south, vid(last_ring, j + 1)])
    out = (verts.astype(np.float32), np.asarray(faces, dtype=np.int64))
    _FACETED_SPHERE_MESH_CACHE[key] = out
    return out


def gt_dist_faceted_sphere_surface(
    points: np.ndarray,
    *,
    bump_scale: float = 0.72,
    sigma: float = 0.20,
    sharpness: float = 3.2,
) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    norm = np.linalg.norm(p, axis=1)
    dirs = p / np.maximum(norm[:, None], 1e-8)
    r = _faceted_sphere_radius(dirs, bump_scale=float(bump_scale), sigma=float(sigma), sharpness=float(sharpness))
    return np.abs(norm - r).astype(np.float32)


def _spiky_sphere_control_directions(n_ctrl: int = 58) -> np.ndarray:
    n_ctrl = max(18, int(n_ctrl))
    dirs: list[np.ndarray] = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
    ]
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_ctrl - len(dirs)):
        y = 1.0 - 2.0 * (float(i) + 0.5) / float(max(1, n_ctrl - len(dirs)))
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * float(i)
        dirs.append(np.asarray([math.cos(theta) * r, y, math.sin(theta) * r], dtype=np.float32))
    arr = np.asarray(dirs, dtype=np.float32)
    return (arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def _spiky_sphere_control_radii(*, spike_scale: float = 0.86, n_ctrl: int = 58) -> np.ndarray:
    n_ctrl = max(18, int(n_ctrl))
    i = np.arange(n_ctrl, dtype=np.float32)
    offsets = 0.18 * np.sin(1.71 * i + 0.25) + 0.14 * np.cos(3.83 * i - 0.40)
    spike_idx = np.asarray([6, 9, 13, 17, 21, 26, 30, 35, 39, 44, 48, 53], dtype=np.int64)
    pit_idx = np.asarray([7, 11, 15, 23, 28, 33, 41, 46, 51, 56], dtype=np.int64)
    spike_idx = spike_idx[spike_idx < n_ctrl]
    pit_idx = pit_idx[pit_idx < n_ctrl]
    offsets[spike_idx] += np.linspace(0.88, 1.18, len(spike_idx), dtype=np.float32)
    offsets[pit_idx] -= np.linspace(0.58, 0.78, len(pit_idx), dtype=np.float32)
    for idx, val in zip([0, 4, 12, 25], [0.95, 0.80, 0.72, 0.90]):
        if idx < n_ctrl:
            offsets[idx] += np.float32(val)
    for idx, val in zip([3, 8, 19, 37], [0.60, 0.72, 0.64, 0.68]):
        if idx < n_ctrl:
            offsets[idx] -= np.float32(val)
    offsets = np.clip(offsets, -0.82, 1.22)
    return np.clip(1.0 + float(spike_scale) * offsets, 0.22, 2.25).astype(np.float32)


def spiky_sphere_control_points(*, spike_scale: float = 0.86, n_ctrl: int = 58) -> np.ndarray:
    dirs = _spiky_sphere_control_directions(n_ctrl=int(n_ctrl))
    radii = _spiky_sphere_control_radii(spike_scale=float(spike_scale), n_ctrl=int(n_ctrl))
    return (dirs * radii[:, None]).astype(np.float32)


def _spiky_sphere_radius(
    directions: np.ndarray,
    *,
    spike_scale: float = 0.86,
    sigma: float = 0.105,
    sharpness: float = 8.0,
    n_ctrl: int = 58,
) -> np.ndarray:
    d = np.asarray(directions, dtype=np.float32)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
    ctrl_dirs = _spiky_sphere_control_directions(n_ctrl=int(n_ctrl))
    ctrl_r = _spiky_sphere_control_radii(spike_scale=float(spike_scale), n_ctrl=int(n_ctrl))
    dots = np.clip(d @ ctrl_dirs.T, -1.0, 1.0)
    ang = np.arccos(dots)
    base_w = np.exp(-0.5 * (ang / max(float(sigma), 1e-6)) ** 2).astype(np.float32)
    w = np.power(base_w, max(float(sharpness), 1.0)).astype(np.float32)
    top_k = min(3, int(w.shape[1]))
    cutoff = np.partition(w, -top_k, axis=1)[:, -top_k][:, None]
    w = np.where(w >= cutoff, w, 0.0).astype(np.float32)
    smooth = (w @ ctrl_r) / np.maximum(np.sum(w, axis=1), 1e-8)
    hard = ctrl_r[np.argmax(base_w, axis=1)]
    nearest = np.max(base_w, axis=1)
    blend = np.clip(0.76 + 0.22 * nearest, 0.76, 0.98).astype(np.float32)
    return ((1.0 - blend) * smooth + blend * hard).astype(np.float32)


def _spiky_sphere_mesh(
    *,
    n_lat: int = 24,
    n_lon: int = 48,
    spike_scale: float = 0.86,
    sigma: float = 0.105,
    sharpness: float = 8.0,
    n_ctrl: int = 58,
) -> tuple[np.ndarray, np.ndarray]:
    key = (
        int(n_lat),
        int(n_lon),
        round(float(spike_scale), 6),
        round(float(sigma), 6),
        round(float(sharpness), 6),
        int(n_ctrl),
    )
    cached = _SPIKY_SPHERE_MESH_CACHE.get(key)
    if cached is not None:
        return cached

    n_lat = max(4, int(n_lat))
    n_lon = max(8, int(n_lon))
    dirs: list[np.ndarray] = [np.asarray([0.0, 0.0, 1.0], dtype=np.float32)]
    for row in range(1, n_lat):
        phi = math.pi * float(row) / float(n_lat)
        theta_offset = (0.5 * (row % 2)) * (2.0 * math.pi / float(n_lon))
        for col in range(n_lon):
            theta = 2.0 * math.pi * float(col) / float(n_lon) + theta_offset
            dirs.append(np.asarray([math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)], dtype=np.float32))
    dirs.append(np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    dirs_arr = np.asarray(dirs, dtype=np.float32)
    radii = _spiky_sphere_radius(
        dirs_arr,
        spike_scale=float(spike_scale),
        sigma=float(sigma),
        sharpness=float(sharpness),
        n_ctrl=int(n_ctrl),
    )
    verts = (dirs_arr * radii[:, None]).astype(np.float32)

    north = 0
    south = len(verts) - 1

    def vid(ring: int, col: int) -> int:
        return 1 + (ring - 1) * n_lon + (col % n_lon)

    faces: list[list[int]] = []
    for col in range(n_lon):
        faces.append([north, vid(1, col), vid(1, col + 1)])
    for ring in range(1, n_lat - 1):
        for col in range(n_lon):
            a = vid(ring, col)
            b = vid(ring, col + 1)
            c = vid(ring + 1, col)
            d = vid(ring + 1, col + 1)
            faces.append([a, c, b])
            faces.append([b, c, d])
    last_ring = n_lat - 1
    for col in range(n_lon):
        faces.append([vid(last_ring, col), south, vid(last_ring, col + 1)])
    out = (verts.astype(np.float32), np.asarray(faces, dtype=np.int64))
    _SPIKY_SPHERE_MESH_CACHE[key] = out
    return out


def gt_dist_spiky_sphere_surface(
    points: np.ndarray,
    *,
    spike_scale: float = 0.86,
    sigma: float = 0.105,
    sharpness: float = 8.0,
    n_ctrl: int = 58,
) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    norm = np.linalg.norm(p, axis=1)
    dirs = p / np.maximum(norm[:, None], 1e-8)
    r = _spiky_sphere_radius(
        dirs,
        spike_scale=float(spike_scale),
        sigma=float(sigma),
        sharpness=float(sharpness),
        n_ctrl=int(n_ctrl),
    )
    return np.abs(norm - r).astype(np.float32)


def _genus3_circle_curve(
    center: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    radius: float,
    n: int,
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, int(n), endpoint=False, dtype=np.float32)
    return (
        center[None, :]
        + float(radius) * (np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v[None, :])
    ).astype(np.float32)


def _genus3_bezier_curve(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, *, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float32)[:, None]
    return ((1.0 - t) ** 3 * p0 + 3.0 * (1.0 - t) ** 2 * t * p1 + 3.0 * (1.0 - t) * t ** 2 * p2 + t ** 3 * p3).astype(np.float32)


def _genus3_tube_skeleton_curves() -> list[np.ndarray]:
    c1 = np.asarray([-1.35, -0.05, 0.00], dtype=np.float32)
    c2 = np.asarray([0.00, 0.20, 0.25], dtype=np.float32)
    c3 = np.asarray([1.35, -0.08, 0.03], dtype=np.float32)
    loop1 = _genus3_circle_curve(c1, _unit_vec([1.0, 0.0, 0.0]), _unit_vec([0.0, 0.0, 1.0]), radius=0.56, n=180)
    loop2 = _genus3_circle_curve(c2, _unit_vec([1.0, 0.0, 0.0]), _unit_vec([0.0, 1.0, 0.0]), radius=0.58, n=180)
    loop3 = _genus3_circle_curve(
        c3,
        _unit_vec([0.86, 0.18, 0.47]),
        _unit_vec([-0.10, 0.98, -0.14]),
        radius=0.56,
        n=180,
    )
    bridge12 = _genus3_bezier_curve(
        c1 + np.asarray([0.56, 0.0, 0.0], dtype=np.float32),
        np.asarray([-0.78, 0.34, 0.18], dtype=np.float32),
        np.asarray([-0.50, 0.48, 0.27], dtype=np.float32),
        c2 + np.asarray([-0.58, 0.0, 0.0], dtype=np.float32),
        n=85,
    )
    bridge23 = _genus3_bezier_curve(
        c2 + np.asarray([0.58, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.50, 0.42, 0.25], dtype=np.float32),
        np.asarray([0.80, 0.24, 0.08], dtype=np.float32),
        c3 + np.asarray([-0.50, -0.10, -0.15], dtype=np.float32),
        n=85,
    )
    return [loop1, loop2, loop3, bridge12, bridge23]


def _genus3_tube_segments() -> np.ndarray:
    pieces: list[np.ndarray] = []
    for i, curve in enumerate(_genus3_tube_skeleton_curves()):
        cc = np.vstack([curve, curve[:1]]) if i < 3 else curve
        pieces.append(np.stack([cc[:-1], cc[1:]], axis=1))
    return np.concatenate(pieces, axis=0).astype(np.float32)


def _genus3_tube_mesh(
    *,
    nx: int = 112,
    ny: int = 88,
    nz: int = 88,
    radius: float = 0.165,
) -> tuple[np.ndarray, np.ndarray]:
    key = (int(nx), int(ny), int(nz), round(float(radius), 6))
    cached = _GENUS3_TUBE_MESH_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from skimage.measure import marching_cubes  # type: ignore
    except Exception as e:
        raise RuntimeError("3d_genus3_tube_surface requires scikit-image for marching_cubes") from e

    segments = _genus3_tube_segments()
    seg_a = segments[:, 0]
    seg_b = segments[:, 1]
    seg_ab = seg_b - seg_a
    seg_len2 = np.sum(seg_ab * seg_ab, axis=1).clip(1e-8)

    xv = np.linspace(-2.05, 2.05, int(nx), dtype=np.float32)
    yv = np.linspace(-1.05, 1.05, int(ny), dtype=np.float32)
    zv = np.linspace(-0.95, 1.05, int(nz), dtype=np.float32)
    xg, yg, zg = np.meshgrid(xv, yv, zv, indexing="ij")
    pts = np.column_stack([xg.reshape(-1), yg.reshape(-1), zg.reshape(-1)]).astype(np.float32)

    min_d2 = np.full((len(pts),), np.inf, dtype=np.float32)
    chunk = 16000
    seg_chunk = 75
    for start in range(0, len(pts), chunk):
        p = pts[start : start + chunk]
        local = np.full((len(p),), np.inf, dtype=np.float32)
        for s0 in range(0, len(seg_a), seg_chunk):
            a = seg_a[s0 : s0 + seg_chunk]
            ab = seg_ab[s0 : s0 + seg_chunk]
            l2 = seg_len2[s0 : s0 + seg_chunk]
            ap = p[:, None, :] - a[None, :, :]
            t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / l2[None, :], 0.0, 1.0)
            closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
            d2 = np.sum((p[:, None, :] - closest) ** 2, axis=2)
            local = np.minimum(local, np.min(d2, axis=1))
        min_d2[start : start + chunk] = local

    field = np.sqrt(min_d2).reshape((int(nx), int(ny), int(nz))) - float(radius)
    spacing = (float(xv[1] - xv[0]), float(yv[1] - yv[0]), float(zv[1] - zv[0]))
    verts, faces, _normals, _values = marching_cubes(field, level=0.0, spacing=spacing)
    verts[:, 0] += float(xv[0])
    verts[:, 1] += float(yv[0])
    verts[:, 2] += float(zv[0])
    out = (verts.astype(np.float32), faces.astype(np.int64))
    _GENUS3_TUBE_MESH_CACHE[key] = out
    return out


def _sample_mesh_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    tri = np.asarray(verts, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    prob = area / max(float(np.sum(area)), 1e-12)
    idx = rng.choice(len(tri), size=int(n), replace=True, p=prob)
    chosen = tri[idx]
    r1 = rng.random((int(n), 1), dtype=np.float32)
    r2 = rng.random((int(n), 1), dtype=np.float32)
    sr1 = np.sqrt(r1)
    pts = (1.0 - sr1) * chosen[:, 0] + sr1 * (1.0 - r2) * chosen[:, 1] + sr1 * r2 * chosen[:, 2]
    return pts.astype(np.float32)


def sample_double_torus_extra_handle_on(n: int, rng: np.random.Generator) -> np.ndarray:
    verts, faces = _double_torus_extra_handle_mesh()
    return _sample_mesh_surface(verts, faces, int(n), rng).astype(np.float32)


def sample_double_torus_extra_handle_off(n: int, rng: np.random.Generator, max_iters: int = 50) -> np.ndarray:
    lo = np.asarray([-2.65, -2.15, -1.45], dtype=np.float32)
    hi = np.asarray([2.65, 1.85, 1.45], dtype=np.float32)
    out = []
    for _ in range(max_iters):
        cur_len = int(sum(len(a) for a in out))
        if cur_len >= int(n):
            break
        m = int((int(n) - cur_len) * 1.8) + 64
        pts = rng.uniform(lo, hi, size=(m, 3)).astype(np.float32)
        sel = pts[np.abs(_double_torus_extra_handle_sdf(pts)) > 0.16]
        out.append(sel)
    pts = np.concatenate(out, axis=0) if out else rng.uniform(lo, hi, size=(int(n), 3)).astype(np.float32)
    if len(pts) < int(n):
        missing = int(n) - len(pts)
        pts = np.concatenate([pts, rng.uniform(lo, hi, size=(missing, 3)).astype(np.float32)], axis=0)
    return pts[: int(n)].astype(np.float32)


def sample_genus3_tube_surface_on(n: int, rng: np.random.Generator) -> np.ndarray:
    verts, faces = _genus3_tube_mesh()
    return _sample_mesh_surface(verts, faces, int(n), rng).astype(np.float32)


def sample_faceted_sphere_surface_on(n: int, rng: np.random.Generator) -> np.ndarray:
    verts, faces = _faceted_sphere_mesh()
    return _sample_mesh_surface(verts, faces, int(n), rng).astype(np.float32)


def sample_faceted_sphere_surface_off(n: int, rng: np.random.Generator, max_iters: int = 50) -> np.ndarray:
    lo = np.asarray([-2.0, -2.0, -2.0], dtype=np.float32)
    hi = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
    out = []
    for _ in range(max_iters):
        cur_len = int(sum(len(a) for a in out))
        if cur_len >= int(n):
            break
        m = int((int(n) - cur_len) * 1.8) + 64
        pts = rng.uniform(lo, hi, size=(m, 3)).astype(np.float32)
        sel = pts[gt_dist_faceted_sphere_surface(pts) > 0.14]
        out.append(sel)
    pts = np.concatenate(out, axis=0) if out else rng.uniform(lo, hi, size=(int(n), 3)).astype(np.float32)
    if len(pts) < int(n):
        missing = int(n) - len(pts)
        pts = np.concatenate([pts, rng.uniform(lo, hi, size=(missing, 3)).astype(np.float32)], axis=0)
    return pts[: int(n)].astype(np.float32)


def sample_spiky_sphere_surface_on(n: int, rng: np.random.Generator) -> np.ndarray:
    verts, faces = _spiky_sphere_mesh()
    return _sample_mesh_surface(verts, faces, int(n), rng).astype(np.float32)


def sample_spiky_sphere_surface_off(n: int, rng: np.random.Generator, max_iters: int = 50) -> np.ndarray:
    lo = np.asarray([-2.35, -2.35, -2.35], dtype=np.float32)
    hi = np.asarray([2.35, 2.35, 2.35], dtype=np.float32)
    out = []
    for _ in range(max_iters):
        cur_len = int(sum(len(a) for a in out))
        if cur_len >= int(n):
            break
        m = int((int(n) - cur_len) * 1.8) + 64
        pts = rng.uniform(lo, hi, size=(m, 3)).astype(np.float32)
        sel = pts[gt_dist_spiky_sphere_surface(pts) > 0.16]
        out.append(sel)
    pts = np.concatenate(out, axis=0) if out else rng.uniform(lo, hi, size=(int(n), 3)).astype(np.float32)
    if len(pts) < int(n):
        missing = int(n) - len(pts)
        pts = np.concatenate([pts, rng.uniform(lo, hi, size=(missing, 3)).astype(np.float32)], axis=0)
    return pts[: int(n)].astype(np.float32)


def gt_dist_square(points: np.ndarray, half: float = 1.0) -> np.ndarray:
    linf = np.maximum(np.abs(points[:, 0]), np.abs(points[:, 1]))
    return np.abs(linf - half)


def gt_dist_two_horizontal_lines(points: np.ndarray, y_abs: float = 1.0) -> np.ndarray:
    return np.abs(np.abs(np.asarray(points)[:, 1]) - float(y_abs)).astype(np.float32)


def build_datasets() -> Dict[str, DatasetSpec]:
    ds: Dict[str, DatasetSpec] = {}
    ds["3d_spiral"] = DatasetSpec(
        name="3d_spiral",
        dim=3,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_spiral_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_spiral_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_spiral_off(n, rng),
        gt_distance_fn=lambda x: gt_dist_spiral(x),
        plot_bounds=(np.array([-2.2, -2.2, -2.8]), np.array([2.2, 2.2, 2.8])),
    )
    ds["3d_paraboloid"] = DatasetSpec(
        name="3d_paraboloid",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_paraboloid_on(n, rng, xy_range=1.2, z_scale=1.0),
        eval_on_sampler=lambda n, rng: sample_paraboloid_on(n, rng, xy_range=1.2, z_scale=1.0),
        eval_off_sampler=lambda n, rng: sample_paraboloid_off(n, rng, xy_range=1.2, z_max=3.0),
        gt_distance_fn=lambda x: gt_dist_paraboloid(x),
        plot_bounds=(np.array([-1.8, -1.8, 0.0]), np.array([1.8, 1.8, 3.2])),
    )
    ds["3d_twosphere"] = DatasetSpec(
        name="3d_twosphere",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_two_sphere_outer_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_two_sphere_outer_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_two_sphere_outer_off(n, rng),
        gt_distance_fn=lambda x: gt_dist_two_sphere_outer(x),
        plot_bounds=(np.array([-2.8, -2.2, -2.2]), np.array([2.8, 2.2, 2.2])),
    )
    ds["3d_double_torus_surface"] = DatasetSpec(
        name="3d_double_torus_surface",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_double_torus_outer_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_double_torus_outer_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_double_torus_outer_off(n, rng),
        gt_distance_fn=lambda x: gt_dist_double_torus_outer(x),
        plot_bounds=(np.array([-2.8, -1.8, -1.15]), np.array([2.8, 1.8, 1.15])),
    )
    ds["3d_double_torus_extra_handle_surface"] = DatasetSpec(
        name="3d_double_torus_extra_handle_surface",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_double_torus_extra_handle_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_double_torus_extra_handle_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_double_torus_extra_handle_off(n, rng),
        gt_distance_fn=lambda x: np.abs(_double_torus_extra_handle_sdf(x)).astype(np.float32),
        plot_bounds=(np.array([-2.65, -2.15, -1.45]), np.array([2.65, 1.85, 1.45])),
    )
    ds["3d_genus3_tube_surface"] = DatasetSpec(
        name="3d_genus3_tube_surface",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_genus3_tube_surface_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_genus3_tube_surface_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_double_torus_outer_off(n, rng),
        gt_distance_fn=lambda x: np.zeros((len(x),), dtype=np.float32),
        plot_bounds=(np.array([-2.1, -1.1, -1.0]), np.array([2.1, 1.1, 1.1])),
    )
    ds["3d_faceted_sphere_surface"] = DatasetSpec(
        name="3d_faceted_sphere_surface",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_faceted_sphere_surface_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_faceted_sphere_surface_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_faceted_sphere_surface_off(n, rng),
        gt_distance_fn=lambda x: gt_dist_faceted_sphere_surface(x),
        plot_bounds=(np.array([-1.65, -1.65, -1.65]), np.array([1.65, 1.65, 1.65])),
    )
    ds["3d_spiky_sphere_surface"] = DatasetSpec(
        name="3d_spiky_sphere_surface",
        dim=3,
        latent_dim_default=2,
        train_on_sampler=lambda n, rng: sample_spiky_sphere_surface_on(n, rng),
        eval_on_sampler=lambda n, rng: sample_spiky_sphere_surface_on(n, rng),
        eval_off_sampler=lambda n, rng: sample_spiky_sphere_surface_off(n, rng),
        gt_distance_fn=lambda x: gt_dist_spiky_sphere_surface(x),
        plot_bounds=(np.array([-2.25, -2.25, -2.25]), np.array([2.25, 2.25, 2.25])),
    )
    ds["2d_square"] = DatasetSpec(
        name="2d_square",
        dim=2,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_square_on(n, rng, half=1.0),
        eval_on_sampler=lambda n, rng: sample_square_on(n, rng, half=1.0),
        eval_off_sampler=lambda n, rng: sample_square_off(n, rng, half=1.0),
        gt_distance_fn=lambda x: gt_dist_square(x, half=1.0),
        plot_bounds=(np.array([-2.3, -2.3]), np.array([2.3, 2.3])),
    )
    ds["2d_two_horizontal_lines"] = DatasetSpec(
        name="2d_two_horizontal_lines",
        dim=2,
        latent_dim_default=1,
        train_on_sampler=lambda n, rng: sample_two_horizontal_lines_on(n, rng, x_half_width=2.0, y_abs=1.0),
        eval_on_sampler=lambda n, rng: sample_two_horizontal_lines_on(n, rng, x_half_width=2.0, y_abs=1.0),
        eval_off_sampler=lambda n, rng: sample_two_horizontal_lines_off(n, rng, x_half_width=2.0, y_abs=1.0),
        gt_distance_fn=lambda x: gt_dist_two_horizontal_lines(x, y_abs=1.0),
        plot_bounds=(np.array([-2.4, -1.8]), np.array([2.4, 1.8])),
    )
    return ds


def make_sine_manifold(n: int) -> np.ndarray:
    t = np.random.uniform(-math.pi, math.pi, size=(n, 1))
    x = np.concatenate([t, np.sin(t)], axis=1)
    return x.astype(np.float32)


def make_sine_grid(n: int) -> np.ndarray:
    t = np.linspace(-math.pi, math.pi, n).reshape(-1, 1)
    x = np.concatenate([t, np.sin(t)], axis=1)
    return x.astype(np.float32)


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return ((a + math.pi) % (2.0 * math.pi) - math.pi).astype(np.float32)


def _sample_rows(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == 0:
        return arr
    if len(arr) >= n:
        idx = np.random.choice(len(arr), size=n, replace=False)
    else:
        idx = np.random.choice(len(arr), size=n, replace=True)
    return arr[idx].astype(np.float32)


def _planar_arm_line_n2(cfg) -> Tuple[np.ndarray, np.ndarray]:
    l1, l2 = 1.0, 0.8
    y_line = 0.3
    t1 = np.linspace(-math.pi, math.pi, max(4 * cfg.n_grid, 4096), dtype=np.float32)
    rhs = (y_line - l1 * np.sin(t1)) / l2
    valid = np.abs(rhs) <= 1.0
    t1v = t1[valid]
    a = np.arcsin(rhs[valid]).astype(np.float32)
    t2a = _wrap_to_pi(a - t1v)
    t2b = _wrap_to_pi((math.pi - a) - t1v)
    x_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(t1v), t2a], axis=1),
            np.stack([_wrap_to_pi(t1v), t2b], axis=1),
        ],
        axis=0,
    ).astype(np.float32)
    x_train = _sample_rows(x_all, max(1, cfg.n_train))
    grid = _sample_rows(x_all, max(1, cfg.n_grid))
    return x_train, grid


def _planar_arm_line_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    l1, l2, l3 = 1.0, 0.8, 0.6
    y_line = 0.35

    n_cand = max(8 * cfg.n_train, 12000)
    t1 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)
    t2 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)
    rhs = (y_line - l1 * np.sin(t1) - l2 * np.sin(t1 + t2)) / l3
    valid = np.abs(rhs) <= 1.0
    t1v = t1[valid]
    t2v = t2[valid]
    a = np.arcsin(rhs[valid]).astype(np.float32)
    t12 = t1v + t2v
    t3a = _wrap_to_pi(a - t12)
    t3b = _wrap_to_pi((math.pi - a) - t12)
    x_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(t1v), _wrap_to_pi(t2v), t3a], axis=1),
            np.stack([_wrap_to_pi(t1v), _wrap_to_pi(t2v), t3b], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    m = max(24, int(math.sqrt(max(cfg.n_grid, 1024))))
    g1 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    g2 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    tg1, tg2 = np.meshgrid(g1, g2)
    t1g = tg1.ravel().astype(np.float32)
    t2g = tg2.ravel().astype(np.float32)
    rhs_g = (y_line - l1 * np.sin(t1g) - l2 * np.sin(t1g + t2g)) / l3
    valid_g = np.abs(rhs_g) <= 1.0
    t1gv = t1g[valid_g]
    t2gv = t2g[valid_g]
    ag = np.arcsin(rhs_g[valid_g]).astype(np.float32)
    t12g = t1gv + t2gv
    t3ga = _wrap_to_pi(ag - t12g)
    t3gb = _wrap_to_pi((math.pi - ag) - t12g)
    grid_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(t1gv), _wrap_to_pi(t2gv), t3ga], axis=1),
            np.stack([_wrap_to_pi(t1gv), _wrap_to_pi(t2gv), t3gb], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    x_train = _sample_rows(x_all, max(1, cfg.n_train))
    grid = _sample_rows(grid_all, max(1, cfg.n_grid))
    return x_train, grid


def _spatial_arm_plane_n4(cfg) -> Tuple[np.ndarray, np.ndarray]:
    # 4-DoF arm (q1 yaw, q2/q3/q4 pitch chain) constrained to z=z_plane.
    l1, l2, l3 = 1.0, 0.8, 0.6
    z_plane = 0.35

    n_cand = max(12 * cfg.n_train, 24000)
    q1 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)
    q2 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)
    q3 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)

    z12 = l1 * np.sin(q2) + l2 * np.sin(q2 + q3)
    rhs = (z_plane - z12) / l3
    valid = np.abs(rhs) <= 1.0
    q1v = q1[valid]
    q2v = q2[valid]
    q3v = q3[valid]
    a = np.arcsin(rhs[valid]).astype(np.float32)
    q23 = q2v + q3v
    q4a = _wrap_to_pi(a - q23)
    q4b = _wrap_to_pi((math.pi - a) - q23)
    x_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(q1v), _wrap_to_pi(q2v), _wrap_to_pi(q3v), q4a], axis=1),
            np.stack([_wrap_to_pi(q1v), _wrap_to_pi(q2v), _wrap_to_pi(q3v), q4b], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    # Dense candidate grid for planning/visualization.
    m = max(14, int(round(max(2, cfg.n_grid) ** (1.0 / 3.0))))
    g1 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    g2 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    g3 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    tg1, tg2, tg3 = np.meshgrid(g1, g2, g3, indexing="ij")
    q1g = tg1.ravel().astype(np.float32)
    q2g = tg2.ravel().astype(np.float32)
    q3g = tg3.ravel().astype(np.float32)
    z12g = l1 * np.sin(q2g) + l2 * np.sin(q2g + q3g)
    rhsg = (z_plane - z12g) / l3
    vg = np.abs(rhsg) <= 1.0
    q1gv = q1g[vg]
    q2gv = q2g[vg]
    q3gv = q3g[vg]
    ag = np.arcsin(rhsg[vg]).astype(np.float32)
    q23g = q2gv + q3gv
    q4ga = _wrap_to_pi(ag - q23g)
    q4gb = _wrap_to_pi((math.pi - ag) - q23g)
    grid_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(q1gv), _wrap_to_pi(q2gv), _wrap_to_pi(q3gv), q4ga], axis=1),
            np.stack([_wrap_to_pi(q1gv), _wrap_to_pi(q2gv), _wrap_to_pi(q3gv), q4gb], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    x_train = _sample_rows(x_all, max(1, cfg.n_train))
    grid = _sample_rows(grid_all, max(1, cfg.n_grid))
    return x_train, grid


def _spatial_arm_plane_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    # 3-DoF arm (q1 yaw, q2/q3 pitch chain) constrained to z=z_plane.
    l1, l2 = 1.0, 0.8
    z_plane = 0.35

    n_cand = max(12 * cfg.n_train, 18000)
    q1 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)
    q2 = np.random.uniform(-math.pi, math.pi, size=(n_cand,)).astype(np.float32)

    rhs = (z_plane - l1 * np.sin(q2)) / l2
    valid = np.abs(rhs) <= 1.0
    q1v = q1[valid]
    q2v = q2[valid]
    a = np.arcsin(rhs[valid]).astype(np.float32)
    q3a = _wrap_to_pi(a - q2v)
    q3b = _wrap_to_pi((math.pi - a) - q2v)
    x_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(q1v), _wrap_to_pi(q2v), q3a], axis=1),
            np.stack([_wrap_to_pi(q1v), _wrap_to_pi(q2v), q3b], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    m = max(26, int(round(max(2, cfg.n_grid) ** 0.5)))
    g1 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    g2 = np.linspace(-math.pi, math.pi, m, dtype=np.float32)
    tg1, tg2 = np.meshgrid(g1, g2, indexing="ij")
    q1g = tg1.ravel().astype(np.float32)
    q2g = tg2.ravel().astype(np.float32)
    rhsg = (z_plane - l1 * np.sin(q2g)) / l2
    vg = np.abs(rhsg) <= 1.0
    q1gv = q1g[vg]
    q2gv = q2g[vg]
    ag = np.arcsin(rhsg[vg]).astype(np.float32)
    q3ga = _wrap_to_pi(ag - q2gv)
    q3gb = _wrap_to_pi((math.pi - ag) - q2gv)
    grid_all = np.concatenate(
        [
            np.stack([_wrap_to_pi(q1gv), _wrap_to_pi(q2gv), q3ga], axis=1),
            np.stack([_wrap_to_pi(q1gv), _wrap_to_pi(q2gv), q3gb], axis=1),
        ],
        axis=0,
    ).astype(np.float32)

    x_train = _sample_rows(x_all, max(1, cfg.n_train))
    grid = _sample_rows(grid_all, max(1, cfg.n_grid))
    return x_train, grid


def _spatial_arm_ellip_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    # 3-DoF arm (q1 yaw, q2/q3 pitch chain) constrained to workspace ellipse:
    # x = a cos(t), y = b sin(t), z = z0  (codim=2 in 3D joint space).
    # The circle dataset name is an alias for this ellipse parameterization.
    l1, l2 = 1.0, 0.8
    a = 1.35
    b = 0.95
    z0 = 0.35

    def _build(n_target: int, *, use_grid: bool = False) -> np.ndarray:
        n_target = max(1, int(n_target))
        if use_grid:
            t = np.linspace(-math.pi, math.pi, n_target, endpoint=False, dtype=np.float32)
        else:
            t = np.random.uniform(-math.pi, math.pi, size=(n_target,)).astype(np.float32)

        x = (a * np.cos(t)).astype(np.float32)
        y = (b * np.sin(t)).astype(np.float32)
        rho = np.sqrt(np.maximum(x * x + y * y, 1e-8)).astype(np.float32)
        q1 = np.arctan2(y, x).astype(np.float32)

        rho2 = rho * rho + z0 * z0
        c3 = (rho2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        c3 = np.clip(c3, -1.0, 1.0).astype(np.float32)
        s3_mag = np.sqrt(np.maximum(0.0, 1.0 - c3 * c3)).astype(np.float32)

        pick_elbow_up = np.random.rand(n_target) < 0.5
        s3 = np.where(pick_elbow_up, s3_mag, -s3_mag).astype(np.float32)
        q3 = np.arctan2(s3, c3).astype(np.float32)
        q2 = (
            np.arctan2(np.full_like(rho, z0, dtype=np.float32), rho)
            - np.arctan2(l2 * s3, l1 + l2 * c3)
        ).astype(np.float32)
        return np.stack([_wrap_to_pi(q1), _wrap_to_pi(q2), _wrap_to_pi(q3)], axis=1).astype(np.float32)

    x_train = _build(cfg.n_train, use_grid=False)
    grid = _build(cfg.n_grid, use_grid=True)
    return x_train, grid


def _spatial_arm_up_n6(cfg) -> Tuple[np.ndarray, np.ndarray]:
    # 6-DoF arm with end-effector tool-axis "upward" orientation.
    # Kinematic orientation chain uses axes: z, y, y, y, x, z.
    # Tool axis is local +x; "up" means world-axis alignment with +z:
    # h1 = a_x = 0, h2 = a_y = 0, and enforce branch a_z > 0.
    # This defines codim=2 manifold in 6D.
    def _rot_x(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

    def _rot_y(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

    def _rot_z(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _tool_axis_world(q: np.ndarray) -> np.ndarray:
        # q shape: (6,)
        R = np.eye(3, dtype=np.float64)
        axes = ("z", "y", "y", "y", "x", "z")
        for j, ax in enumerate(axes):
            a = float(q[j])
            if ax == "x":
                R = R @ _rot_x(a)
            elif ax == "y":
                R = R @ _rot_y(a)
            else:
                R = R @ _rot_z(a)
        # tool/gripper pointing axis (local +x)
        return R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def _solve_q56_for_up(q14: np.ndarray, n_restart: int = 6) -> np.ndarray | None:
        # Solve h=[a_x,a_y]=0 for q5,q6 with branch a_z>0.
        best_q = None
        best_v = 1e18
        for _ in range(max(1, n_restart)):
            q = np.zeros((6,), dtype=np.float64)
            q[:4] = q14.astype(np.float64)
            q[4] = np.random.uniform(-math.pi, math.pi)
            q[5] = np.random.uniform(-math.pi, math.pi)
            for _it in range(32):
                a = _tool_axis_world(q)
                r = np.array([a[0], a[1]], dtype=np.float64)
                v = float(np.linalg.norm(r))
                if v < best_v and a[2] > 0.0:
                    best_v = v
                    best_q = q.copy()
                if v < 1e-6 and a[2] > 0.0:
                    return _wrap_to_pi(q.astype(np.float32))
                eps = 1e-4
                J = np.zeros((2, 2), dtype=np.float64)
                for k, jidx in enumerate((4, 5)):
                    qd = q.copy()
                    qd[jidx] += eps
                    ad = _tool_axis_world(qd)
                    rd = np.array([ad[0], ad[1]], dtype=np.float64)
                    J[:, k] = (rd - r) / eps
                JTJ = J.T @ J + 1e-6 * np.eye(2, dtype=np.float64)
                step = np.linalg.solve(JTJ, J.T @ r)
                step = np.clip(step, -0.35, 0.35)
                q[4] -= step[0]
                q[5] -= step[1]
                q = _wrap_to_pi(q.astype(np.float32)).astype(np.float64)
        if best_q is not None and best_v < 2e-3:
            return _wrap_to_pi(best_q.astype(np.float32))
        return None

    def _build_set(n_target: int) -> np.ndarray:
        out = []
        trials = 0
        max_trials = max(2000, 80 * n_target)
        while len(out) < n_target and trials < max_trials:
            trials += 1
            q14 = np.random.uniform(-math.pi, math.pi, size=(4,)).astype(np.float32)
            q = _solve_q56_for_up(q14, n_restart=6)
            if q is not None:
                out.append(q.astype(np.float32))
        if len(out) == 0:
            # Fallback to avoid empty dataset.
            q14 = np.random.uniform(-math.pi, math.pi, size=(n_target, 4)).astype(np.float32)
            q56 = np.zeros((n_target, 2), dtype=np.float32)
            return np.concatenate([q14, q56], axis=1).astype(np.float32)
        arr = np.stack(out, axis=0).astype(np.float32)
        return _sample_rows(arr, n_target)

    n_train = max(1, int(cfg.n_train))
    n_grid = max(1, int(cfg.n_grid))
    x_train = _build_set(n_train)
    grid = _build_set(n_grid)
    return x_train, grid


def sine_surface_affine_params(cfg=None) -> tuple[float, np.ndarray]:
    return float(_SINE_SURFACE_AFFINE_SCALE), _SINE_SURFACE_AFFINE_OFFSET_XYZ.copy()


def sine_surface_apply_affine_xy(xy: np.ndarray, cfg=None) -> np.ndarray:
    scale, offset = sine_surface_affine_params(cfg)
    arr = np.asarray(xy, dtype=np.float32)
    return (float(scale) * arr + offset[:2][None, :]).astype(np.float32)


def sine_surface_apply_affine_scalar(v: float | np.ndarray, cfg=None) -> float | np.ndarray:
    scale, _ = sine_surface_affine_params(cfg)
    arr = np.asarray(v, dtype=np.float32)
    out = (float(scale) * arr).astype(np.float32)
    if np.isscalar(v):
        return float(out.reshape(-1)[0])
    return out


def _surface_base_z_and_normal_from_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Canonical wave surface: z = a1*sin(fx*x) + a2*cos(fy*y)
    a1, a2 = 0.55, 0.35
    fx, fy = 1.2, 1.0
    z = (a1 * np.sin(fx * x) + a2 * np.cos(fy * y)).astype(np.float32)
    dzdx = a1 * fx * np.cos(fx * x)
    dzdy = -a2 * fy * np.sin(fy * y)
    n = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=1).astype(np.float64)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return z.astype(np.float32), n.astype(np.float32)


def sine_surface_z_and_normal_from_xy(x: np.ndarray, y: np.ndarray, cfg=None) -> tuple[np.ndarray, np.ndarray]:
    scale, offset = sine_surface_affine_params(cfg)
    x_base = (np.asarray(x, dtype=np.float32) - float(offset[0])) / float(scale)
    y_base = (np.asarray(y, dtype=np.float32) - float(offset[1])) / float(scale)
    z_base, n = _surface_base_z_and_normal_from_xy(x_base, y_base)
    z = (float(scale) * z_base + float(offset[2])).astype(np.float32)
    return z.astype(np.float32), n.astype(np.float32)


def _surface_frame_from_xy(x: np.ndarray, y: np.ndarray, cfg=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale, offset = sine_surface_affine_params(cfg)
    x_base = (np.asarray(x, dtype=np.float32) - float(offset[0])) / float(scale)
    y_base = (np.asarray(y, dtype=np.float32) - float(offset[1])) / float(scale)
    z, nvec = sine_surface_z_and_normal_from_xy(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), cfg)
    a1, fx = 0.55, 1.2
    t1 = np.stack([np.ones_like(x_base), np.zeros_like(x_base), a1 * fx * np.cos(fx * x_base)], axis=1).astype(np.float64)
    t1 /= (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12)
    t2 = np.cross(nvec.astype(np.float64), t1)
    t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12)
    return z.astype(np.float32), nvec.astype(np.float32), t1.astype(np.float32), t2.astype(np.float32)


def _rpy_from_rotmat_zyx(R: np.ndarray) -> np.ndarray:
    # R = Rz(yaw) Ry(pitch) Rx(roll)
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _surf_normal_n6(cfg) -> Tuple[np.ndarray, np.ndarray]:
    # 6D workspace pose: [x, y, z, roll, pitch, yaw]
    # Constraints:
    # 1) position lies on wave surface z=f(x,y)
    # 2) orientation's local z-axis aligns with surface normal
    # 3) free spin psi around normal remains unconstrained
    def _sample(n: int, seed_offset: int) -> np.ndarray:
        rng = np.random.default_rng(int(cfg.seed) + seed_offset)
        scale, offset = sine_surface_affine_params(cfg)
        x_base = rng.uniform(-2.0, 2.0, size=(n,)).astype(np.float32)
        y_base = rng.uniform(-2.0, 2.0, size=(n,)).astype(np.float32)
        z_base, nvec = _surface_base_z_and_normal_from_xy(x_base, y_base)
        x = (float(scale) * x_base + float(offset[0])).astype(np.float32)
        y = (float(scale) * y_base + float(offset[1])).astype(np.float32)
        z = (float(scale) * z_base + float(offset[2])).astype(np.float32)
        _z_world, _nvec_world, t1, t2 = _surface_frame_from_xy(x, y, cfg)

        psi = rng.uniform(-math.pi, math.pi, size=(n,)).astype(np.float64)
        c = np.cos(psi)[:, None]
        s = np.sin(psi)[:, None]
        x_axis = c * t1.astype(np.float64) + s * t2.astype(np.float64)
        y_axis = -s * t1.astype(np.float64) + c * t2.astype(np.float64)
        z_axis = _nvec_world.astype(np.float64)

        rpy = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            R = np.stack([x_axis[i], y_axis[i], z_axis[i]], axis=1).astype(np.float64)
            rpy[i] = _rpy_from_rotmat_zyx(R)
        rpy = _wrap_to_pi(rpy.astype(np.float32))
        return np.concatenate([x[:, None], y[:, None], z[:, None], rpy], axis=1).astype(np.float32)

    x_train = _sample(max(1, int(cfg.n_train)), seed_offset=0)
    grid = _sample(max(1, int(cfg.n_grid)), seed_offset=1)
    return x_train, grid


def _surf_normal_traj_n6(cfg) -> Tuple[np.ndarray, np.ndarray]:
    return _surf_normal_demo_sampler_n6(cfg)


def _traj_points_from_grid_random_turn(
    grid: np.ndarray,
    *,
    n_train: int,
    seed: int,
    traj_count: int,
    traj_len: int,
    traj_knn: int,
    forward_weight: float,
    turn_noise: float,
    turn_prob: float,
    direction_blend: float,
    visit_weight: float,
    distance_weight: float,
    boundary_weight: float,
    boundary_steer_weight: float,
    boundary_margin_ratio: float,
) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float32)
    if len(g) == 0:
        return np.zeros((max(1, int(n_train)), g.shape[1] if g.ndim == 2 else 3), dtype=np.float32)

    rng = np.random.default_rng(int(seed) + 971)
    n = max(1, int(n_train))
    n_traj = max(1, int(traj_count))
    t_len = max(2, int(traj_len))
    knn = _knn_indices(g, k=max(2, int(traj_knn)))

    eps = 1e-8
    starts: list[int] = []
    starts.append(int(rng.integers(0, len(g))))
    while len(starts) < n_traj:
        cand = int(rng.integers(0, len(g)))
        if cand in starts:
            continue
        d2 = np.sum((g[np.array(starts)] - g[cand : cand + 1]) ** 2, axis=1)
        if float(np.min(d2)) > 0.03 * float(np.mean(np.var(g, axis=0) + eps)):
            starts.append(cand)
        elif float(rng.uniform()) < 0.08:
            starts.append(cand)

    visit = np.zeros((len(g),), dtype=np.int32)
    xy_min = g[:, :2].min(axis=0)
    xy_max = g[:, :2].max(axis=0)
    xy_span = np.maximum(xy_max - xy_min, 1e-8)
    margin = float(max(0.0, boundary_margin_ratio))
    xy_lo = xy_min + margin * xy_span
    xy_hi = xy_max - margin * xy_span

    seq = []
    for ti in range(n_traj):
        cur = int(starts[ti % len(starts)])
        prev_idx = -1
        prev_dir = None
        turn_state = None
        for _step in range(t_len):
            seq.append(g[cur].astype(np.float32))
            visit[cur] += 1

            nbr = knn[cur]
            nbr = nbr[nbr != cur]
            if prev_idx >= 0:
                nbr = nbr[nbr != prev_idx]
            if len(nbr) == 0:
                cur = int(rng.integers(0, len(g)))
                prev_idx = -1
                prev_dir = None
                continue

            cand = nbr[: min(18, len(nbr))]
            vec = (g[cand] - g[cur]).astype(np.float32)
            d = np.linalg.norm(vec, axis=1) + eps
            u = vec / d[:, None]
            cov = visit[cand].astype(np.float32)
            cov_gain = (np.max(cov) - cov) if len(cov) > 0 else np.zeros_like(d)

            inward_dir = None
            if float(boundary_weight) > 0.0 and margin > 0.0:
                cur_xy = g[cur, :2]
                inward_xy = np.zeros((2,), dtype=np.float32)
                inward_xy += np.maximum(xy_lo - cur_xy, 0.0).astype(np.float32) / xy_span.astype(np.float32)
                inward_xy -= np.maximum(cur_xy - xy_hi, 0.0).astype(np.float32) / xy_span.astype(np.float32)
                inward_norm = float(np.linalg.norm(inward_xy))
                if inward_norm > 1e-8:
                    inward = np.array([inward_xy[0], inward_xy[1], 0.0], dtype=np.float32)
                    inward_dir = inward / max(float(np.linalg.norm(inward)), 1e-8)

            boundary_penalty = np.zeros_like(d)
            if float(boundary_weight) > 0.0 and margin > 0.0:
                xy = g[cand, :2]
                below = np.maximum(xy_lo.reshape(1, 2) - xy, 0.0) / xy_span.reshape(1, 2)
                above = np.maximum(xy - xy_hi.reshape(1, 2), 0.0) / xy_span.reshape(1, 2)
                boundary_penalty = np.sum(below + above, axis=1).astype(np.float32)

            if prev_dir is None:
                score = (
                    -float(distance_weight) * d
                    + float(visit_weight) * cov_gain
                    - float(boundary_weight) * boundary_penalty
                )
                if inward_dir is not None:
                    score = score + float(boundary_steer_weight) * np.sum(u * inward_dir.reshape(1, -1), axis=1)
                if float(turn_prob) > 0.0:
                    score = score + rng.normal(scale=0.03 * float(turn_prob), size=score.shape).astype(np.float32)
            else:
                random_turn = rng.normal(size=prev_dir.shape).astype(np.float32)
                random_turn = random_turn - float(np.dot(random_turn, prev_dir)) * prev_dir
                random_turn_norm = float(np.linalg.norm(random_turn))
                if random_turn_norm > 1e-8:
                    random_turn = random_turn / random_turn_norm
                    if turn_state is None:
                        turn_state = random_turn.astype(np.float32)
                    else:
                        turn_state = (0.88 * turn_state + 0.12 * random_turn).astype(np.float32)
                        turn_state = turn_state / max(float(np.linalg.norm(turn_state)), 1e-8)
                desired = prev_dir.astype(np.float32)
                if turn_state is not None and float(turn_noise) > 0.0:
                    desired = (desired + float(turn_noise) * turn_state).astype(np.float32)
                    desired = desired / max(float(np.linalg.norm(desired)), 1e-8)
                if inward_dir is not None:
                    desired = (desired + float(boundary_steer_weight) * inward_dir).astype(np.float32)
                    desired = desired / max(float(np.linalg.norm(desired)), 1e-8)
                align = np.clip(np.sum(u * desired.reshape(1, -1), axis=1), -1.0, 1.0)
                score = (
                    float(forward_weight) * align
                    - float(distance_weight) * d
                    + float(visit_weight) * cov_gain
                    - float(boundary_weight) * boundary_penalty
                )
            nxt = int(cand[int(np.argmax(score))])
            new_vec = (g[nxt] - g[cur]).astype(np.float32)
            new_norm = float(np.linalg.norm(new_vec))
            if new_norm > 1e-8:
                new_u = new_vec / new_norm
                if prev_dir is None:
                    prev_dir = new_u.astype(np.float32)
                else:
                    blend = float(np.clip(direction_blend, 0.0, 1.0))
                    prev_dir = (blend * prev_dir + (1.0 - blend) * new_u).astype(np.float32)
                    prev_dir = prev_dir / max(float(np.linalg.norm(prev_dir)), 1e-8)
            prev_idx = cur
            cur = nxt

    arr = np.asarray(seq, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((n, g.shape[1]), dtype=np.float32)
    if len(arr) >= n:
        return arr[:n].astype(np.float32)
    rep = int(math.ceil(float(n) / float(len(arr))))
    tiled = np.tile(arr, (rep, 1)).astype(np.float32)
    return tiled[:n].astype(np.float32)


def _surf_normal_demo_sampler_n6(cfg) -> Tuple[np.ndarray, np.ndarray]:
    _, grid = _surf_normal_n6(cfg)
    xyz_grid = grid[:, :3].astype(np.float32)

    n_train = max(1, int(cfg.n_train))
    traj_count_raw = int(getattr(cfg, "traj_count", max(12, n_train // 64)))
    traj_count = int(max(1, min(n_train, traj_count_raw)))
    traj_len = int(max(8, getattr(cfg, "traj_len", int(math.ceil(n_train / max(traj_count, 1))))))
    traj_knn = int(max(4, getattr(cfg, "traj_knn", 20)))
    xyz_train = _traj_points_from_grid_random_turn(
        xyz_grid,
        n_train=n_train,
        seed=int(getattr(cfg, "seed", 0)),
        traj_count=traj_count,
        traj_len=traj_len,
        traj_knn=traj_knn,
        forward_weight=float(getattr(cfg, "traj_turn_forward_weight", 0.7)),
        turn_noise=float(getattr(cfg, "traj_turn_noise", 0.55)),
        turn_prob=float(getattr(cfg, "traj_turn_prob", 0.30)),
        direction_blend=float(getattr(cfg, "traj_turn_direction_blend", 0.35)),
        visit_weight=float(getattr(cfg, "traj_turn_visit_weight", 1.35)),
        distance_weight=float(getattr(cfg, "traj_turn_distance_weight", 0.08)),
        boundary_weight=float(getattr(cfg, "traj_turn_boundary_weight", 1.5)),
        boundary_steer_weight=float(getattr(cfg, "traj_turn_boundary_steer_weight", 0.25)),
        boundary_margin_ratio=float(getattr(cfg, "traj_turn_boundary_margin_ratio", 0.10)),
    ).astype(np.float32)

    x = xyz_train[:, 0].astype(np.float32)
    y = xyz_train[:, 1].astype(np.float32)
    _z_world, nvec, t1, t2 = _surface_frame_from_xy(x, y, cfg)

    rng = np.random.default_rng(int(cfg.seed) + 11)
    psi_step_std = float(max(1e-4, getattr(cfg, "traj_psi_step_std", 0.07)))
    psi = np.zeros((n_train,), dtype=np.float64)
    seg_len = max(2, int(math.ceil(n_train / max(traj_count, 1))))
    for k in range(traj_count):
        a = k * seg_len
        b = min(n_train, (k + 1) * seg_len)
        if a >= b:
            continue
        psi0 = rng.uniform(-math.pi, math.pi)
        dpsi = rng.normal(scale=psi_step_std, size=(b - a,)).astype(np.float64)
        psi_seg = psi0 + np.cumsum(dpsi)
        psi[a:b] = psi_seg
    psi = ((psi + math.pi) % (2.0 * math.pi) - math.pi).astype(np.float64)

    c = np.cos(psi)[:, None]
    s = np.sin(psi)[:, None]
    x_axis = c * t1.astype(np.float64) + s * t2.astype(np.float64)
    y_axis = -s * t1.astype(np.float64) + c * t2.astype(np.float64)
    z_axis = nvec.astype(np.float64)

    rpy = np.zeros((n_train, 3), dtype=np.float32)
    for i in range(n_train):
        R = np.stack([x_axis[i], y_axis[i], z_axis[i]], axis=1).astype(np.float64)
        rpy[i] = _rpy_from_rotmat_zyx(R)
    rpy = _wrap_to_pi(rpy.astype(np.float32))

    x_train = np.concatenate([xyz_train.astype(np.float32), rpy], axis=1).astype(np.float32)
    return x_train, grid.astype(np.float32)


def _dual_arm_guided_insertion_curve(
    s: np.ndarray,
    *,
    x_span: float,
    y_amp: float,
    y_freq: float,
    z_base: float,
    z_amp: float,
    z_freq: float,
) -> tuple[np.ndarray, np.ndarray]:
    del z_amp, z_freq
    ss = s.astype(np.float32)
    x = (float(x_span) * ss).astype(np.float32)
    y = (float(y_amp) * np.sin(float(y_freq) * np.pi * ss)).astype(np.float32)
    z = np.full_like(ss, float(z_base), dtype=np.float32)

    dx = np.full_like(ss, float(x_span), dtype=np.float32)
    dy = (float(y_amp) * float(y_freq) * np.pi * np.cos(float(y_freq) * np.pi * ss)).astype(np.float32)
    dz = np.zeros_like(ss, dtype=np.float32)

    pos = np.stack([x, y, z], axis=1).astype(np.float32)
    tan = np.stack([dx, dy, dz], axis=1).astype(np.float32)
    tan /= (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12)
    return pos, tan


def _dual_arm_guided_insertion_frame(
    s: np.ndarray,
    *,
    x_span: float,
    y_amp: float,
    y_freq: float,
    z_base: float,
    z_amp: float,
    z_freq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos, tan = _dual_arm_guided_insertion_curve(
        s,
        x_span=x_span,
        y_amp=y_amp,
        y_freq=y_freq,
        z_base=z_base,
        z_amp=z_amp,
        z_freq=z_freq,
    )
    ref_up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(s), 1))
    alt_up = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (len(s), 1))
    use_alt = np.abs(np.sum(tan * ref_up, axis=1)) > 0.95
    ref = ref_up.copy()
    ref[use_alt] = alt_up[use_alt]

    normal = ref - np.sum(ref * tan, axis=1, keepdims=True) * tan
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)
    binormal = np.cross(tan, normal).astype(np.float32)
    binormal /= (np.linalg.norm(binormal, axis=1, keepdims=True) + 1e-12)
    normal = np.cross(binormal, tan).astype(np.float32)
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)
    return pos.astype(np.float32), tan.astype(np.float32), normal.astype(np.float32), binormal.astype(np.float32)


def _dual_arm_pose_from_s_u(
    s: np.ndarray,
    u: np.ndarray,
    *,
    grasp_span: float,
    x_span: float,
    y_amp: float,
    y_freq: float,
    z_base: float,
    z_amp: float,
    z_freq: float,
    right_hand_opposite: bool = False,
) -> np.ndarray:
    ss = np.asarray(s, dtype=np.float32).reshape(-1)
    uu = np.asarray(u, dtype=np.float32).reshape(-1)
    if len(ss) != len(uu):
        raise ValueError(f"s and u must have same length, got {len(ss)} and {len(uu)}")
    n = int(len(ss))
    center, tang, _normal, _binormal = _dual_arm_guided_insertion_frame(
        ss,
        x_span=x_span,
        y_amp=y_amp,
        y_freq=y_freq,
        z_base=z_base,
        z_amp=z_amp,
        z_freq=z_freq,
    )
    center[:, 2] += uu

    z_axis = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
    y_axis = np.cross(z_axis, tang).astype(np.float32)
    y_axis /= (np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-12)
    z_axis = np.cross(tang, y_axis).astype(np.float32)
    z_axis /= (np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-12)

    offset = (0.5 * float(grasp_span) * tang).astype(np.float32)
    pos_1 = (center - offset).astype(np.float32)
    pos_2 = (center + offset).astype(np.float32)

    rpy_1 = np.zeros((n, 3), dtype=np.float32)
    rpy_2 = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        R = np.stack([tang[i], y_axis[i], z_axis[i]], axis=1).astype(np.float64)
        rpy_1[i] = _rpy_from_rotmat_zyx(R)
        if right_hand_opposite:
            rpy_2[i] = _rpy_from_rotmat_zyx(R @ np.diag([-1.0, 1.0, -1.0]))
        else:
            rpy_2[i] = rpy_1[i]
    rpy_1 = _wrap_to_pi(rpy_1.astype(np.float32))
    rpy_2 = _wrap_to_pi(rpy_2.astype(np.float32))

    pose_1 = np.concatenate([pos_1, rpy_1], axis=1).astype(np.float32)
    pose_2 = np.concatenate([pos_2, rpy_2], axis=1).astype(np.float32)
    return np.concatenate([pose_1, pose_2], axis=1).astype(np.float32)


def _dual_arm_demo_height_profile(tau: np.ndarray, rng: np.random.Generator, z_half_range: float) -> np.ndarray:
    tt = np.asarray(tau, dtype=np.float32).reshape(-1)
    half = float(max(1e-8, z_half_range))
    freq1 = float(rng.uniform(0.70, 1.45))
    freq2 = float(rng.uniform(1.60, 2.60))
    phase1 = float(rng.uniform(0.0, 2.0 * math.pi))
    phase2 = float(rng.uniform(0.0, 2.0 * math.pi))
    mix = float(rng.uniform(0.18, 0.42))
    offset = float(rng.uniform(-0.18, 0.18))
    profile = (
        np.sin(2.0 * math.pi * freq1 * tt + phase1)
        + mix * np.sin(2.0 * math.pi * freq2 * tt + phase2)
        + offset
    ).astype(np.float32)
    profile -= float(np.mean(profile))
    denom = max(float(np.max(np.abs(profile))), 1e-6)
    amp = float(rng.uniform(0.45, 0.88)) * half
    u = (amp * profile / denom).astype(np.float32)
    return np.clip(u, -0.98 * half, 0.98 * half).astype(np.float32)


def _dual_arm_demo_trajectory_samples(cfg, *, n_train: int, traj_count: int, traj_len: int) -> np.ndarray:
    grasp_span = float(getattr(cfg, "dual_arm_grasp_span", 1.0))
    x_span = float(getattr(cfg, "dual_arm_curve_x_span", 1.4))
    y_amp = float(getattr(cfg, "dual_arm_curve_y_amp", 0.55))
    y_freq = float(getattr(cfg, "dual_arm_curve_y_freq", 1.0))
    z_base = float(getattr(cfg, "dual_arm_curve_z_base", 0.2))
    z_amp = float(getattr(cfg, "dual_arm_curve_z_amp", 0.35))
    z_freq = float(getattr(cfg, "dual_arm_curve_z_freq", 0.7))
    z_half_range = float(getattr(cfg, "dual_arm_vertical_half_range", z_amp))
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 2903)

    seq: list[np.ndarray] = []
    n_demo = int(max(1, traj_count))
    t_len = int(max(2, traj_len))
    for _ in range(n_demo):
        tau = np.linspace(0.0, 1.0, num=t_len, dtype=np.float32)
        margin = float(rng.uniform(0.00, 0.12))
        direction = 1.0 if float(rng.uniform()) < 0.5 else -1.0
        s0, s1 = -1.0 + margin, 1.0 - margin
        s = (s0 + (s1 - s0) * tau).astype(np.float32)
        if direction < 0.0:
            s = s[::-1].copy()
        u = _dual_arm_demo_height_profile(tau, rng, z_half_range)
        seq.append(
            _dual_arm_pose_from_s_u(
                s,
                u,
                grasp_span=grasp_span,
                x_span=x_span,
                y_amp=y_amp,
                y_freq=y_freq,
                z_base=z_base,
                z_amp=z_amp,
                z_freq=z_freq,
                right_hand_opposite=False,
            )
        )

    arr = np.concatenate(seq, axis=0).astype(np.float32) if seq else np.zeros((0, 12), dtype=np.float32)
    n = int(max(1, n_train))
    if len(arr) >= n:
        return arr[:n].astype(np.float32)
    rep = int(math.ceil(float(n) / max(float(len(arr)), 1.0)))
    return np.tile(arr, (rep, 1))[:n].astype(np.float32)


def _dual_arm_pose_yoffset_samex_plane_sameori_n12(cfg) -> Tuple[np.ndarray, np.ndarray]:
    # Dual-arm guided insertion manifold.
    # x = [ee1_pose(6), ee2_pose(6)] with end-effectors rigidly attached to the same object.
    # The object center lies on a vertical ribbon generated by lifting a horizontal guide
    # curve along world z. Free variables: path progress s and height offset u.
    grasp_span = float(getattr(cfg, "dual_arm_grasp_span", 1.0))
    x_span = float(getattr(cfg, "dual_arm_curve_x_span", 1.4))
    y_amp = float(getattr(cfg, "dual_arm_curve_y_amp", 0.55))
    y_freq = float(getattr(cfg, "dual_arm_curve_y_freq", 1.0))
    z_base = float(getattr(cfg, "dual_arm_curve_z_base", 0.2))
    z_amp = float(getattr(cfg, "dual_arm_curve_z_amp", 0.35))
    z_freq = float(getattr(cfg, "dual_arm_curve_z_freq", 0.7))
    z_half_range = float(getattr(cfg, "dual_arm_vertical_half_range", z_amp))

    def _sample(n: int, seed_offset: int) -> np.ndarray:
        rng = np.random.default_rng(int(cfg.seed) + seed_offset)
        s = rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32)
        u = rng.uniform(-z_half_range, z_half_range, size=(n,)).astype(np.float32)

        return _dual_arm_pose_from_s_u(
            s,
            u,
            grasp_span=grasp_span,
            x_span=x_span,
            y_amp=y_amp,
            y_freq=y_freq,
            z_base=z_base,
            z_amp=z_amp,
            z_freq=z_freq,
            right_hand_opposite=False,
        )

    x_train = _sample(max(1, int(cfg.n_train)), seed_offset=0)
    grid = _sample(max(1, int(cfg.n_grid)), seed_offset=1)
    return x_train, grid


def _workspace_pose_rpy_embed(x: np.ndarray) -> np.ndarray:
    q = np.asarray(x, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] < 6:
        return q.astype(np.float32, copy=False)
    blocks = []
    for base in range(0, q.shape[1], 6):
        if base + 6 <= q.shape[1]:
            pos = q[:, base : base + 3]
            rpy = q[:, base + 3 : base + 6]
            blocks.append(
                np.concatenate(
                    [pos, np.sin(rpy).astype(np.float32), np.cos(rpy).astype(np.float32)],
                    axis=1,
                ).astype(np.float32)
            )
        else:
            blocks.append(q[:, base:].astype(np.float32))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    pts = points.astype(np.float32)
    n = int(pts.shape[0])
    kk = int(max(2, min(k, n)))
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(pts.astype(np.float64))
        _, idx = tree.query(pts.astype(np.float64), k=kk)
        if idx.ndim == 1:
            idx = idx[:, None]
        return idx.astype(np.int64)
    except Exception:
        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
        idx = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        return idx.astype(np.int64)


def _traj_points_from_grid(
    grid: np.ndarray,
    *,
    n_train: int,
    seed: int,
    traj_count: int,
    traj_len: int,
    traj_knn: int,
    diverse_starts: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float32)
    if len(g) == 0:
        return np.zeros((max(1, int(n_train)), g.shape[1] if g.ndim == 2 else 3), dtype=np.float32)

    rng = np.random.default_rng(int(seed) + 971)
    n = max(1, int(n_train))
    n_traj = max(1, int(traj_count))
    t_len = max(2, int(traj_len))
    knn = _knn_indices(g, k=max(2, int(traj_knn)))

    eps = 1e-8
    starts: list[int] = []
    if not diverse_starts:
        # Choose diverse starts to improve global coverage.
        if len(g) > 0:
            starts.append(int(rng.integers(0, len(g))))
        while len(starts) < n_traj:
            cand = int(rng.integers(0, len(g)))
            if cand in starts:
                continue
            if len(starts) == 0:
                starts.append(cand)
                continue
            d2 = np.sum((g[np.array(starts)] - g[cand : cand + 1]) ** 2, axis=1)
            # Keep starts reasonably far apart.
            if float(np.min(d2)) > 0.03 * float(np.mean(np.var(g, axis=0) + eps)):
                starts.append(cand)
            elif float(rng.uniform()) < 0.08:
                starts.append(cand)

    visit = np.zeros((len(g),), dtype=np.int32)

    def _pick_dynamic_start(used_starts: list[int]) -> int:
        pool = int(min(len(g), max(256, 32 * n_traj)))
        if len(g) <= pool:
            cand_idx = np.arange(len(g), dtype=np.int64)
        else:
            cand_idx = rng.choice(len(g), size=pool, replace=False).astype(np.int64)
        if len(cand_idx) == 0:
            return int(rng.integers(0, len(g)))
        if len(used_starts) == 0:
            v = visit[cand_idx].astype(np.float32)
            min_v = float(np.min(v))
            good = cand_idx[v <= min_v + 1e-6]
            return int(good[int(rng.integers(0, len(good)))])

        d2 = np.sum(
            (g[cand_idx][:, None, :] - g[np.array(used_starts, dtype=np.int64)][None, :, :]) ** 2,
            axis=2,
        )
        d_score = np.min(d2, axis=1).astype(np.float32)
        v = visit[cand_idx].astype(np.float32)
        v_score = (np.max(v) - v) if len(v) > 0 else np.zeros_like(d_score)
        d_norm = d_score / (float(np.max(d_score)) + 1e-8)
        v_norm = v_score / (float(np.max(v_score)) + 1e-8)
        score = d_norm + 0.70 * v_norm
        return int(cand_idx[int(np.argmax(score))])
    seq = []
    empty_neighbor_warn_count = 0
    for ti in range(n_traj):
        if diverse_starts:
            cur = _pick_dynamic_start(starts)
            starts.append(int(cur))
        else:
            cur = int(starts[ti % len(starts)])
        prev_idx = -1
        prev_dir = None
        for _step in range(t_len):
            seq.append(g[cur].astype(np.float32))
            visit[cur] += 1
            nbr = knn[cur]
            nbr = nbr[nbr != cur]
            if prev_idx >= 0:
                nbr = nbr[nbr != prev_idx]
            if len(nbr) == 0:
                empty_neighbor_warn_count += 1
                if empty_neighbor_warn_count <= 10:
                    print(
                        "[warn][traj] empty neighbor set after filtering; "
                        f"traj={ti} step={_step} cur={cur} prev={prev_idx} "
                        f"n_grid={len(g)} traj_knn={int(traj_knn)}"
                    )
                cur = int(rng.integers(0, len(g)))
                prev_idx = -1
                prev_dir = None
                continue

            cand = nbr[: min(14, len(nbr))]
            vec = (g[cand] - g[cur]).astype(np.float32)
            d = np.linalg.norm(vec, axis=1) + eps
            u = vec / d[:, None]
            cov = visit[cand].astype(np.float32)
            # Coverage gain: prefer less visited regions.
            cov_gain = (np.max(cov) - cov) if len(cov) > 0 else np.zeros_like(d)

            if prev_dir is None:
                # First move: prefer local smooth step (short distance).
                score = -d + 0.65 * cov_gain
            else:
                a = np.clip(np.sum(u * prev_dir.reshape(1, -1), axis=1), -1.0, 1.0)
                # Strongly prefer forward continuation, weakly prefer short step.
                score = 3.2 * a - 0.22 * d + 0.90 * cov_gain
                # Occasionally allow controlled turn to avoid dead loops.
                if float(rng.uniform()) < 0.08:
                    score = score + rng.normal(scale=0.35, size=score.shape).astype(np.float32)

            pick = int(np.argmax(score))
            nxt = int(cand[pick])
            new_vec = (g[nxt] - g[cur]).astype(np.float32)
            new_norm = float(np.linalg.norm(new_vec))
            if new_norm > 1e-8:
                new_u = new_vec / new_norm
                if prev_dir is None:
                    prev_dir = new_u.astype(np.float32)
                else:
                    prev_dir = (0.82 * prev_dir + 0.18 * new_u).astype(np.float32)
                    prev_dir = prev_dir / max(float(np.linalg.norm(prev_dir)), 1e-8)
            prev_idx = cur
            cur = nxt

    arr = np.asarray(seq, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((n, g.shape[1]), dtype=np.float32)
    # Keep trajectory ordering in x_train so downstream consumers can visualize
    # contiguous motion segments instead of shuffled manifold points.
    if len(arr) >= n:
        return arr[:n].astype(np.float32)
    rep = int(math.ceil(float(n) / float(len(arr))))
    tiled = np.tile(arr, (rep, 1)).astype(np.float32)
    return tiled[:n].astype(np.float32)


def _traj_points_from_mesh_edges(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    n_train: int,
    seed: int,
    traj_count: int,
    traj_len: int,
) -> np.ndarray:
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if len(v) == 0:
        return np.zeros((max(1, int(n_train)), 3), dtype=np.float32)

    neighbors: list[set[int]] = [set() for _ in range(len(v))]
    for tri in f:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    adj = [np.asarray(sorted(nbr), dtype=np.int64) for nbr in neighbors]

    rng = np.random.default_rng(int(seed) + 1987)
    n = max(1, int(n_train))
    n_traj = max(1, int(traj_count))
    t_len = max(2, int(traj_len))
    visit = np.zeros((len(v),), dtype=np.int32)
    starts: list[int] = []
    eps = 1e-8

    def _pick_start() -> int:
        pool = min(len(v), max(512, 48 * n_traj))
        cand = np.arange(len(v), dtype=np.int64) if len(v) <= pool else rng.choice(len(v), size=pool, replace=False)
        if len(starts) == 0:
            min_visit = np.min(visit[cand])
            good = cand[visit[cand] <= min_visit]
            return int(good[int(rng.integers(0, len(good)))])
        d2 = np.sum((v[cand][:, None, :] - v[np.asarray(starts, dtype=np.int64)][None, :, :]) ** 2, axis=2)
        d_score = np.min(d2, axis=1)
        visit_score = np.max(visit[cand]) - visit[cand]
        score = d_score / (np.max(d_score) + eps) + 0.7 * visit_score / (np.max(visit_score) + eps)
        return int(cand[int(np.argmax(score))])

    seq: list[np.ndarray] = []
    for _ti in range(n_traj):
        cur = _pick_start()
        starts.append(cur)
        prev = -1
        prev_dir = None
        for _step in range(t_len):
            seq.append(v[cur].astype(np.float32))
            visit[cur] += 1
            cand = adj[cur]
            if prev >= 0:
                cand = cand[cand != prev]
            if len(cand) == 0:
                cur = _pick_start()
                prev = -1
                prev_dir = None
                continue

            vec = v[cand] - v[cur : cur + 1]
            dist = np.linalg.norm(vec, axis=1) + eps
            unit = vec / dist[:, None]
            cov = visit[cand].astype(np.float32)
            cov_gain = np.max(cov) - cov
            if prev_dir is None:
                score = -0.4 * dist + 0.8 * cov_gain
            else:
                forward = np.clip(np.sum(unit * prev_dir.reshape(1, -1), axis=1), -1.0, 1.0)
                score = 2.6 * forward - 0.25 * dist + 0.9 * cov_gain
                if float(rng.uniform()) < 0.06:
                    score = score + rng.normal(scale=0.25, size=score.shape).astype(np.float32)
            nxt = int(cand[int(np.argmax(score))])
            new_vec = v[nxt] - v[cur]
            new_norm = float(np.linalg.norm(new_vec))
            if new_norm > eps:
                new_dir = (new_vec / new_norm).astype(np.float32)
                prev_dir = new_dir if prev_dir is None else (0.84 * prev_dir + 0.16 * new_dir).astype(np.float32)
                prev_dir = prev_dir / max(float(np.linalg.norm(prev_dir)), eps)
            prev = cur
            cur = nxt

    arr = np.asarray(seq, dtype=np.float32)
    if len(arr) >= n:
        return arr[:n].astype(np.float32)
    rep = int(math.ceil(float(n) / float(max(len(arr), 1))))
    return np.tile(arr, (rep, 1))[:n].astype(np.float32)


def _project_to_nearest_mesh_vertices(points: np.ndarray, verts: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    v = np.asarray(verts, dtype=np.float32)
    if len(p) == 0 or len(v) == 0:
        return p.astype(np.float32)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        _, idx = cKDTree(v.astype(np.float64)).query(p.astype(np.float64), k=1)
        return v[np.asarray(idx, dtype=np.int64)].astype(np.float32)
    except Exception:
        out = np.empty_like(p, dtype=np.float32)
        chunk = 512
        for start in range(0, len(p), chunk):
            q = p[start : start + chunk]
            d2 = np.sum((q[:, None, :] - v[None, :, :]) ** 2, axis=2)
            out[start : start + chunk] = v[np.argmin(d2, axis=1)]
        return out.astype(np.float32)


def _genus3_tube_sdf(points: np.ndarray, *, radius: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    segments = _genus3_tube_segments()
    seg_a = segments[:, 0]
    seg_b = segments[:, 1]
    seg_ab = seg_b - seg_a
    seg_len2 = np.sum(seg_ab * seg_ab, axis=1).clip(1e-8)
    out = np.empty((len(p),), dtype=np.float32)
    chunk = 512
    for start in range(0, len(p), chunk):
        q = p[start : start + chunk]
        ap = q[:, None, :] - seg_a[None, :, :]
        t = np.clip(np.sum(ap * seg_ab[None, :, :], axis=2) / seg_len2[None, :], 0.0, 1.0)
        closest = seg_a[None, :, :] + t[:, :, None] * seg_ab[None, :, :]
        d2 = np.sum((q[:, None, :] - closest) ** 2, axis=2)
        out[start : start + chunk] = np.sqrt(np.min(d2, axis=1)).astype(np.float32) - float(radius)
    return out.astype(np.float32)


def _curve_tube_surface_points(
    curve: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    *,
    radius: float,
    closed: bool,
) -> np.ndarray:
    c = np.asarray(curve, dtype=np.float32)
    uu = np.asarray(u, dtype=np.float32).reshape(-1)
    th = np.asarray(theta, dtype=np.float32).reshape(-1)
    n_curve = int(len(c))
    if n_curve < 2:
        return np.repeat(c[:1], len(uu), axis=0).astype(np.float32)

    if closed:
        tt = np.mod(uu, 1.0) * float(n_curve)
        i0 = np.floor(tt).astype(np.int64) % n_curve
        i1 = (i0 + 1) % n_curve
        im = (i0 - 1) % n_curve
        frac = (tt - np.floor(tt)).astype(np.float32)[:, None]
    else:
        tt = np.clip(uu, 0.0, 1.0) * float(n_curve - 1)
        i0 = np.clip(np.floor(tt).astype(np.int64), 0, n_curve - 2)
        i1 = np.clip(i0 + 1, 0, n_curve - 1)
        im = np.clip(i0 - 1, 0, n_curve - 1)
        frac = (tt - np.floor(tt)).astype(np.float32)[:, None]

    center = (1.0 - frac) * c[i0] + frac * c[i1]
    normal_vertices, binormal_vertices = _curve_parallel_transport_frames(c, closed=closed)
    normal = (1.0 - frac) * normal_vertices[i0] + frac * normal_vertices[i1]
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    binormal = (1.0 - frac) * binormal_vertices[i0] + frac * binormal_vertices[i1]
    binormal = binormal - np.sum(binormal * normal, axis=1, keepdims=True) * normal
    binormal = binormal / np.maximum(np.linalg.norm(binormal, axis=1, keepdims=True), 1e-8)
    offset = np.cos(th)[:, None] * normal + np.sin(th)[:, None] * binormal
    return (center + float(radius) * offset).astype(np.float32)


def _curve_parallel_transport_frames(curve: np.ndarray, *, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(curve, dtype=np.float32)
    n = int(len(c))
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    if n == 1:
        return (
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        )

    tangents = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        if closed:
            t = c[(i + 1) % n] - c[(i - 1) % n]
        elif i == 0:
            t = c[1] - c[0]
        elif i == n - 1:
            t = c[-1] - c[-2]
        else:
            t = c[i + 1] - c[i - 1]
        tangents[i] = t / max(float(np.linalg.norm(t)), 1e-8)

    refs = [
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    ]
    ref = min(refs, key=lambda r: abs(float(np.dot(r, tangents[0]))))
    normal0 = ref - float(np.dot(ref, tangents[0])) * tangents[0]
    normal0 = normal0 / max(float(np.linalg.norm(normal0)), 1e-8)

    normals = np.empty((n, 3), dtype=np.float32)
    normals[0] = normal0.astype(np.float32)
    for i in range(1, n):
        prev = normals[i - 1]
        t = tangents[i]
        cur = prev - float(np.dot(prev, t)) * t
        cur_norm = float(np.linalg.norm(cur))
        if cur_norm < 1e-6:
            ref = min(refs, key=lambda r: abs(float(np.dot(r, t))))
            cur = ref - float(np.dot(ref, t)) * t
            cur_norm = float(np.linalg.norm(cur))
        normals[i] = (cur / max(cur_norm, 1e-8)).astype(np.float32)
    binormals = np.cross(tangents, normals)
    binormals = binormals / np.maximum(np.linalg.norm(binormals, axis=1, keepdims=True), 1e-8)
    return normals.astype(np.float32), binormals.astype(np.float32)


def _genus3_tube_param_trajectory_samples(
    *,
    n_train: int,
    seed: int,
    traj_count: int,
    traj_len: int,
    radius: float,
    sdf_tol: float = 0.018,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 2467)
    curves = _genus3_tube_skeleton_curves()
    closed_flags = [True, True, True, False, False]
    lengths = []
    for curve, closed in zip(curves, closed_flags):
        cc = np.vstack([curve, curve[:1]]) if closed else curve
        lengths.append(float(np.sum(np.linalg.norm(np.diff(cc, axis=0), axis=1))))
    prob = np.asarray(lengths, dtype=np.float64)
    prob = prob / max(float(np.sum(prob)), 1e-12)

    n = max(1, int(n_train))
    n_traj = max(1, int(traj_count))
    t_len = max(2, int(traj_len))
    seq: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(800, 80 * n_traj)
    while len(seq) < n_traj and attempts < max_attempts:
        attempts += 1
        ti = len(seq)
        if ti < len(curves) * 5:
            piece = ti % len(curves)
        else:
            piece = int(rng.choice(len(curves), p=prob))
        curve = curves[piece]
        closed = bool(closed_flags[piece])
        base_theta = float(rng.uniform(0.0, 2.0 * math.pi))
        theta_rate = float(rng.uniform(-0.035, 0.035))
        direction = -1.0 if float(rng.uniform()) < 0.5 else 1.0
        speed = float(rng.uniform(0.45, 0.82)) / max(float(len(curve)), 1.0)
        if closed:
            u0 = float(rng.uniform(0.0, 1.0))
            u = u0 + direction * speed * np.arange(t_len, dtype=np.float32)
        else:
            span = speed * float(t_len - 1)
            if span >= 0.74:
                speed = 0.74 / float(t_len - 1)
                span = speed * float(t_len - 1)
            lo = 0.13 + max(0.0, -direction) * span
            hi = 0.87 - max(0.0, direction) * span
            if hi <= lo:
                continue
            u0 = float(rng.uniform(lo, hi))
            u = u0 + direction * speed * np.arange(t_len, dtype=np.float32)
        theta = base_theta + theta_rate * np.arange(t_len, dtype=np.float32)
        cand = _curve_tube_surface_points(curve, u, theta, radius=float(radius), closed=closed)
        sdf = _genus3_tube_sdf(cand, radius=float(radius))
        jumps = np.linalg.norm(np.diff(cand, axis=0), axis=1) if len(cand) > 1 else np.zeros((0,), dtype=np.float32)
        if float(np.max(np.abs(sdf))) > float(sdf_tol):
            continue
        if len(jumps) and float(np.max(jumps)) > 0.055:
            continue
        seq.append(cand.astype(np.float32))

    if len(seq) < n_traj:
        raise RuntimeError(
            "Could not generate enough valid genus3 trajectory demonstrations; "
            f"accepted={len(seq)} requested={n_traj} attempts={attempts}. "
            "Try increasing genus3_traj_sdf_tol or reducing traj_count."
        )

    arr = np.concatenate(seq, axis=0).astype(np.float32) if seq else np.zeros((0, 3), dtype=np.float32)
    if len(arr) >= n:
        return arr[:n].astype(np.float32)
    rep = int(math.ceil(float(n) / float(max(len(arr), 1))))
    return np.tile(arr, (rep, 1))[:n].astype(np.float32)


def _periodic_spline_eval(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    pts = np.asarray(ctrl, dtype=np.float64)
    uu = np.asarray(u, dtype=np.float64).reshape(-1)
    uu = np.mod(uu, 1.0)
    try:
        from scipy.interpolate import CubicSpline  # type: ignore

        t = np.arange(len(pts) + 1, dtype=np.float64)
        spl = CubicSpline(t, np.vstack([pts, pts[0]]), axis=0, bc_type="periodic")
        out = spl(uu * float(len(pts)))
        return out.astype(np.float32)
    except Exception:
        tt = uu * float(len(pts))
        i = np.floor(tt).astype(np.int64) % len(pts)
        a = (tt - np.floor(tt))[:, None]
        out = (1.0 - a) * pts[i] + a * pts[(i + 1) % len(pts)]
        return out.astype(np.float32)


def _knot_spline_union_trefoil(u: np.ndarray) -> np.ndarray:
    tt = 2.0 * np.pi * np.asarray(u, dtype=np.float64).reshape(-1)
    pts = np.stack(
        [
            np.sin(tt) + 2.0 * np.sin(2.0 * tt),
            np.cos(tt) - 2.0 * np.cos(2.0 * tt),
            -np.sin(3.0 * tt),
        ],
        axis=1,
    ).astype(np.float64)
    pts = pts / 3.0 * 0.92
    pts[:, 0] -= 0.18
    pts[:, 1] += 0.02
    return pts.astype(np.float32)


def _knot_spline_union_ctrl() -> np.ndarray:
    return np.asarray(
        [
            [-0.6832485175, -0.1869631121, 1.2338364248],
            [-0.9055007283, 0.0333952859, 1.0760709951],
            [-0.9466254902, 0.2952325059, 0.9766133287],
            [-1.0291257419, 0.5422489407, 0.5551845233],
            [-0.7406820360, 0.6787981286, 0.2546298765],
            [-0.6309173010, 0.7024205460, -0.0928510093],
            [-0.2880547156, 0.6909751395, -0.3813481965],
            [-0.1033013595, 0.4739314522, -0.3482443787],
            [0.0851252607, 0.2413390683, -0.3602270332],
            [0.2360337251, 0.0190282391, -0.0533962052],
            [0.1222081770, -0.2150753185, 0.1924455473],
            [0.1199479769, -0.3802434174, 0.6037724316],
            [-0.2131649105, -0.4454541061, 0.9432002366],
            [-0.3692044633, -0.3092958388, 1.1295074523],
        ],
        dtype=np.float32,
    )


def _knot_spline_union_curve(component: int, u: np.ndarray) -> np.ndarray:
    if int(component) == 0:
        return _knot_spline_union_trefoil(u)
    return _periodic_spline_eval(_knot_spline_union_ctrl(), u)


def _closed_polyline_eval(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    pts = np.asarray(ctrl, dtype=np.float32)
    uu = np.mod(np.asarray(u, dtype=np.float64).reshape(-1), 1.0)
    tt = uu * float(len(pts))
    idx = np.floor(tt).astype(np.int64) % len(pts)
    alpha = (tt - np.floor(tt))[:, None].astype(np.float32)
    out = (1.0 - alpha) * pts[idx] + alpha * pts[(idx + 1) % len(pts)]
    return out.astype(np.float32)


def _knot_spline_triplet_intersect_ctrl() -> np.ndarray:
    red_ctrl = _knot_spline_union_ctrl()
    a = _periodic_spline_eval(red_ctrl, np.asarray([0.875], dtype=np.float32))[0]
    b = _periodic_spline_eval(red_ctrl, np.asarray([0.955], dtype=np.float32))[0]
    mid = 0.5 * (a + b)
    d1 = np.asarray([0.46, -0.18, 0.18], dtype=np.float32)
    d2 = np.asarray([0.12, 0.42, -0.10], dtype=np.float32)
    d3 = np.asarray([-0.34, 0.20, -0.20], dtype=np.float32)
    d4 = np.asarray([-0.18, -0.36, 0.14], dtype=np.float32)
    ctrl = np.vstack([a, mid + d1, b, mid + d2, mid + d3, mid + d4]).astype(np.float32)
    for i in (1, 3, 4, 5):
        ctrl[i, 0] += 0.10
        ctrl[i, 1] -= 0.03
    return ctrl.astype(np.float32)


def _red_curve_normal_at(u: float) -> np.ndarray:
    red_ctrl = _knot_spline_union_ctrl()
    eps = 1e-3
    p_prev = _periodic_spline_eval(red_ctrl, np.asarray([u - eps], dtype=np.float32))[0]
    p_next = _periodic_spline_eval(red_ctrl, np.asarray([u + eps], dtype=np.float32))[0]
    tangent = p_next - p_prev
    tangent /= np.linalg.norm(tangent) + 1e-8
    ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    normal = np.cross(tangent, ref)
    if float(np.linalg.norm(normal)) < 1e-5:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        normal = np.cross(tangent, ref)
    normal /= np.linalg.norm(normal) + 1e-8
    return normal.astype(np.float32)


def _knot_spline_triplet_near_intersect_ctrl() -> np.ndarray:
    ctrl = _knot_spline_triplet_intersect_ctrl().copy()
    gap = 0.12
    push = 0.28
    n0 = _red_curve_normal_at(0.875)
    n2 = _red_curve_normal_at(0.955)
    ctrl[0] = ctrl[0] + gap * n0
    ctrl[2] = ctrl[2] - gap * n2
    ctrl[1] = ctrl[1] + push * n0
    ctrl[3] = ctrl[3] - push * n2
    ctrl[5] = ctrl[5] + 0.5 * push * n0
    ctrl = ctrl + np.asarray([0.08, -0.24, 0.28], dtype=np.float32)[None, :]
    return ctrl.astype(np.float32)


def _knot_spline_triplet_intersect_curve(component: int, u: np.ndarray) -> np.ndarray:
    comp = int(component)
    if comp == 0:
        return _knot_spline_union_trefoil(u)
    if comp == 1:
        return _periodic_spline_eval(_knot_spline_union_ctrl(), u)
    return _closed_polyline_eval(_knot_spline_triplet_intersect_ctrl(), u)


def _knot_spline_triplet_near_intersect_curve(component: int, u: np.ndarray) -> np.ndarray:
    comp = int(component)
    if comp == 0:
        return _knot_spline_union_trefoil(u)
    if comp == 1:
        return _periodic_spline_eval(_knot_spline_union_ctrl(), u)
    return _closed_polyline_eval(_knot_spline_triplet_near_intersect_ctrl(), u)


def _knot_spline_triplet_simple_params(variant: str) -> tuple[np.ndarray, float, float, np.ndarray]:
    v = str(variant or "right_small_tilt")
    if v == "right_mid_tilt":
        center = np.asarray([1.05, -0.55, 0.65], dtype=np.float32)
        a, b = 0.48, 0.30
        frame = (_rotation_z(math.radians(18.0)) @ _rotation_y(math.radians(35.0))).astype(np.float32)
    elif v == "upper_right":
        center = np.asarray([0.85, 0.55, 0.95], dtype=np.float32)
        a, b = 0.36, 0.24
        frame = (_rotation_z(math.radians(-25.0)) @ _rotation_y(math.radians(-30.0))).astype(np.float32)
    elif v == "far_right":
        center = np.asarray([1.25, -0.35, 0.45], dtype=np.float32)
        a, b = 0.42, 0.27
        frame = (_rotation_z(math.radians(18.0)) @ _rotation_y(math.radians(35.0))).astype(np.float32)
    elif v == "above_red":
        center = np.asarray([0.20, 0.15, 1.45], dtype=np.float32)
        a, b = 0.36, 0.24
        frame = np.eye(3, dtype=np.float32)
    elif v == "right_flat_small":
        center = np.asarray([1.15, -0.62, 0.42], dtype=np.float32)
        a, b = 0.30, 0.18
        frame = np.eye(3, dtype=np.float32)
    elif v == "right_flat_mid":
        center = np.asarray([1.18, -0.58, 0.50], dtype=np.float32)
        a, b = 0.34, 0.20
        frame = np.eye(3, dtype=np.float32)
    elif v == "right_low_flat":
        center = np.asarray([1.05, -0.78, 0.22], dtype=np.float32)
        a, b = 0.32, 0.18
        frame = np.eye(3, dtype=np.float32)
    elif v == "right_xz_small":
        center = np.asarray([1.14, -0.62, 0.45], dtype=np.float32)
        a, b = 0.30, 0.18
        frame = _rotation_x(math.radians(90.0)).astype(np.float32)
    elif v == "right_xz_tiny":
        center = np.asarray([1.14, -0.62, 0.45], dtype=np.float32)
        a, b = 0.25, 0.15
        frame = _rotation_x(math.radians(90.0)).astype(np.float32)
    elif v == "right_xz_far":
        center = np.asarray([1.28, -0.62, 0.45], dtype=np.float32)
        a, b = 0.28, 0.17
        frame = _rotation_x(math.radians(90.0)).astype(np.float32)
    elif v == "right_xz_back":
        center = np.asarray([1.14, -0.78, 0.45], dtype=np.float32)
        a, b = 0.28, 0.17
        frame = _rotation_x(math.radians(90.0)).astype(np.float32)
    elif v == "right_gentle_tilt_small":
        center = np.asarray([1.12, -0.60, 0.44], dtype=np.float32)
        a, b = 0.30, 0.18
        frame = (_rotation_z(math.radians(8.0)) @ _rotation_y(math.radians(15.0))).astype(np.float32)
    else:
        center = np.asarray([0.95, -0.55, 0.55], dtype=np.float32)
        a, b = 0.38, 0.24
        frame = (_rotation_z(math.radians(18.0)) @ _rotation_y(math.radians(35.0))).astype(np.float32)
    return center, float(a), float(b), frame


def _knot_spline_triplet_simple_curve(component: int, u: np.ndarray, variant: str = "right_small_tilt") -> np.ndarray:
    comp = int(component)
    if comp == 0:
        return _knot_spline_union_trefoil(u)
    if comp == 1:
        return _periodic_spline_eval(_knot_spline_union_ctrl(), u)
    uu = np.asarray(u, dtype=np.float32).reshape(-1)
    th = (2.0 * math.pi * uu).astype(np.float32)
    center, a, b, frame = _knot_spline_triplet_simple_params(variant)
    local = np.stack(
        [
            a * np.cos(th),
            b * np.sin(th),
            np.zeros_like(th, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return (local @ frame.T + center[None, :]).astype(np.float32)


def _mask_periodic_gaps(u: np.ndarray, gaps: tuple[tuple[float, float], ...]) -> np.ndarray:
    uu = np.mod(np.asarray(u, dtype=np.float64).reshape(-1), 1.0)
    keep = np.ones_like(uu, dtype=bool)
    for center, width in gaps:
        c = float(center) % 1.0
        half = 0.5 * max(0.0, float(width))
        d = np.abs(((uu - c + 0.5) % 1.0) - 0.5)
        keep &= d > half
    return keep


def _sample_component_demo_u(
    n: int,
    *,
    component: int,
    cfg,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(component) == 0:
        base_gaps = ((0.16, 0.035), (0.43, 0.030), (0.72, 0.032))
    elif int(component) == 1:
        base_gaps = ((0.22, 0.030), (0.55, 0.035), (0.82, 0.030))
    else:
        base_gaps = ((0.12, 0.030), (0.47, 0.035), (0.76, 0.030))
    gap_scale = max(0.0, float(getattr(cfg, "knot_spline_gap_scale", 1.0)))
    gaps = tuple((c, w * gap_scale) for c, w in base_gaps)

    bank_n = max(4096, 24 * max(1, int(n)))
    phase = float(rng.uniform(0.0, 1.0 / float(bank_n)))
    u_bank = np.mod(np.linspace(0.0, 1.0, bank_n, endpoint=False) + phase, 1.0)
    u_bank = np.sort(u_bank[_mask_periodic_gaps(u_bank, gaps)])
    if len(u_bank) == 0:
        return np.linspace(0.0, 1.0, max(1, int(n)), endpoint=False, dtype=np.float32)

    count = max(1, int(n))
    idx = np.rint(np.linspace(0, len(u_bank) - 1, count)).astype(np.int64)
    return u_bank[idx].astype(np.float32)


def _knot_spline_union_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(cfg.n_train))
    n_grid = max(2, int(cfg.n_grid))
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 3907)

    n0 = n_train // 2
    n1 = n_train - n0
    u0 = _sample_component_demo_u(n0, component=0, cfg=cfg, rng=rng)
    u1 = _sample_component_demo_u(n1, component=1, cfg=cfg, rng=rng)
    x_train = np.concatenate(
        [
            _knot_spline_union_curve(0, u0),
            _knot_spline_union_curve(1, u1),
        ],
        axis=0,
    ).astype(np.float32)

    g0 = n_grid // 2
    g1 = n_grid - g0
    ug0 = np.linspace(0.0, 1.0, max(1, g0), endpoint=False, dtype=np.float32)
    ug1 = np.linspace(0.0, 1.0, max(1, g1), endpoint=False, dtype=np.float32)
    grid = np.concatenate(
        [
            _knot_spline_union_curve(0, ug0),
            _knot_spline_union_curve(1, ug1),
        ],
        axis=0,
    ).astype(np.float32)
    return x_train, grid


def _split_counts(total: int, count: int) -> list[int]:
    n = max(1, int(count))
    total = max(1, int(total))
    base = total // n
    rem = total % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _component_gap_spec(component: int, cfg) -> tuple[tuple[float, float], ...]:
    if int(component) == 0:
        base_gaps = ((0.16, 0.035), (0.43, 0.030), (0.72, 0.032))
    elif int(component) == 1:
        base_gaps = ((0.22, 0.030), (0.55, 0.035), (0.82, 0.030))
    else:
        base_gaps = ((0.12, 0.030), (0.47, 0.035), (0.76, 0.030))
    gap_scale = max(0.0, float(getattr(cfg, "knot_spline_gap_scale", 1.0)))
    return tuple((c, w * gap_scale) for c, w in base_gaps)


def _allowed_u_segments(
    *,
    component: int,
    cfg,
    bank_n: int,
    phase: float,
) -> list[np.ndarray]:
    u_bank = np.mod(np.linspace(0.0, 1.0, int(bank_n), endpoint=False) + float(phase), 1.0)
    u_bank = np.sort(u_bank[_mask_periodic_gaps(u_bank, _component_gap_spec(component, cfg))])
    if len(u_bank) == 0:
        return [np.linspace(0.0, 1.0, max(2, int(bank_n)), endpoint=False, dtype=np.float32)]
    jumps = np.flatnonzero(np.diff(u_bank) > (2.5 / float(max(2, int(bank_n))))) + 1
    segments = [seg.astype(np.float32) for seg in np.split(u_bank, jumps) if len(seg) > 0]
    return segments or [u_bank.astype(np.float32)]


def _sample_component_demo_traj_u(
    *,
    component: int,
    traj_lengths: list[int],
    cfg,
    rng: np.random.Generator,
) -> np.ndarray:
    if not traj_lengths:
        return np.zeros((0,), dtype=np.float32)
    max_len = max(2, max(int(v) for v in traj_lengths))
    bank_n = max(4096, 96 * max_len * max(1, len(traj_lengths)))
    phase = float(rng.uniform(0.0, 1.0 / float(bank_n)))
    segments = _allowed_u_segments(component=component, cfg=cfg, bank_n=bank_n, phase=phase)

    chunks: list[np.ndarray] = []
    seg_assignments = [j % len(segments) for j in range(len(traj_lengths))]
    seg_counts = [seg_assignments.count(i) for i in range(len(segments))]
    seg_seen = [0 for _ in range(len(segments))]
    for j, length in enumerate(traj_lengths):
        length = max(1, int(length))
        seg_idx = seg_assignments[j]
        seg = segments[seg_idx]
        count_in_seg = max(1, seg_counts[seg_idx])
        rank_in_seg = seg_seen[seg_idx]
        seg_seen[seg_idx] += 1

        if len(seg) <= length:
            idx = np.rint(np.linspace(0, len(seg) - 1, length)).astype(np.int64)
        else:
            # Each trajectory is a contiguous parameter-space arc.  Multiple
            # trajectories assigned to the same allowed arc tile most of it while
            # leaving small unobserved intervals between demos.
            margin = 0.06 / float(count_in_seg)
            lo_frac = (rank_in_seg + margin) / float(count_in_seg)
            hi_frac = (rank_in_seg + 1.0 - margin) / float(count_in_seg)
            jitter = float(rng.uniform(-0.015, 0.015)) / float(count_in_seg)
            lo_frac = min(max(lo_frac + jitter, 0.0), 1.0)
            hi_frac = min(max(hi_frac + jitter, lo_frac), 1.0)
            lo = int(round(lo_frac * float(len(seg) - 1)))
            hi = int(round(hi_frac * float(len(seg) - 1)))
            if hi <= lo:
                hi = min(len(seg) - 1, lo + 1)
            idx = np.rint(np.linspace(lo, hi, length)).astype(np.int64)
        chunks.append(seg[idx].astype(np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _knot_spline_union_traj_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(cfg.n_train))
    n_grid = max(2, int(cfg.n_grid))
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 3929)

    traj_count = int(max(2, min(n_train, getattr(cfg, "traj_count", max(8, n_train // 32)))))
    comp0_traj = traj_count // 2
    comp1_traj = traj_count - comp0_traj
    n0 = n_train // 2
    n1 = n_train - n0
    u0 = _sample_component_demo_traj_u(
        component=0,
        traj_lengths=_split_counts(n0, comp0_traj),
        cfg=cfg,
        rng=rng,
    )
    u1 = _sample_component_demo_traj_u(
        component=1,
        traj_lengths=_split_counts(n1, comp1_traj),
        cfg=cfg,
        rng=rng,
    )
    x_train = np.concatenate(
        [
            _knot_spline_union_curve(0, u0),
            _knot_spline_union_curve(1, u1),
        ],
        axis=0,
    ).astype(np.float32)

    g0 = n_grid // 2
    g1 = n_grid - g0
    ug0 = np.linspace(0.0, 1.0, max(1, g0), endpoint=False, dtype=np.float32)
    ug1 = np.linspace(0.0, 1.0, max(1, g1), endpoint=False, dtype=np.float32)
    grid = np.concatenate(
        [
            _knot_spline_union_curve(0, ug0),
            _knot_spline_union_curve(1, ug1),
        ],
        axis=0,
    ).astype(np.float32)
    return x_train[:n_train], grid


def _knot_spline_triplet_intersect_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(cfg.n_train))
    n_grid = max(3, int(cfg.n_grid))
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 3919)

    counts = [n_train // 3, n_train // 3, n_train - 2 * (n_train // 3)]
    train_parts = []
    for comp, count in enumerate(counts):
        u_comp = _sample_component_demo_u(count, component=comp, cfg=cfg, rng=rng)
        train_parts.append(_knot_spline_triplet_intersect_curve(comp, u_comp))
    x_train = np.concatenate(train_parts, axis=0).astype(np.float32)

    grid_counts = [n_grid // 3, n_grid // 3, n_grid - 2 * (n_grid // 3)]
    grid_parts = []
    for comp, count in enumerate(grid_counts):
        u_comp = np.linspace(0.0, 1.0, max(1, count), endpoint=False, dtype=np.float32)
        grid_parts.append(_knot_spline_triplet_intersect_curve(comp, u_comp))
    grid = np.concatenate(grid_parts, axis=0).astype(np.float32)
    return x_train, grid


def _knot_spline_triplet_near_intersect_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(cfg.n_train))
    n_grid = max(3, int(cfg.n_grid))
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 3929)

    counts = [n_train // 3, n_train // 3, n_train - 2 * (n_train // 3)]
    train_parts = []
    for comp, count in enumerate(counts):
        u_comp = _sample_component_demo_u(count, component=comp, cfg=cfg, rng=rng)
        train_parts.append(_knot_spline_triplet_near_intersect_curve(comp, u_comp))
    x_train = np.concatenate(train_parts, axis=0).astype(np.float32)

    grid_counts = [n_grid // 3, n_grid // 3, n_grid - 2 * (n_grid // 3)]
    grid_parts = []
    for comp, count in enumerate(grid_counts):
        u_comp = np.linspace(0.0, 1.0, max(1, count), endpoint=False, dtype=np.float32)
        grid_parts.append(_knot_spline_triplet_near_intersect_curve(comp, u_comp))
    grid = np.concatenate(grid_parts, axis=0).astype(np.float32)
    return x_train, grid


def _knot_spline_triplet_simple_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(cfg.n_train))
    n_grid = max(3, int(cfg.n_grid))
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 3937)
    variant = str(getattr(cfg, "triplet_simple_variant", "right_small_tilt"))

    counts = [n_train // 3, n_train // 3, n_train - 2 * (n_train // 3)]
    train_parts = []
    for comp, count in enumerate(counts):
        u_comp = _sample_component_demo_u(count, component=comp, cfg=cfg, rng=rng)
        train_parts.append(_knot_spline_triplet_simple_curve(comp, u_comp, variant=variant))
    x_train = np.concatenate(train_parts, axis=0).astype(np.float32)

    grid_counts = [n_grid // 3, n_grid // 3, n_grid - 2 * (n_grid // 3)]
    grid_parts = []
    for comp, count in enumerate(grid_counts):
        u_comp = np.linspace(0.0, 1.0, max(1, count), endpoint=False, dtype=np.float32)
        grid_parts.append(_knot_spline_triplet_simple_curve(comp, u_comp, variant=variant))
    grid = np.concatenate(grid_parts, axis=0).astype(np.float32)
    return x_train, grid


def _intersecting_loops_ellipse(u: np.ndarray) -> np.ndarray:
    uu = np.asarray(u, dtype=np.float32).reshape(-1)
    th = (2.0 * math.pi * uu).astype(np.float32)
    return np.stack([1.05 * np.cos(th), 0.72 * np.sin(th)], axis=1).astype(np.float32)


def _intersecting_loops_poly_ctrl() -> np.ndarray:
    anchors = _intersecting_loops_ellipse(np.asarray([0.62, 0.78], dtype=np.float32))
    a = anchors[0]
    b = anchors[1]
    return np.asarray(
        [
            a,
            [-0.10, -1.02],
            [0.88, -0.78],
            b,
            [0.26, 0.48],
            [-0.74, 0.14],
        ],
        dtype=np.float32,
    )


def _intersecting_loops_curve(component: int, u: np.ndarray) -> np.ndarray:
    if int(component) == 0:
        return _intersecting_loops_ellipse(u)
    return _closed_polyline_eval(_intersecting_loops_poly_ctrl(), u)


def _sample_intersecting_loops_demo_u(
    n: int,
    *,
    component: int,
    cfg,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(component) == 0:
        base_gaps = ((0.18, 0.035), (0.52, 0.030), (0.84, 0.035))
    else:
        base_gaps = ((0.16, 0.040), (0.49, 0.035), (0.74, 0.035))
    gap_scale = max(0.0, float(getattr(cfg, "intersecting_loops_gap_scale", 1.0)))
    gaps = tuple((c, w * gap_scale) for c, w in base_gaps)

    bank_n = max(4096, 24 * max(1, int(n)))
    phase = float(rng.uniform(0.0, 1.0 / float(bank_n)))
    u_bank = np.mod(np.linspace(0.0, 1.0, bank_n, endpoint=False) + phase, 1.0)
    u_bank = np.sort(u_bank[_mask_periodic_gaps(u_bank, gaps)])
    if len(u_bank) == 0:
        return np.linspace(0.0, 1.0, max(1, int(n)), endpoint=False, dtype=np.float32)

    count = max(1, int(n))
    idx = np.rint(np.linspace(0, len(u_bank) - 1, count)).astype(np.int64)
    return u_bank[idx].astype(np.float32)


def _intersecting_loops_n2(cfg) -> Tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(cfg.n_train))
    n_grid = max(2, int(cfg.n_grid))
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 2179)

    n0 = n_train // 2
    n1 = n_train - n0
    u0 = _sample_intersecting_loops_demo_u(n0, component=0, cfg=cfg, rng=rng)
    u1 = _sample_intersecting_loops_demo_u(n1, component=1, cfg=cfg, rng=rng)
    x_train = np.concatenate(
        [
            _intersecting_loops_curve(0, u0),
            _intersecting_loops_curve(1, u1),
        ],
        axis=0,
    ).astype(np.float32)

    g0 = n_grid // 2
    g1 = n_grid - g0
    ug0 = np.linspace(0.0, 1.0, max(1, g0), endpoint=False, dtype=np.float32)
    ug1 = np.linspace(0.0, 1.0, max(1, g1), endpoint=False, dtype=np.float32)
    grid = np.concatenate(
        [
            _intersecting_loops_curve(0, ug0),
            _intersecting_loops_curve(1, ug1),
        ],
        axis=0,
    ).astype(np.float32)
    return x_train, grid


def _nested_ellipses_union_curve(component: int, u: np.ndarray) -> np.ndarray:
    uu = np.asarray(u, dtype=np.float32).reshape(-1)
    th = (2.0 * math.pi * uu).astype(np.float32)
    if int(component) == 0:
        a, b, z = 0.95, 0.62, 0.0
    else:
        a, b, z = 0.48, 0.31, 0.0
    return np.stack(
        [
            a * np.cos(th),
            b * np.sin(th),
            np.full_like(th, z, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _sample_nested_ellipse_demo_u(
    n: int,
    *,
    component: int,
    cfg,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(component) == 0:
        base_gaps = ((0.18, 0.030), (0.58, 0.035))
    else:
        base_gaps = ((0.34, 0.030), (0.76, 0.035))
    gap_scale = max(0.0, float(getattr(cfg, "nested_ellipse_gap_scale", 1.0)))
    gaps = tuple((c, w * gap_scale) for c, w in base_gaps)

    bank_n = max(4096, 24 * max(1, int(n)))
    phase = float(rng.uniform(0.0, 1.0 / float(bank_n)))
    u_bank = np.mod(np.linspace(0.0, 1.0, bank_n, endpoint=False) + phase, 1.0)
    u_bank = np.sort(u_bank[_mask_periodic_gaps(u_bank, gaps)])
    if len(u_bank) == 0:
        return np.linspace(0.0, 1.0, max(1, int(n)), endpoint=False, dtype=np.float32)

    count = max(1, int(n))
    idx = np.rint(np.linspace(0, len(u_bank) - 1, count)).astype(np.int64)
    return u_bank[idx].astype(np.float32)


def _nested_ellipses_union_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(cfg.n_train))
    n_grid = max(2, int(cfg.n_grid))
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 4201)

    n0 = n_train // 2
    n1 = n_train - n0
    u0 = _sample_nested_ellipse_demo_u(n0, component=0, cfg=cfg, rng=rng)
    u1 = _sample_nested_ellipse_demo_u(n1, component=1, cfg=cfg, rng=rng)
    x_train = np.concatenate(
        [
            _nested_ellipses_union_curve(0, u0),
            _nested_ellipses_union_curve(1, u1),
        ],
        axis=0,
    ).astype(np.float32)

    g0 = n_grid // 2
    g1 = n_grid - g0
    ug0 = np.linspace(0.0, 1.0, max(1, g0), endpoint=False, dtype=np.float32)
    ug1 = np.linspace(0.0, 1.0, max(1, g1), endpoint=False, dtype=np.float32)
    grid = np.concatenate(
        [
            _nested_ellipses_union_curve(0, ug0),
            _nested_ellipses_union_curve(1, ug1),
        ],
        axis=0,
    ).astype(np.float32)
    return x_train, grid


def _linked_ellipses_union_curve(component: int, u: np.ndarray) -> np.ndarray:
    uu = np.asarray(u, dtype=np.float32).reshape(-1)
    th = (2.0 * math.pi * uu).astype(np.float32)
    if int(component) == 0:
        # Horizontal link in the xy-plane.
        x = 0.95 * np.cos(th)
        y = 0.62 * np.sin(th)
        z = np.zeros_like(th, dtype=np.float32)
    else:
        # Vertical link in the xz-plane. It crosses the horizontal link's
        # spanning disk once inside and once outside, so the components are linked
        # without intersecting each other.
        x = 0.82 + 0.35 * np.cos(th)
        y = np.zeros_like(th, dtype=np.float32)
        z = 0.82 * np.sin(th)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _sample_linked_ellipse_demo_u(
    n: int,
    *,
    component: int,
    cfg,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(component) == 0:
        base_gaps = ((0.18, 0.030), (0.58, 0.035))
    else:
        base_gaps = ((0.32, 0.030), (0.73, 0.035))
    gap_scale = max(0.0, float(getattr(cfg, "linked_ellipse_gap_scale", 0.0)))
    gaps = tuple((c, w * gap_scale) for c, w in base_gaps)

    bank_n = max(4096, 24 * max(1, int(n)))
    phase = float(rng.uniform(0.0, 1.0 / float(bank_n)))
    u_bank = np.mod(np.linspace(0.0, 1.0, bank_n, endpoint=False) + phase, 1.0)
    u_bank = np.sort(u_bank[_mask_periodic_gaps(u_bank, gaps)])
    if len(u_bank) == 0:
        return np.linspace(0.0, 1.0, max(1, int(n)), endpoint=False, dtype=np.float32)

    count = max(1, int(n))
    idx = np.rint(np.linspace(0, len(u_bank) - 1, count)).astype(np.int64)
    return u_bank[idx].astype(np.float32)


def _linked_ellipses_union_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(cfg.n_train))
    n_grid = max(2, int(cfg.n_grid))
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 4219)

    n0 = n_train // 2
    n1 = n_train - n0
    u0 = _sample_linked_ellipse_demo_u(n0, component=0, cfg=cfg, rng=rng)
    u1 = _sample_linked_ellipse_demo_u(n1, component=1, cfg=cfg, rng=rng)
    x_train = np.concatenate(
        [
            _linked_ellipses_union_curve(0, u0),
            _linked_ellipses_union_curve(1, u1),
        ],
        axis=0,
    ).astype(np.float32)

    g0 = n_grid // 2
    g1 = n_grid - g0
    ug0 = np.linspace(0.0, 1.0, max(1, g0), endpoint=False, dtype=np.float32)
    ug1 = np.linspace(0.0, 1.0, max(1, g1), endpoint=False, dtype=np.float32)
    grid = np.concatenate(
        [
            _linked_ellipses_union_curve(0, ug0),
            _linked_ellipses_union_curve(1, ug1),
        ],
        axis=0,
    ).astype(np.float32)
    return x_train, grid


def _double_torus_surface_grid(n_grid: int) -> np.ndarray:
    major_r, tube_r, ca, cb, frame_a, frame_b = _double_torus_params()
    m = max(8, int(math.ceil(math.sqrt(max(1, int(n_grid)) / 1.75))))
    u = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False, dtype=np.float32)
    v = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False, dtype=np.float32)
    ug, vg = np.meshgrid(u, v)
    uu = ug.reshape(-1)
    vv = vg.reshape(-1)
    pa = _torus_surface_from_uv(uu, vv, center=ca, frame=frame_a, major_r=major_r, tube_r=tube_r)
    pb = _torus_surface_from_uv(uu, vv, center=cb, frame=frame_b, major_r=major_r, tube_r=tube_r)
    grid = np.concatenate(
        [
            _double_torus_outer_from_surface_points(pa, other_center=cb, other_frame=frame_b),
            _double_torus_outer_from_surface_points(pb, other_center=ca, other_frame=frame_a),
        ],
        axis=0,
    ).astype(np.float32)
    if len(grid) >= int(n_grid):
        idx = np.rint(np.linspace(0, len(grid) - 1, int(n_grid))).astype(np.int64)
        return grid[idx].astype(np.float32)
    return grid.astype(np.float32)


def _double_torus_surface_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    rng_train = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 4311)
    x_train = sample_double_torus_outer_on(int(cfg.n_train), rng_train).astype(np.float32)
    grid = _double_torus_surface_grid(int(cfg.n_grid)).astype(np.float32)
    return x_train, grid


def _warped_double_torus_surface_n4(cfg) -> Tuple[np.ndarray, np.ndarray]:
    x3, grid3 = _double_torus_surface_n3(cfg)
    return lift_double_torus_outer_to_4d(x3), lift_double_torus_outer_to_4d(grid3)


def _genus3_tube_surface_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    rng_train = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 5813)
    rng_grid = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 5814)
    verts, faces = _genus3_tube_mesh(
        nx=int(getattr(cfg, "genus3_mesh_nx", 112)),
        ny=int(getattr(cfg, "genus3_mesh_ny", 88)),
        nz=int(getattr(cfg, "genus3_mesh_nz", 88)),
        radius=float(getattr(cfg, "genus3_tube_radius", 0.165)),
    )
    x_train = _sample_mesh_surface(verts, faces, int(cfg.n_train), rng_train).astype(np.float32)
    grid = _sample_mesh_surface(verts, faces, int(cfg.n_grid), rng_grid).astype(np.float32)
    return x_train, grid


def _genus3_tube_surface_traj_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    rng_grid = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 5814)
    verts, faces = _genus3_tube_mesh(
        nx=int(getattr(cfg, "genus3_mesh_nx", 112)),
        ny=int(getattr(cfg, "genus3_mesh_ny", 88)),
        nz=int(getattr(cfg, "genus3_mesh_nz", 88)),
        radius=float(getattr(cfg, "genus3_tube_radius", 0.165)),
    )
    n_train = max(1, int(getattr(cfg, "n_train", 1)))
    traj_count_raw = int(getattr(cfg, "traj_count", max(12, n_train // 24)))
    traj_count = int(max(1, min(n_train, traj_count_raw)))
    traj_len = int(max(8, getattr(cfg, "traj_len", int(math.ceil(n_train / max(traj_count, 1))))))
    x_train = _genus3_tube_param_trajectory_samples(
        n_train=n_train,
        seed=int(getattr(cfg, "seed", 0)),
        traj_count=traj_count,
        traj_len=traj_len,
        radius=float(getattr(cfg, "genus3_tube_radius", 0.165)),
        sdf_tol=float(getattr(cfg, "genus3_traj_sdf_tol", 0.018)),
    )
    grid = _sample_mesh_surface(verts, faces, int(cfg.n_grid), rng_grid).astype(np.float32)
    return x_train.astype(np.float32), grid


def _faceted_sphere_surface_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    rng_train = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 6101)
    rng_grid = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 6102)
    verts, faces = _faceted_sphere_mesh(
        n_lat=int(getattr(cfg, "faceted_sphere_n_lat", 18)),
        n_lon=int(getattr(cfg, "faceted_sphere_n_lon", 36)),
        bump_scale=float(getattr(cfg, "faceted_sphere_bump_scale", 0.72)),
        sigma=float(getattr(cfg, "faceted_sphere_sigma", 0.20)),
        sharpness=float(getattr(cfg, "faceted_sphere_sharpness", 3.2)),
    )
    x_train = _sample_mesh_surface(verts, faces, int(cfg.n_train), rng_train).astype(np.float32)
    grid = _sample_mesh_surface(verts, faces, int(cfg.n_grid), rng_grid).astype(np.float32)
    return x_train, grid


def _spiky_sphere_surface_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    rng_train = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 6201)
    rng_grid = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 6202)
    verts, faces = _spiky_sphere_mesh(
        n_lat=int(getattr(cfg, "spiky_sphere_n_lat", 24)),
        n_lon=int(getattr(cfg, "spiky_sphere_n_lon", 48)),
        spike_scale=float(getattr(cfg, "spiky_sphere_spike_scale", 0.86)),
        sigma=float(getattr(cfg, "spiky_sphere_sigma", 0.105)),
        sharpness=float(getattr(cfg, "spiky_sphere_sharpness", 8.0)),
        n_ctrl=int(getattr(cfg, "spiky_sphere_n_ctrl", 58)),
    )
    x_train = _sample_mesh_surface(verts, faces, int(cfg.n_train), rng_train).astype(np.float32)
    grid = _sample_mesh_surface(verts, faces, int(cfg.n_grid), rng_grid).astype(np.float32)
    return x_train, grid


def _double_torus_extra_handle_surface_n3(cfg) -> Tuple[np.ndarray, np.ndarray]:
    rng_train = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 5911)
    rng_grid = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 5912)
    verts, faces = _double_torus_extra_handle_mesh(
        nx=int(getattr(cfg, "extra_handle_mesh_nx", 112)),
        ny=int(getattr(cfg, "extra_handle_mesh_ny", 96)),
        nz=int(getattr(cfg, "extra_handle_mesh_nz", 88)),
        bridge_radius=float(getattr(cfg, "extra_handle_bridge_radius", 0.18)),
    )
    x_train = _sample_mesh_surface(verts, faces, int(cfg.n_train), rng_train).astype(np.float32)
    grid = _sample_mesh_surface(verts, faces, int(cfg.n_grid), rng_grid).astype(np.float32)
    return x_train, grid


def _warped_double_torus_extra_handle_surface_n4(cfg) -> Tuple[np.ndarray, np.ndarray]:
    x3, grid3 = _double_torus_extra_handle_surface_n3(cfg)
    return lift_double_torus_extra_handle_to_4d(x3), lift_double_torus_extra_handle_to_4d(grid3)


def _warped_double_torus_extra_handle_surface_traj_n4(cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg_3d = SimpleNamespace(**vars(cfg))
    x3, grid3 = generate_dataset("3d_double_torus_extra_handle_surface_traj", cfg_3d)
    return lift_double_torus_extra_handle_to_4d(x3), lift_double_torus_extra_handle_to_4d(grid3)


def generate_dataset(name: str, cfg) -> Tuple[np.ndarray, np.ndarray]:
    if name.endswith("_traj"):
        base = str(name)[: -len("_traj")]
        if base == "12d_dual_arm":
            cfg_base = SimpleNamespace(**vars(cfg))
            n_train = max(1, int(getattr(cfg, "n_train", 1)))
            cfg_base.n_grid = int(
                max(
                    int(getattr(cfg, "n_grid", 1)),
                    int(getattr(cfg, "traj_grid_min", 1)),
                )
            )
            _, grid = generate_dataset(base, cfg_base)
            traj_count_raw = int(getattr(cfg, "traj_count", max(16, n_train // 64)))
            traj_count = int(max(1, min(n_train, traj_count_raw)))
            traj_len = int(max(8, getattr(cfg, "traj_len", int(math.ceil(n_train / max(traj_count, 1))))))
            x_traj = _dual_arm_demo_trajectory_samples(
                cfg,
                n_train=n_train,
                traj_count=traj_count,
                traj_len=traj_len,
            )
            return x_traj.astype(np.float32), grid.astype(np.float32)
        if base == "3d_genus3_tube_surface":
            cfg_base = SimpleNamespace(**vars(cfg))
            cfg_base.n_grid = int(
                max(
                    int(getattr(cfg, "n_grid", 1)),
                    int(getattr(cfg, "traj_grid_min", 1)),
                )
            )
            return _genus3_tube_surface_traj_n3(cfg_base)
        if base in TRAJ_3D_CODIM1_BASES:
            cfg_base = SimpleNamespace(**vars(cfg))
            n_train = max(1, int(getattr(cfg, "n_train", 1)))
            cfg_base.n_grid = int(
                max(
                    int(getattr(cfg, "n_grid", 1)),
                    int(getattr(cfg, "traj_grid_min", 1)),
                )
            )
            # Keep dense reference manifold for eval; train set becomes trajectory samples.
            x_base, grid = generate_dataset(base, cfg_base)
            n_train = max(1, int(getattr(cfg, "n_train", len(x_base))))
            traj_count_raw = int(getattr(cfg, "traj_count", max(12, n_train // 24)))
            traj_count = int(max(1, min(n_train, traj_count_raw)))
            traj_len = int(max(8, getattr(cfg, "traj_len", int(math.ceil(n_train / max(traj_count, 1))))))
            traj_knn = int(max(4, getattr(cfg, "traj_knn", 16)))
            x_traj = _traj_points_from_grid(
                grid.astype(np.float32),
                n_train=n_train,
                seed=int(getattr(cfg, "seed", 0)),
                traj_count=traj_count,
                traj_len=traj_len,
                traj_knn=traj_knn,
                diverse_starts=(
                    base
                    in (
                        "3d_spatial_arm_ellip_n3",
                        "3d_double_torus_surface",
                        "3d_double_torus_extra_handle_surface",
                    )
                ),
            )
            return x_traj.astype(np.float32), grid.astype(np.float32)

    if name == "3d_spiral":
        rng_train = np.random.default_rng(int(cfg.seed))
        rng_grid = np.random.default_rng(int(cfg.seed) + 1)
        x = sample_spiral_on(int(cfg.n_train), rng_train).astype(np.float32)
        grid = sample_spiral_on(int(cfg.n_grid), rng_grid).astype(np.float32)
        return x, grid
    if name == "3d_knot_spline_union":
        return _knot_spline_union_n3(cfg)
    if name in ("linked_curves_3d", "3d_knot_spline_union_traj"):
        return _knot_spline_union_traj_n3(cfg)
    if name == "3d_knot_spline_triplet_intersect":
        return _knot_spline_triplet_intersect_n3(cfg)
    if name == "3d_knot_spline_triplet_near_intersect":
        return _knot_spline_triplet_near_intersect_n3(cfg)
    if name in ("3d_knot_spline_triplet", "3d_knot_spline_triplet_simple"):
        return _knot_spline_triplet_simple_n3(cfg)
    if name == "3d_nested_ellipses_union":
        return _nested_ellipses_union_n3(cfg)
    if name == "3d_linked_ellipses_union":
        return _linked_ellipses_union_n3(cfg)
    if name == "3d_double_torus_surface":
        return _double_torus_surface_n3(cfg)
    if name == "3d_double_torus_extra_handle_surface":
        return _double_torus_extra_handle_surface_n3(cfg)
    if name == "3d_genus3_tube_surface":
        return _genus3_tube_surface_n3(cfg)
    if name == "3d_genus3_tube_surface_traj":
        return _genus3_tube_surface_traj_n3(cfg)
    if name == "3d_faceted_sphere_surface":
        return _faceted_sphere_surface_n3(cfg)
    if name == "3d_spiky_sphere_surface":
        return _spiky_sphere_surface_n3(cfg)
    if name == "4d_warped_double_torus_surface":
        return _warped_double_torus_surface_n4(cfg)
    if name == "4d_warped_double_torus_extra_handle_surface":
        return _warped_double_torus_extra_handle_surface_n4(cfg)
    if name in ("multi_handle_4d", "4d_warped_double_torus_extra_handle_surface_traj"):
        return _warped_double_torus_extra_handle_surface_traj_n4(cfg)
    if name == "3d_paraboloid":
        rng_train = np.random.default_rng(int(cfg.seed))
        rng_grid = np.random.default_rng(int(cfg.seed) + 1)
        x = sample_paraboloid_on(int(cfg.n_train), rng_train, xy_range=1.2, z_scale=1.0).astype(np.float32)
        grid = sample_paraboloid_on(int(cfg.n_grid), rng_grid, xy_range=1.2, z_scale=1.0).astype(np.float32)
        return x, grid
    if name == "3d_twosphere":
        rng_train = np.random.default_rng(int(cfg.seed))
        rng_grid = np.random.default_rng(int(cfg.seed) + 1)
        x = sample_two_sphere_outer_on(int(cfg.n_train), rng_train).astype(np.float32)
        grid = sample_two_sphere_outer_on(int(cfg.n_grid), rng_grid).astype(np.float32)
        return x, grid
    if name == "2d_square":
        rng_train = np.random.default_rng(int(cfg.seed))
        rng_grid = np.random.default_rng(int(cfg.seed) + 1)
        x = sample_square_on(int(cfg.n_train), rng_train, half=1.0).astype(np.float32)
        grid = sample_square_on(int(cfg.n_grid), rng_grid, half=1.0).astype(np.float32)
        return x, grid
    if name == "2d_two_horizontal_lines":
        rng_train = np.random.default_rng(int(cfg.seed))
        n_train = int(cfg.n_train)
        n_grid = int(cfg.n_grid)
        x = sample_two_horizontal_lines_on(n_train, rng_train, x_half_width=2.0, y_abs=1.0).astype(np.float32)
        gx = np.linspace(-2.0, 2.0, max(2, n_grid // 2), dtype=np.float32)
        grid = np.concatenate(
            [
                np.stack([gx, np.ones_like(gx)], axis=1),
                np.stack([gx, -np.ones_like(gx)], axis=1),
            ],
            axis=0,
        ).astype(np.float32)
        return x, grid
    if name == "2d_figure_eight":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        x = np.concatenate([np.sin(t), np.sin(2 * t)], axis=1).astype(np.float32)
        tg = np.linspace(-math.pi, math.pi, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([np.sin(tg), np.sin(2 * tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_intersecting_loops":
        return _intersecting_loops_n2(cfg)
    if name == "2d_ellipse":
        t = np.random.uniform(0.0, 2.0 * math.pi, size=(cfg.n_train, 1))
        a, b = 2.0, 1.0
        x = np.concatenate([a * np.cos(t), b * np.sin(t)], axis=1).astype(np.float32)
        hole_centers = np.array([0.6, 2.2, 4.0], dtype=np.float32)
        hole_width = 0.25 * 1.15
        mask = np.ones(len(t), dtype=bool)
        for c in hole_centers:
            mask &= np.abs(t.squeeze() - c) > hole_width
        x = x[mask]
        tg = np.linspace(0.0, 2.0 * math.pi, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([a * np.cos(tg), b * np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "3d_saddle_surface":
        n = cfg.n_train
        xy = np.random.uniform(-1.5, 1.5, size=(n, 2))
        z = 0.3 * (xy[:, 0:1] ** 2 - xy[:, 1:2] ** 2)
        x = np.concatenate([xy, z], axis=1).astype(np.float32)
        m = int(math.sqrt(cfg.n_grid))
        gx = np.linspace(-1.5, 1.5, m)
        gy = np.linspace(-1.5, 1.5, m)
        gx, gy = np.meshgrid(gx, gy)
        gz = 0.3 * (gx ** 2 - gy ** 2)
        grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
        return x, grid
    if name == "3d_sphere_surface":
        n = cfg.n_train
        u = np.random.uniform(0.0, 2.0 * math.pi, size=(n, 1))
        v = np.random.uniform(0.0, math.pi, size=(n, 1))
        r = 1.5
        x = np.concatenate(
            [
                r * np.cos(u) * np.sin(v),
                r * np.sin(u) * np.sin(v),
                r * np.cos(v),
            ],
            axis=1,
        ).astype(np.float32)
        m = int(math.sqrt(cfg.n_grid))
        ug = np.linspace(0.0, 2.0 * math.pi, m)
        vg = np.linspace(0.0, math.pi, m)
        ug, vg = np.meshgrid(ug, vg)
        grid = np.stack(
            [
                r * np.cos(ug) * np.sin(vg),
                r * np.sin(ug) * np.sin(vg),
                r * np.cos(vg),
            ],
            axis=2,
        ).reshape(-1, 3).astype(np.float32)
        return x, grid
    if name == "3d_torus_surface":
        n = cfg.n_train
        u = np.random.uniform(0.0, 2.0 * math.pi, size=(n, 1))
        v = np.random.uniform(0.0, 2.0 * math.pi, size=(n, 1))
        R, r = 2.0, 0.6
        x = np.concatenate(
            [
                (R + r * np.cos(v)) * np.cos(u),
                (R + r * np.cos(v)) * np.sin(u),
                r * np.sin(v),
            ],
            axis=1,
        ).astype(np.float32)
        m = int(math.sqrt(cfg.n_grid))
        ug = np.linspace(0.0, 2.0 * math.pi, m)
        vg = np.linspace(0.0, 2.0 * math.pi, m)
        ug, vg = np.meshgrid(ug, vg)
        grid = np.stack(
            [
                (R + r * np.cos(vg)) * np.cos(ug),
                (R + r * np.cos(vg)) * np.sin(ug),
                r * np.sin(vg),
            ],
            axis=2,
        ).reshape(-1, 3).astype(np.float32)
        return x, grid
    if name == "2d_planar_arm_line_n2":
        return _planar_arm_line_n2(cfg)
    if name == "3d_planar_arm_line_n3":
        return _planar_arm_line_n3(cfg)
    if name == "3d_spatial_arm_plane_n3":
        return _spatial_arm_plane_n3(cfg)
    if name in ("3d_spatial_arm_ellip_n3", "3d_spatial_arm_circle_n3"):
        return _spatial_arm_ellip_n3(cfg)
    if name == "6d_spatial_arm_up_n6":
        return _spatial_arm_up_n6(cfg)
    if name == "6d_surf_normal":
        return _surf_normal_n6(cfg)
    if name == "6d_surf_normal_traj":
        return _surf_normal_traj_n6(cfg)
    if name == "12d_dual_arm":
        return _dual_arm_pose_yoffset_samex_plane_sameori_n12(cfg)
    if name == "2d_noisy_sine":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        y = np.sin(t) + 0.1 * np.random.randn(cfg.n_train, 1)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_sine":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_sparse_sine":
        n_sparse = max(45, cfg.n_train // 8)
        t = np.random.uniform(-math.pi, math.pi, size=(n_sparse, 1))
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_discontinuous":
        n_half = cfg.n_train // 2
        t1 = np.random.uniform(-math.pi, -0.7, size=(n_half, 1))
        t2 = np.random.uniform(0.7, math.pi, size=(cfg.n_train - n_half, 1))
        t = np.vstack([t1, t2])
        y = np.sin(t)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_looped_spiro":
        t = np.random.uniform(0.0, 2.0 * math.pi, size=(cfg.n_train, 1))
        petals = 4.0
        base = 1.2
        amp = 0.35
        r = base + amp * np.cos(petals * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        x = np.concatenate([x, y], axis=1).astype(np.float32)
        tg = np.linspace(0.0, 2.0 * math.pi, cfg.n_grid).reshape(-1, 1)
        rg = base + amp * np.cos(petals * tg)
        gx = rg * np.cos(tg)
        gy = rg * np.sin(tg)
        grid = np.concatenate([gx, gy], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_sharp_star":
        t = np.random.uniform(0.0, 2.0 * math.pi, size=(cfg.n_train, 1))
        scale = 2.0
        r = 1.0 + 0.35 * np.cos(5 * t) + 0.15 * np.cos(10 * t)
        x = scale * r * np.cos(t)
        y = scale * r * np.sin(t)
        x = np.concatenate([x, y], axis=1).astype(np.float32)
        tg = np.linspace(0.0, 2.0 * math.pi, cfg.n_grid).reshape(-1, 1)
        rg = 1.0 + 0.35 * np.cos(5 * tg) + 0.15 * np.cos(10 * tg)
        gx = scale * rg * np.cos(tg)
        gy = scale * rg * np.sin(tg)
        grid = np.concatenate([gx, gy], axis=1).astype(np.float32)
        return x, grid
    if name == "2d_hetero_noise":
        t = np.random.uniform(-math.pi, math.pi, size=(cfg.n_train, 1))
        sigma = 0.02 + 0.3 * np.exp(-0.5 * (t / 0.8) ** 2)
        y = np.sin(t) + sigma * np.random.randn(cfg.n_train, 1)
        x = np.concatenate([t, y], axis=1).astype(np.float32)
        tg = np.linspace(-4.0, 4.0, cfg.n_grid).reshape(-1, 1)
        grid = np.concatenate([tg, np.sin(tg)], axis=1).astype(np.float32)
        return x, grid
    raise ValueError(f"unknown dataset: {name}")
