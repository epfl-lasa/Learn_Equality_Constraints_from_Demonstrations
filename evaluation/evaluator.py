from __future__ import annotations

from typing import Any, Callable
from types import SimpleNamespace
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from datasets.fixed_demonstrations import FIXED_DEMONSTRATION_DIMS, load_fixed_demonstrations
from datasets.constraint_datasets import (
    generate_dataset,
    lift_xy_to_3d_var,
    lift_xy_to_3d_zero,
    sine_surface_z_and_normal_from_xy,
)
from datasets.real_iiwa.registry import (
    REAL_IIWA_DATASETS,
    canonical_real_iiwa_dataset_name,
    plane_constraint,
    real_iiwa_coordinate_space,
    is_real_iiwa_dataset,
    real_iiwa_neural_surface_z_normal,
    real_iiwa_normalized_to_world_xyz,
    real_iiwa_orientation_representation,
    real_iiwa_surface_distance,
    real_iiwa_surface_grid,
    real_iiwa_surface_type,
)

DEFAULT_EVAL_CFG: dict[str, Any] = {
    "device": "cpu",
    "eval_pad_ratio": 0.6,
    "eval_min_axis_span_ratio": 0.08,
    "zero_eps_quantile": 90.0,
    "eval_proj_n_points": 4096,
    "eval_gt_n_grid": 8128,
    "eval_chamfer_near_ratio": 0.75,
    "eval_chamfer_near_source": "gt",
    "eval_chamfer_near_noise_std_ratio": 0.1,
    "eval_tau_ratio": 0.02,
}
EVALUATOR_FIXED_SEED = 2026

# Centralized evaluator override hooks.
# Priority: defaults < run config < method override < dataset override.
EVAL_METHOD_OVERRIDES: dict[str, dict[str, Any]] = {
    "oncl": {},
    "margin": {},
    "dataaug": {},
    "vae": {},
    "ecomann": {},
}
EVAL_DATASET_OVERRIDES: dict[str, dict[str, Any]] = {
    # Empty entries inherit the run config and evaluator defaults.
    "2d_figure_eight": {},
    "2d_ellipse": {},
    "2d_noisy_sine": {},
    "2d_sine": {},
    "2d_sparse_sine": {},
    "2d_discontinuous": {},
    "2d_two_horizontal_lines": {},
    "2d_looped_spiro": {},
    "2d_sharp_star": {},
    "2d_hetero_noise": {},
    "2d_planar_arm_line_n2": {},
    "3d_saddle_surface": {},
    "3d_sphere_surface": {},
    "3d_torus_surface": {},
    "3d_planar_arm_line_n3": {},
    "3d_spatial_arm_plane_n3": {},
    "3d_spatial_arm_ellip_n3": {},
    "3d_spatial_arm_circle_n3": {},
    "6d_spatial_arm_up_n6": {},
    "6d_spatial_arm_up_n6_py": {},
    "6d_workspace_sine_surface_pose": {},
    "6d_workspace_sine_surface_pose_traj": {},
    "6d_workspace_sine_surface_pose_traj_turn": {},
    "12d_dual_arm": {},
    "12d_dual_arm_traj": {},
}


# Reset the sampling stream for each evaluation call.
def _fixed_rng() -> np.random.Generator:
    return np.random.default_rng(EVALUATOR_FIXED_SEED)


_GT_GRID_CACHE: dict[tuple[str, int], np.ndarray] = {}
_UR5_LIMITS_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def resolve_gt_grid(
    dataset_name: str,
    cfg: Any,
    x_train: np.ndarray | None = None,
) -> np.ndarray:
    if dataset_name in FIXED_DEMONSTRATION_DIMS:
        return load_fixed_demonstrations(dataset_name)["eval_grid"]
    cfg_ds = SimpleNamespace(**vars(cfg)) if hasattr(cfg, "__dict__") else SimpleNamespace()
    cfg_ds.n_grid = int(getattr(cfg, "eval_gt_n_grid", DEFAULT_EVAL_CFG["eval_gt_n_grid"]))
    if not hasattr(cfg_ds, "n_train"):
        cfg_ds.n_train = int(max(64, getattr(cfg_ds, "n_grid", 4096)))
    if not hasattr(cfg_ds, "seed"):
        cfg_ds.seed = 0

    cache_key = (
        str(dataset_name),
        int(getattr(cfg_ds, "n_grid", 4096)),
        int(getattr(cfg_ds, "seed", 0)),
        float(getattr(cfg_ds, "z_amp1", 0.35)),
        float(getattr(cfg_ds, "z_amp2", 0.20)),
        float(getattr(cfg_ds, "z_freq1", 1.5)),
        float(getattr(cfg_ds, "z_freq2", 1.2)),
        float(getattr(cfg_ds, "faceted_sphere_bump_scale", 0.72)),
        float(getattr(cfg_ds, "faceted_sphere_sigma", 0.20)),
        float(getattr(cfg_ds, "faceted_sphere_sharpness", 3.2)),
        float(getattr(cfg_ds, "spiky_sphere_spike_scale", 0.86)),
        float(getattr(cfg_ds, "spiky_sphere_sigma", 0.105)),
        float(getattr(cfg_ds, "spiky_sphere_sharpness", 8.0)),
        int(getattr(cfg_ds, "spiky_sphere_n_ctrl", 58)),
    )
    if cache_key in _GT_GRID_CACHE:
        return _GT_GRID_CACHE[cache_key]

    name = str(dataset_name)
    if is_real_iiwa_dataset(name):
        out = real_iiwa_surface_grid(name, cfg_ds, x_train=x_train).astype(np.float32)
        _GT_GRID_CACHE[cache_key] = out
        return out
    if name.startswith("3d_0z_"):
        base = name[len("3d_0z_"):]
        if base.endswith("_traj"):
            base = base[: -len("_traj")]
        x2, grid2 = generate_dataset(base, cfg_ds)
        src = grid2 if grid2 is not None else x2
        out = lift_xy_to_3d_zero(src).astype(np.float32)
        _GT_GRID_CACHE[cache_key] = out
        return out
    if name.startswith("3d_vz_"):
        base = name[len("3d_vz_"):]
        if base.endswith("_traj"):
            base = base[: -len("_traj")]
        x2, grid2 = generate_dataset(base, cfg_ds)
        src = grid2 if grid2 is not None else x2
        out = lift_xy_to_3d_var(src, cfg_ds).astype(np.float32)
        _GT_GRID_CACHE[cache_key] = out
        return out
    try:
        x, grid = generate_dataset(name, cfg_ds)
        src = grid if grid is not None else x
        out = src.astype(np.float32)
        _GT_GRID_CACHE[cache_key] = out
        return out
    except Exception:
        if x_train is None:
            raise
        return x_train.astype(np.float32)


def resolve_eval_cfg(
    base_cfg: Any,
    method_key: str | None = None,
    dataset_name: str | None = None,
) -> Any:
    vals: dict[str, Any] = dict(DEFAULT_EVAL_CFG)
    if hasattr(base_cfg, "__dict__"):
        vals.update(vars(base_cfg))
    if method_key:
        vals.update(EVAL_METHOD_OVERRIDES.get(method_key, {}))
    if dataset_name:
        vals.update(EVAL_DATASET_OVERRIDES.get(dataset_name, {}))
    return SimpleNamespace(**vals)


def _wrap_pi_np(x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _mixed_unit_pose_angle_dims(dataset_name: str | None, dim: int) -> tuple[int, ...]:
    name = str(dataset_name or "")
    if name in (
        "6d_workspace_sine_surface_pose",
        "6d_workspace_sine_surface_pose_traj",
        "6d_workspace_sine_surface_pose_traj_turn",
        "6d_surf_normal",
        "6d_surf_normal_traj",
    ) and dim >= 6:
        return (3, 4, 5)
    if name in ("12d_dual_arm", "12d_dual_arm_traj") and dim >= 12:
        return (3, 4, 5, 9, 10, 11)
    return ()


def _circular_center_half_width(angles: np.ndarray, pad_ratio: float) -> tuple[float, float]:
    a = _wrap_pi_np(np.asarray(angles, dtype=np.float32).reshape(-1))
    if len(a) == 0:
        return 0.0, float(np.pi)
    theta = np.sort((a + np.pi) % (2.0 * np.pi)).astype(np.float64)
    if len(theta) == 1:
        return float(a[0]), 1e-6
    gaps = np.diff(np.concatenate([theta, theta[:1] + 2.0 * np.pi]))
    gap_idx = int(np.argmax(gaps))
    largest_gap = float(gaps[gap_idx])
    arc_width = float(max(0.0, 2.0 * np.pi - largest_gap))
    arc_start = float(theta[(gap_idx + 1) % len(theta)])
    center_theta = arc_start + 0.5 * arc_width
    center = float(_wrap_pi_np(np.asarray([center_theta - np.pi], dtype=np.float32))[0])
    half = float(min(np.pi, max(1e-6, 0.5 * arc_width * max(1.0 + float(pad_ratio), 1e-6))))
    return center, half


def eval_bounds_from_train(
    x_train: np.ndarray,
    cfg: Any,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mins = x_train.min(axis=0)
    maxs = x_train.max(axis=0)
    angle_dims = _mixed_unit_pose_angle_dims(dataset_name, int(x_train.shape[1]))
    pad_ratio = float(getattr(cfg, "eval_pad_ratio", DEFAULT_EVAL_CFG["eval_pad_ratio"]))
    if angle_dims:
        center = 0.5 * (mins + maxs)
        span = np.maximum(maxs - mins, 1e-6)
        half = 0.5 * span * max(1.0 + pad_ratio, 1e-6)
        for d in angle_dims:
            c_d, h_d = _circular_center_half_width(x_train[:, int(d)], pad_ratio)
            center[int(d)] = c_d
            half[int(d)] = h_d
        return (center - half).astype(np.float32), (center + half).astype(np.float32)

    span_raw = maxs - mins
    min_axis_ratio = float(getattr(cfg, "eval_min_axis_span_ratio", DEFAULT_EVAL_CFG["eval_min_axis_span_ratio"]))
    ref_span = max(float(np.max(span_raw)), 1e-6)
    min_axis_span = max(0.0, min_axis_ratio) * ref_span
    span = np.maximum(span_raw, min_axis_span)
    scale = max(1.0 + pad_ratio, 1e-6)
    center = 0.5 * (mins + maxs)
    half = 0.5 * span * scale
    return center - half, center + half


def _clip_points_to_eval_bounds(
    x: np.ndarray,
    x_train: np.ndarray,
    cfg: Any,
    dataset_name: str | None = None,
) -> np.ndarray:
    """Evaluation-only safety clip to the padded train-domain bounds."""
    mins, maxs = eval_bounds_from_train(x_train, cfg, dataset_name=dataset_name)
    center = (0.5 * (mins + maxs)).astype(np.float32)
    out = x.astype(np.float32).copy()
    finite_rows = np.isfinite(out).all(axis=1)
    if not np.all(finite_rows):
        out[~finite_rows] = center
    angle_dims = _mixed_unit_pose_angle_dims(dataset_name, int(out.shape[1]))
    non_angle_dims = [i for i in range(out.shape[1]) if i not in set(angle_dims)]
    if non_angle_dims:
        out[:, non_angle_dims] = np.clip(
            out[:, non_angle_dims],
            mins.reshape(1, -1)[:, non_angle_dims].astype(np.float32),
            maxs.reshape(1, -1)[:, non_angle_dims].astype(np.float32),
        ).astype(np.float32)
    if angle_dims:
        out[:, list(angle_dims)] = _wrap_pi_np(out[:, list(angle_dims)])
    return out


def sample_eval_seed_points(
    x_train: np.ndarray,
    cfg: Any,
    dataset_name: str | None = None,
) -> np.ndarray:
    rng = _fixed_rng()
    n_seed = max(64, int(getattr(cfg, "eval_proj_n_points", DEFAULT_EVAL_CFG["eval_proj_n_points"])))
    mins, maxs = eval_bounds_from_train(x_train, cfg, dataset_name=dataset_name)
    angle_dims = _mixed_unit_pose_angle_dims(dataset_name, int(x_train.shape[1]))
    if str(dataset_name or "").startswith("6d_spatial_arm_up_n6"):
        lim = _ur5_joint_limits_from_urdf()
        if lim is not None:
            lo, hi = lim
            if int(x_train.shape[1]) >= 6:
                mins = lo.astype(np.float32)
                maxs = hi.astype(np.float32)
                angle_dims = ()
    span = np.maximum(maxs - mins, 1e-6).astype(np.float32)
    near_ratio = float(np.clip(cfg.eval_chamfer_near_ratio, 0.0, 1.0))
    n_near = int(round(n_seed * near_ratio))
    n_box = max(0, n_seed - n_near)
    out = []
    if n_near > 0:
        near_source = x_train.astype(np.float32)
        near_source_name = str(getattr(cfg, "eval_chamfer_near_source", "gt")).strip().lower()
        if near_source_name in {"gt", "true", "true_manifold", "manifold"} and dataset_name:
            try:
                gt_grid = resolve_gt_grid(str(dataset_name), cfg, x_train=x_train).astype(np.float32)
                if gt_grid.ndim == 2 and gt_grid.shape[1] == x_train.shape[1] and len(gt_grid) > 0:
                    non_angle_dims = [i for i in range(gt_grid.shape[1]) if i not in set(angle_dims)]
                    in_bounds = np.isfinite(gt_grid).all(axis=1)
                    if non_angle_dims:
                        lo = mins.reshape(1, -1)[:, non_angle_dims]
                        hi = maxs.reshape(1, -1)[:, non_angle_dims]
                        in_bounds &= np.all((gt_grid[:, non_angle_dims] >= lo) & (gt_grid[:, non_angle_dims] <= hi), axis=1)
                    near_source = gt_grid[in_bounds] if np.any(in_bounds) else gt_grid
            except Exception:
                near_source = x_train.astype(np.float32)
        idx = rng.integers(0, len(near_source), size=n_near)
        x0_near = near_source[idx].astype(np.float32).copy()
        noise_std = float(max(cfg.eval_chamfer_near_noise_std_ratio, 1e-6))
        x0_near = x0_near + rng.normal(size=x0_near.shape).astype(np.float32) * (
            noise_std * span.reshape(1, -1)
        )
        non_angle_dims = [i for i in range(x0_near.shape[1]) if i not in set(angle_dims)]
        if non_angle_dims:
            x0_near[:, non_angle_dims] = np.clip(
                x0_near[:, non_angle_dims],
                mins.reshape(1, -1)[:, non_angle_dims],
                maxs.reshape(1, -1)[:, non_angle_dims],
            )
        if angle_dims:
            x0_near[:, list(angle_dims)] = _wrap_pi_np(x0_near[:, list(angle_dims)])
        out.append(x0_near.astype(np.float32))
    if n_box > 0:
        x0_box = rng.uniform(mins, maxs, size=(n_box, len(mins))).astype(np.float32)
        if angle_dims:
            x0_box[:, list(angle_dims)] = _wrap_pi_np(x0_box[:, list(angle_dims)])
        out.append(x0_box.astype(np.float32))
    if not out:
        x0 = rng.uniform(mins, maxs, size=(n_seed, len(mins))).astype(np.float32)
        if angle_dims:
            x0[:, list(angle_dims)] = _wrap_pi_np(x0[:, list(angle_dims)])
        return x0.astype(np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)


def _ur5_joint_limits_from_urdf() -> tuple[np.ndarray, np.ndarray] | None:
    global _UR5_LIMITS_CACHE
    if _UR5_LIMITS_CACHE is not None:
        return _UR5_LIMITS_CACHE
    try:
        from datasets.ur5_pybullet_utils import resolve_ur5_kinematics_cfg

        kin = resolve_ur5_kinematics_cfg({})
        urdf_path = str(kin.get("urdf_path", "")).strip()
        if (not urdf_path) or (not os.path.exists(urdf_path)):
            return None
        root = ET.parse(urdf_path).getroot()
        lo_list: list[float] = []
        hi_list: list[float] = []
        for j in root.findall("joint"):
            if str(j.attrib.get("type", "")).strip().lower() != "revolute":
                continue
            lim = j.find("limit")
            if lim is None:
                continue
            try:
                lo_j = float(lim.attrib.get("lower", "-3.141592653589793"))
                hi_j = float(lim.attrib.get("upper", "3.141592653589793"))
            except Exception:
                continue
            if (not np.isfinite(lo_j)) or (not np.isfinite(hi_j)) or (hi_j <= lo_j):
                continue
            lo_list.append(lo_j)
            hi_list.append(hi_j)
            if len(lo_list) >= 6:
                break
        if len(lo_list) < 6:
            return None
        _UR5_LIMITS_CACHE = (
            np.asarray(lo_list[:6], dtype=np.float32),
            np.asarray(hi_list[:6], dtype=np.float32),
        )
        return _UR5_LIMITS_CACHE
    except Exception:
        return None


def compute_eps_stop(
    model: nn.Module,
    x_train: np.ndarray,
    cfg: Any,
    *,
    return_h_on: bool = False,
) -> float | tuple[float, np.ndarray]:
    with torch.no_grad():
        f_on = model(torch.from_numpy(x_train.astype(np.float32)).to(cfg.device))
        if f_on.dim() == 1:
            f_on = f_on.unsqueeze(1)
        h_on = torch.linalg.norm(f_on, dim=1).detach().cpu().numpy().reshape(-1)
    eps = float(np.percentile(np.abs(h_on), float(cfg.zero_eps_quantile)))
    if bool(return_h_on):
        return eps, h_on.astype(np.float32)
    return eps


def _nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.full((len(a),), float("nan"), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(b.astype(np.float64))
        d, _ = tree.query(a.astype(np.float64), k=1)
        return d.astype(np.float32)
    except Exception:
        aa = a.astype(np.float32)
        bb = b.astype(np.float32)
        d2 = np.sum((aa[:, None, :] - bb[None, :, :]) ** 2, axis=2)
        return np.sqrt(np.maximum(np.min(d2, axis=1), 0.0)).astype(np.float32)


def _true_projection(x: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(grid.astype(np.float64))
        d, idx = tree.query(x.astype(np.float64), k=1)
        return grid[idx], d.astype(np.float32)
    except Exception:
        xx = x.astype(np.float32)
        gg = grid.astype(np.float32)
        d2 = np.sum((xx[:, None, :] - gg[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        d = np.sqrt(np.maximum(d2[np.arange(len(x)), idx], 0.0)).astype(np.float32)
        return gg[idx], d


def _workspace_pose_analytic_target_embed(embed_xyz_zaxis: np.ndarray, cfg: Any | None = None) -> np.ndarray:
    x = embed_xyz_zaxis[:, 0].astype(np.float32)
    y = embed_xyz_zaxis[:, 1].astype(np.float32)
    z_true, n_true = sine_surface_z_and_normal_from_xy(x, y, cfg=cfg)
    tgt = np.concatenate(
        [x[:, None], y[:, None], z_true[:, None], n_true.astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    return tgt


def _workspace_pose_analytic_dist_embed(embed_xyz_zaxis: np.ndarray, cfg: Any | None = None) -> np.ndarray:
    tgt = _workspace_pose_analytic_target_embed(embed_xyz_zaxis, cfg=cfg)
    d = np.linalg.norm(
        embed_xyz_zaxis.astype(np.float32) - tgt.astype(np.float32),
        axis=1,
    ).astype(np.float32)
    return d


def _dual_arm_pose_params(cfg: Any) -> dict[str, float]:
    required = (
        "dual_arm_grasp_span",
        "dual_arm_curve_x_span",
        "dual_arm_curve_y_amp",
        "dual_arm_curve_y_freq",
        "dual_arm_curve_z_base",
        "dual_arm_vertical_half_range",
    )
    missing = [k for k in required if not hasattr(cfg, k)]
    if missing:
        raise ValueError(
            "12D dual-arm analytic evaluation requires task config fields "
            f"{list(required)}, but missing {missing}. "
            "This usually means env/task parameters were not propagated into eval cfg."
        )
    z_half = float(getattr(cfg, "dual_arm_vertical_half_range"))
    return {
        "grasp_span": float(getattr(cfg, "dual_arm_grasp_span")),
        "x_span": float(getattr(cfg, "dual_arm_curve_x_span")),
        "y_amp": float(getattr(cfg, "dual_arm_curve_y_amp")),
        "y_freq": float(getattr(cfg, "dual_arm_curve_y_freq")),
        "z_base": float(getattr(cfg, "dual_arm_curve_z_base")),
        "z_amp": float(getattr(cfg, "dual_arm_curve_z_amp", z_half)),
        "z_freq": float(getattr(cfg, "dual_arm_curve_z_freq", 0.7)),
        "z_half_range": z_half,
    }


def _rpy_zyx_to_rotmat_batch(rpy: np.ndarray) -> np.ndarray:
    rr = rpy[:, 0].astype(np.float32)
    pp = rpy[:, 1].astype(np.float32)
    yy = rpy[:, 2].astype(np.float32)
    cr = np.cos(rr)
    sr = np.sin(rr)
    cp = np.cos(pp)
    sp = np.sin(pp)
    cy = np.cos(yy)
    sy = np.sin(yy)
    R = np.zeros((len(rpy), 3, 3), dtype=np.float32)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R.astype(np.float32)


def _dual_arm_curve_center_tnb_from_s(s: np.ndarray, cfg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = _dual_arm_pose_params(cfg)
    ss = s.astype(np.float32)
    x = (p["x_span"] * ss).astype(np.float32)
    y = (p["y_amp"] * np.sin(p["y_freq"] * np.pi * ss)).astype(np.float32)
    z = np.full_like(ss, p["z_base"], dtype=np.float32)
    dx = np.full_like(ss, p["x_span"], dtype=np.float32)
    dy = (p["y_amp"] * p["y_freq"] * np.pi * np.cos(p["y_freq"] * np.pi * ss)).astype(np.float32)
    dz = np.zeros_like(ss, dtype=np.float32)

    center = np.stack([x, y, z], axis=1).astype(np.float32)
    tang = np.stack([dx, dy, dz], axis=1).astype(np.float32)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)

    z_axis = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(ss), 1))
    y_axis = np.cross(z_axis, tang).astype(np.float32)
    y_axis /= (np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-12)
    z_axis = np.cross(tang, y_axis).astype(np.float32)
    z_axis /= (np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-12)
    return center.astype(np.float32), tang.astype(np.float32), y_axis.astype(np.float32), z_axis.astype(np.float32)


def _dual_arm_pose_embed_raw(x_raw: np.ndarray) -> np.ndarray:
    x = x_raw.astype(np.float32)

    def _pose_embed(p: np.ndarray) -> np.ndarray:
        pos = p[:, :3].astype(np.float32)
        rpy = p[:, 3:6].astype(np.float32)
        return np.concatenate([pos, np.cos(rpy), np.sin(rpy)], axis=1).astype(np.float32)

    return np.concatenate([_pose_embed(x[:, :6]), _pose_embed(x[:, 6:12])], axis=1).astype(np.float32)


def _dual_arm_pose_task_embed_raw(x_raw: np.ndarray) -> np.ndarray:
    x = x_raw.astype(np.float32)

    def _pose_embed(p: np.ndarray) -> np.ndarray:
        pos = p[:, :3].astype(np.float32)
        R = _rpy_zyx_to_rotmat_batch(p[:, 3:6].astype(np.float32))
        # Similar to 6D sine pose's [position, task axis] distance, but include
        # the full task frame so tangent alignment and same orientation are both visible.
        return np.concatenate(
            [pos, R[:, :, 0], R[:, :, 1], R[:, :, 2]],
            axis=1,
        ).astype(np.float32)

    return np.concatenate([_pose_embed(x[:, :6]), _pose_embed(x[:, 6:12])], axis=1).astype(np.float32)


def _dual_arm_pose_analytic_target_raw(x_raw: np.ndarray, cfg: Any) -> np.ndarray:
    x = x_raw.astype(np.float32)
    p = _dual_arm_pose_params(cfg)

    center_obs = (0.5 * (x[:, 0:3] + x[:, 6:9])).astype(np.float32)
    s_grid = np.linspace(-1.0, 1.0, 2048, dtype=np.float32)
    center_grid, _, _, _ = _dual_arm_curve_center_tnb_from_s(s_grid, cfg)
    d2 = np.sum((center_obs[:, None, 0:2] - center_grid[None, :, 0:2]) ** 2, axis=2)
    idx = np.argmin(d2, axis=1)
    s_star = s_grid[idx].astype(np.float32)

    center, tang, y_axis, z_axis = _dual_arm_curve_center_tnb_from_s(s_star, cfg)
    u_star = np.clip(center_obs[:, 2] - p["z_base"], -p["z_half_range"], p["z_half_range"])
    center[:, 2] = p["z_base"] + u_star.astype(np.float32)

    R_tgt = np.stack([tang, y_axis, z_axis], axis=2).astype(np.float32)
    sy = -np.clip(R_tgt[:, 2, 0], -1.0, 1.0)
    pitch = np.arcsin(sy).astype(np.float32)
    cp = np.cos(pitch)
    roll = np.where(
        np.abs(cp) > 1e-8,
        np.arctan2(R_tgt[:, 2, 1], R_tgt[:, 2, 2]),
        0.0,
    ).astype(np.float32)
    yaw = np.where(
        np.abs(cp) > 1e-8,
        np.arctan2(R_tgt[:, 1, 0], R_tgt[:, 0, 0]),
        np.arctan2(-R_tgt[:, 0, 1], R_tgt[:, 1, 1]),
    ).astype(np.float32)
    rpy = np.stack([roll, pitch, yaw], axis=1).astype(np.float32)

    offset = (0.5 * p["grasp_span"] * tang).astype(np.float32)
    pose_1 = np.concatenate([center - offset, rpy], axis=1).astype(np.float32)
    pose_2 = np.concatenate([center + offset, rpy], axis=1).astype(np.float32)
    return np.concatenate([pose_1, pose_2], axis=1).astype(np.float32)


def _dual_arm_pose_analytic_target_embed(x_raw: np.ndarray, cfg: Any) -> np.ndarray:
    return _dual_arm_pose_embed_raw(_dual_arm_pose_analytic_target_raw(x_raw, cfg))


def _dual_arm_pose_analytic_dist_embed(x_raw: np.ndarray, cfg: Any) -> np.ndarray:
    src = _dual_arm_pose_task_embed_raw(x_raw)
    tgt = _dual_arm_pose_task_embed_raw(_dual_arm_pose_analytic_target_raw(x_raw, cfg))
    return np.linalg.norm(src - tgt, axis=1).astype(np.float32)


def _rotation_geodesic_deg_rpy(rpy_a: np.ndarray, rpy_b: np.ndarray) -> np.ndarray:
    ra = _rpy_zyx_to_rotmat_batch(np.asarray(rpy_a, dtype=np.float32))
    rb = _rpy_zyx_to_rotmat_batch(np.asarray(rpy_b, dtype=np.float32))
    r_rel = np.einsum("nij,njk->nik", np.transpose(ra, (0, 2, 1)), rb)
    tr = r_rel[:, 0, 0] + r_rel[:, 1, 1] + r_rel[:, 2, 2]
    return np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))).astype(np.float32)


def dual_arm_pose_true_constraint_error_arrays(x_raw: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    x = np.asarray(x_raw, dtype=np.float32)
    target = _dual_arm_pose_analytic_target_raw(x[:, :12], cfg)
    left_pos = np.linalg.norm(x[:, 0:3] - target[:, 0:3], axis=1).astype(np.float32)
    right_pos = np.linalg.norm(x[:, 6:9] - target[:, 6:9], axis=1).astype(np.float32)
    left_ori = _rotation_geodesic_deg_rpy(x[:, 3:6], target[:, 3:6]).astype(np.float32)
    right_ori = _rotation_geodesic_deg_rpy(x[:, 9:12], target[:, 9:12]).astype(np.float32)
    center = (0.5 * (x[:, 0:3] + x[:, 6:9])).astype(np.float32)
    center_t = (0.5 * (target[:, 0:3] + target[:, 6:9])).astype(np.float32)
    p = _dual_arm_pose_params(cfg)
    span = np.linalg.norm(x[:, 6:9] - x[:, 0:3], axis=1).astype(np.float32)
    return {
        "left_pos_err": left_pos,
        "right_pos_err": right_pos,
        "mean_pos_err": (0.5 * (left_pos + right_pos)).astype(np.float32),
        "left_ori_err_deg": left_ori,
        "right_ori_err_deg": right_ori,
        "mean_ori_err_deg": (0.5 * (left_ori + right_ori)).astype(np.float32),
        "analytic_vector_dist": _dual_arm_pose_analytic_dist_embed(x[:, :12], cfg).astype(np.float32),
        "center_err": np.linalg.norm(center - center_t, axis=1).astype(np.float32),
        "span_err": np.abs(span - float(p["grasp_span"])).astype(np.float32),
    }


def evaluate_bidirectional_chamfer(
    model: nn.Module,
    x_train: np.ndarray,
    cfg: Any,
    project_fn: Callable[[nn.Module, np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    dataset_name: str | None = None,
    embed_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    postprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    eps_stop_override: float | None = None,
    x0_override: np.ndarray | None = None,
    learned_samples_override: np.ndarray | None = None,
) -> dict[str, float]:
    rng = _fixed_rng()
    if dataset_name is not None:
        gt_grid = resolve_gt_grid(str(dataset_name), cfg, x_train=x_train)
        n_gt = len(gt_grid)
        idx = rng.choice(len(gt_grid), size=n_gt, replace=False)
        gt_samples = gt_grid[idx].astype(np.float32)
    else:
        n_gt = min(len(x_train), max(64, int(getattr(cfg, "eval_gt_n_grid", DEFAULT_EVAL_CFG["eval_gt_n_grid"]))))
        idx = rng.choice(len(x_train), size=n_gt, replace=False)
        gt_samples = x_train[idx].astype(np.float32)
    if postprocess_fn is not None:
        gt_samples = postprocess_fn(gt_samples).astype(np.float32)

    eps_stop = float(eps_stop_override) if eps_stop_override is not None else float(compute_eps_stop(model, x_train, cfg))
    if learned_samples_override is not None:
        learned_samples = _clip_points_to_eval_bounds(
            learned_samples_override.astype(np.float32),
            x_train,
            cfg,
            dataset_name=dataset_name,
        )
    else:
        x0 = (
            x0_override.astype(np.float32)
            if x0_override is not None
            else sample_eval_seed_points(x_train, cfg, dataset_name=dataset_name)
        )
        if postprocess_fn is not None:
            x0 = postprocess_fn(x0)
        learned_samples, _ = project_fn(model, x0, eps_stop)
        learned_samples = _clip_points_to_eval_bounds(
            learned_samples.astype(np.float32),
            x_train,
            cfg,
            dataset_name=dataset_name,
        )
        learned_samples = learned_samples.astype(np.float32)
        if postprocess_fn is not None:
            learned_samples = postprocess_fn(learned_samples)

    if embed_fn is not None:
        a = embed_fn(gt_samples)
        b = embed_fn(learned_samples)
        dist_space = "workspace"
    elif dataset_name is not None and is_real_iiwa_dataset(str(dataset_name)) and real_iiwa_coordinate_space(str(dataset_name)) == "surface_model_normalized":
        a = real_iiwa_normalized_to_world_xyz(str(dataset_name), gt_samples)
        b = real_iiwa_normalized_to_world_xyz(str(dataset_name), learned_samples)
        dist_space = "real_iiwa_world"
    else:
        a = gt_samples
        b = learned_samples
        dist_space = "data_space"

    d_gt_to_learned = _nn_dist(a, b)
    d_learned_to_gt = _nn_dist(b, a)
    return {
        "bidirectional_chamfer": float(np.mean(d_gt_to_learned) + np.mean(d_learned_to_gt)),
        "gt_to_learned_mean": float(np.mean(d_gt_to_learned)),
        "learned_to_gt_mean": float(np.mean(d_learned_to_gt)),
        "gt_to_learned_p95": float(np.percentile(d_gt_to_learned, 95)),
        "learned_to_gt_p95": float(np.percentile(d_learned_to_gt, 95)),
        "dist_space": dist_space,
        "n_gt": int(len(gt_samples)),
        "n_learned": int(len(learned_samples)),
        "eps_stop": float(eps_stop),
    }


def evaluate_projection_metrics(
    model: nn.Module,
    x_train: np.ndarray,
    cfg: Any,
    project_fn: Callable[[nn.Module, np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    dataset_name: str | None = None,
    embed_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    postprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    return_artifacts: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, Any]]:
    if dataset_name is not None:
        grid = resolve_gt_grid(str(dataset_name), cfg, x_train=x_train)
    else:
        grid = x_train.astype(np.float32)
    if postprocess_fn is not None:
        grid = postprocess_fn(grid).astype(np.float32)

    eval_eps_used, h_on = compute_eps_stop(model, x_train, cfg, return_h_on=True)
    x_eval = sample_eval_seed_points(x_train, cfg, dataset_name=dataset_name)
    if postprocess_fn is not None:
        x_eval = postprocess_fn(x_eval).astype(np.float32)
    with torch.no_grad():
        f_eval = model(torch.from_numpy(x_eval.astype(np.float32)).to(cfg.device))
        if f_eval.dim() == 1:
            f_eval = f_eval.unsqueeze(1)
        h_eval = torch.linalg.norm(f_eval, dim=1).detach().cpu().numpy().reshape(-1)
    on_mean_v = float(np.mean(0.5 * (h_on ** 2)))

    use_workspace_analytic = (
        str(dataset_name)
        in (
            "6d_workspace_sine_surface_pose",
            "6d_workspace_sine_surface_pose_traj",
            "6d_workspace_sine_surface_pose_traj_turn",
            "6d_surf_normal",
            "6d_surf_normal_traj",
        )
        and embed_fn is not None
    )
    use_dual_arm_workspace_analytic = (
        str(dataset_name) in ("12d_dual_arm", "12d_dual_arm_traj")
        and embed_fn is not None
    )
    use_real_iiwa_analytic = dataset_name is not None and is_real_iiwa_dataset(str(dataset_name))

    if embed_fn is not None:
        grid_metric = embed_fn(grid)
        x_eval_metric = embed_fn(x_eval)
        if (
            use_workspace_analytic
            and x_eval_metric.ndim == 2
            and x_eval_metric.shape[1] >= 6
        ):
            d_true_eval = _workspace_pose_analytic_dist_embed(
                x_eval_metric[:, :6].astype(np.float32),
                cfg,
            )
            dist_space = "workspace_analytic"
        elif use_dual_arm_workspace_analytic and x_eval.ndim == 2 and x_eval.shape[1] >= 12:
            d_true_eval = _dual_arm_pose_analytic_dist_embed(
                x_eval[:, :12].astype(np.float32),
                cfg,
            )
            dist_space = "workspace_analytic"
        else:
            _, d_true_eval = _true_projection(x_eval_metric, grid_metric)
            dist_space = "workspace"
    else:
        grid_metric = grid
        x_eval_metric = x_eval
        if use_real_iiwa_analytic:
            d_true_eval = real_iiwa_surface_distance(str(dataset_name), x_eval_metric)
            dist_space = "real_iiwa_analytic"
        else:
            _, d_true_eval = _true_projection(x_eval_metric, grid_metric)
            dist_space = "data_space"

    proj, steps = project_fn(model, x_eval, eval_eps_used)
    proj = _clip_points_to_eval_bounds(
        proj.astype(np.float32),
        x_train,
        cfg,
        dataset_name=dataset_name,
    )
    if postprocess_fn is not None:
        proj = postprocess_fn(proj).astype(np.float32)

    if embed_fn is not None:
        proj_metric_all = embed_fn(proj)
        if (
            use_workspace_analytic
            and proj_metric_all.ndim == 2
            and proj_metric_all.shape[1] >= 6
            and x_eval_metric.ndim == 2
            and x_eval_metric.shape[1] >= 6
        ):
            proj_true = _workspace_pose_analytic_target_embed(
                x_eval_metric[:, :6].astype(np.float32),
                cfg,
            )
        elif use_dual_arm_workspace_analytic and x_eval.ndim == 2 and x_eval.shape[1] >= 12:
            proj_true = _dual_arm_pose_analytic_target_embed(
                x_eval[:, :12].astype(np.float32),
                cfg,
            )
        else:
            proj_true, _ = _true_projection(x_eval_metric, grid_metric)
    else:
        proj_metric_all = proj
        proj_true, _ = _true_projection(x_eval_metric, grid_metric)

    proj_mask = np.isfinite(proj).all(axis=1)
    if np.any(proj_mask):
        proj_to_trueproj = np.linalg.norm(proj_metric_all[proj_mask] - proj_true[proj_mask], axis=1)
        if (
            use_workspace_analytic
            and proj_metric_all.ndim == 2
            and proj_metric_all.shape[1] >= 6
        ):
            proj_final_true_dist = _workspace_pose_analytic_dist_embed(
                proj_metric_all[proj_mask, :6].astype(np.float32),
                cfg,
            )
        elif use_dual_arm_workspace_analytic and proj.ndim == 2 and proj.shape[1] >= 12:
            proj_final_true_dist = _dual_arm_pose_analytic_dist_embed(
                proj[proj_mask, :12].astype(np.float32),
                cfg,
            )
        elif use_real_iiwa_analytic:
            proj_final_true_dist = real_iiwa_surface_distance(str(dataset_name), proj_metric_all[proj_mask])
        else:
            proj_final_true_dist = _nn_dist(proj_metric_all[proj_mask], grid_metric)
        with torch.no_grad():
            f_proj = model(torch.from_numpy(proj[proj_mask].astype(np.float32)).to(cfg.device))
            if f_proj.dim() == 1:
                f_proj = f_proj.unsqueeze(1)
            h_proj = torch.linalg.norm(f_proj, dim=1).detach().cpu().numpy().reshape(-1)
        proj_residual = np.abs(h_proj)
        proj_steps_mean = float(np.mean(steps[proj_mask]))
    else:
        proj_to_trueproj = np.array([float("nan")], dtype=np.float32)
        proj_final_true_dist = np.array([float("nan")], dtype=np.float32)
        proj_residual = np.array([float("nan")], dtype=np.float32)
        proj_steps_mean = float("nan")

    tau = float(cfg.eval_tau_ratio) * float(np.mean(grid_metric.max(axis=0) - grid_metric.min(axis=0)))

    # Recall: measure how many GT manifold points are classified as on-manifold by learned constraint.
    with torch.no_grad():
        f_gt = model(torch.from_numpy(grid.astype(np.float32)).to(cfg.device))
        if f_gt.dim() == 1:
            f_gt = f_gt.unsqueeze(1)
        h_gt = torch.linalg.norm(f_gt, dim=1).detach().cpu().numpy().reshape(-1)
    pred_on_gt = np.abs(h_gt) < eval_eps_used
    coverage = float(np.mean(pred_on_gt)) if pred_on_gt.size > 0 else float("nan")

    # False-positive rate on query points outside the GT distance threshold.
    pred_zero_eval = np.abs(h_eval) < eval_eps_used
    far_eval = d_true_eval >= tau
    false_pos = float(np.mean(pred_zero_eval[far_eval])) if np.any(far_eval) else float("nan")

    # Precision: use projected points as predicted on-manifold set and check GT distance against tau.
    if np.any(proj_mask):
        pred_on_true_ratio = float(np.mean(proj_final_true_dist < tau))
    else:
        pred_on_true_ratio = float("nan")

    metrics = {
        "pred_on_mean_v": on_mean_v,
        "proj_manifold_dist": float(np.mean(proj_final_true_dist)),
        "proj_v_residual": float(np.mean(proj_residual)),
        "proj_true_dist": float(np.mean(proj_to_trueproj)),
        "proj_steps": proj_steps_mean,
        "pred_recall": coverage,
        "pred_FPrate": false_pos,
        "pred_precision": pred_on_true_ratio,
        "eval_eps_used": float(eval_eps_used),
        "dist_space": dist_space,
    }
    if use_real_iiwa_analytic and np.any(proj_mask):
        metrics.update(_real_iiwa_projection_error_metrics(str(dataset_name), proj[proj_mask]))
    if not bool(return_artifacts):
        return metrics
    artifacts = {
        "x_eval": x_eval.astype(np.float32),
        "proj": proj.astype(np.float32),
        "steps": steps.astype(np.float32),
        "eval_eps_used": float(eval_eps_used),
    }
    return metrics, artifacts


def _summary_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _real_iiwa_projection_error_metrics(dataset_name: str, proj: np.ndarray) -> dict[str, float]:
    pts = np.asarray(proj, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return {}
    canonical = canonical_real_iiwa_dataset_name(dataset_name)
    if canonical is None:
        return {}
    spec = REAL_IIWA_DATASETS[canonical]
    if real_iiwa_coordinate_space(dataset_name) == "surface_model_normalized":
        pts_world = real_iiwa_normalized_to_world_xyz(dataset_name, pts[:, :3])
    else:
        pts_world = pts[:, :3]

    if real_iiwa_surface_type(dataset_name) == "neural_height_field":
        z_pred, normal = real_iiwa_neural_surface_z_normal(dataset_name, pts_world[:, 0], pts_world[:, 1])
        grad_norm = np.linalg.norm(normal[:, :2] / np.maximum(normal[:, 2:3], 1e-12), axis=1)
        pos_m = np.abs(pts_world[:, 2] - z_pred) / np.sqrt(1.0 + grad_norm**2)
    else:
        normal_one, d = plane_constraint()
        normal = np.tile(normal_one.reshape(1, 3), (len(pts_world), 1)).astype(np.float32)
        pos_m = np.abs(pts_world[:, :3].astype(np.float64) @ normal_one + float(d))

    out = _summary_stats("real_iiwa_proj_position_error_mm", np.asarray(pos_m, dtype=np.float64) * 1000.0)
    out["real_iiwa_proj_position_error_m_mean"] = float(np.mean(pos_m)) if len(pos_m) else float("nan")

    if pts.shape[1] >= 6 and real_iiwa_orientation_representation(dataset_name) == "tool_z":
        tool_z = pts[:, 3:6].astype(np.float32)
        tool_z = tool_z / np.maximum(np.linalg.norm(tool_z, axis=1, keepdims=True), 1e-12)
        sign = float(spec.get("target_tool_axis_sign", -1.0))
        target_tool_z = (sign * normal.astype(np.float32))
        target_tool_z = target_tool_z / np.maximum(np.linalg.norm(target_tool_z, axis=1, keepdims=True), 1e-12)
        dots = np.clip(np.sum(tool_z * target_tool_z, axis=1), -1.0, 1.0)
        err_deg = np.degrees(np.arccos(dots))
        out.update(_summary_stats("real_iiwa_proj_orientation_error_deg", err_deg))
        axis_l2 = np.linalg.norm(tool_z - target_tool_z, axis=1)
        out.update(_summary_stats("real_iiwa_proj_orientation_axis_l2", axis_l2))
    return out
