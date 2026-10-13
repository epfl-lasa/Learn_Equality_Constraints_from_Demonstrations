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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from planning.planner import build_linear_path, plan_path
from planning.plan_surf_normal import (
    DATASET_NAME,
    DEFAULT_CKPT,
    _build_data_pool,
    _choose_device,
    _error_to_true_constraint,
    _load_model,
    _planner_cfg,
    _point_project_path,
    _plot_error_distribution_paper,
    _resolve_path,
    _rpy_zyx_to_local_z,
    _save_pointwise_csv,
    _workspace_surface_z_and_normal_from_xy,
)
from datasets.constraint_datasets import sine_surface_apply_affine_xy


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _rpy_from_rotmat_zyx(R: np.ndarray) -> np.ndarray:
    # R = Rz(yaw) Ry(pitch) Rx(roll)
    sy = -float(R[2, 0])
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = float(np.arcsin(sy))
    cp = float(np.cos(pitch))
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _generate_highfreq_waypoints(
    *,
    n_waypoints: int,
    x_center: float,
    y_start: float,
    y_end: float,
    amp_x: float,
    snake_freq: float,
    surface_cfg: Any | None = None,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(max(3, n_waypoints)), dtype=np.float32)
    # Waypoint parameters are specified in the canonical sine workspace and
    # then mapped into the current environment scale/offset.
    y_base = float(y_start) + (float(y_end) - float(y_start)) * t
    y_mid = 0.5 * (float(y_start) + float(y_end))
    y_base = y_mid + 1.10 * (y_base - y_mid)
    x_base = float(x_center) + float(amp_x) * np.sin(2.0 * np.pi * float(snake_freq) * t)
    x_base = np.clip(x_base, -1.9, 1.9).astype(np.float32)
    y_base = np.clip(y_base, -1.9, 1.9).astype(np.float32)
    xy = sine_surface_apply_affine_xy(
        np.stack([x_base, y_base], axis=1).astype(np.float32),
        surface_cfg,
    )
    x = xy[:, 0].astype(np.float32)
    y = xy[:, 1].astype(np.float32)
    if len(x) >= 5:
        # Widen the sweep near the ends so the path reads more clearly as a
        # left-to-right cleaning motion over the surface patch.
        x[:2] -= 0.028
        x[-3:] += 0.028
    z, nvec = _workspace_surface_z_and_normal_from_xy(x.astype(np.float32), y.astype(np.float32), surface_cfg)
    pos = np.stack([x, y, z], axis=1).astype(np.float32)

    tang = np.zeros_like(pos, dtype=np.float32)
    tang[1:-1] = pos[2:] - pos[:-2]
    tang[0] = pos[1] - pos[0]
    tang[-1] = pos[-1] - pos[-2]

    rpy = np.zeros((len(pos), 3), dtype=np.float32)
    for i in range(len(pos)):
        z_axis = _normalize(nvec[i])
        t_axis = tang[i] - float(np.dot(tang[i], z_axis)) * z_axis
        x_axis = _normalize(t_axis)
        y_axis = _normalize(np.cross(z_axis, x_axis))
        x_axis = _normalize(np.cross(y_axis, z_axis))
        R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
        rpy[i] = _rpy_from_rotmat_zyx(R)
    return np.concatenate([pos, rpy], axis=1).astype(np.float32)


def _plan_waypoint_chain(
    *,
    model: Any,
    cfg: Any,
    waypoints: np.ndarray,
    seg_waypoints: int,
    planner_mode: str,
) -> tuple[np.ndarray, list[float]]:
    full: list[np.ndarray] = []
    plan_times: list[float] = []
    planner_mode_norm = str(planner_mode).strip().lower()
    if planner_mode_norm not in {"traj_opt", "point_project"}:
        raise ValueError(f"unsupported planner_mode: {planner_mode}")
    for i in range(waypoints.shape[0] - 1):
        s = waypoints[i]
        g = waypoints[i + 1]
        t0 = time.time()
        if planner_mode_norm == "traj_opt":
            seg = plan_path(
                model=model,
                x_start=s,
                x_goal=g,
                cfg=cfg,
                planner_name="traj_opt",
                n_waypoints=int(seg_waypoints),
                dataset_name=DATASET_NAME,
                periodic_joint=False,
            ).astype(np.float32)
        else:
            init = build_linear_path(
                np.asarray(s, dtype=np.float32),
                np.asarray(g, dtype=np.float32),
                n_waypoints=int(seg_waypoints),
                periodic=False,
            ).astype(np.float32)
            seg = _point_project_path(
                model,
                init,
                x_start=s,
                x_goal=g,
                device=str(cfg.device),
                proj_steps=int(cfg.projector["steps"]),
                proj_alpha=float(cfg.projector["alpha"]),
                proj_min_steps=int(cfg.projector["min_steps"]),
                f_abs_stop=None,
            ).astype(np.float32)
        plan_times.append(float(time.time() - t0))
        if i == 0:
            full.append(seg)
        else:
            full.append(seg[1:])
    traj = np.concatenate(full, axis=0).astype(np.float32) if len(full) > 0 else waypoints.copy()
    return traj, plan_times


def _plot_waypoints_cleaning_paper(
    *,
    waypoints: np.ndarray,
    traj: np.ndarray,
    out_path: str,
    surface_cfg: Any | None = None,
) -> None:
    grid_xy = sine_surface_apply_affine_xy(
        np.asarray(
            [
                [-2.0, -2.0],
                [-2.0, 2.0],
                [2.0, -2.0],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        ),
        surface_cfg,
    )
    gx = np.linspace(float(np.min(grid_xy[:, 0])), float(np.max(grid_xy[:, 0])), 120).astype(np.float32)
    gy = np.linspace(float(np.min(grid_xy[:, 1])), float(np.max(grid_xy[:, 1])), 120).astype(np.float32)
    gxx, gyy = np.meshgrid(gx, gy)
    z_grid, _ = _workspace_surface_z_and_normal_from_xy(gxx.reshape(-1), gyy.reshape(-1), surface_cfg)
    z_grid = z_grid.reshape(gxx.shape)

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
            gxx,
            gyy,
            z_grid,
            color="#22d3ee",
            alpha=0.30,
            linewidth=0.23,
            edgecolor=(0, 0, 0, 0.24),
            antialiased=True,
            shade=True,
        )

        col = "#b91c1c"
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            color=col,
            linewidth=1.1,
            alpha=0.95,
        )
        ax.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            waypoints[:, 2],
            s=10,
            c="#111827",
            alpha=0.95,
        )

        idx = np.arange(0, traj.shape[0], 10, dtype=int)
        if len(idx) > 0 and idx[-1] != traj.shape[0] - 1:
            idx = np.concatenate([idx, np.array([traj.shape[0] - 1], dtype=int)])
        rpy = traj[idx, 3:6]
        dirs = _rpy_zyx_to_local_z(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        ax.quiver(
            traj[idx, 0],
            traj[idx, 1],
            traj[idx, 2],
            dirs[:, 0],
            dirs[:, 1],
            dirs[:, 2],
            length=0.24,
            normalize=True,
            color=col,
            linewidths=0.75,
            alpha=0.88,
            arrow_length_ratio=0.40,
        )

        ax.set_xlabel("x", fontsize=8, labelpad=1)
        ax.set_ylabel("y", fontsize=8, labelpad=1)
        ax.set_zlabel("z", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=7, pad=0)
        x_pad = 0.05 * float(np.max(gx) - np.min(gx))
        y_pad = 0.05 * float(np.max(gy) - np.min(gy))
        ax.set_xlim(float(np.min(gx) - x_pad), float(np.max(gx) + x_pad))
        ax.set_ylim(float(np.min(gy) - y_pad), float(np.max(gy) + y_pad))
        z_vals_surface = np.asarray(z_grid[np.isfinite(z_grid)], dtype=np.float32)
        z_vals_traj = traj[:, 2].astype(np.float32)
        z_all = np.concatenate([z_vals_surface, z_vals_traj], axis=0)
        z_pad = 0.15
        ax.set_zlim(float(np.min(z_all) - z_pad), 0.75)
        ax.view_init(elev=60, azim=-64)
        try:
            ax.dist = 8.5
        except Exception:
            pass

        h_surface = Patch(facecolor="#22d3ee", edgecolor=(0, 0, 0, 0.24), alpha=0.30)
        h_traj = Line2D([0], [0], color=col, linewidth=1.5)
        h_wp = Line2D([0], [0], color="#111827", marker="o", linestyle="None", markersize=4.0)
        ax.legend(
            [h_surface, h_traj, h_wp],
            ["True equality constraint", "Plan with learned constraint", "Waypoints"],
            loc="upper left",
            frameon=False,
        )
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.998)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task2 surfnormal_waypoints: repeated waypoint-to-waypoint planning on learned constraint."
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Checkpoint path.")
    parser.add_argument(
        "--outdir",
        default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/surfnormal_waypoints_paper",
        help="Output directory.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--n-waypoints", type=int, default=11, help="Number of cleaning waypoints on snake curve.")
    parser.add_argument("--curve-amp-x", type=float, default=1.4, help="Snake amplitude along x.")
    parser.add_argument("--snake-freq", type=float, default=2.5, help="Snake oscillation frequency along x.")
    parser.add_argument("--curve-center-x", type=float, default=0.0, help="Curve center x.")
    parser.add_argument("--curve-y-start", type=float, default=1., help="Snake start y (positive side).")
    parser.add_argument("--curve-y-end", type=float, default=-1.35, help="Snake end y (negative side).")
    parser.add_argument("--seg-waypoints", type=int, default=52, help="Trajectory waypoints per segment.")

    parser.add_argument("--planner-mode", choices=["traj_opt", "point_project"], default="traj_opt", help="Planning mode.")
    parser.add_argument("--opt-steps", type=int, default=1240, help="traj_opt iterations.")
    parser.add_argument("--opt-lr", type=float, default=0.01, help="traj_opt learning rate.")
    parser.add_argument("--lam-manifold", type=float, default=1.0, help="Manifold loss weight.")
    parser.add_argument("--lam-len-joint", type=float, default=0.4, help="Path length loss weight.")
    parser.add_argument("--lam-smooth", type=float, default=0.9, help="Smoothness loss weight.")
    parser.add_argument("--trust-scale", type=float, default=0.8, help="Trust region scale.")
    parser.add_argument("--proj-steps", type=int, default=120, help="Projector steps.")
    parser.add_argument("--proj-alpha", type=float, default=0.3, help="Projector alpha.")
    parser.add_argument("--proj-min-steps", type=int, default=30, help="Projector min steps.")
    args = parser.parse_args()

    ckpt_path = _resolve_path(args.ckpt)
    outdir = _resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(int(args.seed))
    device = _choose_device(str(args.device))
    model, ckpt = _load_model(ckpt_path, device=device)
    surface_cfg = SimpleNamespace(**dict(ckpt.get("cfg", {})))
    _x_train, _pool = _build_data_pool(ckpt, seed=int(args.seed))

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
        obstacle_enable=False,
        obstacle_center_xy=(0.0, 0.0),
        obstacle_radius=0.0,
        obstacle_margin=0.0,
        lam_obstacle=0.0,
    )

    waypoints = _generate_highfreq_waypoints(
        n_waypoints=int(args.n_waypoints),
        x_center=float(args.curve_center_x),
        y_start=float(args.curve_y_start),
        y_end=float(args.curve_y_end),
        amp_x=float(args.curve_amp_x),
        snake_freq=float(args.snake_freq),
        surface_cfg=surface_cfg,
    )
    traj, plan_times = _plan_waypoint_chain(
        model=model,
        cfg=cfg,
        waypoints=waypoints,
        seg_waypoints=int(args.seg_waypoints),
        planner_mode=str(args.planner_mode),
    )

    pos_err, ang_err_deg = _error_to_true_constraint(traj.astype(np.float32))
    point_csv = os.path.join(outdir, "surfnormal_waypoints_pointwise_errors.csv")
    # Reuse pointwise csv writer by passing single pseudo-case split.
    pseudo_case = SimpleNamespace(traj=traj)
    _save_pointwise_csv(point_csv, [pseudo_case], pos_err, ang_err_deg)

    traj_fig = os.path.join(outdir, "surfnormal_waypoints_planning.png")
    _plot_waypoints_cleaning_paper(waypoints=waypoints, traj=traj, out_path=traj_fig, surface_cfg=surface_cfg)
    dist_fig = os.path.join(outdir, "surfnormal_waypoints_error_distribution_paper.png")
    _plot_error_distribution_paper(pos_err=pos_err, ang_err_deg=ang_err_deg, out_path=dist_fig)

    summary = {
        "task": "surfnormal_waypoints",
        "dataset": DATASET_NAME,
        "ckpt": ckpt_path,
        "seed": int(args.seed),
        "n_waypoints": int(args.n_waypoints),
        "segments": int(max(0, args.n_waypoints - 1)),
        "mean_position_error": float(np.mean(pos_err)),
        "std_position_error": float(np.std(pos_err)),
        "mean_orientation_error_deg": float(np.mean(ang_err_deg)),
        "std_orientation_error_deg": float(np.std(ang_err_deg)),
        "mean_segment_plan_seconds": float(np.mean(plan_times) if len(plan_times) > 0 else 0.0),
        "outputs": {
            "planning_plot": traj_fig,
            "distribution_plot": dist_fig,
            "pointwise_csv": point_csv,
        },
    }
    summary_path = os.path.join(outdir, "surfnormal_waypoints_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        "[summary] "
        f"pos_mean={summary['mean_position_error']:.6f}, pos_std={summary['std_position_error']:.6f}, "
        f"ori_mean_deg={summary['mean_orientation_error_deg']:.4f}, ori_std_deg={summary['std_orientation_error_deg']:.4f}, "
        f"seg_plan_time_mean={summary['mean_segment_plan_seconds']:.3f}s"
    )
    print(f"[saved] {summary_path}")
    print(f"[saved] {traj_fig}")
    print(f"[saved] {dist_fig}")
    print(f"[saved] {point_csv}")


if __name__ == "__main__":
    main()
