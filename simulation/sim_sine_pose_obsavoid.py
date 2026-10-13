#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from planning.plan_surf_normal import (
    DEFAULT_CKPT,
    PairCase,
    _build_data_pool,
    _choose_device,
    _error_to_true_constraint,
    _load_model,
    _plan_with_obstacle,
    _plot_error_distribution_paper,
    _plot_planning_paper,
    _planner_cfg,
    _resolve_path,
)
from datasets.constraint_datasets import (
    sine_surface_affine_params,
    sine_surface_apply_affine_scalar,
    sine_surface_apply_affine_xy,
    sine_surface_z_and_normal_from_xy,
)
from simulation.ik_controller import IKConfig, JointTrackConfig, UR5TrajectoryController, _FFmpegVideoWriter


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = [float(v) for v in a]
    bx, by, bz, bw = [float(v) for v in b]
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    roll = float(rpy[0])
    pitch = float(rpy[1])
    yaw = float(rpy[2])
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    q = np.asarray([qx, qy, qz, qw], dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    return q


def _quat_xyzw_to_rpy_zyx(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp_clip = np.clip(sinp, -1.0, 1.0)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (np.pi / 2.0), np.arcsin(sinp_clip))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=1).astype(np.float32)


def _quat_inverse_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    out = q.copy()
    out[:, :3] *= -1.0
    return out.astype(np.float32)


def _apply_quat_offset_batch(q_xyzw: np.ndarray, q_off_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32)
    q_off = np.asarray(q_off_xyzw, dtype=np.float32).reshape(4)
    out = np.zeros_like(q, dtype=np.float32)
    for i in range(len(q)):
        out[i] = _quat_multiply(q[i], q_off)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out.astype(np.float32)


def _error_to_surface_constraint(
    x_pose: np.ndarray,
    *,
    surface_cfg=None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_pose, dtype=np.float32)
    return _error_to_true_constraint(x, surface_cfg)


def _downsample_traj_for_plot(traj: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    n = len(arr)
    m = int(max(2, max_points))
    if n <= m:
        return arr.astype(np.float32)
    idx = np.linspace(0, n - 1, num=m, dtype=np.int32)
    idx = np.unique(np.clip(idx, 0, n - 1))
    if idx[-1] != n - 1:
        idx = np.concatenate([idx, np.asarray([n - 1], dtype=np.int32)])
    return arr[idx].astype(np.float32)


def _make_surface_patch_xyz_grid(
    pose_path_raw: np.ndarray,
    *,
    surface_cfg=None,
    pad_x_neg: float = 0.06,
    pad_x_pos: float = 0.18,
    pad_y_neg: float = 0.20,
    pad_y_pos: float = 0.08,
    grid_size: int = 28,
) -> np.ndarray:
    xyz = np.asarray(pose_path_raw[:, :3], dtype=np.float32)
    # Keep the visual patch biased away from the robot base side so the surface
    # does not visually overlap the arm. This is visualization-only.
    x_lo = float(np.min(xyz[:, 0]) - pad_x_neg)
    x_hi = float(np.max(xyz[:, 0]) + pad_x_pos)
    y_lo = float(np.min(xyz[:, 1]) - pad_y_neg)
    y_hi = float(np.max(xyz[:, 1]) + pad_y_pos)
    gx = np.linspace(x_lo, x_hi, num=int(max(8, grid_size)), dtype=np.float32)
    gy = np.linspace(y_lo, y_hi, num=int(max(8, grid_size)), dtype=np.float32)
    gxx, gyy = np.meshgrid(gx, gy)
    gzz, _ = _workspace_surface_z_and_normal_from_xy(gxx.reshape(-1), gyy.reshape(-1), surface_cfg)
    return np.stack([gxx, gyy, gzz.reshape(gxx.shape)], axis=2).astype(np.float32)


def _orientation_error_deg(q_des_xyzw: np.ndarray, q_meas_xyzw: np.ndarray) -> np.ndarray:
    qd = np.asarray(q_des_xyzw, dtype=np.float32)
    qm = np.asarray(q_meas_xyzw, dtype=np.float32)
    out = np.zeros((len(qd),), dtype=np.float32)
    for i in range(len(qd)):
        q_err = _quat_multiply(qd[i], _quat_conjugate(qm[i]))
        q_err /= max(float(np.linalg.norm(q_err)), 1e-8)
        w = float(np.clip(abs(q_err[3]), 0.0, 1.0))
        out[i] = float(np.degrees(2.0 * math.acos(w)))
    return out


def _quat_to_local_z_axis(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]
    z_x = 2.0 * (x * z + y * w)
    z_y = 2.0 * (y * z - x * w)
    z_z = 1.0 - 2.0 * (x * x + y * y)
    out = np.stack([z_x, z_y, z_z], axis=1).astype(np.float32)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out


def _workspace_surface_z_and_normal_from_xy(x: np.ndarray, y: np.ndarray, surface_cfg=None) -> tuple[np.ndarray, np.ndarray]:
    return sine_surface_z_and_normal_from_xy(x, y, surface_cfg)


def _plot_tracking_errors(
    *,
    segment_series: list[dict[str, np.ndarray]],
    out_path: str,
) -> None:
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    ):
        n_seg = max(1, len(segment_series))
        fig, axes = plt.subplots(
            n_seg,
            3,
            figsize=(7.2, max(2.2 * n_seg, 3.0)),
            sharex=False,
        )
        if n_seg == 1:
            axes = np.asarray([axes], dtype=object)
        colors = ("#2563eb", "#dc2626", "#15803d")
        ylabels = ("Pos Err", "Axis Align Err (deg)", "Joint Err")
        for row_idx, seg in enumerate(segment_series):
            t = np.asarray(seg["time_s"], dtype=np.float32)
            pos = np.asarray(seg["pos_track_err"], dtype=np.float32)
            ori = np.asarray(seg["axis_align_err_deg"], dtype=np.float32)
            joint = np.asarray(seg["joint_err_norm"], dtype=np.float32)
            vals = (pos, ori, joint)
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                ax.plot(t, vals[col_idx], color=colors[col_idx], lw=1.0)
                ax.grid(alpha=0.22)
                if row_idx == 0:
                    ax.set_title(ylabels[col_idx], fontsize=8)
                if col_idx == 0:
                    seg_label = f"Seg {row_idx + 1}"
                    if bool(seg.get("forced_cross_obstacle", False)):
                        seg_label += " cross"
                    else:
                        seg_label += " free"
                    ax.set_ylabel(seg_label, fontsize=8)
                if row_idx == n_seg - 1:
                    ax.set_xlabel("Time (s)")
        fig.tight_layout(pad=0.45)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def _concat_videos_ffmpeg(segment_paths: list[str], out_path: str) -> str | None:
    valid_paths = [os.path.abspath(p) for p in segment_paths if p and os.path.exists(p)]
    if not valid_paths:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return valid_paths[-1]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        concat_path = f.name
        for path in valid_paths:
            escaped = path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    try:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-c",
            "copy",
            os.path.abspath(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.abspath(out_path)
    except Exception:
        return valid_paths[-1]
    finally:
        try:
            os.remove(concat_path)
        except OSError:
            pass


def _video_frame_schedule(n_steps: int, *, sim_dt: float, video_fps: int) -> np.ndarray:
    n = int(max(0, n_steps))
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)
    capture_every = max(1, int(round(1.0 / max(float(sim_dt) * float(video_fps), 1e-8))))
    # Match track_joint_trajectory(): one initial post-settle frame, then
    # frames sampled from control steps at the writer cadence.
    idx = [-1]
    for i in range(n):
        if ((i + 1) % capture_every == 0) or (i == n - 1):
            idx.append(i)
    return np.asarray(idx, dtype=np.int32)


def _make_error_curve_video(
    *,
    pos_err: np.ndarray,
    ori_err_deg: np.ndarray,
    sim_dt: float,
    video_fps: int,
    video_slowdown: float,
    out_path: str,
    width: int = 760,
    height: int = 430,
) -> str | None:
    pos = np.asarray(pos_err, dtype=np.float32).reshape(-1)
    ori = np.asarray(ori_err_deg, dtype=np.float32).reshape(-1)
    n = int(min(len(pos), len(ori)))
    if n <= 0:
        return None
    idx = _video_frame_schedule(n, sim_dt=float(sim_dt), video_fps=int(video_fps))
    if len(idx) == 0:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_fps = float(video_fps) / max(float(video_slowdown), 1e-3)
    writer = _FFmpegVideoWriter(out_path=os.path.abspath(out_path), width=int(width), height=int(height), fps=out_fps)

    t = np.arange(n, dtype=np.float32) * float(sim_dt)
    pos_mm = pos * 1000.0
    pos_lim = max(float(np.percentile(pos_mm[np.isfinite(pos_mm)], 99)) if np.isfinite(pos_mm).any() else 0.0, 0.1)
    ori_lim = max(float(np.percentile(ori[np.isfinite(ori)], 99)) if np.isfinite(ori).any() else 0.0, 1e-2)
    pos_lim *= 1.15
    ori_lim *= 1.15
    fig, axes = plt.subplots(2, 1, figsize=(float(width) / 100.0, float(height) / 100.0), dpi=100, sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    axes[0].set_facecolor("white")
    axes[1].set_facecolor("white")
    line_pos, = axes[0].plot([], [], color="#2563eb", lw=2.8, solid_capstyle="round")
    fill_pos = axes[0].fill_between([], [], [], color="#93c5fd", alpha=0.35)
    dot_pos, = axes[0].plot([], [], marker="o", color="#1d4ed8", ms=6, linestyle="None")
    vline_pos = axes[0].axvline(0.0, color="#1d4ed8", lw=1.2, ls="--", alpha=0.55)
    line_ori, = axes[1].plot([], [], color="#dc2626", lw=2.8, solid_capstyle="round")
    fill_ori = axes[1].fill_between([], [], [], color="#fca5a5", alpha=0.35)
    dot_ori, = axes[1].plot([], [], marker="o", color="#b91c1c", ms=6, linestyle="None")
    vline_ori = axes[1].axvline(0.0, color="#b91c1c", lw=1.2, ls="--", alpha=0.55)
    axes[0].set_ylabel("mm", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("deg", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0.0, pos_lim)
    axes[1].set_ylim(0.0, ori_lim)
    axes[0].set_xlim(float(t[0]), float(t[-1]) if len(t) > 1 else float(t[0] + sim_dt))
    for ax in axes:
        ax.grid(alpha=0.18, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=10, width=0.8, length=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
    axes[0].text(
        0.01, 0.96, "Position error to true constraint", transform=axes[0].transAxes,
        fontsize=12, fontweight="bold", color="#1e3a8a", va="top"
    )
    axes[1].text(
        0.01, 0.96, "Orientation error to true constraint", transform=axes[1].transAxes,
        fontsize=12, fontweight="bold", color="#991b1b", va="top"
    )
    value_pos = axes[0].text(
        0.99, 0.92, "", transform=axes[0].transAxes,
        fontsize=15, fontweight="bold", color="#1d4ed8", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="#dbeafe", ec="#93c5fd", lw=0.9)
    )
    value_ori = axes[1].text(
        0.99, 0.92, "", transform=axes[1].transAxes,
        fontsize=15, fontweight="bold", color="#b91c1c", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fee2e2", ec="#fca5a5", lw=0.9)
    )
    fig.subplots_adjust(left=0.10, right=0.992, bottom=0.11, top=0.985, hspace=0.34)
    try:
        for k_raw in idx.tolist():
            k = int(np.clip(k_raw, 0, n - 1))
            kk = k + 1
            x_now = float(t[k]) if k_raw >= 0 else 0.0
            line_pos.set_data(t[:kk], pos_mm[:kk])
            dot_pos.set_data([x_now], [pos_mm[k]])
            vline_pos.set_xdata([x_now, x_now])
            line_ori.set_data(t[:kk], ori[:kk])
            dot_ori.set_data([x_now], [ori[k]])
            vline_ori.set_xdata([x_now, x_now])
            value_pos.set_text(f"{pos_mm[k]:.1f} mm")
            value_ori.set_text(f"{ori[k]:.2f}°")
            fill_pos.remove()
            fill_ori.remove()
            fill_pos = axes[0].fill_between(t[:kk], pos_mm[:kk], np.zeros((kk,), dtype=np.float32), color="#93c5fd", alpha=0.35)
            fill_ori = axes[1].fill_between(t[:kk], ori[:kk], np.zeros((kk,), dtype=np.float32), color="#fca5a5", alpha=0.35)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
            writer.append_data(frame)
    finally:
        try:
            writer.close()
        finally:
            plt.close(fig)
    return os.path.abspath(out_path)


def _cleanup_segment_videos(segment_paths: list[str], final_video_path: str | None) -> None:
    final_abs = os.path.abspath(final_video_path) if final_video_path else None
    for path in segment_paths:
        if not path:
            continue
        try:
            path_abs = os.path.abspath(path)
            if final_abs is not None and path_abs == final_abs:
                continue
            if os.path.exists(path_abs):
                os.remove(path_abs)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan a 6D sine-pose trajectory, solve it to UR5 joints, and track it in PyBullet."
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Checkpoint path for the learned constraint model.")
    parser.add_argument(
        "--outdir",
        default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/sim_sine_pose_obsavoid",
        help="Output directory.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    parser.add_argument("--seed", type=int, default=91262, help="Random seed.")
    parser.add_argument("--n-waypoints", type=int, default=86, help="Pose waypoints produced by the planner.")
    parser.add_argument("--pair-min-dist", type=float, default=1.05, help="Min start-goal xyz distance.")
    parser.add_argument("--pair-max-dist", type=float, default=3.4, help="Max start-goal xyz distance.")
    parser.add_argument("--pair-max-y", type=float, default=1.2, help="Require sampled start/goal y <= this value.")
    parser.add_argument("--pair-tries", type=int, default=1600, help="Pair sampling retries.")
    parser.add_argument("--planner-mode", choices=["traj_opt", "point_project"], default="traj_opt", help="Planning mode.")
    parser.add_argument("--pair-force-cross-obstacle", type=int, default=1, help="1 to force obstacle crossing.")
    parser.add_argument("--n-force-cross-trajs", type=int, default=3, help="How many trajectories should be sampled with explicit obstacle-crossing geometry.")
    parser.add_argument(
        "--pair-cross-radius-scale",
        type=float,
        default=1.03,
        help="Intersection radius multiplier for forced-cross pair sampling.",
    )
    parser.add_argument(
        "--pair-cross-side-margin",
        type=float,
        default=0.12,
        help="Minimum |x-center_x| for endpoints when forcing obstacle crossing.",
    )
    parser.add_argument(
        "--pair-cross-y-band",
        type=float,
        default=1.35,
        help="Maximum |y-center_y| for endpoints when forcing obstacle crossing.",
    )
    parser.add_argument("--obstacle-cx", type=float, default=0.0, help="Obstacle center x in workspace.")
    parser.add_argument("--obstacle-cy", type=float, default=-0.7, help="Obstacle center y in workspace.")
    parser.add_argument("--obstacle-radius", type=float, default=0.5, help="Obstacle radius in xy plane.")
    parser.add_argument("--obstacle-margin", type=float, default=0.15, help="Safety margin in traj_opt.")
    parser.add_argument("--lam-obstacle", type=float, default=20.0, help="Obstacle penalty weight in traj_opt.")
    parser.add_argument("--opt-steps", type=int, default=1240, help="traj_opt iterations.")
    parser.add_argument("--opt-lr", type=float, default=0.01, help="traj_opt learning rate.")
    parser.add_argument("--lam-manifold", type=float, default=1.0, help="Manifold loss weight.")
    parser.add_argument("--lam-len-joint", type=float, default=0.4, help="Path length loss weight.")
    parser.add_argument("--lam-smooth", type=float, default=0.2, help="Smoothness loss weight.")
    parser.add_argument("--trust-scale", type=float, default=0.8, help="Trust-region step cap.")
    parser.add_argument("--proj-steps", type=int, default=120, help="Projector steps for plotting/projection config.")
    parser.add_argument("--proj-alpha", type=float, default=0.3, help="Projector alpha.")
    parser.add_argument("--proj-min-steps", type=int, default=30, help="Projector minimum steps.")
    parser.add_argument("--surface-points", type=int, default=5000, help="Projected points used to render the surface.")
    parser.add_argument("--surface-grid", type=int, default=120, help="Grid resolution for surface rendering.")
    parser.add_argument("--surface-knn", type=int, default=40, help="KNN count for learned surface interpolation.")
    parser.add_argument(
        "--surface-mask-percentile",
        type=float,
        default=93.0,
        help="Percentile threshold for masking unsupported learned-surface regions.",
    )
    parser.add_argument(
        "--surface-source",
        choices=["true", "learned"],
        default="true",
        help="Use true or learned surface for the paper-style trajectory plot.",
    )
    parser.add_argument(
        "--plot-exec-max-points",
        type=int,
        default=180,
        help="Maximum number of executed trajectory points kept in the 3D planning plot.",
    )
    parser.add_argument(
        "--plot-n-paths",
        type=int,
        default=4,
        help="Number of trajectories to plan/render, and also the number of planned/executed path pairs shown in the static planning plot.",
    )
    parser.add_argument(
        "--ik-iters", type=int, default=64, help="Per-point DLS IK iterations."
    )
    parser.add_argument("--ik-method", choices=["pybullet", "dls"], default="pybullet", help="IK backend.")
    parser.add_argument("--ik-damping", type=float, default=0.05, help="DLS damping coefficient.")
    parser.add_argument("--ik-step-size", type=float, default=0.6, help="DLS step size.")
    parser.add_argument("--ik-max-delta", type=float, default=0.18, help="Max joint update norm per IK iteration.")
    parser.add_argument("--ik-pos-tol", type=float, default=0.004, help="IK position tolerance in meters.")
    parser.add_argument("--ik-ori-tol-deg", type=float, default=2.5, help="IK orientation tolerance in degrees.")
    parser.add_argument("--sim-dt", type=float, default=1.0 / 240.0, help="PyBullet simulation step.")
    parser.add_argument("--max-joint-speed", type=float, default=0.9, help="Joint speed limit for time parameterization.")
    parser.add_argument("--min-segment-time", type=float, default=0.04, help="Minimum time per path segment.")
    parser.add_argument("--endpoint-ramp-time", type=float, default=0.20, help="Ease-in/out duration for the joint reference endpoints.")
    parser.add_argument("--reference-smooth-passes", type=int, default=12, help="Number of light smoothing passes applied to the joint reference.")
    parser.add_argument("--terminal-hold-time", type=float, default=0.35, help="Extra hold time at the final reference state so the servo settles before reset.")
    parser.add_argument(
        "--trajectory-time-scale",
        type=float,
        default=1.0,
        help="Scale the reference trajectory duration by this factor without changing controller update logic.",
    )
    parser.add_argument("--position-gain", type=float, default=0.32, help="Joint position servo gain.")
    parser.add_argument("--velocity-gain", type=float, default=1.05, help="Joint velocity servo gain.")
    parser.add_argument("--max-force", type=float, default=140.0, help="Per-joint max servo force.")
    parser.add_argument("--settle-steps", type=int, default=24, help="Extra settle steps before tracking.")
    parser.add_argument("--draw-ee-trace", type=int, default=1, help="1 to draw executed end-effector trace in GUI.")
    parser.add_argument("--draw-ref-trace", type=int, default=0, help="1 to draw planned end-effector trace in GUI.")
    parser.add_argument(
        "--keep-traces",
        type=int,
        default=0,
        help="1 to preserve previously drawn trajectory traces instead of clearing them between segments.",
    )
    parser.add_argument("--draw-surface-wireframe", type=int, default=1, help="1 to draw the local sine surface patch in PyBullet.")
    parser.add_argument("--hide-gripper", type=int, default=1, help="1 to hide the Robotiq gripper and render the probe as the visible end-effector.")
    parser.add_argument("--trace-stride", type=int, default=16, help="Draw one line segment every this many control samples.")
    parser.add_argument("--trace-width", type=float, default=3.0, help="Debug line width for trajectory traces.")
    parser.add_argument("--surface-line-stride", type=int, default=4, help="Grid stride for sine-surface wireframe rendering.")
    parser.add_argument("--surface-line-width", type=float, default=1.2, help="Line width for sine-surface wireframe.")
    parser.add_argument(
        "--gui",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="0: no render/video; 1: offscreen render and save video; 2: GUI render only, no video logging.",
    )
    parser.add_argument("--realtime", type=int, default=0, help="1 to sleep between sim steps.")
    parser.add_argument("--video-name", default="sinepose_ur5_tracking.mp4", help="MP4 filename.")
    parser.add_argument("--video-slowdown", type=float, default=1.0, help="Slow down saved video playback by this factor without changing control.")
    parser.add_argument("--save-error-video", type=int, default=1, help="1 to save a synchronized error-curve video.")
    args = parser.parse_args()

    ckpt_path = _resolve_path(args.ckpt)
    outdir = _resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(int(args.seed))
    device = _choose_device(str(args.device))
    model, ckpt = _load_model(ckpt_path, device=device)
    surface_cfg = SimpleNamespace(**dict(ckpt.get("cfg", {})))
    _, pool = _build_data_pool(ckpt, seed=int(args.seed))
    surface_scale, surface_offset = sine_surface_affine_params(surface_cfg)
    obstacle_center = sine_surface_apply_affine_xy(
        np.asarray([[float(args.obstacle_cx), float(args.obstacle_cy)]], dtype=np.float32),
        surface_cfg,
    )[0]
    obstacle_radius = float(sine_surface_apply_affine_scalar(float(args.obstacle_radius), surface_cfg))
    obstacle_margin = float(sine_surface_apply_affine_scalar(float(args.obstacle_margin), surface_cfg))
    pair_min_dist = float(sine_surface_apply_affine_scalar(float(args.pair_min_dist), surface_cfg))
    pair_max_dist = float(sine_surface_apply_affine_scalar(float(args.pair_max_dist), surface_cfg))
    pair_max_y = float(float(args.pair_max_y) * float(surface_scale) + float(surface_offset[1]))
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
        obstacle_center_xy=(float(obstacle_center[0]), float(obstacle_center[1])),
        obstacle_radius=float(obstacle_radius),
        obstacle_margin=float(obstacle_margin),
        lam_obstacle=float(args.lam_obstacle),
    )

    n_render_trajs = int(max(1, args.plot_n_paths))
    n_force_cross = int(np.clip(args.n_force_cross_trajs, 0, n_render_trajs))
    rng = np.random.default_rng(int(args.seed))
    selected_endpoints: list[np.ndarray] = []
    cases: list[PairCase] = []
    if n_force_cross > 0:
        cases.extend(
            _plan_with_obstacle(
                model=model,
                cfg=cfg,
                pool=pool,
                rng=rng,
                n_trajs=n_force_cross,
                n_waypoints=int(args.n_waypoints),
                planner_mode=str(args.planner_mode),
                center=obstacle_center.astype(np.float32),
                radius=float(obstacle_radius),
                min_dist=float(pair_min_dist),
                max_dist=float(pair_max_dist),
                max_y=float(pair_max_y),
                diverse_min_dist=0.9,
                force_cross_obstacle=bool(args.pair_force_cross_obstacle),
                cross_radius_scale=float(args.pair_cross_radius_scale),
                cross_side_margin=float(pair_cross_side_margin),
                cross_y_band=float(pair_cross_y_band),
                pair_tries=int(args.pair_tries),
                selected_endpoints=selected_endpoints,
            )
        )
    n_free = int(max(0, n_render_trajs - n_force_cross))
    if n_free > 0:
        cases.extend(
            _plan_with_obstacle(
                model=model,
                cfg=cfg,
                pool=pool,
                rng=rng,
                n_trajs=n_free,
                n_waypoints=int(args.n_waypoints),
                planner_mode=str(args.planner_mode),
                center=obstacle_center.astype(np.float32),
                radius=float(obstacle_radius),
                min_dist=float(pair_min_dist),
                max_dist=float(pair_max_dist),
                max_y=float(pair_max_y),
                diverse_min_dist=0.9,
                force_cross_obstacle=False,
                cross_radius_scale=float(args.pair_cross_radius_scale),
                cross_side_margin=float(pair_cross_side_margin),
                cross_y_band=float(pair_cross_y_band),
                pair_tries=int(args.pair_tries),
                selected_endpoints=selected_endpoints,
            )
        )

    ik_cfg = IKConfig(
        method=str(args.ik_method),
        max_iters=int(args.ik_iters),
        damping=float(args.ik_damping),
        step_size=float(args.ik_step_size),
        max_delta_norm=float(args.ik_max_delta),
        pos_tol=float(args.ik_pos_tol),
        ori_tol_rad=math.radians(float(args.ik_ori_tol_deg)),
    )
    track_cfg = JointTrackConfig(
        sim_dt=float(args.sim_dt),
        max_joint_speed=float(args.max_joint_speed),
        min_segment_time=float(args.min_segment_time),
        trajectory_time_scale=float(args.trajectory_time_scale),
        endpoint_ramp_time=float(args.endpoint_ramp_time),
        reference_smooth_passes=int(args.reference_smooth_passes),
        terminal_hold_time=float(args.terminal_hold_time),
        position_gain=float(args.position_gain),
        velocity_gain=float(args.velocity_gain),
        max_force=float(args.max_force),
        settle_steps=int(args.settle_steps),
        draw_ee_trace=bool(args.draw_ee_trace),
        draw_ref_trace=bool(args.draw_ref_trace),
        draw_surface_wireframe=bool(args.draw_surface_wireframe),
        preserve_trace_history=bool(args.keep_traces),
        enable_keyboard_pause=True,
        trace_stride=int(args.trace_stride),
        trace_width=float(args.trace_width),
        surface_line_stride=int(args.surface_line_stride),
        surface_line_width=float(args.surface_line_width),
        realtime=bool(args.realtime),
        video_slowdown=float(args.video_slowdown),
    )

    save_video = int(args.gui) == 1
    use_gui = int(args.gui) == 2
    final_video_path = os.path.join(outdir, str(args.video_name)) if save_video else None
    final_error_video_path = None
    if int(args.save_error_video) == 1:
        video_root, video_ext = os.path.splitext(str(args.video_name))
        final_error_video_path = os.path.join(outdir, f"{video_root}_errors{video_ext or '.mp4'}")
    segment_video_paths: list[str] = []
    segment_error_video_paths: list[str] = []
    segment_metrics: list[dict[str, object]] = []
    exec_cases: list[PairCase] = []
    plan_cases: list[PairCase] = []
    error_plot_segments: list[dict[str, np.ndarray | bool]] = []
    time_chunks: list[np.ndarray] = []
    pos_track_err_chunks: list[np.ndarray] = []
    ori_track_err_chunks: list[np.ndarray] = []
    joint_err_chunks: list[np.ndarray] = []
    surface_pos_err_chunks: list[np.ndarray] = []
    surface_ori_err_chunks: list[np.ndarray] = []
    planned_surface_pos_err_chunks: list[np.ndarray] = []
    planned_surface_ori_err_chunks: list[np.ndarray] = []
    arrays_to_save: dict[str, np.ndarray] = {}
    with UR5TrajectoryController(
        gui=use_gui,
        hide_gripper=bool(args.hide_gripper),
        sim_dt=float(args.sim_dt),
    ) as ctrl:
        ctrl.set_camera(
            distance=1.55,
            yaw=38.0,
            pitch=-40.0,
            target_position=[0.10, -0.18, 1.02],
            fov=52.0,
        )
        time_offset = 0.0
        tool_off_rpy = np.asarray(ik_cfg.tool_frame_offset_rpy, dtype=np.float32).reshape(1, 3)
        tool_off_quat = _rpy_to_quat_xyzw(tool_off_rpy[0])
        tool_off_inv = _quat_inverse_xyzw(tool_off_quat[None, :])[0]
        common_pose_path = np.concatenate([c.traj.astype(np.float32) for c in cases], axis=0)
        common_surface_xyz_grid = _make_surface_patch_xyz_grid(
            common_pose_path,
            surface_cfg=surface_cfg,
        )
        for case_idx, case in enumerate(cases):
            pose_path_raw = case.traj.astype(np.float32)
            pose_path = pose_path_raw.copy()
            plan_cases.append(
                PairCase(
                    start=pose_path[0].astype(np.float32),
                    goal=pose_path[-1].astype(np.float32),
                    waypoint=None,
                    traj=pose_path.astype(np.float32),
                    plan_seconds=float(case.plan_seconds),
                    min_obstacle_dist_xy=float(case.min_obstacle_dist_xy),
                )
            )
            planned_surface_pos_err, planned_surface_ori_err_deg = _error_to_true_constraint(
                pose_path,
                surface_cfg=surface_cfg,
            )
            joint_path, ik_summary = ctrl.solve_pose_path_ik(pose_path, cfg=ik_cfg)
            time_s, q_ref, qd_ref = ctrl.time_parameterize_joint_path(joint_path, cfg=track_cfg)
            pose_path_dense = np.zeros((len(time_s), 3), dtype=np.float32)
            t_src_dense = np.linspace(0.0, float(time_s[-1]), num=len(pose_path), dtype=np.float32)
            for j in range(3):
                pose_path_dense[:, j] = np.interp(time_s, t_src_dense, pose_path[:, j]).astype(np.float32)
            segment_video_path = None
            if final_video_path:
                root, ext = os.path.splitext(final_video_path)
                segment_video_path = f"{root}_segment_{case_idx + 1:02d}{ext or '.mp4'}"
            segment_error_video_path = None
            if final_error_video_path:
                root, ext = os.path.splitext(final_error_video_path)
                segment_error_video_path = f"{root}_segment_{case_idx + 1:02d}{ext or '.mp4'}"
            track = ctrl.track_joint_trajectory(
                q_ref,
                qd_ref=qd_ref,
                cfg=track_cfg,
                video_path=segment_video_path,
                ee_ref_pos=pose_path_dense,
                surface_xyz_grid=common_surface_xyz_grid,
                obstacle_center_xy=obstacle_center.astype(np.float32),
                obstacle_radius=float(obstacle_radius),
                reuse_static_scene=True,
            )
            if track.get("video_path"):
                segment_video_paths.append(str(track["video_path"]))

            time_exec = np.arange(len(track["q_meas"]), dtype=np.float32) * float(track["sim_dt"])
            q_des_quat = np.stack([_rpy_to_quat_xyzw(r) for r in pose_path[:, 3:6]], axis=0).astype(np.float32)
            q_des_quat_dense = np.zeros((len(time_exec), 4), dtype=np.float32)
            pos_des_dense = np.zeros((len(time_exec), 3), dtype=np.float32)
            t_src = np.linspace(0.0, float(time_exec[-1]), num=len(pose_path), dtype=np.float32)
            for j in range(3):
                pos_des_dense[:, j] = np.interp(time_exec, t_src, pose_path[:, j]).astype(np.float32)
            for j in range(4):
                q_des_quat_dense[:, j] = np.interp(time_exec, t_src, q_des_quat[:, j]).astype(np.float32)
            q_des_quat_dense /= np.maximum(np.linalg.norm(q_des_quat_dense, axis=1, keepdims=True), 1e-8)

            ee_quat_task = _apply_quat_offset_batch(track["ee_quat_xyzw"], tool_off_inv)
            pos_track_err = np.linalg.norm(track["ee_pos"] - pos_des_dense, axis=1).astype(np.float32)
            ori_track_err_deg = _orientation_error_deg(q_des_quat_dense, ee_quat_task).astype(np.float32)
            exec_rpy_task = _quat_xyzw_to_rpy_zyx(ee_quat_task)
            exec_pose = np.concatenate([track["ee_pos"], exec_rpy_task], axis=1).astype(np.float32)
            exec_pose_plot = _downsample_traj_for_plot(exec_pose, max_points=int(args.plot_exec_max_points))
            surface_pos_err, surface_ori_err_deg = _error_to_surface_constraint(
                exec_pose,
                surface_cfg=surface_cfg,
            )
            if segment_error_video_path is not None:
                err_vid = _make_error_curve_video(
                    pos_err=surface_pos_err.astype(np.float32),
                    ori_err_deg=surface_ori_err_deg.astype(np.float32),
                    sim_dt=float(track["sim_dt"]),
                    video_fps=int(track_cfg.video_fps),
                    video_slowdown=float(track_cfg.video_slowdown),
                    out_path=segment_error_video_path,
                )
                if isinstance(err_vid, str) and err_vid:
                    segment_error_video_paths.append(err_vid)
            exec_case = PairCase(
                start=exec_pose_plot[0].astype(np.float32),
                goal=exec_pose_plot[-1].astype(np.float32),
                waypoint=None,
                traj=exec_pose_plot.astype(np.float32),
                plan_seconds=0.0,
                min_obstacle_dist_xy=float(
                    np.min(
                        np.linalg.norm(
                            exec_pose[:, :2] - obstacle_center.astype(np.float32)[None, :],
                            axis=1,
                        )
                    )
                    - float(obstacle_radius)
                ),
            )
            exec_cases.append(exec_case)
            error_plot_segments.append(
                {
                    "time_s": time_exec.astype(np.float32),
                    "pos_track_err": pos_track_err.astype(np.float32),
                    "axis_align_err_deg": surface_ori_err_deg.astype(np.float32),
                    "joint_err_norm": track["joint_err_norm"].astype(np.float32),
                    "forced_cross_obstacle": bool(case_idx < n_force_cross),
                }
            )
            time_chunks.append((time_exec + float(time_offset)).astype(np.float32))
            pos_track_err_chunks.append(pos_track_err)
            ori_track_err_chunks.append(ori_track_err_deg)
            joint_err_chunks.append(track["joint_err_norm"].astype(np.float32))
            surface_pos_err_chunks.append(surface_pos_err.astype(np.float32))
            surface_ori_err_chunks.append(surface_ori_err_deg.astype(np.float32))
            planned_surface_pos_err_chunks.append(planned_surface_pos_err.astype(np.float32))
            planned_surface_ori_err_chunks.append(planned_surface_ori_err_deg.astype(np.float32))
            segment_metrics.append(
                {
                    "segment_index": case_idx + 1,
                    "forced_cross_obstacle": bool(case_idx < n_force_cross),
                    "planner_waypoints": int(len(pose_path)),
                    "controller_samples": int(len(track["q_meas"])),
                    "plan_seconds": float(case.plan_seconds),
                    "planner_min_obstacle_dist_xy": float(case.min_obstacle_dist_xy),
                    "video_path": track.get("video_path"),
                    "ik": {
                        "first_pos_err": float(ik_summary.get("first_pos_err", 0.0)),
                        "first_ori_err_deg": float(np.degrees(ik_summary.get("first_ori_err_rad", 0.0))),
                        "mean_pos_err": float(ik_summary["mean_pos_err"]),
                        "max_pos_err": float(ik_summary["max_pos_err"]),
                        "mean_ori_err_deg": float(np.degrees(ik_summary["mean_ori_err_rad"])),
                        "max_ori_err_deg": float(np.degrees(ik_summary["max_ori_err_rad"])),
                        "mean_iters": float(ik_summary["mean_iters"]),
                    },
                    "tracking": {
                        "mean_joint_err_norm": float(track["mean_joint_err_norm"]),
                        "max_joint_err_norm": float(track["max_joint_err_norm"]),
                        "mean_pos_track_err": float(np.mean(pos_track_err)),
                        "max_pos_track_err": float(np.max(pos_track_err)),
                        "mean_ori_track_err_deg": float(np.mean(ori_track_err_deg)),
                        "max_ori_track_err_deg": float(np.max(ori_track_err_deg)),
                    },
                    "executed_constraint_error": {
                        "mean_surface_pos_err": float(np.mean(surface_pos_err)),
                        "std_surface_pos_err": float(np.std(surface_pos_err)),
                        "max_surface_pos_err": float(np.max(surface_pos_err)),
                        "mean_surface_ori_err_deg": float(np.mean(surface_ori_err_deg)),
                        "std_surface_ori_err_deg": float(np.std(surface_ori_err_deg)),
                        "max_surface_ori_err_deg": float(np.max(surface_ori_err_deg)),
                    },
                    "planned_constraint_error": {
                        "mean_surface_pos_err": float(np.mean(planned_surface_pos_err)),
                        "std_surface_pos_err": float(np.std(planned_surface_pos_err)),
                        "max_surface_pos_err": float(np.max(planned_surface_pos_err)),
                        "mean_surface_ori_err_deg": float(np.mean(planned_surface_ori_err_deg)),
                        "std_surface_ori_err_deg": float(np.std(planned_surface_ori_err_deg)),
                        "max_surface_ori_err_deg": float(np.max(planned_surface_ori_err_deg)),
                    },
                }
            )
            arrays_to_save[f"segment_{case_idx + 1:02d}_pose_path"] = pose_path.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_joint_path"] = joint_path.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_time_s"] = time_s.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_time_exec"] = time_exec.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_q_ref"] = q_ref.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_qd_ref"] = qd_ref.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_q_meas"] = track["q_meas"].astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_ee_pos"] = track["ee_pos"].astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_ee_quat_xyzw"] = track["ee_quat_xyzw"].astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_ee_quat_task_xyzw"] = ee_quat_task.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_pos_track_err"] = pos_track_err.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_ori_track_err_deg"] = ori_track_err_deg.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_planned_surface_pos_err"] = planned_surface_pos_err.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_planned_surface_ori_err_deg"] = planned_surface_ori_err_deg.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_surface_pos_err"] = surface_pos_err.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_surface_ori_err_deg"] = surface_ori_err_deg.astype(np.float32)
            arrays_to_save[f"segment_{case_idx + 1:02d}_exec_pose"] = exec_pose.astype(np.float32)
            time_offset += float(time_exec[-1]) + float(track_cfg.sim_dt)

    time_exec_all = np.concatenate(time_chunks, axis=0).astype(np.float32)
    pos_track_err_all = np.concatenate(pos_track_err_chunks, axis=0).astype(np.float32)
    ori_track_err_deg_all = np.concatenate(ori_track_err_chunks, axis=0).astype(np.float32)
    joint_err_all = np.concatenate(joint_err_chunks, axis=0).astype(np.float32)
    planned_surface_pos_err_all = np.concatenate(planned_surface_pos_err_chunks, axis=0).astype(np.float32)
    planned_surface_ori_err_deg_all = np.concatenate(planned_surface_ori_err_chunks, axis=0).astype(np.float32)
    surface_pos_err_all = np.concatenate(surface_pos_err_chunks, axis=0).astype(np.float32)
    surface_ori_err_deg_all = np.concatenate(surface_ori_err_chunks, axis=0).astype(np.float32)

    error_plot = os.path.join(outdir, "sinepose_ur5_tracking_errors.png")
    _plot_tracking_errors(
        segment_series=error_plot_segments,
        out_path=error_plot,
    )
    traj_plot = os.path.join(outdir, "sinepose_tracking_obstacle_planning.png")
    n_plot_paths = int(max(1, min(n_render_trajs, len(plan_cases), len(exec_cases))))
    plan_plot_cases = plan_cases[:n_plot_paths]
    exec_plot_cases = exec_cases[:n_plot_paths]
    plot_cases = plan_plot_cases + exec_plot_cases
    plot_labels = [f"Planned path {i + 1}" for i in range(len(plan_plot_cases))] + [
        f"Executed path {i + 1}" for i in range(len(exec_plot_cases))
    ]
    plan_colors = ["#94a3b8", "#64748b", "#475569", "#334155"][: len(plan_plot_cases)]
    exec_colors = ["#2563eb", "#0f766e", "#d97706", "#7c3aed"][: len(exec_plot_cases)]
    plot_colors = plan_colors + exec_colors
    plot_linestyles = ["--"] * len(plan_plot_cases) + ["-"] * len(exec_plot_cases)
    _plot_planning_paper(
        model=model,
        cfg=cfg,
        pool=pool,
        surface_points=int(args.surface_points),
        surface_grid=int(args.surface_grid),
        surface_knn=int(args.surface_knn),
        surface_mask_percentile=float(args.surface_mask_percentile),
        surface_source=str(args.surface_source),
        cases=plot_cases,
        center=obstacle_center.astype(np.float32),
        radius=float(obstacle_radius),
        out_path=traj_plot,
        traj_labels=plot_labels,
        traj_colors=plot_colors,
        traj_linestyles=plot_linestyles,
        global_paper_view=True,
        surface_cfg=surface_cfg,
        orientation_arrow_length=0.045,
    )
    dist_plot = os.path.join(outdir, "6d_surf_normal_traj_oncl_tracking_error_distribution_paper.png")
    _plot_error_distribution_paper(
        pos_err=planned_surface_pos_err_all.astype(np.float32),
        ang_err_deg=planned_surface_ori_err_deg_all.astype(np.float32),
        pos_err_compare=surface_pos_err_all.astype(np.float32),
        ang_err_deg_compare=surface_ori_err_deg_all.astype(np.float32),
        primary_label="Planned",
        compare_label="Executed",
        out_path=dist_plot,
    )

    merged_video_path = _concat_videos_ffmpeg(segment_video_paths, final_video_path) if final_video_path else None
    if merged_video_path:
        _cleanup_segment_videos(segment_video_paths, merged_video_path)
    merged_error_video_path = _concat_videos_ffmpeg(segment_error_video_paths, final_error_video_path) if final_error_video_path else None
    if merged_error_video_path:
        _cleanup_segment_videos(segment_error_video_paths, merged_error_video_path)
    np.savez_compressed(
        os.path.join(outdir, "sinepose_ur5_tracking_arrays.npz"),
        time_exec_all=time_exec_all,
        pos_track_err_all=pos_track_err_all,
        ori_track_err_deg_all=ori_track_err_deg_all,
        joint_err_all=joint_err_all,
        planned_surface_pos_err_all=planned_surface_pos_err_all,
        planned_surface_ori_err_deg_all=planned_surface_ori_err_deg_all,
        surface_pos_err_all=surface_pos_err_all,
        surface_ori_err_deg_all=surface_ori_err_deg_all,
        **arrays_to_save,
    )

    mean_ik_pos = float(np.mean([seg["ik"]["mean_pos_err"] for seg in segment_metrics])) if segment_metrics else 0.0
    mean_track_pos = float(np.mean([seg["tracking"]["mean_pos_track_err"] for seg in segment_metrics])) if segment_metrics else 0.0
    mean_track_ori = float(np.mean([seg["tracking"]["mean_ori_track_err_deg"] for seg in segment_metrics])) if segment_metrics else 0.0
    mean_joint_err = float(np.mean([seg["tracking"]["mean_joint_err_norm"] for seg in segment_metrics])) if segment_metrics else 0.0
    summary = {
        "task": "sinepose_ur5_tracking",
        "ckpt": ckpt_path,
        "video_path": merged_video_path,
        "error_video_path": merged_error_video_path,
        "segment_video_paths": ([] if merged_video_path else segment_video_paths),
        "segment_error_video_paths": ([] if merged_error_video_path else segment_error_video_paths),
        "error_plot": os.path.abspath(error_plot),
        "traj_plot": os.path.abspath(traj_plot),
        "distribution_plot": os.path.abspath(dist_plot),
        "planner_waypoints": int(sum(seg["planner_waypoints"] for seg in segment_metrics)),
        "controller_samples": int(sum(seg["controller_samples"] for seg in segment_metrics)),
        "n_render_trajs": int(len(segment_metrics)),
        "n_force_cross_trajs": int(n_force_cross),
        "environment": {
            "obstacle_center_xy": [float(obstacle_center[0]), float(obstacle_center[1])],
            "obstacle_radius": float(obstacle_radius),
        },
        "ik": {
            "mean_pos_err": mean_ik_pos,
            "max_pos_err": float(max(seg["ik"]["max_pos_err"] for seg in segment_metrics)) if segment_metrics else 0.0,
            "mean_ori_err_deg": float(np.mean([seg["ik"]["mean_ori_err_deg"] for seg in segment_metrics])) if segment_metrics else 0.0,
            "max_ori_err_deg": float(max(seg["ik"]["max_ori_err_deg"] for seg in segment_metrics)) if segment_metrics else 0.0,
            "mean_iters": float(np.mean([seg["ik"]["mean_iters"] for seg in segment_metrics])) if segment_metrics else 0.0,
        },
        "tracking": {
            "mean_joint_err_norm": mean_joint_err,
            "max_joint_err_norm": float(np.max(joint_err_all)) if len(joint_err_all) > 0 else 0.0,
            "mean_pos_track_err": mean_track_pos,
            "max_pos_track_err": float(np.max(pos_track_err_all)) if len(pos_track_err_all) > 0 else 0.0,
            "mean_ori_track_err_deg": mean_track_ori,
            "max_ori_track_err_deg": float(np.max(ori_track_err_deg_all)) if len(ori_track_err_deg_all) > 0 else 0.0,
        },
        "executed_constraint_error": {
            "mean_surface_pos_err": float(np.mean(surface_pos_err_all)) if len(surface_pos_err_all) > 0 else 0.0,
            "std_surface_pos_err": float(np.std(surface_pos_err_all)) if len(surface_pos_err_all) > 0 else 0.0,
            "max_surface_pos_err": float(np.max(surface_pos_err_all)) if len(surface_pos_err_all) > 0 else 0.0,
            "mean_surface_ori_err_deg": float(np.mean(surface_ori_err_deg_all)) if len(surface_ori_err_deg_all) > 0 else 0.0,
            "std_surface_ori_err_deg": float(np.std(surface_ori_err_deg_all)) if len(surface_ori_err_deg_all) > 0 else 0.0,
            "max_surface_ori_err_deg": float(np.max(surface_ori_err_deg_all)) if len(surface_ori_err_deg_all) > 0 else 0.0,
        },
        "mean_position_error": float(np.mean(surface_pos_err_all)) if len(surface_pos_err_all) > 0 else 0.0,
        "std_position_error": float(np.std(surface_pos_err_all)) if len(surface_pos_err_all) > 0 else 0.0,
        "mean_orientation_error_deg": float(np.mean(surface_ori_err_deg_all)) if len(surface_ori_err_deg_all) > 0 else 0.0,
        "std_orientation_error_deg": float(np.std(surface_ori_err_deg_all)) if len(surface_ori_err_deg_all) > 0 else 0.0,
        "controller_cfg": {
            "sim_dt": float(track_cfg.sim_dt),
            "max_joint_speed": float(track_cfg.max_joint_speed),
            "min_segment_time": float(track_cfg.min_segment_time),
            "trajectory_time_scale": float(track_cfg.trajectory_time_scale),
            "endpoint_ramp_time": float(track_cfg.endpoint_ramp_time),
            "reference_smooth_passes": int(track_cfg.reference_smooth_passes),
            "terminal_hold_time": float(track_cfg.terminal_hold_time),
            "position_gain": float(track_cfg.position_gain),
            "velocity_gain": float(track_cfg.velocity_gain),
            "max_force": float(track_cfg.max_force),
            "settle_steps": int(track_cfg.settle_steps),
            "draw_ee_trace": bool(track_cfg.draw_ee_trace),
            "draw_ref_trace": bool(track_cfg.draw_ref_trace),
            "trace_stride": int(track_cfg.trace_stride),
            "trace_width": float(track_cfg.trace_width),
            "video_slowdown": float(track_cfg.video_slowdown),
        },
        "ik_cfg": {
            "method": str(ik_cfg.method),
            "max_iters": int(ik_cfg.max_iters),
            "damping": float(ik_cfg.damping),
            "step_size": float(ik_cfg.step_size),
            "max_delta_norm": float(ik_cfg.max_delta_norm),
            "pos_tol": float(ik_cfg.pos_tol),
            "ori_tol_deg": float(np.degrees(ik_cfg.ori_tol_rad)),
            "search_first_seed": bool(ik_cfg.search_first_seed),
            "seed_search_samples": int(ik_cfg.seed_search_samples),
            "seed_search_ori_weight": float(ik_cfg.seed_search_ori_weight),
            "tool_frame_offset_rpy": [float(v) for v in ik_cfg.tool_frame_offset_rpy],
        },
        "segments": segment_metrics,
    }
    summary_path = os.path.join(outdir, "sinepose_ur5_tracking_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        "[summary] "
        f"ik_pos_mean={summary['ik']['mean_pos_err']:.5f}, "
        f"track_pos_mean={summary['tracking']['mean_pos_track_err']:.5f}, "
        f"track_ori_mean_deg={summary['tracking']['mean_ori_track_err_deg']:.3f}, "
        f"joint_err_mean={summary['tracking']['mean_joint_err_norm']:.5f}, "
        f"exec_surface_pos_mean={summary['mean_position_error']:.5f}, "
        f"exec_surface_pos_std={summary['std_position_error']:.5f}, "
        f"exec_surface_ori_mean_deg={summary['mean_orientation_error_deg']:.3f}, "
        f"exec_surface_ori_std_deg={summary['std_orientation_error_deg']:.3f}"
    )
    print(f"[saved] {summary_path}")
    print(f"[saved] {error_plot}")
    print(f"[saved] {traj_plot}")
    print(f"[saved] {dist_plot}")
    if merged_video_path:
        print(f"[saved] {merged_video_path}")
    if merged_error_video_path:
        print(f"[saved] {merged_error_video_path}")


if __name__ == "__main__":
    main()
