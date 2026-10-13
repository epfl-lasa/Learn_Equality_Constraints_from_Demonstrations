#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.fixed_demonstrations import load_fixed_demonstrations
from datasets.ur5_pybullet_utils import resolve_dual_ur5_base_cfg
from simulation.sim_dual_arm_ur5_demo import (
    DualUR5DemoSim,
    _compose_task_local_offset_quat_xyzw,
    _left_task_grasp_flip_xyzw,
    _load_planned_path_npz,
    _parse_rpy_deg_triplet,
    _right_task_grasp_flip_xyzw,
    _resolve_path,
)
from evaluation.evaluator import dual_arm_pose_true_constraint_error_arrays
from simulation.ik_controller import _FFmpegVideoWriter


DATASET_NAME = "12d_dual_arm_traj"
DEMO_SPEED_MULTIPLIER = 3.0


def _summary_stats(vals: np.ndarray) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max": float(np.max(arr)),
    }


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
        qa = q[i]
        ax, ay, az, aw = [float(v) for v in qa.reshape(4)]
        bx, by, bz, bw = [float(v) for v in q_off.reshape(4)]
        out[i] = np.asarray(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float32,
        )
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out.astype(np.float32)


def _quat_xyzw_to_rpy_zyx_batch(q_xyzw: np.ndarray) -> np.ndarray:
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


def _parse_snapshot_steps(text: str) -> list[int]:
    raw = str(text).strip()
    if not raw:
        return []
    vals: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        vals.append(int(token))
    return vals


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


def _sampled_frame_indices(n_steps: int, *, sim_dt: float, video_fps: int) -> np.ndarray:
    n = int(max(0, n_steps))
    if n <= 0:
        return np.zeros((0,), dtype=np.int32)
    capture_every = max(1, int(round(1.0 / max(float(sim_dt) * float(video_fps), 1e-8))))
    idx = [0]
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
    title: str,
    width: int = 760,
    height: int = 430,
) -> str | None:
    pos = np.asarray(pos_err, dtype=np.float32).reshape(-1)
    ori = np.asarray(ori_err_deg, dtype=np.float32).reshape(-1)
    n = int(min(len(pos), len(ori)))
    if n <= 0:
        return None
    idx = _sampled_frame_indices(n, sim_dt=float(sim_dt), video_fps=int(video_fps))
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
    title_fs = 12
    value_fs = 15
    axes[0].text(
        0.01, 0.96, "Position error to true constraint", transform=axes[0].transAxes,
        fontsize=title_fs, fontweight="bold", color="#1e3a8a", va="top"
    )
    axes[1].text(
        0.01, 0.96, "Orientation error to true constraint", transform=axes[1].transAxes,
        fontsize=title_fs, fontweight="bold", color="#991b1b", va="top"
    )
    value_pos = axes[0].text(
        0.99, 0.92, "", transform=axes[0].transAxes,
        fontsize=value_fs, fontweight="bold", color="#1d4ed8", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="#dbeafe", ec="#93c5fd", lw=0.9)
    )
    value_ori = axes[1].text(
        0.99, 0.92, "", transform=axes[1].transAxes,
        fontsize=value_fs, fontweight="bold", color="#b91c1c", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fee2e2", ec="#fca5a5", lw=0.9)
    )
    fig.subplots_adjust(left=0.10, right=0.992, bottom=0.11, top=0.985, hspace=0.34)
    try:
        for k in idx.tolist():
            k = int(np.clip(k, 0, n - 1))
            kk = k + 1
            line_pos.set_data(t[:kk], pos_mm[:kk])
            dot_pos.set_data([t[k]], [pos_mm[k]])
            vline_pos.set_xdata([t[k], t[k]])
            line_ori.set_data(t[:kk], ori[:kk])
            dot_ori.set_data([t[k]], [ori[k]])
            vline_ori.set_xdata([t[k], t[k]])
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


def _path_from_12d_pose_array(path12: np.ndarray) -> dict[str, np.ndarray]:
    path = np.asarray(path12, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] < 12:
        raise ValueError(f"expected path with shape (T,12), got {path.shape}")
    pose_left = path[:, 0:6].astype(np.float32)
    pose_right = path[:, 6:12].astype(np.float32)
    p_left = pose_left[:, 0:3].astype(np.float32)
    p_right = pose_right[:, 0:3].astype(np.float32)
    center = (0.5 * (p_left + p_right)).astype(np.float32)
    return {
        "pose_left": pose_left,
        "pose_right": pose_right,
        "center": center,
        "p_left": p_left,
        "p_right": p_right,
    }


def _load_train_demo_path(cfg: SimpleNamespace, *, demo_index: int) -> dict[str, np.ndarray]:
    blocks = _load_train_demo_blocks(cfg)
    idx = int(np.clip(int(demo_index), 0, len(blocks) - 1))
    return _path_from_12d_pose_array(blocks[idx].astype(np.float32))


def _load_train_demo_blocks(_cfg: SimpleNamespace) -> list[np.ndarray]:
    demo = load_fixed_demonstrations(DATASET_NAME)
    stops = np.cumsum(demo["trajectory_lengths"])
    starts = np.concatenate(([0], stops[:-1]))
    return [demo["x_train"][start:stop].copy() for start, stop in zip(starts, stops)]


def _parse_int_list(text: str) -> list[int]:
    raw = str(text).strip()
    if not raw:
        return []
    out: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _parse_crop_frac(text: str) -> tuple[float, float, float, float] | None:
    raw = str(text).strip()
    if not raw:
        return None
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError("snapshot crop must have four comma-separated fractions: x0,y0,x1,y1")
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render a planned 12D dual-arm path in PyBullet.")
    ap.add_argument("--gui", type=int, default=1, help="0: no video, 1: DIRECT render video, 2: GUI render only")
    ap.add_argument("--dataset-config", type=str, default="configs/datasets/12d_dual_arm_traj.json")
    ap.add_argument(
        "--path-source",
        type=str,
        default="planned",
        choices=("planned", "demo"),
        help="planned: load a path from dual_arm_learned_plans.npz; demo: render training demonstrations.",
    )
    ap.add_argument(
        "--plan-outdir",
        type=str,
        default="",
        help="When --path-source=planned, planner output dir containing dual_arm_learned_plans.npz. Render outputs are also written here.",
    )
    ap.add_argument("--planned-path-npz", type=str, default="", help="Optional explicit .npz path; overrides loading from --plan-outdir")
    ap.add_argument("--planned-path-key", type=str, default="path_0", help="Path key inside planned-path-npz")
    ap.add_argument(
        "--render-outdir",
        type=str,
        default="",
        help="Optional explicit render output dir. Defaults to --plan-outdir for planned mode, or outputs/bench/dual_arm_demo_render for demo mode.",
    )
    ap.add_argument(
        "--demo-index",
        type=int,
        default=-1,
        help="Single training demo index when --path-source=demo. If omitted, uses the first --demo-count demos.",
    )
    ap.add_argument(
        "--demo-indices",
        type=str,
        default="",
        help="Optional comma-separated training demo indices to render sequentially, e.g. 0,3,5.",
    )
    ap.add_argument(
        "--demo-count",
        type=int,
        default=5,
        help="When --path-source=demo and no explicit demo index list is given, render the first N training demos.",
    )
    ap.add_argument(
        "--orientation-mode",
        type=str,
        default="task",
        choices=("none", "task"),
        help="IK orientation target for tracking the planned task-space path.",
    )
    ap.add_argument(
        "--task-roll-deg",
        type=float,
        default=0.0,
        help="Fixed extra roll offset in degrees about the task-frame tangent axis when --orientation-mode=task.",
    )
    ap.add_argument(
        "--task-offset-rpy-deg",
        type=str,
        default="0,0,90",
        help="Extra fixed local task-frame RPY offset in degrees as roll,pitch,yaw. Applied after --task-roll-deg.",
    )
    ap.add_argument("--ik-iters", type=int, default=160)
    ap.add_argument("--sim-dt", type=float, default=1.0 / 180.0)
    ap.add_argument("--max-joint-speed", type=float, default=0.75)
    ap.add_argument("--min-segment-time", type=float, default=0.06)
    ap.add_argument("--terminal-hold-time", type=float, default=0.45)
    ap.add_argument("--max-force", type=float, default=260.0)
    ap.add_argument("--video-width", type=int, default=1280)
    ap.add_argument("--video-height", type=int, default=900)
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--video-slowdown", type=float, default=1.2)
    ap.add_argument("--save-error-video", type=int, default=1, help="1 to save a synchronized error-curve video.")
    ap.add_argument("--trace-stride", type=int, default=18)
    ap.add_argument(
        "--arm-trace-radius",
        type=float,
        default=0.0,
        help="Rendered radius for the two arm end-effector traces. Set <=0 to disable; default disables arm traces.",
    )
    ap.add_argument("--save-snapshots", type=int, default=0, help="1 to save PNG snapshots using the same camera as gui=1 video rendering.")
    ap.add_argument("--snapshot-steps", type=str, default="", help="Comma-separated control-step indices for snapshots, e.g. 40,120,240.")
    ap.add_argument("--snapshot-dir", type=str, default="", help="Optional snapshot output directory. Defaults to <plan-outdir>/snapshots.")
    ap.add_argument(
        "--snapshot-crop-frac",
        type=str,
        default="0.10,0.08,0.86,0.80",
        help="Relative crop box for saved snapshots only, formatted as x0,y0,x1,y1.",
    )
    ap.add_argument("--realtime", type=int, default=0)
    return ap.parse_args()


def _run_segment(
    *,
    path: dict[str, np.ndarray],
    cfg: SimpleNamespace,
    base_cfg: dict[str, list[float]],
    gui_mode: int,
    orientation_mode: str,
    task_offset_quat: np.ndarray,
    ik_iters: int,
    sim_dt: float,
    max_joint_speed: float,
    min_segment_time: float,
    terminal_hold_time: float,
    max_force: float,
    video_path: str | None,
    video_width: int,
    video_height: int,
    video_fps: int,
    video_slowdown: float,
    realtime: bool,
    trace_stride: int,
    arm_trace_radius: float,
    save_snapshots: bool,
    snapshot_steps: list[int] | None,
    snapshot_dir: str | None,
    snapshot_crop_frac: tuple[float, float, float, float] | None,
) -> dict[str, object]:
    sim = DualUR5DemoSim(
        gui=(int(gui_mode) == 2),
        urdf_path=None,
        ee_link_index=None,
        tool_axis=None,
        base_cfg=base_cfg,
        sim_dt=float(sim_dt),
    )
    try:
        sim.build_guide_visuals(path["center"], path["p_left"], path["p_right"], cfg=cfg, draw_center_guide=False)
        q_left, ik_left = sim.solve_ik_path(
            sim.left,
            path["pose_left"],
            orientation_mode=str(orientation_mode),
            task_offset_quat_xyzw=task_offset_quat,
            side="left",
            max_iters=int(ik_iters),
            seed_offset=0,
        )
        q_right, ik_right = sim.solve_ik_path(
            sim.right,
            path["pose_right"],
            orientation_mode=str(orientation_mode),
            task_offset_quat_xyzw=task_offset_quat,
            side="right",
            max_iters=int(ik_iters),
            seed_offset=101,
        )
        t, ql_ref, qr_ref, qdl_ref, qdr_ref = sim.time_parameterize(
            q_left,
            q_right,
            max_joint_speed=float(max_joint_speed),
            min_segment_time=float(min_segment_time),
            terminal_hold_time=float(terminal_hold_time),
        )
        track = sim.track(
            ql_ref,
            qr_ref,
            qdl_ref,
            qdr_ref,
            video_path=video_path,
            video_width=int(video_width),
            video_height=int(video_height),
            video_fps=int(video_fps),
            video_slowdown=float(video_slowdown),
            realtime=bool(realtime),
            trace_stride=int(trace_stride),
            max_force=float(max_force),
            arm_trace_radius=float(arm_trace_radius),
            draw_object_trace=True,
            object_trace_radius=0.003,
            snapshot_steps=(snapshot_steps if save_snapshots else None),
            snapshot_dir=snapshot_dir,
            snapshot_crop_frac=(snapshot_crop_frac if save_snapshots else None),
        )
        ref_left_pos = path["pose_left"][:, 0:3].astype(float)
        ref_right_pos = path["pose_right"][:, 0:3].astype(float)
        exec_left_pos = np.asarray(track["ee_left"], dtype=float)
        exec_right_pos = np.asarray(track["ee_right"], dtype=float)
        if len(exec_left_pos) != len(ref_left_pos):
            t_ref = np.linspace(0.0, 1.0, len(ref_left_pos), dtype=float)
            t_exec = np.linspace(0.0, 1.0, len(exec_left_pos), dtype=float)
            ref_left_pos = np.stack([np.interp(t_exec, t_ref, ref_left_pos[:, j]) for j in range(3)], axis=1)
            ref_right_pos = np.stack([np.interp(t_exec, t_ref, ref_right_pos[:, j]) for j in range(3)], axis=1)
        exec_left_err = np.linalg.norm(exec_left_pos - ref_left_pos, axis=1).astype(np.float32)
        exec_right_err = np.linalg.norm(exec_right_pos - ref_right_pos, axis=1).astype(np.float32)
        exec_left_quat_task = np.asarray(track["ee_left_quat_xyzw"], dtype=np.float32)
        exec_right_quat_task = np.asarray(track["ee_right_quat_xyzw"], dtype=np.float32)
        left_flip_inv = _quat_inverse_xyzw(_left_task_grasp_flip_xyzw().reshape(1, 4))[0]
        right_flip_inv = _quat_inverse_xyzw(_right_task_grasp_flip_xyzw().reshape(1, 4))[0]
        task_offset_inv = _quat_inverse_xyzw(task_offset_quat.reshape(1, 4))[0]
        exec_left_quat_task = _apply_quat_offset_batch(exec_left_quat_task, left_flip_inv)
        exec_left_quat_task = _apply_quat_offset_batch(exec_left_quat_task, task_offset_inv)
        exec_right_quat_task = _apply_quat_offset_batch(exec_right_quat_task, right_flip_inv)
        exec_right_quat_task = _apply_quat_offset_batch(exec_right_quat_task, task_offset_inv)
        exec_pose_task = np.concatenate(
            [
                np.asarray(track["ee_left"], dtype=np.float32),
                _quat_xyzw_to_rpy_zyx_batch(exec_left_quat_task),
                np.asarray(track["ee_right"], dtype=np.float32),
                _quat_xyzw_to_rpy_zyx_batch(exec_right_quat_task),
            ],
            axis=1,
        ).astype(np.float32)
        exec_true_err = dual_arm_pose_true_constraint_error_arrays(exec_pose_task, cfg)
        return {
            "ik_left": ik_left,
            "ik_right": ik_right,
            "track": track,
            "t": t,
            "video_path": video_path,
            "exec_left_err": exec_left_err,
            "exec_right_err": exec_right_err,
            "exec_true_err": exec_true_err,
        }
    finally:
        sim.close()


def _aggregate_ik_stats(segment_results: list[dict[str, object]], side: str) -> dict[str, float]:
    key = "ik_left" if str(side).strip().lower().startswith("l") else "ik_right"
    mean_pos = np.asarray([float(seg[key]["mean_pos_err"]) for seg in segment_results], dtype=np.float32)
    max_pos = np.asarray([float(seg[key]["max_pos_err"]) for seg in segment_results], dtype=np.float32)
    mean_ori = np.asarray(
        [float(seg[key].get("mean_ori_err_deg", float("nan"))) for seg in segment_results],
        dtype=np.float32,
    )
    max_ori = np.asarray(
        [float(seg[key].get("max_ori_err_deg", float("nan"))) for seg in segment_results],
        dtype=np.float32,
    )
    return {
        "mean_pos_err": float(np.mean(mean_pos)),
        "std_pos_err": float(np.std(mean_pos)),
        "max_pos_err": float(np.max(max_pos)),
        "mean_ori_err_deg": float(np.nanmean(mean_ori)),
        "std_ori_err_deg": float(np.nanstd(mean_ori)),
        "max_ori_err_deg": float(np.nanmax(max_ori)),
    }


def main() -> None:
    args = _parse_args()
    with open(_resolve_path(str(args.dataset_config)), "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = SimpleNamespace(**cfg_dict)
    base_cfg = resolve_dual_ur5_base_cfg(cfg)

    path_source = str(args.path_source).strip().lower()
    npz_path = str(args.planned_path_npz).strip()
    plan_outdir_raw = str(args.plan_outdir).strip()
    render_outdir_raw = str(args.render_outdir).strip()
    summary_path_key = str(args.planned_path_key)
    summary_npz_path: str | None = None
    demo_indices: list[int] = []
    paths_to_render: list[dict[str, np.ndarray]] = []
    segment_labels: list[str] = []

    if path_source == "planned":
        plan_outdir = _resolve_path(plan_outdir_raw) if plan_outdir_raw else ""
        if not npz_path:
            if not plan_outdir_raw:
                raise ValueError("When --path-source=planned, provide either --planned-path-npz or --plan-outdir.")
            npz_path = os.path.join(plan_outdir, "dual_arm_learned_plans.npz")
        else:
            if not plan_outdir_raw:
                plan_outdir = os.path.dirname(_resolve_path(npz_path))
        npz_path = _resolve_path(npz_path)
        path = _load_planned_path_npz(npz_path, str(args.planned_path_key))
        paths_to_render = [path]
        segment_labels = [str(args.planned_path_key)]
        render_outdir = _resolve_path(render_outdir_raw) if render_outdir_raw else plan_outdir
        summary_npz_path = npz_path
    else:
        demo_indices = _parse_int_list(str(args.demo_indices))
        if not demo_indices:
            if int(args.demo_index) >= 0:
                demo_indices = [int(args.demo_index)]
            else:
                demo_indices = list(range(int(max(1, args.demo_count))))
        if not demo_indices:
            raise RuntimeError("no demo indices selected")
        demo_blocks = _load_train_demo_blocks(cfg)
        n_blocks = len(demo_blocks)
        demo_indices = [int(np.clip(int(v), 0, n_blocks - 1)) for v in demo_indices]
        paths_to_render = [_path_from_12d_pose_array(demo_blocks[idx].astype(np.float32)) for idx in demo_indices]
        segment_labels = [f"demo_{idx}" for idx in demo_indices]
        render_outdir = (
            _resolve_path(render_outdir_raw)
            if render_outdir_raw
            else _resolve_path("outputs/bench/dual_arm_demo_render")
        )
        summary_path_key = "demo_indices_" + "_".join(str(int(v)) for v in demo_indices)

    os.makedirs(render_outdir, exist_ok=True)
    task_offset_rpy_deg = _parse_rpy_deg_triplet(str(args.task_offset_rpy_deg))
    task_offset_quat = _compose_task_local_offset_quat_xyzw(
        task_roll_rad=np.deg2rad(float(args.task_roll_deg)),
        extra_rpy_rad=np.deg2rad(task_offset_rpy_deg.astype(np.float32)),
    )
    snapshot_steps = [] if path_source == "demo" else _parse_snapshot_steps(str(args.snapshot_steps))
    snapshot_dir = _resolve_path(str(args.snapshot_dir)) if str(args.snapshot_dir).strip() else os.path.join(render_outdir, "snapshots")
    snapshot_crop_frac = _parse_crop_frac(str(args.snapshot_crop_frac))

    final_video_path = None
    final_error_video_path = None
    if int(args.gui) == 1:
        video_name = "dual_arm_demo_path.mp4" if path_source == "demo" else "dual_arm_planned_path.mp4"
        final_video_path = os.path.join(render_outdir, video_name)
    if int(args.save_error_video) == 1:
        error_video_name = "dual_arm_demo_path_errors.mp4" if path_source == "demo" else "dual_arm_planned_path_errors.mp4"
        final_error_video_path = os.path.join(render_outdir, error_video_name)

    max_joint_speed = float(args.max_joint_speed)
    min_segment_time = float(args.min_segment_time)
    terminal_hold_time = float(args.terminal_hold_time)
    video_slowdown = float(args.video_slowdown)
    if path_source == "demo":
        max_joint_speed = max_joint_speed * DEMO_SPEED_MULTIPLIER
        min_segment_time = max(min_segment_time / DEMO_SPEED_MULTIPLIER, 1e-3)
        terminal_hold_time = max(terminal_hold_time / DEMO_SPEED_MULTIPLIER, 1e-3)
        video_slowdown = max(video_slowdown / DEMO_SPEED_MULTIPLIER, 1e-3)

    save_snapshots = bool(int(args.save_snapshots)) and path_source != "demo"
    segment_results: list[dict[str, object]] = []
    segment_video_paths: list[str] = []
    segment_error_video_paths: list[str] = []
    segment_dir = os.path.join(render_outdir, "_segments")
    if len(paths_to_render) > 1 and (int(args.gui) == 1 or final_error_video_path is not None):
        os.makedirs(segment_dir, exist_ok=True)

    for seg_idx, path in enumerate(paths_to_render):
        seg_video_path = final_video_path
        seg_error_video_path = final_error_video_path
        if len(paths_to_render) > 1 and (int(args.gui) == 1 or final_error_video_path is not None):
            if int(args.gui) == 1:
                seg_video_path = os.path.join(segment_dir, f"dual_arm_demo_path_{seg_idx:02d}.mp4")
            if final_error_video_path is not None:
                seg_error_video_path = os.path.join(segment_dir, f"dual_arm_demo_path_errors_{seg_idx:02d}.mp4")
        seg = _run_segment(
            path=path,
            cfg=cfg,
            base_cfg=base_cfg,
            gui_mode=int(args.gui),
            orientation_mode=str(args.orientation_mode),
            task_offset_quat=task_offset_quat,
            ik_iters=int(args.ik_iters),
            sim_dt=float(args.sim_dt),
            max_joint_speed=max_joint_speed,
            min_segment_time=min_segment_time,
            terminal_hold_time=terminal_hold_time,
            max_force=float(args.max_force),
            video_path=seg_video_path,
            video_width=int(args.video_width),
            video_height=int(args.video_height),
            video_fps=int(args.video_fps),
            video_slowdown=video_slowdown,
            realtime=bool(int(args.realtime)),
            trace_stride=int(args.trace_stride),
            arm_trace_radius=float(args.arm_trace_radius),
            save_snapshots=save_snapshots,
            snapshot_steps=snapshot_steps,
            snapshot_dir=snapshot_dir,
            snapshot_crop_frac=snapshot_crop_frac,
        )
        seg["label"] = segment_labels[seg_idx] if seg_idx < len(segment_labels) else f"segment_{seg_idx}"
        if path_source == "demo":
            seg["demo_index"] = int(demo_indices[seg_idx])
        segment_results.append(seg)
        if isinstance(seg.get("video_path"), str) and seg.get("video_path"):
            segment_video_paths.append(str(seg["video_path"]))
        if seg_error_video_path is not None:
            err_vid = _make_error_curve_video(
                pos_err=np.asarray(seg["exec_true_err"]["mean_pos_err"], dtype=np.float32),
                ori_err_deg=np.asarray(seg["exec_true_err"]["mean_ori_err_deg"], dtype=np.float32),
                sim_dt=float(args.sim_dt),
                video_fps=int(args.video_fps),
                video_slowdown=video_slowdown,
                out_path=seg_error_video_path,
                title=f"{seg['label']}: execution error vs time" if "label" in seg else f"segment_{seg_idx}: execution error vs time",
            )
            if isinstance(err_vid, str) and err_vid:
                seg["error_video_path"] = err_vid
                segment_error_video_paths.append(err_vid)
        print(
            f"[segment {seg_idx}] "
            f"{seg['label']} "
            f"left_ik={seg['ik_left']['mean_pos_err']:.4f}/{seg['ik_left']['max_pos_err']:.4f}, "
            f"right_ik={seg['ik_right']['mean_pos_err']:.4f}/{seg['ik_right']['max_pos_err']:.4f}, "
            f"exec_pos={float(np.mean(seg['exec_left_err'])):.4f}/{float(np.mean(seg['exec_right_err'])):.4f}, "
            f"exec_true_pos={_summary_stats(seg['exec_true_err']['mean_pos_err'])['mean']:.4f}, "
            f"exec_true_ori_deg={_summary_stats(seg['exec_true_err']['mean_ori_err_deg'])['mean']:.2f}"
        )

    merged_video_path = None
    merged_error_video_path = None
    if int(args.gui) == 1:
        if len(segment_video_paths) > 1 and final_video_path is not None:
            merged_video_path = _concat_videos_ffmpeg(segment_video_paths, final_video_path)
            if merged_video_path:
                _cleanup_segment_videos(segment_video_paths, merged_video_path)
        elif segment_video_paths:
            merged_video_path = segment_video_paths[-1]
        if len(segment_error_video_paths) > 1 and final_error_video_path is not None:
            merged_error_video_path = _concat_videos_ffmpeg(segment_error_video_paths, final_error_video_path)
            if merged_error_video_path:
                _cleanup_segment_videos(segment_error_video_paths, merged_error_video_path)
        elif segment_error_video_paths:
            merged_error_video_path = segment_error_video_paths[-1]
    elif final_error_video_path is not None:
        if len(segment_error_video_paths) > 1:
            merged_error_video_path = _concat_videos_ffmpeg(segment_error_video_paths, final_error_video_path)
            if merged_error_video_path:
                _cleanup_segment_videos(segment_error_video_paths, merged_error_video_path)
        elif segment_error_video_paths:
            merged_error_video_path = segment_error_video_paths[-1]

    exec_left_all = np.concatenate([np.asarray(seg["exec_left_err"], dtype=np.float32) for seg in segment_results], axis=0)
    exec_right_all = np.concatenate([np.asarray(seg["exec_right_err"], dtype=np.float32) for seg in segment_results], axis=0)
    exec_true_all = {
        k: np.concatenate([np.asarray(seg["exec_true_err"][k], dtype=np.float32) for seg in segment_results], axis=0)
        for k in (
            "left_pos_err",
            "right_pos_err",
            "mean_pos_err",
            "left_ori_err_deg",
            "right_ori_err_deg",
            "mean_ori_err_deg",
            "analytic_vector_dist",
            "center_err",
            "span_err",
        )
    }
    snapshot_paths_all: list[str] = []
    for seg in segment_results:
        snapshot_paths_all.extend([str(v) for v in seg["track"].get("snapshot_paths", [])])
    summary = {
        "path_source": path_source,
        "planned_path_npz": summary_npz_path,
        "planned_path_key": summary_path_key,
        "demo_index": (int(args.demo_index) if (path_source == "demo" and int(args.demo_index) >= 0) else None),
        "demo_indices": ([int(v) for v in demo_indices] if path_source == "demo" else None),
        "demo_count": (int(len(demo_indices)) if path_source == "demo" else None),
        "demo_speed_multiplier": (float(DEMO_SPEED_MULTIPLIER) if path_source == "demo" else None),
        "orientation_mode": str(args.orientation_mode),
        "task_roll_deg": float(args.task_roll_deg),
        "task_offset_rpy_deg": [float(v) for v in task_offset_rpy_deg.tolist()],
        "sim_dt": float(args.sim_dt),
        "n_segments": int(len(segment_results)),
        "n_task_waypoints": int(sum(len(path["pose_left"]) for path in paths_to_render)),
        "n_control_steps": int(sum(len(np.asarray(seg["t"])) for seg in segment_results)),
        "duration_s": float(sum(float(np.asarray(seg["t"])[-1]) if len(np.asarray(seg["t"])) else 0.0 for seg in segment_results)),
        "max_joint_speed_used": max_joint_speed,
        "min_segment_time_used": min_segment_time,
        "terminal_hold_time_used": terminal_hold_time,
        "video_slowdown_used": video_slowdown,
        "left_ik": (_aggregate_ik_stats(segment_results, "left") if len(segment_results) > 1 else segment_results[0]["ik_left"]),
        "right_ik": (_aggregate_ik_stats(segment_results, "right") if len(segment_results) > 1 else segment_results[0]["ik_right"]),
        "exec_pos_err_left": _summary_stats(exec_left_all),
        "exec_pos_err_right": _summary_stats(exec_right_all),
        "exec_pos_err_left_mean": float(np.mean(exec_left_all)),
        "exec_pos_err_right_mean": float(np.mean(exec_right_all)),
        "exec_pos_err_left_max": float(np.max(exec_left_all)),
        "exec_pos_err_right_max": float(np.max(exec_right_all)),
        "exec_true_constraint": {
            "left_pos_err": _summary_stats(exec_true_all["left_pos_err"]),
            "right_pos_err": _summary_stats(exec_true_all["right_pos_err"]),
            "mean_pos_err": _summary_stats(exec_true_all["mean_pos_err"]),
            "left_ori_err_deg": _summary_stats(exec_true_all["left_ori_err_deg"]),
            "right_ori_err_deg": _summary_stats(exec_true_all["right_ori_err_deg"]),
            "mean_ori_err_deg": _summary_stats(exec_true_all["mean_ori_err_deg"]),
            "analytic_vector_dist": _summary_stats(exec_true_all["analytic_vector_dist"]),
            "center_err": _summary_stats(exec_true_all["center_err"]),
            "span_err": _summary_stats(exec_true_all["span_err"]),
        },
        "mean_joint_err_left": float(np.mean([float(seg["track"]["mean_joint_err_left"]) for seg in segment_results])),
        "mean_joint_err_right": float(np.mean([float(seg["track"]["mean_joint_err_right"]) for seg in segment_results])),
        "max_joint_err_left": float(np.max([float(seg["track"]["max_joint_err_left"]) for seg in segment_results])),
        "max_joint_err_right": float(np.max([float(seg["track"]["max_joint_err_right"]) for seg in segment_results])),
        "arm_trace_radius": float(args.arm_trace_radius),
        "save_snapshots": save_snapshots,
        "snapshot_steps": [int(v) for v in snapshot_steps],
        "snapshot_dir": (snapshot_dir if save_snapshots else None),
        "snapshot_crop_frac": (list(snapshot_crop_frac) if (snapshot_crop_frac is not None and save_snapshots) else None),
        "snapshot_paths": snapshot_paths_all,
        "base_cfg": base_cfg,
        "video_path": merged_video_path,
        "error_video_path": merged_error_video_path,
        "segment_video_paths": ([] if merged_video_path else segment_video_paths),
        "segment_error_video_paths": ([] if merged_error_video_path else segment_error_video_paths),
        "segment_summaries": [
            {
                "label": str(seg["label"]),
                "demo_index": seg.get("demo_index"),
                "n_task_waypoints": int(len(paths_to_render[idx]["pose_left"])),
                "n_control_steps": int(len(np.asarray(seg["t"]))),
                "duration_s": float(np.asarray(seg["t"])[-1]) if len(np.asarray(seg["t"])) else 0.0,
                "left_ik": seg["ik_left"],
                "right_ik": seg["ik_right"],
                "exec_pos_err_left": _summary_stats(np.asarray(seg["exec_left_err"], dtype=np.float32)),
                "exec_pos_err_right": _summary_stats(np.asarray(seg["exec_right_err"], dtype=np.float32)),
                "exec_true_constraint": {
                    "mean_pos_err": _summary_stats(np.asarray(seg["exec_true_err"]["mean_pos_err"], dtype=np.float32)),
                    "mean_ori_err_deg": _summary_stats(np.asarray(seg["exec_true_err"]["mean_ori_err_deg"], dtype=np.float32)),
                    "analytic_vector_dist": _summary_stats(np.asarray(seg["exec_true_err"]["analytic_vector_dist"], dtype=np.float32)),
                },
                "video_path": (None if merged_video_path else seg.get("video_path")),
                "error_video_path": (None if merged_error_video_path else seg.get("error_video_path")),
            }
            for idx, seg in enumerate(segment_results)
        ],
    }
    os.makedirs(render_outdir, exist_ok=True)
    summary_name = "dual_arm_demo_path_summary.json" if path_source == "demo" else "dual_arm_planned_path_summary.json"
    summary_path = os.path.join(render_outdir, summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        "[summary] "
        f"left_ik={summary['left_ik']['mean_pos_err']:.4f}/{summary['left_ik']['max_pos_err']:.4f}, "
        f"right_ik={summary['right_ik']['mean_pos_err']:.4f}/{summary['right_ik']['max_pos_err']:.4f}, "
        f"exec_pos={summary['exec_pos_err_left_mean']:.4f}/{summary['exec_pos_err_right_mean']:.4f}, "
        f"exec_true_pos={summary['exec_true_constraint']['mean_pos_err']['mean']:.4f}, "
        f"exec_true_ori_deg={summary['exec_true_constraint']['mean_ori_err_deg']['mean']:.2f}, "
        f"joint_err={summary['mean_joint_err_left']:.4f}/{summary['mean_joint_err_right']:.4f}"
    )
    print(f"[saved] {summary_path}")
    if merged_video_path:
        print(f"[saved] {merged_video_path}")
    if merged_error_video_path:
        print(f"[saved] {merged_error_video_path}")


if __name__ == "__main__":
    main()
