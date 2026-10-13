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

import numpy as np

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.constraint_datasets import generate_dataset, sine_surface_z_and_normal_from_xy  # noqa: E402
from simulation.ik_controller import IKConfig, UR5TrajectoryController, _FFmpegVideoWriter  # noqa: E402


DATASET_NAME = "6d_surf_normal_traj"
DEFAULT_DATASET_CONFIG = "configs/datasets/6d_surf_normal_traj.json"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_repo_root(), path))


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


def _make_surface_patch_xyz_grid(
    pose_path_raw: np.ndarray,
    *,
    surface_cfg=None,
    pad_x_neg: float = 0.06,
    pad_x_pos: float = 0.18,
    pad_y_neg: float = 0.20,
    pad_y_pos: float = 0.08,
    grid_size: int = 64,
) -> np.ndarray:
    xyz = np.asarray(pose_path_raw[:, :3], dtype=np.float32)
    x_lo = float(np.min(xyz[:, 0]) - pad_x_neg)
    x_hi = float(np.max(xyz[:, 0]) + pad_x_pos)
    y_lo = float(np.min(xyz[:, 1]) - pad_y_neg)
    y_hi = float(np.max(xyz[:, 1]) + pad_y_pos)
    gx = np.linspace(x_lo, x_hi, num=int(max(8, grid_size)), dtype=np.float32)
    gy = np.linspace(y_lo, y_hi, num=int(max(8, grid_size)), dtype=np.float32)
    gxx, gyy = np.meshgrid(gx, gy)
    gzz, _ = sine_surface_z_and_normal_from_xy(gxx.reshape(-1), gyy.reshape(-1), surface_cfg)
    return np.stack([gxx, gyy, gzz.reshape(gxx.shape)], axis=2).astype(np.float32)


def _load_demo_cfg(path: str, *, seed: int) -> SimpleNamespace:
    cfg_path = _resolve_path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = dict(json.load(f))
    cfg.setdefault("n_train", 3000)
    cfg.setdefault("n_grid", int(cfg.get("traj_gene_n_grid", max(4096, int(cfg["n_train"])))))
    cfg["n_grid"] = int(max(int(cfg.get("n_grid", 1)), int(cfg.get("traj_gene_n_grid", cfg.get("n_grid", 1)))))
    cfg["seed"] = int(seed)
    return SimpleNamespace(**cfg)


def _split_demo_segments(x_train: np.ndarray, cfg: SimpleNamespace) -> list[np.ndarray]:
    x = np.asarray(x_train, dtype=np.float32)
    n_train = int(x.shape[0])
    traj_count = int(max(1, min(n_train, getattr(cfg, "traj_count", max(12, n_train // 64)))))
    seg_len = int(max(2, math.ceil(n_train / max(traj_count, 1))))
    out: list[np.ndarray] = []
    for i in range(traj_count):
        a = int(i * seg_len)
        b = int(min(n_train, (i + 1) * seg_len))
        if b - a >= 2:
            out.append(x[a:b].astype(np.float32))
    return out


def _select_segments(
    segments: list[np.ndarray],
    *,
    n_demos: int,
    selection: str,
    start_index: int = 0,
    explicit_indices: list[int] | None = None,
    skip_indices: list[int] | None = None,
) -> list[tuple[int, np.ndarray]]:
    if not segments:
        return []
    skip_set = set()
    if skip_indices is not None:
        skip_set = {int(np.clip(int(v), 0, len(segments) - 1)) for v in skip_indices}
    if explicit_indices is not None and len(explicit_indices) > 0:
        idx: list[int] = []
        seen: set[int] = set()
        for v in explicit_indices:
            i = int(np.clip(int(v), 0, len(segments) - 1))
            if i not in seen and i not in skip_set:
                idx.append(i)
                seen.add(i)
        return [(i, segments[i].astype(np.float32)) for i in idx]
    n = int(max(1, min(int(n_demos), len(segments))))
    start = int(np.clip(int(start_index), 0, max(0, len(segments) - 1)))
    if selection == "first":
        candidates = list(range(start, len(segments)))
    elif selection == "uniform":
        candidates = np.linspace(start, len(segments) - 1, num=max(n + len(skip_set), n), dtype=np.int32).tolist()
        candidates = [int(v) for v in dict.fromkeys(candidates)]
    else:
        candidates = list(range(len(segments) - 1, start - 1, -1))
    idx = []
    for i in candidates:
        if i in skip_set:
            continue
        idx.append(int(i))
        if len(idx) >= n:
            break
    if selection == "last":
        idx = list(reversed(idx))
    return [(i, segments[i].astype(np.float32)) for i in idx]


def _stride_demo(demo: np.ndarray, stride: int) -> np.ndarray:
    arr = np.asarray(demo, dtype=np.float32)
    s = int(max(1, stride))
    if s <= 1 or len(arr) <= 2:
        return arr.astype(np.float32)
    idx = np.arange(0, len(arr), s, dtype=np.int32)
    if int(idx[-1]) != len(arr) - 1:
        idx = np.concatenate([idx, np.asarray([len(arr) - 1], dtype=np.int32)])
    return arr[idx].astype(np.float32)


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _smooth_rows(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if len(x) < 3:
        return x.astype(np.float32)
    padded = np.concatenate([x[:1], x, x[-1:]], axis=0)
    return (0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]).astype(np.float32)


def _prepare_playback_pose_path(
    pose_path: np.ndarray,
    cfg: SimpleNamespace,
    *,
    interp_steps: int,
    smooth_passes: int,
) -> np.ndarray:
    src = np.asarray(pose_path, dtype=np.float32)
    if src.ndim != 2 or src.shape[1] < 6:
        raise ValueError("pose_path must have shape (N, >=6)")
    if len(src) <= 1:
        return src[:, :6].astype(np.float32)

    steps = int(max(1, interp_steps))
    if steps > 1:
        u = np.arange(len(src), dtype=np.float32)
        u_new = np.linspace(0.0, float(len(src) - 1), num=(len(src) - 1) * steps + 1, dtype=np.float32)
        out = np.zeros((len(u_new), 6), dtype=np.float32)
        out[:, 0] = np.interp(u_new, u, src[:, 0]).astype(np.float32)
        out[:, 1] = np.interp(u_new, u, src[:, 1]).astype(np.float32)
        z_surface, _ = sine_surface_z_and_normal_from_xy(out[:, 0], out[:, 1], cfg)
        out[:, 2] = np.asarray(z_surface, dtype=np.float32)
        rpy_unwrapped = np.unwrap(src[:, 3:6].astype(np.float64), axis=0)
        for j in range(3):
            out[:, 3 + j] = np.interp(u_new, u, rpy_unwrapped[:, j]).astype(np.float32)
        out[:, 3:6] = _wrap_to_pi(out[:, 3:6])
    else:
        out = src[:, :6].copy().astype(np.float32)

    passes = int(max(0, smooth_passes))
    for _ in range(passes):
        endpoints = out[[0, -1]].copy()
        out[:, 0:2] = _smooth_rows(out[:, 0:2])
        z_surface, _ = sine_surface_z_and_normal_from_xy(out[:, 0], out[:, 1], cfg)
        out[:, 2] = np.asarray(z_surface, dtype=np.float32)
        rpy_unwrapped = np.unwrap(out[:, 3:6].astype(np.float64), axis=0).astype(np.float32)
        out[:, 3:6] = _wrap_to_pi(_smooth_rows(rpy_unwrapped))
        out[0] = endpoints[0]
        out[-1] = endpoints[1]
    return out.astype(np.float32)


def _surface_z_residual_from_xyz(xyz: np.ndarray, cfg: SimpleNamespace) -> np.ndarray:
    arr = np.asarray(xyz, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("xyz must have shape (N, >=3)")
    z_surface, _ = sine_surface_z_and_normal_from_xy(arr[:, 0], arr[:, 1], cfg)
    return (arr[:, 2].astype(np.float32) - np.asarray(z_surface, dtype=np.float32)).astype(np.float32)


def _trace_visual_xyz(
    xyz: np.ndarray,
    cfg: SimpleNamespace | None,
    *,
    clearance: float,
    follow_surface: bool,
) -> np.ndarray:
    arr = np.asarray(xyz, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("xyz must have shape (N, >=3)")
    clearance = float(max(0.0, clearance))
    if cfg is None:
        arr[:, 2] += clearance
        return arr[:, :3].astype(np.float32)
    z_surface, normal = sine_surface_z_and_normal_from_xy(arr[:, 0], arr[:, 1], cfg)
    normal = np.asarray(normal, dtype=np.float32)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    if bool(follow_surface):
        arr[:, 2] = np.asarray(z_surface, dtype=np.float32)
    else:
        arr[:, 2] = np.maximum(arr[:, 2], np.asarray(z_surface, dtype=np.float32))
    arr[:, :3] += normal[:, :3] * clearance
    return arr[:, :3].astype(np.float32)


def _render_direct_joint_playback(
    ctrl: UR5TrajectoryController,
    joint_path: np.ndarray,
    *,
    video_path: str | None,
    surface_xyz_grid: np.ndarray,
    surface_cfg: SimpleNamespace | None,
    reuse_static_scene: bool,
    draw_surface: bool,
    ref_xyz_path: np.ndarray | None,
    draw_ref_trace: bool,
    draw_trace: bool,
    trace_stride: int,
    trace_width: float,
    trace_surface_clearance: float,
    trace_follow_surface: bool,
    surface_line_stride: int,
    surface_line_width: float,
    video_width: int,
    video_height: int,
    play_fps: float,
    render_frame_stride: int,
    realtime: bool,
    sim_dt: float,
) -> dict[str, np.ndarray | float | str | None]:
    import time

    q_path = np.asarray(joint_path, dtype=np.float32)
    if q_path.ndim != 2 or q_path.shape[1] != 6:
        raise ValueError("joint_path must have shape (N, 6)")

    writer = None
    video_out_path = os.path.abspath(video_path) if video_path else None
    if video_out_path:
        os.makedirs(os.path.dirname(video_out_path), exist_ok=True)
        writer = _FFmpegVideoWriter(
            out_path=video_out_path,
            width=int(video_width),
            height=int(video_height),
            fps=float(max(1e-3, play_fps)),
        )

    if not bool(reuse_static_scene):
        ctrl._clear_surface_visuals()
    if bool(draw_surface) and (not bool(reuse_static_scene) or len(ctrl._surface_body_ids) == 0):
        ctrl._build_surface_visuals(
            np.asarray(surface_xyz_grid, dtype=np.float32),
            stride=int(max(1, surface_line_stride)),
            line_width=float(max(0.5, surface_line_width)),
        )
    ctrl._clear_trace_visuals()

    q_log = np.zeros_like(q_path, dtype=np.float32)
    ee_pos = np.zeros((len(q_path), 3), dtype=np.float32)
    ee_quat = np.zeros((len(q_path), 4), dtype=np.float32)
    trace_stride = int(max(1, trace_stride))
    trace_radius = float(max(0.0012, 0.0008 * float(trace_width)))
    trace_clearance = float(max(float(trace_surface_clearance), 3.0 * trace_radius))
    trace_rgba = (0.10, 0.70, 0.25, 0.92)
    prev_pos_vis: np.ndarray | None = None
    render_frame_stride = int(max(1, render_frame_stride))
    capture_seconds = 0.0
    write_seconds = 0.0
    close_seconds = 0.0
    frames_written = 0

    if bool(draw_ref_trace) and ref_xyz_path is not None:
        ref = np.asarray(ref_xyz_path, dtype=np.float32)
        if ref.ndim == 2 and ref.shape[1] >= 3 and len(ref) >= 2:
            ref_xyz = ref[:, :3]
            ref_idx = np.arange(0, len(ref_xyz), trace_stride, dtype=np.int32)
            if int(ref_idx[-1]) != len(ref_xyz) - 1:
                ref_idx = np.concatenate([ref_idx, np.asarray([len(ref_xyz) - 1], dtype=np.int32)])
            ref_xyz = _trace_visual_xyz(
                ref_xyz[ref_idx],
                surface_cfg,
                clearance=max(0.006, 0.75 * trace_clearance),
                follow_surface=True,
            )
            for a, b in zip(ref_xyz[:-1], ref_xyz[1:]):
                ctrl._add_visual_cylinder_segment(
                    a,
                    b,
                    radius=max(0.0009, 0.7 * trace_radius),
                    rgba=(0.10, 0.30, 0.95, 0.86),
                    specular=(0.03, 0.04, 0.08),
                    body_list=ctrl._trace_body_ids,
                )

    t0 = time.time()
    for i, q in enumerate(q_path):
        ctrl.reset_joint_state(q)
        ctrl._p.stepSimulation(physicsClientId=ctrl.client_id)
        q_i, _qd_i = ctrl.get_joint_state()
        pos_i, quat_i = ctrl.get_ee_pose()
        q_log[i] = q_i.astype(np.float32)
        ee_pos[i] = pos_i.astype(np.float32)
        ee_quat[i] = quat_i.astype(np.float32)
        if bool(draw_trace) and i % trace_stride == 0:
            pos_vis = _trace_visual_xyz(
                pos_i.reshape(1, 3),
                surface_cfg,
                clearance=trace_clearance,
                follow_surface=bool(trace_follow_surface),
            )[0]
            if prev_pos_vis is not None:
                ctrl._add_visual_cylinder_segment(
                    prev_pos_vis,
                    pos_vis,
                    radius=trace_radius,
                    rgba=trace_rgba,
                    specular=(0.03, 0.05, 0.03),
                    body_list=ctrl._trace_body_ids,
                )
            prev_pos_vis = pos_vis.copy()
        if writer is not None:
            write_this_frame = (i % render_frame_stride == 0) or (i == len(q_path) - 1)
            if write_this_frame:
                t_cap = time.time()
                frame = ctrl.capture_frame(width=int(video_width), height=int(video_height))
                capture_seconds += float(time.time() - t_cap)
                t_write = time.time()
                writer.append_data(frame)
                write_seconds += float(time.time() - t_write)
                frames_written += 1
        if bool(realtime):
            time.sleep(float(max(0.0, sim_dt)))

    if writer is not None:
        t_close = time.time()
        writer.close()
        close_seconds = float(time.time() - t_close)
    ctrl._clear_trace_visuals()
    wall_seconds = float(time.time() - t0)
    return {
        "sim_dt": float(sim_dt),
        "wall_seconds": wall_seconds,
        "capture_seconds": float(capture_seconds),
        "write_seconds": float(write_seconds),
        "close_seconds": float(close_seconds),
        "frames_written": int(frames_written),
        "render_fps_wall": float(frames_written) / max(wall_seconds, 1e-9),
        "q_set": q_log.astype(np.float32),
        "ee_pos": ee_pos.astype(np.float32),
        "ee_quat_xyzw": ee_quat.astype(np.float32),
        "video_path": video_out_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast PyBullet playback renderer for SurfNormal demonstration trajectories."
    )
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG, help="Dataset config JSON.")
    parser.add_argument(
        "--outdir",
        default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/surfnormal_demonstrations",
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Dataset generation seed.")
    parser.add_argument("--n-demos", type=int, default=10, help="Number of demonstration trajectories to render.")
    parser.add_argument(
        "--demo-start-index",
        type=int,
        default=0,
        help="Start demo index for contiguous selection modes.",
    )
    parser.add_argument(
        "--demo-indices",
        default="",
        help="Comma-separated explicit demo indices to render; overrides selection/start-index/n-demos.",
    )
    parser.add_argument(
        "--skip-demo-indices",
        default="",
        help="Comma-separated demo indices to exclude before selection; useful for skipping bad late demos while keeping later ones preferred.",
    )
    parser.add_argument(
        "--selection",
        choices=["last", "first", "uniform"],
        default="last",
        help="Which demonstration trajectories to render.",
    )
    parser.add_argument("--demo-stride", type=int, default=2, help="Keep every Nth point from each demonstration.")

    parser.add_argument("--ik-method", choices=["pybullet", "dls"], default="pybullet", help="IK backend.")
    parser.add_argument("--ik-iters", type=int, default=64, help="Per-point IK iterations.")
    parser.add_argument("--ik-damping", type=float, default=0.05, help="DLS damping coefficient.")
    parser.add_argument("--ik-step-size", type=float, default=0.6, help="DLS step size.")
    parser.add_argument("--ik-max-delta", type=float, default=0.18, help="Max joint update norm per IK iteration.")
    parser.add_argument("--ik-pos-tol", type=float, default=0.004, help="IK position tolerance in meters.")
    parser.add_argument("--ik-ori-tol-deg", type=float, default=2.5, help="IK orientation tolerance in degrees.")

    parser.add_argument("--sim-dt", type=float, default=1.0 / 240.0, help="PyBullet simulation step.")
    parser.add_argument("--play-fps", type=float, default=60.0, help="Playback FPS for direct demo frames.")
    parser.add_argument(
        "--playback-interp-steps",
        type=int,
        default=1,
        help="Insert this many direct-playback intervals per original demo segment; 1 keeps original points.",
    )
    parser.add_argument(
        "--playback-smooth-passes",
        "--demo-smooth-passes",
        dest="playback_smooth_passes",
        type=int,
        default=0,
        help="Render-only smoothing passes applied after interpolation while keeping points on the analytic surface.",
    )
    parser.add_argument(
        "--render-frame-stride",
        type=int,
        default=1,
        help="Write one video frame every N demo waypoints; use >1 for fast preview rendering.",
    )

    parser.add_argument("--draw-ee-trace", type=int, default=1, help="1 to draw executed end-effector traces.")
    parser.add_argument("--draw-ref-trace", type=int, default=0, help="1 to draw reference traces.")
    parser.add_argument("--draw-surface-wireframe", type=int, default=1, help="1 to draw the sine surface workpiece.")
    parser.add_argument("--hide-gripper", type=int, default=1, help="1 to hide the Robotiq gripper.")
    parser.add_argument("--trace-stride", type=int, default=6, help="Draw one trace segment every this many control samples.")
    parser.add_argument("--trace-width", type=float, default=2.2, help="Trajectory trace width.")
    parser.add_argument(
        "--trace-surface-clearance",
        type=float,
        default=0.018,
        help="Visual clearance above the analytic surface for rendered trace tubes.",
    )
    parser.add_argument(
        "--trace-follow-surface",
        type=int,
        default=1,
        help="1 to draw trace tubes on the analytic surface plus clearance; 0 to draw measured EE positions plus clearance.",
    )
    parser.add_argument("--surface-line-stride", type=int, default=4, help="Surface visual grid stride.")
    parser.add_argument("--surface-line-width", type=float, default=1.2, help="Surface visual line width.")
    parser.add_argument(
        "--surface-visual-grid",
        type=int,
        default=64,
        help="Grid resolution used to render the analytic sine surface.",
    )

    parser.add_argument("--gui", type=int, choices=[0, 1, 2], default=1, help="0: no video; 1: offscreen video; 2: GUI render only, no video logging.")
    parser.add_argument("--realtime", type=int, default=0, help="1 to sleep between sim steps.")
    parser.add_argument("--video-name", default="surfnormal_demonstrations.mp4", help="Merged MP4 filename.")
    parser.add_argument("--video-slowdown", type=float, default=1.0, help="Saved video slowdown; below 1 speeds playback up.")
    parser.add_argument("--video-width", type=int, default=1024, help="Video frame width.")
    parser.add_argument("--video-height", type=int, default=768, help="Video frame height.")

    parser.add_argument("--camera-distance", type=float, default=1.55, help="Camera distance.")
    parser.add_argument("--camera-yaw", type=float, default=34.0, help="Camera yaw in degrees.")
    parser.add_argument("--camera-pitch", type=float, default=-38.0, help="Camera pitch in degrees.")
    parser.add_argument("--camera-fov", type=float, default=54.0, help="Camera vertical FOV in degrees.")
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=[0.40, -0.22, 1.07],
        metavar=("X", "Y", "Z"),
        help="Camera target position.",
    )
    args = parser.parse_args()

    outdir = _resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    demo_cfg = _load_demo_cfg(str(args.dataset_config), seed=int(args.seed))
    x_train, _grid = generate_dataset(DATASET_NAME, demo_cfg)
    segments = _split_demo_segments(x_train, demo_cfg)
    explicit_indices = [int(v.strip()) for v in str(args.demo_indices).split(",") if v.strip()] if str(args.demo_indices).strip() else None
    skip_indices = [int(v.strip()) for v in str(args.skip_demo_indices).split(",") if v.strip()] if str(args.skip_demo_indices).strip() else None
    selected = _select_segments(
        segments,
        n_demos=int(args.n_demos),
        selection=str(args.selection),
        start_index=int(args.demo_start_index),
        explicit_indices=explicit_indices,
        skip_indices=skip_indices,
    )
    selected = [(i, _stride_demo(seg, int(args.demo_stride))) for i, seg in selected]
    if not selected:
        raise RuntimeError("no demonstration trajectories were selected")

    ik_cfg = IKConfig(
        method=str(args.ik_method),
        max_iters=int(args.ik_iters),
        damping=float(args.ik_damping),
        step_size=float(args.ik_step_size),
        max_delta_norm=float(args.ik_max_delta),
        pos_tol=float(args.ik_pos_tol),
        ori_tol_rad=math.radians(float(args.ik_ori_tol_deg)),
    )
    save_video = int(args.gui) == 1
    use_gui = int(args.gui) == 2
    final_video_path = os.path.join(outdir, str(args.video_name)) if save_video else None
    segment_video_paths: list[str] = []
    arrays_to_save: dict[str, np.ndarray] = {}
    summaries: list[dict[str, object]] = []

    common_pose_path = np.concatenate([seg for _, seg in selected], axis=0).astype(np.float32)
    surface_xyz_grid = _make_surface_patch_xyz_grid(
        common_pose_path,
        surface_cfg=demo_cfg,
        grid_size=int(args.surface_visual_grid),
    )
    play_fps = float(args.play_fps) / max(float(args.video_slowdown), 1e-3)

    with UR5TrajectoryController(
        gui=use_gui,
        hide_gripper=bool(args.hide_gripper),
        sim_dt=float(args.sim_dt),
    ) as ctrl:
        ctrl.set_camera(
            distance=float(args.camera_distance),
            yaw=float(args.camera_yaw),
            pitch=float(args.camera_pitch),
            target_position=[float(v) for v in args.camera_target],
            fov=float(args.camera_fov),
        )
        for local_idx, (demo_idx, pose_path_raw) in enumerate(selected, start=1):
            pose_path_source = pose_path_raw.astype(np.float32)
            pose_path = _prepare_playback_pose_path(
                pose_path_source,
                demo_cfg,
                interp_steps=int(args.playback_interp_steps),
                smooth_passes=int(args.playback_smooth_passes),
            )
            demo_surface_residual = _surface_z_residual_from_xyz(pose_path[:, :3], demo_cfg)
            joint_path, ik_summary = ctrl.solve_pose_path_ik(pose_path, cfg=ik_cfg)
            joint_step_norm = (
                np.linalg.norm(np.diff(joint_path.astype(np.float32), axis=0), axis=1).astype(np.float32)
                if len(joint_path) > 1
                else np.zeros((0,), dtype=np.float32)
            )

            segment_video_path = None
            if final_video_path:
                root, ext = os.path.splitext(final_video_path)
                segment_video_path = f"{root}_demo_{local_idx:02d}{ext or '.mp4'}"

            direct = _render_direct_joint_playback(
                ctrl,
                joint_path,
                video_path=segment_video_path,
                surface_xyz_grid=surface_xyz_grid,
                surface_cfg=demo_cfg,
                reuse_static_scene=True,
                draw_surface=bool(args.draw_surface_wireframe),
                ref_xyz_path=pose_path[:, :3],
                draw_ref_trace=bool(args.draw_ref_trace),
                draw_trace=bool(args.draw_ee_trace),
                trace_stride=int(args.trace_stride),
                trace_width=float(args.trace_width),
                trace_surface_clearance=float(args.trace_surface_clearance),
                trace_follow_surface=bool(args.trace_follow_surface),
                surface_line_stride=int(args.surface_line_stride),
                surface_line_width=float(args.surface_line_width),
                video_width=int(args.video_width),
                video_height=int(args.video_height),
                play_fps=float(play_fps),
                render_frame_stride=int(args.render_frame_stride),
                realtime=bool(args.realtime),
                sim_dt=float(args.sim_dt),
            )
            if direct.get("video_path"):
                segment_video_paths.append(str(direct["video_path"]))

            prefix = f"demo_{local_idx:02d}"
            arrays_to_save[f"{prefix}_dataset_index"] = np.asarray([demo_idx], dtype=np.int32)
            arrays_to_save[f"{prefix}_pose_path_source"] = pose_path_source.astype(np.float32)
            arrays_to_save[f"{prefix}_pose_path"] = pose_path.astype(np.float32)
            arrays_to_save[f"{prefix}_joint_path"] = joint_path.astype(np.float32)
            arrays_to_save[f"{prefix}_joint_step_norm"] = joint_step_norm.astype(np.float32)
            arrays_to_save[f"{prefix}_q_set"] = np.asarray(direct["q_set"], dtype=np.float32)
            arrays_to_save[f"{prefix}_ee_pos"] = np.asarray(direct["ee_pos"], dtype=np.float32)
            arrays_to_save[f"{prefix}_ee_quat_xyzw"] = np.asarray(direct["ee_quat_xyzw"], dtype=np.float32)
            arrays_to_save[f"{prefix}_demo_surface_z_residual"] = demo_surface_residual.astype(np.float32)
            arrays_to_save[f"{prefix}_ee_surface_z_residual"] = _surface_z_residual_from_xyz(
                np.asarray(direct["ee_pos"], dtype=np.float32),
                demo_cfg,
            ).astype(np.float32)
            arrays_to_save[f"{prefix}_ik_pos_errs"] = np.asarray(ik_summary["pos_errs"], dtype=np.float32)
            arrays_to_save[f"{prefix}_ik_ori_errs_rad"] = np.asarray(ik_summary["ori_errs_rad"], dtype=np.float32)

            ee_surface_residual = arrays_to_save[f"{prefix}_ee_surface_z_residual"]

            summaries.append(
                {
                    "demo_local_index": int(local_idx),
                    "demo_dataset_index": int(demo_idx),
                    "source_waypoints": int(pose_path_source.shape[0]),
                    "pose_waypoints": int(pose_path.shape[0]),
                    "frames": int(joint_path.shape[0]),
                    "video_frames": int(direct["frames_written"]),
                    "duration_s": float(direct["frames_written"]) / max(float(play_fps), 1e-6),
                    "ik": {
                        "mean_pos_err": float(ik_summary["mean_pos_err"]),
                        "max_pos_err": float(ik_summary["max_pos_err"]),
                        "first_pos_err": float(ik_summary["first_pos_err"]),
                        "mean_ori_err_deg": float(np.degrees(float(ik_summary["mean_ori_err_rad"]))),
                        "max_ori_err_deg": float(np.degrees(float(ik_summary["max_ori_err_rad"]))),
                        "first_ori_err_deg": float(np.degrees(float(ik_summary["first_ori_err_rad"]))),
                        "mean_iters": float(ik_summary["mean_iters"]),
                        "mean_joint_step_norm": float(np.mean(joint_step_norm)) if len(joint_step_norm) > 0 else 0.0,
                        "max_joint_step_norm": float(np.max(joint_step_norm)) if len(joint_step_norm) > 0 else 0.0,
                    },
                    "surface_constraint": {
                        "demo_mean_abs_z_residual": float(np.mean(np.abs(demo_surface_residual))),
                        "demo_max_abs_z_residual": float(np.max(np.abs(demo_surface_residual))),
                        "ee_mean_abs_z_residual": float(np.mean(np.abs(ee_surface_residual))),
                        "ee_max_abs_z_residual": float(np.max(np.abs(ee_surface_residual))),
                    },
                    "playback": {
                        "mode": "direct_reset",
                        "tracking_error": 0.0,
                        "playback_interp_steps": int(args.playback_interp_steps),
                        "playback_smooth_passes": int(args.playback_smooth_passes),
                        "play_fps": float(play_fps),
                        "render_frame_stride": int(args.render_frame_stride),
                        "trace_surface_clearance": float(args.trace_surface_clearance),
                        "trace_follow_surface": bool(args.trace_follow_surface),
                        "wall_seconds": float(direct["wall_seconds"]),
                        "render_fps_wall": float(direct["render_fps_wall"]),
                        "capture_seconds": float(direct["capture_seconds"]),
                        "write_seconds": float(direct["write_seconds"]),
                        "close_seconds": float(direct["close_seconds"]),
                    },
                    "video_path": direct.get("video_path"),
                }
            )
            print(
                f"[demo {local_idx:02d}/{len(selected):02d}] "
                f"dataset_idx={demo_idx}, source_waypoints={pose_path_source.shape[0]}, "
                f"waypoints={pose_path.shape[0]}, "
                f"frames={joint_path.shape[0]}, "
                f"video_frames={int(direct['frames_written'])}, "
                f"demo_surface_max={float(np.max(np.abs(demo_surface_residual))):.5f}m, "
                f"ik_pos_max={float(ik_summary['max_pos_err']):.5f}m, "
                f"joint_step_max={float(np.max(joint_step_norm)) if len(joint_step_norm) > 0 else 0.0:.4f}rad, "
                f"ee_surface_max={float(np.max(np.abs(ee_surface_residual))):.5f}m, "
                f"duration={float(direct['frames_written']) / max(float(play_fps), 1e-6):.2f}s, "
                f"wall={float(direct['wall_seconds']):.2f}s, "
                f"render_fps={float(direct['render_fps_wall']):.1f}"
            )

    merged_video_path = _concat_videos_ffmpeg(segment_video_paths, final_video_path) if final_video_path else None
    arrays_path = os.path.join(outdir, "surfnormal_demonstrations_arrays.npz")
    np.savez_compressed(arrays_path, **arrays_to_save)
    summary = {
        "task": "surfnormal_demonstration_render",
        "dataset": DATASET_NAME,
        "dataset_config": os.path.abspath(_resolve_path(str(args.dataset_config))),
        "seed": int(args.seed),
        "n_available_demos": int(len(segments)),
        "n_rendered_demos": int(len(selected)),
        "selection": str(args.selection),
        "demo_stride": int(args.demo_stride),
        "playback_cfg": {
            "mode": "direct_reset",
            "sim_dt": float(args.sim_dt),
            "playback_interp_steps": int(args.playback_interp_steps),
            "playback_smooth_passes": int(args.playback_smooth_passes),
            "play_fps": float(play_fps),
            "requested_play_fps": float(args.play_fps),
            "video_slowdown": float(args.video_slowdown),
            "render_frame_stride": int(args.render_frame_stride),
            "trace_surface_clearance": float(args.trace_surface_clearance),
            "trace_follow_surface": bool(args.trace_follow_surface),
            "video_width": int(args.video_width),
            "video_height": int(args.video_height),
            "surface_visual_grid": int(args.surface_visual_grid),
        },
        "video_path": merged_video_path,
        "segment_video_paths": segment_video_paths,
        "arrays": arrays_path,
        "demos": summaries,
    }
    summary_path = os.path.join(outdir, "surfnormal_demonstrations_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[saved] {summary_path}")
    print(f"[saved] {arrays_path}")
    if merged_video_path:
        print(f"[saved] {merged_video_path}")


if __name__ == "__main__":
    main()
