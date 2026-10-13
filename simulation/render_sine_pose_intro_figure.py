#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.render_sine_pose_demonstrations import (  # noqa: E402
    DATASET_NAME,
    DEFAULT_DATASET_CONFIG,
    _load_demo_cfg,
    _make_surface_patch_xyz_grid,
    _prepare_playback_pose_path,
    _repo_root,
    _resolve_path,
    _select_segments,
    _split_demo_segments,
    _stride_demo,
    _trace_visual_xyz,
)
from datasets.constraint_datasets import generate_dataset, sine_surface_z_and_normal_from_xy  # noqa: E402
from simulation.ik_controller import (  # noqa: E402
    IKConfig,
    UR5TrajectoryController,
    _quat_conjugate,
    _quat_multiply,
    _rpy_to_quat_xyzw,
)


def _crop_frame(
    frame: np.ndarray,
    *,
    crop_left: float,
    crop_right: float,
    crop_top: float,
    crop_bottom: float,
) -> np.ndarray:
    img = np.asarray(frame, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    h, w = img.shape[:2]
    l = int(round(float(np.clip(crop_left, 0.0, 0.45)) * w))
    r = int(round(float(np.clip(crop_right, 0.0, 0.45)) * w))
    t = int(round(float(np.clip(crop_top, 0.0, 0.45)) * h))
    b = int(round(float(np.clip(crop_bottom, 0.0, 0.45)) * h))
    x0 = min(max(0, l), max(0, w - 2))
    x1 = max(x0 + 1, min(w, w - r))
    y0 = min(max(0, t), max(0, h - 2))
    y1 = max(y0 + 1, min(h, h - b))
    cropped = img[y0:y1, x0:x1, :]
    return cropped.copy()


def _crop_point(
    xy_px: tuple[float, float],
    *,
    width: int,
    height: int,
    crop_left: float,
    crop_right: float,
    crop_top: float,
    crop_bottom: float,
) -> tuple[float, float] | None:
    x, y = float(xy_px[0]), float(xy_px[1])
    l = float(np.clip(crop_left, 0.0, 0.45)) * float(width)
    r = float(np.clip(crop_right, 0.0, 0.45)) * float(width)
    t = float(np.clip(crop_top, 0.0, 0.45)) * float(height)
    b = float(np.clip(crop_bottom, 0.0, 0.45)) * float(height)
    x0 = l
    x1 = float(width) - r
    y0 = t
    y1 = float(height) - b
    if x < x0 or x > x1 or y < y0 or y > y1:
        return None
    return x - x0, y - y0


def _project_world_to_pixel(
    xyz_world: np.ndarray,
    *,
    camera_target: list[float],
    camera_distance: float,
    camera_yaw: float,
    camera_pitch: float,
    camera_fov: float,
    width: int,
    height: int,
    p,
) -> tuple[float, float] | None:
    view = np.asarray(
        p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[float(v) for v in camera_target],
            distance=float(camera_distance),
            yaw=float(camera_yaw),
            pitch=float(camera_pitch),
            roll=0.0,
            upAxisIndex=2,
        ),
        dtype=np.float32,
    ).reshape((4, 4), order="F")
    proj = np.asarray(
        p.computeProjectionMatrixFOV(
            fov=float(camera_fov),
            aspect=float(width) / max(float(height), 1.0),
            nearVal=0.02,
            farVal=6.0,
        ),
        dtype=np.float32,
    ).reshape((4, 4), order="F")
    pos_h = np.asarray([float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2]), 1.0], dtype=np.float32)
    clip = proj @ (view @ pos_h)
    w = float(clip[3])
    if abs(w) < 1e-8:
        return None
    ndc = clip[:3] / w
    if float(ndc[2]) < -1.2 or float(ndc[2]) > 1.2:
        return None
    px = (float(ndc[0]) * 0.5 + 0.5) * float(width)
    py = (1.0 - (float(ndc[1]) * 0.5 + 0.5)) * float(height)
    return px, py


def _pick_annotated_point(
    pts_world: np.ndarray,
    *,
    target_xy_norm: tuple[float, float],
    camera_target: list[float],
    camera_distance: float,
    camera_yaw: float,
    camera_pitch: float,
    camera_fov: float,
    width: int,
    height: int,
    crop_left: float,
    crop_right: float,
    crop_top: float,
    crop_bottom: float,
    p,
) -> tuple[np.ndarray, tuple[float, float]] | None:
    pts = np.asarray(pts_world, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return None
    target = np.asarray(target_xy_norm, dtype=np.float32)
    best_idx = None
    best_score = None
    best_crop_xy = None
    for i, pt in enumerate(pts):
        px = _project_world_to_pixel(
            pt,
            camera_target=camera_target,
            camera_distance=camera_distance,
            camera_yaw=camera_yaw,
            camera_pitch=camera_pitch,
            camera_fov=camera_fov,
            width=width,
            height=height,
            p=p,
        )
        if px is None:
            continue
        crop_xy = _crop_point(
            px,
            width=width,
            height=height,
            crop_left=crop_left,
            crop_right=crop_right,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
        )
        if crop_xy is None:
            continue
        cx, cy = crop_xy
        score = float(
            (cx / max(1.0, float(width) * (1.0 - crop_left - crop_right)) - target[0]) ** 2
            + (cy / max(1.0, float(height) * (1.0 - crop_top - crop_bottom)) - target[1]) ** 2
        )
        if best_score is None or score < best_score:
            best_score = score
            best_idx = i
            best_crop_xy = (float(cx), float(cy))
    if best_idx is None or best_crop_xy is None:
        return None
    return pts[int(best_idx)].copy(), best_crop_xy


def _draw_overlay_boxes(
    frame: np.ndarray,
    *,
    traj_rgba: tuple[float, float, float, float],
    arrow_rgba: tuple[float, float, float, float],
    sample_rgba: tuple[float, float, float, float],
) -> np.ndarray:
    img = np.asarray(frame, dtype=np.uint8)
    h, w = img.shape[:2]
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()

    box_fill = (1.0, 1.0, 1.0, 0.84)
    box_edge = (0.58, 0.62, 0.69, 0.95)
    text_rgb = (0.14, 0.17, 0.21, 1.0)

    left_box_w = 0.39
    right_box_w = 0.47
    legend_h = 0.2
    constraints_h = 0.2
    left_x = 0.055
    right_x = 1.0 - left_x - right_box_w
    bottom = 0.055

    for (bx, by, bw, bh) in [(left_x, bottom, left_box_w, legend_h), (right_x, bottom, right_box_w, constraints_h)]:
        ax.add_patch(
            FancyBboxPatch(
                (bx, by),
                bw,
                bh,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.4,
                edgecolor=box_edge,
                facecolor=box_fill,
                transform=ax.transAxes,
            )
        )

    # Legend box
    sym_x = left_x + 0.005
    txt_x = left_x + 0.055
    row0 = bottom + legend_h - 0.038
    row_gap = 0.05

    ax.plot([sym_x, sym_x + 0.04], [row0, row0], color=traj_rgba[:3], linewidth=5.0,
            transform=ax.transAxes, solid_capstyle="round")
    ax.text(txt_x, row0, r"Demonstrated states $\mathcal{D}$", fontsize=18, va="center", color=text_rgb, transform=ax.transAxes)

    y2 = row0 - row_gap
    ax.add_patch(
        FancyArrowPatch(
            (sym_x, y2 - 0.012),
            (sym_x + 0.04, y2 + 0.015),
            mutation_scale=16,
            linewidth=3.0,
            arrowstyle="-|>",
            color=arrow_rgba[:3],
            transform=ax.transAxes,
        )
    )
    ax.text(txt_x, y2, "End-effector orientations", fontsize=18, va="center", color=text_rgb, transform=ax.transAxes)

    y3 = row0 - 2 * row_gap
    ax.add_patch(Circle((sym_x + 0.02, y3), 0.0085, transform=ax.transAxes, color=sample_rgba[:3], alpha=0.95))
    ax.text(txt_x, y3, r"Perturbed samples $\widetilde{\mathcal{D}}$", fontsize=18, va="center", color=text_rgb, transform=ax.transAxes)

    # Constraints box
    c_y = bottom
    ax.text(right_x + 0.018, c_y + constraints_h - 0.034, "Constraint learning objectives", fontsize=18, fontweight="semibold",
            color=text_rgb, transform=ax.transAxes)
    csym_x = right_x + 0.013
    ctxt_x = right_x + 0.078
    crow0 = c_y + constraints_h - 0.07
    crow_gap = 0.053

    ax.plot([csym_x, csym_x + 0.04], [crow0, crow0], color=traj_rgba[:3], linewidth=5.0,
            transform=ax.transAxes, solid_capstyle="round")
    ax.text(ctxt_x, crow0, r"$h_\theta(x)=0$", fontsize=18, va="center",
            color=text_rgb, transform=ax.transAxes)

    cy2 = crow0 - crow_gap
    ax.add_patch(Circle((csym_x + 0.02, cy2), 0.0085, transform=ax.transAxes, color=sample_rgba[:3], alpha=0.95))
    ax.text(ctxt_x, cy2, r"$\|\nabla h_\theta^{(i)}(x)\|=1$  and", fontsize=18, va="center",
            color=text_rgb, transform=ax.transAxes)

    cy3 = crow0 - 2 * crow_gap
    # ax.add_patch(Circle((csym_x + 0.02, cy3), 0.0085, transform=ax.transAxes, color=sample_rgba[:3], alpha=0.95))
    ax.text(ctxt_x, cy3, r"$\nabla h_\theta^{(i)}(x)^\top \nabla h_\theta^{(j)}(x)=0,\ \forall i\neq j$", fontsize=18, va="center",
            color=text_rgb, transform=ax.transAxes)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    out = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return out


def _parse_index_list(raw: str) -> list[int]:
    txt = str(raw).strip()
    if not txt:
        return []
    out: list[int] = []
    for token in txt.split(","):
        tok = str(token).strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _quat_rotate_vec(q_xyzw: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float32).reshape(4)
    v = np.asarray(v_xyz, dtype=np.float32).reshape(3)
    qv = np.asarray([v[0], v[1], v[2], 0.0], dtype=np.float32)
    out = _quat_multiply(_quat_multiply(q, qv), _quat_conjugate(q))
    return out[:3].astype(np.float32)


def _safe_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float32).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-8)
    ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(n, ref))) > 0.95:
        ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    t1 = np.cross(n, ref).astype(np.float32)
    t1 = t1 / max(float(np.linalg.norm(t1)), 1e-8)
    t2 = np.cross(n, t1).astype(np.float32)
    t2 = t2 / max(float(np.linalg.norm(t2)), 1e-8)
    return t1.astype(np.float32), t2.astype(np.float32)


def _draw_polyline(
    ctrl: UR5TrajectoryController,
    pts_xyz: np.ndarray,
    *,
    radius: float,
    rgba: tuple[float, float, float, float],
    body_list: list[int],
) -> None:
    pts = np.asarray(pts_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return
    for a, b in zip(pts[:-1], pts[1:]):
        ctrl._add_visual_cylinder_segment(
            a,
            b,
            radius=float(radius),
            rgba=rgba,
            specular=(0.03, 0.03, 0.03),
            body_list=body_list,
        )


def _draw_point_cloud(
    ctrl: UR5TrajectoryController,
    pts_xyz: np.ndarray,
    *,
    radius: float,
    rgba: tuple[float, float, float, float],
    body_list: list[int],
) -> None:
    pts = np.asarray(pts_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return
    for p in pts:
        ctrl._add_visual_sphere_marker(
            p,
            radius=float(radius),
            rgba=rgba,
            specular=(0.02, 0.02, 0.02),
            body_list=body_list,
        )


def _draw_orientation_arrows(
    ctrl: UR5TrajectoryController,
    pose_path: np.ndarray,
    surface_cfg: SimpleNamespace,
    *,
    stride: int,
    axis: str,
    sign: float,
    length: float,
    shaft_radius: float,
    head_radius: float,
    base_clearance: float,
    rgba: tuple[float, float, float, float],
    body_list: list[int],
) -> None:
    poses = np.asarray(pose_path, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] < 6 or len(poses) == 0:
        return
    axis_map = {
        "x": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    }
    local_axis = axis_map[str(axis).lower()]
    idx = np.arange(0, len(poses), int(max(1, stride)), dtype=np.int32)
    if int(idx[-1]) != len(poses) - 1:
        idx = np.concatenate([idx, np.asarray([len(poses) - 1], dtype=np.int32)])
    sel = poses[idx]
    base_pts = _trace_visual_xyz(
        sel[:, :3],
        surface_cfg,
        clearance=float(max(0.0, base_clearance)),
        follow_surface=True,
    )
    for p_vis, pose in zip(base_pts, sel):
        quat = _rpy_to_quat_xyzw(pose[3:6].astype(np.float32))
        axis_world = _quat_rotate_vec(quat, local_axis) * float(sign)
        axis_world = axis_world / max(float(np.linalg.norm(axis_world)), 1e-8)
        tip = p_vis + axis_world.astype(np.float32) * float(length)
        shaft_end = p_vis + axis_world.astype(np.float32) * float(0.74 * length)
        ctrl._add_visual_cylinder_segment(
            p_vis,
            shaft_end,
            radius=float(shaft_radius),
            rgba=rgba,
            specular=(0.04, 0.03, 0.03),
            body_list=body_list,
        )
        h1, h2 = _safe_tangent_basis(axis_world)
        head_len = float(0.26 * length)
        head_span = float(0.10 * length)
        head_starts = [
            tip - axis_world * head_len + h1 * head_span,
            tip - axis_world * head_len - h1 * head_span,
            tip - axis_world * head_len + h2 * head_span,
            tip - axis_world * head_len - h2 * head_span,
        ]
        for hs in head_starts:
            ctrl._add_visual_cylinder_segment(
                hs.astype(np.float32),
                tip.astype(np.float32),
                radius=float(head_radius),
                rgba=rgba,
                specular=(0.04, 0.03, 0.03),
                body_list=body_list,
            )


def _sample_visible_perturbations(
    pose_paths: list[np.ndarray],
    surface_cfg: SimpleNamespace,
    *,
    n_samples: int,
    tangent_std: float,
    normal_std: float,
    base_clearance: float,
    seed: int,
) -> np.ndarray:
    paths = [np.asarray(p, dtype=np.float32) for p in pose_paths if len(np.asarray(p, dtype=np.float32)) > 0]
    if not paths or int(n_samples) <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    all_xyz = np.concatenate([p[:, :3] for p in paths], axis=0).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    sel_idx = rng.choice(len(all_xyz), size=int(n_samples), replace=(len(all_xyz) < int(n_samples)))
    anchors = all_xyz[sel_idx].astype(np.float32)
    z_surf, normals = sine_surface_z_and_normal_from_xy(anchors[:, 0], anchors[:, 1], surface_cfg)
    anchors[:, 2] = np.asarray(z_surf, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    pts = np.zeros((len(anchors), 3), dtype=np.float32)
    for i, (a, nrm) in enumerate(zip(anchors, normals)):
        t1, t2 = _safe_tangent_basis(nrm)
        off_t1 = float(rng.normal(0.0, float(tangent_std)))
        off_t2 = float(rng.normal(0.0, float(tangent_std)))
        off_n = float(base_clearance + rng.normal(0.0, float(normal_std)))
        pts[i] = (
            a.astype(np.float32)
            + t1 * off_t1
            + t2 * off_t2
            + nrm.astype(np.float32) * off_n
        ).astype(np.float32)
    return pts.astype(np.float32)


def _select_robot_pose_path(
    selected: list[tuple[int, np.ndarray]],
    demo_cfg: SimpleNamespace,
    *,
    local_index: int,
    smooth_passes: int,
) -> np.ndarray:
    idx = int(np.clip(int(local_index), 0, max(0, len(selected) - 1)))
    pose_src = np.asarray(selected[idx][1], dtype=np.float32)
    return _prepare_playback_pose_path(
        pose_src,
        demo_cfg,
        interp_steps=1,
        smooth_passes=int(max(0, smooth_passes)),
    ).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a single PyBullet intro figure for the SurfNormal task.")
    ap.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG, help="Dataset config JSON.")
    ap.add_argument(
        "--outdir",
        default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/surfnormal_intro_figure",
        help="Output directory.",
    )
    ap.add_argument("--image-name", default="surfnormal_intro_figure.png", help="Output image filename.")
    ap.add_argument("--seed", type=int, default=0, help="Dataset generation seed.")
    ap.add_argument("--n-demos", type=int, default=2, help="Number of demonstration trajectories to draw.")
    ap.add_argument("--demo-start-index", type=int, default=0, help="Start demo index for contiguous selection.")
    ap.add_argument("--demo-indices", default="", help="Comma-separated explicit demo indices to render.")
    ap.add_argument("--skip-demo-indices", default="", help="Comma-separated demo indices to exclude.")
    ap.add_argument("--selection", choices=["last", "first", "uniform"], default="last", help="Trajectory selection mode.")
    ap.add_argument("--demo-stride", type=int, default=2, help="Keep every Nth point from each selected demonstration.")
    ap.add_argument("--demo-smooth-passes", type=int, default=1, help="Render-only smoothing passes for displayed trajectories.")

    ap.add_argument("--sample-count", type=int, default=84, help="Number of visible perturbed sample points.")
    ap.add_argument("--sample-tangent-std", type=float, default=0.025, help="Tangential std for visible perturbed samples.")
    ap.add_argument("--sample-normal-std", type=float, default=0.010, help="Normal-direction std for visible perturbed samples.")
    ap.add_argument("--sample-base-clearance", type=float, default=0.014, help="Base lift above the surface for visible perturbed samples.")

    ap.add_argument("--orientation-arrow-stride", type=int, default=8, help="Show one orientation arrow every N trajectory points.")
    ap.add_argument("--orientation-arrow-axis", choices=["x", "y", "z"], default="z", help="Local axis used for orientation arrows.")
    ap.add_argument("--orientation-arrow-sign", type=float, default=1.0, help="Multiply the arrow direction by this sign.")
    ap.add_argument("--orientation-arrow-length", type=float, default=0.070, help="Orientation arrow length in meters.")
    ap.add_argument("--orientation-arrow-radius", type=float, default=0.0015, help="Orientation arrow shaft radius.")
    ap.add_argument("--orientation-arrow-head-radius", type=float, default=0.0023, help="Orientation arrow head segment radius.")
    ap.add_argument("--orientation-arrow-clearance", type=float, default=0.016, help="Normal lift used for arrow bases.")

    ap.add_argument("--traj-radius", type=float, default=0.0035, help="Trajectory tube radius.")
    ap.add_argument("--traj-clearance", type=float, default=0.012, help="Normal lift for trajectory tubes.")
    ap.add_argument("--sample-radius", type=float, default=0.0065, help="Radius of perturbed sample spheres.")

    ap.add_argument("--show-robot", type=int, default=1, help="1 to place the UR5 at a representative demo pose.")
    ap.add_argument("--robot-demo-local-index", type=int, default=0, help="Which selected demo to use for the representative robot pose.")
    ap.add_argument("--robot-pose-fraction", type=float, default=0.55, help="Pose fraction along the chosen demo for robot placement.")
    ap.add_argument("--hide-gripper", type=int, default=1, help="1 to hide the Robotiq gripper.")

    ap.add_argument("--ik-method", choices=["pybullet", "dls"], default="pybullet", help="IK backend for representative robot placement.")
    ap.add_argument("--ik-iters", type=int, default=64, help="Per-point IK iterations.")
    ap.add_argument("--ik-damping", type=float, default=0.05, help="DLS damping coefficient.")
    ap.add_argument("--ik-step-size", type=float, default=0.6, help="DLS step size.")
    ap.add_argument("--ik-max-delta", type=float, default=0.18, help="Max joint update norm per IK iteration.")
    ap.add_argument("--ik-pos-tol", type=float, default=0.004, help="IK position tolerance in meters.")
    ap.add_argument("--ik-ori-tol-deg", type=float, default=2.5, help="IK orientation tolerance in degrees.")

    ap.add_argument("--sim-dt", type=float, default=1.0 / 240.0, help="PyBullet simulation step.")
    ap.add_argument("--surface-visual-grid", type=int, default=64, help="Grid resolution for the rendered sine surface.")
    ap.add_argument("--surface-line-stride", type=int, default=4, help="Surface visual grid stride.")
    ap.add_argument("--surface-line-width", type=float, default=1.2, help="Surface visual line width.")

    ap.add_argument("--camera-distance", type=float, default=1.10, help="Camera distance.")
    ap.add_argument("--camera-yaw", type=float, default=34.0, help="Camera yaw in degrees.")
    ap.add_argument("--camera-pitch", type=float, default=-42.0, help="Camera pitch in degrees.")
    ap.add_argument("--camera-fov", type=float, default=54.0, help="Camera vertical FOV in degrees.")
    ap.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=[0.38, -0.22, 1.03],
        metavar=("X", "Y", "Z"),
        help="Camera target position.",
    )
    ap.add_argument("--image-width", type=int, default=1400, help="Output image width.")
    ap.add_argument("--image-height", type=int, default=980, help="Output image height.")
    ap.add_argument("--crop-left", type=float, default=0.16, help="Fraction to crop from the left edge after rendering.")
    ap.add_argument("--crop-right", type=float, default=0.18, help="Fraction to crop from the right edge after rendering.")
    ap.add_argument("--crop-top", type=float, default=0.05, help="Fraction to crop from the top edge after rendering.")
    ap.add_argument("--crop-bottom", type=float, default=0.26, help="Fraction to crop from the bottom edge after rendering.")
    ap.add_argument("--gui", type=int, choices=[0, 1], default=0, help="0: offscreen render, 1: open GUI and still save a screenshot.")
    args = ap.parse_args()

    outdir = _resolve_path(str(args.outdir))
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, str(args.image_name))

    demo_cfg = _load_demo_cfg(str(args.dataset_config), seed=int(args.seed))
    x_train, _grid = generate_dataset(DATASET_NAME, demo_cfg)
    x_train = np.asarray(x_train, dtype=np.float32)
    segments = _split_demo_segments(x_train, demo_cfg)
    explicit_indices = _parse_index_list(str(args.demo_indices))
    skip_indices = _parse_index_list(str(args.skip_demo_indices))
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

    pose_paths = [
        _prepare_playback_pose_path(
            np.asarray(seg, dtype=np.float32),
            demo_cfg,
            interp_steps=1,
            smooth_passes=int(max(0, args.demo_smooth_passes)),
        ).astype(np.float32)
        for _, seg in selected
    ]
    common_pose_path = np.concatenate([p for p in pose_paths], axis=0).astype(np.float32)
    surface_xyz_grid = _make_surface_patch_xyz_grid(
        common_pose_path,
        surface_cfg=demo_cfg,
        grid_size=int(args.surface_visual_grid),
    )

    sample_xyz = _sample_visible_perturbations(
        pose_paths,
        demo_cfg,
        n_samples=int(args.sample_count),
        tangent_std=float(args.sample_tangent_std),
        normal_std=float(args.sample_normal_std),
        base_clearance=float(args.sample_base_clearance),
        seed=int(args.seed) + 17,
    )

    traj_rgba = (0.08, 0.36, 0.92, 0.96)
    sample_rgba = (0.36, 0.40, 0.45, 0.78)
    arrow_rgba = (0.70, 0.12, 0.14, 0.96)

    ik_cfg = IKConfig(
        method=str(args.ik_method),
        max_iters=int(args.ik_iters),
        damping=float(args.ik_damping),
        step_size=float(args.ik_step_size),
        max_delta_norm=float(args.ik_max_delta),
        pos_tol=float(args.ik_pos_tol),
        ori_tol_rad=math.radians(float(args.ik_ori_tol_deg)),
    )

    summary: dict[str, object] = {
        "task": "surfnormal_intro_figure",
        "dataset": DATASET_NAME,
        "dataset_config": os.path.abspath(_resolve_path(str(args.dataset_config))),
        "seed": int(args.seed),
        "selected_demo_indices": [int(i) for i, _ in selected],
        "n_selected_demos": int(len(selected)),
        "sample_count": int(len(sample_xyz)),
        "output_image": os.path.abspath(out_png),
        "cwd": _repo_root(),
    }

    with UR5TrajectoryController(
        gui=bool(int(args.gui) == 1),
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
        ctrl._clear_surface_visuals()
        ctrl._clear_trace_visuals()
        ctrl._clear_marker_visuals()
        ctrl._build_surface_visuals(
            np.asarray(surface_xyz_grid, dtype=np.float32),
            stride=int(max(1, args.surface_line_stride)),
            line_width=float(max(0.5, args.surface_line_width)),
        )

        for traj_idx, pose_path in enumerate(pose_paths):
            vis_xyz = _trace_visual_xyz(
                pose_path[:, :3],
                demo_cfg,
                clearance=float(max(0.0, args.traj_clearance)),
                follow_surface=True,
            )
            _draw_polyline(
                ctrl,
                vis_xyz,
                radius=float(args.traj_radius),
                rgba=traj_rgba,
                body_list=ctrl._trace_body_ids,
            )
            _draw_orientation_arrows(
                ctrl,
                pose_path,
                demo_cfg,
                stride=int(args.orientation_arrow_stride),
                axis=str(args.orientation_arrow_axis),
                sign=float(args.orientation_arrow_sign),
                length=float(args.orientation_arrow_length),
                shaft_radius=float(args.orientation_arrow_radius),
                head_radius=float(args.orientation_arrow_head_radius),
                base_clearance=float(args.orientation_arrow_clearance),
                rgba=arrow_rgba,
                body_list=ctrl._marker_body_ids,
            )

        _draw_point_cloud(
            ctrl,
            sample_xyz.astype(np.float32),
            radius=float(args.sample_radius),
            rgba=sample_rgba,
            body_list=ctrl._marker_body_ids,
        )

        if bool(args.show_robot):
            try:
                robot_pose_path = _select_robot_pose_path(
                    selected,
                    demo_cfg,
                    local_index=int(args.robot_demo_local_index),
                    smooth_passes=int(max(0, args.demo_smooth_passes)),
                )
                joint_path, ik_summary = ctrl.solve_pose_path_ik(robot_pose_path, cfg=ik_cfg)
                idx = int(np.clip(round((len(joint_path) - 1) * float(args.robot_pose_fraction)), 0, max(0, len(joint_path) - 1)))
                ctrl.reset_joint_state(np.asarray(joint_path[idx], dtype=np.float32))
                ctrl._p.stepSimulation(physicsClientId=ctrl.client_id)
                summary["robot_pose_demo_dataset_index"] = int(selected[int(np.clip(int(args.robot_demo_local_index), 0, len(selected) - 1))][0])
                summary["robot_pose_path_index"] = int(idx)
                summary["robot_pose_ik_max_pos_err"] = float(ik_summary["max_pos_err"])
                summary["robot_pose_ik_max_ori_err_deg"] = float(np.degrees(float(ik_summary["max_ori_err_rad"])))
            except Exception as e:
                summary["robot_pose_error"] = str(e)

        frame = ctrl.capture_frame(width=int(args.image_width), height=int(args.image_height))
        frame = _crop_frame(
            frame,
            crop_left=float(args.crop_left),
            crop_right=float(args.crop_right),
            crop_top=float(args.crop_top),
            crop_bottom=float(args.crop_bottom),
        )
        frame = _draw_overlay_boxes(
            frame,
            traj_rgba=traj_rgba,
            arrow_rgba=arrow_rgba,
            sample_rgba=sample_rgba,
        )
        imageio.imwrite(out_png, frame)

    summary_path = os.path.join(outdir, "surfnormal_intro_figure_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_png}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
