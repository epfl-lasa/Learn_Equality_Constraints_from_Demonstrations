#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from planning.plan_surf_normal import (  # noqa: E402
    DEFAULT_CKPT,
    _choose_device,
    _error_to_true_constraint,
    _load_model,
    _plot_error_distribution_paper,
    _planner_cfg,
    _resolve_path,
    _save_pointwise_csv,
)
from planning.plan_surf_normal_waypoints import (  # noqa: E402
    DATASET_NAME,
    _generate_highfreq_waypoints,
    _plan_waypoint_chain,
    _workspace_surface_z_and_normal_from_xy,
)
from simulation.sim_sine_pose_obsavoid import (  # noqa: E402
    _apply_quat_offset_batch,
    _downsample_traj_for_plot,
    _make_error_curve_video,
    _make_surface_patch_xyz_grid,
    _orientation_error_deg,
    _plot_tracking_errors,
    _quat_inverse_xyzw,
    _quat_xyzw_to_rpy_zyx,
    _rpy_to_quat_xyzw,
)
from simulation.ik_controller import IKConfig, JointTrackConfig, UR5TrajectoryController  # noqa: E402


def _plot_waypoints_tracking_paper(
    *,
    waypoints: np.ndarray,
    planned_traj: np.ndarray,
    executed_traj: np.ndarray,
    out_path: str,
    surface_cfg=None,
) -> None:
    xyz_all = np.concatenate([planned_traj[:, :3], executed_traj[:, :3], waypoints[:, :3]], axis=0).astype(np.float32)
    x_lo = float(np.min(xyz_all[:, 0]) - 0.10)
    x_hi = float(np.max(xyz_all[:, 0]) + 0.10)
    y_lo = float(np.min(xyz_all[:, 1]) - 0.12)
    y_hi = float(np.max(xyz_all[:, 1]) + 0.12)
    gx = np.linspace(x_lo, x_hi, 90, dtype=np.float32)
    gy = np.linspace(y_lo, y_hi, 90, dtype=np.float32)
    gxx, gyy = np.meshgrid(gx, gy)
    gzz, _ = _workspace_surface_z_and_normal_from_xy(gxx.reshape(-1), gyy.reshape(-1), surface_cfg)
    gzz = gzz.reshape(gxx.shape)

    exec_plot = _downsample_traj_for_plot(executed_traj.astype(np.float32), max_points=220)
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    ):
        fig = plt.figure(figsize=(3.55, 3.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            gxx,
            gyy,
            gzz,
            color="#22d3ee",
            alpha=0.28,
            linewidth=0.18,
            edgecolor=(0, 0, 0, 0.22),
            antialiased=True,
            shade=True,
        )
        ax.plot(
            planned_traj[:, 0],
            planned_traj[:, 1],
            planned_traj[:, 2],
            color="#6b7280",
            linewidth=1.0,
            linestyle="--",
            alpha=0.95,
        )
        ax.plot(
            exec_plot[:, 0],
            exec_plot[:, 1],
            exec_plot[:, 2],
            color="#2563eb",
            linewidth=1.3,
            alpha=0.95,
        )
        ax.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            waypoints[:, 2],
            s=6,
            c="#111827",
            alpha=0.95,
        )

        ax.set_xlabel("x", fontsize=8, labelpad=1)
        ax.set_ylabel("y", fontsize=8, labelpad=1)
        ax.set_zlabel("z", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=7, pad=0)
        z_all = np.concatenate([gzz[np.isfinite(gzz)], xyz_all[:, 2]], axis=0).astype(np.float32)
        ax.set_xlim(float(np.min(gx)), float(np.max(gx)))
        ax.set_ylim(float(np.min(gy)), float(np.max(gy)))
        ax.set_zlim(float(np.min(z_all) - 0.06), float(np.max(z_all) + 0.10))
        ax.view_init(elev=58, azim=-66)
        try:
            ax.dist = 8.5
        except Exception:
            pass

        handles = [
            Patch(facecolor="#22d3ee", edgecolor=(0, 0, 0, 0.22), alpha=0.28),
            Line2D([0], [0], color="#6b7280", linewidth=1.3, linestyle="--"),
            Line2D([0], [0], color="#2563eb", linewidth=1.5),
            Line2D([0], [0], color="#111827", marker="o", linestyle="None", markersize=3.0),
        ]
        labels = [
            "True equality constraint",
            "Planned path",
            "Executed path",
            "Waypoints",
        ]
        ax.legend(handles, labels, loc="upper left", frameon=False)
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.998)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a single UR5 execution for the sine-pose waypoint transfer task."
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Checkpoint path for the learned constraint model.")
    parser.add_argument(
        "--outdir",
        default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed/oncl/sim_sine_pose_waypoints",
        help="Output directory.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--n-waypoints", type=int, default=11, help="Number of cleaning waypoints on the snake curve.")
    parser.add_argument("--curve-amp-x", type=float, default=1.4, help="Snake amplitude along x.")
    parser.add_argument("--snake-freq", type=float, default=2.5, help="Snake oscillation frequency.")
    parser.add_argument("--curve-center-x", type=float, default=0.0, help="Curve center x.")
    parser.add_argument("--curve-y-start", type=float, default=1.0, help="Snake start y.")
    parser.add_argument("--curve-y-end", type=float, default=-1.35, help="Snake end y.")
    parser.add_argument("--seg-waypoints", type=int, default=52, help="Trajectory waypoints per planned segment.")
    parser.add_argument("--planner-mode", choices=["traj_opt", "point_project"], default="traj_opt", help="Planning mode.")

    parser.add_argument("--opt-steps", type=int, default=1240, help="traj_opt iterations.")
    parser.add_argument("--opt-lr", type=float, default=0.01, help="traj_opt learning rate.")
    parser.add_argument("--lam-manifold", type=float, default=1.0, help="Manifold loss weight.")
    parser.add_argument("--lam-len-joint", type=float, default=0.4, help="Path length loss weight.")
    parser.add_argument("--lam-smooth", type=float, default=0.9, help="Smoothness loss weight.")
    parser.add_argument("--trust-scale", type=float, default=0.8, help="Trust-region scale.")
    parser.add_argument("--proj-steps", type=int, default=120, help="Projector steps.")
    parser.add_argument("--proj-alpha", type=float, default=0.3, help="Projector alpha.")
    parser.add_argument("--proj-min-steps", type=int, default=30, help="Projector minimum steps.")

    parser.add_argument("--ik-method", choices=["pybullet", "dls"], default="pybullet", help="IK backend.")
    parser.add_argument("--ik-iters", type=int, default=64, help="Per-point IK iterations.")
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
    parser.add_argument("--terminal-hold-time", type=float, default=0.35, help="Extra hold time at the terminal reference.")
    parser.add_argument("--position-gain", type=float, default=0.32, help="Joint position servo gain.")
    parser.add_argument("--velocity-gain", type=float, default=1.05, help="Joint velocity servo gain.")
    parser.add_argument("--max-force", type=float, default=140.0, help="Per-joint max servo force.")
    parser.add_argument("--settle-steps", type=int, default=24, help="Extra settle steps before tracking.")

    parser.add_argument("--draw-ee-trace", type=int, default=1, help="1 to draw executed end-effector trace.")
    parser.add_argument("--draw-ref-trace", type=int, default=0, help="1 to draw reference end-effector trace.")
    parser.add_argument("--draw-surface-wireframe", type=int, default=1, help="1 to draw the local sine surface patch.")
    parser.add_argument("--hide-gripper", type=int, default=1, help="1 to hide the Robotiq gripper.")
    parser.add_argument("--trace-stride", type=int, default=16, help="Render one trajectory segment every this many samples.")
    parser.add_argument("--trace-width", type=float, default=3.0, help="Trajectory trace width.")
    parser.add_argument("--surface-line-stride", type=int, default=4, help="Surface wireframe stride.")
    parser.add_argument("--surface-line-width", type=float, default=1.2, help="Surface wireframe line width.")
    parser.add_argument("--waypoint-marker-radius", type=float, default=0.006, help="Waypoint marker sphere radius in the PyBullet render.")

    parser.add_argument("--camera-distance", type=float, default=1.85, help="Saved-video camera distance.")
    parser.add_argument("--camera-yaw", type=float, default=38.0, help="Saved-video camera yaw in degrees.")
    parser.add_argument("--camera-pitch", type=float, default=-42.0, help="Saved-video camera pitch in degrees.")
    parser.add_argument("--camera-fov", type=float, default=52.0, help="Saved-video camera vertical FOV in degrees.")
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=[0.10, -0.18, 1.02],
        metavar=("X", "Y", "Z"),
        help="Saved-video camera target position.",
    )

    parser.add_argument(
        "--gui",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="0: no render/video; 1: offscreen render+video; 2: GUI render only, no video logging.",
    )
    parser.add_argument("--realtime", type=int, default=0, help="1 to sleep between sim steps.")
    parser.add_argument("--video-name", default="sinepose_waypoints_ur5_tracking.mp4", help="MP4 filename.")
    parser.add_argument("--video-slowdown", type=float, default=1, help="Slow down saved video playback.")
    parser.add_argument("--save-error-video", type=int, default=1, help="1 to save a synchronized error-curve video.")
    args = parser.parse_args()

    ckpt_path = _resolve_path(args.ckpt)
    outdir = _resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(int(args.seed))
    device = _choose_device(str(args.device))
    model, ckpt = _load_model(ckpt_path, device=device)
    surface_cfg = SimpleNamespace(**dict(ckpt.get("cfg", {})))

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
    pose_path = _plan_waypoint_chain(
        model=model,
        cfg=cfg,
        waypoints=waypoints,
        seg_waypoints=int(args.seg_waypoints),
        planner_mode=str(args.planner_mode),
    )[0].astype(np.float32)

    planned_surface_pos_err, planned_surface_ori_err_deg = _error_to_true_constraint(
        pose_path,
        surface_cfg=surface_cfg,
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
        keep_visuals_on_finish=bool(int(args.gui) == 2),
        enable_keyboard_pause=True,
        trace_stride=int(args.trace_stride),
        trace_width=float(args.trace_width),
        surface_line_stride=int(args.surface_line_stride),
        surface_line_width=float(args.surface_line_width),
        waypoint_marker_radius=float(args.waypoint_marker_radius),
        realtime=bool(args.realtime),
        video_slowdown=float(args.video_slowdown),
    )

    save_video = int(args.gui) == 1
    use_gui = int(args.gui) == 2
    video_path = os.path.join(outdir, str(args.video_name)) if save_video else None
    error_video_path = None
    if int(args.save_error_video) == 1:
        video_root, video_ext = os.path.splitext(str(args.video_name))
        error_video_path = os.path.join(outdir, f"{video_root}_errors{video_ext or '.mp4'}")

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
        surface_xyz_grid = _make_surface_patch_xyz_grid(pose_path, surface_cfg=surface_cfg)
        joint_path, ik_summary = ctrl.solve_pose_path_ik(pose_path, cfg=ik_cfg)
        time_s, q_ref, qd_ref = ctrl.time_parameterize_joint_path(joint_path, cfg=track_cfg)

        pose_path_dense = np.zeros((len(time_s), 3), dtype=np.float32)
        t_src_dense = np.linspace(0.0, float(time_s[-1]), num=len(pose_path), dtype=np.float32)
        for j in range(3):
            pose_path_dense[:, j] = np.interp(time_s, t_src_dense, pose_path[:, j]).astype(np.float32)

        track = ctrl.track_joint_trajectory(
            q_ref,
            qd_ref=qd_ref,
            cfg=track_cfg,
            video_path=video_path,
            ee_ref_pos=pose_path_dense,
            surface_xyz_grid=surface_xyz_grid,
            waypoint_xyz=waypoints[:, :3].astype(np.float32),
            obstacle_center_xy=None,
            obstacle_radius=None,
        )

    time_exec = np.arange(len(track["q_meas"]), dtype=np.float32) * float(track["sim_dt"])
    pos_des_dense = np.zeros((len(time_exec), 3), dtype=np.float32)
    q_des_quat = np.stack([_rpy_to_quat_xyzw(r) for r in pose_path[:, 3:6]], axis=0).astype(np.float32)
    q_des_quat_dense = np.zeros((len(time_exec), 4), dtype=np.float32)
    t_src = np.linspace(0.0, float(time_exec[-1]), num=len(pose_path), dtype=np.float32)
    for j in range(3):
        pos_des_dense[:, j] = np.interp(time_exec, t_src, pose_path[:, j]).astype(np.float32)
    for j in range(4):
        q_des_quat_dense[:, j] = np.interp(time_exec, t_src, q_des_quat[:, j]).astype(np.float32)
    q_des_quat_dense /= np.maximum(np.linalg.norm(q_des_quat_dense, axis=1, keepdims=True), 1e-8)

    tool_off_quat = _rpy_to_quat_xyzw(np.asarray(ik_cfg.tool_frame_offset_rpy, dtype=np.float32))
    tool_off_inv = _quat_inverse_xyzw(tool_off_quat[None, :])[0]
    ee_quat_task = _apply_quat_offset_batch(track["ee_quat_xyzw"], tool_off_inv)
    exec_rpy = _quat_xyzw_to_rpy_zyx(ee_quat_task)
    exec_pose = np.concatenate([track["ee_pos"], exec_rpy], axis=1).astype(np.float32)

    pos_track_err = np.linalg.norm(track["ee_pos"] - pos_des_dense, axis=1).astype(np.float32)
    ori_track_err_deg = _orientation_error_deg(q_des_quat_dense, ee_quat_task).astype(np.float32)
    surface_pos_err, surface_ori_err_deg = _error_to_true_constraint(exec_pose, surface_cfg=surface_cfg)
    saved_error_video_path = None
    if error_video_path is not None:
        saved_error_video_path = _make_error_curve_video(
            pos_err=surface_pos_err.astype(np.float32),
            ori_err_deg=surface_ori_err_deg.astype(np.float32),
            sim_dt=float(track["sim_dt"]),
            video_fps=int(track_cfg.video_fps),
            video_slowdown=float(track_cfg.video_slowdown),
            out_path=error_video_path,
        )

    arrays_path = os.path.join(outdir, "sinepose_waypoints_ur5_tracking_arrays.npz")
    np.savez_compressed(
        arrays_path,
        pose_path=pose_path.astype(np.float32),
        waypoints=waypoints.astype(np.float32),
        joint_path=joint_path.astype(np.float32),
        q_ref=q_ref.astype(np.float32),
        qd_ref=qd_ref.astype(np.float32),
        q_meas=track["q_meas"].astype(np.float32),
        qd_meas=track["qd_meas"].astype(np.float32),
        ee_pos=track["ee_pos"].astype(np.float32),
        ee_quat_xyzw=track["ee_quat_xyzw"].astype(np.float32),
        ee_quat_task_xyzw=ee_quat_task.astype(np.float32),
        pos_track_err=pos_track_err.astype(np.float32),
        ori_track_err_deg=ori_track_err_deg.astype(np.float32),
        surface_pos_err=surface_pos_err.astype(np.float32),
        surface_ori_err_deg=surface_ori_err_deg.astype(np.float32),
        planned_surface_pos_err=planned_surface_pos_err.astype(np.float32),
        planned_surface_ori_err_deg=planned_surface_ori_err_deg.astype(np.float32),
    )

    plan_case = SimpleNamespace(traj=pose_path.astype(np.float32))
    exec_case = SimpleNamespace(traj=exec_pose.astype(np.float32))
    planned_csv = os.path.join(outdir, "sinepose_waypoints_planned_pointwise_errors.csv")
    executed_csv = os.path.join(outdir, "sinepose_waypoints_executed_pointwise_errors.csv")
    _save_pointwise_csv(planned_csv, [plan_case], planned_surface_pos_err, planned_surface_ori_err_deg)
    _save_pointwise_csv(executed_csv, [exec_case], surface_pos_err, surface_ori_err_deg)

    tracking_fig = os.path.join(outdir, "sinepose_waypoints_tracking_overview.png")
    _plot_waypoints_tracking_paper(
        waypoints=waypoints.astype(np.float32),
        planned_traj=pose_path.astype(np.float32),
        executed_traj=exec_pose.astype(np.float32),
        out_path=tracking_fig,
        surface_cfg=surface_cfg,
    )
    dist_fig = os.path.join(outdir, "sinepose_waypoints_tracking_error_distribution_paper.png")
    _plot_error_distribution_paper(
        pos_err=surface_pos_err.astype(np.float32),
        ang_err_deg=surface_ori_err_deg.astype(np.float32),
        pos_err_compare=planned_surface_pos_err.astype(np.float32),
        ang_err_deg_compare=planned_surface_ori_err_deg.astype(np.float32),
        primary_label="Executed",
        compare_label="Planned",
        out_path=dist_fig,
    )
    tracking_err_fig = os.path.join(outdir, "sinepose_waypoints_ur5_tracking_errors.png")
    _plot_tracking_errors(
        segment_series=[
            {
                "time_s": time_exec.astype(np.float32),
                "pos_track_err": pos_track_err.astype(np.float32),
                "axis_align_err_deg": surface_ori_err_deg.astype(np.float32),
                "joint_err_norm": track["joint_err_norm"].astype(np.float32),
                "forced_cross_obstacle": False,
            }
        ],
        out_path=tracking_err_fig,
    )

    summary = {
        "task": "sinepose_waypoints_tracking",
        "dataset": DATASET_NAME,
        "ckpt": ckpt_path,
        "seed": int(args.seed),
        "n_waypoints": int(args.n_waypoints),
        "planner_waypoints_per_segment": int(args.seg_waypoints),
        "planner_pose_waypoints_total": int(pose_path.shape[0]),
        "ik_method": str(ik_summary.get("method", args.ik_method)),
        "ik": {
            "mean_pos_err": float(ik_summary["mean_pos_err"]),
            "max_pos_err": float(ik_summary["max_pos_err"]),
            "mean_ori_err_deg": float(np.degrees(float(ik_summary["mean_ori_err_rad"]))),
            "max_ori_err_deg": float(np.degrees(float(ik_summary["max_ori_err_rad"]))),
            "mean_iters": float(ik_summary["mean_iters"]),
        },
        "tracking": {
            "mean_pos_track_err": float(np.mean(pos_track_err)),
            "max_pos_track_err": float(np.max(pos_track_err)),
            "mean_ori_track_err_deg": float(np.mean(ori_track_err_deg)),
            "max_ori_track_err_deg": float(np.max(ori_track_err_deg)),
            "mean_joint_err_norm": float(np.mean(track["joint_err_norm"])),
            "max_joint_err_norm": float(np.max(track["joint_err_norm"])),
        },
        "planned_constraint_error": {
            "mean_surface_pos_err": float(np.mean(planned_surface_pos_err)),
            "std_surface_pos_err": float(np.std(planned_surface_pos_err)),
            "max_surface_pos_err": float(np.max(planned_surface_pos_err)),
            "mean_surface_ori_err_deg": float(np.mean(planned_surface_ori_err_deg)),
            "std_surface_ori_err_deg": float(np.std(planned_surface_ori_err_deg)),
            "max_surface_ori_err_deg": float(np.max(planned_surface_ori_err_deg)),
        },
        "executed_constraint_error": {
            "mean_surface_pos_err": float(np.mean(surface_pos_err)),
            "std_surface_pos_err": float(np.std(surface_pos_err)),
            "max_surface_pos_err": float(np.max(surface_pos_err)),
            "mean_surface_ori_err_deg": float(np.mean(surface_ori_err_deg)),
            "std_surface_ori_err_deg": float(np.std(surface_ori_err_deg)),
            "max_surface_ori_err_deg": float(np.max(surface_ori_err_deg)),
        },
        "mean_position_error": float(np.mean(surface_pos_err)),
        "std_position_error": float(np.std(surface_pos_err)),
        "mean_orientation_error_deg": float(np.mean(surface_ori_err_deg)),
        "std_orientation_error_deg": float(np.std(surface_ori_err_deg)),
        "render": {
            "camera_distance": float(args.camera_distance),
            "camera_yaw": float(args.camera_yaw),
            "camera_pitch": float(args.camera_pitch),
            "camera_fov": float(args.camera_fov),
            "camera_target": [float(v) for v in args.camera_target],
            "waypoint_marker_radius": float(args.waypoint_marker_radius),
        },
        "outputs": {
            "video": str(track.get("video_path")) if track.get("video_path") else None,
            "error_video": str(saved_error_video_path) if saved_error_video_path else None,
            "tracking_overview_plot": tracking_fig,
            "distribution_plot": dist_fig,
            "tracking_error_plot": tracking_err_fig,
            "planned_pointwise_csv": planned_csv,
            "executed_pointwise_csv": executed_csv,
            "arrays": arrays_path,
        },
    }
    summary_path = os.path.join(outdir, "sinepose_waypoints_ur5_tracking_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        "[summary] "
        f"ik_pos_mean={summary['ik']['mean_pos_err']:.5f}, "
        f"track_pos_mean={summary['tracking']['mean_pos_track_err']:.5f}, "
        f"exec_surface_pos_mean={summary['mean_position_error']:.5f}, "
        f"exec_surface_pos_std={summary['std_position_error']:.5f}, "
        f"exec_surface_ori_mean_deg={summary['mean_orientation_error_deg']:.3f}, "
        f"exec_surface_ori_std_deg={summary['std_orientation_error_deg']:.3f}"
    )
    print(f"[saved] {summary_path}")
    print(f"[saved] {tracking_fig}")
    print(f"[saved] {dist_fig}")
    print(f"[saved] {tracking_err_fig}")
    print(f"[saved] {planned_csv}")
    print(f"[saved] {executed_csv}")
    print(f"[saved] {arrays_path}")
    if track.get("video_path"):
        print(f"[saved] {track['video_path']}")
    if saved_error_video_path:
        print(f"[saved] {saved_error_video_path}")


if __name__ == "__main__":
    main()
