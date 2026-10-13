#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import numpy as np

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.constraint_datasets import _dual_arm_guided_insertion_frame, _rpy_from_rotmat_zyx
from datasets.ur5_pybullet_utils import (
    _make_pybullet_friendly_urdf,
    pick_default_ee_link_index,
    resolve_dual_ur5_base_cfg,
    resolve_ur5_kinematics_cfg,
)
from simulation.ik_controller import _FFmpegVideoWriter


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_REPO_ROOT, path))


def _crop_frame_by_frac(
    frame: np.ndarray,
    crop_frac: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if crop_frac is None:
        return frame
    x0f, y0f, x1f, y1f = [float(v) for v in crop_frac]
    x0f = min(max(x0f, 0.0), 0.98)
    y0f = min(max(y0f, 0.0), 0.98)
    x1f = min(max(x1f, x0f + 1e-4), 1.0)
    y1f = min(max(y1f, y0f + 1e-4), 1.0)
    h, w = frame.shape[:2]
    x0 = int(round(x0f * w))
    y0 = int(round(y0f * h))
    x1 = int(round(x1f * w))
    y1 = int(round(y1f * h))
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    return frame[y0:y1, x0:x1].copy()


@dataclass
class RobotHandle:
    robot_id: int
    arm_joint_indices: list[int]
    ik_joint_indices: list[int]
    arm_ik_positions: list[int]
    ee_link_index: int
    q_lo: np.ndarray
    q_hi: np.ndarray
    q_range: np.ndarray
    home_q: np.ndarray
    gripper_joint_indices: list[int]


def _preferred_home_q(side: str, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    side_s = str(side).strip().lower()
    if side_s.startswith("r"):
        home = np.asarray([-0.75, -1.95, 2.20, -1.85, -1.57, -0.15], dtype=np.float32)
    else:
        home = np.asarray([0.75, -1.95, 2.20, -1.85, -1.57, 0.15], dtype=np.float32)
    return np.clip(home, q_lo, q_hi).astype(np.float32)


def _rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in np.asarray(rpy, dtype=np.float32).reshape(3)]
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    q = np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )
    q /= max(float(np.linalg.norm(q)), 1e-8)
    return q


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = [float(v) for v in np.asarray(a, dtype=np.float32).reshape(4)]
    bx, by, bz, bw = [float(v) for v in np.asarray(b, dtype=np.float32).reshape(4)]
    q = np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )
    q /= max(float(np.linalg.norm(q)), 1e-8)
    return q


def _right_task_grasp_flip_xyzw() -> np.ndarray:
    # Fixed 180deg flip about local tool-y, so the right gripper approaches
    # the shared link from the opposite side instead of mirroring the left hand.
    return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def _left_task_grasp_flip_xyzw() -> np.ndarray:
    # Fixed 180deg flip about local tool-x to match this gripper's grasp frame
    # to the task/object frame convention used in the dataset.
    return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _quat_from_local_x_rotation_xyzw(angle_rad: float) -> np.ndarray:
    half = 0.5 * float(angle_rad)
    return np.asarray([math.sin(half), 0.0, 0.0, math.cos(half)], dtype=np.float32)


def _parse_rpy_deg_triplet(text: str) -> np.ndarray:
    raw = str(text).strip()
    if not raw:
        return np.zeros((3,), dtype=np.float32)
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(vals) != 3:
        raise ValueError("task offset RPY must have three comma-separated values: roll,pitch,yaw")
    return np.asarray(vals, dtype=np.float32)


def _compose_task_local_offset_quat_xyzw(*, task_roll_rad: float, extra_rpy_rad: np.ndarray | None) -> np.ndarray:
    quat = _quat_from_local_x_rotation_xyzw(float(task_roll_rad))
    if extra_rpy_rad is not None:
        extra = np.asarray(extra_rpy_rad, dtype=np.float32).reshape(-1)
        if len(extra) >= 3 and float(np.linalg.norm(extra[:3])) > 1e-8:
            quat = _quat_multiply_xyzw(quat, _rpy_to_quat_xyzw(extra[:3]))
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    return quat.astype(np.float32)


def _quat_from_z_axis(axis: np.ndarray) -> np.ndarray:
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    v = np.asarray(axis, dtype=np.float32).reshape(3)
    v /= max(float(np.linalg.norm(v)), 1e-8)
    dot = float(np.clip(np.dot(z_axis, v), -1.0, 1.0))
    if dot > 1.0 - 1e-7:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if dot < -1.0 + 1e-7:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ax = np.cross(z_axis, v).astype(np.float32)
    ax /= max(float(np.linalg.norm(ax)), 1e-8)
    ang = math.acos(dot)
    s = math.sin(0.5 * ang)
    return np.asarray([ax[0] * s, ax[1] * s, ax[2] * s, math.cos(0.5 * ang)], dtype=np.float32)


def _quat_xyzw_to_rpy_zyx(p: Any, quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    rot = np.asarray(p.getMatrixFromQuaternion([float(v) for v in quat]), dtype=np.float32).reshape(3, 3)
    return np.asarray(_rpy_from_rotmat_zyx(rot.astype(np.float64)), dtype=np.float32)


def _make_demo_path(cfg: Any, *, n_steps: int, seed: int) -> dict[str, np.ndarray]:
    n = int(max(2, n_steps))
    grasp_span = float(getattr(cfg, "dual_arm_grasp_span", 0.45))
    x_span = float(getattr(cfg, "dual_arm_curve_x_span", 0.45))
    y_amp = float(getattr(cfg, "dual_arm_curve_y_amp", 0.12))
    y_freq = float(getattr(cfg, "dual_arm_curve_y_freq", 1.0))
    z_base = float(getattr(cfg, "dual_arm_curve_z_base", 0.76))
    z_amp = float(getattr(cfg, "dual_arm_curve_z_amp", 0.08))
    z_freq = float(getattr(cfg, "dual_arm_curve_z_freq", 0.7))
    z_half_range = float(getattr(cfg, "dual_arm_vertical_half_range", z_amp))

    # Use a monotonic guide-curve sweep for a clean physical demonstration.
    # The dataset sampler is stochastic; this script intentionally renders one
    # continuous representative task path.
    rng = np.random.default_rng(int(seed))
    s = np.linspace(-0.92, 0.92, num=n, dtype=np.float32)
    center, tang, _normal, _binormal = _dual_arm_guided_insertion_frame(
        s,
        x_span=x_span,
        y_amp=y_amp,
        y_freq=y_freq,
        z_base=z_base,
        z_amp=z_amp,
        z_freq=z_freq,
    )
    tau = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    # Smooth but diverse height profiles. Normalize before scaling so the
    # rendered center always stays inside the allowed vertical ribbon.
    freq1 = float(rng.uniform(0.70, 1.45))
    freq2 = float(rng.uniform(1.60, 2.60))
    phase1 = float(rng.uniform(0.0, 2.0 * math.pi))
    phase2 = float(rng.uniform(0.0, 2.0 * math.pi))
    mix = float(rng.uniform(0.18, 0.42))
    offset = float(rng.uniform(-0.18, 0.18))
    profile = (
        np.sin(2.0 * math.pi * freq1 * tau + phase1)
        + mix * np.sin(2.0 * math.pi * freq2 * tau + phase2)
        + offset
    ).astype(np.float32)
    profile -= float(np.mean(profile))
    denom = max(float(np.max(np.abs(profile))), 1e-6)
    amp = float(rng.uniform(0.45, 0.88)) * z_half_range
    u = (amp * profile / denom).astype(np.float32)
    u = np.clip(u, -0.98 * z_half_range, 0.98 * z_half_range).astype(np.float32)
    center[:, 2] += u

    z_axis = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
    y_axis = np.cross(z_axis, tang).astype(np.float32)
    y_axis /= np.maximum(np.linalg.norm(y_axis, axis=1, keepdims=True), 1e-8)
    z_axis = np.cross(tang, y_axis).astype(np.float32)
    z_axis /= np.maximum(np.linalg.norm(z_axis, axis=1, keepdims=True), 1e-8)

    offset = (0.5 * grasp_span * tang).astype(np.float32)
    p_left = (center - offset).astype(np.float32)
    p_right = (center + offset).astype(np.float32)
    rpy_left = np.zeros((n, 3), dtype=np.float32)
    rpy_right = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        r_left = np.stack([tang[i], y_axis[i], z_axis[i]], axis=1).astype(np.float64)
        # Right hand uses a fixed 180 degree transform relative to the object,
        # so the two tool x-axes point toward each other across the grasped bar.
        r_right = r_left @ np.diag([-1.0, 1.0, -1.0])
        rpy_left[i] = _rpy_from_rotmat_zyx(r_left)
        rpy_right[i] = _rpy_from_rotmat_zyx(r_right)
    pose_left = np.concatenate([p_left, rpy_left], axis=1).astype(np.float32)
    pose_right = np.concatenate([p_right, rpy_right], axis=1).astype(np.float32)
    return {
        "pose_left": pose_left,
        "pose_right": pose_right,
        "center": center.astype(np.float32),
        "p_left": p_left.astype(np.float32),
        "p_right": p_right.astype(np.float32),
    }


def _path_from_12d_pose_array(path12: np.ndarray) -> dict[str, np.ndarray]:
    path = np.asarray(path12, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] < 12:
        raise ValueError(f"expected planned path with shape (T,12), got {path.shape}")
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


def _load_planned_path_npz(path: str, key: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    selected = str(key)
    if selected not in data.files:
        if len(data.files) == 1:
            selected = data.files[0]
        else:
            raise KeyError(f"path key '{key}' not found in {path}; available keys={data.files}")
    return _path_from_12d_pose_array(np.asarray(data[selected], dtype=np.float32))


class DualUR5DemoSim:
    def __init__(
        self,
        *,
        gui: bool,
        urdf_path: str | None,
        ee_link_index: int | None,
        tool_axis: str | None,
        base_cfg: dict[str, list[float]],
        sim_dt: float,
    ) -> None:
        try:
            import pybullet as p  # type: ignore
            import pybullet_data  # type: ignore
        except Exception as e:
            raise RuntimeError(f"pybullet unavailable: {e}")
        self.p = p
        self.gui = bool(gui)
        self.client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(float(sim_dt), physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.sim_dt = float(sim_dt)
        self.trace_body_ids: list[int] = []
        self.scene_body_ids: list[int] = []
        self.virtual_link_body_ids: list[int] = []
        self.temp_mesh_paths: list[str] = []
        self.camera_distance = 1.96
        self.camera_yaw = 33.0
        self.camera_pitch = -50.0
        self.camera_target = np.asarray([0.02, -0.03, 0.73], dtype=np.float32)
        self.camera_fov = 52.0

        kin_cfg = resolve_ur5_kinematics_cfg(
            {"urdf_path": urdf_path, "ee_link_index": ee_link_index, "tool_axis": tool_axis}
        )
        load_path = str(kin_cfg["urdf_path"])
        self._patched_urdf: str | None = None
        txt = ""
        try:
            txt = open(load_path, "r", encoding="utf-8").read()
        except Exception:
            pass
        if "package://" in txt:
            self._patched_urdf = _make_pybullet_friendly_urdf(load_path)
            load_path = self._patched_urdf
        self.ee_link_index_override = kin_cfg.get("ee_link_index")
        self._build_scene(base_cfg)
        self.left = self._load_robot(
            load_path,
            base_xyz=base_cfg["dual_arm_left_base_xyz"],
            base_rpy=base_cfg["dual_arm_left_base_rpy"],
            side="left",
        )
        self.right = self._load_robot(
            load_path,
            base_xyz=base_cfg["dual_arm_right_base_xyz"],
            base_rpy=base_cfg["dual_arm_right_base_rpy"],
            side="right",
        )
        self._set_camera()
        self._disable_default_motors(self.left)
        self._disable_default_motors(self.right)

    def close(self) -> None:
        for bid in list(self.trace_body_ids) + list(self.scene_body_ids):
            try:
                self.p.removeBody(int(bid), physicsClientId=self.client_id)
            except Exception:
                pass
        for bid in list(self.virtual_link_body_ids):
            try:
                self.p.removeBody(int(bid), physicsClientId=self.client_id)
            except Exception:
                pass
        try:
            self.p.disconnect(physicsClientId=self.client_id)
        except Exception:
            pass
        if self._patched_urdf:
            try:
                os.remove(self._patched_urdf)
            except Exception:
                pass
        for mesh_path in list(self.temp_mesh_paths):
            try:
                os.remove(mesh_path)
            except Exception:
                pass

    def _set_camera(self) -> None:
        if self.gui:
            try:
                self.p.resetDebugVisualizerCamera(
                    cameraDistance=float(self.camera_distance),
                    cameraYaw=float(self.camera_yaw),
                    cameraPitch=float(self.camera_pitch),
                    cameraTargetPosition=[float(v) for v in self.camera_target],
                    physicsClientId=self.client_id,
                )
            except Exception:
                pass

    def _load_robot(self, urdf_path: str, *, base_xyz: list[float], base_rpy: list[float], side: str) -> RobotHandle:
        p = self.p
        rid = p.loadURDF(
            urdf_path,
            basePosition=[float(v) for v in base_xyz],
            baseOrientation=[float(v) for v in _rpy_to_quat_xyzw(np.asarray(base_rpy, dtype=np.float32))],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client_id,
        )
        arm: list[int] = []
        ik_joints: list[int] = []
        lo: list[float] = []
        hi: list[float] = []
        gripper: list[int] = []
        nj = p.getNumJoints(rid, physicsClientId=self.client_id)
        for j in range(nj):
            info = p.getJointInfo(rid, j, physicsClientId=self.client_id)
            jt = int(info[2])
            if jt != p.JOINT_FIXED:
                ik_joints.append(int(j))
            if jt == p.JOINT_REVOLUTE:
                if len(arm) < 6:
                    arm.append(int(j))
                    lj, hj = float(info[8]), float(info[9])
                    if (not np.isfinite(lj)) or (not np.isfinite(hj)) or hj <= lj:
                        lj, hj = -math.pi, math.pi
                    lo.append(lj)
                    hi.append(hj)
                else:
                    gripper.append(int(j))
            elif jt == p.JOINT_PRISMATIC:
                gripper.append(int(j))
        if len(arm) < 6:
            raise RuntimeError(f"UR5 model has fewer than 6 revolute arm joints: {len(arm)}")
        q_lo = np.asarray(lo[:6], dtype=np.float32)
        q_hi = np.asarray(hi[:6], dtype=np.float32)
        home = _preferred_home_q(side, q_lo, q_hi)
        ee_cfg = None if int(self.ee_link_index_override) < 0 else int(self.ee_link_index_override)
        ee_idx = int(ee_cfg) if ee_cfg is not None else pick_default_ee_link_index(rid, arm[-1], self.client_id)
        handle = RobotHandle(
            robot_id=int(rid),
            arm_joint_indices=arm[:6],
            ik_joint_indices=ik_joints,
            arm_ik_positions=[int(ik_joints.index(j)) for j in arm[:6]],
            ee_link_index=int(ee_idx),
            q_lo=q_lo,
            q_hi=q_hi,
            q_range=np.maximum(q_hi - q_lo, 1e-3).astype(np.float32),
            home_q=home,
            gripper_joint_indices=gripper,
        )
        self._set_gripper_closed(handle)
        return handle

    def _disable_default_motors(self, robot: RobotHandle) -> None:
        self.p.setJointMotorControlArray(
            robot.robot_id,
            robot.arm_joint_indices,
            controlMode=self.p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * len(robot.arm_joint_indices),
            forces=[0.0] * len(robot.arm_joint_indices),
            physicsClientId=self.client_id,
        )

    def _set_gripper_closed(self, robot: RobotHandle) -> None:
        p = self.p
        for j in robot.gripper_joint_indices:
            info = p.getJointInfo(robot.robot_id, j, physicsClientId=self.client_id)
            lo, hi = float(info[8]), float(info[9])
            if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
                q = 0.0
            else:
                q = lo + 0.78 * (hi - lo)
            p.resetJointState(robot.robot_id, int(j), float(q), targetVelocity=0.0, physicsClientId=self.client_id)
            try:
                p.setJointMotorControl2(
                    robot.robot_id,
                    int(j),
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=float(q),
                    force=25.0,
                    physicsClientId=self.client_id,
                )
            except Exception:
                pass

    def _reset_robot_q(self, robot: RobotHandle, q: np.ndarray) -> None:
        qv = np.clip(np.asarray(q, dtype=np.float32).reshape(6), robot.q_lo, robot.q_hi)
        for i, j in enumerate(robot.arm_joint_indices):
            self.p.resetJointState(
                robot.robot_id,
                int(j),
                targetValue=float(qv[i]),
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )

    def _get_q(self, robot: RobotHandle) -> np.ndarray:
        sts = self.p.getJointStates(robot.robot_id, robot.arm_joint_indices, physicsClientId=self.client_id)
        return np.asarray([float(s[0]) for s in sts], dtype=np.float32)

    def _get_ee_pose(self, robot: RobotHandle) -> tuple[np.ndarray, np.ndarray]:
        ls = self.p.getLinkState(
            robot.robot_id,
            robot.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        pos = np.asarray(ls[4], dtype=np.float32)
        quat = np.asarray(ls[5], dtype=np.float32)
        quat /= max(float(np.linalg.norm(quat)), 1e-8)
        return pos, quat

    def _build_scene(self, base_cfg: dict[str, list[float]]) -> None:
        # Use the high support boxes already included in the UR5 URDF; no extra table.
        return

    def _add_box(
        self,
        *,
        center: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        rgba: tuple[float, float, float, float],
    ) -> int:
        vis = self.p.createVisualShape(
            self.p.GEOM_BOX,
            halfExtents=[float(v) for v in half_extents],
            rgbaColor=[float(v) for v in rgba],
            specularColor=[0.03, 0.03, 0.03],
            physicsClientId=self.client_id,
        )
        bid = self.p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(vis),
            baseCollisionShapeIndex=-1,
            basePosition=[float(v) for v in center],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        self.scene_body_ids.append(int(bid))
        return int(bid)

    def _add_cylinder_segment(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        radius: float,
        rgba: tuple[float, float, float, float],
        body_list: list[int] | None = None,
    ) -> int | None:
        a3 = np.asarray(a, dtype=np.float32).reshape(3)
        b3 = np.asarray(b, dtype=np.float32).reshape(3)
        d = b3 - a3
        length = float(np.linalg.norm(d))
        if length <= 1e-6 or not np.isfinite(length):
            return None
        vis = self.p.createVisualShape(
            self.p.GEOM_CYLINDER,
            radius=float(radius),
            length=float(length),
            rgbaColor=[float(v) for v in rgba],
            specularColor=[0.04, 0.04, 0.04],
            physicsClientId=self.client_id,
        )
        bid = self.p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(vis),
            baseCollisionShapeIndex=-1,
            basePosition=[float(v) for v in (0.5 * (a3 + b3))],
            baseOrientation=[float(v) for v in _quat_from_z_axis(d)],
            physicsClientId=self.client_id,
        )
        if body_list is not None:
            body_list.append(int(bid))
        return int(bid)

    def _write_tube_mesh_obj(
        self,
        points: np.ndarray,
        *,
        radius: float,
        radial_segments: int = 10,
    ) -> str | None:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        finite = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite]
        if len(pts) < 2:
            return None

        keep = [0]
        for i in range(1, len(pts)):
            if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > 1e-5:
                keep.append(i)
        pts = pts[np.asarray(keep, dtype=np.int64)]
        if len(pts) < 2:
            return None

        radial_n = int(max(5, radial_segments))
        tangents = np.zeros_like(pts)
        tangents[0] = pts[1] - pts[0]
        tangents[-1] = pts[-1] - pts[-2]
        if len(pts) > 2:
            tangents[1:-1] = pts[2:] - pts[:-2]
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-8)

        normals = np.zeros_like(pts)
        ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(ref, tangents[0]))) > 0.92:
            ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        n0 = ref - tangents[0] * float(np.dot(ref, tangents[0]))
        n0 /= max(float(np.linalg.norm(n0)), 1e-8)
        normals[0] = n0
        for i in range(1, len(pts)):
            n = normals[i - 1] - tangents[i] * float(np.dot(normals[i - 1], tangents[i]))
            nn = float(np.linalg.norm(n))
            if nn <= 1e-6:
                ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
                if abs(float(np.dot(ref, tangents[i]))) > 0.92:
                    ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                n = ref - tangents[i] * float(np.dot(ref, tangents[i]))
                nn = float(np.linalg.norm(n))
            normals[i] = n / max(nn, 1e-8)

        binormals = np.cross(tangents, normals).astype(np.float32)
        binormals /= np.maximum(np.linalg.norm(binormals, axis=1, keepdims=True), 1e-8)

        verts: list[np.ndarray] = []
        for pnt, normal, binormal in zip(pts, normals, binormals):
            for j in range(radial_n):
                theta = 2.0 * math.pi * float(j) / float(radial_n)
                offset = math.cos(theta) * normal + math.sin(theta) * binormal
                verts.append(pnt + float(radius) * offset)

        tmp = tempfile.NamedTemporaryFile(prefix="dual_arm_guide_curve_", suffix=".obj", delete=False)
        mesh_path = tmp.name
        tmp.close()
        with open(mesh_path, "w", encoding="utf-8") as f:
            f.write("o guide_curve_tube\n")
            for v in verts:
                f.write(f"v {float(v[0]):.7f} {float(v[1]):.7f} {float(v[2]):.7f}\n")
            for i in range(len(pts) - 1):
                row0 = i * radial_n
                row1 = (i + 1) * radial_n
                for j in range(radial_n):
                    a = row0 + j + 1
                    b = row0 + ((j + 1) % radial_n) + 1
                    c = row1 + ((j + 1) % radial_n) + 1
                    d = row1 + j + 1
                    f.write(f"f {a} {b} {c}\n")
                    f.write(f"f {a} {c} {d}\n")
        self.temp_mesh_paths.append(mesh_path)
        return mesh_path

    def _add_tube_curve(
        self,
        points: np.ndarray,
        *,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> int | None:
        mesh_path = self._write_tube_mesh_obj(points, radius=radius, radial_segments=10)
        if mesh_path is None:
            return None
        vis = self.p.createVisualShape(
            self.p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[1.0, 1.0, 1.0],
            rgbaColor=[float(v) for v in rgba],
            specularColor=[0.04, 0.04, 0.04],
            physicsClientId=self.client_id,
        )
        bid = self.p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(vis),
            baseCollisionShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        return int(bid)

    def _write_vertical_ribbon_mesh_obj(
        self,
        cfg: Any,
        *,
        n_s: int = 96,
        n_u: int = 18,
    ) -> str:
        x_span = float(getattr(cfg, "dual_arm_curve_x_span", 0.45))
        y_amp = float(getattr(cfg, "dual_arm_curve_y_amp", 0.12))
        y_freq = float(getattr(cfg, "dual_arm_curve_y_freq", 1.0))
        z_base = float(getattr(cfg, "dual_arm_curve_z_base", 0.68))
        z_half_range = float(getattr(cfg, "dual_arm_vertical_half_range", 0.06))
        s_vals = np.linspace(-1.0, 1.0, num=int(max(2, n_s)), dtype=np.float32)
        u_vals = np.linspace(-z_half_range, z_half_range, num=int(max(2, n_u)), dtype=np.float32)

        verts: list[tuple[float, float, float]] = []
        for s in s_vals:
            x = x_span * float(s)
            y = y_amp * math.sin(y_freq * math.pi * float(s))
            for u in u_vals:
                verts.append((x, y, z_base + float(u)))

        tmp = tempfile.NamedTemporaryFile(prefix="dual_arm_vertical_ribbon_", suffix=".obj", delete=False)
        mesh_path = tmp.name
        tmp.close()
        with open(mesh_path, "w", encoding="utf-8") as f:
            f.write("o dual_arm_vertical_ribbon\n")
            for v in verts:
                f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
            nu = len(u_vals)
            for i in range(len(s_vals) - 1):
                for j in range(len(u_vals) - 1):
                    a = i * nu + j + 1
                    b = (i + 1) * nu + j + 1
                    c = (i + 1) * nu + (j + 1) + 1
                    d = i * nu + (j + 1) + 1
                    # Keep a single winding so alpha blending stays visually light.
                    f.write(f"f {a} {b} {c}\n")
                    f.write(f"f {a} {c} {d}\n")
        self.temp_mesh_paths.append(mesh_path)
        return mesh_path

    def _add_vertical_ribbon_surface(self, cfg: Any) -> int:
        mesh_path = self._write_vertical_ribbon_mesh_obj(cfg)
        vis = self.p.createVisualShape(
            self.p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[1.0, 1.0, 1.0],
            rgbaColor=[0.74, 0.81, 0.88, 0.14],
            specularColor=[0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )
        bid = self.p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(vis),
            baseCollisionShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        return int(bid)

    def build_guide_visuals(
        self,
        center: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        *,
        cfg: Any | None = None,
        draw_center_guide: bool = True,
    ) -> None:
        del left, right
        center = np.asarray(center, dtype=np.float32)
        if cfg is not None:
            self.scene_body_ids.append(self._add_vertical_ribbon_surface(cfg))
        if bool(draw_center_guide):
            guide_id = self._add_tube_curve(center, radius=0.003, rgba=(0.05, 0.18, 0.58, 0.88))
            if guide_id is not None:
                self.scene_body_ids.append(guide_id)

    def solve_ik_path(
        self,
        robot: RobotHandle,
        pose: np.ndarray,
        *,
        orientation_mode: str,
        task_offset_quat_xyzw: np.ndarray,
        side: str,
        max_iters: int,
        seed_offset: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        p = self.p
        poses = np.asarray(pose, dtype=np.float32)
        mode = str(orientation_mode).strip().lower()
        if mode not in ("none", "task"):
            raise ValueError(f"unknown orientation_mode '{orientation_mode}'")
        task_offset_quat = np.asarray(task_offset_quat_xyzw, dtype=np.float32).reshape(4)
        task_offset_quat /= max(float(np.linalg.norm(task_offset_quat)), 1e-8)
        q_prev = robot.home_q.copy()
        q_out = np.zeros((len(poses), 6), dtype=np.float32)
        pos_err = np.zeros((len(poses),), dtype=np.float32)
        ori_err = np.zeros((len(poses),), dtype=np.float32)
        joint_damping = [0.05] * len(robot.ik_joint_indices)
        for i in range(len(poses)):
            target_pos = poses[i, :3].astype(np.float32)
            rest_q = np.clip((0.50 * q_prev + 0.50 * robot.home_q), robot.q_lo, robot.q_hi).astype(np.float32)
            kwargs = dict(
                bodyUniqueId=robot.robot_id,
                endEffectorLinkIndex=robot.ee_link_index,
                targetPosition=[float(v) for v in target_pos],
                lowerLimits=[float(v) for v in robot.q_lo],
                upperLimits=[float(v) for v in robot.q_hi],
                jointRanges=[float(v) for v in robot.q_range],
                restPoses=[float(v) for v in rest_q],
                jointDamping=joint_damping,
                solver=p.IK_DLS,
                maxNumIterations=int(max(1, max_iters)),
                residualThreshold=1e-5,
                physicsClientId=self.client_id,
            )
            target_quat = None
            if mode == "task":
                target_quat = _rpy_to_quat_xyzw(poses[i, 3:6])
                target_quat = _quat_multiply_xyzw(target_quat, task_offset_quat)
                if str(side).lower().startswith("r"):
                    target_quat = _quat_multiply_xyzw(target_quat, _right_task_grasp_flip_xyzw())
                else:
                    target_quat = _quat_multiply_xyzw(target_quat, _left_task_grasp_flip_xyzw())
            if target_quat is not None:
                kwargs["targetOrientation"] = [float(v) for v in target_quat]
            sol = np.asarray(p.calculateInverseKinematics(**kwargs), dtype=np.float32).reshape(-1)
            if len(sol) < len(robot.ik_joint_indices):
                raise RuntimeError(f"IK returned {len(sol)} joints; expected at least {len(robot.ik_joint_indices)}")
            q = np.asarray([sol[j] for j in robot.arm_ik_positions], dtype=np.float32)
            q = np.clip(q, robot.q_lo, robot.q_hi).astype(np.float32)
            self._reset_robot_q(robot, q)
            ee_pos, ee_quat = self._get_ee_pose(robot)
            pos_err[i] = float(np.linalg.norm(target_pos - ee_pos))
            if target_quat is not None:
                dot = float(abs(np.dot(target_quat, ee_quat)))
                ori_err[i] = float(2.0 * math.acos(np.clip(dot, 0.0, 1.0)))
            q_out[i] = q
            q_prev = q.copy()
        return q_out, {
            "mean_pos_err": float(np.mean(pos_err)),
            "max_pos_err": float(np.max(pos_err)),
            "first_pos_err": float(pos_err[0]) if len(pos_err) else 0.0,
            "mean_ori_err_deg": float(np.degrees(np.mean(ori_err))) if mode != "none" else 0.0,
            "max_ori_err_deg": float(np.degrees(np.max(ori_err))) if mode != "none" else 0.0,
        }

    def time_parameterize(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        *,
        max_joint_speed: float,
        min_segment_time: float,
        terminal_hold_time: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ql = np.asarray(q_left, dtype=np.float32)
        qr = np.asarray(q_right, dtype=np.float32)
        seg_dt = [0.0]
        vmax = float(max(max_joint_speed, 1e-3))
        min_dt = float(max(min_segment_time, 1e-3))
        for i in range(1, len(ql)):
            dq = max(float(np.max(np.abs(ql[i] - ql[i - 1]))), float(np.max(np.abs(qr[i] - qr[i - 1]))))
            seg_dt.append(max(dq / vmax, min_dt))
        t_src = np.cumsum(np.asarray(seg_dt, dtype=np.float32))
        total_t = float(max(t_src[-1], min_dt))
        n_out = int(max(2, math.ceil(total_t / self.sim_dt))) + 1
        t = np.linspace(0.0, total_t, num=n_out, dtype=np.float32)
        out_l = np.zeros((len(t), 6), dtype=np.float32)
        out_r = np.zeros((len(t), 6), dtype=np.float32)
        for j in range(6):
            out_l[:, j] = np.interp(t, t_src, ql[:, j]).astype(np.float32)
            out_r[:, j] = np.interp(t, t_src, qr[:, j]).astype(np.float32)
        qd_l = np.gradient(out_l, self.sim_dt, axis=0).astype(np.float32)
        qd_r = np.gradient(out_r, self.sim_dt, axis=0).astype(np.float32)
        qd_l = np.clip(qd_l, -vmax, vmax).astype(np.float32)
        qd_r = np.clip(qd_r, -vmax, vmax).astype(np.float32)
        n_hold = int(max(0, round(float(terminal_hold_time) / self.sim_dt)))
        if n_hold > 0:
            t_hold = t[-1] + self.sim_dt * np.arange(1, n_hold + 1, dtype=np.float32)
            t = np.concatenate([t, t_hold]).astype(np.float32)
            out_l = np.concatenate([out_l, np.repeat(out_l[-1:], n_hold, axis=0)], axis=0).astype(np.float32)
            out_r = np.concatenate([out_r, np.repeat(out_r[-1:], n_hold, axis=0)], axis=0).astype(np.float32)
            qd_l = np.concatenate([qd_l, np.zeros((n_hold, 6), dtype=np.float32)], axis=0)
            qd_r = np.concatenate([qd_r, np.zeros((n_hold, 6), dtype=np.float32)], axis=0)
        return t, out_l, out_r, qd_l, qd_r

    def capture_frame(self, *, width: int, height: int) -> np.ndarray:
        view = self.p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[float(v) for v in self.camera_target],
            distance=float(self.camera_distance),
            yaw=float(self.camera_yaw),
            pitch=float(self.camera_pitch),
            roll=0.0,
            upAxisIndex=2,
        )
        proj = self.p.computeProjectionMatrixFOV(
            fov=float(self.camera_fov),
            aspect=float(width) / max(float(height), 1.0),
            nearVal=0.02,
            farVal=6.0,
        )
        renderer = self.p.ER_BULLET_HARDWARE_OPENGL if self.gui else self.p.ER_TINY_RENDERER
        _, _, rgba, _, _ = self.p.getCameraImage(
            width=int(width),
            height=int(height),
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=renderer,
            physicsClientId=self.client_id,
        )
        frame = np.asarray(rgba, dtype=np.uint8).reshape(int(height), int(width), 4)[..., :3].copy()
        bg_mask = (frame[..., 0] >= 245) & (frame[..., 1] >= 245) & (frame[..., 2] >= 245)
        if np.any(bg_mask):
            frame[bg_mask] = np.asarray([220, 228, 232], dtype=np.uint8)
        return frame

    def _set_position_control(self, robot: RobotHandle, q: np.ndarray, qd: np.ndarray, *, force: float) -> None:
        self.p.setJointMotorControlArray(
            robot.robot_id,
            robot.arm_joint_indices,
            controlMode=self.p.POSITION_CONTROL,
            targetPositions=[float(v) for v in q],
            targetVelocities=[float(v) for v in qd],
            positionGains=[0.34] * 6,
            velocityGains=[1.05] * 6,
            forces=[float(force)] * 6,
            physicsClientId=self.client_id,
        )

    def _reset_or_create_virtual_links(self, a: np.ndarray, b: np.ndarray) -> None:
        a3 = np.asarray(a, dtype=np.float32).reshape(3)
        b3 = np.asarray(b, dtype=np.float32).reshape(3)
        d = b3 - a3
        length = float(np.linalg.norm(d))
        if length <= 1e-6:
            return
        u = d / length
        mid = 0.5 * (a3 + b3)
        visual_extra = 0.065
        visual_length = float(length + 2.0 * visual_extra)
        if len(self.virtual_link_body_ids) == 0:
            link_vis = self.p.createVisualShape(
                self.p.GEOM_CYLINDER,
                radius=0.011,
                length=visual_length,
                rgbaColor=[0.95, 0.55, 0.10, 1.0],
                specularColor=[0.20, 0.14, 0.06],
                physicsClientId=self.client_id,
            )
            link_bid = int(
                self.p.createMultiBody(
                    baseMass=0.0,
                    baseVisualShapeIndex=int(link_vis),
                    baseCollisionShapeIndex=-1,
                    basePosition=[float(v) for v in mid],
                    baseOrientation=[float(v) for v in _quat_from_z_axis(u)],
                    physicsClientId=self.client_id,
                )
            )
            sphere_vis = self.p.createVisualShape(
                self.p.GEOM_SPHERE,
                radius=0.030,
                rgbaColor=[0.04, 0.16, 0.95, 1.0],
                specularColor=[0.12, 0.12, 0.22],
                physicsClientId=self.client_id,
            )
            sphere_bid = int(
                self.p.createMultiBody(
                    baseMass=0.0,
                    baseVisualShapeIndex=int(sphere_vis),
                    baseCollisionShapeIndex=-1,
                    basePosition=[float(v) for v in mid],
                    baseOrientation=[0.0, 0.0, 0.0, 1.0],
                    physicsClientId=self.client_id,
                )
            )
            self.virtual_link_body_ids = [link_bid, sphere_bid]
        else:
            self.p.resetBasePositionAndOrientation(
                int(self.virtual_link_body_ids[0]),
                [float(v) for v in mid],
                [float(v) for v in _quat_from_z_axis(u)],
                physicsClientId=self.client_id,
            )
            if len(self.virtual_link_body_ids) > 1:
                self.p.resetBasePositionAndOrientation(
                    int(self.virtual_link_body_ids[1]),
                    [float(v) for v in mid],
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=self.client_id,
                )

    def _reset_or_create_rod(self, a: np.ndarray, b: np.ndarray) -> None:
        self._reset_or_create_virtual_links(a, b)

    def track(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        qd_left: np.ndarray,
        qd_right: np.ndarray,
        *,
        video_path: str | None,
        video_width: int,
        video_height: int,
        video_fps: int,
        video_slowdown: float,
        realtime: bool,
        trace_stride: int,
        max_force: float,
        arm_trace_radius: float = 0.0055,
        draw_object_trace: bool = False,
        object_trace_radius: float = 0.003,
        snapshot_steps: list[int] | None = None,
        snapshot_dir: str | None = None,
        snapshot_crop_frac: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        ql = np.asarray(q_left, dtype=np.float32)
        qr = np.asarray(q_right, dtype=np.float32)
        qdl = np.asarray(qd_left, dtype=np.float32)
        qdr = np.asarray(qd_right, dtype=np.float32)
        self._reset_robot_q(self.left, ql[0])
        self._reset_robot_q(self.right, qr[0])
        self._set_position_control(self.left, ql[0], qdl[0], force=max_force)
        self._set_position_control(self.right, qr[0], qdr[0], force=max_force)
        for _ in range(32):
            self.p.stepSimulation(physicsClientId=self.client_id)
        pos_l0, _ = self._get_ee_pose(self.left)
        pos_r0, _ = self._get_ee_pose(self.right)
        self._reset_or_create_virtual_links(pos_l0, pos_r0)

        writer = None
        if video_path:
            os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
            out_fps = float(video_fps) / max(float(video_slowdown), 1e-3)
            writer = _FFmpegVideoWriter(out_path=video_path, width=int(video_width), height=int(video_height), fps=out_fps)
            writer.append_data(self.capture_frame(width=int(video_width), height=int(video_height)))

        n = len(ql)
        q_err_l = np.zeros((n,), dtype=np.float32)
        q_err_r = np.zeros((n,), dtype=np.float32)
        ee_l = np.zeros((n, 3), dtype=np.float32)
        ee_r = np.zeros((n, 3), dtype=np.float32)
        ee_l_rpy = np.zeros((n, 3), dtype=np.float32)
        ee_r_rpy = np.zeros((n, 3), dtype=np.float32)
        ee_l_quat = np.zeros((n, 4), dtype=np.float32)
        ee_r_quat = np.zeros((n, 4), dtype=np.float32)
        prev_l: np.ndarray | None = None
        prev_r: np.ndarray | None = None
        prev_c: np.ndarray | None = None
        trace_radius = (None if float(arm_trace_radius) <= 0.0 else float(max(0.0015, arm_trace_radius)))
        object_radius = float(max(0.0015, object_trace_radius))
        snapshot_step_set = {int(v) for v in (snapshot_steps or []) if int(v) >= 0}
        snapshot_dir_abs = None
        saved_snapshot_paths: list[str] = []
        if snapshot_step_set:
            snapshot_dir_abs = os.path.abspath(snapshot_dir or "snapshots")
            os.makedirs(snapshot_dir_abs, exist_ok=True)
        capture_every = max(1, int(round(1.0 / max(self.sim_dt * float(video_fps), 1e-8))))
        t0 = time.time()
        for i in range(n):
            self._set_position_control(self.left, ql[i], qdl[i], force=max_force)
            self._set_position_control(self.right, qr[i], qdr[i], force=max_force)
            self.p.stepSimulation(physicsClientId=self.client_id)
            if realtime:
                time.sleep(self.sim_dt)
            q_meas_l = self._get_q(self.left)
            q_meas_r = self._get_q(self.right)
            pos_l, quat_l = self._get_ee_pose(self.left)
            pos_r, quat_r = self._get_ee_pose(self.right)
            q_err_l[i] = float(np.linalg.norm(ql[i] - q_meas_l))
            q_err_r[i] = float(np.linalg.norm(qr[i] - q_meas_r))
            ee_l[i] = pos_l
            ee_r[i] = pos_r
            ee_l_rpy[i] = _quat_xyzw_to_rpy_zyx(self.p, quat_l)
            ee_r_rpy[i] = _quat_xyzw_to_rpy_zyx(self.p, quat_r)
            ee_l_quat[i] = quat_l.astype(np.float32)
            ee_r_quat[i] = quat_r.astype(np.float32)
            self._reset_or_create_virtual_links(pos_l, pos_r)
            if i % int(max(1, trace_stride)) == 0:
                center = (0.5 * (pos_l + pos_r)).astype(np.float32)
                if prev_l is not None:
                    if trace_radius is not None:
                        self._add_cylinder_segment(
                            prev_l,
                            pos_l,
                            radius=trace_radius,
                            rgba=(0.10, 0.62, 0.25, 0.94),
                            body_list=self.trace_body_ids,
                        )
                        self._add_cylinder_segment(
                            prev_r,
                            pos_r,
                            radius=trace_radius,
                            rgba=(0.85, 0.23, 0.18, 0.94),
                            body_list=self.trace_body_ids,
                        )
                    if bool(draw_object_trace) and prev_c is not None:
                        self._add_cylinder_segment(
                            prev_c,
                            center,
                            radius=object_radius,
                            rgba=(0.05, 0.18, 0.58, 0.88),
                            body_list=self.trace_body_ids,
                        )
                prev_l = pos_l.copy()
                prev_r = pos_r.copy()
                prev_c = center.copy()
            if snapshot_dir_abs is not None and i in snapshot_step_set:
                frame = self.capture_frame(width=int(video_width), height=int(video_height))
                frame = _crop_frame_by_frac(frame, snapshot_crop_frac)
                snap_path = os.path.join(snapshot_dir_abs, f"snapshot_step_{int(i):04d}.png")
                imageio.imwrite(snap_path, frame)
                saved_snapshot_paths.append(snap_path)
            if writer is not None and ((i + 1) % capture_every == 0 or i == n - 1):
                writer.append_data(self.capture_frame(width=int(video_width), height=int(video_height)))
        if writer is not None:
            writer.close()
        return {
            "wall_seconds": float(time.time() - t0),
            "mean_joint_err_left": float(np.mean(q_err_l)),
            "mean_joint_err_right": float(np.mean(q_err_r)),
            "max_joint_err_left": float(np.max(q_err_l)),
            "max_joint_err_right": float(np.max(q_err_r)),
            "ee_left": ee_l.astype(np.float32),
            "ee_right": ee_r.astype(np.float32),
            "ee_left_rpy": ee_l_rpy.astype(np.float32),
            "ee_right_rpy": ee_r_rpy.astype(np.float32),
            "ee_left_quat_xyzw": ee_l_quat.astype(np.float32),
            "ee_right_quat_xyzw": ee_r_quat.astype(np.float32),
            "snapshot_paths": saved_snapshot_paths,
        }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render a dual-UR5 12D guided-insertion demonstration.")
    ap.add_argument("--gui", type=int, default=1, help="0: no video, 1: DIRECT render video, 2: GUI render only, no video logging")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-config", type=str, default="configs/datasets/12d_dual_arm_traj.json")
    ap.add_argument("--outdir", type=str, default="outputs/bench/dual_arm_ur5_demo")
    ap.add_argument("--n-steps", type=int, default=96)
    ap.add_argument("--planned-path-npz", type=str, default="", help="Optional .npz from plan_dual_arm_from_learned_constraint.py.")
    ap.add_argument("--planned-path-key", type=str, default="path_0", help="Path array key inside --planned-path-npz.")
    ap.add_argument(
        "--orientation-mode",
        type=str,
        default="task",
        choices=("none", "task"),
        help="IK orientation target: none=position only, task=planned task frame with optional fixed local roll bias.",
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
    ap.add_argument("--use-orientation", type=int, default=-1, help="1 forces task orientation; 0 selects none unless --orientation-mode is supplied.")
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
    ap.add_argument("--trace-stride", type=int, default=18)
    ap.add_argument("--realtime", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    with open(args.dataset_config, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = SimpleNamespace(**cfg_dict, seed=int(args.seed))
    orientation_mode = str(args.orientation_mode).strip().lower()
    if int(args.use_orientation) == 1:
        orientation_mode = "task"
    elif int(args.use_orientation) == 0 and "--orientation-mode" not in sys.argv:
        orientation_mode = "none"
    task_offset_rpy_deg = _parse_rpy_deg_triplet(str(args.task_offset_rpy_deg))
    task_offset_quat = _compose_task_local_offset_quat_xyzw(
        task_roll_rad=np.deg2rad(float(args.task_roll_deg)),
        extra_rpy_rad=np.deg2rad(task_offset_rpy_deg.astype(np.float32)),
    )
    base_cfg = resolve_dual_ur5_base_cfg(cfg)
    if str(args.planned_path_npz).strip():
        path = _load_planned_path_npz(_resolve_path(str(args.planned_path_npz)), str(args.planned_path_key))
    else:
        path = _make_demo_path(cfg, n_steps=int(args.n_steps), seed=int(args.seed))
    os.makedirs(args.outdir, exist_ok=True)
    video_path = None
    if int(args.gui) == 1:
        video_path = os.path.join(args.outdir, "dual_arm_ur5_demo.mp4")

    sim = DualUR5DemoSim(
        gui=(int(args.gui) == 2),
        urdf_path=None,
        ee_link_index=None,
        tool_axis=None,
        base_cfg=base_cfg,
        sim_dt=float(args.sim_dt),
    )
    try:
        sim.build_guide_visuals(path["center"], path["p_left"], path["p_right"], cfg=cfg)
        q_left, ik_left = sim.solve_ik_path(
            sim.left,
            path["pose_left"],
            orientation_mode=orientation_mode,
            task_offset_quat_xyzw=task_offset_quat,
            side="left",
            max_iters=int(args.ik_iters),
            seed_offset=0,
        )
        q_right, ik_right = sim.solve_ik_path(
            sim.right,
            path["pose_right"],
            orientation_mode=orientation_mode,
            task_offset_quat_xyzw=task_offset_quat,
            side="right",
            max_iters=int(args.ik_iters),
            seed_offset=101,
        )
        t, ql_ref, qr_ref, qdl_ref, qdr_ref = sim.time_parameterize(
            q_left,
            q_right,
            max_joint_speed=float(args.max_joint_speed),
            min_segment_time=float(args.min_segment_time),
            terminal_hold_time=float(args.terminal_hold_time),
        )
        track = sim.track(
            ql_ref,
            qr_ref,
            qdl_ref,
            qdr_ref,
            video_path=video_path,
            video_width=int(args.video_width),
            video_height=int(args.video_height),
            video_fps=int(args.video_fps),
            video_slowdown=float(args.video_slowdown),
            realtime=bool(args.realtime),
            trace_stride=int(args.trace_stride),
            max_force=float(args.max_force),
        )
        summary = {
            "seed": int(args.seed),
            "orientation_mode": orientation_mode,
            "task_roll_deg": float(args.task_roll_deg),
            "task_offset_rpy_deg": [float(v) for v in task_offset_rpy_deg.tolist()],
            "use_orientation": bool(orientation_mode != "none"),
            "sim_dt": float(args.sim_dt),
            "n_task_waypoints": int(args.n_steps),
            "n_control_steps": int(len(t)),
            "duration_s": float(t[-1]) if len(t) else 0.0,
            "left_ik": ik_left,
            "right_ik": ik_right,
            "mean_joint_err_left": track["mean_joint_err_left"],
            "mean_joint_err_right": track["mean_joint_err_right"],
            "max_joint_err_left": track["max_joint_err_left"],
            "max_joint_err_right": track["max_joint_err_right"],
            "base_cfg": base_cfg,
            "video_path": video_path,
        }
        summary_path = os.path.join(args.outdir, "dual_arm_ur5_demo_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(
            "[summary] "
            f"left_ik={ik_left['mean_pos_err']:.4f}/{ik_left['max_pos_err']:.4f}, "
            f"right_ik={ik_right['mean_pos_err']:.4f}/{ik_right['max_pos_err']:.4f}, "
            f"joint_err={track['mean_joint_err_left']:.4f}/{track['mean_joint_err_right']:.4f}"
        )
        print(f"[saved] {summary_path}")
        if video_path:
            print(f"[saved] {video_path}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
