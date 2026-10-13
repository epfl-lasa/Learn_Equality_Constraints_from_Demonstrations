from __future__ import annotations

import numpy as np

from datasets.ur5_pybullet_utils import UR5_LINK_LENGTHS
from datasets.ur5_n6_dataset import ur5_fk_chain_n6, ur5_tool_axis_n6


def is_arm_dataset(name: str) -> bool:
    return "arm_" in str(name)


def is_workspace_pose_dataset(name: str) -> bool:
    return str(name) in (
        "6d_surf_normal",
        "6d_surf_normal_traj",
        "12d_dual_arm",
        "12d_dual_arm_traj",
    )


def wrap_np_pi(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def wrap_workspace_pose_rpy_np(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    if y.ndim == 2 and y.shape[1] >= 6:
        for base in range(0, y.shape[1], 6):
            if base + 6 <= y.shape[1]:
                y[:, base + 3 : base + 6] = wrap_np_pi(y[:, base + 3 : base + 6])
    return y


def _rpy_zyx_to_local_z_np(rpy: np.ndarray) -> np.ndarray:
    roll = rpy[:, 0]
    pitch = rpy[:, 1]
    yaw = rpy[:, 2]
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    z_x = cy * sp * cr + sy * sr
    z_y = sy * sp * cr - cy * sr
    z_z = cp * cr
    z = np.stack([z_x, z_y, z_z], axis=1).astype(np.float32)
    z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return z


def planar_fk(q: np.ndarray, lengths: list[float]) -> np.ndarray:
    t, j = q.shape
    joints = np.zeros((t, j + 1, 2), dtype=np.float32)
    ang = np.cumsum(q, axis=1)
    for k in range(j):
        joints[:, k + 1, 0] = joints[:, k, 0] + float(lengths[k]) * np.cos(ang[:, k])
        joints[:, k + 1, 1] = joints[:, k, 1] + float(lengths[k]) * np.sin(ang[:, k])
    return joints


def spatial_fk_n3(q: np.ndarray, lengths: list[float]) -> np.ndarray:
    t = q.shape[0]
    l1, l2 = float(lengths[0]), float(lengths[1])
    q1 = q[:, 0]
    t2 = q[:, 1]
    t3 = q[:, 1] + q[:, 2]
    r1 = l1 * np.cos(t2)
    z1 = l1 * np.sin(t2)
    r2 = r1 + l2 * np.cos(t3)
    z2 = z1 + l2 * np.sin(t3)
    joints = np.zeros((t, 3, 3), dtype=np.float32)
    joints[:, 1, 0] = r1 * np.cos(q1)
    joints[:, 1, 1] = r1 * np.sin(q1)
    joints[:, 1, 2] = z1
    joints[:, 2, 0] = r2 * np.cos(q1)
    joints[:, 2, 1] = r2 * np.sin(q1)
    joints[:, 2, 2] = z2
    return joints


def spatial_fk_n4(q: np.ndarray, lengths: list[float]) -> np.ndarray:
    t = q.shape[0]
    l1, l2, l3 = float(lengths[0]), float(lengths[1]), float(lengths[2])
    q1 = q[:, 0]
    t2 = q[:, 1]
    t3 = q[:, 1] + q[:, 2]
    t4 = q[:, 1] + q[:, 2] + q[:, 3]
    r1 = l1 * np.cos(t2)
    z1 = l1 * np.sin(t2)
    r2 = r1 + l2 * np.cos(t3)
    z2 = z1 + l2 * np.sin(t3)
    r3 = r2 + l3 * np.cos(t4)
    z3 = z2 + l3 * np.sin(t4)
    joints = np.zeros((t, 4, 3), dtype=np.float32)
    joints[:, 1, 0] = r1 * np.cos(q1)
    joints[:, 1, 1] = r1 * np.sin(q1)
    joints[:, 1, 2] = z1
    joints[:, 2, 0] = r2 * np.cos(q1)
    joints[:, 2, 1] = r2 * np.sin(q1)
    joints[:, 2, 2] = z2
    joints[:, 3, 0] = r3 * np.cos(q1)
    joints[:, 3, 1] = r3 * np.sin(q1)
    joints[:, 3, 2] = z3
    return joints


def spatial_fk_n6(q: np.ndarray, use_pybullet: bool = True) -> np.ndarray:
    return ur5_fk_chain_n6(q.astype(np.float32), use_pybullet=use_pybullet)


def spatial_tool_axis_n6(q: np.ndarray, use_pybullet: bool = True) -> np.ndarray:
    return ur5_tool_axis_n6(q.astype(np.float32), use_pybullet=use_pybullet)


def spatial_fk(q: np.ndarray, lengths: list[float], use_pybullet_n6: bool = True) -> np.ndarray:
    if q.shape[1] == 3:
        return spatial_fk_n3(q, lengths)
    if q.shape[1] == 4:
        return spatial_fk_n4(q, lengths)
    if q.shape[1] == 6:
        _ = lengths
        return spatial_fk_n6(q, use_pybullet=use_pybullet_n6)
    raise ValueError(f"unsupported spatial dof={q.shape[1]}")


def workspace_embed_for_eval(
    name: str,
    q: np.ndarray,
    *,
    ur5_use_pybullet_n6: bool = True,
) -> np.ndarray:
    q = q.astype(np.float32)
    base_name = str(name[:-5] if str(name).endswith("_traj") else name)
    if name == "2d_planar_arm_line_n2":
        return planar_fk(q, [1.0, 0.8])[:, -1, :].astype(np.float32)
    if name == "3d_planar_arm_line_n3":
        return planar_fk(q, [1.0, 0.8, 0.6])[:, -1, :].astype(np.float32)
    if base_name in ("3d_spatial_arm_plane_n3", "3d_spatial_arm_ellip_n3", "3d_spatial_arm_circle_n3"):
        return spatial_fk_n3(q, [1.0, 0.8])[:, -1, :].astype(np.float32)
    if base_name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
        # Arm-up datasets are orientation-constrained; evaluate in orientation space only.
        tool = spatial_tool_axis_n6(q, use_pybullet=ur5_use_pybullet_n6)
        return tool.astype(np.float32)
    if base_name == "12d_dual_arm" and q.shape[1] >= 12:
        pose_1 = q[:, :6].astype(np.float32)
        pose_2 = q[:, 6:12].astype(np.float32)

        def _pose_embed(p: np.ndarray) -> np.ndarray:
            pos = p[:, :3].astype(np.float32)
            rpy = p[:, 3:6].astype(np.float32)
            return np.concatenate([pos, np.cos(rpy), np.sin(rpy)], axis=1).astype(np.float32)

        return np.concatenate([_pose_embed(pose_1), _pose_embed(pose_2)], axis=1).astype(np.float32)
    if is_workspace_pose_dataset(name) and q.shape[1] >= 6:
        pos = q[:, :3].astype(np.float32)
        z_axis = _rpy_zyx_to_local_z_np(q[:, 3:6].astype(np.float32))
        return np.concatenate([pos, z_axis], axis=1).astype(np.float32)
    return q.astype(np.float32)
