from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def quat_to_matrix_xyzw(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-15)


def ros_transform_to_matrix(transform) -> np.ndarray:
    t = transform.translation
    q = transform.rotation
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_to_matrix_xyzw((q.x, q.y, q.z, q.w))
    out[:3, 3] = (t.x, t.y, t.z)
    return out


def matrix_to_pose_row(stamp: float, pose: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
    qx, qy, qz, qw = matrix_to_quat_xyzw(pose[:3, :3])
    x, y, z = pose[:3, 3]
    return (float(stamp), float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw))
