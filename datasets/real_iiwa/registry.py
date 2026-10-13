from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_IIWA_TASKS_ROOT = REPO_ROOT / "datasets" / "real_iiwa" / "tasks"
REAL_IIWA_SURFACE_MODELS_ROOT = REPO_ROOT / "datasets" / "real_iiwa" / "surface_models"

PLANE_SURFACE_NORMAL = np.asarray(
    [0.0027779415162709365, 0.0074624800006471495, 0.9999682967140369],
    dtype=np.float64,
)
PLANE_SURFACE_NORMAL = PLANE_SURFACE_NORMAL / np.linalg.norm(PLANE_SURFACE_NORMAL)
PLANE_SURFACE_D = -0.16089767061815902



REAL_IIWA_DATASETS: dict[str, dict[str, Any]] = {
    "RealKuka": {
        "task_id": "RealKuka",
        "train_set": "human_demo_paper_toolz_arc3mm_cropped_v001",
        "description": "Real KUKA iiwa task with human_demo_paper thinned by 3 mm xyz arc length and cropped to remove enter/leave segments.",
        "true_codim": 4,
        "surface_type": "neural_height_field",
        "surface_model": "calibrated_tcp1845_neural_1d_surface_model",
        "coordinate_space": "surface_model_normalized",
        "orientation_condition": "recorded_human_tool_z_compared_to_minus_surface_normal",
        "target_tool_axis": "tcp_local_z",
        "target_tool_axis_sign": -1.0,
        "orientation_representation": "tool_z",
        "x_space": "surface_model_normalized_uvz_plus_tool_z",
    },
}



def canonical_real_iiwa_dataset_name(name: str) -> str | None:
    key = str(name)
    if key in REAL_IIWA_DATASETS:
        return key
    return None


def is_real_iiwa_dataset(name: str) -> bool:
    return canonical_real_iiwa_dataset_name(name) is not None


def plane_constraint() -> tuple[np.ndarray, float]:
    return PLANE_SURFACE_NORMAL.astype(np.float64, copy=True), float(PLANE_SURFACE_D)


def real_iiwa_surface_type(dataset_name: str) -> str:
    canonical = canonical_real_iiwa_dataset_name(dataset_name)
    if canonical is None:
        raise KeyError(f"unknown real iiwa dataset {dataset_name!r}")
    return str(REAL_IIWA_DATASETS[canonical].get("surface_type", "plane"))


def real_iiwa_coordinate_space(dataset_name: str) -> str:
    canonical = canonical_real_iiwa_dataset_name(dataset_name)
    if canonical is None:
        raise KeyError(f"unknown real iiwa dataset {dataset_name!r}")
    return str(REAL_IIWA_DATASETS[canonical].get("coordinate_space", "world"))


def real_iiwa_orientation_representation(dataset_name: str) -> str:
    canonical = canonical_real_iiwa_dataset_name(dataset_name)
    if canonical is None:
        raise KeyError(f"unknown real iiwa dataset {dataset_name!r}")
    return str(REAL_IIWA_DATASETS[canonical].get("orientation_representation", "none"))


def load_task_json(task_id: str) -> dict[str, Any]:
    path = REAL_IIWA_TASKS_ROOT / str(task_id) / "task.json"
    return json.loads(path.read_text())


def _find_train_set(task: dict[str, Any], train_set: str) -> dict[str, Any]:
    for entry in task.get("train_sets", []):
        if entry.get("name") == train_set or Path(str(entry.get("path", ""))).stem == train_set:
            return dict(entry)
    raise KeyError(f"train set {train_set!r} not found for task {task.get('task_id')!r}")


def load_real_iiwa_train_set(
    dataset_name: str,
    *,
    x_key: str = "x",
) -> dict[str, Any]:
    canonical = canonical_real_iiwa_dataset_name(dataset_name)
    if canonical is None:
        raise KeyError(f"unknown real iiwa dataset {dataset_name!r}")
    spec = dict(REAL_IIWA_DATASETS[canonical])
    task_id = str(spec["task_id"])
    task_root = REAL_IIWA_TASKS_ROOT / task_id
    task = load_task_json(task_id)
    train_entry = _find_train_set(task, str(spec["train_set"]))
    npz_path = task_root / str(train_entry["path"])
    with np.load(npz_path, allow_pickle=False) as data:
        x = np.asarray(data[x_key], dtype=np.float32)
        arrays = {k: data[k] for k in data.files}
    normal, d = plane_constraint()
    return {
        "name": canonical,
        "spec": spec,
        "task": task,
        "task_root": task_root,
        "train_entry": train_entry,
        "npz_path": npz_path,
        "x": x,
        "arrays": arrays,
        "plane_normal": normal,
        "plane_d": d,
    }


def _rpy_from_rotmat_zyx(rot: np.ndarray) -> np.ndarray:
    sy = -float(rot[2, 0])
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = float(np.arcsin(sy))
    cp = float(np.cos(pitch))
    if abs(cp) > 1e-8:
        roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-rot[0, 1], rot[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _quat_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _target_tangent_basis(target_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = np.asarray(target_z, dtype=np.float64)
    z_axis = z_axis / np.maximum(np.linalg.norm(z_axis), 1e-12)
    ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(ref @ z_axis)) > 0.95:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = ref - float(ref @ z_axis) * z_axis
    e1 = e1 / np.maximum(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(z_axis, e1)
    e2 = e2 / np.maximum(np.linalg.norm(e2), 1e-12)
    return e1, e2


def _spin_range_from_demo(loaded: dict[str, Any], target_z: np.ndarray) -> tuple[float, float, float]:
    arrays = loaded.get("arrays", {})
    quat = np.asarray(arrays.get("tcp_quat", np.zeros((0, 4))), dtype=np.float64)
    e1, e2 = _target_tangent_basis(target_z)
    angles: list[float] = []
    for q in quat:
        rot = _quat_to_matrix_xyzw(q)
        x_axis = rot[:, 0]
        x_proj = x_axis - float(x_axis @ target_z) * target_z
        norm = float(np.linalg.norm(x_proj))
        if norm < 1e-9:
            continue
        x_proj = x_proj / norm
        angles.append(float(np.arctan2(float(x_proj @ e2), float(x_proj @ e1))))
    if not angles:
        return -np.pi, np.pi, 0.0
    unwrapped = np.unwrap(np.asarray(angles, dtype=np.float64))
    center = float(np.median(unwrapped))
    lo = float(np.quantile(unwrapped, 0.02))
    hi = float(np.quantile(unwrapped, 0.98))
    pad = max(0.08, 0.20 * max(hi - lo, 1e-6))
    return lo - pad, hi + pad, center


def _wrap_rpy_near(rpy: np.ndarray, center: np.ndarray) -> np.ndarray:
    out = np.asarray(rpy, dtype=np.float64).copy()
    c = np.asarray(center, dtype=np.float64).reshape(3)
    out += 2.0 * np.pi * np.round((c - out) / (2.0 * np.pi))
    return out


def _rotmat_from_z_axis_canonical(target_z: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(target_z, dtype=np.float64)
    z_axis = z_axis / np.maximum(np.linalg.norm(z_axis, axis=1, keepdims=True), 1e-12)
    ref = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), (len(z_axis), 1))
    near_parallel = np.abs(np.sum(ref * z_axis, axis=1)) > 0.95
    ref[near_parallel] = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = ref - np.sum(ref * z_axis, axis=1, keepdims=True) * z_axis
    x_axis = x_axis / np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.maximum(np.linalg.norm(y_axis, axis=1, keepdims=True), 1e-12)
    return np.stack([x_axis, y_axis, z_axis], axis=2)


def _rpy_from_z_axis_canonical(target_z: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    rot = _rotmat_from_z_axis_canonical(target_z)
    rpy = np.stack([_rpy_from_rotmat_zyx(r) for r in rot], axis=0).astype(np.float64)
    if center is not None:
        rpy = np.stack([_wrap_rpy_near(v, center) for v in rpy], axis=0)
    return rpy.astype(np.float64)


class _Surface1DTiltMLP:
    """Lazy wrapper around the calibrated torch height-field model."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._torch = None
        self._model = None
        self._checkpoint = None

    def _load(self):
        if self._model is not None:
            return self._torch, self._model, self._checkpoint
        import torch
        from torch import nn

        class Surface1DTiltMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.curve = nn.Sequential(
                    nn.Linear(1, 32),
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1),
                )
                self.v_slope = nn.Parameter(torch.zeros(1))
                self.bias = nn.Parameter(torch.zeros(1))

            def forward(self, u, v):
                return self.curve(u) + self.v_slope * v + self.bias

        ckpt_path = self.model_dir / "calibrated_tcp1845_neural_1d_surface_mlp.pt"
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = Surface1DTiltMLP()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._torch = torch
        self._model = model
        self._checkpoint = checkpoint
        return torch, model, checkpoint

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch, model, checkpoint = self._load()
        x_arr = np.asarray(x, dtype=np.float32).reshape(-1)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        xy = torch.from_numpy(np.stack([x_arr, y_arr], axis=1).astype(np.float32))
        u_axis = torch.as_tensor(checkpoint["changing_axis_xy"], dtype=torch.float32)
        v_axis = torch.as_tensor(checkpoint["invariant_axis_xy"], dtype=torch.float32)
        u = ((xy @ u_axis[:, None]) - torch.as_tensor(checkpoint["u_mean"])) / torch.as_tensor(checkpoint["u_scale"])
        v = ((xy @ v_axis[:, None]) - torch.as_tensor(checkpoint["v_mean"])) / torch.as_tensor(checkpoint["v_scale"])
        u.requires_grad_(True)
        v.requires_grad_(True)
        normalized_z = model(u, v)
        grad_u, grad_v = torch.autograd.grad(normalized_z.sum(), (u, v))
        z_scale = torch.as_tensor(checkpoint["z_scale"])
        z = normalized_z * z_scale + torch.as_tensor(checkpoint["z_mean"])
        dz_du = grad_u * z_scale / torch.as_tensor(checkpoint["u_scale"])
        dz_dv = grad_v * z_scale / torch.as_tensor(checkpoint["v_scale"])
        gradient_xy = dz_du * u_axis.reshape(1, 2) + dz_dv * v_axis.reshape(1, 2)
        normal = torch.cat([-gradient_xy, torch.ones((len(x_arr), 1), dtype=torch.float32)], dim=1)
        normal = normal / torch.linalg.norm(normal, dim=1, keepdim=True).clamp_min(1e-12)
        return z.detach().cpu().numpy().reshape(-1).astype(np.float32), normal.detach().cpu().numpy().astype(np.float32)

    def evaluate_normalized_z(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        torch, model, _checkpoint = self._load()
        u_t = torch.from_numpy(np.asarray(u, dtype=np.float32).reshape(-1, 1))
        v_t = torch.from_numpy(np.asarray(v, dtype=np.float32).reshape(-1, 1))
        with torch.no_grad():
            z = model(u_t, v_t)
        return z.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def world_to_normalized(self, points: np.ndarray) -> np.ndarray:
        _torch, _model, checkpoint = self._load()
        pts = np.asarray(points, dtype=np.float32)
        u_axis = np.asarray(checkpoint["changing_axis_xy"], dtype=np.float32)
        v_axis = np.asarray(checkpoint["invariant_axis_xy"], dtype=np.float32)
        u = (pts[:, :2] @ u_axis.reshape(2, 1)).reshape(-1)
        v = (pts[:, :2] @ v_axis.reshape(2, 1)).reshape(-1)
        u_norm = (u - float(np.asarray(checkpoint["u_mean"]).reshape(-1)[0])) / float(np.asarray(checkpoint["u_scale"]).reshape(-1)[0])
        v_norm = (v - float(np.asarray(checkpoint["v_mean"]).reshape(-1)[0])) / float(np.asarray(checkpoint["v_scale"]).reshape(-1)[0])
        z_norm = (pts[:, 2] - float(np.asarray(checkpoint["z_mean"]).reshape(-1)[0])) / float(np.asarray(checkpoint["z_scale"]).reshape(-1)[0])
        return np.stack([u_norm, v_norm, z_norm], axis=1).astype(np.float32)

    def normalized_to_world(self, points: np.ndarray) -> np.ndarray:
        _torch, _model, checkpoint = self._load()
        pts = np.asarray(points, dtype=np.float32)
        u_axis = np.asarray(checkpoint["changing_axis_xy"], dtype=np.float32)
        v_axis = np.asarray(checkpoint["invariant_axis_xy"], dtype=np.float32)
        u = pts[:, 0] * float(np.asarray(checkpoint["u_scale"]).reshape(-1)[0]) + float(np.asarray(checkpoint["u_mean"]).reshape(-1)[0])
        v = pts[:, 1] * float(np.asarray(checkpoint["v_scale"]).reshape(-1)[0]) + float(np.asarray(checkpoint["v_mean"]).reshape(-1)[0])
        xy = u.reshape(-1, 1) * u_axis.reshape(1, 2) + v.reshape(-1, 1) * v_axis.reshape(1, 2)
        z = pts[:, 2] * float(np.asarray(checkpoint["z_scale"]).reshape(-1)[0]) + float(np.asarray(checkpoint["z_mean"]).reshape(-1)[0])
        return np.concatenate([xy, z.reshape(-1, 1)], axis=1).astype(np.float32)


_NEURAL_SURFACE_CACHE: dict[str, _Surface1DTiltMLP] = {}


def _neural_surface_model(dataset_name: str) -> _Surface1DTiltMLP:
    canonical = canonical_real_iiwa_dataset_name(dataset_name)
    if canonical is None:
        raise KeyError(f"unknown real iiwa dataset {dataset_name!r}")
    spec = REAL_IIWA_DATASETS[canonical]
    model_name = str(spec.get("surface_model", ""))
    if not model_name:
        raise ValueError(f"real iiwa dataset {dataset_name!r} has no neural surface model")
    if model_name not in _NEURAL_SURFACE_CACHE:
        _NEURAL_SURFACE_CACHE[model_name] = _Surface1DTiltMLP(REAL_IIWA_SURFACE_MODELS_ROOT / model_name)
    return _NEURAL_SURFACE_CACHE[model_name]


def real_iiwa_neural_surface_z_normal(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _neural_surface_model(dataset_name).evaluate(x, y)


def real_iiwa_world_to_normalized_xyz(dataset_name: str, points: np.ndarray) -> np.ndarray:
    return _neural_surface_model(dataset_name).world_to_normalized(points)


def real_iiwa_normalized_to_world_xyz(dataset_name: str, points: np.ndarray) -> np.ndarray:
    return _neural_surface_model(dataset_name).normalized_to_world(points)


def real_iiwa_surface_distance(
    dataset_name: str,
    points: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"real iiwa surface distance requires points with shape (N,>=3), got {pts.shape}")
    tool_z = pts[:, 3:6].astype(np.float32) if pts.shape[1] >= 6 else None
    if real_iiwa_coordinate_space(dataset_name) == "surface_model_normalized":
        pts_world = real_iiwa_normalized_to_world_xyz(dataset_name, pts[:, :3])
    else:
        pts_world = pts[:, :3]
    if real_iiwa_surface_type(dataset_name) == "neural_height_field":
        z_pred, normal = real_iiwa_neural_surface_z_normal(dataset_name, pts_world[:, 0], pts_world[:, 1])
        grad_norm = np.linalg.norm(normal[:, :2] / np.maximum(normal[:, 2:3], 1e-12), axis=1)
        pos_dist = (np.abs(pts_world[:, 2] - z_pred) / np.sqrt(1.0 + grad_norm**2)).astype(np.float32)
        if tool_z is not None and real_iiwa_orientation_representation(dataset_name) == "tool_z":
            tool_z = tool_z / np.maximum(np.linalg.norm(tool_z, axis=1, keepdims=True), 1e-12)
            sign = float(REAL_IIWA_DATASETS[canonical_real_iiwa_dataset_name(dataset_name)].get("target_tool_axis_sign", -1.0))
            target_tool_z = (sign * normal.astype(np.float32))
            axis_dist = np.linalg.norm(tool_z - target_tool_z, axis=1).astype(np.float32)
            return np.sqrt(pos_dist**2 + axis_dist**2).astype(np.float32)
        return pos_dist.astype(np.float32)
    normal, d = plane_constraint()
    pos_dist = np.abs(pts_world[:, :3].astype(np.float64) @ normal + float(d)).astype(np.float32)
    if tool_z is not None and real_iiwa_orientation_representation(dataset_name) == "tool_z":
        tool_z = tool_z / np.maximum(np.linalg.norm(tool_z, axis=1, keepdims=True), 1e-12)
        sign = float(REAL_IIWA_DATASETS[canonical_real_iiwa_dataset_name(dataset_name)].get("target_tool_axis_sign", -1.0))
        target_tool_z = np.tile((sign * normal).reshape(1, 3), (len(tool_z), 1)).astype(np.float32)
        axis_dist = np.linalg.norm(tool_z - target_tool_z, axis=1).astype(np.float32)
        return np.sqrt(pos_dist**2 + axis_dist**2).astype(np.float32)
    return pos_dist.astype(np.float32)


def real_iiwa_plane_grid(
    dataset_name: str,
    cfg: Any,
    *,
    x_train: np.ndarray | None = None,
) -> np.ndarray:
    loaded = load_real_iiwa_train_set(dataset_name)
    normal = np.asarray(loaded["plane_normal"], dtype=np.float64)
    d = float(loaded["plane_d"])
    train = np.asarray(x_train if x_train is not None else loaded["x"], dtype=np.float64)
    if train.ndim != 2 or train.shape[1] < 3:
        raise ValueError(f"real iiwa plane grid requires 3D position data, got shape {train.shape}")

    n_grid = int(getattr(cfg, "eval_gt_n_grid", getattr(cfg, "n_grid", 4096)))
    n_side = max(8, int(np.ceil(np.sqrt(max(1, n_grid)))))
    pad_ratio = float(getattr(cfg, "real_iiwa_plane_grid_pad_ratio", 0.20))
    xy = train[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - pad_ratio * span
    hi = hi + pad_ratio * span
    xs = np.linspace(float(lo[0]), float(hi[0]), n_side, dtype=np.float64)
    ys = np.linspace(float(lo[1]), float(hi[1]), n_side, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    if abs(float(normal[2])) < 1e-9:
        raise ValueError("real iiwa plane normal has near-zero z component; cannot build z=f(x,y) grid")
    zz = -(normal[0] * xx + normal[1] * yy + d) / normal[2]
    grid = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1)
    if grid.shape[0] > n_grid:
        grid = grid[:n_grid]
    if train.shape[1] >= 6:
        target_z = -normal
        e1, e2 = _target_tangent_basis(target_z)
        spin_lo, spin_hi, _ = _spin_range_from_demo(loaded, target_z)
        n_spin = max(4, int(getattr(cfg, "real_iiwa_pose_grid_n_spin", 12)))
        n_xy = max(4, int(np.ceil(np.sqrt(max(1, int(np.ceil(n_grid / n_spin)))))))
        xs = np.linspace(float(lo[0]), float(hi[0]), n_xy, dtype=np.float64)
        ys = np.linspace(float(lo[1]), float(hi[1]), n_xy, dtype=np.float64)
        spins = np.linspace(float(spin_lo), float(spin_hi), n_spin, dtype=np.float64)
        rpy_center = np.median(train[:, 3:6], axis=0) if train.shape[1] >= 6 else np.zeros(3, dtype=np.float64)
        rows: list[np.ndarray] = []
        for x_val in xs:
            for y_val in ys:
                z_val = -(normal[0] * x_val + normal[1] * y_val + d) / normal[2]
                for theta in spins:
                    x_axis = np.cos(theta) * e1 + np.sin(theta) * e2
                    x_axis = x_axis / np.maximum(np.linalg.norm(x_axis), 1e-12)
                    y_axis = np.cross(target_z, x_axis)
                    y_axis = y_axis / np.maximum(np.linalg.norm(y_axis), 1e-12)
                    rot = np.stack([x_axis, y_axis, target_z], axis=1)
                    rpy = _wrap_rpy_near(_rpy_from_rotmat_zyx(rot), rpy_center)
                    rows.append(np.concatenate([np.asarray([x_val, y_val, z_val], dtype=np.float64), rpy], axis=0))
        grid6 = np.stack(rows, axis=0)
        if grid6.shape[0] > n_grid:
            grid6 = grid6[:n_grid]
        return grid6.astype(np.float32)
    return grid.astype(np.float32)


def real_iiwa_neural_surface_grid(
    dataset_name: str,
    cfg: Any,
    *,
    x_train: np.ndarray | None = None,
) -> np.ndarray:
    loaded = load_real_iiwa_train_set(dataset_name)
    train = np.asarray(x_train if x_train is not None else loaded["x"], dtype=np.float64)
    if train.ndim != 2 or train.shape[1] < 3:
        raise ValueError(f"real iiwa neural surface grid requires 3D position data, got shape {train.shape}")

    n_grid = int(getattr(cfg, "eval_gt_n_grid", getattr(cfg, "n_grid", 4096)))
    n_side = max(8, int(np.ceil(np.sqrt(max(1, n_grid)))))
    pad_ratio = float(getattr(cfg, "real_iiwa_surface_grid_pad_ratio", getattr(cfg, "real_iiwa_plane_grid_pad_ratio", 0.10)))
    if real_iiwa_coordinate_space(dataset_name) == "surface_model_normalized":
        uv = train[:, :2]
        lo = uv.min(axis=0)
        hi = uv.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        lo = lo - pad_ratio * span
        hi = hi + pad_ratio * span
        us = np.linspace(float(lo[0]), float(hi[0]), n_side, dtype=np.float64)
        vs = np.linspace(float(lo[1]), float(hi[1]), n_side, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs, indexing="xy")
        z_norm = _neural_surface_model(dataset_name).evaluate_normalized_z(uu.reshape(-1), vv.reshape(-1))
        grid = np.stack([uu.reshape(-1), vv.reshape(-1), z_norm.astype(np.float64)], axis=1)
        if grid.shape[0] > n_grid:
            grid = grid[:n_grid]
        if train.shape[1] >= 6:
            world = real_iiwa_normalized_to_world_xyz(dataset_name, grid.astype(np.float32)).astype(np.float64)
            _z_world, normal = real_iiwa_neural_surface_z_normal(dataset_name, world[:, 0], world[:, 1])
            sign = float(loaded["spec"].get("target_tool_axis_sign", -1.0))
            target_z = sign * normal.astype(np.float64)
            if str(loaded["spec"].get("orientation_representation", "none")) == "tool_z":
                target_z = target_z / np.maximum(np.linalg.norm(target_z, axis=1, keepdims=True), 1e-12)
                return np.concatenate([grid.astype(np.float64), target_z], axis=1).astype(np.float32)
            rpy_center = np.median(train[:, 3:6], axis=0)
            rpy = _rpy_from_z_axis_canonical(target_z, center=rpy_center)
            return np.concatenate([grid.astype(np.float64), rpy], axis=1).astype(np.float32)
        return grid.astype(np.float32)

    xy = train[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - pad_ratio * span
    hi = hi + pad_ratio * span

    model_dir = REAL_IIWA_SURFACE_MODELS_ROOT / str(loaded["spec"].get("surface_model"))
    json_path = model_dir / "calibrated_tcp1845_neural_1d_surface.json"
    if json_path.exists():
        meta = json.loads(json_path.read_text())
        bounds = meta.get("xy_bounds", {})
        if "min" in bounds and "max" in bounds:
            bmin = np.asarray(bounds["min"], dtype=np.float64)
            bmax = np.asarray(bounds["max"], dtype=np.float64)
            lo = np.maximum(lo, bmin)
            hi = np.minimum(hi, bmax)

    xs = np.linspace(float(lo[0]), float(hi[0]), n_side, dtype=np.float64)
    ys = np.linspace(float(lo[1]), float(hi[1]), n_side, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    z, _normal = real_iiwa_neural_surface_z_normal(dataset_name, xx.reshape(-1), yy.reshape(-1))
    grid = np.stack([xx.reshape(-1), yy.reshape(-1), z.astype(np.float64)], axis=1)
    if grid.shape[0] > n_grid:
        grid = grid[:n_grid]
    if train.shape[1] >= 6:
        _z_world, normal = real_iiwa_neural_surface_z_normal(dataset_name, grid[:, 0], grid[:, 1])
        sign = float(loaded["spec"].get("target_tool_axis_sign", -1.0))
        target_z = sign * normal.astype(np.float64)
        if str(loaded["spec"].get("orientation_representation", "none")) == "tool_z":
            target_z = target_z / np.maximum(np.linalg.norm(target_z, axis=1, keepdims=True), 1e-12)
            return np.concatenate([grid.astype(np.float64), target_z], axis=1).astype(np.float32)
        rpy_center = np.median(train[:, 3:6], axis=0)
        rpy = _rpy_from_z_axis_canonical(target_z, center=rpy_center)
        return np.concatenate([grid.astype(np.float64), rpy], axis=1).astype(np.float32)
    return grid.astype(np.float32)


def real_iiwa_surface_grid(
    dataset_name: str,
    cfg: Any,
    *,
    x_train: np.ndarray | None = None,
) -> np.ndarray:
    if real_iiwa_surface_type(dataset_name) == "neural_height_field":
        return real_iiwa_neural_surface_grid(dataset_name, cfg, x_train=x_train)
    return real_iiwa_plane_grid(dataset_name, cfg, x_train=x_train)
