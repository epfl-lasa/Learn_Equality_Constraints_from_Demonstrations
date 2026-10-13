from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def mixed_unit_angle_dims(dataset_name: str | None, dim: int) -> tuple[int, ...]:
    name = str(dataset_name or "")
    if name in {
        "6d_workspace_sine_surface_pose",
        "6d_workspace_sine_surface_pose_traj",
        "6d_workspace_sine_surface_pose_traj_turn",
        "6d_surf_normal",
        "6d_surf_normal_traj",
    } and dim >= 6:
        return (3, 4, 5)
    if name in {"12d_dual_arm", "12d_dual_arm_traj"} and dim >= 12:
        return (3, 4, 5, 9, 10, 11)
    return ()


@dataclass(frozen=True)
class FeatureNormalizer:
    center: np.ndarray
    scale: np.ndarray
    angle_dims: tuple[int, ...] = ()

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        *,
        dataset_name: str | None = None,
        enable: bool = True,
    ) -> "FeatureNormalizer":
        xx = np.asarray(x, dtype=np.float32)
        dim = int(xx.shape[1])
        angle_dims = mixed_unit_angle_dims(dataset_name, dim) if enable else ()
        center = np.mean(xx, axis=0).astype(np.float32)
        scale = np.ones((dim,), dtype=np.float32)
        if not angle_dims:
            return cls(center=np.zeros((dim,), dtype=np.float32), scale=scale, angle_dims=())

        for j in range(dim):
            vals = xx[:, j].astype(np.float32)
            if j in angle_dims:
                vals_w = _wrap_pi(vals)
                center[j] = np.float32(
                    np.arctan2(float(np.mean(np.sin(vals_w))), float(np.mean(np.cos(vals_w))))
                )
                dev = np.abs(_wrap_pi(vals_w - center[j]))
                span = 2.0 * float(np.quantile(dev, 0.95))
                if span < 1e-6:
                    span = 2.0 * float(np.max(dev))
            else:
                lo, hi = np.quantile(vals, [0.05, 0.95])
                span = float(hi - lo)
                if span < 1e-6:
                    span = float(np.max(vals) - np.min(vals))
            scale[j] = np.float32(max(span, 1e-3))
        return cls(center=center.astype(np.float32), scale=scale.astype(np.float32), angle_dims=tuple(angle_dims))

    @property
    def enabled(self) -> bool:
        return bool(self.angle_dims)

    def transform(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float32)
        out = xx - self.center.reshape(1, -1)
        for j in self.angle_dims:
            out[:, int(j)] = _wrap_pi(out[:, int(j)])
        return (out / self.scale.reshape(1, -1)).astype(np.float32)

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        yy = np.asarray(x_norm, dtype=np.float32)
        out = yy * self.scale.reshape(1, -1) + self.center.reshape(1, -1)
        for j in self.angle_dims:
            out[:, int(j)] = _wrap_pi(out[:, int(j)])
        return out.astype(np.float32)


class FeatureNormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, normalizer: FeatureNormalizer) -> None:
        super().__init__()
        self.model = model
        self.normalizer = normalizer
        self.register_buffer("feature_center", torch.from_numpy(normalizer.center.astype(np.float32)))
        self.register_buffer("feature_scale", torch.from_numpy(normalizer.scale.astype(np.float32)))
        self.angle_dims = tuple(int(v) for v in normalizer.angle_dims)

    def _transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        z = x - self.feature_center.to(device=x.device, dtype=x.dtype).view(1, -1)
        if self.angle_dims:
            z = z.clone()
            for j in self.angle_dims:
                if 0 <= j < int(z.shape[1]):
                    z[:, j] = torch.remainder(z[:, j] + torch.pi, 2.0 * torch.pi) - torch.pi
        return z / self.feature_scale.to(device=x.device, dtype=x.dtype).clamp_min(1e-6).view(1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self._transform_tensor(x))
