from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, out_dim: int = 1) -> None:
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.SiLU())
            dim = hidden
        layers.append(nn.Linear(dim, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NormalizedMLP(nn.Module):
    """MLP with input centering, scaling and optional angle wrapping.

    Checkpoint loaders use this class for model_type="normalized_mlp".
    DataAug training wraps a base MLP with FeatureNormalizedModel.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        depth: int,
        out_dim: int = 1,
        center: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        angle_dims: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if center is None:
            center = torch.zeros((in_dim,), dtype=torch.float32)
        if scale is None:
            scale = torch.ones((in_dim,), dtype=torch.float32)
        self.register_buffer("input_center", center.detach().clone().float().view(in_dim))
        self.register_buffer("input_scale", scale.detach().clone().float().view(in_dim).clamp_min(1e-6))
        self.angle_dims = tuple(int(v) for v in angle_dims)
        self.net = MLP(in_dim=in_dim, hidden=hidden, depth=depth, out_dim=out_dim)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        z = x - self.input_center.to(device=x.device, dtype=x.dtype)
        if self.angle_dims:
            z = z.clone()
            for j in self.angle_dims:
                if 0 <= j < z.shape[1]:
                    z[:, j] = torch.remainder(z[:, j] + torch.pi, 2.0 * torch.pi) - torch.pi
        return z / self.input_scale.to(device=x.device, dtype=x.dtype).clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._normalize(x))
