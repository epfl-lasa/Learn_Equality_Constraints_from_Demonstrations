#!/usr/bin/env python3
"""Evaluate the calibrated ball-center surface at world-frame XY."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Surface1DTiltMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.curve = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1))
        self.v_slope = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, v):
        return self.curve(u) + self.v_slope * v + self.bias


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Surface1DTiltMLP()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def evaluate(model, checkpoint, x, y):
    xy = torch.tensor([[x, y]], dtype=torch.float32)
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
    gradient_xy = dz_du * u_axis + dz_dv * v_axis
    upward_normal = torch.cat((-gradient_xy.reshape(-1), torch.ones(1)))
    upward_normal /= torch.linalg.norm(upward_normal)
    return {
        "frame": "world",
        "x_m": float(x), "y_m": float(y), "z_m": float(z.item()),
        "gradient_dz_dx_dz_dy": [float(value) for value in gradient_xy.reshape(-1)],
        "upward_normal": [float(value) for value in upward_normal],
        "tool_z_for_surface_normal_mode": [float(-value) for value in upward_normal],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).with_name(
                            "calibrated_tcp1845_neural_1d_surface_mlp.pt"))
    args = parser.parse_args()
    model, checkpoint = load_model(args.model)
    print(json.dumps(evaluate(model, checkpoint, args.x, args.y), indent=2))


if __name__ == "__main__":
    main()
