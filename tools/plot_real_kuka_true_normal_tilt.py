#!/usr/bin/env python3
"""Plot true-surface normal tilt over the human demonstration region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_NPZ = (
    PROJECT_ROOT
    / "datasets/real_iiwa/tasks/RealKuka/processed/train_sets/"
    / "human_demo_paper_toolz_arc3mm_cropped_v001.npz"
)
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "datasets/real_iiwa/surface_models/calibrated_tcp1845_neural_1d_surface_model/"
    / "calibrated_tcp1845_neural_1d_surface_mlp.pt"
)
DEFAULT_OUTDIR = PROJECT_ROOT / "datasets/real_iiwa/tasks/RealKuka/figures"


class Surface1DTiltMLP(nn.Module):
    def __init__(self) -> None:
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

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.curve(u) + self.v_slope * v + self.bias


def load_surface_model(path: Path) -> tuple[Surface1DTiltMLP, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Surface1DTiltMLP()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def true_surface_normals(model: Surface1DTiltMLP, checkpoint: dict, xy_world: np.ndarray) -> np.ndarray:
    xy = torch.as_tensor(xy_world, dtype=torch.float32)
    u_axis = torch.as_tensor(checkpoint["changing_axis_xy"], dtype=torch.float32)
    v_axis = torch.as_tensor(checkpoint["invariant_axis_xy"], dtype=torch.float32)
    u = ((xy @ u_axis[:, None]) - torch.as_tensor(checkpoint["u_mean"])) / torch.as_tensor(
        checkpoint["u_scale"]
    )
    v = ((xy @ v_axis[:, None]) - torch.as_tensor(checkpoint["v_mean"])) / torch.as_tensor(
        checkpoint["v_scale"]
    )
    u.requires_grad_(True)
    v.requires_grad_(True)
    normalized_z = model(u, v)
    grad_u, grad_v = torch.autograd.grad(normalized_z.sum(), (u, v))
    z_scale = torch.as_tensor(checkpoint["z_scale"])
    dz_du = grad_u * z_scale / torch.as_tensor(checkpoint["u_scale"])
    dz_dv = grad_v * z_scale / torch.as_tensor(checkpoint["v_scale"])
    gradient_xy = dz_du * u_axis.reshape(1, 2) + dz_dv * v_axis.reshape(1, 2)
    normal = torch.cat((-gradient_xy, torch.ones((len(xy), 1), dtype=torch.float32)), dim=1)
    normal = normal / torch.linalg.norm(normal, dim=1, keepdim=True).clamp_min(1e-9)
    return normal.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-npz", type=Path, default=DEFAULT_DEMO_NPZ)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    data = np.load(args.demo_npz)
    xy = np.asarray(data["position_world"][:, :2], dtype=np.float32)
    stored_normal = np.asarray(data["surface_normal"], dtype=np.float32)
    model, checkpoint = load_surface_model(args.model)
    normal = true_surface_normals(model, checkpoint, xy)

    tilt_deg = np.degrees(np.arctan2(np.linalg.norm(normal[:, :2], axis=1), normal[:, 2]))
    stored_tilt_deg = np.degrees(
        np.arctan2(np.linalg.norm(stored_normal[:, :2], axis=1), stored_normal[:, 2])
    )
    normal_delta_deg = np.degrees(
        np.arccos(np.clip(np.sum(normal * stored_normal, axis=1), -1.0, 1.0))
    )

    stats = {
        "definition": "tilt angle between the 1D true surface upward normal and world +Z",
        "demo_npz": str(args.demo_npz),
        "surface_model": str(args.model),
        "sample_count": int(len(tilt_deg)),
        "tilt_deg": {
            "mean": float(np.mean(tilt_deg)),
            "std": float(np.std(tilt_deg)),
            "median": float(np.median(tilt_deg)),
            "min": float(np.min(tilt_deg)),
            "max": float(np.max(tilt_deg)),
            "p05": float(np.percentile(tilt_deg, 5)),
            "p95": float(np.percentile(tilt_deg, 95)),
        },
        "stored_npz_normal_check_delta_deg": {
            "mean": float(np.mean(normal_delta_deg)),
            "max": float(np.max(normal_delta_deg)),
        },
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "human_demo_true_surface_normal_tilt_samples.csv"
    np.savetxt(
        csv_path,
        np.column_stack([xy, normal, tilt_deg]),
        delimiter=",",
        header="x_world_m,y_world_m,normal_x,normal_y,normal_z,tilt_deg",
        comments="",
    )
    summary_path = args.outdir / "human_demo_true_surface_normal_tilt_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=False)

    bins = np.linspace(max(0.0, float(np.floor(tilt_deg.min()))), float(np.ceil(tilt_deg.max())) + 0.25, 24)
    weights = np.full_like(tilt_deg, 100.0 / len(tilt_deg), dtype=np.float64)
    axes[0].hist(
        tilt_deg,
        bins=bins,
        weights=weights,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
    )
    axes[0].axvline(stats["tilt_deg"]["mean"], color="#D55E00", linewidth=1.6, label="mean")
    axes[0].axvline(stats["tilt_deg"]["median"], color="#009E73", linewidth=1.4, linestyle="--", label="median")
    text = (
        f"mean {stats['tilt_deg']['mean']:.2f} deg\n"
        f"std {stats['tilt_deg']['std']:.2f} deg\n"
        f"5-95% {stats['tilt_deg']['p05']:.2f}-{stats['tilt_deg']['p95']:.2f} deg"
    )
    axes[0].text(
        0.98,
        0.95,
        text,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92},
    )
    axes[0].set_title("Normal tilt distribution")
    axes[0].set_xlabel("Tilt from world +Z (deg)")
    axes[0].set_ylabel("Human demo samples (%)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(True, axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))

    sc = axes[1].scatter(
        xy[:, 0],
        xy[:, 1],
        c=tilt_deg,
        s=9,
        cmap="viridis",
        linewidths=0,
        alpha=0.9,
    )
    axes[1].set_title("Demonstration region")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.55)
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02)
    cbar.set_label("Tilt (deg)")

    fig.suptitle("1D true constraint surface normal tilt over human demonstrations", y=0.98, fontsize=12)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.20, top=0.78, wspace=0.28)
    png_path = args.outdir / "human_demo_true_surface_normal_tilt_distribution.png"
    pdf_path = args.outdir / "human_demo_true_surface_normal_tilt_distribution.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(json.dumps(stats, indent=2))
    print(png_path)
    print(pdf_path)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
