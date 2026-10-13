#!/usr/bin/env python3
"""Compact slide visualization of the learned RealKuka constraint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.real_iiwa.registry import (  # noqa: E402
    real_iiwa_neural_surface_grid,
    real_iiwa_normalized_to_world_xyz,
)
from methods.utils.projection import project_points_with_steps_numpy  # noqa: E402
from planning.checkpoint_loader import load_constraint_model_from_checkpoint  # noqa: E402


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/bench/paper/oncl/RealKuka/RealKuka_oncl_seed0_model.pt"
)
DEFAULT_DEMO_NPZ = (
    PROJECT_ROOT
    / "datasets/real_iiwa/tasks/RealKuka/processed/train_sets/"
    / "human_demo_paper_toolz_arc3mm_cropped_v001.npz"
)
DEFAULT_OUTDIR = PROJECT_ROOT / "datasets/real_iiwa/tasks/RealKuka/figures"


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _axis_equal_3d(ax, xyz: np.ndarray, *, z_scale: float = 1.25) -> None:
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-6)
    radius = 0.5 * max(float(span[0]), float(span[1]), float(span[2]) * float(z_scale))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius / float(z_scale), center[2] + radius / float(z_scale))
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0 / float(z_scale)))
    except Exception:
        pass


def _build_learned_samples(
    *,
    checkpoint: Path,
    n_side: int,
    pad_ratio: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, _ckpt = load_constraint_model_from_checkpoint(str(checkpoint), device=device)
    cfg = SimpleNamespace(eval_gt_n_grid=int(n_side) * int(n_side), real_iiwa_surface_grid_pad_ratio=float(pad_ratio))
    init_grid = real_iiwa_neural_surface_grid("RealKuka", cfg).astype(np.float32)
    projected, steps = project_points_with_steps_numpy(
        model,
        init_grid,
        device=device,
        proj_steps=120,
        proj_alpha=0.3,
        proj_min_steps=30,
        f_abs_stop=None,
    )
    world_xyz = real_iiwa_normalized_to_world_xyz("RealKuka", projected[:, :3]).astype(np.float32)
    tool_z = _normalize_rows(projected[:, 3:6].astype(np.float32))
    return world_xyz, tool_z, steps.astype(np.float32)


def _arrow_indices(n_side: int, arrows_per_side: int) -> np.ndarray:
    loc = np.linspace(0.16, 0.84, int(arrows_per_side))
    ij = np.round(loc * (int(n_side) - 1)).astype(int)
    rows = []
    for i in ij:
        for j in ij:
            rows.append(i * int(n_side) + j)
    return np.asarray(rows, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--demo-npz", type=Path, default=DEFAULT_DEMO_NPZ)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-side", type=int, default=42)
    parser.add_argument("--pad-ratio", type=float, default=0.035)
    parser.add_argument("--arrows-per-side", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arrow-length-m", type=float, default=0.038)
    args = parser.parse_args()

    xyz, tool_z, steps = _build_learned_samples(
        checkpoint=args.checkpoint,
        n_side=int(args.n_side),
        pad_ratio=float(args.pad_ratio),
        device=str(args.device),
    )
    demo = np.load(args.demo_npz)
    demo_xyz = np.asarray(demo["position_world"], dtype=np.float32)
    demo_stride = max(1, len(demo_xyz) // 500)

    tri = mtri.Triangulation(xyz[:, 0], xyz[:, 1])
    arrow_idx = _arrow_indices(int(args.n_side), int(args.arrows_per_side))
    arrow_idx = arrow_idx[(0 <= arrow_idx) & (arrow_idx < len(xyz))]
    # Plot upward learned normal for readability; the actual tool local z-axis is the opposite direction.
    normal_dir = -tool_z[arrow_idx]

    args.outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    fig = plt.figure(figsize=(3.35, 2.45))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(
        tri,
        xyz[:, 2],
        color="#4C78A8",
        alpha=0.43,
        linewidth=0.18,
        edgecolor=(0.12, 0.22, 0.36, 0.18),
        antialiased=True,
        shade=True,
    )
    surf.set_label("learned constraint")

    ax.plot(
        demo_xyz[::demo_stride, 0],
        demo_xyz[::demo_stride, 1],
        demo_xyz[::demo_stride, 2],
        color="#2f2f2f",
        linewidth=1.0,
        alpha=0.78,
        label="human demonstration",
    )
    ax.quiver(
        xyz[arrow_idx, 0],
        xyz[arrow_idx, 1],
        xyz[arrow_idx, 2],
        normal_dir[:, 0],
        normal_dir[:, 1],
        normal_dir[:, 2],
        length=float(args.arrow_length_m),
        normalize=True,
        color="#D55E00",
        linewidths=0.85,
        arrow_length_ratio=0.27,
        label="orientation",
    )
    ax.scatter(
        xyz[arrow_idx, 0],
        xyz[arrow_idx, 1],
        xyz[arrow_idx, 2],
        s=9,
        color="#D55E00",
        depthshade=False,
        alpha=0.95,
    )

    ax.set_xlabel("x", labelpad=-4, fontsize=6)
    ax.set_ylabel("y", labelpad=-4, fontsize=6)
    ax.set_zlabel("z", labelpad=-6, fontsize=6)
    ax.tick_params(axis="both", labelsize=6, pad=-4)
    ax.tick_params(axis="z", labelsize=6, pad=-2)
    all_xyz = np.concatenate([xyz, demo_xyz[::demo_stride]], axis=0)
    _axis_equal_3d(ax, all_xyz, z_scale=1.85)
    ax.view_init(elev=26, azim=-62)
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.2,
        labelspacing=0.28,
        fontsize=6,
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    stem = "real_kuka_learned_constraint_orientation_slide_compact"
    png_path = args.outdir / f"{stem}.png"
    pdf_path = args.outdir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    summary = {
        "checkpoint": str(args.checkpoint),
        "demo_npz": str(args.demo_npz),
        "n_learned_samples": int(len(xyz)),
        "n_orientation_arrows": int(len(arrow_idx)),
        "projection_steps_mean": float(np.mean(steps)),
        "projection_steps_max": float(np.max(steps)),
        "note": "Orientation arrows show the upward learned normal for readability; tool local z-axis is the opposite direction.",
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    summary_path = args.outdir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
