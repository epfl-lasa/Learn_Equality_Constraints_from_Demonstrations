#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ros.transforms import quat_to_matrix_xyzw


def _task_root(path: Path) -> Path | None:
    p = path.resolve()
    for parent in p.parents:
        if parent.parent.name == "tasks":
            return parent
    return None


def _default_out_path(dataset: Path, orientation_axis: str) -> Path:
    task_root = _task_root(dataset)
    if task_root is None:
        return Path("outputs/iiwa_replay/trajectory_plots") / f"{dataset.stem}_tcp_orientation_{orientation_axis}.png"

    rel_parts = dataset.resolve().relative_to(task_root).parts
    if "extracted" in rel_parts:
        idx = rel_parts.index("extracted")
        stem = rel_parts[idx + 1] if idx + 1 < len(rel_parts) else dataset.stem
    elif "crops" in rel_parts or "train_sets" in rel_parts:
        stem = dataset.stem
    else:
        stem = dataset.stem
    return task_root / "figures" / f"{stem}_tcp_orientation_{orientation_axis}.png"


def _axis_limits(points: np.ndarray, pad_ratio: float = 0.55) -> list[tuple[float, float]]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    span = max(float((hi - lo).max()), 0.05)
    return [(float(c - pad_ratio * span), float(c + pad_ratio * span)) for c in center]


def _orientation_axes(quat_xyzw: np.ndarray, axis: str) -> np.ndarray:
    axis_vec = {
        "x": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    }[axis]
    dirs = np.empty((quat_xyzw.shape[0], 3), dtype=np.float64)
    for i, quat in enumerate(quat_xyzw):
        dirs[i] = quat_to_matrix_xyzw(quat) @ axis_vec
    norm = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / np.maximum(norm, 1e-12)


def _style_axes(ax, limits: list[tuple[float, float]]) -> None:
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=24, azim=-58)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot iiwa TCP samples with orientation arrows.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to the task's figures/ directory for real_iiwa task datasets.",
    )
    parser.add_argument("--arrow-every", type=int, default=12)
    parser.add_argument("--arrow-length", type=float, default=0.035)
    parser.add_argument("--orientation-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    if not bool(args.show):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(args.dataset, allow_pickle=True)
    points_raw = np.asarray(data["x_raw"] if "x_raw" in data else data["tcp_pos"], dtype=np.float64)
    points = points_raw[:, :3]
    quat = np.asarray(data["tcp_quat"], dtype=np.float64)
    traj_id = (
        np.asarray(data["traj_id"], dtype=np.int64)
        if "traj_id" in data
        else np.zeros((points.shape[0],), dtype=np.int64)
    )
    dirs = _orientation_axes(quat, str(args.orientation_axis))
    limits = _axis_limits(points)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    arrow_every = max(1, int(args.arrow_every))
    arrow_len = float(args.arrow_length)
    for tid in np.unique(traj_id):
        mask = traj_id == tid
        idx = np.flatnonzero(mask)
        pts = points[idx]
        c = colors[int(tid) % len(colors)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=c, lw=0.8, alpha=0.45)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=c,
            s=float(args.point_size),
            depthshade=True,
            label=f"traj {int(tid)} ({len(pts)} pts)",
        )
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="limegreen", edgecolor="black", s=85, marker="o")
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color="red", edgecolor="black", s=90, marker="X")

        arrow_idx = idx[::arrow_every]
        # Always include the final point's orientation if it was skipped.
        if arrow_idx.size and arrow_idx[-1] != idx[-1]:
            arrow_idx = np.concatenate([arrow_idx, idx[-1:]])
        arrow_pts = points[arrow_idx]
        arrow_dirs = dirs[arrow_idx]
        ax.quiver(
            arrow_pts[:, 0],
            arrow_pts[:, 1],
            arrow_pts[:, 2],
            arrow_dirs[:, 0],
            arrow_dirs[:, 1],
            arrow_dirs[:, 2],
            length=arrow_len,
            normalize=True,
            color=c,
            linewidth=1.0,
            alpha=0.95,
            arrow_length_ratio=0.35,
        )

    _style_axes(ax, limits)
    ax.set_title(f"TCP trajectory + orientation (+{args.orientation_axis.upper()} axis, every {arrow_every} pts)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = args.out or _default_out_path(args.dataset, str(args.orientation_axis))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi))
    print(out)
    if bool(args.show):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
