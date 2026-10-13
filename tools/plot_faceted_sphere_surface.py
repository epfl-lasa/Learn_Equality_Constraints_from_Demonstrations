#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.constraint_datasets import (
    _faceted_sphere_mesh,
    _spiky_sphere_mesh,
    faceted_sphere_control_points,
    spiky_sphere_control_points,
)
from experiments.config_loader import load_layered_config
from experiments.dataset_resolve import resolve_dataset
from methods.models.mlp import MLP


def _axis_equal_3d(ax, pts: np.ndarray, *, zoom: float = 1.0) -> None:
    p = np.asarray(pts, dtype=np.float32)
    mins = np.nanmin(p, axis=0)
    maxs = np.nanmax(p, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = radius / max(float(zoom), 1e-6)
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))


def _load_learned_surface(
    ckpt_path: str,
    *,
    bounds: tuple[np.ndarray, np.ndarray],
    resolution: int,
    max_faces: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from skimage import measure  # type: ignore
    except Exception as e:
        raise RuntimeError("learned faceted-sphere plotting requires scikit-image") from e

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MLP(
        in_dim=int(ckpt["in_dim"]),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
        out_dim=int(ckpt["constraint_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(str(device))
    model.eval()

    lo, hi = bounds
    xs = np.linspace(float(lo[0]), float(hi[0]), int(resolution), dtype=np.float32)
    ys = np.linspace(float(lo[1]), float(hi[1]), int(resolution), dtype=np.float32)
    zs = np.linspace(float(lo[2]), float(hi[2]), int(resolution), dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).astype(np.float32)
    vals = np.empty((len(pts),), dtype=np.float32)
    bs = 32768
    with torch.no_grad():
        for start in range(0, len(pts), bs):
            x = torch.from_numpy(pts[start : start + bs]).to(str(device))
            f = model(x)
            if f.dim() == 1:
                f = f.unsqueeze(1)
            vals[start : start + bs] = f[:, 0].detach().cpu().numpy().astype(np.float32)
    vol = vals.reshape((len(xs), len(ys), len(zs)))
    verts, faces, _normals, _values = measure.marching_cubes(
        vol,
        level=0.0,
        spacing=(float(xs[1] - xs[0]), float(ys[1] - ys[0]), float(zs[1] - zs[0])),
    )
    verts[:, 0] += float(xs[0])
    verts[:, 1] += float(ys[0])
    verts[:, 2] += float(zs[0])
    faces = faces.astype(np.int64)
    if len(faces) > int(max_faces):
        rng = np.random.default_rng(20260723)
        faces = faces[rng.choice(len(faces), size=int(max_faces), replace=False)]
    return verts.astype(np.float32), faces.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the 3D faceted sphere dataset.")
    ap.add_argument("--dataset", default="3d_faceted_sphere_surface_traj")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/analysis/faceted_sphere_preview/3d_faceted_sphere_surface.png")
    ap.add_argument("--show-demos", action="store_true")
    ap.add_argument("--show-control-points", action="store_true")
    ap.add_argument("--learned-ckpt", default="")
    ap.add_argument("--learned-resolution", type=int, default=96)
    ap.add_argument("--learned-max-faces", type=int, default=60000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--demo-max-count", type=int, default=24)
    ap.add_argument("--demo-stride", type=int, default=2)
    ap.add_argument("--surface-alpha", type=float, default=0.78)
    ap.add_argument("--surface-edge-alpha", type=float, default=0.13)
    ap.add_argument("--demo-line-width", type=float, default=0.8)
    ap.add_argument("--demo-line-alpha", type=float, default=0.62)
    ap.add_argument("--demo-lift", type=float, default=1.018)
    ap.add_argument("--figsize", default="3.4,3.0")
    ap.add_argument("--elev", type=float, default=19.0)
    ap.add_argument("--azim", type=float, default=-46.0)
    ap.add_argument("--zoom", type=float, default=1.16)
    args = ap.parse_args()

    cfg_map, _ = load_layered_config(os.path.join(_PROJECT_ROOT, "configs"), "oncl", str(args.dataset))
    cfg_map["seed"] = int(args.seed)
    cfg = SimpleNamespace(**cfg_map)
    ds = resolve_dataset(str(args.dataset), cfg)
    x_train = np.asarray(ds["x_train"], dtype=np.float32)

    if str(args.dataset).startswith("3d_spiky_sphere_surface"):
        true_verts, true_faces = _spiky_sphere_mesh(
            n_lat=int(getattr(cfg, "spiky_sphere_n_lat", 24)),
            n_lon=int(getattr(cfg, "spiky_sphere_n_lon", 48)),
            spike_scale=float(getattr(cfg, "spiky_sphere_spike_scale", 0.86)),
            sigma=float(getattr(cfg, "spiky_sphere_sigma", 0.105)),
            sharpness=float(getattr(cfg, "spiky_sphere_sharpness", 8.0)),
            n_ctrl=int(getattr(cfg, "spiky_sphere_n_ctrl", 58)),
        )
        ctrl = spiky_sphere_control_points(
            spike_scale=float(getattr(cfg, "spiky_sphere_spike_scale", 0.86)),
            n_ctrl=int(getattr(cfg, "spiky_sphere_n_ctrl", 58)),
        )
    else:
        true_verts, true_faces = _faceted_sphere_mesh(
            n_lat=int(getattr(cfg, "faceted_sphere_n_lat", 18)),
            n_lon=int(getattr(cfg, "faceted_sphere_n_lon", 36)),
            bump_scale=float(getattr(cfg, "faceted_sphere_bump_scale", 0.72)),
            sigma=float(getattr(cfg, "faceted_sphere_sigma", 0.20)),
            sharpness=float(getattr(cfg, "faceted_sphere_sharpness", 3.2)),
        )
        ctrl = faceted_sphere_control_points(bump_scale=float(getattr(cfg, "faceted_sphere_bump_scale", 0.72)))
    if str(args.learned_ckpt).strip():
        pad = np.asarray([0.22, 0.22, 0.22], dtype=np.float32)
        verts, faces = _load_learned_surface(
            str(args.learned_ckpt),
            bounds=(np.min(true_verts, axis=0) - pad, np.max(true_verts, axis=0) + pad),
            resolution=int(args.learned_resolution),
            max_faces=int(args.learned_max_faces),
            device=str(args.device),
        )
        surface_color = "#ef4444"
        surface_label = "Learned zero surface"
    else:
        verts, faces = true_verts, true_faces
        surface_color = "#5abf8a"
        surface_label = "True surface"

    parts = [p.strip() for p in str(args.figsize).replace("x", ",").split(",") if p.strip()]
    figsize = (float(parts[0]), float(parts[1])) if len(parts) == 2 else (3.4, 3.0)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with plt.rc_context({"font.size": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        poly = Poly3DCollection(
            verts[faces],
            facecolors=mpl.colors.to_rgba(surface_color, alpha=float(args.surface_alpha)),
            edgecolor=(0.05, 0.18, 0.13, float(args.surface_edge_alpha)),
            linewidth=0.16,
        )
        ax.add_collection3d(poly)
        handles: list[object] = [Patch(facecolor=surface_color, edgecolor="none", alpha=0.9, label=surface_label)]

        if bool(args.show_demos):
            traj_count = max(1, int(getattr(cfg, "traj_count", 1)))
            traj_len = max(2, int(np.ceil(len(x_train) / max(traj_count, 1))))
            starts = list(range(0, len(x_train), traj_len))[:: max(1, int(args.demo_stride))]
            if int(args.demo_max_count) > 0:
                starts = starts[: int(args.demo_max_count)]
            for start in starts:
                seg = x_train[start : start + traj_len]
                if len(seg) < 2:
                    continue
                seg_plot = seg * float(args.demo_lift)
                ax.plot(
                    seg_plot[:, 0],
                    seg_plot[:, 1],
                    seg_plot[:, 2],
                    color="#2563eb",
                    linewidth=float(args.demo_line_width),
                    alpha=float(args.demo_line_alpha),
                    zorder=20,
                )
            handles.append(Line2D([0], [0], color="#2563eb", linewidth=1.2, alpha=0.85, label="Demonstrations"))

        if bool(args.show_control_points):
            ctrl_plot = ctrl * 1.045
            ax.scatter(ctrl_plot[:, 0], ctrl_plot[:, 1], ctrl_plot[:, 2], s=22, c="#111827", alpha=0.95, depthshade=False)
            handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", markersize=4, label="Control points"))

        _axis_equal_3d(ax, true_verts, zoom=float(args.zoom))
        ax.view_init(elev=float(args.elev), azim=float(args.azim))
        ax.set_xlabel("$x_1$", labelpad=-2)
        ax.set_ylabel("$x_2$", labelpad=-2)
        ax.set_zlabel("$x_3$", labelpad=-2)
        ax.tick_params(pad=-2)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.legend(handles=handles, loc="upper left", frameon=False, borderaxespad=0.2)
        fig.tight_layout(pad=0.05)
        fig.savefig(out_path, dpi=260, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
