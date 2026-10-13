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

from datasets.constraint_datasets import _genus3_tube_mesh
from experiments.config_loader import load_layered_config
from experiments.dataset_resolve import resolve_dataset
from methods.models.mlp import MLP


def _parse_figsize(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(text).lower().replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--figsize must look like WIDTH,HEIGHT, for example 1.95,1.42")
    return float(parts[0]), float(parts[1])


def _axis_equal_3d(ax, pts: np.ndarray, *, zoom: float = 1.0) -> None:
    p = np.asarray(pts, dtype=np.float32)
    mins = np.nanmin(p, axis=0)
    maxs = np.nanmax(p, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius / max(float(zoom), 1e-6), 1e-6)
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - 0.78 * radius), float(center[2] + 0.78 * radius))


def _match_height(path: str, *, target_height: int, crop_top: int) -> None:
    if int(target_height) <= 0:
        return
    try:
        from PIL import Image
    except Exception:
        return
    im = Image.open(path).convert("RGBA")
    if im.size[1] > int(target_height):
        top = min(max(0, int(crop_top)), im.size[1] - int(target_height))
        im.crop((0, top, im.size[0], top + int(target_height))).save(path)
        return
    if im.size[1] < int(target_height):
        out = Image.new("RGBA", (im.size[0], int(target_height)), (255, 255, 255, 255))
        y = (int(target_height) - im.size[1]) // 2
        out.alpha_composite(im, (0, y))
        out.save(path)


def _load_cfg(seed: int | None) -> SimpleNamespace:
    cfg_map, _ = load_layered_config(os.path.join(_PROJECT_ROOT, "configs"), "oncl", "3d_genus3_tube_surface_traj")
    if seed is not None:
        cfg_map["seed"] = int(seed)
    return SimpleNamespace(**cfg_map)


def _load_learned_surface(
    ckpt_path: str,
    *,
    bounds: tuple[np.ndarray, np.ndarray],
    resolution: int,
    max_faces: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from skimage import measure  # type: ignore
    except Exception as e:
        raise RuntimeError("learned genus3 plotting requires scikit-image") from e

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
        faces = faces[rng.choice(len(faces), size=int(max_faces), replace=False)]
    return verts.astype(np.float32), faces.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compact true/learned plots for the 3D genus-3 handle dataset.")
    ap.add_argument("--mode", choices=("true", "learned"), required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figsize", default="1.95,1.42")
    ap.add_argument("--seed", type=int, default=2116)
    ap.add_argument("--max-faces", type=int, default=5200)
    ap.add_argument("--learned-resolution", type=int, default=88)
    ap.add_argument("--surface-alpha", type=float, default=0.46)
    ap.add_argument("--show-demos", action="store_true")
    ap.add_argument("--demo-as-trajectories", action="store_true")
    ap.add_argument("--demo-point-size", type=float, default=0.18)
    ap.add_argument("--demo-point-alpha", type=float, default=0.16)
    ap.add_argument("--demo-line-width", type=float, default=0.28)
    ap.add_argument("--demo-line-alpha", type=float, default=0.36)
    ap.add_argument("--demo-color", default="#111827")
    ap.add_argument("--demo-color-cycle", action="store_true")
    ap.add_argument("--demo-max-count", type=int, default=24)
    ap.add_argument("--demo-stride", type=int, default=1)
    ap.add_argument("--demo-indices", default="")
    ap.add_argument("--demo-trajectory-scatter", action="store_true")
    ap.add_argument("--demo-break-jump", type=float, default=0.0)
    ap.add_argument("--zoom", type=float, default=1.22)
    ap.add_argument("--elev", type=float, default=17.0)
    ap.add_argument("--azim", type=float, default=-78.0)
    ap.add_argument("--hide-axis-labels", action="store_true")
    ap.add_argument("--legend-y", type=float, default=0.86)
    ap.add_argument("--target-height", type=int, default=362)
    ap.add_argument("--crop-top", type=int, default=59)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = _load_cfg(int(args.seed))
    dataset_name = "3d_genus3_tube_surface_traj" if bool(args.show_demos) else "3d_genus3_tube_surface"
    ds = resolve_dataset(dataset_name, cfg)
    x_train = np.asarray(ds["x_train"], dtype=np.float32)
    rng = np.random.default_rng(int(args.seed) + 7501)

    true_verts, true_faces = _genus3_tube_mesh(
        nx=int(getattr(cfg, "genus3_mesh_nx", 112)),
        ny=int(getattr(cfg, "genus3_mesh_ny", 88)),
        nz=int(getattr(cfg, "genus3_mesh_nz", 88)),
        radius=float(getattr(cfg, "genus3_tube_radius", 0.165)),
    )
    true_faces_plot = np.asarray(true_faces, dtype=np.int64)
    if len(true_faces_plot) > int(args.max_faces):
        true_faces_plot = true_faces_plot[rng.choice(len(true_faces_plot), size=int(args.max_faces), replace=False)]

    if str(args.mode) == "learned":
        if not str(args.ckpt).strip():
            raise ValueError("--ckpt is required for --mode learned")
        pad = np.asarray([0.20, 0.15, 0.15], dtype=np.float32)
        verts, faces = _load_learned_surface(
            str(args.ckpt),
            bounds=(np.min(true_verts, axis=0) - pad, np.max(true_verts, axis=0) + pad),
            resolution=int(args.learned_resolution),
            max_faces=int(args.max_faces),
            rng=rng,
            device=str(args.device),
        )
        color = "#ef4444"
        label = "Learned zero surface"
    else:
        verts, faces = true_verts, true_faces_plot
        color = "#45b885"
        label = "True surface"

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with plt.rc_context(
        {
            "font.size": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
        }
    ):
        fig = plt.figure(figsize=_parse_figsize(args.figsize))
        ax = fig.add_subplot(111, projection="3d")
        poly = Poly3DCollection(
            verts[faces],
            facecolors=mpl.colors.to_rgba(color, alpha=float(args.surface_alpha)),
            edgecolor=(0, 0, 0, 0.018),
            linewidth=0.025,
        )
        ax.add_collection3d(poly)
        handles: list[object] = [
            Patch(facecolor=color, edgecolor="none", alpha=0.85, label=label),
        ]
        if bool(args.show_demos):
            if bool(args.demo_as_trajectories):
                traj_count = max(1, int(getattr(cfg, "traj_count", 1)))
                traj_len = max(2, int(getattr(cfg, "traj_len", int(np.ceil(len(x_train) / max(traj_count, 1))))))
                if str(args.demo_indices).strip():
                    traj_ids = [int(x.strip()) for x in str(args.demo_indices).split(",") if x.strip()]
                    starts = [ti * traj_len for ti in traj_ids if 0 <= ti * traj_len < len(x_train)]
                else:
                    starts = list(range(0, len(x_train), traj_len))
                    starts = starts[:: max(1, int(args.demo_stride))]
                    if int(args.demo_max_count) > 0:
                        starts = starts[: int(args.demo_max_count)]
                cycle_colors = ["#2563eb", "#dc2626", "#7c3aed", "#ea580c", "#0891b2", "#16a34a"]
                for demo_i, start in enumerate(starts):
                    seg = x_train[start : start + traj_len]
                    if len(seg) < 2:
                        continue
                    demo_color = cycle_colors[demo_i % len(cycle_colors)] if bool(args.demo_color_cycle) else str(args.demo_color)
                    pieces = [seg]
                    if float(args.demo_break_jump) > 0.0:
                        jump = np.linalg.norm(np.diff(seg, axis=0), axis=1)
                        cuts = np.where(jump > float(args.demo_break_jump))[0] + 1
                        pieces = [p for p in np.split(seg, cuts) if len(p) >= 2]
                    for piece in pieces:
                        ax.plot(
                            piece[:, 0],
                            piece[:, 1],
                            piece[:, 2],
                            color=demo_color,
                            linewidth=float(args.demo_line_width),
                            alpha=float(args.demo_line_alpha),
                            zorder=7,
                        )
                    if bool(args.demo_trajectory_scatter):
                        ax.scatter(
                            seg[:, 0],
                            seg[:, 1],
                            seg[:, 2],
                            s=float(args.demo_point_size),
                            c=demo_color,
                            alpha=min(0.95, float(args.demo_line_alpha) + 0.1),
                            depthshade=False,
                            zorder=8,
                        )
                handles.append(Line2D([0], [0], color=cycle_colors[0] if bool(args.demo_color_cycle) else str(args.demo_color), linewidth=0.8, alpha=0.65, label="Demonstrations"))
            else:
                ax.scatter(
                    x_train[:, 0],
                    x_train[:, 1],
                    x_train[:, 2],
                    s=float(args.demo_point_size),
                    c=str(args.demo_color),
                    alpha=float(args.demo_point_alpha),
                    depthshade=False,
                    zorder=6,
                )
                handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=str(args.demo_color), alpha=0.45, markersize=2.2, label="Demonstrations"))

        _axis_equal_3d(ax, true_verts, zoom=float(args.zoom))
        ax.view_init(elev=float(args.elev), azim=float(args.azim))
        ax.set_xlabel("" if bool(args.hide_axis_labels) else "$x_1$", labelpad=0)
        ax.set_ylabel("" if bool(args.hide_axis_labels) else "$x_2$", labelpad=0)
        ax.set_zlabel("")
        if not bool(args.hide_axis_labels):
            maxs = np.max(true_verts, axis=0)
            ax.text(float(maxs[0] + 0.05), float(maxs[1] + 0.03), float(maxs[2] + 0.04), "$x_3$", fontsize=6)
        ax.tick_params(labelsize=5, pad=-2)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.set_proj_type("persp")
        try:
            ax.dist = 7.0 / max(float(args.zoom), 1e-6)
        except Exception:
            pass
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.03, float(args.legend_y)),
            ncol=1,
            fontsize=5,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.1,
            handletextpad=0.3,
        )
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    _match_height(out_path, target_height=int(args.target_height), crop_top=int(args.crop_top))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
