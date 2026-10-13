from __future__ import annotations

import math
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MaxNLocator
from torch import nn

from evaluation.evaluator import DEFAULT_EVAL_CFG, eval_bounds_from_train, resolve_gt_grid
from datasets.constraint_datasets import generate_dataset, sine_surface_z_and_normal_from_xy
from datasets.ur5_pybullet_utils import (
    UR5_LINK_LENGTHS,
    _make_pybullet_friendly_urdf,
    pick_default_ee_link_index,
    resolve_ur5_kinematics_cfg,
    resolve_ur5_render_cfg,
)
from planning.kinematics import (
    is_arm_dataset,
    planar_fk,
    spatial_fk,
    spatial_tool_axis_n6,
)
from planning.planner import (
    init_path_joint_spline,
    init_path_via_workspace_ik,
    pick_far_pair_workspace_planar,
    plan_path,
)
from methods.utils.projection import project_points_with_steps_numpy, project_trajectory_numpy
from plotting.plot_common import plot_contour_traj_2d

N6_WORKSPACE_VIS_POINTS_DEFAULT = 90
ZERO_EPS_QUANTILE_DEFAULT = float(DEFAULT_EVAL_CFG["zero_eps_quantile"])


def _split_polyline_on_jumps(points: np.ndarray, *, jump_factor: float = 8.0) -> list[np.ndarray]:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[0] <= 1:
        return [pts] if pts.size else []
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    positive = steps[steps > 1e-12]
    if len(positive) == 0:
        return [pts]
    med = float(np.median(positive))
    if not np.isfinite(med) or med <= 0.0:
        return [pts]
    cut = np.flatnonzero(steps > float(jump_factor) * med) + 1
    if len(cut) == 0:
        return [pts]
    bounds = np.concatenate([[0], cut, [pts.shape[0]]]).astype(np.int64)
    return [pts[int(a): int(b)] for a, b in zip(bounds[:-1], bounds[1:]) if int(b) - int(a) > 1]


def _curve_component_count(dataset_name: str | None) -> int | None:
    name = str(dataset_name or "")
    if "triplet" in name:
        return 3
    if (
        "linked_curves_3d" in name
        or "knot_spline_union_traj" in name
        or "knot_spline_union" in name
        or "nested_ellipses_union" in name
        or "linked_ellipses_union" in name
    ):
        return 2
    return None


def _split_counts(total: int, component_count: int) -> list[int]:
    if total <= 0 or component_count <= 0:
        return []
    n = min(int(component_count), int(total))
    base = int(total) // n
    out = [base for _ in range(n)]
    out[-1] = int(total) - base * (n - 1)
    return out


def _split_gt_components(gt_points: np.ndarray | None, component_count: int | None) -> list[np.ndarray]:
    if gt_points is None or len(gt_points) == 0:
        return []
    pts = np.asarray(gt_points, dtype=np.float32)
    if component_count is None or component_count <= 1:
        return [pts]
    out: list[np.ndarray] = []
    start = 0
    for count in _split_counts(len(pts), int(component_count)):
        end = start + int(count)
        if end > start:
            out.append(pts[start:end])
        start = end
    return out


def _assign_points_to_gt_components(
    points: np.ndarray,
    gt_components: list[np.ndarray],
) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0 or len(gt_components) <= 1:
        return [pts] if len(pts) else []

    try:
        from scipy.spatial import cKDTree  # type: ignore

        dists = []
        for comp in gt_components:
            if len(comp) == 0:
                dists.append(np.full((len(pts),), np.inf, dtype=np.float32))
            else:
                d, _ = cKDTree(np.asarray(comp, dtype=np.float32)).query(pts, k=1)
                dists.append(np.asarray(d, dtype=np.float32))
        labels = np.argmin(np.stack(dists, axis=1), axis=1)
    except Exception:
        dists = []
        chunk = 512
        for comp in gt_components:
            comp_arr = np.asarray(comp, dtype=np.float32)
            cur = np.full((len(pts),), np.inf, dtype=np.float32)
            if len(comp_arr) > 0:
                for s in range(0, len(pts), chunk):
                    e = min(len(pts), s + chunk)
                    d2 = np.sum((pts[s:e, None, :] - comp_arr[None, :, :]) ** 2, axis=2)
                    cur[s:e] = np.sqrt(np.maximum(np.min(d2, axis=1), 0.0))
            dists.append(cur)
        labels = np.argmin(np.stack(dists, axis=1), axis=1)

    grouped = []
    for i in range(len(gt_components)):
        part = pts[labels == i]
        if len(part) > 0:
            grouped.append(part)
    return grouped


def _proj_from_cfg(cfg: Any) -> tuple[float, int, int]:
    proj = getattr(cfg, "projector", None)
    if isinstance(proj, dict):
        alpha = float(proj.get("alpha", 0.3))
        steps = int(proj.get("steps", 100))
        min_steps = int(proj.get("min_steps", 30))
        return alpha, steps, min_steps
    return (
        float(getattr(cfg, "proj_alpha", 0.3)),
        int(getattr(cfg, "proj_steps", 100)),
        int(getattr(cfg, "proj_min_steps", 30)),
    )


def _pln(cfg: Any, key: str, default: Any) -> Any:
    pln = getattr(cfg, "planner", None)
    if isinstance(pln, dict) and key in pln:
        return pln[key]
    return default

def _plot_training_diagnostics(hist: dict[str, np.ndarray], out_path: str, title: str) -> None:
    ep = hist["epoch"]
    f_abs = hist["f_abs_mean"]
    gnorm = hist["grad_norm_mean"]
    ortho = hist["ortho_err"]
    k = int(f_abs.shape[1]) if f_abs.ndim == 2 else 1

    ncols = 3
    plt.figure(figsize=(4.0 * ncols, 3.6))

    ax1 = plt.subplot(1, ncols, 1)
    for i in range(k):
        ax1.plot(ep, f_abs[:, i], lw=1.6, label=f"|f{i+1}| mean")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("on-data mean")
    ax1.set_title("|f_i| on data")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2 = plt.subplot(1, ncols, 2)
    for i in range(k):
        ax2.plot(ep, gnorm[:, i], lw=1.6, label=f"||grad f{i+1}|| mean")
    ax2.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.6)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("on-data mean")
    ax2.set_title("Gradient Norm")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    ax3 = plt.subplot(1, ncols, 3)
    ax3.plot(ep, ortho, lw=1.8, color="#ef4444")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("mean |J J^T - I|")
    ax3.set_title("Orthogonality Error")
    ax3.grid(alpha=0.25)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _plot_constraint_2d(
    model: nn.Module,
    x_train: np.ndarray,
    traj: np.ndarray,
    out_path: str,
    title: str,
    axis_labels: tuple[str, str],
    cfg: Any,
    worst_traj: np.ndarray | None = None,
    worst_x0: np.ndarray | None = None,
) -> None:
    plot_contour_traj_2d(
        model=model,
        x_train=x_train,
        traj=traj,
        out_path=out_path,
        title=title,
        axis_labels=axis_labels,
        cfg=cfg,
        line_color="green",
        worst_traj=worst_traj,
        worst_x0=worst_x0,
    )


def _plot_zero_surfaces_3d(
    model: nn.Module,
    x_train: np.ndarray,
    traj: np.ndarray,
    out_path: str,
    title: str,
    axis_labels: tuple[str, str, str],
    cfg: Any,
    intersection_points: np.ndarray | None = None,
) -> None:
    # Reuse evaluator bounds so near-constant axes (e.g., 3d_0z_*) also get a usable visual span.
    mins, maxs = eval_bounds_from_train(x_train, cfg)

    n = max(16, int(cfg.surface_plot_n))
    xx, yy, zz = np.meshgrid(
        np.linspace(float(mins[0]), float(maxs[0]), n),
        np.linspace(float(mins[1]), float(maxs[1]), n),
        np.linspace(float(mins[2]), float(maxs[2]), n),
        indexing="ij",
    )
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    device = next(model.parameters()).device

    out_list = []
    with torch.no_grad():
        for s in range(0, len(pts), max(1024, int(cfg.surface_eval_chunk))):
            e = min(len(pts), s + max(1024, int(cfg.surface_eval_chunk)))
            f = model(torch.from_numpy(pts[s:e]).to(device))
            if f.dim() == 1:
                f = f.unsqueeze(1)
            out_list.append(f.detach().cpu().numpy())
    f_all = np.concatenate(out_list, axis=0)
    f1 = f_all[:, 0]
    has_f2 = f_all.shape[1] >= 2
    f2 = f_all[:, 1] if has_f2 else None

    with torch.no_grad():
        f_train = model(torch.from_numpy(x_train).to(device))
        if f_train.dim() == 1:
            f_train = f_train.unsqueeze(1)
        eps_q = float(getattr(cfg, "zero_eps_quantile", ZERO_EPS_QUANTILE_DEFAULT))
        eps1 = max(float(np.percentile(np.abs(f_train[:, 0].detach().cpu().numpy()), eps_q)), 1e-4)
        if has_f2:
            eps2 = max(float(np.percentile(np.abs(f_train[:, 1].detach().cpu().numpy()), eps_q)), 1e-4)
            h_on = torch.linalg.norm(f_train[:, :2], dim=1).detach().cpu().numpy()
            eps_h = max(float(np.percentile(h_on, eps_q)), 1e-6)

    rendered = False
    verts1 = faces1 = verts2 = faces2 = None
    if bool(cfg.surface_use_marching_cubes):
        try:
            from skimage import measure  # type: ignore

            f1_vol = f1.reshape(n, n, n)
            dx = float((maxs[0] - mins[0]) / max(1, n - 1))
            dy = float((maxs[1] - mins[1]) / max(1, n - 1))
            dz = float((maxs[2] - mins[2]) / max(1, n - 1))
            lvl1 = 0.0 if (float(np.min(f1_vol)) <= 0.0 <= float(np.max(f1_vol))) else eps1
            vol1 = f1_vol if lvl1 == 0.0 else np.abs(f1_vol)
            verts1, faces1, _, _ = measure.marching_cubes(vol1, level=lvl1, spacing=(dx, dy, dz))
            verts1 += np.array([mins[0], mins[1], mins[2]], dtype=np.float32)
            if has_f2 and f2 is not None:
                f2_vol = f2.reshape(n, n, n)
                lvl2 = 0.0 if (float(np.min(f2_vol)) <= 0.0 <= float(np.max(f2_vol))) else eps2
                vol2 = f2_vol if lvl2 == 0.0 else np.abs(f2_vol)
                verts2, faces2, _, _ = measure.marching_cubes(vol2, level=lvl2, spacing=(dx, dy, dz))
                verts2 += np.array([mins[0], mins[1], mins[2]], dtype=np.float32)
            rendered = True
        except Exception:
            rendered = False

    p1 = p2 = p12 = None
    if has_f2 and f2 is not None:
        if intersection_points is not None and len(intersection_points) > 0:
            cand = intersection_points.astype(np.float32)
            with torch.no_grad():
                fc = model(torch.from_numpy(cand).to(device))
                if fc.dim() == 1:
                    fc = fc.unsqueeze(1)
                hc = torch.linalg.norm(fc[:, :2], dim=1).detach().cpu().numpy()
            p12 = cand[hc <= eps_h]
        else:
            h_grid = np.sqrt(f1 * f1 + f2 * f2)
            p12 = pts[h_grid <= eps_h]
        if len(p12) > 1200:
            idx = np.random.choice(len(p12), size=1200, replace=False)
            p12 = p12[idx]
    if not rendered:
        p1 = pts[np.abs(f1) <= eps1]
        if len(p1) > cfg.surface_max_points:
            idx = np.random.choice(len(p1), size=int(cfg.surface_max_points), replace=False)
            p1 = p1[idx]
        if has_f2 and f2 is not None:
            p2 = pts[np.abs(f2) <= eps2]
            if len(p2) > cfg.surface_max_points:
                idx = np.random.choice(len(p2), size=int(cfg.surface_max_points), replace=False)
                p2 = p2[idx]

    train_plot = x_train
    if len(train_plot) > cfg.plot_train_max_points:
        idx = np.random.choice(len(train_plot), size=int(cfg.plot_train_max_points), replace=False)
        train_plot = train_plot[idx]
    step = max(1, int(np.ceil(traj.shape[1] / max(1, int(cfg.plot_traj_max_count)))))
    traj_ids = list(range(0, traj.shape[1], step))
    stride = max(1, int(cfg.plot_traj_stride))

    def _draw_common(ax) -> None:
        ax.scatter(train_plot[:, 0], train_plot[:, 1], train_plot[:, 2], s=5, c="gray", alpha=0.24, label="train")
        for i in traj_ids:
            ax.plot(
                traj[::stride, i, 0], traj[::stride, i, 1], traj[::stride, i, 2],
                "-", color="green", linewidth=1.2 if i == traj_ids[0] else 0.8, alpha=0.65,
                label="traj" if i == traj_ids[0] else None,
            )
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        ax.set_xlim(float(mins[0]), float(maxs[0]))
        ax.set_ylim(float(mins[1]), float(maxs[1]))
        ax.set_zlim(float(mins[2]), float(maxs[2]))
        ax.set_proj_type("persp")
        ax.view_init(elev=28, azim=-42)

    def _add_surface(ax, verts, faces, color: str, alpha: float, label: str) -> None:
        poly = Poly3DCollection(verts[faces], alpha=alpha, facecolor=color, edgecolor=(0, 0, 0, 0.12), linewidth=0.1)
        poly.set_label(label)
        ax.add_collection3d(poly)

    if has_f2:
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132, projection="3d")
        ax3 = fig.add_subplot(133, projection="3d")
        if rendered and verts1 is not None and faces1 is not None and verts2 is not None and faces2 is not None:
            _add_surface(ax1, verts1, faces1, "#22d3ee", 0.24, "f1=0")
            _add_surface(ax2, verts2, faces2, "#f472b6", 0.24, "f2=0")
            _add_surface(ax3, verts1, faces1, "#22d3ee", 0.14, "f1=0")
            _add_surface(ax3, verts2, faces2, "#f472b6", 0.14, "f2=0")
            if p12 is not None and len(p12) > 0:
                ax3.scatter(
                    p12[:, 0], p12[:, 1], p12[:, 2],
                    s=2.0, c="red", alpha=0.9,
                    label=f"intersection (q{eps_q:.0f})",
                )
        else:
            if p1 is not None and len(p1) > 0:
                ax1.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=1.2, c="#22d3ee", alpha=0.26, label=f"f1≈0 (q{eps_q:.0f}={eps1:.3g})")
                ax3.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=1.0, c="#22d3ee", alpha=0.18, label=f"f1≈0 (q{eps_q:.0f}={eps1:.3g})")
            if p2 is not None and len(p2) > 0:
                ax2.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=1.2, c="#f472b6", alpha=0.26, label=f"f2≈0 (q{eps_q:.0f}={eps2:.3g})")
                ax3.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=1.0, c="#f472b6", alpha=0.18, label=f"f2≈0 (q{eps_q:.0f}={eps2:.3g})")
            if p12 is not None and len(p12) > 0:
                ax3.scatter(
                    p12[:, 0], p12[:, 1], p12[:, 2],
                    s=2.0, c="red", alpha=0.9,
                    label=f"intersection (q{eps_q:.0f})",
                )
        _draw_common(ax1)
        _draw_common(ax2)
        _draw_common(ax3)
        ax1.set_title("f1")
        ax2.set_title("f2")
        ax3.set_title("f1 + f2 + intersection")
        ax1.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper left", fontsize=8)
        ax3.legend(loc="upper left", fontsize=8)
    else:
        fig = plt.figure(figsize=(7, 6))
        ax1 = fig.add_subplot(111, projection="3d")
        if rendered and verts1 is not None and faces1 is not None:
            _add_surface(ax1, verts1, faces1, "#ef4444", 0.24, "f1=0")
        elif p1 is not None and len(p1) > 0:
            ax1.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=1.5, c="#ef4444", alpha=0.30, label=f"f1≈0 (q{eps_q:.0f}={eps1:.3g})")
        _draw_common(ax1)
        ax1.set_title("f1 (codim=1)")
        ax1.legend(loc="upper left", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    if bool(cfg.show_3d_plot):
        plt.show()
    plt.close(fig)


def _plot_constraint_surface_paper_3d(
    model: nn.Module,
    x_train: np.ndarray,
    out_path: str,
    axis_labels: tuple[str, str, str],
    cfg: Any,
    intersection_points: np.ndarray | None = None,
    dataset_name: str | None = None,
) -> None:
    # Paper-oriented variant: keep only the combined (third) view.
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    n = max(16, int(cfg.surface_plot_n))
    xx, yy, zz = np.meshgrid(
        np.linspace(float(mins[0]), float(maxs[0]), n),
        np.linspace(float(mins[1]), float(maxs[1]), n),
        np.linspace(float(mins[2]), float(maxs[2]), n),
        indexing="ij",
    )
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    device = next(model.parameters()).device

    out_list = []
    with torch.no_grad():
        for s in range(0, len(pts), max(1024, int(cfg.surface_eval_chunk))):
            e = min(len(pts), s + max(1024, int(cfg.surface_eval_chunk)))
            f = model(torch.from_numpy(pts[s:e]).to(device))
            if f.dim() == 1:
                f = f.unsqueeze(1)
            out_list.append(f.detach().cpu().numpy())
    f_all = np.concatenate(out_list, axis=0)
    f1 = f_all[:, 0]
    if f_all.shape[1] < 2:
        return
    f2 = f_all[:, 1]

    with torch.no_grad():
        f_train = model(torch.from_numpy(x_train).to(device))
        if f_train.dim() == 1:
            f_train = f_train.unsqueeze(1)
        eps_q = float(getattr(cfg, "zero_eps_quantile", ZERO_EPS_QUANTILE_DEFAULT))
        eps1 = max(float(np.percentile(np.abs(f_train[:, 0].detach().cpu().numpy()), eps_q)), 1e-4)
        eps2 = max(float(np.percentile(np.abs(f_train[:, 1].detach().cpu().numpy()), eps_q)), 1e-4)
        h_on = torch.linalg.norm(f_train[:, :2], dim=1).detach().cpu().numpy()
        eps_h = max(float(np.percentile(h_on, eps_q)), 1e-6)

    rendered = False
    verts1 = faces1 = verts2 = faces2 = None
    if bool(cfg.surface_use_marching_cubes):
        try:
            from skimage import measure  # type: ignore

            f1_vol = f1.reshape(n, n, n)
            f2_vol = f2.reshape(n, n, n)
            dx = float((maxs[0] - mins[0]) / max(1, n - 1))
            dy = float((maxs[1] - mins[1]) / max(1, n - 1))
            dz = float((maxs[2] - mins[2]) / max(1, n - 1))

            lvl1 = 0.0 if (float(np.min(f1_vol)) <= 0.0 <= float(np.max(f1_vol))) else eps1
            lvl2 = 0.0 if (float(np.min(f2_vol)) <= 0.0 <= float(np.max(f2_vol))) else eps2
            vol1 = f1_vol if lvl1 == 0.0 else np.abs(f1_vol)
            vol2 = f2_vol if lvl2 == 0.0 else np.abs(f2_vol)
            verts1, faces1, _, _ = measure.marching_cubes(vol1, level=lvl1, spacing=(dx, dy, dz))
            verts2, faces2, _, _ = measure.marching_cubes(vol2, level=lvl2, spacing=(dx, dy, dz))
            verts1 += np.array([mins[0], mins[1], mins[2]], dtype=np.float32)
            verts2 += np.array([mins[0], mins[1], mins[2]], dtype=np.float32)
            rendered = True
        except Exception:
            rendered = False

    p1 = pts[np.abs(f1) <= eps1]
    p2 = pts[np.abs(f2) <= eps2]
    if len(p1) > int(cfg.surface_max_points):
        p1 = p1[np.random.choice(len(p1), size=int(cfg.surface_max_points), replace=False)]
    if len(p2) > int(cfg.surface_max_points):
        p2 = p2[np.random.choice(len(p2), size=int(cfg.surface_max_points), replace=False)]

    if intersection_points is not None and len(intersection_points) > 0:
        cand = intersection_points.astype(np.float32)
        with torch.no_grad():
            fc = model(torch.from_numpy(cand).to(device))
            if fc.dim() == 1:
                fc = fc.unsqueeze(1)
            hc = torch.linalg.norm(fc[:, :2], dim=1).detach().cpu().numpy()
        p12 = cand[hc <= eps_h]
    else:
        p12 = pts[np.sqrt(f1 * f1 + f2 * f2) <= eps_h]
    learned_curve_max_points = int(getattr(cfg, "plot_learned_curve_max_points", 4096))
    if len(p12) > learned_curve_max_points:
        p12 = p12[np.random.choice(len(p12), size=learned_curve_max_points, replace=False)]

    gt_plot = None
    if dataset_name is not None:
        try:
            gt_plot = resolve_gt_grid(str(dataset_name), cfg, x_train=x_train)
        except Exception:
            gt_plot = None
    component_count = _curve_component_count(dataset_name)
    gt_components = _split_gt_components(gt_plot, component_count)
    learned_plot = None
    if gt_plot is not None and len(gt_plot) > 0 and bool(getattr(cfg, "plot_project_gt_curve", True)):
        try:
            proj_alpha, proj_steps, proj_min_steps = _proj_from_cfg(cfg)
            learned_plot, _ = project_points_with_steps_numpy(
                model,
                np.asarray(gt_plot, dtype=np.float32),
                device=str(device),
                proj_steps=int(proj_steps),
                proj_alpha=float(proj_alpha),
                proj_min_steps=int(proj_min_steps),
                f_abs_stop=eps_h,
            )
            learned_plot = np.asarray(learned_plot, dtype=np.float32)
        except Exception:
            learned_plot = None
    learned_plot_components = _split_gt_components(learned_plot, component_count)

    focus_twist_ellipse = str(dataset_name) == "3d_vz_2d_ellipse_traj"
    plot_mins = np.asarray(mins, dtype=np.float32).copy()
    plot_maxs = np.asarray(maxs, dtype=np.float32).copy()
    if focus_twist_ellipse:
        zoom_parts: list[np.ndarray] = []
        if gt_plot is not None and len(gt_plot) > 0:
            zoom_parts.append(np.asarray(gt_plot, dtype=np.float32)[:, :3])
        if learned_plot is not None and len(learned_plot) > 0:
            zoom_parts.append(np.asarray(learned_plot, dtype=np.float32)[:, :3])
        elif p12 is not None and len(p12) > 0:
            zoom_parts.append(np.asarray(p12, dtype=np.float32)[:, :3])
        if zoom_parts:
            zoom_pts = np.concatenate(zoom_parts, axis=0)
            zoom_pts = zoom_pts[np.isfinite(zoom_pts).all(axis=1)]
            if len(zoom_pts) > 0:
                zmin = zoom_pts.min(axis=0)
                zmax = zoom_pts.max(axis=0)
                span = np.maximum(zmax - zmin, 1e-6)
                pad = np.maximum(0.06 * span, 0.02 * float(np.max(span)))
                plot_mins = zmin - pad
                plot_maxs = zmax + pad

    compact = bool(getattr(cfg, "paper_plot_compact", False))
    show_legend = bool(getattr(cfg, "paper_plot_show_legend", True))
    is_linked_curves = str(dataset_name) == "linked_curves_3d"
    axis_fontsize = 6 if compact else 8
    tick_fontsize = 5 if compact else 7
    legend_fontsize = 5 if compact else 7
    tick_nbins = 3 if compact else 5
    true_lw = 2.2 if compact else 3.2
    learned_lw = 0.7 if compact else 0.85
    surface_edge_lw = 0.06 if compact else 0.1

    def _order_curve_points(points: np.ndarray) -> np.ndarray:
        if points.ndim != 2 or points.shape[0] < 3:
            return points
        center = np.mean(points, axis=0, keepdims=True)
        dist_to_center = np.linalg.norm(points - center, axis=1)
        start_idx = int(np.argmax(dist_to_center))
        remaining = np.ones(points.shape[0], dtype=bool)
        order = [start_idx]
        remaining[start_idx] = False
        current = start_idx
        for _ in range(points.shape[0] - 1):
            rest_idx = np.flatnonzero(remaining)
            if len(rest_idx) == 0:
                break
            d = np.linalg.norm(points[rest_idx] - points[current], axis=1)
            nxt = int(rest_idx[int(np.argmin(d))])
            order.append(nxt)
            remaining[nxt] = False
            current = nxt
        return points[np.asarray(order, dtype=np.int32)]

    def _draw_curve_segments(
        ax,
        points: np.ndarray,
        *,
        color: str,
        linewidth: float,
        alpha: float,
        label: str,
        linestyle: str | tuple[int, tuple[float, ...]] = "-",
        preserve_order: bool = False,
        scatter_only: bool = False,
        scatter_size: float = 1.6,
    ) -> None:
        if points is None or len(points) <= 1:
            return
        line = np.asarray(points, dtype=np.float32)
        if not preserve_order:
            line = _order_curve_points(line)
        first = True
        for seg in _split_polyline_on_jumps(line):
            if len(seg) <= 1:
                continue
            if scatter_only:
                ax.scatter(
                    seg[:, 0],
                    seg[:, 1],
                    seg[:, 2],
                    s=float(scatter_size),
                    c=color,
                    alpha=alpha,
                    label=label if first else None,
                    depthshade=False,
                )
            else:
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    seg[:, 2],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    linestyle=linestyle,
                    label=label if first else None,
                )
            first = False

    def _latex_axis_label(lbl: str) -> str:
        s = str(lbl).strip()
        # q1 -> q_1 for paper-style math labels.
        if len(s) >= 2 and s[0] in ("q", "x") and s[1:].isdigit():
            return rf"${s[0]}_{{{s[1:]}}}$"
        return s

    def _draw_common(ax) -> None:
        if focus_twist_ellipse:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
        else:
            ax.set_xlabel(_latex_axis_label(axis_labels[0]), fontsize=axis_fontsize, labelpad=0 if compact else 1)
            ax.set_ylabel(_latex_axis_label(axis_labels[1]), fontsize=axis_fontsize, labelpad=0 if compact else 1)
        # Place q3 label near the top end of the z-axis itself.
        ax.set_zlabel("")
        if not focus_twist_ellipse:
            z_pad = 0.03 * float(maxs[2] - mins[2])
            x_pad = 0.02 * float(maxs[0] - mins[0])
            y_pad = 0.02 * float(maxs[1] - mins[1])
            ax.text(
                float(maxs[0] + x_pad),
                float(maxs[1] + y_pad),
                float(maxs[2] + z_pad),
                _latex_axis_label(axis_labels[2]),
                fontsize=axis_fontsize,
            )
        # Pull tick labels closer to axes.
        ax.tick_params(labelsize=tick_fontsize, pad=-2 if compact else 0)
        ax.set_xlim(float(plot_mins[0]), float(plot_maxs[0]))
        ax.set_ylim(float(plot_mins[1]), float(plot_maxs[1]))
        ax.set_zlim(float(plot_mins[2]), float(plot_maxs[2]))
        # Increase tick density so ticks are not too sparse/far apart.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=tick_nbins))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_nbins))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=tick_nbins))
        if focus_twist_ellipse:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_box_aspect((1.0, 1.0, 0.72))
        ax.set_proj_type("persp")
        ax.view_init(elev=28, azim=-42)

    def _add_surface(ax, verts, faces, color: str, alpha: float, label: str) -> None:
        poly = Poly3DCollection(
            verts[faces],
            alpha=alpha,
            facecolor=color,
            edgecolor=(0, 0, 0, 0.08),
            linewidth=surface_edge_lw,
            label=label,
        )
        ax.add_collection3d(poly)

    # Match paper boxplot compact typography style.
    fig_size = (1.72, 1.44) if compact else (3.45, 2.9)
    cfg_fig_size = getattr(cfg, "paper_plot_figsize", None)
    if cfg_fig_size is not None:
        try:
            fig_size = (float(cfg_fig_size[0]), float(cfg_fig_size[1]))
        except Exception:
            fig_size = (1.72, 1.44) if compact else fig_size
    elif str(dataset_name) == "3d_vz_2d_ellipse_traj":
        fig_size = (1.55, 1.30) if compact else (3.05, 2.55)

    with plt.rc_context(
        {
            "font.size": axis_fontsize,
            "axes.labelsize": axis_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "legend.fontsize": legend_fontsize,
        }
    ):
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection="3d")
        h1_label = r"$h_{\theta}^{(1)}=0$" if compact else r"$h_{\theta}^{(1)} = 0$"
        h2_label = r"$h_{\theta}^{(2)}=0$" if compact else r"$h_{\theta}^{(2)} = 0$"
        int_label = "Learned" if compact else "Learned constraint"
        gt_label = "True" if compact else "True equality constraint"

        if rendered and verts1 is not None and faces1 is not None and verts2 is not None and faces2 is not None:
            surface_alpha = 0.18 if focus_twist_ellipse else 0.12
            _add_surface(ax, verts1, faces1, "#22d3ee", surface_alpha, h1_label)
            _add_surface(ax, verts2, faces2, "#f472b6", surface_alpha, h2_label)
        else:
            if len(p1) > 0:
                ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=0.9, c="#22d3ee", alpha=0.20, label=h1_label)
            if len(p2) > 0:
                ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=0.9, c="#f472b6", alpha=0.20, label=h2_label)

        if gt_components:
            first_gt = True
            gt_colors = ["#2563eb", "#111827"] if is_linked_curves else ["#374151"]
            for comp_idx, comp in enumerate(gt_components):
                if len(comp) <= 1:
                    continue
                gt_color = gt_colors[comp_idx % len(gt_colors)]
                gt_comp_label = gt_label
                if is_linked_curves and len(gt_components) > 1:
                    gt_comp_label = f"True constraint branch {comp_idx + 1}"
                _draw_curve_segments(
                    ax,
                    comp,
                    color=gt_color,
                    linewidth=true_lw,
                    alpha=0.92,
                    label=gt_comp_label if (first_gt or is_linked_curves) else None,
                    preserve_order=True,
                )
                first_gt = False

        if learned_plot_components:
            learned_scatter_only = bool(getattr(cfg, "plot_learned_curve_scatter_only", False))
            first_learned = True
            for comp in learned_plot_components:
                if len(comp) <= 1:
                    continue
                _draw_curve_segments(
                    ax,
                    comp,
                    color="#ef4444",
                    linewidth=learned_lw,
                    alpha=0.98,
                    linestyle=(0, (2.4, 1.4)),
                    label=int_label if first_learned else None,
                    preserve_order=True,
                    scatter_only=learned_scatter_only,
                )
                first_learned = False
        elif len(p12) > 0:
            learned_components = _assign_points_to_gt_components(p12, gt_components)
            first_learned = True
            for comp in learned_components:
                if len(comp) <= 1:
                    continue
                _draw_curve_segments(
                    ax,
                    comp,
                    color="#ef4444",
                    linewidth=learned_lw,
                    alpha=0.98,
                    linestyle=(0, (2.4, 1.4)),
                    label=int_label if first_learned else None,
                )
                first_learned = False

        _draw_common(ax)

        # No title for paper style.
        if show_legend:
            if compact:
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.47, 1.02),
                    ncol=2,
                    fontsize=legend_fontsize,
                    frameon=False,
                    borderaxespad=0.0,
                    columnspacing=0.75,
                    handlelength=1.15,
                    handletextpad=0.25,
                )
            else:
                if focus_twist_ellipse:
                    handles = [
                        Patch(facecolor="#0891b2", edgecolor="#0891b2", alpha=0.48, label=h1_label),
                        Patch(facecolor="#db2777", edgecolor="#db2777", alpha=0.48, label=h2_label),
                        Line2D([0], [0], color="#4b5563", linewidth=true_lw, label=gt_label),
                        Line2D(
                            [0],
                            [0],
                            color="#ef4444",
                            linewidth=learned_lw,
                            linestyle=(0, (2.4, 1.4)),
                            label=int_label,
                        ),
                    ]
                    ax.legend(
                        handles=handles,
                        loc="upper left",
                        fontsize=legend_fontsize,
                        frameon=False,
                    )
                elif is_linked_curves:
                    handles = [
                        Patch(facecolor="#22d3ee", edgecolor="#22d3ee", alpha=0.24, label=h1_label),
                        Patch(facecolor="#f472b6", edgecolor="#f472b6", alpha=0.24, label=h2_label),
                        Line2D([0], [0], color="none", linewidth=0, label=""),
                        Line2D([0], [0], color="#2563eb", linewidth=true_lw, label="True constraint branch 1"),
                        Line2D([0], [0], color="#111827", linewidth=true_lw, label="True constraint branch 2"),
                        Line2D(
                            [0],
                            [0],
                            color="#ef4444",
                            linewidth=learned_lw,
                            linestyle=(0, (2.4, 1.4)),
                            label=int_label,
                        ),
                    ]
                    ax.legend(
                        handles=handles,
                        loc="upper left",
                        ncol=2,
                        fontsize=legend_fontsize,
                        frameon=False,
                        columnspacing=0.7,
                        handlelength=1.4,
                        handletextpad=0.35,
                        borderaxespad=0.0,
                    )
                else:
                    ax.legend(loc="upper left", fontsize=legend_fontsize, frameon=False)

        # Keep right margin so z-label is visible.
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.998)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


def _plot_highdim_pca(
    x_train: np.ndarray,
    traj: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    # Lightweight fallback for data_dim > 3: visualize train manifold and trajectories in PCA-2D.
    x = x_train.astype(np.float32)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:2].T  # (D,2)
    emb_train = (x - mu) @ basis

    x0 = traj[0]
    xT = traj[-1]
    emb_start = (x0 - mu) @ basis
    emb_end = (xT - mu) @ basis

    plt.figure(figsize=(7.0, 6.0))
    if len(emb_train) > 2500:
        idx = np.random.choice(len(emb_train), size=2500, replace=False)
        emb_train = emb_train[idx]
    plt.scatter(emb_train[:, 0], emb_train[:, 1], s=5, c="gray", alpha=0.25, label="train")

    step = max(1, int(np.ceil(traj.shape[1] / 32)))
    for i in range(0, traj.shape[1], step):
        tr = (traj[:, i, :] - mu) @ basis
        plt.plot(tr[:, 0], tr[:, 1], "-", color="green", lw=0.9, alpha=0.7)
    plt.scatter(emb_start[:, 0], emb_start[:, 1], s=12, c="royalblue", alpha=0.9, label="traj start")
    plt.scatter(emb_end[:, 0], emb_end[:, 1], s=12, c="crimson", alpha=0.9, label="traj end")
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.title(title + " (PCA-2D)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _rpy_zyx_to_local_z(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    # R = Rz(yaw) Ry(pitch) Rx(roll), local z-axis in world is R[:,2].
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    z_x = cy * sp * cr + sy * sr
    z_y = sy * sp * cr - cy * sr
    z_z = cp * cr
    z = np.stack([z_x, z_y, z_z], axis=1).astype(np.float32)
    z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return z


def _plot_workspace_pose_orientation_3d(
    x_train: np.ndarray,
    eval_proj: np.ndarray,
    out_path: str,
    title: str,
    surface_cfg: Any | None = None,
) -> None:
    # eval_proj expected shape (N,6): [x,y,z,roll,pitch,yaw]
    if eval_proj is None or len(eval_proj) == 0 or eval_proj.shape[1] < 6:
        return
    pts = eval_proj[:, :3].astype(np.float32)
    rpy = eval_proj[:, 3:6].astype(np.float32)
    dirs = _rpy_zyx_to_local_z(rpy[:, 0], rpy[:, 1], rpy[:, 2])

    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    train_plot = x_train[:, :3] if x_train.shape[1] >= 3 else x_train
    if len(train_plot) > 0:
        xmin, xmax = float(np.min(train_plot[:, 0])), float(np.max(train_plot[:, 0]))
        ymin, ymax = float(np.min(train_plot[:, 1])), float(np.max(train_plot[:, 1]))
    else:
        xmin = float(np.min(pts[:, 0]))
        xmax = float(np.max(pts[:, 0]))
        ymin = float(np.min(pts[:, 1]))
        ymax = float(np.max(pts[:, 1]))
    pad_x = max(0.05, 0.08 * max(xmax - xmin, 1e-3))
    pad_y = max(0.05, 0.08 * max(ymax - ymin, 1e-3))
    xx, yy = np.meshgrid(
        np.linspace(xmin - pad_x, xmax + pad_x, 80),
        np.linspace(ymin - pad_y, ymax + pad_y, 80),
    )
    zz, _ = sine_surface_z_and_normal_from_xy(xx, yy, cfg=surface_cfg)
    ax.plot_surface(xx, yy, zz, rstride=2, cstride=2, alpha=0.18, linewidth=0.0, color="#22d3ee")

    if len(train_plot) > 2500:
        idx = np.random.choice(len(train_plot), size=2500, replace=False)
        train_plot = train_plot[idx]
    ax.scatter(train_plot[:, 0], train_plot[:, 1], train_plot[:, 2], s=4, c="gray", alpha=0.20, label="train")

    m = min(len(pts), 500)
    if len(pts) > m:
        idx = np.random.choice(len(pts), size=m, replace=False)
        pts_q = pts[idx]
        dirs_q = dirs[idx]
    else:
        pts_q = pts
        dirs_q = dirs
    ax.scatter(pts_q[:, 0], pts_q[:, 1], pts_q[:, 2], s=8, c="#ef4444", alpha=0.75, label="eval proj")

    ax.quiver(
        pts_q[:, 0],
        pts_q[:, 1],
        pts_q[:, 2],
        dirs_q[:, 0],
        dirs_q[:, 1],
        dirs_q[:, 2],
        length=0.20,
        normalize=True,
        color="#111827",
        linewidths=0.7,
        alpha=0.75,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    x_all = np.concatenate([pts[:, 0], xx.reshape(-1)], axis=0)
    y_all = np.concatenate([pts[:, 1], yy.reshape(-1)], axis=0)
    ax.set_xlim(float(np.min(x_all) - 0.03), float(np.max(x_all) + 0.03))
    ax.set_ylim(float(np.min(y_all) - 0.03), float(np.max(y_all) + 0.03))
    z_all = np.concatenate([pts[:, 2], zz.reshape(-1)], axis=0)
    pad = 0.03
    ax.set_zlim(float(np.min(z_all) - pad), float(np.max(z_all) + pad))
    ax.view_init(elev=24, azim=-52)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_workspace_pose_projection_error_distributions(
    x_before: np.ndarray,
    x_after: np.ndarray,
    out_path: str,
    title: str,
    surface_cfg: Any | None = None,
) -> None:
    # Inputs expected shape (N,6): [x,y,z,roll,pitch,yaw].
    if (
        x_before is None
        or x_after is None
        or len(x_before) == 0
        or len(x_after) == 0
        or x_before.shape[1] < 6
        or x_after.shape[1] < 6
    ):
        return

    n = int(min(len(x_before), len(x_after)))
    xb = x_before[:n].astype(np.float32, copy=False)
    xa = x_after[:n].astype(np.float32, copy=False)
    finite = np.isfinite(xb).all(axis=1) & np.isfinite(xa).all(axis=1)
    if not np.any(finite):
        return
    xb = xb[finite]
    xa = xa[finite]

    zb_true, nb_true = sine_surface_z_and_normal_from_xy(xb[:, 0], xb[:, 1], cfg=surface_cfg)
    za_true, na_true = sine_surface_z_and_normal_from_xy(xa[:, 0], xa[:, 1], cfg=surface_cfg)
    pos_err_before = np.abs(xb[:, 2] - zb_true)
    pos_err_after = np.abs(xa[:, 2] - za_true)

    zb_axis = _rpy_zyx_to_local_z(xb[:, 3], xb[:, 4], xb[:, 5])
    za_axis = _rpy_zyx_to_local_z(xa[:, 3], xa[:, 4], xa[:, 5])
    cos_b = np.clip(np.sum(zb_axis * nb_true, axis=1), -1.0, 1.0)
    cos_a = np.clip(np.sum(za_axis * na_true, axis=1), -1.0, 1.0)
    ang_err_before = np.degrees(np.arccos(cos_b))
    ang_err_after = np.degrees(np.arccos(cos_a))

    pos_cap = float(np.percentile(np.concatenate([pos_err_before, pos_err_after], axis=0), 99))
    pos_cap = max(pos_cap, 1e-4)
    pos_bins = np.linspace(0.0, pos_cap, 50)

    ang_cap = float(np.percentile(np.concatenate([ang_err_before, ang_err_after], axis=0), 99))
    ang_cap = max(ang_cap, 1.0)
    ang_bins = np.linspace(0.0, ang_cap, 50)

    fig = plt.figure(figsize=(10.0, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(pos_err_before, bins=pos_bins, color="#64748b", alpha=0.72, label="before")
    ax1.hist(pos_err_after, bins=pos_bins, color="#16a34a", alpha=0.58, label="after")
    ax1.set_xlabel("position distance to true manifold")
    ax1.set_ylabel("count")
    ax1.set_title("|z - z_true(x,y)|")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(ang_err_before, bins=ang_bins, color="#64748b", alpha=0.72, label="before")
    ax2.hist(ang_err_after, bins=ang_bins, color="#16a34a", alpha=0.58, label="after")
    ax2.set_xlabel("orientation angle error (deg)")
    ax2.set_title("angle(local z-axis, true normal)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(
        "[workspace_pose_err] "
        f"pos_mean: {float(np.mean(pos_err_before)):.5f}->{float(np.mean(pos_err_after)):.5f}, "
        f"ang_mean_deg: {float(np.mean(ang_err_before)):.3f}->{float(np.mean(ang_err_after)):.3f}"
    )
    print(f"saved: {out_path}")


def _rpy_zyx_to_rotmat_batch(rpy: np.ndarray) -> np.ndarray:
    rr = rpy[:, 0].astype(np.float32)
    pp = rpy[:, 1].astype(np.float32)
    yy = rpy[:, 2].astype(np.float32)
    cr = np.cos(rr)
    sr = np.sin(rr)
    cp = np.cos(pp)
    sp = np.sin(pp)
    cy = np.cos(yy)
    sy = np.sin(yy)

    R = np.zeros((len(rpy), 3, 3), dtype=np.float32)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R.astype(np.float32)


def _rotation_geodesic_deg_batch(rpy_a: np.ndarray, rpy_b: np.ndarray) -> np.ndarray:
    Ra = _rpy_zyx_to_rotmat_batch(rpy_a.astype(np.float32))
    Rb = _rpy_zyx_to_rotmat_batch(rpy_b.astype(np.float32))
    Rrel = np.einsum("nij,njk->nik", np.transpose(Ra, (0, 2, 1)), Rb)
    tr = Rrel[:, 0, 0] + Rrel[:, 1, 1] + Rrel[:, 2, 2]
    cosv = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosv)).astype(np.float32)


def _dual_arm_curve_center_tnb_from_s(
    s: np.ndarray,
    *,
    grasp_span: float,
    x_span: float,
    y_amp: float,
    y_freq: float,
    z_base: float,
    z_amp: float,
    z_freq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del grasp_span, z_amp, z_freq
    ss = s.astype(np.float32)
    x = (float(x_span) * ss).astype(np.float32)
    y = (float(y_amp) * np.sin(float(y_freq) * np.pi * ss)).astype(np.float32)
    z = np.full_like(ss, float(z_base), dtype=np.float32)
    dx = np.full_like(ss, float(x_span), dtype=np.float32)
    dy = (float(y_amp) * float(y_freq) * np.pi * np.cos(float(y_freq) * np.pi * ss)).astype(np.float32)
    dz = np.zeros_like(ss, dtype=np.float32)

    center = np.stack([x, y, z], axis=1).astype(np.float32)
    tang = np.stack([dx, dy, dz], axis=1).astype(np.float32)
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)

    z_axis = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(ss), 1))
    y_axis = np.cross(z_axis, tang).astype(np.float32)
    y_axis /= (np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-12)
    z_axis = np.cross(tang, y_axis).astype(np.float32)
    z_axis /= (np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-12)
    return center.astype(np.float32), tang.astype(np.float32), y_axis.astype(np.float32), z_axis.astype(np.float32)


def _dual_arm_pose_analytic_target(
    x: np.ndarray,
    *,
    grasp_span: float,
    x_span: float,
    y_amp: float,
    y_freq: float,
    z_base: float,
    z_amp: float,
    z_freq: float,
    z_half_range: float | None = None,
) -> np.ndarray:
    xx = x.astype(np.float32)
    center_obs = (0.5 * (xx[:, 0:3] + xx[:, 6:9])).astype(np.float32)
    s_grid = np.linspace(-1.0, 1.0, 2048, dtype=np.float32)
    center_grid, _, _, _ = _dual_arm_curve_center_tnb_from_s(
        s_grid,
        grasp_span=grasp_span,
        x_span=x_span,
        y_amp=y_amp,
        y_freq=y_freq,
        z_base=z_base,
        z_amp=z_amp,
        z_freq=z_freq,
    )
    d2 = np.sum((center_obs[:, None, 0:2] - center_grid[None, :, 0:2]) ** 2, axis=2)
    s_star = s_grid[np.argmin(d2, axis=1)].astype(np.float32)

    center, tang, y_axis, z_axis = _dual_arm_curve_center_tnb_from_s(
        s_star,
        grasp_span=grasp_span,
        x_span=x_span,
        y_amp=y_amp,
        y_freq=y_freq,
        z_base=z_base,
        z_amp=z_amp,
        z_freq=z_freq,
    )
    if z_half_range is None:
        z_half_range = float(z_amp)
    u_star = np.clip(center_obs[:, 2] - float(z_base), -float(z_half_range), float(z_half_range))
    center[:, 2] = float(z_base) + u_star.astype(np.float32)

    R = np.stack([tang, y_axis, z_axis], axis=2).astype(np.float32)
    sy = -np.clip(R[:, 2, 0], -1.0, 1.0)
    pitch = np.arcsin(sy).astype(np.float32)
    cp = np.cos(pitch)
    roll = np.where(np.abs(cp) > 1e-8, np.arctan2(R[:, 2, 1], R[:, 2, 2]), 0.0).astype(np.float32)
    yaw = np.where(
        np.abs(cp) > 1e-8,
        np.arctan2(R[:, 1, 0], R[:, 0, 0]),
        np.arctan2(-R[:, 0, 1], R[:, 1, 1]),
    ).astype(np.float32)
    rpy = np.stack([roll, pitch, yaw], axis=1).astype(np.float32)

    offset = (0.5 * float(grasp_span) * tang).astype(np.float32)
    pose_1 = np.concatenate([center - offset, rpy], axis=1).astype(np.float32)
    pose_2 = np.concatenate([center + offset, rpy], axis=1).astype(np.float32)
    return np.concatenate([pose_1, pose_2], axis=1).astype(np.float32)


def _dual_arm_pose_task_frame_embed(x: np.ndarray) -> np.ndarray:
    xx = x.astype(np.float32, copy=False)

    def _pose_embed(pose: np.ndarray) -> np.ndarray:
        pos = pose[:, :3].astype(np.float32)
        R = _rpy_zyx_to_rotmat_batch(pose[:, 3:6].astype(np.float32))
        return np.concatenate([pos, R[:, :, 0], R[:, :, 1], R[:, :, 2]], axis=1).astype(np.float32)

    return np.concatenate([_pose_embed(xx[:, :6]), _pose_embed(xx[:, 6:12])], axis=1).astype(np.float32)


def _plot_dual_arm_pose_projection_error_distributions(
    x_before: np.ndarray,
    x_after: np.ndarray,
    out_path: str,
    title: str,
    *,
    grasp_span: float,
    x_span: float,
    y_amp: float,
    y_freq: float,
    z_base: float,
    z_amp: float,
    z_freq: float,
    z_half_range: float | None = None,
) -> None:
    if (
        x_before is None
        or x_after is None
        or len(x_before) == 0
        or len(x_after) == 0
        or x_before.shape[1] < 12
        or x_after.shape[1] < 12
    ):
        return

    n = int(min(len(x_before), len(x_after)))
    xb = x_before[:n].astype(np.float32, copy=False)
    xa = x_after[:n].astype(np.float32, copy=False)
    finite = np.isfinite(xb).all(axis=1) & np.isfinite(xa).all(axis=1)
    if not np.any(finite):
        return
    xb = xb[finite]
    xa = xa[finite]

    tb = _dual_arm_pose_analytic_target(
        xb,
        grasp_span=grasp_span,
        x_span=x_span,
        y_amp=y_amp,
        y_freq=y_freq,
        z_base=z_base,
        z_amp=z_amp,
        z_freq=z_freq,
        z_half_range=z_half_range,
    )
    ta = _dual_arm_pose_analytic_target(
        xa,
        grasp_span=grasp_span,
        x_span=x_span,
        y_amp=y_amp,
        y_freq=y_freq,
        z_base=z_base,
        z_amp=z_amp,
        z_freq=z_freq,
        z_half_range=z_half_range,
    )

    pos_b1 = np.linalg.norm(xb[:, 0:3] - tb[:, 0:3], axis=1).astype(np.float32)
    pos_b2 = np.linalg.norm(xb[:, 6:9] - tb[:, 6:9], axis=1).astype(np.float32)
    pos_a1 = np.linalg.norm(xa[:, 0:3] - ta[:, 0:3], axis=1).astype(np.float32)
    pos_a2 = np.linalg.norm(xa[:, 6:9] - ta[:, 6:9], axis=1).astype(np.float32)
    pos_err_before = (0.5 * (pos_b1 + pos_b2)).astype(np.float32)
    pos_err_after = (0.5 * (pos_a1 + pos_a2)).astype(np.float32)

    ang_b1 = _rotation_geodesic_deg_batch(xb[:, 3:6], tb[:, 3:6])
    ang_b2 = _rotation_geodesic_deg_batch(xb[:, 9:12], tb[:, 9:12])
    ang_a1 = _rotation_geodesic_deg_batch(xa[:, 3:6], ta[:, 3:6])
    ang_a2 = _rotation_geodesic_deg_batch(xa[:, 9:12], ta[:, 9:12])
    ang_err_before = (0.5 * (ang_b1 + ang_b2)).astype(np.float32)
    ang_err_after = (0.5 * (ang_a1 + ang_a2)).astype(np.float32)

    vec_err_before = np.linalg.norm(
        _dual_arm_pose_task_frame_embed(xb) - _dual_arm_pose_task_frame_embed(tb),
        axis=1,
    ).astype(np.float32)
    vec_err_after = np.linalg.norm(
        _dual_arm_pose_task_frame_embed(xa) - _dual_arm_pose_task_frame_embed(ta),
        axis=1,
    ).astype(np.float32)

    pos_cap = float(np.percentile(np.concatenate([pos_err_before, pos_err_after], axis=0), 99))
    pos_cap = max(pos_cap, 1e-4)
    pos_bins = np.linspace(0.0, pos_cap, 50)

    ang_cap = float(np.percentile(np.concatenate([ang_err_before, ang_err_after], axis=0), 99))
    ang_cap = max(ang_cap, 1.0)
    ang_bins = np.linspace(0.0, ang_cap, 50)

    vec_cap = float(np.percentile(np.concatenate([vec_err_before, vec_err_after], axis=0), 99))
    vec_cap = max(vec_cap, 1e-4)
    vec_bins = np.linspace(0.0, vec_cap, 50)

    fig = plt.figure(figsize=(14.4, 4.2))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.hist(pos_err_before, bins=pos_bins, color="#64748b", alpha=0.72, label="before")
    ax1.hist(pos_err_after, bins=pos_bins, color="#16a34a", alpha=0.58, label="after")
    ax1.set_xlabel("mean EE position error")
    ax1.set_ylabel("count")
    ax1.set_title("mean(||p_i - p_i,true||), i in {1,2}")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(ang_err_before, bins=ang_bins, color="#64748b", alpha=0.72, label="before")
    ax2.hist(ang_err_after, bins=ang_bins, color="#16a34a", alpha=0.58, label="after")
    ax2.set_xlabel("mean EE orientation error (deg)")
    ax2.set_title("mean geodesic angle to true shared pose")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(vec_err_before, bins=vec_bins, color="#64748b", alpha=0.72, label="before")
    ax3.hist(vec_err_after, bins=vec_bins, color="#16a34a", alpha=0.58, label="after")
    ax3.set_xlabel("analytic embedding L2 error")
    ax3.set_title("evaluator-style vector distance")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(
        "[dual_arm_pose_err] "
        f"pos_mean: {float(np.mean(pos_err_before)):.5f}->{float(np.mean(pos_err_after)):.5f}, "
        f"ang_mean_deg: {float(np.mean(ang_err_before)):.3f}->{float(np.mean(ang_err_after)):.3f}, "
        f"vec_mean: {float(np.mean(vec_err_before)):.5f}->{float(np.mean(vec_err_after)):.5f}"
    )
    print(f"saved: {out_path}")


def _plot_dual_arm_guided_insertion_orientation_3d(
    x_train: np.ndarray,
    eval_proj: np.ndarray,
    out_path: str,
    title: str,
    task_cfg: Any | None = None,
) -> None:
    if eval_proj is None or len(eval_proj) == 0 or eval_proj.shape[1] < 12:
        return

    xx = eval_proj.astype(np.float32, copy=False)
    p1 = xx[:, 0:3].astype(np.float32)
    p2 = xx[:, 6:9].astype(np.float32)
    center = (0.5 * (p1 + p2)).astype(np.float32)

    fig = plt.figure(figsize=(9.2, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    if x_train is not None and len(x_train) > 0 and x_train.shape[1] >= 12:
        train_center = (0.5 * (x_train[:, 0:3] + x_train[:, 6:9])).astype(np.float32)
        if len(train_center) > 3000:
            idx = np.random.choice(len(train_center), size=3000, replace=False)
            train_center = train_center[idx]
        ax.scatter(
            train_center[:, 0],
            train_center[:, 1],
            train_center[:, 2],
            s=4,
            c="#94a3b8",
            alpha=0.18,
            label="train object centers",
        )

    # eval_proj is an unordered projected sample cloud, so show it as scatter only.
    if len(center) > 1200:
        idx_eval = np.random.choice(len(center), size=1200, replace=False)
        center_s = center[idx_eval]
    else:
        center_s = center

    ax.scatter(center_s[:, 0], center_s[:, 1], center_s[:, 2], s=9, c="#111827", alpha=0.36, label="projected centers")

    # Overlay the vertical-ribbon center manifold.
    s_true = np.linspace(-1.0, 1.0, 240, dtype=np.float32)
    center_mid, tang_true, _y_true, z_true = _dual_arm_curve_center_tnb_from_s(
        s_true,
        grasp_span=1.0,
        x_span=float(getattr(task_cfg, "dual_arm_curve_x_span", 0.45)),
        y_amp=float(getattr(task_cfg, "dual_arm_curve_y_amp", 0.12)),
        y_freq=float(getattr(task_cfg, "dual_arm_curve_y_freq", 1.0)),
        z_base=float(getattr(task_cfg, "dual_arm_curve_z_base", 0.68)),
        z_amp=float(getattr(task_cfg, "dual_arm_curve_z_amp", getattr(task_cfg, "dual_arm_vertical_half_range", 0.12))),
        z_freq=float(getattr(task_cfg, "dual_arm_curve_z_freq", 0.7)),
    )
    z_half_range = float(getattr(task_cfg, "dual_arm_vertical_half_range", getattr(task_cfg, "dual_arm_curve_z_amp", 0.12)))
    center_low = center_mid.copy()
    center_low[:, 2] = center_mid[:, 2] - z_half_range
    center_high = center_mid.copy()
    center_high[:, 2] = center_mid[:, 2] + z_half_range

    ax.plot(center_mid[:, 0], center_mid[:, 1], center_mid[:, 2], "--", color="#0f172a", lw=2.0, alpha=0.95, label="centerline")
    ax.plot(center_low[:, 0], center_low[:, 1], center_low[:, 2], "-", color="#64748b", lw=1.2, alpha=0.65, label="ribbon z bounds")
    ax.plot(center_high[:, 0], center_high[:, 1], center_high[:, 2], "-", color="#64748b", lw=1.2, alpha=0.65)

    true_idx = np.linspace(0, len(center_mid) - 1, 14, dtype=int)
    ax.quiver(
        center_mid[true_idx, 0], center_mid[true_idx, 1], center_mid[true_idx, 2],
        tang_true[true_idx, 0], tang_true[true_idx, 1], tang_true[true_idx, 2],
        length=0.08, normalize=True, color="#2563eb", linewidths=0.8, alpha=0.85,
    )

    pts_all = np.concatenate([center_s, center_low, center_high, center_mid], axis=0)
    mins = np.min(pts_all, axis=0)
    maxs = np.max(pts_all, axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    pad = 0.15 * span
    ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    ax.set_zlim(float(mins[2] - pad[2]), float(maxs[2] + pad[2]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-56)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ur5_eval_projection_workspace_orientation_3d(
    q_train: np.ndarray,
    q_eval_proj: np.ndarray,
    out_path: str,
    title: str,
    use_pybullet_n6: bool,
) -> None:
    if q_eval_proj is None or len(q_eval_proj) == 0 or q_eval_proj.shape[1] < 6:
        return

    def _ee_and_dir(q: np.ndarray, use_pybullet: bool) -> tuple[np.ndarray, np.ndarray]:
        joints = spatial_fk(q.astype(np.float32), list(UR5_LINK_LENGTHS), use_pybullet_n6=use_pybullet)
        ee = joints[:, -1, :].astype(np.float32)
        d = spatial_tool_axis_n6(q.astype(np.float32), use_pybullet=use_pybullet).astype(np.float32)
        return ee, d

    qtr = q_train.astype(np.float32)
    qpr = q_eval_proj.astype(np.float32)

    try:
        tr_pos, tr_dir = _ee_and_dir(qtr, use_pybullet=bool(use_pybullet_n6))
        pr_pos, pr_dir = _ee_and_dir(qpr, use_pybullet=bool(use_pybullet_n6))
    except Exception as e:
        # Keep visualization available even when pybullet backend is unavailable in headless env.
        print(f"[warn] UR5 workspace vis fallback to analytic kinematics: {e}")
        tr_pos, tr_dir = _ee_and_dir(qtr, use_pybullet=False)
        pr_pos, pr_dir = _ee_and_dir(qpr, use_pybullet=False)

    if len(tr_pos) > 1400:
        idx = np.random.choice(len(tr_pos), size=1400, replace=False)
        tr_pos = tr_pos[idx]
        tr_dir = tr_dir[idx]
    if len(pr_pos) > 900:
        idx = np.random.choice(len(pr_pos), size=900, replace=False)
        pr_pos = pr_pos[idx]
        pr_dir = pr_dir[idx]

    fig = plt.figure(figsize=(10.4, 4.8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.scatter(tr_pos[:, 0], tr_pos[:, 1], tr_pos[:, 2], s=4, c="#9ca3af", alpha=0.20, label="train workspace")
    ax1.scatter(pr_pos[:, 0], pr_pos[:, 1], pr_pos[:, 2], s=8, c="#ef4444", alpha=0.72, label="eval proj workspace")
    qstep = max(1, len(pr_pos) // 90)
    qidx = np.arange(0, len(pr_pos), qstep, dtype=int)
    if len(qidx) > 0 and qidx[-1] != len(pr_pos) - 1:
        qidx = np.concatenate([qidx, np.array([len(pr_pos) - 1], dtype=int)])
    if len(qidx) > 0:
        ax1.quiver(
            pr_pos[qidx, 0], pr_pos[qidx, 1], pr_pos[qidx, 2],
            pr_dir[qidx, 0], pr_dir[qidx, 1], pr_dir[qidx, 2],
            length=0.18, normalize=True, color="#111827", linewidths=0.7, alpha=0.75
        )
    all_pos = np.concatenate([tr_pos, pr_pos], axis=0)
    mins = np.min(all_pos, axis=0)
    maxs = np.max(all_pos, axis=0)
    ctr = 0.5 * (mins + maxs)
    half = 0.55 * float(max(np.max(maxs - mins), 1e-3))
    ax1.set_xlim(float(ctr[0] - half), float(ctr[0] + half))
    ax1.set_ylim(float(ctr[1] - half), float(ctr[1] + half))
    ax1.set_zlim(float(ctr[2] - half), float(ctr[2] + half))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("Workspace Position + Tool Axis")
    ax1.legend(loc="best", fontsize=8)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ax2.scatter(tr_dir[:, 0], tr_dir[:, 1], tr_dir[:, 2], s=5, c="#9ca3af", alpha=0.22, label="train tool axis")
    ax2.scatter(pr_dir[:, 0], pr_dir[:, 1], pr_dir[:, 2], s=8, c="#ef4444", alpha=0.75, label="eval proj tool axis")
    ax2.quiver([0.0], [0.0], [0.0], [up[0]], [up[1]], [up[2]], color="#16a34a", linewidths=2.0, length=1.0)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_zlim(-1.05, 1.05)
    ax2.set_xlabel("dx")
    ax2.set_ylabel("dy")
    ax2.set_zlabel("dz")
    ax2.set_title("Tool Axis on Unit Sphere")
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _plot_planar_arm_planning(
    model: nn.Module,
    name: str,
    x_train: np.ndarray,
    out_path: str,
    cfg: Any,
    render_pybullet: bool = True,
) -> list[np.ndarray]:
    from planning.planner import _plot_planar_arm_planning as _core_plot_planar_arm_planning

    return _core_plot_planar_arm_planning(
        model=model,
        name=name,
        x_train=x_train,
        out_path=out_path,
        cfg=cfg,
        render_pybullet=render_pybullet,
    )
def _plot_projection_value_distribution(
    model: nn.Module,
    x_train: np.ndarray,
    cfg: Any,
    out_path: str,
    use_pybullet_n6: bool,
) -> None:
    # Same noisy samples before/after projection; compare orientation angle error distributions.
    mins, maxs = eval_bounds_from_train(x_train, cfg)
    span = np.maximum(maxs - mins, 1e-6).astype(np.float32)
    n0 = max(180, N6_WORKSPACE_VIS_POINTS_DEFAULT * 3)
    idx = np.random.randint(0, len(x_train), size=n0)
    x0 = x_train[idx].astype(np.float32).copy()
    noise_std = float(max(cfg.oncl_near_std_ratio, 1e-4))
    x0 = x0 + np.random.randn(*x0.shape).astype(np.float32) * (noise_std * span.reshape(1, -1))
    x0 = np.clip(x0, mins.reshape(1, -1), maxs.reshape(1, -1)).astype(np.float32)

    with torch.no_grad():
        f_on = model(torch.from_numpy(x_train.astype(np.float32)).to(cfg.device))
        if f_on.dim() == 1:
            f_on = f_on.unsqueeze(1)
        h_on = torch.linalg.norm(f_on, dim=1).detach().cpu().numpy()
    eps_q = float(getattr(cfg, "zero_eps_quantile", ZERO_EPS_QUANTILE_DEFAULT))
    eps_stop = float(np.percentile(np.abs(h_on), eps_q))

    proj_alpha, proj_steps, proj_min_steps = _proj_from_cfg(cfg)
    traj = project_trajectory_numpy(
        model,
        x0,
        device=str(cfg.device),
        proj_steps=int(proj_steps),
        proj_alpha=float(proj_alpha),
        proj_min_steps=int(proj_min_steps),
        f_abs_stop=eps_stop,
    )
    q_end = traj[-1].astype(np.float32)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    a0 = spatial_tool_axis_n6(x0.astype(np.float32), use_pybullet=use_pybullet_n6)
    a1 = spatial_tool_axis_n6(q_end.astype(np.float32), use_pybullet=use_pybullet_n6)
    c0 = np.clip(np.sum(a0 * up.reshape(1, 3), axis=1), -1.0, 1.0)
    c1 = np.clip(np.sum(a1 * up.reshape(1, 3), axis=1), -1.0, 1.0)
    ang0 = np.degrees(np.arccos(c0))
    ang1 = np.degrees(np.arccos(c1))

    cap = float(np.percentile(np.concatenate([ang0, ang1], axis=0), 99))
    cap = max(cap, 1.0)
    bins = np.linspace(0.0, cap, 50)
    fig = plt.figure(figsize=(9.2, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(ang0, bins=bins, color="#64748b", alpha=0.9)
    ax1.set_title("before projection")
    ax1.set_xlabel("orientation error (deg)")
    ax1.set_ylabel("count")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(ang1, bins=bins, color="#16a34a", alpha=0.9)
    ax2.set_title("after projection")
    ax2.set_xlabel("orientation error (deg)")
    ax2.grid(alpha=0.25)

    fig.suptitle("Noisy samples: orientation-angle error before/after projection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(
        f"[proj_err] before mean={float(np.mean(ang0)):.3f} deg, after mean={float(np.mean(ang1)):.3f} deg"
    )
    print(f"saved: {out_path}")


def _plot_ur5_projection_error_distribution_from_pairs(
    q_eval: np.ndarray,
    q_eval_proj: np.ndarray,
    out_path: str,
    title: str,
    use_pybullet_n6: bool,
) -> None:
    # Compare orientation angle to +z before/after projection using matched eval pairs.
    if (
        q_eval is None
        or q_eval_proj is None
        or len(q_eval) == 0
        or len(q_eval_proj) == 0
        or q_eval.shape[1] < 6
        or q_eval_proj.shape[1] < 6
    ):
        return

    n = int(min(len(q_eval), len(q_eval_proj)))
    qb = q_eval[:n].astype(np.float32, copy=False)
    qa = q_eval_proj[:n].astype(np.float32, copy=False)
    finite = np.isfinite(qb).all(axis=1) & np.isfinite(qa).all(axis=1)
    if not np.any(finite):
        return
    qb = qb[finite]
    qa = qa[finite]

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ab = spatial_tool_axis_n6(qb, use_pybullet=use_pybullet_n6)
    aa = spatial_tool_axis_n6(qa, use_pybullet=use_pybullet_n6)
    cb = np.clip(np.sum(ab * up.reshape(1, 3), axis=1), -1.0, 1.0)
    ca = np.clip(np.sum(aa * up.reshape(1, 3), axis=1), -1.0, 1.0)
    ang_before = np.degrees(np.arccos(cb))
    ang_after = np.degrees(np.arccos(ca))

    cap = float(np.percentile(np.concatenate([ang_before, ang_after], axis=0), 99))
    cap = max(cap, 1.0)
    bins = np.linspace(0.0, cap, 50)

    fig = plt.figure(figsize=(9.2, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(ang_before, bins=bins, color="#64748b", alpha=0.9)
    ax1.set_title("before projection")
    ax1.set_xlabel("orientation error (deg)")
    ax1.set_ylabel("count")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(ang_after, bins=bins, color="#16a34a", alpha=0.9)
    ax2.set_title("after projection")
    ax2.set_xlabel("orientation error (deg)")
    ax2.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(
        f"[ur5_proj_err] before mean={float(np.mean(ang_before)):.3f} deg, "
        f"after mean={float(np.mean(ang_after)):.3f} deg"
    )
    print(f"saved: {out_path}")


def _render_ur5_pybullet_trajectories(q_paths: list[np.ndarray], cfg: Any) -> None:
    if not q_paths:
        return
    kin_ov = {}
    for k, a in (("urdf_path", "ur5_urdf_path"), ("ee_link_index", "ur5_ee_link_index"), ("tool_axis", "ur5_tool_axis")):
        if hasattr(cfg, a):
            kin_ov[k] = getattr(cfg, a)
    ren_ov = {}
    for k, a in (
        ("grasp_offset", "ur5_grasp_offset"),
        ("grasp_axis_shift", "ur5_grasp_axis_shift"),
        ("debug_axes", "ur5_debug_axes"),
        ("cylinder_rotate_90", "ur5_cylinder_rotate_90"),
        ("gripper_close_ratio", "ur5_gripper_close_ratio"),
    ):
        if hasattr(cfg, a):
            ren_ov[k] = getattr(cfg, a)
    kin = resolve_ur5_kinematics_cfg(kin_ov)
    ren = resolve_ur5_render_cfg(ren_ov)

    urdf_path = str(kin["urdf_path"]).strip()
    if not urdf_path:
        print("[warn] skip pybullet render: urdf_path is empty")
        return
    try:
        import pybullet as p  # type: ignore
        import pybullet_data  # type: ignore
    except Exception as e:
        print(f"[warn] skip pybullet render: {e}")
        return

    cid = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=cid)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.7, cameraYaw=48.0, cameraPitch=-28.0, cameraTargetPosition=[0.0, 0.0, 0.5], physicsClientId=cid
    )
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    floor = p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.02], useFixedBase=True, physicsClientId=cid)
    p.changeVisualShape(floor, -1, rgbaColor=[0.96, 0.97, 0.99, 1.0], physicsClientId=cid)
    try:
        rid = p.loadURDF(urdf_path, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)
    except Exception:
        patched = _make_pybullet_friendly_urdf(urdf_path)
        rid = p.loadURDF(patched, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)

    rev = []
    nj = p.getNumJoints(rid, physicsClientId=cid)
    for j in range(nj):
        info = p.getJointInfo(rid, j, physicsClientId=cid)
        if int(info[2]) == p.JOINT_REVOLUTE:
            rev.append(int(j))
    if len(rev) < 6:
        print(f"[warn] skip pybullet render: URDF has only {len(rev)} revolute joints")
        p.disconnect(physicsClientId=cid)
        return
    arm = rev[:6]
    ee_raw = int(kin["ee_link_index"])
    ee_idx = ee_raw if ee_raw >= 0 else pick_default_ee_link_index(rid, arm[-1], cid)
    try:
        ee_info = p.getJointInfo(rid, ee_idx, physicsClientId=cid)
        ee_name = ee_info[12].decode("utf-8", errors="ignore")
        print(f"[pybullet] ee link index={ee_idx}, name={ee_name}")
    except Exception:
        pass
    tool_axis_name = str(kin["tool_axis"]).strip().lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(tool_axis_name, 2)
    grasp_offset = float(ren["grasp_offset"])
    grasp_axis_shift = float(ren["grasp_axis_shift"])
    debug_axes = bool(ren["debug_axes"])
    cyl_rotate_90 = bool(ren["cylinder_rotate_90"])

    cyl_len = 0.22
    cyl_rad = 0.03
    cyl_vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(cyl_rad),
        length=float(cyl_len),
        rgbaColor=[0.95, 0.75, 0.15, 0.9],
        physicsClientId=cid,
    )
    cyl_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=int(cyl_vis),
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        physicsClientId=cid,
    )
    top_line_id = -1
    # Try to anchor object by actual fingertip geometry (3-finger gripper).
    name_to_idx: dict[str, int] = {}
    for j in range(nj):
        info = p.getJointInfo(rid, j, physicsClientId=cid)
        lname = info[12].decode("utf-8", errors="ignore")
        name_to_idx[lname] = int(j)
    tip_names = [
        "gripperfinger_1_link_3",
        "gripperfinger_2_link_3",
        "gripperfinger_middle_link_3",
    ]
    tip_idx = [name_to_idx[nm] for nm in tip_names if nm in name_to_idx]
    gripper_joint_idx = None
    gripper_joint_limits = (-1.0, 1.0)
    for j in range(nj):
        info = p.getJointInfo(rid, j, physicsClientId=cid)
        jname = info[1].decode("utf-8", errors="ignore")
        if "gripperrobotiq_hand_joint" in jname:
            gripper_joint_idx = int(j)
            lo = float(info[8])
            hi = float(info[9])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                gripper_joint_limits = (lo, hi)
            break
    if debug_axes:
        print("[pybullet] link indices (subset):")
        for nm in ("base_link", "wrist_3_link", "tool0", "ee_link", "gripperpalm",
                   "gripperfinger_1_link_3", "gripperfinger_2_link_3", "gripperfinger_middle_link_3"):
            if nm in name_to_idx:
                print(f"  - {nm}: {name_to_idx[nm]}")
        print(f"[pybullet] using ee_idx={ee_idx}, tool_axis={tool_axis_name}")

    def _axis_vec_from_quat(quat_xyzw: list[float]) -> np.ndarray:
        mat = np.asarray(p.getMatrixFromQuaternion(quat_xyzw), dtype=np.float32).reshape(3, 3)
        v = mat[:, int(axis_idx)]
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _orient_cylinder_quat(ee_quat: list[float]) -> list[float]:
        # PyBullet cylinder local axis is +Z. Rotate it to requested tool axis, then apply ee orientation.
        if axis_idx == 2:  # z
            q_off = [0.0, 0.0, 0.0, 1.0]
        elif axis_idx == 0:  # x
            q_off = p.getQuaternionFromEuler([0.0, float(np.pi / 2.0), 0.0])
        else:  # y
            q_off = p.getQuaternionFromEuler([float(-np.pi / 2.0), 0.0, 0.0])
        _, q = p.multiplyTransforms([0.0, 0.0, 0.0], ee_quat, [0.0, 0.0, 0.0], q_off)
        return list(q)

    def _quat_from_axis(axis: np.ndarray, ref_axis: np.ndarray) -> list[float]:
        # Build quaternion whose local +Z aligns with axis, and x is stabilized by ref_axis.
        z = axis / max(float(np.linalg.norm(axis)), 1e-8)
        x0 = ref_axis - np.dot(ref_axis, z) * z
        if np.linalg.norm(x0) < 1e-8:
            x0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            x0 = x0 - np.dot(x0, z) * z
        x = x0 / max(float(np.linalg.norm(x0)), 1e-8)
        y = np.cross(z, x)
        y = y / max(float(np.linalg.norm(y)), 1e-8)
        x = np.cross(y, z)
        R = np.stack([x, y, z], axis=1).astype(np.float32)
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return [float(qx), float(qy), float(qz), float(qw)]

    def _apply_cylinder_extra_rotation(quat_xyzw: np.ndarray) -> np.ndarray:
        if not cyl_rotate_90:
            return quat_xyzw
        # Rotate cylinder by +90 deg around its local X so gripper tends to contact curved side.
        q_extra = p.getQuaternionFromEuler([float(np.pi / 2.0), 0.0, 0.0])
        _, q_new = p.multiplyTransforms([0.0, 0.0, 0.0], quat_xyzw.tolist(), [0.0, 0.0, 0.0], q_extra)
        return np.asarray(q_new, dtype=np.float32)

    def _cyl_axis_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
        mat = np.asarray(p.getMatrixFromQuaternion(quat_xyzw.tolist()), dtype=np.float32).reshape(3, 3)
        v = mat[:, 2]  # cylinder local Z
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _tip_center() -> np.ndarray | None:
        if len(tip_idx) < 3:
            return None
        pts = []
        for li in tip_idx[:3]:
            ls = p.getLinkState(rid, li, computeForwardKinematics=True, physicsClientId=cid)
            pts.append(np.asarray(ls[4], dtype=np.float32))
        return (pts[0] + pts[1] + pts[2]) / 3.0

    def _cylinder_pose_from_gripper(ee_pos: np.ndarray, ee_quat: list[float]) -> tuple[np.ndarray, np.ndarray]:
        # Preferred: infer from fingertip geometry.
        tool = _axis_vec_from_quat(ee_quat)
        if len(tip_idx) >= 3:
            pts = []
            for li in tip_idx[:3]:
                ls = p.getLinkState(rid, li, computeForwardKinematics=True, physicsClientId=cid)
                pts.append(np.asarray(ls[4], dtype=np.float32))
            center = (pts[0] + pts[1] + pts[2]) / 3.0
            center = center + tool * float(grasp_axis_shift)
            q = np.asarray(_quat_from_axis(tool, tool), dtype=np.float32)
            q = _apply_cylinder_extra_rotation(q)
            return center, q
        # Fallback: ee frame + offset.
        center = ee_pos + tool * float(grasp_offset)
        q = np.asarray(_quat_from_axis(tool, tool), dtype=np.float32)
        q = _apply_cylinder_extra_rotation(q)
        return center, q

    dt = float(max(float(_pln(cfg, "pybullet_real_time_dt", 0.06)), 0.01))
    try:
        # Initialize to first frame before enabling rendering to avoid the startup "disconnected" flash.
        q0 = q_paths[0][0]
        for k, j in enumerate(arm):
            p.resetJointState(rid, j, float(q0[k]), targetVelocity=0.0, physicsClientId=cid)
        if gripper_joint_idx is not None:
            lo, hi = gripper_joint_limits
            r = float(np.clip(float(ren["gripper_close_ratio"]), 0.0, 1.0))
            qg = lo + r * (hi - lo)
            p.resetJointState(rid, gripper_joint_idx, qg, targetVelocity=0.0, physicsClientId=cid)
        ls0 = p.getLinkState(rid, ee_idx, computeForwardKinematics=True, physicsClientId=cid)
        ee_pos0 = np.asarray(ls0[4], dtype=np.float32)
        ee_quat0 = list(ls0[5])
        cyl_pos0, cyl_quat0 = _cylinder_pose_from_gripper(ee_pos0, ee_quat0)
        # Fix a rigid grasp offset in ee frame to avoid tiny frame-to-frame jitter.
        R0 = np.asarray(p.getMatrixFromQuaternion(ee_quat0), dtype=np.float32).reshape(3, 3)
        tip0 = _tip_center()
        if tip0 is not None:
            local_off = (R0.T @ (tip0 - ee_pos0)).astype(np.float32)
            local_off = local_off + np.array([0.0, 0.0, float(grasp_axis_shift)], dtype=np.float32)
        else:
            local_off = np.array([0.0, 0.0, float(grasp_offset)], dtype=np.float32)
        q_ref = np.asarray(_orient_cylinder_quat(ee_quat0), dtype=np.float32)
        q_ref = _apply_cylinder_extra_rotation(q_ref)
        _, ee_inv = p.invertTransform([0.0, 0.0, 0.0], ee_quat0)
        _, q_offset = p.multiplyTransforms([0.0, 0.0, 0.0], ee_inv, [0.0, 0.0, 0.0], q_ref.tolist())
        q_offset = np.asarray(q_offset, dtype=np.float32)
        cyl_pos0 = ee_pos0 + (R0 @ local_off).astype(np.float32)
        _, q0w = p.multiplyTransforms([0.0, 0.0, 0.0], ee_quat0, [0.0, 0.0, 0.0], q_offset.tolist())
        p.resetBasePositionAndOrientation(cyl_id, cyl_pos0.tolist(), list(q0w), physicsClientId=cid)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=cid)
        for cidx, q_path in enumerate(q_paths):
            print(f"[pybullet] rendering case {cidx + 1}/{len(q_paths)} ...")
            for i in range(q_path.shape[0]):
                q = q_path[i]
                for k, j in enumerate(arm):
                    p.resetJointState(rid, j, float(q[k]), targetVelocity=0.0, physicsClientId=cid)
                if gripper_joint_idx is not None:
                    lo, hi = gripper_joint_limits
                    r = float(np.clip(float(ren["gripper_close_ratio"]), 0.0, 1.0))
                    qg = lo + r * (hi - lo)
                    p.resetJointState(rid, gripper_joint_idx, qg, targetVelocity=0.0, physicsClientId=cid)
                ls = p.getLinkState(rid, ee_idx, computeForwardKinematics=True, physicsClientId=cid)
                ee_pos = np.asarray(ls[4], dtype=np.float32)
                ee_quat = list(ls[5])  # xyzw
                R = np.asarray(p.getMatrixFromQuaternion(ee_quat), dtype=np.float32).reshape(3, 3)
                cyl_pos = ee_pos + (R @ local_off).astype(np.float32)
                _, q_cur = p.multiplyTransforms([0.0, 0.0, 0.0], ee_quat, [0.0, 0.0, 0.0], q_offset.tolist())
                cyl_quat = np.asarray(q_cur, dtype=np.float32)
                axis = _cyl_axis_from_quat(cyl_quat)
                p.resetBasePositionAndOrientation(cyl_id, cyl_pos.tolist(), cyl_quat.tolist(), physicsClientId=cid)

                p0 = cyl_pos + axis * (0.5 * cyl_len)
                p1 = cyl_pos + axis * (0.5 * cyl_len + 0.18)
                top_line_id = p.addUserDebugLine(
                    p0.tolist(),
                    p1.tolist(),
                    [1.0, 0.1, 0.1],
                    lineWidth=4.0,
                    lifeTime=0.0,
                    replaceItemUniqueId=int(top_line_id),
                    physicsClientId=cid,
                )
                if debug_axes:
                    mat = np.asarray(p.getMatrixFromQuaternion(ee_quat), dtype=np.float32).reshape(3, 3)
                    o = ee_pos
                    lx = o + 0.10 * mat[:, 0]
                    ly = o + 0.10 * mat[:, 1]
                    lz = o + 0.10 * mat[:, 2]
                    p.addUserDebugLine(o.tolist(), lx.tolist(), [1, 0, 0], 2.0, 1e-3, physicsClientId=cid)
                    p.addUserDebugLine(o.tolist(), ly.tolist(), [0, 1, 0], 2.0, 1e-3, physicsClientId=cid)
                    p.addUserDebugLine(o.tolist(), lz.tolist(), [0, 0, 1], 2.0, 1e-3, physicsClientId=cid)
                time.sleep(dt)
            time.sleep(0.6)
    finally:
        p.disconnect(physicsClientId=cid)
