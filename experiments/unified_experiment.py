from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from datasets.constraint_datasets import set_seed
from experiments.dataset_resolve import resolve_dataset
from evaluation.eval_runner import run_eval_metrics
from methods.utils.projection import (
    project_trajectory_numpy,
    project_points_with_steps_numpy,
    true_projection,
)
from planning.planner import (
    plan_path,
    _plot_planar_arm_planning,
    plan_linear_then_model_project,
    resolve_periodic_mode,
)
from methods.models.autoencoder import VariationalAutoEncoder
from planning.kinematics import (
    is_arm_dataset as _is_arm_dataset,
    is_workspace_pose_dataset as _is_workspace_pose_dataset,
    workspace_embed_for_eval as _workspace_embed_for_eval,
    wrap_np_pi as _wrap_np_pi,
    wrap_workspace_pose_rpy_np as _wrap_workspace_pose_rpy_np,
)
from methods.utils.feature_normalizer import FeatureNormalizedModel, FeatureNormalizer
from methods import dataaug as dataaug_method
from methods import vae as vae_base
from plotting import vae_plots as vae_plots
from methods import ecomann as ecomann_base
from plotting.ecomann_plots import save_training_diagnostics_2d as save_ecomann_training_diagnostics_2d
from plotting.ecomann_plots import save_loss_curves as save_ecomann_loss_curves
from methods import oncl as ve
from plotting.dataaug_plots import (
    plot_knn_normals,
    plot_loss_curves,
    plot_planned_paths,
)
from plotting.plot_common import plot_contour_traj_2d
from plotting.plot_common import plot_planned_paths_3d
from plotting.oncl_plots import (
    _plot_constraint_surface_paper_3d,
    _plot_constraint_2d,
    _plot_dual_arm_guided_insertion_orientation_3d,
    _plot_dual_arm_pose_projection_error_distributions,
    _plot_training_diagnostics,
    _plot_zero_surfaces_3d,
    _plot_ur5_eval_projection_workspace_orientation_3d,
    _plot_ur5_projection_error_distribution_from_pairs,
    _plot_workspace_pose_orientation_3d,
    _plot_workspace_pose_projection_error_distributions,
)
from evaluation.evaluator import eval_bounds_from_train, resolve_gt_grid

from experiments.config_loader import apply_overrides, load_layered_config

VALID_METHODS = {"oncl", "igr", "margin", "dataaug", "vae", "ecomann"}
DUAL_ARM_12D_DATASETS = {
    "12d_dual_arm",
    "12d_dual_arm_traj",
}
ARM_UP_6D_DATASETS = {
    "6d_spatial_arm_up_n6",
    "6d_spatial_arm_up_n6_py",
    "6d_spatial_arm_up_n6_traj",
    "6d_spatial_arm_up_n6_py_traj",
}


def _seed_token(seed: Any) -> str:
    text = str(seed).strip()
    if not text:
        text = "none"
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in text)


def _dataset_artifact_dir(out_root: str, method: str, dataset: str) -> str:
    outdir = os.path.join(str(out_root), str(method), str(dataset))
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _seeded_eval_model_paths(
    *,
    out_root: str,
    method: str,
    dataset: str,
    file_tag: str,
    seed: Any,
) -> tuple[str, str]:
    outdir = _dataset_artifact_dir(out_root, method, dataset)
    seed_part = _seed_token(seed)
    stem = f"{dataset}_{file_tag}_seed{seed_part}"
    return (
        os.path.join(outdir, f"{stem}_eval.json"),
        os.path.join(outdir, f"{stem}_model.pt"),
    )


def _ur5_use_pybullet_n6(dataset: str) -> bool:
    n = str(dataset)
    return n.startswith("6d_spatial_arm_up_n6") and ("_py" not in n)


@dataclass
class VAEConfig:
    seed: int = 36
    device: str = "auto"
    n_train: int = 512
    traj_gene_n_grid: int = 4096
    latent_dim: int = -1
    hidden_dims: tuple[int, ...] = (64, 32)
    epochs: int = 500
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta_final: float = 0.05
    warmup_epochs: int = 440
    ur5_backend: str = "auto"
    viz_enable: bool = True
    viz_sample_latent_n: int = 150
    viz_max_eval_points: int = 2000
    projector: dict[str, Any] = field(
        default_factory=lambda: {"alpha": 0.3, "steps": 80, "min_steps": 20}
    )
    planner: dict[str, Any] = field(
        default_factory=lambda: {
            "method": "traj_opt",
            "opt_steps": 1240,
            "opt_lr": 0.01,
            "opt_lam_smooth": 0.2,
            "lam_manifold": 1.0,
            "lam_len_joint": 0.40,
            "trust_scale": 0.8,
            "anim_fps": 6,
            "anim_stride": 1,
            "save_gif": True,
            "pybullet_render": False,
            "pybullet_real_time_dt": 0.06,
        }
    )


def _print_eval_lines(dataset: str, metrics: dict[str, Any]) -> None:
    print(
        f"[eval] {dataset} | proj_dist={float(metrics.get('proj_manifold_dist', float('nan'))):.6f} "
        f"| pred_recall={float(metrics.get('pred_recall', float('nan'))):.6f} "
        f"| pred_FPrate={float(metrics.get('pred_FPrate', float('nan'))):.6f} "
        f"| chamfer={float(metrics.get('bidirectional_chamfer', float('nan'))):.6f} "
        f"| gt->learned={float(metrics.get('gt_to_learned_mean', float('nan'))):.6f} "
        f"| learned->gt={float(metrics.get('learned_to_gt_mean', float('nan'))):.6f} "
        f"| space={metrics.get('dist_space', 'unknown')}"
    )


def _nn_dist_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.full((len(a),), float("nan"), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(b.astype(np.float64))
        d, _ = tree.query(a.astype(np.float64), k=1)
        return d.astype(np.float32)
    except Exception:
        aa = a.astype(np.float32)
        bb = b.astype(np.float32)
        d2 = np.sum((aa[:, None, :] - bb[None, :, :]) ** 2, axis=2)
        return np.sqrt(np.maximum(np.min(d2, axis=1), 0.0)).astype(np.float32)


def _worst_case_traj_2d(
    *,
    dataset: str,
    model: nn.Module,
    x_train: np.ndarray,
    cfg_for_project: Any,
    cfg_for_grid: Any,
    eval_artifacts: dict[str, Any],
    project_traj_fn: Any = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    x_eval = np.asarray(eval_artifacts.get("x_eval", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    proj = np.asarray(eval_artifacts.get("proj", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    if x_eval.ndim != 2 or proj.ndim != 2 or x_eval.shape[1] != 2 or proj.shape[1] != 2:
        return None, None
    n = int(min(len(x_eval), len(proj)))
    if n <= 0:
        return None, None
    x_eval = x_eval[:n]
    proj = proj[:n]
    finite = np.isfinite(x_eval).all(axis=1) & np.isfinite(proj).all(axis=1)
    if not np.any(finite):
        return None, None
    x_eval = x_eval[finite]
    proj = proj[finite]

    gt_grid = resolve_gt_grid(str(dataset), cfg_for_grid, x_train=x_train).astype(np.float32)
    if gt_grid.ndim != 2 or gt_grid.shape[1] != 2 or len(gt_grid) == 0:
        return None, None

    d_final = _nn_dist_numpy(proj, gt_grid)
    if len(d_final) == 0 or not np.isfinite(d_final).any():
        return None, None
    n_worst = int(min(24, max(1, np.ceil(0.05 * len(d_final)))))
    idx = np.argsort(d_final)[-n_worst:]
    x0_worst = x_eval[idx].astype(np.float32, copy=False)

    eps_used = float(eval_artifacts.get("eval_eps_used", 0.0))
    if not np.isfinite(eps_used) or eps_used <= 0.0:
        eps_used = 1e-6
    if callable(project_traj_fn):
        worst_traj = project_traj_fn(x0_worst.astype(np.float32))
    else:
        proj_cfg = getattr(cfg_for_project, "projector", {}) or {}
        worst_traj = project_trajectory_numpy(
            model,
            x0_worst,
            device=str(getattr(cfg_for_project, "device", "cpu")),
            proj_steps=int(proj_cfg.get("steps", 80)),
            proj_alpha=float(proj_cfg.get("alpha", 0.3)),
            proj_min_steps=int(proj_cfg.get("min_steps", 0)),
            f_abs_stop=eps_used,
        )
    return worst_traj, x0_worst


def _worst_case_traj_3d(
    *,
    dataset: str,
    model: nn.Module,
    x_train: np.ndarray,
    cfg_for_project: Any,
    cfg_for_grid: Any,
    eval_artifacts: dict[str, Any],
    project_traj_fn: Any = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    x_eval = np.asarray(eval_artifacts.get("x_eval", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    proj = np.asarray(eval_artifacts.get("proj", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    if x_eval.ndim != 2 or proj.ndim != 2 or x_eval.shape[1] != 3 or proj.shape[1] != 3:
        return None, None
    n = int(min(len(x_eval), len(proj)))
    if n <= 0:
        return None, None
    x_eval = x_eval[:n]
    proj = proj[:n]
    finite = np.isfinite(x_eval).all(axis=1) & np.isfinite(proj).all(axis=1)
    if not np.any(finite):
        return None, None
    x_eval = x_eval[finite]
    proj = proj[finite]

    gt_grid = resolve_gt_grid(str(dataset), cfg_for_grid, x_train=x_train).astype(np.float32)
    if gt_grid.ndim != 2 or gt_grid.shape[1] != 3 or len(gt_grid) == 0:
        return None, None

    if _is_arm_dataset(dataset) or _is_workspace_pose_dataset(dataset):
        use_pybullet_n6 = _ur5_use_pybullet_n6(str(dataset))
        gt_metric = _workspace_embed_for_eval(str(dataset), gt_grid, ur5_use_pybullet_n6=use_pybullet_n6)
        proj_metric = _workspace_embed_for_eval(str(dataset), proj, ur5_use_pybullet_n6=use_pybullet_n6)
    else:
        gt_metric = gt_grid
        proj_metric = proj

    d_final = _nn_dist_numpy(proj_metric, gt_metric)
    if len(d_final) == 0 or not np.isfinite(d_final).any():
        return None, None
    n_worst = int(min(24, max(1, np.ceil(0.05 * len(d_final)))))
    idx = np.argsort(d_final)[-n_worst:]
    x0_worst = x_eval[idx].astype(np.float32, copy=False)

    eps_used = float(eval_artifacts.get("eval_eps_used", 0.0))
    if not np.isfinite(eps_used) or eps_used <= 0.0:
        eps_used = 1e-6
    if callable(project_traj_fn):
        worst_traj = project_traj_fn(x0_worst.astype(np.float32))
    else:
        proj_cfg = getattr(cfg_for_project, "projector", {}) or {}
        worst_traj = project_trajectory_numpy(
            model,
            x0_worst,
            device=str(getattr(cfg_for_project, "device", "cpu")),
            proj_steps=int(proj_cfg.get("steps", 80)),
            proj_alpha=float(proj_cfg.get("alpha", 0.3)),
            proj_min_steps=int(proj_cfg.get("min_steps", 0)),
            f_abs_stop=eps_used,
        )
    return worst_traj, x0_worst


def _plot_worst_projection_3d(
    *,
    x_train: np.ndarray,
    worst_traj: np.ndarray,
    worst_x0: np.ndarray,
    out_path: str,
    title: str,
    axis_labels: tuple[str, str, str],
) -> None:
    if worst_traj is None or worst_x0 is None or len(worst_traj) == 0 or len(worst_x0) == 0:
        return
    train_plot = x_train
    if len(train_plot) > 2000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(train_plot), size=2000, replace=False)
        train_plot = train_plot[idx]

    fig = plt.figure(figsize=(8.2, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(train_plot[:, 0], train_plot[:, 1], train_plot[:, 2], s=5, c="gray", alpha=0.18, label="train")
    for i in range(worst_traj.shape[1]):
        ax.plot(
            worst_traj[:, i, 0], worst_traj[:, i, 1], worst_traj[:, i, 2],
            "-", color="red", linewidth=1.0, alpha=0.9,
            label="worst traj (top 5%)" if i == 0 else None,
        )
    ax.scatter(
        worst_x0[:, 0], worst_x0[:, 1], worst_x0[:, 2],
        s=16, c="royalblue", alpha=0.95, label="worst starts",
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _axis_labels_for_dataset(dataset: str, dim: int) -> tuple[str, ...]:
    if _is_arm_dataset(dataset):
        return tuple([f"q{i+1}" for i in range(dim)])
    return tuple([f"x{i+1}" for i in range(dim)])


def _make_vis_cfg_for_method(dataset: str, run_cfg: Any, eval_cfg: Any) -> Any:
    # Use the run's projection and bounds settings with the ONCL plot helpers.
    vis_vals = asdict(ve.build_cfg(dataset))
    if hasattr(run_cfg, "__dict__"):
        for k, v in vars(run_cfg).items():
            vis_vals[k] = v
    if hasattr(eval_cfg, "__dict__"):
        vis_vals.update(vars(eval_cfg))
    return SimpleNamespace(**vis_vals)


def _save_common_method_plots(
    *,
    dataset: str,
    method_tag: str,
    model: nn.Module,
    x_train: np.ndarray,
    outdir: str,
    vis_cfg: Any,
    eval_artifacts: dict[str, Any],
    project_traj_fn: Any = None,
) -> None:
    dim = int(x_train.shape[1])
    labels = _axis_labels_for_dataset(dataset, dim)
    mins, maxs = eval_bounds_from_train(x_train, vis_cfg)
    rng = np.random.default_rng(int(getattr(vis_cfg, "seed", 0)) + 3001)
    viz_proj_traj_count = max(8, int(getattr(vis_cfg, "viz_proj_traj_count", 64)))
    x0 = rng.uniform(mins, maxs, size=(viz_proj_traj_count, dim)).astype(np.float32)

    with torch.no_grad():
        f_on = model(torch.from_numpy(x_train.astype(np.float32)).to(str(getattr(vis_cfg, "device", "cpu"))))
        if f_on.dim() == 1:
            f_on = f_on.unsqueeze(1)
        h_on = torch.linalg.norm(f_on, dim=1).detach().cpu().numpy().reshape(-1)
    q = float(getattr(vis_cfg, "zero_eps_quantile", 90.0))
    eps_stop = float(np.percentile(np.abs(h_on), q))
    if callable(project_traj_fn):
        traj = project_traj_fn(x0.astype(np.float32))
    else:
        traj = project_trajectory_numpy(
            model,
            x0,
            device=str(getattr(vis_cfg, "device", "cpu")),
            proj_steps=int((getattr(vis_cfg, "projector", {}) or {}).get("steps", 80)),
            proj_alpha=float((getattr(vis_cfg, "projector", {}) or {}).get("alpha", 0.3)),
            proj_min_steps=int((getattr(vis_cfg, "projector", {}) or {}).get("min_steps", 0)),
            f_abs_stop=eps_stop,
        )

    if dim == 2:
        worst_traj, worst_x0 = _worst_case_traj_2d(
            dataset=str(dataset),
            model=model,
            x_train=x_train,
            cfg_for_project=vis_cfg,
            cfg_for_grid=vis_cfg,
            eval_artifacts=eval_artifacts,
            project_traj_fn=project_traj_fn,
        )
        out_path = os.path.join(outdir, f"{dataset}_{method_tag}_contour_traj.png")
        _plot_constraint_2d(
            model=model,
            x_train=x_train,
            traj=traj,
            out_path=out_path,
            title=f"{dataset}: {method_tag}",
            axis_labels=(labels[0], labels[1]),
            cfg=vis_cfg,
            worst_traj=worst_traj,
            worst_x0=worst_x0,
        )
        return
    if dim == 3:
        out_path = os.path.join(outdir, f"{dataset}_{method_tag}_zero_surfaces_3d.png")
        _plot_zero_surfaces_3d(
            model=model,
            x_train=x_train,
            traj=traj,
            out_path=out_path,
            title=f"{dataset}: {method_tag}",
            axis_labels=(labels[0], labels[1], labels[2]),
            cfg=vis_cfg,
            intersection_points=eval_artifacts.get("proj", None),
        )
        out_paper = os.path.join(outdir, f"{dataset}_{method_tag}_constraint_surface_paper.png")
        _plot_constraint_surface_paper_3d(
            model=model,
            x_train=x_train,
            out_path=out_paper,
            axis_labels=(labels[0], labels[1], labels[2]),
            cfg=vis_cfg,
            intersection_points=eval_artifacts.get("proj", None),
            dataset_name=str(dataset),
        )
        worst_traj_3d, worst_x0_3d = _worst_case_traj_3d(
            dataset=str(dataset),
            model=model,
            x_train=x_train,
            cfg_for_project=vis_cfg,
            cfg_for_grid=vis_cfg,
            eval_artifacts=eval_artifacts,
            project_traj_fn=project_traj_fn,
        )
        out_worst = os.path.join(outdir, f"{dataset}_{method_tag}_worst_projection_3d.png")
        _plot_worst_projection_3d(
            x_train=x_train,
            worst_traj=worst_traj_3d,
            worst_x0=worst_x0_3d,
            out_path=out_worst,
            title=f"{dataset}: {method_tag} worst-case projection trajectories",
            axis_labels=(labels[0], labels[1], labels[2]),
        )


def _save_udf_plots(
    *,
    method: str,
    dataset: str,
    model: nn.Module,
    x_train: np.ndarray,
    grid: np.ndarray,
    cfg: Any,
    vis_cfg: Any,
    eval_artifacts: dict[str, Any],
    train_history: dict[str, Any],
    train_artifacts: dict[str, Any],
    outdir: str,
    ds_info: dict[str, Any],
    n_basis: np.ndarray,
) -> None:
    x_eval = eval_artifacts.get("x_eval", np.zeros((0, x_train.shape[1]), dtype=np.float32)).astype(np.float32)
    if len(x_eval) == 0:
        return

    n_train = len(x_train)
    idx_list = [
        min(n_train - 1, int(round((n_train - 1) * r)))
        for r in np.linspace(0.05, 0.95, 15)
    ]
    k = dataaug_method.effective_knn_norm_estimation_points(cfg, n_train)
    plot_knn_normals(
        x_train,
        idx_list=idx_list,
        k=k,
        out_path=os.path.join(outdir, f"{dataset}_{method}_knn_normal.png"),
        title=f"{dataset}: KNN + Normal ({method})",
        cfg=vis_cfg,
        grid=grid,
        n_basis=n_basis,
        off_bank=np.asarray(train_artifacts.get("off_bank", np.zeros((0, 0, x_train.shape[1]), dtype=np.float32)), dtype=np.float32),
        off_bank_preview=np.asarray(train_artifacts.get("off_bank_preview", np.zeros((0, 0, x_train.shape[1]), dtype=np.float32)), dtype=np.float32),
        off_bank_preview_mask=np.asarray(train_artifacts.get("off_bank_preview_mask", np.zeros((0, 0), dtype=bool)), dtype=bool),
    )

    n_plot = 16 if x_train.shape[1] == 3 else 128
    n_plot = min(n_plot, len(x_eval))
    x0 = x_eval[:n_plot].astype(np.float32)
    eps_used = float(eval_artifacts.get("eval_eps_used", 0.0))
    if not np.isfinite(eps_used) or eps_used <= 0.0:
        eps_used = 1e-6
    traj = project_trajectory_numpy(
        model,
        x0,
        device=str(getattr(cfg, "device", "cpu")),
        proj_steps=int((getattr(cfg, "projector", {}) or {}).get("steps", 100)),
        proj_alpha=float((getattr(cfg, "projector", {}) or {}).get("alpha", 0.3)),
        proj_min_steps=int((getattr(cfg, "projector", {}) or {}).get("min_steps", 0)),
        f_abs_stop=eps_used,
    )
    if int(x_train.shape[1]) == 2:
        axis_labels = _axis_labels_for_dataset(dataset, int(x_train.shape[1]))
        worst_traj, worst_x0 = _worst_case_traj_2d(
            dataset=str(dataset),
            model=model,
            x_train=x_train,
            cfg_for_project=cfg,
            cfg_for_grid=vis_cfg,
            eval_artifacts=eval_artifacts,
        )
        plot_contour_traj_2d(
            model=model,
            x_train=x_train,
            traj=traj,
            out_path=os.path.join(outdir, f"{dataset}_{method}_contour_traj.png"),
            title=f"{dataset}: {method.capitalize()} Baseline",
            axis_labels=(axis_labels[0], axis_labels[1]),
            cfg=vis_cfg,
            line_color="green",
            worst_traj=worst_traj,
            worst_x0=worst_x0,
        )
    plot_loss_curves(
        train_history,
        out_path=os.path.join(outdir, f"{dataset}_{method}_loss_curves.png"),
        title=f"{dataset}: {method.capitalize()} Baseline Losses",
        cfg=vis_cfg,
    )

    if x_train.shape[1] == 3:
        if _is_arm_dataset(dataset):
            out_plan = os.path.join(outdir, f"{dataset}_{method}_planning_demo.png")
            _plot_planar_arm_planning(
                model,
                str(dataset),
                x_train,
                out_plan,
                cfg,
                render_pybullet=False,
            )
            return
        plan_rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 77)
        n_pairs = 4
        replace = len(x_eval) < 2 * n_pairs
        picks = plan_rng.choice(len(x_eval), size=2 * n_pairs, replace=replace)
        plan_pairs = [
            (x_eval[picks[2 * i]], x_eval[picks[2 * i + 1]])
            for i in range(n_pairs)
        ]
        plans_proj: list[np.ndarray] = []
        plans_constr: list[np.ndarray] = []
        pln = getattr(cfg, "planner", None)
        planner_name = str(pln.get("method", "traj_opt")) if isinstance(pln, dict) else "traj_opt"
        n_waypoints = int(pln.get("steps", 64)) + 1 if isinstance(pln, dict) else 65
        use_linear = planner_name.lower() == "linear_proj"
        for x_start, x_goal in plan_pairs:
            x_start = true_projection(x_start[None, :], grid)[0][0]
            x_goal = true_projection(x_goal[None, :], grid)[0][0]
            planned = plan_path(
                model=model,
                x_start=x_start,
                x_goal=x_goal,
                cfg=cfg,
                planner_name=planner_name,
                n_waypoints=n_waypoints,
                dataset_name=dataset,
                periodic_joint=bool(ds_info.get("periodic_joint", False)),
                f_abs_stop=eps_used,
            )
            if use_linear:
                plans_proj.append(planned)
            else:
                plans_constr.append(planned)
        axis_labels = _axis_labels_for_dataset(dataset, 3)
        plot_planned_paths_3d(
            x_train=x_train,
            plans_proj=plans_proj,
            plans_constr=plans_constr,
            out_path=os.path.join(outdir, f"{dataset}_{method}_planner_paths_3d.png"),
            title=f"{dataset}: {method.capitalize()} Planned Paths (3D)",
            axis_labels=(axis_labels[0], axis_labels[1], axis_labels[2]),
        )
        return
    if x_train.shape[1] == 6 and str(dataset) in ARM_UP_6D_DATASETS:
        out_plan = os.path.join(outdir, f"{dataset}_{method}_planning_demo.png")
        _plot_planar_arm_planning(
            model,
            str(dataset),
            x_train,
            out_plan,
            cfg,
            render_pybullet=False,
        )
        return
    if x_train.shape[1] != 2:
        return
    if str(dataset) == "2d_planar_arm_line_n2":
        out_plan = os.path.join(outdir, f"{dataset}_{method}_planning_demo.png")
        _plot_planar_arm_planning(
            model,
            str(dataset),
            x_train,
            out_plan,
            cfg,
            render_pybullet=False,
        )
        return
    plan_rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 77)
    n_pairs = 4
    replace = len(x_eval) < 2 * n_pairs
    picks = plan_rng.choice(len(x_eval), size=2 * n_pairs, replace=replace)
    plan_pairs = [
        (x_eval[picks[2 * i]], x_eval[picks[2 * i + 1]])
        for i in range(n_pairs)
    ]
    plans_proj: list[np.ndarray] = []
    plans_constr: list[np.ndarray] = []
    pln = getattr(cfg, "planner", None)
    planner_name = str(pln.get("method", "traj_opt")) if isinstance(pln, dict) else "traj_opt"
    n_waypoints = int(pln.get("steps", 64)) + 1 if isinstance(pln, dict) else 65
    use_linear = planner_name.lower() == "linear_proj"
    for x_start, x_goal in plan_pairs:
        x_start = true_projection(x_start[None, :], grid)[0][0]
        x_goal = true_projection(x_goal[None, :], grid)[0][0]
        planned = plan_path(
            model=model,
            x_start=x_start,
            x_goal=x_goal,
            cfg=cfg,
            planner_name=planner_name,
            n_waypoints=n_waypoints,
            dataset_name=dataset,
            periodic_joint=bool(ds_info.get("periodic_joint", False)),
            f_abs_stop=eps_used,
        )
        if use_linear:
            plans_proj.append(planned)
        else:
            plans_constr.append(planned)
    plot_planned_paths(
        model,
        x_train,
        grid,
        plans_proj,
        plans_constr,
        vis_cfg,
        out_path=os.path.join(outdir, f"{dataset}_{method}_planner_paths.png"),
        title=f"{dataset}: Delta Planned Paths",
        zero_level_eps=eps_used,
    )


def _apply_projector_subcfg(cfg_obj: Any) -> None:
    # Projection parameters are consumed directly from cfg.projector.
    _ = cfg_obj


def _apply_planner_subcfg(cfg_obj: Any) -> None:
    # Planner parameters are consumed directly from cfg.planner.
    _ = cfg_obj


def _build_cfg_from_mapping_strict(cfg_cls: Any, mapping: dict[str, Any]) -> Any:
    # Strict mode for runner path: every cfg field must come from layered config.
    field_names = set(getattr(cfg_cls, "__dataclass_fields__", {}).keys())
    missing = sorted([k for k in field_names if k not in mapping])
    if missing:
        raise ValueError(
            f"missing required config keys for {cfg_cls.__name__}: {missing}. "
            "Runner mode requires all fields to be defined in config files."
        )
    kwargs = {k: mapping[k] for k in field_names}
    return cfg_cls(**kwargs)


def _attach_extra_cfg_fields(cfg_obj: Any, mapping: dict[str, Any]) -> Any:
    # Layered dataset configs may contain task-specific fields that are not part
    # of the method dataclass. Preserve them on the runtime cfg object so
    # resolve_dataset/evaluation/checkpoint export all see the same task params.
    for key, value in mapping.items():
        if not hasattr(cfg_obj, key):
            setattr(cfg_obj, key, value)
    return cfg_obj


def _cfg_to_serializable_dict(cfg_obj: Any, mapping: dict[str, Any]) -> dict[str, Any]:
    cfg_dict = dict(mapping)
    cfg_dict.update(asdict(cfg_obj))
    for key, value in vars(cfg_obj).items():
        if key not in cfg_dict:
            cfg_dict[key] = value
    return cfg_dict


def _resolve_run_config(
    method: str,
    dataset: str,
    *,
    config_root: str,
    cli_overrides: list[str],
) -> tuple[dict[str, Any], list[str]]:
    cfg_method = "oncl" if str(method) == "igr" else str(method)
    cfg_dict, loaded_paths = load_layered_config(config_root, cfg_method, dataset)
    cfg_dict = apply_overrides(cfg_dict, cli_overrides)
    # Guard against common typo: passing method_dataset id as --dataset
    # (e.g. "oncl_6d_spatial_arm_up_n6_py") instead of pure dataset id.
    ds_prefix = os.path.abspath(os.path.join(os.path.abspath(config_root), "datasets"))
    md_prefix = os.path.abspath(os.path.join(os.path.abspath(config_root), "method_dataset"))
    has_dataset_layer = any(os.path.abspath(p).startswith(ds_prefix) for p in loaded_paths)
    has_method_dataset_layer = any(os.path.abspath(p).startswith(md_prefix) for p in loaded_paths)
    if (not has_dataset_layer) and has_method_dataset_layer and str(dataset).startswith(f"{cfg_method}_"):
        base = str(dataset)[len(f"{cfg_method}_") :]
        raise ValueError(
            f"invalid dataset id '{dataset}': looks like a method_dataset key. "
            f"Use --dataset {base} (without '{cfg_method}_' prefix)."
        )
    _require_method_projector_cfg(
        method=cfg_method,
        config_root=config_root,
        loaded_paths=loaded_paths,
    )
    return cfg_dict, loaded_paths


def _load_config_file(path: str) -> dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            data = json.load(f)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError(f"yaml config requested but PyYAML not available: {path}") from e
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"unsupported config extension: {path}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be object/dict: {path}")
    return data


def _require_method_projector_cfg(*, method: str, config_root: str, loaded_paths: list[str]) -> None:
    # Keep strict requirement for iterative projector-based methods.
    if str(method) not in {"oncl", "igr", "margin", "dataaug"}:
        return
    root = os.path.abspath(str(config_root))
    method_prefix = os.path.join(root, "methods", str(method))
    method_path = next((p for p in loaded_paths if os.path.abspath(p).startswith(os.path.abspath(method_prefix))), None)
    if method_path is None:
        raise ValueError(
            f"missing method config for '{method}': expected configs/methods/{method}.json/.yaml with projector settings"
        )
    method_cfg = _load_config_file(method_path)
    proj = method_cfg.get("projector", None)
    if not isinstance(proj, dict):
        raise ValueError(f"{method_path} must define object key 'projector'")
    missing = [k for k in ("alpha", "steps", "min_steps") if k not in proj]
    if missing:
        raise ValueError(f"{method_path} projector missing keys: {missing}")


def run_oncl_one(
    dataset: str,
    *,
    out_root: str,
    seed_override: int | None,
    cfg_mapping: dict[str, Any],
    method_name: str = "oncl",
    method_tag: str = "oncl",
    force_single_output: bool = False,
) -> dict[str, Any]:
    cfg = _build_cfg_from_mapping_strict(ve.DemoCfg, cfg_mapping)
    cfg = _attach_extra_cfg_fields(cfg, cfg_mapping)
    _apply_projector_subcfg(cfg)
    _apply_planner_subcfg(cfg)
    cfg.train_method_name = str(method_name)

    if seed_override is not None:
        cfg.seed = int(seed_override)

    cfg.device = ve._choose_device(str(cfg.device))
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    set_seed(int(cfg.seed))
    outdir = os.path.join(out_root, str(method_name))
    os.makedirs(outdir, exist_ok=True)

    ds = resolve_dataset(
        dataset,
        cfg,
        optimize_ur5_train_only=True,
        ur5_backend=("pybullet" if _ur5_use_pybullet_n6(str(dataset)) else "analytic"),
    )
    x_train = ds["x_train"]
    data_dim = int(ds.get("data_dim", int(x_train.shape[1])))
    true_codim = int(ds.get("true_codim", 1))
    train_t0 = time.perf_counter()

    if bool(force_single_output):
        codim = 1
        cfg.constraint_dim = 1
        cfg.lam_oncl_ortho = 0.0
    elif str(cfg.constraint_dim).lower() == "auto":
        try:
            est = ve.estimate_codim_local_pca(
                x_train,
                periodic_joint=_is_arm_dataset(dataset),
                sample_ratio=float(cfg.codim_auto_sample_ratio),
                k_neighbors=int(cfg.codim_auto_k_neighbors),
                const_axis_std_ratio=float(cfg.codim_auto_const_axis_std_ratio),
                seed=int(cfg.seed) + 1000,
            )
            codim = int(est["estimated_codim"])
            if codim != true_codim:
                print(
                    "\033[31m"
                    f"[warn] [codim-auto] mismatch for {dataset}: estimated={codim}, true={true_codim}; "
                    "fallback to true codim for training."
                    "\033[0m"
                )
                codim = int(true_codim)
        except Exception as e:
            print(
                "\033[31m"
                f"[warn] [codim-auto] failed for {dataset}: {e}; fallback to true codim={true_codim}."
                "\033[0m"
            )
            codim = int(true_codim)
    else:
        codim = max(1, int(cfg.constraint_dim))
        if data_dim == 2 and codim != 1:
            codim = 1

    model, train_hist = ve.train_oncl_only(cfg, x_train, constraint_dim=codim)
    learned_codim = int(np.asarray(train_hist.get("learned_codim", np.asarray([codim]))).reshape(-1)[0])
    train_seconds = float(time.perf_counter() - train_t0)

    if _is_arm_dataset(dataset):
        post_fn = _wrap_np_pi
    elif _is_workspace_pose_dataset(dataset):
        post_fn = _wrap_workspace_pose_rpy_np
    else:
        post_fn = None
    use_pybullet_n6 = _ur5_use_pybullet_n6(str(dataset))
    embed_fn = (
        lambda q, _name=dataset: _workspace_embed_for_eval(
            _name,
            q,
            ur5_use_pybullet_n6=use_pybullet_n6,
        )
    ) if (_is_arm_dataset(dataset) or _is_workspace_pose_dataset(dataset)) else None

    def project_fn(_model: nn.Module, x0: np.ndarray, eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
        return project_points_with_steps_numpy(
            _model,
            x0.astype(np.float32),
            device=str(cfg.device),
            proj_steps=int((getattr(cfg, "projector", {}) or {}).get("steps", 100)),
            proj_alpha=float((getattr(cfg, "projector", {}) or {}).get("alpha", 0.3)),
            proj_min_steps=int((getattr(cfg, "projector", {}) or {}).get("min_steps", 0)),
            f_abs_stop=float(eps_stop),
        )

    metrics, eval_cfg, eval_artifacts = run_eval_metrics(
        cfg=cfg,
        method_key="oncl",
        dataset_name=dataset,
        model=model,
        x_train=x_train,
        project_fn=project_fn,
        embed_fn=embed_fn,
        postprocess_fn=post_fn,
    )
    metrics["train_seconds"] = float(train_seconds)
    _print_eval_lines(dataset, metrics)

    vis_cfg = _make_vis_cfg_for_method(dataset, cfg, eval_cfg)
    _save_common_method_plots(
        dataset=dataset,
        method_tag=str(method_tag),
        model=model,
        x_train=x_train,
        outdir=outdir,
        vis_cfg=vis_cfg,
        eval_artifacts=eval_artifacts,
    )
    out_diag = os.path.join(outdir, f"{dataset}_{method_tag}_training_diagnostics.png")
    _plot_training_diagnostics(train_hist, out_diag, title=f"{dataset}: training diagnostics (on-data)")

    if int(x_train.shape[1]) == 2:
        if str(dataset) == "2d_planar_arm_line_n2":
            out_plan = os.path.join(outdir, f"{dataset}_{method_tag}_planning_demo.png")
            _plot_planar_arm_planning(
                model,
                str(dataset),
                x_train,
                out_plan,
                cfg,
                render_pybullet=False,
            )
        else:
            x_eval = np.asarray(eval_artifacts.get("x_eval", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            grid_vis = ds.get("grid", None)
            if grid_vis is None:
                grid_vis = x_train
            grid_vis = np.asarray(grid_vis, dtype=np.float32)
            if len(x_eval) >= 8 and len(grid_vis) > 0:
                plan_rng = np.random.default_rng(int(cfg.seed) + 77)
                n_pairs = 4
                replace = len(x_eval) < 2 * n_pairs
                picks = plan_rng.choice(len(x_eval), size=2 * n_pairs, replace=replace)
                pairs = [(x_eval[picks[2 * i]], x_eval[picks[2 * i + 1]]) for i in range(n_pairs)]
                plans_proj: list[np.ndarray] = []
                plans_constr: list[np.ndarray] = []
                pln = getattr(cfg, "planner", None)
                planner_name = str(pln.get("method", "traj_opt")) if isinstance(pln, dict) else "traj_opt"
                n_waypoints = int(pln.get("steps", 64)) + 1 if isinstance(pln, dict) else 65
                use_linear = planner_name.lower() == "linear_proj"
                eps_used = float(eval_artifacts.get("eval_eps_used", 1e-6))
                if not np.isfinite(eps_used) or eps_used <= 0.0:
                    eps_used = 1e-6
                for x_start, x_goal in pairs:
                    x_start = true_projection(x_start[None, :], grid_vis)[0][0]
                    x_goal = true_projection(x_goal[None, :], grid_vis)[0][0]
                    planned = plan_path(
                        model=model,
                        x_start=x_start,
                        x_goal=x_goal,
                        cfg=cfg,
                        planner_name=planner_name,
                        n_waypoints=n_waypoints,
                        dataset_name=dataset,
                        periodic_joint=bool(ds.get("periodic_joint", False)),
                        f_abs_stop=eps_used,
                    )
                    if use_linear:
                        plans_proj.append(planned)
                    else:
                        plans_constr.append(planned)
                out_plan = os.path.join(outdir, f"{dataset}_oncl_planner_paths.png")
                plot_planned_paths(
                    model,
                    x_train,
                    grid_vis,
                    plans_proj,
                    plans_constr,
                    vis_cfg,
                    out_path=out_plan,
                    title=f"{dataset}: {method_name} Planned Paths",
                    zero_level_eps=eps_used,
                )

    if str(dataset) in ("3d_planar_arm_line_n3", "3d_spatial_arm_plane_n3", "3d_spatial_arm_ellip_n3", "3d_spatial_arm_circle_n3") or str(dataset) in ARM_UP_6D_DATASETS:
        out_plan = os.path.join(outdir, f"{dataset}_{method_tag}_planning_demo.png")
        _plot_planar_arm_planning(
            model,
            str(dataset),
            x_train,
            out_plan,
            cfg,
            render_pybullet=False,
        )
    elif int(x_train.shape[1]) == 3:
        x_eval = np.asarray(eval_artifacts.get("x_eval", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        grid_vis = ds.get("grid", None)
        if grid_vis is None:
            grid_vis = x_train
        grid_vis = np.asarray(grid_vis, dtype=np.float32)
        if len(x_eval) >= 8 and len(grid_vis) > 0:
            plan_rng = np.random.default_rng(int(cfg.seed) + 77)
            n_pairs = 4
            replace = len(x_eval) < 2 * n_pairs
            picks = plan_rng.choice(len(x_eval), size=2 * n_pairs, replace=replace)
            pairs = [(x_eval[picks[2 * i]], x_eval[picks[2 * i + 1]]) for i in range(n_pairs)]
            plans_proj: list[np.ndarray] = []
            plans_constr: list[np.ndarray] = []
            pln = getattr(cfg, "planner", None)
            planner_name = str(pln.get("method", "traj_opt")) if isinstance(pln, dict) else "traj_opt"
            n_waypoints = int(pln.get("steps", 64)) + 1 if isinstance(pln, dict) else 65
            use_linear = planner_name.lower() == "linear_proj"
            eps_used = float(eval_artifacts.get("eval_eps_used", 1e-6))
            if not np.isfinite(eps_used) or eps_used <= 0.0:
                eps_used = 1e-6
            for x_start, x_goal in pairs:
                x_start = true_projection(x_start[None, :], grid_vis)[0][0]
                x_goal = true_projection(x_goal[None, :], grid_vis)[0][0]
                planned = plan_path(
                    model=model,
                    x_start=x_start,
                    x_goal=x_goal,
                    cfg=cfg,
                    planner_name=planner_name,
                    n_waypoints=n_waypoints,
                    dataset_name=dataset,
                    periodic_joint=bool(ds.get("periodic_joint", False)),
                    f_abs_stop=eps_used,
                )
                if use_linear:
                    plans_proj.append(planned)
                else:
                    plans_constr.append(planned)
            labels3 = _axis_labels_for_dataset(dataset, 3)
            out_plan = os.path.join(outdir, f"{dataset}_{method_tag}_planner_paths_3d.png")
            plot_planned_paths_3d(
                x_train=x_train,
                plans_proj=plans_proj,
                plans_constr=plans_constr,
                out_path=out_plan,
                title=f"{dataset}: {method_name} Planned Paths (3D)",
                axis_labels=(labels3[0], labels3[1], labels3[2]),
            )

    if str(dataset) in (
        "6d_surf_normal",
        "6d_surf_normal_traj",
    ):
        out_pose = os.path.join(outdir, f"{dataset}_{method_tag}_workspace_pose_orientation.png")
        _plot_workspace_pose_orientation_3d(
            x_train=x_train,
            eval_proj=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_pose,
            title=f"{dataset} ({method_name}): projected eval poses + orientation z-axis",
            surface_cfg=cfg,
        )
        out_err = os.path.join(outdir, f"{dataset}_{method_tag}_workspace_pose_proj_error_distributions.png")
        _plot_workspace_pose_projection_error_distributions(
            x_before=eval_artifacts.get("x_eval", np.zeros((0, 6), dtype=np.float32)),
            x_after=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_err,
            title=f"{dataset} ({method_name}): projection errors before/after",
            surface_cfg=cfg,
        )
    if str(dataset) in DUAL_ARM_12D_DATASETS:
        out_pose = os.path.join(outdir, f"{dataset}_{method_tag}_dual_arm_guided_insertion_orientation.png")
        _plot_dual_arm_guided_insertion_orientation_3d(
            x_train=x_train,
            eval_proj=eval_artifacts.get("proj", np.zeros((0, 12), dtype=np.float32)),
            out_path=out_pose,
            title=f"{dataset} ({method_name}): dual-arm guided insertion poses",
            task_cfg=cfg,
        )
        out_err = os.path.join(outdir, f"{dataset}_{method_tag}_dual_arm_pose_proj_error_distributions.png")
        _plot_dual_arm_pose_projection_error_distributions(
            x_before=eval_artifacts.get("x_eval", np.zeros((0, 12), dtype=np.float32)),
            x_after=eval_artifacts.get("proj", np.zeros((0, 12), dtype=np.float32)),
            out_path=out_err,
            title=f"{dataset} ({method_name}): dual-arm projection errors before/after",
            grasp_span=float(getattr(cfg, "dual_arm_grasp_span", 1.0)),
            x_span=float(getattr(cfg, "dual_arm_curve_x_span", 1.4)),
            y_amp=float(getattr(cfg, "dual_arm_curve_y_amp", 0.55)),
            y_freq=float(getattr(cfg, "dual_arm_curve_y_freq", 1.0)),
            z_base=float(getattr(cfg, "dual_arm_curve_z_base", 0.2)),
            z_amp=float(getattr(cfg, "dual_arm_curve_z_amp", getattr(cfg, "dual_arm_vertical_half_range", 0.35))),
            z_freq=float(getattr(cfg, "dual_arm_curve_z_freq", 0.7)),
            z_half_range=float(getattr(cfg, "dual_arm_vertical_half_range", getattr(cfg, "dual_arm_curve_z_amp", 0.35))),
        )
    if dataset in ARM_UP_6D_DATASETS:
        out_dist = os.path.join(outdir, f"{dataset}_{method_tag}_proj_value_distribution.png")
        _plot_ur5_projection_error_distribution_from_pairs(
            q_eval=eval_artifacts.get("x_eval", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_dist,
            title=f"{dataset} ({method_name}): orientation-angle error before/after projection",
            use_pybullet_n6=_ur5_use_pybullet_n6(str(dataset)),
        )
        out_ws = os.path.join(outdir, f"{dataset}_{method_tag}_eval_proj_workspace_orientation.png")
        _plot_ur5_eval_projection_workspace_orientation_3d(
            q_train=x_train,
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_ws,
            title=f"{dataset} ({method_name}): eval projected points in workspace + tool orientation",
            use_pybullet_n6=_ur5_use_pybullet_n6(str(dataset)),
        )

    eval_path, ckpt_path = _seeded_eval_model_paths(
        out_root=out_root,
        method=str(method_name),
        dataset=str(dataset),
        file_tag=str(method_tag),
        seed=int(cfg.seed),
    )
    cfg_serial = _cfg_to_serializable_dict(cfg, cfg_mapping)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    torch.save(
        {
            "dataset": str(dataset),
            "method": str(method_name),
            "model_state": model.state_dict(),
            "in_dim": int(x_train.shape[1]),
            "constraint_dim": int(learned_codim),
            "hidden": int(cfg.hidden),
            "depth": int(cfg.depth),
            "x_train": x_train.astype(np.float32),
            "train_hist": train_hist,
            "cfg": cfg_serial,
        },
        ckpt_path,
    )

    return {
        "method": str(method_name),
        "dataset": dataset,
        "metrics": _normalize_metrics(metrics),
        "eval_path": eval_path,
        "ckpt_path": ckpt_path,
        "config": {**cfg_serial, "constraint_dim": int(learned_codim)},
    }


def run_udf_one(
    method: str,
    dataset: str,
    *,
    out_root: str,
    seed_override: int | None,
    cfg_mapping: dict[str, Any],
) -> dict[str, Any]:
    cfg = _build_cfg_from_mapping_strict(dataaug_method.Config, cfg_mapping)
    cfg = _attach_extra_cfg_fields(cfg, cfg_mapping)
    _apply_projector_subcfg(cfg)
    _apply_planner_subcfg(cfg)

    if seed_override is not None:
        cfg.seed = int(seed_override)

    if str(cfg.device) == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    set_seed(int(cfg.seed))

    ds = resolve_dataset(
        dataset,
        cfg,
        optimize_ur5_train_only=True,
        ur5_backend=("pybullet" if _ur5_use_pybullet_n6(str(dataset)) else "analytic"),
    )
    x_train = ds["x_train"]
    true_codim = int(ds.get("true_codim", 1))
    train_t0 = time.perf_counter()

    knn_k = dataaug_method.effective_knn_norm_estimation_points(cfg, len(x_train))
    model, train_stats, train_history, train_artifacts = dataaug_method.train_baseline_with_optional_normalization(
        cfg,
        mode=method,
        x=x_train,
        true_codim=true_codim,
        dataset_name=str(dataset),
        knn_k=knn_k,
    )
    train_seconds = float(time.perf_counter() - train_t0)
    feature_normalizer = train_artifacts.get("feature_normalizer", None)
    eval_model: nn.Module = model
    if getattr(feature_normalizer, "enabled", False):
        eval_model = FeatureNormalizedModel(model, feature_normalizer).to(str(cfg.device))
    n_basis = np.asarray(
        train_artifacts.get(
            "n_basis_plot",
            np.zeros((len(x_train), x_train.shape[1], true_codim), dtype=np.float32),
        ),
        dtype=np.float32,
    )

    if _is_arm_dataset(dataset):
        post_fn = _wrap_np_pi
    elif _is_workspace_pose_dataset(dataset):
        post_fn = _wrap_workspace_pose_rpy_np
    else:
        post_fn = None
    use_pybullet_n6 = _ur5_use_pybullet_n6(str(dataset))
    embed_fn = (
        lambda q, _name=dataset: _workspace_embed_for_eval(
            _name,
            q,
            ur5_use_pybullet_n6=use_pybullet_n6,
        )
    ) if (_is_arm_dataset(dataset) or _is_workspace_pose_dataset(dataset)) else None

    project_fn_base = dataaug_method._make_project_fn(cfg)

    def project_fn(_model: nn.Module, x0: np.ndarray, eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
        if getattr(feature_normalizer, "enabled", False):
            x0_norm = feature_normalizer.transform(np.asarray(x0, dtype=np.float32))
            proj_norm, steps = project_fn_base(model, x0_norm, eps_stop)
            return feature_normalizer.inverse_transform(proj_norm), steps
        return project_fn_base(_model, x0, eps_stop)

    metrics, eval_cfg, eval_artifacts = run_eval_metrics(
        cfg=cfg,
        method_key=method,
        dataset_name=dataset,
        model=eval_model,
        x_train=x_train,
        project_fn=project_fn,
        embed_fn=embed_fn,
        postprocess_fn=post_fn,
    )
    metrics["train_seconds"] = float(train_seconds)
    _print_eval_lines(dataset, metrics)

    outdir = os.path.join(out_root, method)
    os.makedirs(outdir, exist_ok=True)
    vis_cfg = _make_vis_cfg_for_method(dataset, cfg, eval_cfg)
    grid_vis = ds.get("grid", None)
    if grid_vis is None:
        grid_vis = x_train
    _save_udf_plots(
        method=str(method),
        dataset=str(dataset),
        model=eval_model,
        x_train=x_train,
        grid=np.asarray(grid_vis, dtype=np.float32),
        cfg=cfg,
        vis_cfg=vis_cfg,
        eval_artifacts=eval_artifacts,
        train_history=train_history,
        train_artifacts=train_artifacts,
        outdir=outdir,
        ds_info=ds,
        n_basis=n_basis,
    )
    if int(x_train.shape[1]) != 2:
        _save_common_method_plots(
            dataset=dataset,
            method_tag=str(method),
            model=eval_model,
            x_train=x_train,
            outdir=outdir,
            vis_cfg=vis_cfg,
            eval_artifacts=eval_artifacts,
        )
    if dataset in ARM_UP_6D_DATASETS:
        out_dist = os.path.join(outdir, f"{dataset}_{method}_proj_value_distribution.png")
        _plot_ur5_projection_error_distribution_from_pairs(
            q_eval=eval_artifacts.get("x_eval", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_dist,
            title=f"{dataset} ({method}): orientation-angle error before/after projection",
            use_pybullet_n6=_ur5_use_pybullet_n6(str(dataset)),
        )
        out_ws = os.path.join(outdir, f"{dataset}_{method}_eval_proj_workspace_orientation.png")
        _plot_ur5_eval_projection_workspace_orientation_3d(
            q_train=x_train,
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_ws,
            title=f"{dataset} ({method}): eval projected points in workspace + tool orientation",
            use_pybullet_n6=_ur5_use_pybullet_n6(str(dataset)),
        )
    if dataset in (
        "6d_surf_normal",
        "6d_surf_normal_traj",
    ):
        out_pose = os.path.join(outdir, f"{dataset}_{method}_workspace_pose_orientation.png")
        _plot_workspace_pose_orientation_3d(
            x_train=x_train,
            eval_proj=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_pose,
            title=f"{dataset} ({method}): projected eval poses + orientation z-axis",
            surface_cfg=cfg,
        )
        out_err = os.path.join(outdir, f"{dataset}_{method}_workspace_pose_proj_error_distributions.png")
        _plot_workspace_pose_projection_error_distributions(
            x_before=eval_artifacts.get("x_eval", np.zeros((0, 6), dtype=np.float32)),
            x_after=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_err,
            title=f"{dataset} ({method}): projection errors before/after",
            surface_cfg=cfg,
        )
    if str(dataset) in DUAL_ARM_12D_DATASETS:
        out_pose = os.path.join(outdir, f"{dataset}_{method}_dual_arm_guided_insertion_orientation.png")
        _plot_dual_arm_guided_insertion_orientation_3d(
            x_train=x_train,
            eval_proj=eval_artifacts.get("proj", np.zeros((0, 12), dtype=np.float32)),
            out_path=out_pose,
            title=f"{dataset} ({method}): dual-arm guided insertion poses",
            task_cfg=cfg,
        )
        out_err = os.path.join(outdir, f"{dataset}_{method}_dual_arm_pose_proj_error_distributions.png")
        _plot_dual_arm_pose_projection_error_distributions(
            x_before=eval_artifacts.get("x_eval", np.zeros((0, 12), dtype=np.float32)),
            x_after=eval_artifacts.get("proj", np.zeros((0, 12), dtype=np.float32)),
            out_path=out_err,
            title=f"{dataset} ({method}): dual-arm projection errors before/after",
            grasp_span=float(getattr(cfg, "dual_arm_grasp_span", 1.0)),
            x_span=float(getattr(cfg, "dual_arm_curve_x_span", 1.4)),
            y_amp=float(getattr(cfg, "dual_arm_curve_y_amp", 0.55)),
            y_freq=float(getattr(cfg, "dual_arm_curve_y_freq", 1.0)),
            z_base=float(getattr(cfg, "dual_arm_curve_z_base", 0.2)),
            z_amp=float(getattr(cfg, "dual_arm_curve_z_amp", getattr(cfg, "dual_arm_vertical_half_range", 0.35))),
            z_freq=float(getattr(cfg, "dual_arm_curve_z_freq", 0.7)),
            z_half_range=float(getattr(cfg, "dual_arm_vertical_half_range", getattr(cfg, "dual_arm_curve_z_amp", 0.35))),
        )

    eval_path, ckpt_path = _seeded_eval_model_paths(
        out_root=out_root,
        method=str(method),
        dataset=str(dataset),
        file_tag=str(method),
        seed=int(cfg.seed),
    )

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    cfg_serial = _cfg_to_serializable_dict(cfg, cfg_mapping)

    torch.save(
        {
            "dataset": str(dataset),
            "method": str(method),
            "model_state": model.state_dict(),
            "in_dim": int(x_train.shape[1]),
            "constraint_dim": 1,
            "hidden": int(cfg.hidden),
            "depth": int(cfg.depth),
            "model_type": "feature_normalized_mlp" if getattr(feature_normalizer, "enabled", False) else "mlp",
            "feature_normalizer": {
                "enabled": bool(getattr(feature_normalizer, "enabled", False)),
                "center": getattr(feature_normalizer, "center", np.zeros((x_train.shape[1],), dtype=np.float32)).tolist(),
                "scale": getattr(feature_normalizer, "scale", np.ones((x_train.shape[1],), dtype=np.float32)).tolist(),
                "angle_dims": list(getattr(feature_normalizer, "angle_dims", ())),
            },
            "x_train": x_train,
            "train_stats": train_stats,
            "cfg": cfg_serial,
        },
        ckpt_path,
    )

    return {
        "method": method,
        "dataset": dataset,
        "metrics": _normalize_metrics(metrics),
        "eval_path": eval_path,
        "ckpt_path": ckpt_path,
        "config": cfg_serial,
    }


def run_ecomann_one(
    dataset: str,
    *,
    out_root: str,
    seed_override: int | None,
    cfg_mapping: dict[str, Any],
) -> dict[str, Any]:
    cfg = _build_cfg_from_mapping_strict(ecomann_base.Config, cfg_mapping)
    cfg = _attach_extra_cfg_fields(cfg, cfg_mapping)
    _apply_projector_subcfg(cfg)
    _apply_planner_subcfg(cfg)

    if seed_override is not None:
        cfg.seed = int(seed_override)
    cfg.device = ecomann_base._choose_device(str(cfg.device))
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    set_seed(int(cfg.seed))
    ds = resolve_dataset(
        dataset,
        cfg,
        optimize_ur5_train_only=True,
        ur5_backend=("pybullet" if _ur5_use_pybullet_n6(str(dataset)) else "analytic"),
    )
    x_train = ds["x_train"]
    true_codim = int(ds.get("true_codim", 1))
    feature_normalizer = FeatureNormalizer.fit(x_train, dataset_name=str(dataset), enable=True)
    x_train_model = feature_normalizer.transform(x_train) if feature_normalizer.enabled else x_train
    if feature_normalizer.enabled:
        denom = float(2 * (2 ** int(x_train_model.shape[1])))
        max_mult = max(1e-6, float(max(2, len(x_train_model) - 2)) / max(denom, 1.0))
        if float(cfg.n_local_neighborhood_mult) > max_mult:
            print(
                f"[ecomann-scale-inv] cap n_local_neighborhood_mult "
                f"{float(cfg.n_local_neighborhood_mult):.6g}->{max_mult:.6g} "
                f"for dataset={dataset}"
            )
            cfg.n_local_neighborhood_mult = float(max_mult)
    train_t0 = time.perf_counter()
    model, train_hist, learned_codim, loader_data = ecomann_base.train_ecomann(
        cfg,
        x_train_model,
        force_codim=true_codim,
        return_loader_data=True,
    )
    train_seconds = float(time.perf_counter() - train_t0)
    eval_model: nn.Module = model
    if feature_normalizer.enabled:
        eval_model = FeatureNormalizedModel(model, feature_normalizer).to(str(cfg.device))

    if _is_arm_dataset(dataset):
        post_fn = _wrap_np_pi
    elif _is_workspace_pose_dataset(dataset):
        post_fn = _wrap_workspace_pose_rpy_np
    else:
        post_fn = None
    use_pybullet_n6 = _ur5_use_pybullet_n6(str(dataset))
    embed_fn = (
        lambda q, _name=dataset: _workspace_embed_for_eval(
            _name,
            q,
            ur5_use_pybullet_n6=use_pybullet_n6,
        )
    ) if (_is_arm_dataset(dataset) or _is_workspace_pose_dataset(dataset)) else None

    def project_fn(_model: nn.Module, x0: np.ndarray, _eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
        # EcoMaNN projector: match upstream stopping rules in
        # smp_manifold_learning.motion_planner.feature.Projection
        # (tol/max_iter/step_size and divergence guard).
        # Add aggressive dq clipping to prevent rare Newton blow-ups.
        p_cfg = getattr(cfg, "projector", {}) or {}
        tol = float(p_cfg.get("tol", 1e-5))
        max_iter = int(p_cfg.get("max_iter", 200))
        step_size = float(p_cfg.get("step_size", 1.0))
        diverge_ratio = float(p_cfg.get("diverge_ratio", 2.0))
        max_dq_norm = float(p_cfg.get("max_dq_norm", 1.0))

        x_in_raw = np.asarray(x0, dtype=np.float32)
        x_in = feature_normalizer.transform(x_in_raw) if feature_normalizer.enabled else x_in_raw
        x_out = np.asarray(x_in, dtype=np.float32).copy()
        steps = np.zeros((len(x_out),), dtype=np.float32)

        for i in range(len(x_out)):
            q = x_out[i].astype(np.float64, copy=True)
            y = np.asarray(model.y(q), dtype=np.float64).reshape(-1)
            y_norm = float(np.linalg.norm(y))
            y0 = float(diverge_ratio * y_norm)
            it = 0
            while (y_norm > tol) and (it < max_iter) and (y_norm < y0):
                try:
                    J = np.asarray(model.J(q), dtype=np.float64)
                    dq = np.linalg.lstsq(J, y, rcond=None)[0]
                    if np.isfinite(max_dq_norm) and max_dq_norm > 0.0:
                        dq_norm = float(np.linalg.norm(dq))
                        if np.isfinite(dq_norm) and dq_norm > max_dq_norm:
                            dq = dq * (max_dq_norm / max(dq_norm, 1e-12))
                except Exception:
                    break
                q = q - (step_size * dq)
                y = np.asarray(model.y(q), dtype=np.float64).reshape(-1)
                y_norm = float(np.linalg.norm(y))
                it += 1
            x_out[i] = q.astype(np.float32)
            steps[i] = float(it)
        if feature_normalizer.enabled:
            x_out = feature_normalizer.inverse_transform(x_out)
        return x_out.astype(np.float32), steps.astype(np.float32)

    metrics, eval_cfg, eval_artifacts = run_eval_metrics(
        cfg=cfg,
        method_key="ecomann",
        dataset_name=dataset,
        model=eval_model,
        x_train=x_train,
        project_fn=project_fn,
        embed_fn=embed_fn,
        postprocess_fn=post_fn,
    )
    metrics["train_seconds"] = float(train_seconds)
    _print_eval_lines(dataset, metrics)

    outdir = os.path.join(out_root, "ecomann")
    os.makedirs(outdir, exist_ok=True)
    vis_cfg = _make_vis_cfg_for_method(dataset, cfg, eval_cfg)
    _save_common_method_plots(
        dataset=dataset,
        method_tag="ecomann",
        model=eval_model,
        x_train=x_train,
        outdir=outdir,
        vis_cfg=vis_cfg,
        eval_artifacts=eval_artifacts,
    )
    extra_plots = save_ecomann_training_diagnostics_2d(
        dataset=dataset,
        outdir=outdir,
        x_train=x_train,
        loader_data=loader_data,
        cfg=cfg,
    )
    for p in extra_plots:
        print(f"[saved] {p}")
    out_loss = save_ecomann_loss_curves(
        dataset=dataset,
        outdir=outdir,
        train_hist=train_hist,
    )
    if out_loss is not None:
        print(f"[saved] {out_loss}")
    if dataset in ARM_UP_6D_DATASETS:
        out_dist = os.path.join(outdir, f"{dataset}_ecomann_proj_value_distribution.png")
        _plot_ur5_projection_error_distribution_from_pairs(
            q_eval=eval_artifacts.get("x_eval", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_dist,
            title=f"{dataset} (ecomann): orientation-angle error before/after projection",
            use_pybullet_n6=_ur5_use_pybullet_n6(str(dataset)),
        )
        out_ws = os.path.join(outdir, f"{dataset}_ecomann_eval_proj_workspace_orientation.png")
        _plot_ur5_eval_projection_workspace_orientation_3d(
            q_train=x_train,
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_ws,
            title=f"{dataset} (ecomann): eval projected points in workspace + tool orientation",
            use_pybullet_n6=_ur5_use_pybullet_n6(str(dataset)),
        )
    if dataset in (
        "6d_surf_normal",
        "6d_surf_normal_traj",
    ):
        out_pose = os.path.join(outdir, f"{dataset}_ecomann_workspace_pose_orientation.png")
        _plot_workspace_pose_orientation_3d(
            x_train=x_train,
            eval_proj=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_pose,
            title=f"{dataset} (ecomann): projected eval poses + orientation z-axis",
            surface_cfg=cfg,
        )
        out_err = os.path.join(outdir, f"{dataset}_ecomann_workspace_pose_proj_error_distributions.png")
        _plot_workspace_pose_projection_error_distributions(
            x_before=eval_artifacts.get("x_eval", np.zeros((0, 6), dtype=np.float32)),
            x_after=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_err,
            title=f"{dataset} (ecomann): projection errors before/after",
            surface_cfg=cfg,
        )
    if str(dataset) in DUAL_ARM_12D_DATASETS:
        out_pose = os.path.join(outdir, f"{dataset}_ecomann_dual_arm_guided_insertion_orientation.png")
        _plot_dual_arm_guided_insertion_orientation_3d(
            x_train=x_train,
            eval_proj=eval_artifacts.get("proj", np.zeros((0, 12), dtype=np.float32)),
            out_path=out_pose,
            title=f"{dataset} (ecomann): dual-arm guided insertion poses",
            task_cfg=cfg,
        )
        out_err = os.path.join(outdir, f"{dataset}_ecomann_dual_arm_pose_proj_error_distributions.png")
        _plot_dual_arm_pose_projection_error_distributions(
            x_before=eval_artifacts.get("x_eval", np.zeros((0, 12), dtype=np.float32)),
            x_after=eval_artifacts.get("proj", np.zeros((0, 12), dtype=np.float32)),
            out_path=out_err,
            title=f"{dataset} (ecomann): dual-arm projection errors before/after",
            grasp_span=float(getattr(cfg, "dual_arm_grasp_span", 1.0)),
            x_span=float(getattr(cfg, "dual_arm_curve_x_span", 1.4)),
            y_amp=float(getattr(cfg, "dual_arm_curve_y_amp", 0.55)),
            y_freq=float(getattr(cfg, "dual_arm_curve_y_freq", 1.0)),
            z_base=float(getattr(cfg, "dual_arm_curve_z_base", 0.2)),
            z_amp=float(getattr(cfg, "dual_arm_curve_z_amp", getattr(cfg, "dual_arm_vertical_half_range", 0.35))),
            z_freq=float(getattr(cfg, "dual_arm_curve_z_freq", 0.7)),
            z_half_range=float(getattr(cfg, "dual_arm_vertical_half_range", getattr(cfg, "dual_arm_curve_z_amp", 0.35))),
        )

    eval_path, ckpt_path = _seeded_eval_model_paths(
        out_root=out_root,
        method="ecomann",
        dataset=str(dataset),
        file_tag="ecomann",
        seed=int(cfg.seed),
    )
    cfg_serial = _cfg_to_serializable_dict(cfg, cfg_mapping)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    torch.save(
        {
            "dataset": str(dataset),
            "method": "ecomann",
            "model_state": model.state_dict(),
            "in_dim": int(x_train.shape[1]),
            "constraint_dim": int(learned_codim),
            "hidden_sizes": [int(v) for v in cfg.hidden_sizes],
            "model_type": "feature_normalized_ecomann" if feature_normalizer.enabled else "ecomann",
            "feature_normalizer": {
                "enabled": bool(feature_normalizer.enabled),
                "center": feature_normalizer.center.tolist(),
                "scale": feature_normalizer.scale.tolist(),
                "angle_dims": list(feature_normalizer.angle_dims),
            },
            "x_train": x_train.astype(np.float32),
            "train_hist": train_hist,
            "cfg": cfg_serial,
        },
        ckpt_path,
    )

    return {
        "method": "ecomann",
        "dataset": dataset,
        "metrics": _normalize_metrics(metrics),
        "eval_path": eval_path,
        "ckpt_path": ckpt_path,
        "config": cfg_serial,
    }


def run_one(
    method: str,
    dataset: str,
    *,
    out_root: str,
    seed_override: int | None,
    config_root: str,
    cli_overrides: list[str],
) -> tuple[dict[str, Any], list[str]]:
    if method not in VALID_METHODS:
        raise ValueError(f"unsupported method '{method}', valid={sorted(VALID_METHODS)}")

    cfg_mapping, loaded_paths = _resolve_run_config(
        method,
        dataset,
        config_root=config_root,
        cli_overrides=cli_overrides,
    )

    if method == "oncl":
        result = run_oncl_one(
            dataset,
            out_root=out_root,
            seed_override=seed_override,
            cfg_mapping=cfg_mapping,
        )
    elif method == "igr":
        result = run_oncl_one(
            dataset,
            out_root=out_root,
            seed_override=seed_override,
            cfg_mapping=cfg_mapping,
            method_name="igr",
            method_tag="igr",
            force_single_output=True,
        )
    elif method == "ecomann":
        result = run_ecomann_one(
            dataset,
            out_root=out_root,
            seed_override=seed_override,
            cfg_mapping=cfg_mapping,
        )
    elif method == "vae":
        result = run_vae_one(
            dataset,
            out_root=out_root,
            seed_override=seed_override,
            cfg_mapping=cfg_mapping,
        )
    else:
        result = run_udf_one(
            method,
            dataset,
            out_root=out_root,
            seed_override=seed_override,
            cfg_mapping=cfg_mapping,
        )

    result["loaded_config_paths"] = loaded_paths
    return result, loaded_paths


def _resolve_vae_latent_dim(cfg: VAEConfig, data_dim: int, true_codim: int) -> int:
    if int(cfg.latent_dim) > 0:
        return int(cfg.latent_dim)
    # Default to intrinsic dimension estimate d-codim, clamp to [1, d-1].
    est = int(data_dim) - int(true_codim)
    return max(1, min(int(data_dim) - 1, est))


def run_vae_one(
    dataset: str,
    *,
    out_root: str,
    seed_override: int | None,
    cfg_mapping: dict[str, Any],
) -> dict[str, Any]:
    cfg = _build_cfg_from_mapping_strict(VAEConfig, cfg_mapping)
    cfg = _attach_extra_cfg_fields(cfg, cfg_mapping)
    _apply_projector_subcfg(cfg)
    _apply_planner_subcfg(cfg)

    if seed_override is not None:
        cfg.seed = int(seed_override)

    if str(cfg.device) == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    set_seed(int(cfg.seed))

    ur5_backend = str(getattr(cfg, "ur5_backend", "auto")).lower().strip()
    if ur5_backend in ("", "auto"):
        ur5_backend = "pybullet" if _ur5_use_pybullet_n6(str(dataset)) else "analytic"
    ds = resolve_dataset(
        dataset,
        cfg,
        optimize_ur5_train_only=True,
        ur5_backend=ur5_backend,
    )
    x_train = ds["x_train"]
    data_dim = int(x_train.shape[1])
    true_codim = int(ds.get("true_codim", 1))
    latent_dim = _resolve_vae_latent_dim(cfg, data_dim, true_codim)
    hidden = tuple(int(v) for v in cfg.hidden_dims)
    train_t0 = time.perf_counter()

    vae_model = VariationalAutoEncoder(in_dim=data_dim, latent_dim=latent_dim, hidden=hidden).to(cfg.device)
    train_cfg = vae_base.TrainConfig(
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        beta_final=float(cfg.beta_final),
        warmup_epochs=int(cfg.warmup_epochs),
        train_log_every=50,
    )
    x_train_t = torch.from_numpy(x_train.astype(np.float32)).to(cfg.device)
    train_history = vae_base.train_variational_autoencoder(
        vae_model, x_train_t, train_cfg, torch.device(cfg.device)
    )
    train_seconds = float(time.perf_counter() - train_t0)

    field_model = vae_base.VAEProjectorField(vae_model).to(cfg.device)
    field_model.eval()

    def _vae_project_once_np(x0: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.from_numpy(x0.astype(np.float32)).to(cfg.device)
            return field_model.project_tensor(xt).detach().cpu().numpy().astype(np.float32)

    def _vae_project_traj_np(x0: np.ndarray) -> np.ndarray:
        x_in = x0.astype(np.float32)
        x_out = _vae_project_once_np(x_in)
        return np.stack([x_in, x_out], axis=0).astype(np.float32)

    def project_fn(_model: nn.Module, x0: np.ndarray, _eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
        x_proj = _vae_project_once_np(x0)
        steps = np.ones((len(x_proj),), dtype=np.float32)
        return x_proj, steps

    if _is_arm_dataset(dataset):
        post_fn = _wrap_np_pi
    elif _is_workspace_pose_dataset(dataset):
        post_fn = _wrap_workspace_pose_rpy_np
    else:
        post_fn = None
    use_pybullet_n6 = str(ds.get("ur5_backend", ur5_backend)).lower() == "pybullet"
    embed_fn = (
        lambda q, _name=dataset: _workspace_embed_for_eval(
            _name,
            q,
            ur5_use_pybullet_n6=use_pybullet_n6,
        )
    ) if (_is_arm_dataset(dataset) or _is_workspace_pose_dataset(dataset)) else None

    metrics, eval_cfg, eval_artifacts = run_eval_metrics(
        cfg=cfg,
        method_key="vae",
        dataset_name=dataset,
        model=field_model,
        x_train=x_train,
        project_fn=project_fn,
        embed_fn=embed_fn,
        postprocess_fn=post_fn,
    )
    metrics["train_seconds"] = float(train_seconds)

    # Extra VAE generative metrics on prior samples z~N(0,1):
    # - gen_manifold_dist: mean NN distance from generated points to GT manifold
    # - gen_chamfer: bidirectional NN distance sum between generated set and GT manifold
    n_gen = max(64, int(getattr(eval_cfg, "eval_proj_n_points", 4096)))
    _, x_gen = vae_base.sample_prior_decode(
        vae_model,
        latent_dim=int(latent_dim),
        n_sample=n_gen,
        seed=int(cfg.seed) + 17,
        device=str(cfg.device),
    )
    gt_grid = resolve_gt_grid(dataset, eval_cfg, x_train=x_train).astype(np.float32)
    if post_fn is not None:
        x_gen = post_fn(x_gen).astype(np.float32)
        gt_grid = post_fn(gt_grid).astype(np.float32)
    if embed_fn is not None:
        gen_metric = embed_fn(x_gen).astype(np.float32)
        gt_metric = embed_fn(gt_grid).astype(np.float32)
    else:
        gen_metric = x_gen.astype(np.float32)
        gt_metric = gt_grid.astype(np.float32)
    d_gen_to_gt = _nn_dist_numpy(gen_metric, gt_metric)
    d_gt_to_gen = _nn_dist_numpy(gt_metric, gen_metric)
    metrics["gen_manifold_dist"] = float(np.mean(d_gen_to_gt))
    metrics["gen_chamfer"] = float(np.mean(d_gen_to_gt) + np.mean(d_gt_to_gen))

    _print_eval_lines(dataset, metrics)
    outdir = os.path.join(out_root, "vae")
    os.makedirs(outdir, exist_ok=True)

    if bool(cfg.viz_enable) and int(x_train.shape[1]) != 6:
        x_eval = eval_artifacts["x_eval"].astype(np.float32)
        x_proj = eval_artifacts["proj"].astype(np.float32)
        if int(cfg.viz_max_eval_points) > 0 and len(x_eval) > int(cfg.viz_max_eval_points):
            idx = np.random.default_rng(int(cfg.seed) + 99).choice(
                len(x_eval), size=int(cfg.viz_max_eval_points), replace=False
            )
            x_eval = x_eval[idx].astype(np.float32)
            x_proj = x_proj[idx].astype(np.float32)

        gt_grid = resolve_gt_grid(dataset, eval_cfg, x_train=x_train).astype(np.float32)
        if post_fn is not None:
            gt_grid = post_fn(gt_grid).astype(np.float32)

        if embed_fn is not None:
            x_eval_metric = embed_fn(x_eval).astype(np.float32)
            gt_metric = embed_fn(gt_grid).astype(np.float32)
        else:
            x_eval_metric = x_eval
            gt_metric = gt_grid
        tau = float(eval_cfg.eval_tau_ratio) * float(
            np.mean(gt_metric.max(axis=0) - gt_metric.min(axis=0))
        )
        d_true = _nn_dist_numpy(x_eval_metric, gt_metric)
        y_true = (d_true < tau).astype(np.int64)

        with torch.no_grad():
            z_eval = vae_base.encode_mu(vae_model, x_eval, device=str(cfg.device))
            n_samp = max(1, int(cfg.viz_sample_latent_n) * 2)
            z_samp, x_dec = vae_base.sample_prior_decode(
                vae_model,
                latent_dim=int(latent_dim),
                n_sample=n_samp,
                seed=int(cfg.seed) + 7,
                device=str(cfg.device),
            )

        cache = {
            "x": x_eval,
            "y_true": y_true,
            "err": np.linalg.norm(x_eval - x_proj, axis=1).astype(np.float32),
            "x_proj": x_proj,
        }
        vae_pack = (cache, z_eval, x_dec, z_samp)
        ds_vis = SimpleNamespace(name=str(dataset), dim=int(data_dim))
        viz_path = os.path.join(outdir, f"{dataset}_vae_visualize_all.png")
        vae_plots.visualize_all(
            ds=ds_vis,
            x_train=x_train,
            ae_pack=None,
            vae_pack=vae_pack,
            latent_dim=int(latent_dim),
            save_path=viz_path,
            show=False,
        )
        print(f"saved: {viz_path}")
    out_loss = os.path.join(outdir, f"{dataset}_vae_loss_curves.png")
    vae_plots.plot_vae_loss_curves(
        train_history,
        out_path=out_loss,
        title=f"{dataset}: VAE Losses",
    )
    print(f"saved: {out_loss}")

    vis_cfg = _make_vis_cfg_for_method(dataset, cfg, eval_cfg)
    _save_common_method_plots(
        dataset=dataset,
        method_tag="vae",
        model=field_model,
        x_train=x_train,
        outdir=outdir,
        vis_cfg=vis_cfg,
        eval_artifacts=eval_artifacts,
        project_traj_fn=_vae_project_traj_np,
    )
    if int(x_train.shape[1]) == 2:
        if str(dataset) == "2d_planar_arm_line_n2":
            out_plan = os.path.join(outdir, f"{dataset}_vae_planning_demo.png")
            _plot_planar_arm_planning(
                field_model,
                str(dataset),
                x_train,
                out_plan,
                cfg,
                render_pybullet=False,
            )
            print(f"saved: {out_plan}")
        x_eval = np.asarray(eval_artifacts.get("x_eval", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        grid_vis = ds.get("grid", None)
        if grid_vis is None:
            grid_vis = x_train
        grid_vis = np.asarray(grid_vis, dtype=np.float32)
        if str(dataset) != "2d_planar_arm_line_n2" and len(x_eval) >= 8 and len(grid_vis) > 0:
            plan_rng = np.random.default_rng(int(cfg.seed) + 77)
            n_pairs = 4
            replace = len(x_eval) < 2 * n_pairs
            picks = plan_rng.choice(len(x_eval), size=2 * n_pairs, replace=replace)
            pairs = [(x_eval[picks[2 * i]], x_eval[picks[2 * i + 1]]) for i in range(n_pairs)]
            plans_proj: list[np.ndarray] = []
            plans_constr: list[np.ndarray] = []
            pln = getattr(cfg, "planner", None)
            planner_name = str(pln.get("method", "traj_opt")) if isinstance(pln, dict) else "traj_opt"
            n_waypoints = int(pln.get("steps", 64)) + 1 if isinstance(pln, dict) else 65
            use_linear = planner_name.lower() == "linear_proj"
            eps_used = float(eval_artifacts.get("eval_eps_used", 1e-6))
            if not np.isfinite(eps_used) or eps_used <= 0.0:
                eps_used = 1e-6
            for x_start, x_goal in pairs:
                x_start = true_projection(x_start[None, :], grid_vis)[0][0]
                x_goal = true_projection(x_goal[None, :], grid_vis)[0][0]
                if use_linear:
                    periodic = resolve_periodic_mode(
                        periodic_joint=bool(ds.get("periodic_joint", False)),
                        dataset_name=dataset,
                    )
                    planned = plan_linear_then_model_project(
                        model=field_model,
                        x_start=x_start,
                        x_goal=x_goal,
                        device=str(cfg.device),
                        n_waypoints=n_waypoints,
                        periodic=bool(periodic),
                    )
                else:
                    planned = plan_path(
                        model=field_model,
                        x_start=x_start,
                        x_goal=x_goal,
                        cfg=cfg,
                        planner_name=planner_name,
                        n_waypoints=n_waypoints,
                        dataset_name=dataset,
                        periodic_joint=bool(ds.get("periodic_joint", False)),
                        f_abs_stop=eps_used,
                    )
                if use_linear:
                    plans_proj.append(planned)
                else:
                    plans_constr.append(planned)
            plot_planned_paths(
                field_model,
                x_train,
                grid_vis,
                plans_proj,
                plans_constr,
                _make_vis_cfg_for_method(dataset, cfg, eval_cfg),
                out_path=os.path.join(outdir, f"{dataset}_vae_planner_paths.png"),
                title=f"{dataset}: VAE Planned Paths",
                zero_level_eps=eps_used,
            )
    elif int(x_train.shape[1]) == 3:
        if _is_arm_dataset(dataset):
            out_plan = os.path.join(outdir, f"{dataset}_vae_planning_demo.png")
            _plot_planar_arm_planning(
                field_model,
                str(dataset),
                x_train,
                out_plan,
                cfg,
                render_pybullet=False,
            )
        else:
            x_eval = np.asarray(eval_artifacts.get("x_eval", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            grid_vis = ds.get("grid", None)
            if grid_vis is None:
                grid_vis = x_train
            grid_vis = np.asarray(grid_vis, dtype=np.float32)
            if len(x_eval) >= 8 and len(grid_vis) > 0:
                plan_rng = np.random.default_rng(int(cfg.seed) + 77)
                n_pairs = 4
                replace = len(x_eval) < 2 * n_pairs
                picks = plan_rng.choice(len(x_eval), size=2 * n_pairs, replace=replace)
                pairs = [(x_eval[picks[2 * i]], x_eval[picks[2 * i + 1]]) for i in range(n_pairs)]
                plans_proj: list[np.ndarray] = []
                plans_constr: list[np.ndarray] = []
                pln = getattr(cfg, "planner", None)
                planner_name = str(pln.get("method", "traj_opt")) if isinstance(pln, dict) else "traj_opt"
                n_waypoints = int(pln.get("steps", 64)) + 1 if isinstance(pln, dict) else 65
                use_linear = planner_name.lower() == "linear_proj"
                eps_used = float(eval_artifacts.get("eval_eps_used", 1e-6))
                if not np.isfinite(eps_used) or eps_used <= 0.0:
                    eps_used = 1e-6
                for x_start, x_goal in pairs:
                    x_start = true_projection(x_start[None, :], grid_vis)[0][0]
                    x_goal = true_projection(x_goal[None, :], grid_vis)[0][0]
                    if use_linear:
                        periodic = resolve_periodic_mode(
                            periodic_joint=bool(ds.get("periodic_joint", False)),
                            dataset_name=dataset,
                        )
                        planned = plan_linear_then_model_project(
                            model=field_model,
                            x_start=x_start,
                            x_goal=x_goal,
                            device=str(cfg.device),
                            n_waypoints=n_waypoints,
                            periodic=bool(periodic),
                        )
                    else:
                        planned = plan_path(
                            model=field_model,
                            x_start=x_start,
                            x_goal=x_goal,
                            cfg=cfg,
                            planner_name=planner_name,
                            n_waypoints=n_waypoints,
                            dataset_name=dataset,
                            periodic_joint=bool(ds.get("periodic_joint", False)),
                            f_abs_stop=eps_used,
                        )
                    if use_linear:
                        plans_proj.append(planned)
                    else:
                        plans_constr.append(planned)
                labels3 = _axis_labels_for_dataset(dataset, 3)
                plot_planned_paths_3d(
                    x_train=x_train,
                    plans_proj=plans_proj,
                    plans_constr=plans_constr,
                    out_path=os.path.join(outdir, f"{dataset}_vae_planner_paths_3d.png"),
                    title=f"{dataset}: VAE Planned Paths (3D)",
                    axis_labels=(labels3[0], labels3[1], labels3[2]),
                )
    elif int(x_train.shape[1]) == 6 and str(dataset) in ARM_UP_6D_DATASETS:
        out_plan = os.path.join(outdir, f"{dataset}_vae_planning_demo.png")
        _plot_planar_arm_planning(
            field_model,
            str(dataset),
            x_train,
            out_plan,
            cfg,
            render_pybullet=False,
        )

    if dataset in ARM_UP_6D_DATASETS:
        out_dist = os.path.join(outdir, f"{dataset}_vae_proj_value_distribution.png")
        _plot_ur5_projection_error_distribution_from_pairs(
            q_eval=eval_artifacts.get("x_eval", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_dist,
            title=f"{dataset} (vae): orientation-angle error before/after projection",
            use_pybullet_n6=_ur5_use_pybullet_n6(str(dataset)),
        )
        out_ws = os.path.join(outdir, f"{dataset}_vae_eval_proj_workspace_orientation.png")
        _plot_ur5_eval_projection_workspace_orientation_3d(
            q_train=x_train,
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_ws,
            title=f"{dataset} (vae): eval projected points in workspace + tool orientation",
            use_pybullet_n6=_ur5_use_pybullet_n6(str(dataset)),
        )
        out_ws_gen = os.path.join(outdir, f"{dataset}_vae_eval_workspace_orientation.png")
        _plot_ur5_eval_projection_workspace_orientation_3d(
            q_train=x_train,
            q_eval_proj=x_gen.astype(np.float32),
            out_path=out_ws_gen,
            title=f"{dataset} (vae): generated (z~N(0,1)) in workspace + tool orientation",
            use_pybullet_n6=_ur5_use_pybullet_n6(str(dataset)),
        )
    if dataset in (
        "6d_surf_normal",
        "6d_surf_normal_traj",
    ):
        out_pose = os.path.join(outdir, f"{dataset}_vae_workspace_pose_orientation.png")
        _plot_workspace_pose_orientation_3d(
            x_train=x_train,
            eval_proj=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_pose,
            title=f"{dataset} (vae): projected eval poses + orientation z-axis",
            surface_cfg=cfg,
        )
        out_err = os.path.join(outdir, f"{dataset}_vae_workspace_pose_proj_error_distributions.png")
        _plot_workspace_pose_projection_error_distributions(
            x_before=eval_artifacts.get("x_eval", np.zeros((0, 6), dtype=np.float32)),
            x_after=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
            out_path=out_err,
            title=f"{dataset} (vae): projection errors before/after",
            surface_cfg=cfg,
        )
    if str(dataset) in DUAL_ARM_12D_DATASETS:
        out_pose = os.path.join(outdir, f"{dataset}_vae_dual_arm_guided_insertion_orientation.png")
        _plot_dual_arm_guided_insertion_orientation_3d(
            x_train=x_train,
            eval_proj=eval_artifacts.get("proj", np.zeros((0, 12), dtype=np.float32)),
            out_path=out_pose,
            title=f"{dataset} (vae): dual-arm guided insertion poses",
            task_cfg=cfg,
        )
        out_err = os.path.join(outdir, f"{dataset}_vae_dual_arm_pose_proj_error_distributions.png")
        _plot_dual_arm_pose_projection_error_distributions(
            x_before=eval_artifacts.get("x_eval", np.zeros((0, 12), dtype=np.float32)),
            x_after=eval_artifacts.get("proj", np.zeros((0, 12), dtype=np.float32)),
            out_path=out_err,
            title=f"{dataset} (vae): dual-arm projection errors before/after",
            grasp_span=float(getattr(cfg, "dual_arm_grasp_span", 1.0)),
            x_span=float(getattr(cfg, "dual_arm_curve_x_span", 1.4)),
            y_amp=float(getattr(cfg, "dual_arm_curve_y_amp", 0.55)),
            y_freq=float(getattr(cfg, "dual_arm_curve_y_freq", 1.0)),
            z_base=float(getattr(cfg, "dual_arm_curve_z_base", 0.2)),
            z_amp=float(getattr(cfg, "dual_arm_curve_z_amp", getattr(cfg, "dual_arm_vertical_half_range", 0.35))),
            z_freq=float(getattr(cfg, "dual_arm_curve_z_freq", 0.7)),
            z_half_range=float(getattr(cfg, "dual_arm_vertical_half_range", getattr(cfg, "dual_arm_curve_z_amp", 0.35))),
        )

    eval_path, ckpt_path = _seeded_eval_model_paths(
        out_root=out_root,
        method="vae",
        dataset=str(dataset),
        file_tag="vae",
        seed=int(cfg.seed),
    )
    cfg_serial = _cfg_to_serializable_dict(cfg, cfg_mapping)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            "dataset": str(dataset),
            "method": "vae",
            "model_state": vae_model.state_dict(),
            "in_dim": data_dim,
            "latent_dim": latent_dim,
            "hidden_dims": hidden,
            "x_train": x_train,
            "cfg": cfg_serial,
        },
        ckpt_path,
    )

    return {
        "method": "vae",
        "dataset": dataset,
        "metrics": _normalize_metrics(metrics),
        "eval_path": eval_path,
        "ckpt_path": ckpt_path,
        "config": cfg_serial,
    }
