#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import time
import json
from dataclasses import dataclass, asdict, field
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from evaluation.evaluator import eval_bounds_from_train
from evaluation.evaluator import resolve_gt_grid
from datasets.constraint_datasets import set_seed
from plotting.oncl_plots import (
    _plot_constraint_surface_paper_3d,
    _plot_constraint_2d,
    _plot_highdim_pca,
    _plot_projection_value_distribution,
    _plot_ur5_eval_projection_workspace_orientation_3d,
    _plot_workspace_pose_projection_error_distributions,
    _plot_workspace_pose_orientation_3d,
    _plot_training_diagnostics,
    _plot_zero_surfaces_3d,
    _render_ur5_pybullet_trajectories,
)
from plotting.dataaug_plots import plot_planned_paths
from methods import dataaug as dataaug_method
from experiments.dataset_resolve import resolve_dataset
from evaluation.eval_runner import run_eval_metrics
from planning.kinematics import (
    is_arm_dataset as _is_arm_dataset,
    is_workspace_pose_dataset as _is_workspace_pose_dataset,
    workspace_embed_for_eval as shared_workspace_embed_for_eval,
    wrap_np_pi as _wrap_np_pi,
    wrap_workspace_pose_rpy_np as _wrap_workspace_pose_rpy_np,
)
from methods.utils.projection import (
    project_points_with_steps_numpy,
    project_trajectory_numpy,
)
from methods.models.mlp import MLP
from planning.planner import plan_path
from planning.planner import _plot_planar_arm_planning
from methods.codim_utils import estimate_codim_local_pca

DEFAULT_DATASETS = [
    # "3d_torus_surface",

    "3d_spatial_arm_ellip_n3",
    # "3d_vz_2d_sine",
    # "3d_0z_2d_ellipse"
    # "3d_spatial_arm_ellip_n3",

    # "3d_spatial_arm_plane_n3",
    # "6d_spatial_arm_up_n6_py",
    # "6d_spatial_arm_up_n6",

]
DEFAULT_OUTDIR = "outputs/oncl"

@dataclass
class DemoCfg:
    seed: int = 2116
    device: str = "auto"
    n_train: int = 512
    traj_gene_n_grid: int = 4096
    hidden: int = 128
    depth: int = 3
    lr: float = 2e-4
    lr_decay_step: int = 1000  # <=0 disables LR decay
    lr_decay_gamma: float = 0.5  # multiplicative decay factor
    epochs: int = 2000
    batch_size: int = 128
    lam_oncl: float = 0.25
    lam_oncl_ortho: float = 1.0
    oncl_near_ratio: float = 0.85
    oncl_near_std_ratio: float = 0.05
    train_sample_pad_ratio: float = 0.6
    train_min_axis_span_ratio: float = 0.08
    metric_eval_every: int = 1
    projector: dict[str, Any] = field(
        default_factory=lambda: {"alpha": 0.3, "steps": 100, "min_steps": 30}
    )
    planner: dict[str, Any] = field(
        default_factory=lambda: {
            "pair_min_ratio": 0.15,
            "pair_max_ratio": 0.35,
            "pair_tries": 1200,
            "init_mode": "joint_spline",
            "joint_mid_noise": 0.0,
            "lam_manifold": 1.0,
            "lam_len_joint": 0.40,
            "opt_steps": 1240,
            "opt_lr": 0.01,
            "opt_lam_smooth": 0.2,
            "trust_scale": 0.8,
            "method": "traj_opt",
            "anim_fps": 6,
            "anim_stride": 1,
            "save_gif": True,
            "pybullet_render": False,
            "pybullet_real_time_dt": 0.06
        }
    )
    viz_proj_traj_count: int = 64

    constraint_dim: Any = "auto"     # Can be int or "auto".
    codim_auto_sample_ratio: float = 0.2
    codim_auto_k_neighbors: int = 16
    codim_auto_const_axis_std_ratio: float = 1e-3
    codim_auto_strict_check: bool = True

    show_3d_plot: bool = not False
    surface_plot_n: int = 28
    surface_eval_chunk: int = 8192
    surface_max_points: int = 5000
    plot_train_max_points: int = 1200
    plot_traj_max_count: int = 24
    plot_traj_stride: int = 2
    surface_use_marching_cubes: bool = True
    train_log_every: int = 50
    # UR5 kinematics/render config now lives in datasets/ur5_pybullet_utils.py

# ----------------------------------------------------------------------
# Config layering:
# 1) BASE_CFG = current default values (from DemoCfg)
# 2) DATASET_OVERRIDES = per-dataset custom parameters
# ----------------------------------------------------------------------
BASE_CFG: dict[str, Any] = asdict(DemoCfg())

DATASET_OVERRIDES: dict[str, dict[str, Any]] = {
    # Examples (edit as needed):
    "6d_spatial_arm_up_n6_py": {
        "epochs": 4,#000,
        "constraint_dim": 2,
        "lr": 2e-4,
    },
    "6d_spatial_arm_up_n6": {
        "epochs": 3000,
        "constraint_dim": 2,
        "lr": 3e-4,

    },
}

def build_cfg(dataset_name: str) -> DemoCfg:
    cfg_dict = dict(BASE_CFG)
    ds_ov = DATASET_OVERRIDES.get(str(dataset_name), {})
    cfg_dict.update(ds_ov)
    return DemoCfg(**cfg_dict)


def _choose_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _proj_alpha(cfg: Any) -> float:
    proj = getattr(cfg, "projector", None)
    if isinstance(proj, dict) and "alpha" in proj:
        return float(proj["alpha"])
    return float(getattr(cfg, "proj_alpha", 0.3))


def _proj_steps(cfg: Any) -> int:
    proj = getattr(cfg, "projector", None)
    if isinstance(proj, dict) and "steps" in proj:
        return int(proj["steps"])
    return int(getattr(cfg, "proj_steps", 100))


def _proj_min_steps(cfg: Any) -> int:
    proj = getattr(cfg, "projector", None)
    if isinstance(proj, dict) and "min_steps" in proj:
        return int(proj["min_steps"])
    return int(getattr(cfg, "proj_min_steps", 30))


def _planner_bool(cfg: Any, key: str, default: bool) -> bool:
    pln = getattr(cfg, "planner", None)
    if isinstance(pln, dict) and key in pln:
        return bool(pln[key])
    return bool(default)


def _enable_interactive_backend_if_possible() -> bool:
    for name in ("QtAgg", "TkAgg", "Qt5Agg"):
        try:
            plt.switch_backend(name)
            return True
        except Exception:
            continue
    return False


def _uniform_in_box(n: int, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return np.random.uniform(mins, maxs, size=(n, len(mins))).astype(np.float32)


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
    model: nn.Module,
    dataset_name: str,
    x_train: np.ndarray,
    cfg: DemoCfg,
    eval_cfg: Any,
    eval_artifacts: dict[str, np.ndarray],
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

    gt_grid = resolve_gt_grid(str(dataset_name), eval_cfg, x_train=x_train).astype(np.float32)
    if gt_grid.ndim != 2 or gt_grid.shape[1] != 2 or len(gt_grid) == 0:
        return None, None
    d_final = _nn_dist_numpy(proj, gt_grid)
    if len(d_final) == 0 or not np.isfinite(d_final).any():
        return None, None
    n_worst = int(min(24, max(1, np.ceil(0.05 * len(d_final)))))
    idx = np.argsort(d_final)[-n_worst:]
    x0_worst = x_eval[idx].astype(np.float32, copy=False)

    eps_used = float(eval_artifacts.get("eval_eps_used", float("nan")))
    if not np.isfinite(eps_used) or eps_used <= 0.0:
        with torch.no_grad():
            f_on = model(torch.from_numpy(x_train.astype(np.float32)).to(cfg.device))
            if f_on.dim() == 1:
                f_on = f_on.unsqueeze(1)
            h_on = torch.linalg.norm(f_on, dim=1).detach().cpu().numpy().reshape(-1)
        eps_used = float(np.percentile(np.abs(h_on), float(getattr(eval_cfg, "zero_eps_quantile", 90.0))))

    worst_traj = project_trajectory_numpy(
        model,
        x0_worst,
        device=str(cfg.device),
        proj_steps=_proj_steps(cfg),
        proj_alpha=_proj_alpha(cfg),
        proj_min_steps=_proj_min_steps(cfg),
        f_abs_stop=eps_used,
    )
    return worst_traj, x0_worst


def _resolve_dataset(name: str, cfg: DemoCfg) -> dict[str, Any]:
    ur5_backend = "pybullet" if str(name) == "6d_spatial_arm_up_n6" else "analytic"
    return resolve_dataset(
        name,
        cfg,
        optimize_ur5_train_only=True,
        ur5_backend=ur5_backend,
    )


def _oncl_multi_constraint(model: nn.Module, x: torch.Tensor, lam_ortho: float) -> torch.Tensor:
    # Standard ONCL objective (fixed codim).
    f = model(x)
    if f.dim() == 1:
        f = f.unsqueeze(1)
    k = f.shape[1]
    grads = []
    for i in range(k):
        gi = torch.autograd.grad(f[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
        grads.append(gi)
    g = torch.stack(grads, dim=1)  # (B, k, d)
    row_norm = torch.linalg.norm(g, dim=2)
    loss_row = ((row_norm - 1.0) ** 2).mean()
    if k <= 1:
        return loss_row
    gram = torch.matmul(g, g.transpose(1, 2))
    eye = torch.eye(k, device=x.device, dtype=x.dtype).unsqueeze(0)
    loss_ortho = ((gram - eye) ** 2).mean()
    return loss_row + float(lam_ortho) * loss_ortho


def _sample_xr_batch(
    xb: torch.Tensor,
    x_train_dev: torch.Tensor,
    mins_t: torch.Tensor,
    maxs_t: torch.Tensor,
    span_t: torch.Tensor,
    near_ratio: float,
    near_std_ratio: float,
) -> torch.Tensor:
    bsz, dim = xb.shape
    n_near = max(0, min(int(round(bsz * near_ratio)), bsz))
    n_box = bsz - n_near
    parts = []
    if n_near > 0:
        idx = torch.randint(0, x_train_dev.shape[0], size=(n_near,), device=xb.device)
        x_near = x_train_dev[idx]
        x_near = x_near + torch.randn_like(x_near) * (near_std_ratio * span_t)
        x_near = torch.max(torch.min(x_near, maxs_t), mins_t)
        parts.append(x_near)
    if n_box > 0:
        x_box = torch.rand((n_box, dim), device=xb.device)
        x_box = x_box * (maxs_t - mins_t) + mins_t
        parts.append(x_box)
    xr = torch.cat(parts, dim=0)
    xr = xr[torch.randperm(xr.shape[0], device=xb.device)]
    xr.requires_grad_(True)
    return xr


def train_oncl_only(
    cfg: DemoCfg, x_train: np.ndarray, constraint_dim: int
) -> tuple[nn.Module, dict[str, np.ndarray]]:
    x_t = torch.from_numpy(x_train)
    ds = torch.utils.data.TensorDataset(x_t)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(cfg.device == "cuda"),
        num_workers=0,
    )
    model = MLP(
        in_dim=x_train.shape[1],
        hidden=cfg.hidden,
        depth=cfg.depth,
        out_dim=max(1, int(constraint_dim)),
    ).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = None
    if int(cfg.lr_decay_step) > 0 and float(cfg.lr_decay_gamma) < 1.0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=int(cfg.lr_decay_step),
            gamma=float(cfg.lr_decay_gamma),
        )

    mins = x_train.min(axis=0)
    maxs = x_train.max(axis=0)
    span_raw = maxs - mins
    ref_span = max(float(np.max(span_raw)), 1e-6)
    min_axis_span = max(0.0, float(cfg.train_min_axis_span_ratio)) * ref_span
    span = np.maximum(span_raw, min_axis_span)
    scale = max(1.0 + float(cfg.train_sample_pad_ratio), 1e-6)
    center = 0.5 * (mins + maxs)
    half = 0.5 * span * scale
    mins, maxs = center - half, center + half
    mins_t = torch.from_numpy(mins.astype(np.float32)).to(cfg.device)
    maxs_t = torch.from_numpy(maxs.astype(np.float32)).to(cfg.device)
    x_train_dev = x_t.to(cfg.device)
    span_t = (maxs_t - mins_t).clamp_min(1e-6)
    near_ratio = float(min(max(cfg.oncl_near_ratio, 0.0), 1.0))
    near_std_ratio = float(max(cfg.oncl_near_std_ratio, 1e-5))

    hist: dict[str, list[np.ndarray | float]] = {
        "epoch": [],
        "f_abs_mean": [],
        "grad_norm_mean": [],
        "ortho_err": [],
    }

    model.train()
    n_ep_total = int(cfg.epochs)

    for ep in range(n_ep_total):
        for (xb,) in dl:
            xb = xb.to(cfg.device)
            opt.zero_grad(set_to_none=True)

            f_on = model(xb)
            loss_on = (f_on ** 2).mean()

            xr = _sample_xr_batch(xb, x_train_dev, mins_t, maxs_t, span_t, near_ratio, near_std_ratio)
            loss_eik = _oncl_multi_constraint(model, xr, lam_ortho=float(cfg.lam_oncl_ortho))

            loss = loss_on + cfg.lam_oncl * loss_eik
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if (ep % max(1, int(cfg.metric_eval_every))) == 0:
            x_eval = x_train_dev.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                f_eval = model(x_eval)
                if f_eval.dim() == 1:
                    f_eval = f_eval.unsqueeze(1)
                k = f_eval.shape[1]
                grads = []
                for i in range(k):
                    gi = torch.autograd.grad(
                        f_eval[:, i].sum(),
                        x_eval,
                        create_graph=False,
                        retain_graph=(i < k - 1),
                    )[0]
                    grads.append(gi)
                g = torch.stack(grads, dim=1)  # (N, k, d)
                grad_norm = torch.linalg.norm(g, dim=2)  # (N, k)
                if k > 1:
                    gram = torch.matmul(g, g.transpose(1, 2))
                    eye = torch.eye(k, device=g.device, dtype=g.dtype).unsqueeze(0)
                    ortho_err = torch.mean(torch.abs(gram - eye)).item()
                else:
                    ortho_err = 0.0
            hist["epoch"].append(float(ep + 1))
            hist["f_abs_mean"].append(torch.mean(torch.abs(f_eval), dim=0).detach().cpu().numpy())
            hist["grad_norm_mean"].append(torch.mean(grad_norm, dim=0).detach().cpu().numpy())
            hist["ortho_err"].append(float(ortho_err))
            if ((ep + 1) % max(1, int(cfg.train_log_every))) == 0 or ep == 0 or (ep + 1) == int(cfg.epochs):
                f_m = np.mean(np.abs(f_eval.detach().cpu().numpy()), axis=0)
                g_m = np.mean(grad_norm.detach().cpu().numpy(), axis=0)
                f_s = ", ".join([f"f{i+1}={v:.4f}" for i, v in enumerate(f_m)])
                g_s = ", ".join([f"|grad f{i+1}|={v:.4f}" for i, v in enumerate(g_m)])
                lr_now = float(opt.param_groups[0]["lr"])
                method_name = str(getattr(cfg, "train_method_name", "oncl"))
                print(
                    f"[train] method={method_name} ep={ep+1:4d}/{cfg.epochs} "
                    f"| lr={lr_now:.2e} | {f_s} | {g_s} | ortho={ortho_err:.5f}"
                )
        if scheduler is not None:
            scheduler.step()

    learned_codim = int(max(1, int(constraint_dim)))

    out_hist = {
        "epoch": np.asarray(hist["epoch"], dtype=np.float32),
        "f_abs_mean": np.asarray(hist["f_abs_mean"], dtype=np.float32),
        "grad_norm_mean": np.asarray(hist["grad_norm_mean"], dtype=np.float32),
        "ortho_err": np.asarray(hist["ortho_err"], dtype=np.float32),
        "learned_codim": np.asarray([learned_codim], dtype=np.int32),
    }
    return model, out_hist


def run_dataset(name: str, cfg: DemoCfg, outdir: str) -> None:
    set_seed(cfg.seed)
    print(f"[run] start dataset={name}")
    ds = _resolve_dataset(name, cfg)
    x_train = ds["x_train"]
    data_dim = int(ds["data_dim"])
    axis_labels = ds["axis_labels"]
    print(f"[run] dataset ready: data_dim={data_dim}, n_train={len(x_train)}")

    true_codim = int(ds.get("true_codim", 1))
    if str(cfg.constraint_dim).lower() == "auto":
        try:
            est = estimate_codim_local_pca(
                x_train,
                periodic_joint=_is_arm_dataset(name),
                sample_ratio=float(cfg.codim_auto_sample_ratio),
                k_neighbors=int(cfg.codim_auto_k_neighbors),
                const_axis_std_ratio=float(cfg.codim_auto_const_axis_std_ratio),
                seed=int(cfg.seed) + 1000,
            )
            codim = int(est["estimated_codim"])
            print(
                f"[codim-auto] {name}: estimated={codim}, true={true_codim}, "
                f"mode_frac={est['mode_fraction']:.3f}, k={est['k_neighbors']}, n_sample={est['n_sample']}, "
                f"const_axes={est['n_const_axes']}, d_eff={est['d_eff']}"
            )
            if codim != true_codim:
                print(
                    "\033[31m"
                    f"[warn] [codim-auto] mismatch for {name}: estimated={codim}, true={true_codim}; "
                    "fallback to true codim for training."
                    "\033[0m"
                )
                codim = int(true_codim)
        except Exception as e:
            print(
                "\033[31m"
                f"[warn] [codim-auto] failed for {name}: {e}; fallback to true codim={true_codim}."
                "\033[0m"
            )
            codim = int(true_codim)
    else:
        codim = max(1, int(cfg.constraint_dim))
        if data_dim == 2 and codim != 1:
            print(f"[info] {name}: forcing constraint_dim=1 for 2D data")
            codim = 1

    model, train_hist = train_oncl_only(cfg, x_train, constraint_dim=codim)
    learned_codim = int(np.asarray(train_hist.get("learned_codim", np.asarray([codim]))).reshape(-1)[0])
    print(f"[run] training finished: dataset={name}")
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f"{name}_oncl_model.pt")
    torch.save(
        {
            "dataset": str(name),
            "model_state": model.state_dict(),
            "in_dim": int(x_train.shape[1]),
            "constraint_dim": int(learned_codim),
            "hidden": int(cfg.hidden),
            "depth": int(cfg.depth),
            "x_train": x_train.astype(np.float32),
            "cfg": asdict(cfg),
        },
        ckpt_path,
    )
    print(f"saved: {ckpt_path}")

    def _project_last_with_steps(
        m: nn.Module, x0_eval: np.ndarray, eps: float
    ) -> tuple[np.ndarray, np.ndarray]:
        x0_use = x0_eval.astype(np.float32)
        q_end, steps = project_points_with_steps_numpy(
            m,
            x0_use,
            device=str(cfg.device),
            proj_steps=_proj_steps(cfg),
            proj_alpha=_proj_alpha(cfg),
            proj_min_steps=_proj_min_steps(cfg),
            f_abs_stop=float(eps),
        )
        return q_end, steps

    embed_fn = None
    post_fn = None
    if _is_arm_dataset(name):
        use_pybullet_n6 = str(name) == "6d_spatial_arm_up_n6"
        embed_fn = lambda q, _name=name: shared_workspace_embed_for_eval(  # noqa: E731
            _name,
            q,
            ur5_use_pybullet_n6=use_pybullet_n6,
        )
        post_fn = _wrap_np_pi
    elif _is_workspace_pose_dataset(name):
        use_pybullet_n6 = str(name) == "6d_spatial_arm_up_n6"
        embed_fn = lambda q, _name=name: shared_workspace_embed_for_eval(  # noqa: E731
            _name,
            q,
            ur5_use_pybullet_n6=use_pybullet_n6,
        )
        post_fn = _wrap_workspace_pose_rpy_np

    eval_metrics, eval_cfg, eval_artifacts = run_eval_metrics(
        cfg=cfg,
        method_key="oncl",
        dataset_name=name,
        model=model,
        x_train=x_train,
        project_fn=_project_last_with_steps,
        embed_fn=embed_fn,
        postprocess_fn=post_fn,
    )

    vis_vals = asdict(cfg)
    vis_vals.update(vars(eval_cfg))
    vis_cfg = SimpleNamespace(**vis_vals)

    mins, maxs = eval_bounds_from_train(x_train, eval_cfg)
    x0 = _uniform_in_box(cfg.viz_proj_traj_count, mins, maxs)
    with torch.no_grad():
        f_on_t = model(torch.from_numpy(x_train).to(cfg.device))
        if f_on_t.dim() == 1:
            f_on_t = f_on_t.unsqueeze(1)
        f_on_norm = torch.linalg.norm(f_on_t, dim=1).detach().cpu().numpy().reshape(-1)
    eps_stop = float(np.percentile(np.abs(f_on_norm), eval_cfg.zero_eps_quantile))
    traj = project_trajectory_numpy(
        model,
        x0,
        device=str(cfg.device),
        proj_steps=_proj_steps(cfg),
        proj_alpha=_proj_alpha(cfg),
        proj_min_steps=_proj_min_steps(cfg),
        f_abs_stop=eps_stop,
    )

    out_eval = os.path.join(outdir, f"{name}_oncl_eval.json")
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
    print(
        f"[eval] {name} | proj_dist={eval_metrics['proj_manifold_dist']:.6f} "
        f"| pred_recall={eval_metrics['pred_recall']:.6f} "
        f"| pred_FPrate={eval_metrics['pred_FPrate']:.6f} "
        f"| chamfer={eval_metrics['bidirectional_chamfer']:.6f} "
        f"| gt->learned={eval_metrics['gt_to_learned_mean']:.6f} "
        f"| learned->gt={eval_metrics['learned_to_gt_mean']:.6f} "
        f"| space={eval_metrics['dist_space']}"
    )
    print(
        f"[eval] {name} | proj_steps={eval_metrics['proj_steps']:.2f} "
        f"| proj_true_dist={eval_metrics['proj_true_dist']:.6f} "
        f"| proj_v_residual={eval_metrics['proj_v_residual']:.6f} "
        f"| eval_eps={eval_metrics['eval_eps_used']:.6f} "
        f"| pred_precision={eval_metrics['pred_precision']:.6f}"
    )
    print(f"saved: {out_eval}")

    out_diag = os.path.join(outdir, f"{name}_oncl_training_diag.png")
    _plot_training_diagnostics(train_hist, out_diag, title=f"{name}: training diagnostics (on-data)")
    print(f"saved: {out_diag}")

    planned_paths: list[np.ndarray] = []
    planning_enabled = _planner_bool(cfg, "enable", True)

    if data_dim == 2:
        worst_traj, worst_x0 = _worst_case_traj_2d(
            model=model,
            dataset_name=str(name),
            x_train=x_train,
            cfg=cfg,
            eval_cfg=eval_cfg,
            eval_artifacts=eval_artifacts,
        )
        out_path = os.path.join(outdir, f"{name}_oncl_contour_traj.png")
        _plot_constraint_2d(
            model=model,
            x_train=x_train,
            traj=traj,
            out_path=out_path,
            title=f"{name}: ONCL",
            axis_labels=(axis_labels[0], axis_labels[1]),
            cfg=vis_cfg,
            worst_traj=worst_traj,
            worst_x0=worst_x0,
        )
        print(f"saved: {out_path}")
        if planning_enabled and name == "2d_planar_arm_line_n2":
            out_plan = os.path.join(outdir, f"{name}_oncl_planning_demo.png")
            planned_paths = _plot_planar_arm_planning(model, name, x_train, out_plan, cfg, render_pybullet=False)
        elif planning_enabled:
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
                    x_start = dataaug_method.true_projection(x_start[None, :], grid_vis)[0][0]
                    x_goal = dataaug_method.true_projection(x_goal[None, :], grid_vis)[0][0]
                    planned = plan_path(
                        model=model,
                        x_start=x_start,
                        x_goal=x_goal,
                        cfg=cfg,
                        planner_name=planner_name,
                        n_waypoints=n_waypoints,
                        dataset_name=name,
                        periodic_joint=bool(ds.get("periodic_joint", False)),
                        f_abs_stop=eps_used,
                    )
                    if use_linear:
                        plans_proj.append(planned)
                    else:
                        plans_constr.append(planned)
                out_plan = os.path.join(outdir, f"{name}_oncl_planner_paths.png")
                plot_planned_paths(
                    model,
                    x_train,
                    grid_vis,
                    plans_proj,
                    plans_constr,
                    vis_cfg,
                    out_path=out_plan,
                    title=f"{name}: ONCL Planned Paths",
                    zero_level_eps=eps_used,
                )
                print(f"saved: {out_plan}")
        return

    if data_dim == 3:
        out_path = os.path.join(outdir, f"{name}_oncl_zero_surfaces_3d.png")
        _plot_zero_surfaces_3d(
            model=model,
            x_train=x_train,
            traj=traj,
            out_path=out_path,
            title=f"{name}: ONCL",
            axis_labels=(axis_labels[0], axis_labels[1], axis_labels[2]),
            cfg=vis_cfg,
            intersection_points=eval_artifacts.get("proj", None),
        )
        print(f"saved: {out_path}")
        out_paper = os.path.join(outdir, f"{name}_oncl_constraint_surface_paper.png")
        _plot_constraint_surface_paper_3d(
            model=model,
            x_train=x_train,
            out_path=out_paper,
            axis_labels=(axis_labels[0], axis_labels[1], axis_labels[2]),
            cfg=vis_cfg,
            intersection_points=eval_artifacts.get("proj", None),
            dataset_name=str(name),
        )
        print(f"saved: {out_paper}")
    else:
        out_path = os.path.join(outdir, f"{name}_oncl_pca_traj.png")
        _plot_highdim_pca(
            x_train=x_train,
            traj=traj,
            out_path=out_path,
            title=f"{name}: ONCL",
        )
        print(f"saved: {out_path}")
        if name in (
            "6d_workspace_sine_surface_pose",
            "6d_workspace_sine_surface_pose_traj",
            "6d_workspace_sine_surface_pose_traj_turn",
        ):
            out_pose = os.path.join(outdir, f"{name}_oncl_workspace_pose_orientation.png")
            _plot_workspace_pose_orientation_3d(
                x_train=x_train,
                eval_proj=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
                out_path=out_pose,
                title=f"{name}: projected eval poses + orientation z-axis",
            )
            print(f"saved: {out_pose}")
            out_dist = os.path.join(outdir, f"{name}_oncl_workspace_pose_proj_error_distributions.png")
            _plot_workspace_pose_projection_error_distributions(
                x_before=eval_artifacts.get("x_eval", np.zeros((0, 6), dtype=np.float32)),
                x_after=eval_artifacts.get("proj", np.zeros((0, 6), dtype=np.float32)),
                out_path=out_dist,
                title=f"{name}: projection errors before/after",
            )
            print(f"saved: {out_dist}")
    if planning_enabled and name in ("3d_planar_arm_line_n3", "3d_spatial_arm_plane_n3", "3d_spatial_arm_ellip_n3", "3d_spatial_arm_circle_n3", "6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
        out_plan = os.path.join(outdir, f"{name}_oncl_planning_demo.png")
        planned_paths = _plot_planar_arm_planning(model, name, x_train, out_plan, cfg, render_pybullet=False)
    if name in ("6d_spatial_arm_up_n6", "6d_spatial_arm_up_n6_py"):
        out_dist = os.path.join(outdir, f"{name}_oncl_proj_value_distribution.png")
        _plot_projection_value_distribution(
            model,
            x_train,
            vis_cfg,
            out_dist,
            use_pybullet_n6=(name == "6d_spatial_arm_up_n6"),
        )
        print(f"saved: {out_dist}")
        out_ws = os.path.join(outdir, f"{name}_oncl_eval_proj_workspace_orientation.png")
        _plot_ur5_eval_projection_workspace_orientation_3d(
            q_train=x_train,
            q_eval_proj=eval_artifacts.get("proj", np.zeros((0, x_train.shape[1]), dtype=np.float32)),
            out_path=out_ws,
            title=f"{name}: eval projected points in workspace + tool orientation",
            use_pybullet_n6=(name == "6d_spatial_arm_up_n6"),
        )
        print(f"saved: {out_ws}")
    # Keep pybullet render as the very last step after all figures are saved.
    if planning_enabled and name == "6d_spatial_arm_up_n6" and _planner_bool(cfg, "pybullet_render", False) and len(planned_paths) > 0:
        _render_ur5_pybullet_trajectories(planned_paths, cfg)


def main() -> None:
    interactive_checked = False
    interactive_ok = False
    for name in DEFAULT_DATASETS:
        cfg = build_cfg(str(name))
        cfg.device = _choose_device(str(cfg.device))
        if bool(cfg.show_3d_plot):
            if not interactive_checked:
                interactive_ok = _enable_interactive_backend_if_possible()
                interactive_checked = True
            if not interactive_ok:
                print("[warn] interactive matplotlib backend unavailable; disable show_3d_plot")
                cfg.show_3d_plot = False
        if cfg.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        print(f"[cfg] dataset={name}, n_train={cfg.n_train}, epochs={cfg.epochs}")
        run_dataset(str(name), cfg, DEFAULT_OUTDIR)


if __name__ == "__main__":
    from experiments.unified_experiment import run_one as _run_one_unified

    p = argparse.ArgumentParser(description="oncl direct wrapper (unified runner)")
    p.add_argument("--dataset", default=",".join(DEFAULT_DATASETS), help="single or comma-separated datasets")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config-root", default="configs")
    p.add_argument("--override", action="append", default=[], help="dotted key=value override")
    args = p.parse_args()
    ds_list = [d.strip() for d in str(args.dataset).split(",") if d.strip()]
    for ds_name in ds_list:
        print(f"[run] method=oncl dataset={ds_name}")
        result, loaded_paths = _run_one_unified(
            method="oncl",
            dataset=ds_name,
            out_root=str(args.outdir),
            seed_override=args.seed,
            config_root=str(args.config_root),
            cli_overrides=list(args.override),
        )
        m = result["metrics"]
        print(f"[cfg] loaded_layers={loaded_paths if loaded_paths else '[]'}")
        print(
            f"[done] method=oncl dataset={ds_name} "
            f"proj_dist={m.get('proj_manifold_dist', float('nan')):.6f} "
            f"recall={m.get('pred_recall', float('nan')):.6f} "
            f"FPrate={m.get('pred_FPrate', float('nan')):.6f}"
        )
