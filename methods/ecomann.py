from __future__ import annotations

import os
import sys
import time
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def _add_smp_repo_to_path() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    smp_root = os.path.join(repo_root, "smp_manifold_learning-master")
    if not os.path.isdir(smp_root):
        raise FileNotFoundError(
            f"Bundled ECoMaNN source not found: {smp_root}"
        )
    if smp_root not in sys.path:
        sys.path.insert(0, smp_root)
    # Provide the timer API used by the ECoMaNN data loader on Python 3.8+.
    if not hasattr(time, "clock"):
        time.clock = time.perf_counter  # type: ignore[attr-defined]
    return smp_root


@dataclass
class Config:
    seed: int = 2116
    device: str = "auto"
    n_train: int = 512
    traj_gene_n_grid: int = 4096

    hidden_sizes: tuple[int, ...] = (36, 24, 18, 10)
    epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    use_batch_norm: bool = True
    norm_level_clip_max: float = 100.0

    is_performing_data_augmentation: bool = True
    is_optimizing_local_tangent_space_alignment_loss: bool = True
    is_optimizing_signed_siamese_pairs: bool = True
    is_aligning_lpca_normal_space_eigvecs: bool = True
    is_augmenting_w_rand_comb_of_normaleigvecs: bool = True
    max_rand_comb_normal_dim: int = 3
    aug_epsilon_mult_override: float | None = None
    clean_aug_data: bool = True
    siam_mode: str = "all"
    n_normal_space_traversal: int = 9
    n_local_neighborhood_mult: float = 1.0

    train_log_every: int = 1
    projector: dict[str, Any] = field(
        default_factory=lambda: {"alpha": 0.3, "steps": 80, "min_steps": 20}
    )
    planner: dict[str, Any] = field(
        default_factory=lambda: {
            "enable": False,
            "method": "traj_opt",
            "steps": 64,
            "opt_steps": 1240,
            "opt_lr": 0.01,
            "opt_lam_smooth": 0.2,
            "lam_manifold": 1.0,
            "lam_len_joint": 0.4,
            "trust_scale": 0.8,
            "pair_min_ratio": 0.15,
            "pair_max_ratio": 0.35,
            "pair_tries": 1200,
            "init_mode": "joint_spline",
            "joint_mid_noise": 0.0,
            "anim_fps": 6,
            "anim_stride": 1,
            "save_gif": False,
            "pybullet_render": False,
            "pybullet_real_time_dt": 0.06,
        }
    )


def _choose_device(device: str) -> str:
    if device != "auto":
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_bool(x: Any) -> bool:
    return bool(x)


def train_ecomann(
    cfg: Config,
    x_train: np.ndarray,
    *,
    force_codim: int | None = None,
    return_loader_data: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any], int] | tuple[torch.nn.Module, dict[str, Any], int, dict[str, np.ndarray]]:
    smp_root = _add_smp_repo_to_path()
    from smp_manifold_learning.dataset_loader.ecmnn_dataset_loader import ECMNNDatasetLoader
    from smp_manifold_learning.differentiable_models.ecmnn import EqualityConstraintManifoldNeuralNetwork

    if str(cfg.siam_mode) not in {"all", "no_siam_reflection", "no_siam_same_levelvec", "no_siam_frac_aug"}:
        raise ValueError(f"unsupported siam_mode: {cfg.siam_mode}")

    np.random.seed(int(cfg.seed))
    torch.random.manual_seed(int(cfg.seed))

    device = _choose_device(str(cfg.device))
    x_train = np.asarray(x_train, dtype=np.float32)
    if x_train.ndim != 2:
        raise ValueError(f"x_train must be rank-2, got shape={x_train.shape}")

    old_cwd = os.getcwd()
    loader_data: dict[str, np.ndarray] | None = None
    try:
        os.chdir(smp_root)
        with tempfile.TemporaryDirectory(prefix="ecomann_train_") as td:
            ds_base = os.path.join(td, "train_data")
            np.save(ds_base + ".npy", x_train)

            loader = ECMNNDatasetLoader(
                ds_base,
                is_performing_data_augmentation=_as_bool(cfg.is_performing_data_augmentation),
                N_normal_space_traversal=int(cfg.n_normal_space_traversal),
                is_optimizing_signed_siamese_pairs=_as_bool(cfg.is_optimizing_signed_siamese_pairs),
                clean_aug_data=_as_bool(cfg.clean_aug_data),
                is_aligning_lpca_normal_space_eigvecs=_as_bool(cfg.is_aligning_lpca_normal_space_eigvecs),
                is_augmenting_w_rand_comb_of_normaleigvecs=_as_bool(cfg.is_augmenting_w_rand_comb_of_normaleigvecs),
                max_rand_comb_normal_dim=int(cfg.max_rand_comb_normal_dim),
                aug_epsilon_mult_override=(
                    None if getattr(cfg, "aug_epsilon_mult_override", None) is None
                    else float(cfg.aug_epsilon_mult_override)
                ),
                rand_seed=int(cfg.seed),
                N_local_neighborhood_mult=float(cfg.n_local_neighborhood_mult),
                dim_normal_space_override=(None if force_codim is None else int(force_codim)),
            )
            # Use all available (augmented) data for training.
            batch_train_loader = DataLoader(
                loader.dataset,
                batch_size=int(cfg.batch_size),
                shuffle=True,
                num_workers=0,
                # BatchNorm in training mode requires batch size > 1.
                # Avoid rare tail-batch size=1 crash.
                drop_last=bool(cfg.use_batch_norm),
            )
            if return_loader_data:
                loader_data = {k: np.asarray(v) for k, v in loader.dataset.data.items()}
    finally:
        os.chdir(old_cwd)

    model = EqualityConstraintManifoldNeuralNetwork(
        input_dim=int(loader.dim_ambient),
        hidden_sizes=[int(v) for v in cfg.hidden_sizes],
        output_dim=int(loader.dim_normal_space),
        use_batch_norm=_as_bool(cfg.use_batch_norm),
        drop_p=0.0,
        is_training=True,
        device=device,
    )
    opt = torch.optim.RMSprop(model.nn_model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    hist: dict[str, list[float]] = {
        "loss": [],
        "norm_level": [],
        "J_nspace": [],
        "cov_nspace": [],
        "J_rspace": [],
        "cov_rspace": [],
    }
    if _as_bool(cfg.is_optimizing_signed_siamese_pairs):
        hist["siam_reflection"] = []
        hist["siam_same_levelvec"] = []
        hist["siam_frac_aug"] = []

    for epoch in range(int(cfg.epochs)):
        epoch_t0 = time.perf_counter()
        model.train()
        loss_vals: list[float] = []
        nlevel_vals: list[float] = []
        jns_vals: list[float] = []
        cns_vals: list[float] = []
        jrs_vals: list[float] = []
        crs_vals: list[float] = []
        sref_vals: list[float] = []
        ssame_vals: list[float] = []
        sfrac_vals: list[float] = []
        # Timing breakdown (seconds), averaged per batch at logging time.
        t_batch_total = 0.0
        t_get_components = 0.0
        t_loss_assemble = 0.0
        t_backward_step = 0.0
        n_batches = 0
        n_samples = 0

        for batch_data in batch_train_loader:
            tb0 = time.perf_counter()
            bsz = int(batch_data["data"].shape[0]) if isinstance(batch_data, dict) and "data" in batch_data else 0
            opt.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            comps = model.get_loss_components(batch_data)
            t_get_components += float(time.perf_counter() - t0)

            t1 = time.perf_counter()
            norm_level_loss_raw = comps["norm_level_wnmse_per_dim"].mean()
            clip_max = float(getattr(cfg, "norm_level_clip_max", 0.0))
            if np.isfinite(clip_max) and clip_max > 0.0:
                norm_level_loss = torch.clamp(norm_level_loss_raw, max=clip_max)
            else:
                norm_level_loss = norm_level_loss_raw
            J_nspace_proj_loss = comps["J_nspace_proj_loss_per_dim"].mean()
            cov_nspace_proj_loss = comps["cov_nspace_proj_loss_per_dim"].mean()
            J_rspace_proj_loss = comps["J_rspace_proj_loss_per_dim"].mean()
            cov_rspace_proj_loss = comps["cov_rspace_proj_loss_per_dim"].mean()

            loss = norm_level_loss
            if _as_bool(cfg.is_optimizing_local_tangent_space_alignment_loss):
                loss = loss + J_nspace_proj_loss + cov_nspace_proj_loss + J_rspace_proj_loss + cov_rspace_proj_loss

            if _as_bool(cfg.is_optimizing_signed_siamese_pairs):
                if str(cfg.siam_mode) != "no_siam_reflection":
                    loss = loss + comps["siam_reflection_wnmse_torch"].mean()
                if str(cfg.siam_mode) != "no_siam_frac_aug":
                    loss = loss + comps["siam_frac_aug_wnmse_torch"].mean()
                if str(cfg.siam_mode) != "no_siam_same_levelvec":
                    loss = loss + comps["siam_same_levelvec_wnmse_torch"].mean()
            t_loss_assemble += float(time.perf_counter() - t1)

            loss_np = float(loss.detach().cpu().item())
            if not np.isfinite(loss_np):
                raise RuntimeError(f"EcoMaNN loss became non-finite at epoch={epoch + 1}")
            t2 = time.perf_counter()
            loss.backward()
            opt.step()
            t_backward_step += float(time.perf_counter() - t2)

            loss_vals.append(loss_np)
            nlevel_vals.append(float(norm_level_loss.detach().cpu().item()))
            jns_vals.append(float(J_nspace_proj_loss.detach().cpu().item()))
            cns_vals.append(float(cov_nspace_proj_loss.detach().cpu().item()))
            jrs_vals.append(float(J_rspace_proj_loss.detach().cpu().item()))
            crs_vals.append(float(cov_rspace_proj_loss.detach().cpu().item()))
            if _as_bool(cfg.is_optimizing_signed_siamese_pairs):
                sref_vals.append(float(comps["siam_reflection_wnmse_torch"].mean().detach().cpu().item()))
                ssame_vals.append(float(comps["siam_same_levelvec_wnmse_torch"].mean().detach().cpu().item()))
                sfrac_vals.append(float(comps["siam_frac_aug_wnmse_torch"].mean().detach().cpu().item()))
            t_batch_total += float(time.perf_counter() - tb0)
            n_batches += 1
            n_samples += max(0, bsz)

        model.eval()
        if ((epoch + 1) % max(1, int(cfg.train_log_every))) == 0:
            # Use epoch mini-batch aggregate directly; avoid extra full-dataset recomputation.
            all_n = float(np.mean(nlevel_vals)) if nlevel_vals else float("nan")
            loss_epoch = float(np.mean(loss_vals)) if loss_vals else float("nan")
            epoch_s = float(time.perf_counter() - epoch_t0)
            nb = max(1, int(n_batches))
            bps = (float(n_samples) / max(epoch_s, 1e-9)) if n_samples > 0 else float("nan")
            print(
                f"[EcoMaNN] epoch {epoch + 1}/{int(cfg.epochs)} "
                f"loss={loss_epoch:.6f} all_norm={all_n:.6f} "
                f"| t_epoch={epoch_s:.3f}s t_batch={t_batch_total/nb:.4f}s "
                f"t_comp={t_get_components/nb:.4f}s t_loss={t_loss_assemble/nb:.4f}s "
                f"t_bw={t_backward_step/nb:.4f}s "
                f"{('samples_per_sec=' + format(bps, '.1f')) if np.isfinite(bps) else ''}"
            )

        hist["loss"].append(float(np.mean(loss_vals)) if loss_vals else float("nan"))
        hist["norm_level"].append(float(np.mean(nlevel_vals)) if nlevel_vals else float("nan"))
        hist["J_nspace"].append(float(np.mean(jns_vals)) if jns_vals else float("nan"))
        hist["cov_nspace"].append(float(np.mean(cns_vals)) if cns_vals else float("nan"))
        hist["J_rspace"].append(float(np.mean(jrs_vals)) if jrs_vals else float("nan"))
        hist["cov_rspace"].append(float(np.mean(crs_vals)) if crs_vals else float("nan"))
        if _as_bool(cfg.is_optimizing_signed_siamese_pairs):
            hist["siam_reflection"].append(float(np.mean(sref_vals)) if sref_vals else float("nan"))
            hist["siam_same_levelvec"].append(float(np.mean(ssame_vals)) if ssame_vals else float("nan"))
            hist["siam_frac_aug"].append(float(np.mean(sfrac_vals)) if sfrac_vals else float("nan"))

    model.eval()
    model.is_training = False
    if return_loader_data:
        if loader_data is None:
            raise RuntimeError("EcoMaNN loader data was requested but not available.")
        return model, hist, int(loader.dim_normal_space), loader_data
    return model, hist, int(loader.dim_normal_space)
