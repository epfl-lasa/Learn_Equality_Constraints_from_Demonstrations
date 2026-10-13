from __future__ import annotations

import os
import sys
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch import nn

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from methods.models.autoencoder import VariationalAutoEncoder
from methods.models.mlp import MLP, NormalizedMLP
from methods.utils.feature_normalizer import FeatureNormalizedModel, FeatureNormalizer
from methods.vae import VAEProjectorField


def _feature_normalizer_from_checkpoint(ckpt: dict[str, Any], in_dim: int) -> FeatureNormalizer:
    fd = ckpt.get("feature_normalizer", {}) or {}
    return FeatureNormalizer(
        center=np.asarray(fd.get("center", np.zeros((in_dim,), dtype=np.float32)), dtype=np.float32),
        scale=np.asarray(fd.get("scale", np.ones((in_dim,), dtype=np.float32)), dtype=np.float32),
        angle_dims=tuple(int(v) for v in fd.get("angle_dims", [])),
    )


def _load_mlp_field(
    ckpt: dict[str, Any],
    *,
    device: str,
    model_type: str,
    disable_feature_normalizer: bool = False,
) -> nn.Module:
    in_dim = int(ckpt["in_dim"])
    out_dim = int(ckpt["constraint_dim"])
    hidden = int(ckpt.get("hidden", 128))
    depth = int(ckpt.get("depth", 3))
    state = ckpt["model_state"]

    if model_type == "normalized_mlp":
        model = NormalizedMLP(
            in_dim=in_dim,
            hidden=hidden,
            depth=depth,
            out_dim=out_dim,
            center=state.get("input_center", torch.zeros((in_dim,), dtype=torch.float32)),
            scale=state.get("input_scale", torch.ones((in_dim,), dtype=torch.float32)),
            angle_dims=tuple(int(v) for v in ckpt.get("input_angle_dims", [])),
        ).to(device)
        model.load_state_dict(state)
        return model

    base = MLP(in_dim=in_dim, hidden=hidden, depth=depth, out_dim=out_dim).to(device)
    base.load_state_dict(state)
    if model_type == "feature_normalized_mlp" and not bool(disable_feature_normalizer):
        return FeatureNormalizedModel(base, _feature_normalizer_from_checkpoint(ckpt, in_dim)).to(device)
    return base


def _load_ecomann_field(
    ckpt: dict[str, Any],
    *,
    device: str,
    model_type: str,
    disable_feature_normalizer: bool = False,
) -> nn.Module:
    from methods.ecomann import _add_smp_repo_to_path

    _add_smp_repo_to_path()
    from smp_manifold_learning.differentiable_models.ecmnn import EqualityConstraintManifoldNeuralNetwork

    cfg = ckpt.get("cfg", {}) or {}
    in_dim = int(ckpt["in_dim"])
    out_dim = int(ckpt["constraint_dim"])
    model = EqualityConstraintManifoldNeuralNetwork(
        input_dim=in_dim,
        hidden_sizes=[int(v) for v in ckpt.get("hidden_sizes", (36, 24, 18, 10))],
        output_dim=out_dim,
        use_batch_norm=bool(cfg.get("use_batch_norm", True)),
        drop_p=0.0,
        is_training=False,
        device=device,
    )

    def _train_compat(self: nn.Module, mode: bool = True) -> nn.Module:
        self.nn_model.train(mode)
        self.is_training = bool(mode)
        return self

    model.train = MethodType(_train_compat, model)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if model_type == "feature_normalized_ecomann" and not bool(disable_feature_normalizer):
        return FeatureNormalizedModel(model, _feature_normalizer_from_checkpoint(ckpt, in_dim)).to(device)
    return model.to(device)


def _load_vae_residual_field(ckpt: dict[str, Any], *, device: str) -> nn.Module:
    in_dim = int(ckpt["in_dim"])
    latent_dim = int(ckpt["latent_dim"])
    hidden = tuple(int(v) for v in ckpt.get("hidden_dims", (64, 32)))
    vae = VariationalAutoEncoder(in_dim=in_dim, latent_dim=latent_dim, hidden=hidden).to(device)
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()
    return VAEProjectorField(vae).to(device)


def load_constraint_model_from_checkpoint(
    ckpt_path: str,
    *,
    device: str,
    disable_feature_normalizer: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a trained method checkpoint as a residual field usable by traj-opt.

    ONCL/DataAug/IGR expose their learned MLP constraint fields directly.
    EcoMaNN exposes its equality-constraint network. VAE is adapted to the
    residual field f(x)=x-D(E_mu(x)), so it can share the same planner timing
    path even though its output dimension is the ambient dimension.
    """

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    method = str(ckpt.get("method", "") or "").lower()
    model_type = str(ckpt.get("model_type") or "mlp").lower()

    if method == "vae" or "vae" in model_type:
        model = _load_vae_residual_field(ckpt, device=device)
    elif method == "ecomann" or "ecomann" in model_type:
        model = _load_ecomann_field(
            ckpt,
            device=device,
            model_type=model_type,
            disable_feature_normalizer=bool(disable_feature_normalizer),
        )
    else:
        model = _load_mlp_field(
            ckpt,
            device=device,
            model_type=model_type,
            disable_feature_normalizer=bool(disable_feature_normalizer),
        )

    model.eval()
    return model, ckpt
