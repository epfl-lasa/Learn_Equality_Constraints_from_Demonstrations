from __future__ import annotations

from typing import Any, Callable

import numpy as np
from torch import nn

from evaluation.evaluator import (
    evaluate_bidirectional_chamfer,
    evaluate_projection_metrics,
    resolve_eval_cfg,
)


ProjectFn = Callable[[nn.Module, np.ndarray, float], tuple[np.ndarray, np.ndarray]]


def run_eval_metrics(
    *,
    cfg: Any,
    method_key: str,
    dataset_name: str,
    model: nn.Module,
    x_train: np.ndarray,
    project_fn: ProjectFn,
    embed_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    postprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[dict[str, float], Any, dict[str, np.ndarray]]:
    eval_cfg = resolve_eval_cfg(
        cfg,
        method_key=method_key,
        dataset_name=dataset_name,
    )
    metrics_proj, eval_artifacts = evaluate_projection_metrics(
        model=model,
        x_train=x_train,
        cfg=eval_cfg,
        project_fn=project_fn,
        dataset_name=dataset_name,
        embed_fn=embed_fn,
        postprocess_fn=postprocess_fn,
        return_artifacts=True,
    )
    metrics_chamfer = evaluate_bidirectional_chamfer(
        model=model,
        x_train=x_train,
        cfg=eval_cfg,
        project_fn=project_fn,
        dataset_name=dataset_name,
        embed_fn=embed_fn,
        postprocess_fn=postprocess_fn,
        eps_stop_override=float(eval_artifacts["eval_eps_used"]),
        x0_override=eval_artifacts["x_eval"],
        learned_samples_override=eval_artifacts["proj"],
    )
    out = {**metrics_proj, **metrics_chamfer}
    out["dist_space"] = metrics_proj.get("dist_space", out.get("dist_space", ""))
    return out, eval_cfg, eval_artifacts
