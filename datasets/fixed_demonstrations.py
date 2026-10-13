from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEMONSTRATIONS_ROOT = Path(__file__).resolve().parent / "demonstrations"
FIXED_DEMONSTRATION_DIMS = {
    "6d_spatial_arm_up_n6_py_traj": 6,
    "6d_surf_normal_traj": 6,
    "12d_dual_arm_traj": 12,
}


def load_fixed_demonstrations(dataset_name: str) -> dict[str, Any]:
    """Load the packaged training trajectories and reference samples."""
    if dataset_name not in FIXED_DEMONSTRATION_DIMS:
        raise KeyError(f"unknown fixed demonstration dataset {dataset_name!r}")
    path = DEMONSTRATIONS_ROOT / f"{dataset_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Required demonstration file is missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in ("x_train", "grid", "eval_grid")
        }
        lengths = np.asarray(data["trajectory_lengths"], dtype=np.int64)
        generation_seed = int(data["generation_seed"].item())
        metadata = json.loads(str(data["metadata"].item()))
    dim = FIXED_DEMONSTRATION_DIMS[dataset_name]
    for key, points in arrays.items():
        if points.ndim != 2 or points.shape[1] != dim or len(points) == 0:
            raise ValueError(f"Invalid {key} shape in {path}: {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite {key} samples in {path}")
    if lengths.ndim != 1 or np.any(lengths <= 0) or int(lengths.sum()) != len(arrays["x_train"]):
        raise ValueError(f"Invalid trajectory lengths in {path}")
    return {
        **arrays,
        "trajectory_lengths": lengths,
        "generation_seed": generation_seed,
        "metadata": metadata,
    }
