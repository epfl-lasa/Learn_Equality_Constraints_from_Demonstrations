from __future__ import annotations

from typing import Any, Callable
import os
import glob
import math

import numpy as np
from datasets.fixed_demonstrations import FIXED_DEMONSTRATION_DIMS, load_fixed_demonstrations

from datasets.constraint_datasets import (
    generate_dataset,
    lift_xy_to_3d_var,
    lift_xy_to_3d_zero,
)
from datasets.ur5_n6_dataset import (
    sample_ur5_upward_dataset,
    sample_ur5_upward_dataset_analytic,
)
from datasets.real_iiwa.registry import (
    canonical_real_iiwa_dataset_name,
    is_real_iiwa_dataset,
    load_real_iiwa_train_set,
    real_iiwa_orientation_representation,
    real_iiwa_surface_grid,
)

BASE_2D_DATASETS = [
    "2d_square",
    "2d_figure_eight",
    "2d_intersecting_loops",
    "2d_ellipse",
    "2d_noisy_sine",
    "2d_sine",
    "2d_sparse_sine",
    "2d_discontinuous",
    "2d_two_horizontal_lines",
    "2d_looped_spiro",
    "2d_sharp_star",
    "2d_hetero_noise",
    "2d_planar_arm_line_n2",
]

BASE_3D_DATASETS = [
    "3d_spiral",
    "linked_curves_3d",
    "3d_knot_spline_union",
    "3d_knot_spline_union_traj",
    "3d_knot_spline_triplet",
    "3d_knot_spline_triplet_intersect",
    "3d_knot_spline_triplet_near_intersect",
    "3d_knot_spline_triplet_simple",
    "3d_nested_ellipses_union",
    "3d_linked_ellipses_union",
    "3d_paraboloid",
    "3d_paraboloid_traj",
    "3d_twosphere",
    "3d_twosphere_traj",
    "3d_saddle_surface",
    "3d_saddle_surface_traj",
    "3d_sphere_surface",
    "3d_sphere_surface_traj",
    "3d_torus_surface",
    "3d_torus_surface_traj",
    "3d_double_torus_surface",
    "3d_double_torus_surface_traj",
    "3d_double_torus_extra_handle_surface",
    "3d_double_torus_extra_handle_surface_traj",
    "3d_genus3_tube_surface",
    "3d_genus3_tube_surface_traj",
    "3d_faceted_sphere_surface",
    "3d_faceted_sphere_surface_traj",
    "3d_spiky_sphere_surface",
    "3d_spiky_sphere_surface_traj",
    "3d_planar_arm_line_n3",
    "3d_planar_arm_line_n3_traj",
    "3d_spatial_arm_plane_n3",
    "3d_spatial_arm_plane_n3_traj",
    "3d_spatial_arm_ellip_n3",
    "3d_spatial_arm_ellip_n3_traj",
    "3d_spatial_arm_circle_n3",
]

BASE_4D_DATASETS = [
    "multi_handle_4d",
    "4d_warped_double_torus_surface",
    "4d_warped_double_torus_extra_handle_surface",
    "4d_warped_double_torus_extra_handle_surface_traj",
]

BASE_6D_DATASETS = [
    "6d_spatial_arm_up_n6",
    "6d_spatial_arm_up_n6_py",
    "6d_spatial_arm_up_n6_py_traj",
    "6d_surf_normal",
    "6d_surf_normal_traj",
]

BASE_12D_DATASETS = [
    "12d_dual_arm",
    "12d_dual_arm_traj",
]


def _as_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32)


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _nearest_neighbor_l2_distances(x: np.ndarray) -> np.ndarray:
    clean = np.asarray(x, dtype=np.float64)
    n = int(clean.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(clean)
        dists, _ = tree.query(clean, k=2)
        return np.asarray(dists[:, 1], dtype=np.float64)
    except Exception:
        out = np.empty((n,), dtype=np.float64)
        chunk = 512
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            d2 = np.sum((clean[start:end, None, :] - clean[None, :, :]) ** 2, axis=2)
            rows = np.arange(end - start)
            d2[rows, start + rows] = np.inf
            out[start:end] = np.sqrt(np.maximum(np.min(d2, axis=1), 0.0))
        return out


def _apply_training_noise(x: np.ndarray, cfg: Any, *, angle_columns: tuple[int, ...] = ()) -> tuple[np.ndarray, dict[str, Any] | None]:
    if not bool(getattr(cfg, "train_noise_enable", False)):
        return _as_float32(x), None

    mode = str(getattr(cfg, "train_noise_mode", "raw_span_gaussian")).strip()
    clean = np.asarray(x, dtype=np.float32)
    if mode == "raw_span_gaussian":
        ratio = float(getattr(cfg, "train_noise_raw_span_ratio", 0.0))
        raw_span = np.max(clean, axis=0) - np.min(clean, axis=0)
        std = (max(0.0, ratio) * raw_span).astype(np.float32)
        metadata = {
            "enabled": True,
            "mode": mode,
            "raw_span_ratio": ratio,
            "raw_span": raw_span.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "angle_columns": [int(c) for c in angle_columns],
        }
    elif mode == "median_nn_l2_gaussian":
        alpha = max(0.0, float(getattr(cfg, "train_noise_median_nn_alpha", 0.0)))
        nn_dists = _nearest_neighbor_l2_distances(clean)
        median_nn_l2 = float(np.median(nn_dists)) if len(nn_dists) else 0.0
        target_rms_l2 = float(alpha * median_nn_l2)
        dim = max(1, int(clean.shape[1]))
        per_dim_sigma = float(target_rms_l2 / math.sqrt(float(dim)))
        std = np.full((dim,), per_dim_sigma, dtype=np.float32)
        metadata = {
            "enabled": True,
            "mode": mode,
            "median_nn_alpha": alpha,
            "median_nn_l2": median_nn_l2,
            "target_rms_l2": target_rms_l2,
            "per_dim_sigma": per_dim_sigma,
            "std": std.astype(float).tolist(),
            "angle_columns": [int(c) for c in angle_columns],
        }
    else:
        raise ValueError(f"unsupported train_noise_mode={mode!r}")

    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 104729)
    noisy = clean + rng.normal(0.0, std.reshape(1, -1), size=clean.shape).astype(np.float32)
    for col in angle_columns:
        if 0 <= int(col) < noisy.shape[1]:
            noisy[:, int(col)] = _wrap_to_pi(noisy[:, int(col)])
    return noisy.astype(np.float32), metadata


def _split_positive_count(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    n = max(1, min(int(count), int(total)))
    base = int(total) // n
    rem = int(total) % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _apply_training_missing_segments(x: np.ndarray, cfg: Any) -> tuple[np.ndarray, dict[str, Any] | None]:
    if not bool(getattr(cfg, "train_missing_segment_enable", False)):
        return _as_float32(x), None

    mode = str(getattr(cfg, "train_missing_segment_mode", "trajectory_random_segments")).strip()
    if mode != "trajectory_random_segments":
        raise ValueError(f"unsupported train_missing_segment_mode={mode!r}")

    clean = np.asarray(x, dtype=np.float32)
    n = int(clean.shape[0])
    if n <= 1:
        return clean.astype(np.float32), {
            "enabled": True,
            "mode": mode,
            "total_ratio": float(getattr(cfg, "train_missing_total_ratio", 0.0)),
            "removed_points": 0,
            "kept_points_before_refill": n,
        }

    ratio = min(0.999, max(0.0, float(getattr(cfg, "train_missing_total_ratio", 0.0))))
    count = max(1, int(getattr(cfg, "train_missing_segment_count", 1)))
    keep_n_train = bool(getattr(cfg, "train_missing_keep_n_train", False))
    traj_len = max(1, int(getattr(cfg, "traj_len", n)))
    traj_count = max(1, int(getattr(cfg, "traj_count", int(math.ceil(float(n) / float(traj_len))))))
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 209759)

    pieces: list[np.ndarray] = []
    removed_ranges: list[dict[str, int]] = []
    removed_points = 0
    for traj_i in range(traj_count):
        start = traj_i * traj_len
        if start >= n:
            break
        end = min(n, start + traj_len)
        seg = clean[start:end]
        t = int(seg.shape[0])
        if t <= 1 or ratio <= 0.0:
            pieces.append(seg)
            continue

        target_remove = min(t - 1, int(round(ratio * t)))
        if target_remove <= 0:
            pieces.append(seg)
            continue

        keep = np.ones((t,), dtype=bool)
        for length in _split_positive_count(target_remove, count):
            valid_starts = [
                s for s in range(0, t - int(length) + 1)
                if bool(np.all(keep[s : s + int(length)]))
            ]
            if valid_starts:
                local_start = int(rng.choice(np.asarray(valid_starts, dtype=np.int64)))
                local_end = local_start + int(length)
                keep[local_start:local_end] = False
                removed_ranges.append({
                    "traj": int(traj_i),
                    "start": int(local_start),
                    "end": int(local_end),
                })
                removed_points += int(local_end - local_start)
                continue

            remaining = np.flatnonzero(keep)
            if len(remaining) == 0:
                break
            take = min(int(length), int(len(remaining)))
            idx = rng.choice(remaining, size=take, replace=False)
            keep[idx] = False
            removed_points += int(take)

        pieces.append(seg[keep])

    filtered = np.concatenate(pieces, axis=0).astype(np.float32) if pieces else clean[:0].astype(np.float32)
    kept_before_refill = int(filtered.shape[0])
    if keep_n_train and kept_before_refill > 0 and kept_before_refill < n:
        rep = int(math.ceil(float(n) / float(kept_before_refill)))
        filtered = np.tile(filtered, (rep, 1))[:n].astype(np.float32)

    return filtered.astype(np.float32), {
        "enabled": True,
        "mode": mode,
        "total_ratio": ratio,
        "segment_count": count,
        "keep_n_train": keep_n_train,
        "traj_len": traj_len,
        "traj_count": traj_count,
        "removed_points": int(removed_points),
        "kept_points_before_refill": kept_before_refill,
        "final_points": int(filtered.shape[0]),
        "removed_ranges_preview": removed_ranges[:32],
    }


def _apply_training_subsample(x: np.ndarray, cfg: Any) -> tuple[np.ndarray, dict[str, Any] | None]:
    if not bool(getattr(cfg, "train_subsample_enable", False)):
        return _as_float32(x), None

    mode = str(getattr(cfg, "train_subsample_mode", "trajectory_even")).strip()
    if mode != "trajectory_even":
        raise ValueError(f"unsupported train_subsample_mode={mode!r}")

    clean = np.asarray(x, dtype=np.float32)
    n = int(clean.shape[0])
    if n <= 1:
        return clean.astype(np.float32), {
            "enabled": True,
            "mode": mode,
            "ratio": float(getattr(cfg, "train_subsample_ratio", 1.0)),
            "final_points": n,
        }

    ratio = min(1.0, max(0.0, float(getattr(cfg, "train_subsample_ratio", 1.0))))
    keep_n_train = bool(getattr(cfg, "train_subsample_keep_n_train", False))
    traj_len = max(1, int(getattr(cfg, "traj_len", n)))
    traj_count = max(1, int(getattr(cfg, "traj_count", int(math.ceil(float(n) / float(traj_len))))))

    pieces: list[np.ndarray] = []
    per_traj_keep: list[int] = []
    index_preview: list[dict[str, Any]] = []
    for traj_i in range(traj_count):
        start = traj_i * traj_len
        if start >= n:
            break
        end = min(n, start + traj_len)
        seg = clean[start:end]
        t = int(seg.shape[0])
        keep_count = max(1, min(t, int(round(ratio * t))))
        if keep_count >= t:
            idx = np.arange(t, dtype=np.int64)
        else:
            idx = np.rint(np.linspace(0, t - 1, keep_count)).astype(np.int64)
            idx = np.unique(idx)
            if int(idx.shape[0]) < keep_count:
                missing = keep_count - int(idx.shape[0])
                candidates = np.setdiff1d(np.arange(t, dtype=np.int64), idx, assume_unique=True)
                idx = np.sort(np.concatenate([idx, candidates[:missing]]))
        pieces.append(seg[idx])
        per_traj_keep.append(int(idx.shape[0]))
        if len(index_preview) < 8:
            index_preview.append({
                "traj": int(traj_i),
                "kept": int(idx.shape[0]),
                "indices": idx[:32].astype(int).tolist(),
            })

    filtered = np.concatenate(pieces, axis=0).astype(np.float32) if pieces else clean[:0].astype(np.float32)
    kept_before_refill = int(filtered.shape[0])
    if keep_n_train and kept_before_refill > 0 and kept_before_refill < n:
        rep = int(math.ceil(float(n) / float(kept_before_refill)))
        filtered = np.tile(filtered, (rep, 1))[:n].astype(np.float32)

    return filtered.astype(np.float32), {
        "enabled": True,
        "mode": mode,
        "ratio": ratio,
        "keep_n_train": keep_n_train,
        "traj_len": traj_len,
        "traj_count": traj_count,
        "original_points": n,
        "kept_points_before_refill": kept_before_refill,
        "final_points": int(filtered.shape[0]),
        "points_per_traj_after_subsample": per_traj_keep,
        "subsample_indices_preview": index_preview,
    }


def _apply_training_data_transforms(
    x: np.ndarray,
    cfg: Any,
    *,
    angle_columns: tuple[int, ...] = (),
) -> tuple[np.ndarray, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    if bool(getattr(cfg, "train_subsample_enable", False)) and bool(getattr(cfg, "train_missing_segment_enable", False)):
        raise ValueError("train_subsample_enable and train_missing_segment_enable are mutually exclusive")
    x_base, train_subsample = _apply_training_subsample(x, cfg)
    x_missing, train_missing_segments = _apply_training_missing_segments(x_base, cfg)
    x_train, train_noise = _apply_training_noise(
        x_missing,
        cfg,
        angle_columns=angle_columns,
    )
    return x_train, train_noise, train_missing_segments, train_subsample


_DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
_UR5_CACHE_DIR = os.path.join(_DATASETS_DIR, "cache", "ur5_upward")


def _print_red(msg: str) -> None:
    print(f"\033[31m{msg}\033[0m")


def _cfg_n_grid(cfg: Any, default: int = 4096) -> int:
    if not hasattr(cfg, "traj_gene_n_grid"):
        raise ValueError("missing required config key: traj_gene_n_grid")
    return int(getattr(cfg, "traj_gene_n_grid"))


def _cfg_for_generation(cfg: Any) -> Any:
    from types import SimpleNamespace

    if not hasattr(cfg, "traj_gene_n_grid"):
        raise ValueError("missing required config key: traj_gene_n_grid")
    m = dict(vars(cfg))
    m["n_grid"] = int(getattr(cfg, "traj_gene_n_grid"))
    return SimpleNamespace(**m)


def _ur5_cache_path(
    *,
    dataset_name: str,
    backend: str,
    seed: int,
    n_train: int,
    n_grid: int,
) -> str:
    os.makedirs(_UR5_CACHE_DIR, exist_ok=True)
    safe_name = str(dataset_name).replace("/", "_")
    safe_backend = str(backend).replace("/", "_")
    fname = (
        f"{safe_name}__backend-{safe_backend}"
        f"__seed-{int(seed)}__ntrain-{int(n_train)}__ngrid-{int(n_grid)}.npz"
    )
    return os.path.join(_UR5_CACHE_DIR, fname)


def _try_load_ur5_cache(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            x = np.asarray(data["x_train"], dtype=np.float32)
            grid = np.asarray(data["grid"], dtype=np.float32)
        _print_red(f"[data][ur5][cache] load: {path}")
        return x, grid
    except Exception:
        return None


def _try_load_ur5_cache_relaxed_n_grid(
    *,
    dataset_name: str,
    backend: str,
    seed: int,
    n_train: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    safe_name = str(dataset_name).replace("/", "_")
    safe_backend = str(backend).replace("/", "_")
    pat = (
        f"{safe_name}__backend-{safe_backend}"
        f"__seed-{int(seed)}__ntrain-{int(n_train)}__ngrid-*.npz"
    )
    cand = glob.glob(os.path.join(_UR5_CACHE_DIR, pat))
    if not cand:
        return None
    # Prefer most recently updated cache file.
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return _try_load_ur5_cache(cand[0])


def _save_ur5_cache(path: str, x: np.ndarray, grid: np.ndarray) -> None:
    np.savez_compressed(
        path,
        x_train=np.asarray(x, dtype=np.float32),
        grid=np.asarray(grid, dtype=np.float32),
    )


def _knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    n = int(pts.shape[0])
    kk = int(max(2, min(k, n)))
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(pts.astype(np.float64))
        _, idx = tree.query(pts.astype(np.float64), k=kk)
        if idx.ndim == 1:
            idx = idx[:, None]
        return idx.astype(np.int64)
    except Exception:
        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
        idx = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        return idx.astype(np.int64)


def _traj_points_from_grid_embedded(
    grid: np.ndarray,
    *,
    embed: np.ndarray,
    n_train: int,
    seed: int,
    traj_count: int,
    traj_len: int,
    traj_knn: int,
    diverse_starts: bool = False,
    debug_enabled: bool = False,
    debug_name: str = "traj",
    debug_outdir: str = "outputs/analysis/traj_debug",
    workspace_mapper: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float32)
    e = np.asarray(embed, dtype=np.float32)
    if len(g) == 0:
        return np.zeros((max(1, int(n_train)), g.shape[1] if g.ndim == 2 else 6), dtype=np.float32)

    rng = np.random.default_rng(int(seed) + 1971)
    n = max(1, int(n_train))
    n_traj = max(1, int(traj_count))
    t_len = max(2, int(traj_len))
    knn = _knn_indices(e, k=max(2, int(traj_knn)))

    starts: list[int] = []
    if not diverse_starts:
        starts.append(int(rng.integers(0, len(g))))
        while len(starts) < n_traj:
            cand = int(rng.integers(0, len(g)))
            if cand in starts:
                continue
            d2 = np.sum((e[np.array(starts)] - e[cand : cand + 1]) ** 2, axis=1)
            if float(np.min(d2)) > 0.02 * float(np.mean(np.var(e, axis=0) + 1e-8)):
                starts.append(cand)
            elif float(rng.uniform()) < 0.08:
                starts.append(cand)

    visit = np.zeros((len(g),), dtype=np.int32)
    transitions: list[tuple[int, int, np.ndarray, int, int]] = []
    fallback_events: list[tuple[int, np.ndarray, int, int]] = []

    def _pick_dynamic_start(used_starts: list[int]) -> int:
        pool = int(min(len(g), max(256, 32 * n_traj)))
        if len(g) <= pool:
            cand_idx = np.arange(len(g), dtype=np.int64)
        else:
            cand_idx = rng.choice(len(g), size=pool, replace=False).astype(np.int64)
        if len(cand_idx) == 0:
            return int(rng.integers(0, len(g)))
        if len(used_starts) == 0:
            v = visit[cand_idx].astype(np.float32)
            min_v = float(np.min(v))
            good = cand_idx[v <= min_v + 1e-6]
            return int(good[int(rng.integers(0, len(good)))])

        d2 = np.sum(
            (e[cand_idx][:, None, :] - e[np.array(used_starts, dtype=np.int64)][None, :, :]) ** 2,
            axis=2,
        )
        d_score = np.min(d2, axis=1).astype(np.float32)
        v = visit[cand_idx].astype(np.float32)
        v_score = (np.max(v) - v) if len(v) > 0 else np.zeros_like(d_score)
        d_norm = d_score / (float(np.max(d_score)) + 1e-8)
        v_norm = v_score / (float(np.max(v_score)) + 1e-8)
        score = d_norm + 1*v_norm
        return int(cand_idx[int(np.argmax(score))])

    seq: list[np.ndarray] = []
    empty_neighbor_warn_count = 0
    for ti in range(n_traj):
        if diverse_starts:
            cur = _pick_dynamic_start(starts)
            starts.append(int(cur))
        else:
            cur = int(starts[ti % len(starts)])
        prev_idx = -1
        prev_dir = None
        for _ in range(t_len):
            seq.append(g[cur].astype(np.float32))
            visit[cur] += 1
            nbr = knn[cur]
            nbr = nbr[nbr != cur]
            if prev_idx >= 0:
                nbr = nbr[nbr != prev_idx]
            if len(nbr) == 0:
                raw = knn[cur]
                raw = raw[raw != cur]
                fallback_events.append((int(cur), raw.astype(np.int64), int(ti), int(_)))
                empty_neighbor_warn_count += 1
                if empty_neighbor_warn_count <= 10:
                    print(
                        "[warn][traj] empty neighbor set after filtering; "
                        f"traj={ti} step={_} cur={cur} prev={prev_idx} "
                        f"n_grid={len(g)} traj_knn={int(traj_knn)}"
                    )
                cur = int(rng.integers(0, len(g)))
                prev_idx = -1
                prev_dir = None
                continue

            cand = nbr[: min(14, len(nbr))]
            vec = (e[cand] - e[cur]).astype(np.float32)
            d = np.linalg.norm(vec, axis=1) + 1e-8
            u = vec / d[:, None]
            cov = visit[cand].astype(np.float32)
            cov_gain = (np.max(cov) - cov) if len(cov) > 0 else np.zeros_like(d)

            if prev_dir is None:
                score = -d + 0.65 * cov_gain
            else:
                a = np.clip(np.sum(u * prev_dir.reshape(1, -1), axis=1), -1.0, 1.0)
                score = 3.0 * a - 0.20 * d + 0.90 * cov_gain
                if float(rng.uniform()) < 0.08:
                    score = score + rng.normal(scale=0.35, size=score.shape).astype(np.float32)

            pick = int(np.argmax(score))
            nxt = int(cand[pick])
            transitions.append((int(cur), int(nxt), cand.astype(np.int64), int(ti), int(_)))
            new_vec = (e[nxt] - e[cur]).astype(np.float32)
            new_norm = float(np.linalg.norm(new_vec))
            if new_norm > 1e-8:
                new_u = new_vec / new_norm
                if prev_dir is None:
                    prev_dir = new_u.astype(np.float32)
                else:
                    prev_dir = (0.82 * prev_dir + 0.18 * new_u).astype(np.float32)
                    prev_dir = prev_dir / max(float(np.linalg.norm(prev_dir)), 1e-8)
            prev_idx = cur
            cur = nxt

    arr = np.asarray(seq, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((n, g.shape[1]), dtype=np.float32)

    if bool(debug_enabled):
        try:
            _plot_traj_jump_debug(
                g=g,
                arr=arr,
                transitions=transitions,
                fallback_events=fallback_events,
                traj_count=n_traj,
                traj_len=t_len,
                outdir=str(debug_outdir),
                name=str(debug_name),
                seed=int(seed),
                workspace_mapper=workspace_mapper,
            )
        except Exception as e:
            _print_red(f"[warn][traj_debug] plot failed: {e}")

    if len(arr) >= n:
        return arr[:n].astype(np.float32)
    rep = int(math.ceil(float(n) / float(len(arr))))
    tiled = np.tile(arr, (rep, 1)).astype(np.float32)
    return tiled[:n].astype(np.float32)


def _plot_traj_jump_debug(
    *,
    g: np.ndarray,
    arr: np.ndarray,
    transitions: list[tuple[int, int, np.ndarray, int, int]],
    fallback_events: list[tuple[int, np.ndarray, int, int]],
    traj_count: int,
    traj_len: int,
    outdir: str,
    name: str,
    seed: int,
    workspace_mapper: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    if len(arr) < 2:
        return
    if workspace_mapper is not None:
        g_ws = np.asarray(workspace_mapper(g), dtype=np.float32)
        arr_ws = np.asarray(workspace_mapper(arr), dtype=np.float32)
    else:
        g_ws = np.asarray(g[:, : min(3, g.shape[1])], dtype=np.float32)
        arr_ws = np.asarray(arr[:, : min(3, arr.shape[1])], dtype=np.float32)

    if g_ws.shape[1] < 3:
        pad = 3 - g_ws.shape[1]
        g_ws = np.pad(g_ws, ((0, 0), (0, pad)))
        arr_ws = np.pad(arr_ws, ((0, 0), (0, pad)))

    dists = []
    for cur_i, nxt_i, _, _, _ in transitions:
        d = float(np.linalg.norm(g_ws[nxt_i] - g_ws[cur_i]))
        if np.isfinite(d):
            dists.append(d)
    d_arr = np.asarray(dists, dtype=np.float32)
    if len(d_arr) > 0:
        med = float(np.median(d_arr))
        thr = max(float(3.0 * med), float(np.percentile(d_arr, 95) * 1.5))
    else:
        thr = float("inf")

    jump_events: list[tuple[int, np.ndarray, int, int, int | None, float | None]] = []
    for cur_i, nxt_i, cand_idx, ti, si in transitions:
        d = float(np.linalg.norm(g_ws[nxt_i] - g_ws[cur_i]))
        if np.isfinite(d) and d > thr:
            jump_events.append((cur_i, cand_idx, ti, si, nxt_i, d))
    for cur_i, cand_idx, ti, si in fallback_events:
        jump_events.append((cur_i, cand_idx, ti, si, None, None))
    if len(jump_events) == 0:
        return

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(0)
    gg = g_ws
    if len(gg) > 4000:
        idx = rng.choice(len(gg), size=4000, replace=False)
        gg = gg[idx]
    ax.scatter(gg[:, 0], gg[:, 1], gg[:, 2], s=2.0, c="#9CA3AF", alpha=0.12, linewidths=0, label="grid(ws)")

    colors = ["#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706", "#0891B2"]
    for i in range(max(1, int(traj_count))):
        a = i * int(traj_len)
        if a >= len(arr_ws):
            break
        b = min(len(arr_ws), (i + 1) * int(traj_len))
        seg = arr_ws[a:b]
        if len(seg) < 2:
            continue
        c = colors[i % len(colors)]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], "-", color=c, linewidth=1.0, alpha=0.40)

    shown_jump = False
    shown_knn = False
    shown_to = False
    for cur_i, cand_idx, _, _, nxt_i, _ in jump_events[:48]:
        p = g_ws[int(cur_i)]
        ax.scatter(
            [p[0]], [p[1]], [p[2]],
            s=46, c="#EF4444", edgecolors="black", linewidths=0.7, zorder=8,
            label=("jump point" if not shown_jump else None),
        )
        shown_jump = True
        cc = g_ws[np.asarray(cand_idx, dtype=np.int64)]
        if len(cc) > 0:
            ax.scatter(
                cc[:, 0], cc[:, 1], cc[:, 2],
                s=24, c="#F59E0B", alpha=0.92, linewidths=0,
                label=("knn candidates" if not shown_knn else None),
            )
            shown_knn = True
        if nxt_i is not None:
            q = g_ws[int(nxt_i)]
            ax.plot(
                [p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                "--", color="#EF4444", linewidth=1.1, alpha=0.92,
                label=("jump edge" if not shown_to else None),
            )
            shown_to = True

    ax.set_title(f"{name} traj jump debug (seed={int(seed)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if shown_jump or shown_knn or shown_to:
        ax.legend(loc="best", fontsize=8)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{name}_seed{int(seed)}_traj_jump_debug.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    _print_red(f"[saved][traj_debug] {out_path}")


def resolve_dataset(
    name: str,
    cfg: Any,
    *,
    optimize_ur5_train_only: bool = True,
    ur5_backend: str | None = None,
) -> dict[str, Any]:
    if name in FIXED_DEMONSTRATION_DIMS:
        demo = load_fixed_demonstrations(name)
        arm_up = name == "6d_spatial_arm_up_n6_py_traj"
        dim = FIXED_DEMONSTRATION_DIMS[name]
        if arm_up:
            labels = tuple(f"q{i + 1}" for i in range(6))
            codim = 2
        elif dim == 6:
            labels = ("x", "y", "z", "roll", "pitch", "yaw")
            codim = 3
        else:
            labels = ("x1", "y1", "z1", "roll1", "pitch1", "yaw1", "x2", "y2", "z2", "roll2", "pitch2", "yaw2")
            codim = 10
        result = {
            "name": name,
            "x_train": demo["x_train"],
            "grid": demo["grid"],
            "data_dim": dim,
            "axis_labels": labels,
            "true_codim": codim,
            "periodic_joint": arm_up,
            "demonstration_seed": demo["generation_seed"],
            "trajectory_lengths": demo["trajectory_lengths"],
            "train_noise": None,
            "train_missing_segments": None,
            "train_subsample": None,
        }
        if arm_up:
            result["ur5_backend"] = "analytic"
        return result
    if is_real_iiwa_dataset(name):
        canonical = canonical_real_iiwa_dataset_name(name)
        x_key = str(getattr(cfg, "real_iiwa_x_key", "x"))
        loaded = load_real_iiwa_train_set(str(canonical), x_key=x_key)
        x = np.asarray(loaded["x"], dtype=np.float32)
        orientation_repr = real_iiwa_orientation_representation(str(canonical))
        angle_columns = (
            (3, 4, 5)
            if x.ndim == 2 and int(x.shape[1]) == 6 and orientation_repr == "rpy"
            else ()
        )
        x_train, train_noise, train_missing_segments, train_subsample = _apply_training_data_transforms(
            x,
            cfg,
            angle_columns=angle_columns,
        )
        grid = real_iiwa_surface_grid(str(canonical), cfg, x_train=x_train)
        labels = (
            ("x", "y", "z")
            if int(x_train.shape[1]) == 3
            else ("x", "y", "z", "tool_z_x", "tool_z_y", "tool_z_z")
            if int(x_train.shape[1]) == 6 and orientation_repr == "tool_z"
            else ("x", "y", "z", "roll", "pitch", "yaw")
            if int(x_train.shape[1]) == 6
            else tuple(f"x{i + 1}" for i in range(int(x_train.shape[1])))
        )
        return {
            "name": str(canonical),
            "x_train": _as_float32(x_train),
            "grid": _as_float32(grid),
            "data_dim": int(x_train.shape[1]),
            "axis_labels": labels,
            "true_codim": int(loaded["spec"].get("true_codim", 1)),
            "periodic_joint": False,
            "real_iiwa": True,
            "real_iiwa_task_id": str(loaded["spec"]["task_id"]),
            "real_iiwa_train_set": str(loaded["train_entry"]["name"]),
            "real_iiwa_npz": str(loaded["npz_path"]),
            "plane_normal": np.asarray(loaded["plane_normal"], dtype=np.float32),
            "plane_d": float(loaded["plane_d"]),
            "orientation_condition": str(loaded["spec"].get("orientation_condition", "unknown")),
            "train_noise": train_noise,
            "train_missing_segments": train_missing_segments,
            "train_subsample": train_subsample,
        }

    cfg_gen = _cfg_for_generation(cfg)
    if name in BASE_2D_DATASETS:
        x, grid = generate_dataset(name, cfg_gen)
        labels = ("q1", "q2") if "2d_planar_arm_line_n2" in name else ("x", "y")
        return {
            "name": name,
            "x_train": _as_float32(x),
            "grid": _as_float32(grid),
            "data_dim": 2,
            "axis_labels": labels,
            "true_codim": 1,
            "periodic_joint": bool("arm_" in str(name)),
        }
    if name in BASE_3D_DATASETS:
        x, grid = generate_dataset(name, cfg_gen)
        labels = (
            ("q1", "q2", "q3")
            if (
                "3d_planar_arm_line_n3" in name
                or "3d_spatial_arm_plane_n3" in name
                or "3d_spatial_arm_ellip_n3" in name
                or "3d_spatial_arm_circle_n3" in name
            )
            else ("x", "y", "z")
        )
        true_codim = (
            2
            if (
                name
                in (
                    "3d_spiral",
                    "linked_curves_3d",
                    "3d_knot_spline_union",
                    "3d_knot_spline_union_traj",
                    "3d_knot_spline_triplet",
                    "3d_knot_spline_triplet_intersect",
                    "3d_knot_spline_triplet_near_intersect",
                    "3d_knot_spline_triplet_simple",
                    "3d_nested_ellipses_union",
                    "3d_linked_ellipses_union",
                )
                or name.startswith("3d_spatial_arm_ellip_n3")
                or name.startswith("3d_spatial_arm_circle_n3")
            )
            else 1
        )
        return {
            "name": name,
            "x_train": _as_float32(x),
            "grid": _as_float32(grid),
            "data_dim": 3,
            "axis_labels": labels,
            "true_codim": true_codim,
            "periodic_joint": bool("arm_" in str(name)),
        }
    if name in BASE_4D_DATASETS:
        x, grid = generate_dataset(name, cfg_gen)
        true_codim = (
            2
            if name
            in (
                "4d_warped_double_torus_surface",
                "4d_warped_double_torus_extra_handle_surface",
                "4d_warped_double_torus_extra_handle_surface_traj",
                "multi_handle_4d",
            )
            else 1
        )
        return {
            "name": name,
            "x_train": _as_float32(x),
            "grid": _as_float32(grid),
            "data_dim": 4,
            "axis_labels": ("x", "y", "z", "x4"),
            "true_codim": true_codim,
            "periodic_joint": False,
        }
    if name in BASE_6D_DATASETS:
        if name in (
            "6d_surf_normal",
            "6d_surf_normal_traj",
        ):
            x, grid = generate_dataset(name, cfg_gen)
            x_train, train_noise, train_missing_segments, train_subsample = _apply_training_data_transforms(
                x,
                cfg,
                angle_columns=(3, 4, 5),
            )
            return {
                "name": name,
                "x_train": x_train,
                "grid": _as_float32(grid),
                "data_dim": 6,
                "axis_labels": ("x", "y", "z", "roll", "pitch", "yaw"),
                "true_codim": 3,
                "periodic_joint": False,
                "train_noise": train_noise,
                "train_missing_segments": train_missing_segments,
                "train_subsample": train_subsample,
            }
        is_traj = name.endswith("_traj")
        base_name = str(name[: -len("_traj")] if is_traj else name)
        backend = str(ur5_backend or "").lower().strip()
        if backend in ("", "auto"):
            backend = "analytic" if base_name.endswith("_py") else "pybullet"
        if backend not in ("pybullet", "analytic"):
            raise ValueError(f"unknown ur5_backend '{ur5_backend}', expected 'pybullet' or 'analytic'")

        n_train = int(cfg.n_train)
        cfg_n_grid = _cfg_n_grid(cfg, default=4096)
        if is_traj:
            n_grid = int(
                max(
                    int(cfg_n_grid),
                    int(getattr(cfg, "traj_grid_min", 1)),
                )
            )
        else:
            n_grid = int(max(1, (1 if optimize_ur5_train_only else int(cfg_n_grid))))
        seed = int(cfg.seed)
        cache_path = _ur5_cache_path(
            dataset_name=base_name,
            backend=backend,
            seed=seed,
            n_train=n_train,
            n_grid=n_grid,
        )
        cached = _try_load_ur5_cache(cache_path)
        if cached is None and bool(optimize_ur5_train_only):
            cached = _try_load_ur5_cache_relaxed_n_grid(
                dataset_name=base_name,
                backend=backend,
                seed=seed,
                n_train=n_train,
            )
        if cached is not None:
            x, grid = cached
        else:
            if backend == "pybullet":
                x, grid = sample_ur5_upward_dataset(
                    n_train,
                    n_grid,
                    seed=seed,
                )
            else:
                x, grid = sample_ur5_upward_dataset_analytic(
                    n_train,
                    n_grid,
                    seed=seed,
                )
            _save_ur5_cache(cache_path, x, grid)
        return {
            "name": name,
            "x_train": _as_float32(x),
            "grid": _as_float32(grid),
            "data_dim": 6,
            "axis_labels": ("q1", "q2", "q3", "q4", "q5", "q6"),
            "true_codim": 2,
            "periodic_joint": True,
            "ur5_backend": backend,
        }
    if name in BASE_12D_DATASETS:
        x, grid = generate_dataset(name, cfg_gen)
        x_train, train_noise, train_missing_segments, train_subsample = _apply_training_data_transforms(
            x,
            cfg,
            angle_columns=(3, 4, 5, 9, 10, 11),
        )
        return {
            "name": name,
            "x_train": x_train,
            "grid": _as_float32(grid),
            "data_dim": 12,
            "axis_labels": ("x1", "y1", "z1", "roll1", "pitch1", "yaw1", "x2", "y2", "z2", "roll2", "pitch2", "yaw2"),
            "true_codim": 10,
            "periodic_joint": False,
            "train_noise": train_noise,
            "train_missing_segments": train_missing_segments,
            "train_subsample": train_subsample,
        }
    if name.startswith("3d_vz_") and name.endswith("_traj"):
        base = name[len("3d_vz_") : -len("_traj")]
        if base not in BASE_2D_DATASETS:
            raise ValueError(f"unknown lifted traj dataset base for 3d_vz_: {base}")
        cfg_base = dict(vars(cfg))
        n_train = int(cfg_base.get("n_train", 1))
        cfg_base_grid = int(cfg_base["traj_gene_n_grid"])
        cfg_base["n_grid"] = int(
            max(
                int(cfg_base_grid),
                int(getattr(cfg, "traj_grid_min", 1)),
            )
        )
        from types import SimpleNamespace

        x2, grid2 = generate_dataset(base, SimpleNamespace(**cfg_base))
        x3 = lift_xy_to_3d_var(x2, cfg).astype(np.float32)
        g3 = lift_xy_to_3d_var(grid2, cfg).astype(np.float32)
        traj_count_raw = int(getattr(cfg, "traj_count", max(12, n_train // 24)))
        traj_count = int(max(1, min(n_train, traj_count_raw)))
        traj_len = int(max(8, getattr(cfg, "traj_len", int(math.ceil(n_train / max(traj_count, 1))))))
        traj_knn = int(max(4, getattr(cfg, "traj_knn", 16)))
        x_traj = _traj_points_from_grid_embedded(
            g3,
            embed=g3,
            n_train=n_train,
            seed=int(getattr(cfg, "seed", 0)),
            traj_count=traj_count,
            traj_len=traj_len,
            traj_knn=traj_knn,
            diverse_starts=(str(name) == "3d_vz_2d_ellipse_traj"),
        )
        x_train, train_noise, train_missing_segments, train_subsample = _apply_training_data_transforms(
            x_traj,
            cfg,
        )
        labels = ("q1", "q2", "q3") if "2d_planar_arm_line_n2" in base else ("x", "y", "z")
        return {
            "name": name,
            "x_train": x_train,
            "grid": g3.astype(np.float32),
            "data_dim": 3,
            "axis_labels": labels,
            "true_codim": 2,
            "periodic_joint": bool("arm_" in str(base)),
            "train_noise": train_noise,
            "train_missing_segments": train_missing_segments,
            "train_subsample": train_subsample,
        }
    if name.startswith("3d_0z_"):
        base = name[len("3d_0z_") :]
        if base not in BASE_2D_DATASETS:
            raise ValueError(f"unknown lifted dataset base for 3d_0z_: {base}")
        x2, grid2 = generate_dataset(base, cfg_gen)
        labels = ("q1", "q2", "q3") if "2d_planar_arm_line_n2" in base else ("x", "y", "z")
        return {
            "name": name,
            "x_train": lift_xy_to_3d_zero(x2).astype(np.float32),
            "grid": lift_xy_to_3d_zero(grid2).astype(np.float32),
            "data_dim": 3,
            "axis_labels": labels,
            "true_codim": 2,
            "periodic_joint": bool("arm_" in str(base)),
        }
    if name.startswith("3d_vz_"):
        base = name[len("3d_vz_") :]
        if base not in BASE_2D_DATASETS:
            raise ValueError(f"unknown lifted dataset base for 3d_vz_: {base}")
        x2, grid2 = generate_dataset(base, cfg_gen)
        labels = ("q1", "q2", "q3") if "2d_planar_arm_line_n2" in base else ("x", "y", "z")
        return {
            "name": name,
            "x_train": lift_xy_to_3d_var(x2, cfg).astype(np.float32),
            "grid": lift_xy_to_3d_var(grid2, cfg).astype(np.float32),
            "data_dim": 3,
            "axis_labels": labels,
            "true_codim": 2,
            "periodic_joint": bool("arm_" in str(base)),
        }
    raise ValueError(f"unknown dataset '{name}'")
