#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ros.iiwa_bag import load_iiwa_npz
from datasets.ros.transforms import quat_to_matrix_xyzw


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float64)


def _rpy_from_rotmat_zyx(rot: np.ndarray) -> np.ndarray:
    sy = -float(rot[2, 0])
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = float(np.arcsin(sy))
    cp = float(np.cos(pitch))
    if abs(cp) > 1e-8:
        roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-rot[0, 1], rot[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _quat_to_rpy_zyx(quat_xyzw: np.ndarray) -> np.ndarray:
    if quat_xyzw.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack([_rpy_from_rotmat_zyx(quat_to_matrix_xyzw(q)) for q in quat_xyzw], axis=0)


def _rpy_wrap_stats(rpy_wrapped: np.ndarray) -> dict[str, object]:
    if rpy_wrapped.shape[0] < 2:
        zeros = [0.0, 0.0, 0.0]
        return {
            "raw_max_abs_step_deg": zeros,
            "wrapped_max_abs_step_deg": zeros,
            "raw_jump_gt_pi_count": 0,
            "raw_jump_gt_pi_indices": [],
        }
    raw_step = np.diff(rpy_wrapped, axis=0)
    wrapped_step = _wrap_pi(raw_step)
    raw_jump = np.argwhere(np.abs(raw_step) > np.pi)
    return {
        "raw_max_abs_step_deg": np.degrees(np.max(np.abs(raw_step), axis=0)).astype(float).tolist(),
        "wrapped_max_abs_step_deg": np.degrees(np.max(np.abs(wrapped_step), axis=0)).astype(float).tolist(),
        "raw_jump_gt_pi_count": int(raw_jump.shape[0]),
        "raw_jump_gt_pi_indices": raw_jump[:20].astype(int).tolist(),
    }


def _nearest_indices(source_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    source = np.asarray(source_t, dtype=np.float64).reshape(-1)
    query = np.asarray(query_t, dtype=np.float64).reshape(-1)
    if source.size == 0 or query.size == 0:
        return np.zeros((query.size,), dtype=np.int64)
    idx = np.searchsorted(source, query)
    idx0 = np.clip(idx - 1, 0, source.size - 1)
    idx1 = np.clip(idx, 0, source.size - 1)
    choose_prev = np.abs(query - source[idx0]) <= np.abs(source[idx1] - query)
    return np.where(choose_prev, idx0, idx1).astype(np.int64)


def _dedupe_consecutive(points: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.arange(points.shape[0], dtype=np.int64)
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], step > float(eps)])
    return np.flatnonzero(keep).astype(np.int64)


def _arclength_downsample_indices(points: np.ndarray, target_spacing: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    n = p.shape[0]
    if n <= 2 or float(target_spacing) <= 0.0:
        return np.arange(n, dtype=np.int64)

    keep = [0]
    last = p[0].copy()
    accum = 0.0
    spacing = float(target_spacing)
    for i in range(1, n):
        seg = float(np.linalg.norm(p[i] - p[i - 1]))
        accum += seg
        if accum >= spacing:
            keep.append(i)
            last = p[i].copy()
            accum = 0.0
    if keep[-1] != n - 1:
        keep.append(n - 1)
    return np.asarray(keep, dtype=np.int64)


def _segment_stats(points: np.ndarray) -> dict[str, float]:
    if points.shape[0] < 2:
        return {
            "count": float(points.shape[0]),
            "path_length_m": 0.0,
            "adjacent_mean_m": 0.0,
            "adjacent_median_m": 0.0,
            "adjacent_p90_m": 0.0,
            "adjacent_max_m": 0.0,
        }
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return {
        "count": float(points.shape[0]),
        "path_length_m": float(np.sum(d)),
        "adjacent_mean_m": float(np.mean(d)),
        "adjacent_median_m": float(np.median(d)),
        "adjacent_p90_m": float(np.quantile(d, 0.90)),
        "adjacent_max_m": float(np.max(d)),
    }


def preprocess_one(
    path: Path,
    *,
    crop_start_frac: float,
    crop_end_frac: float,
    target_spacing: float,
    dedupe_eps: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    data = load_iiwa_npz(path)
    t_pose = np.asarray(data["t_pose"], dtype=np.float64)
    tcp_pos = np.asarray(data["tcp_pos"], dtype=np.float64)
    tcp_quat = np.asarray(data["tcp_quat"], dtype=np.float64)
    ee_pos = np.asarray(data["ee_pos"], dtype=np.float64)
    ee_quat = np.asarray(data["ee_quat"], dtype=np.float64)

    n = int(tcp_pos.shape[0])
    a = int(np.floor(float(crop_start_frac) * n))
    b = int(np.floor(float(crop_end_frac) * n))
    a = int(np.clip(a, 0, max(n - 1, 0)))
    b = int(np.clip(b, a + 2, n))

    crop_slice = slice(a, b)
    p_crop = tcp_pos[crop_slice]
    dedupe_local = _dedupe_consecutive(p_crop, eps=dedupe_eps)
    p_dedup = p_crop[dedupe_local]
    down_local = _arclength_downsample_indices(p_dedup, target_spacing=target_spacing)
    selected_local = dedupe_local[down_local]
    selected = selected_local + a

    t_sel = t_pose[selected]
    q_idx = _nearest_indices(np.asarray(data["t_joint"], dtype=np.float64), t_sel)
    vel_idx = _nearest_indices(np.asarray(data["t_ee_vel"], dtype=np.float64), t_sel)

    arrays = {
        "t": t_sel.astype(np.float64),
        "tcp_pos": tcp_pos[selected].astype(np.float64),
        "tcp_quat": tcp_quat[selected].astype(np.float64),
        "ee_pos": ee_pos[selected].astype(np.float64),
        "ee_quat": ee_quat[selected].astype(np.float64),
        "q": np.asarray(data["q"], dtype=np.float64)[q_idx],
        "dq": np.asarray(data["dq"], dtype=np.float64)[q_idx],
        "tcp_linear_vel": np.asarray(data["tcp_linear_vel"], dtype=np.float64)[vel_idx],
        "tcp_angular_vel": np.asarray(data["tcp_angular_vel"], dtype=np.float64)[vel_idx],
    }
    tcp_rpy = _wrap_pi(_quat_to_rpy_zyx(arrays["tcp_quat"]))
    ee_rpy = _wrap_pi(_quat_to_rpy_zyx(arrays["ee_quat"]))
    tcp_rpy_unwrapped = np.unwrap(tcp_rpy, axis=0)
    ee_rpy_unwrapped = np.unwrap(ee_rpy, axis=0)
    arrays.update(
        {
            "tcp_rpy": tcp_rpy,
            "tcp_rpy_unwrapped": tcp_rpy_unwrapped,
            "tcp_pose_rpy": np.concatenate([arrays["tcp_pos"], tcp_rpy], axis=1).astype(np.float64),
            "tcp_pose_rpy_unwrapped": np.concatenate([arrays["tcp_pos"], tcp_rpy_unwrapped], axis=1).astype(np.float64),
            "ee_rpy": ee_rpy,
            "ee_rpy_unwrapped": ee_rpy_unwrapped,
            "ee_pose_rpy": np.concatenate([arrays["ee_pos"], ee_rpy], axis=1).astype(np.float64),
            "ee_pose_rpy_unwrapped": np.concatenate([arrays["ee_pos"], ee_rpy_unwrapped], axis=1).astype(np.float64),
        }
    )
    meta = {
        "source": str(path),
        "original_count": n,
        "crop_start_index": int(a),
        "crop_end_index": int(b),
        "cropped_count": int(b - a),
        "deduped_count": int(p_dedup.shape[0]),
        "selected_count": int(selected.shape[0]),
        "crop_raw_stats": _segment_stats(p_crop),
        "selected_stats": _segment_stats(arrays["tcp_pos"]),
        "tcp_rpy_wrap_stats": _rpy_wrap_stats(tcp_rpy),
    }
    return arrays, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop and downsample iiwa TCP demonstrations for constraint learning.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("datasets/real_iiwa/processed/probe_tcp_pos_8mm.npz"))
    parser.add_argument("--crop-start-frac", type=float, default=0.20)
    parser.add_argument("--crop-end-frac", type=float, default=0.90)
    parser.add_argument(
        "--target-spacing",
        type=float,
        default=0.008,
        help="Target TCP arclength spacing in meters. Default follows SurfNormal median adjacent xyz distance.",
    )
    parser.add_argument("--dedupe-eps", type=float, default=1e-10)
    parser.add_argument(
        "--x-space",
        choices=("tcp_pos", "tcp_pose_rpy", "tcp_pose_rpy_unwrapped"),
        default="tcp_pos",
        help="Which preprocessed representation to expose as x/x_raw.",
    )
    parser.add_argument("--normalize", action="store_true", help="Save x as normalized x_space instead of raw x_space.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_arrays = []
    metadata = []
    traj_id_parts = []
    starts = []
    cursor = 0
    for traj_id, path in enumerate(args.inputs):
        arrays, meta = preprocess_one(
            path,
            crop_start_frac=float(args.crop_start_frac),
            crop_end_frac=float(args.crop_end_frac),
            target_spacing=float(args.target_spacing),
            dedupe_eps=float(args.dedupe_eps),
        )
        all_arrays.append(arrays)
        metadata.append(meta)
        n = arrays["tcp_pos"].shape[0]
        starts.append(cursor)
        traj_id_parts.append(np.full((n,), traj_id, dtype=np.int64))
        cursor += n
        print(
            f"{path.parent.name}: crop {meta['crop_start_index']}:{meta['crop_end_index']} "
            f"cropped={meta['cropped_count']} selected={meta['selected_count']} "
            f"path={meta['selected_stats']['path_length_m']:.3f}m "
            f"median_step={meta['selected_stats']['adjacent_median_m']:.4f}m"
        )

    merged = {name: np.concatenate([a[name] for a in all_arrays], axis=0) for name in all_arrays[0]}
    traj_id_arr = np.concatenate(traj_id_parts, axis=0)
    traj_start_indices = np.asarray(starts, dtype=np.int64)

    x_raw = merged[str(args.x_space)].astype(np.float64)
    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0)
    std = np.maximum(std, 1e-8)
    x = (x_raw - mean) / std if bool(args.normalize) else x_raw.copy()

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(
        {
            "inputs": [str(p) for p in args.inputs],
            "crop_start_frac": float(args.crop_start_frac),
            "crop_end_frac": float(args.crop_end_frac),
            "target_spacing_m": float(args.target_spacing),
            "target_spacing_source": "6d_surf_normal_traj median adjacent xyz distance is approximately 0.008 m.",
            "x_space": str(args.x_space),
            "rpy_convention": (
                "ZYX Euler RPY, matching 6d_surf_normal: "
                "R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Wrapped RPY arrays are in [-pi, pi); "
                "unwrapped arrays are provided for diagnostics or optional continuous-angle training."
            ),
            "normalized_x": bool(args.normalize),
            "trajectories": metadata,
        },
        indent=2,
    )
    np.savez_compressed(
        out,
        x=x.astype(np.float32),
        x_raw=x_raw.astype(np.float32),
        t=merged["t"].astype(np.float64),
        tcp_pos=merged["tcp_pos"].astype(np.float32),
        tcp_quat=merged["tcp_quat"].astype(np.float32),
        tcp_rpy=merged["tcp_rpy"].astype(np.float32),
        tcp_rpy_unwrapped=merged["tcp_rpy_unwrapped"].astype(np.float32),
        tcp_pose_rpy=merged["tcp_pose_rpy"].astype(np.float32),
        tcp_pose_rpy_unwrapped=merged["tcp_pose_rpy_unwrapped"].astype(np.float32),
        ee_pos=merged["ee_pos"].astype(np.float32),
        ee_quat=merged["ee_quat"].astype(np.float32),
        ee_rpy=merged["ee_rpy"].astype(np.float32),
        ee_rpy_unwrapped=merged["ee_rpy_unwrapped"].astype(np.float32),
        ee_pose_rpy=merged["ee_pose_rpy"].astype(np.float32),
        ee_pose_rpy_unwrapped=merged["ee_pose_rpy_unwrapped"].astype(np.float32),
        q=merged["q"].astype(np.float32),
        dq=merged["dq"].astype(np.float32),
        tcp_linear_vel=merged["tcp_linear_vel"].astype(np.float32),
        tcp_angular_vel=merged["tcp_angular_vel"].astype(np.float32),
        traj_id=traj_id_arr,
        traj_start_indices=traj_start_indices,
        normalization_mean=mean.astype(np.float32),
        normalization_std=std.astype(np.float32),
        metadata_json=np.asarray(meta_json),
    )
    print(f"[saved] {out}")
    print(f"total selected points: {x_raw.shape[0]}")
    print(f"normalization mean: {mean}")
    print(f"normalization std: {std}")


if __name__ == "__main__":
    main()
