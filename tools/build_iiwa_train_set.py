#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _dedupe_consecutive(points: np.ndarray, eps: float) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.arange(points.shape[0], dtype=np.int64)
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], step > float(eps)])
    return np.flatnonzero(keep).astype(np.int64)


def _arclength_downsample_indices(points: np.ndarray, target_spacing: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    n = int(p.shape[0])
    if n <= 2 or float(target_spacing) <= 0.0:
        return np.arange(n, dtype=np.int64)
    keep = [0]
    accum = 0.0
    spacing = float(target_spacing)
    for i in range(1, n):
        accum += float(np.linalg.norm(p[i] - p[i - 1]))
        if accum >= spacing:
            keep.append(i)
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


def _task_root(path: Path) -> Path | None:
    p = path.resolve()
    for parent in p.parents:
        if parent.parent.name == "tasks":
            return parent
    return None


def _spacing_label(args: argparse.Namespace) -> str:
    if str(args.subsample_mode) == "none":
        return "nosubsample"
    if str(args.subsample_mode) == "index-stride":
        return f"stride{int(args.index_stride)}"
    mm = int(round(float(args.target_spacing) * 1000.0))
    return f"{mm}mm"


def _default_out_path(task: Path | None, inputs: list[Path], args: argparse.Namespace) -> Path:
    name = str(args.name or "").strip()
    if not name:
        name = f"{args.x_space}_{_spacing_label(args)}_manual_crop_v001"
    if task is not None:
        return task / "processed" / "train_sets" / f"{name}.npz"
    common = _task_root(inputs[0]) if inputs else None
    if common is not None:
        return common / "processed" / "train_sets" / f"{name}.npz"
    return Path("datasets/real_iiwa/processed/train_sets") / f"{name}.npz"


def _default_figure_path(out: Path) -> Path:
    task = _task_root(out)
    if task is not None:
        return task / "figures" / f"{out.stem}_density.png"
    return out.with_name(f"{out.stem}_density.png")


def _axis_limits(points: np.ndarray, pad_ratio: float = 0.55) -> list[tuple[float, float]]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    span = max(float((hi - lo).max()), 0.05)
    return [(float(c - pad_ratio * span), float(c + pad_ratio * span)) for c in center]


def _orientation_axes(quat_xyzw: np.ndarray, axis: str) -> np.ndarray:
    axis_vec = {
        "x": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    }[axis]
    dirs = np.empty((quat_xyzw.shape[0], 3), dtype=np.float64)
    for i, quat in enumerate(quat_xyzw):
        dirs[i] = quat_to_matrix_xyzw(quat) @ axis_vec
    return dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _optional_nearest(data: dict[str, np.ndarray], time_key: str, value_key: str, t_query: np.ndarray) -> np.ndarray | None:
    if time_key not in data or value_key not in data:
        return None
    t = np.asarray(data[time_key], dtype=np.float64)
    values = np.asarray(data[value_key])
    if t.size == 0 or values.shape[:1] != t.shape[:1]:
        return None
    return values[_nearest_indices(t, t_query)]


def _selected_indices(points: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dedupe_idx = _dedupe_consecutive(points, eps=float(args.dedupe_eps))
    if str(args.subsample_mode) == "none":
        local = np.arange(dedupe_idx.shape[0], dtype=np.int64)
    elif str(args.subsample_mode) == "index-stride":
        stride = max(1, int(args.index_stride))
        local = np.arange(0, dedupe_idx.shape[0], stride, dtype=np.int64)
        if local.size and local[-1] != dedupe_idx.shape[0] - 1:
            local = np.concatenate([local, np.asarray([dedupe_idx.shape[0] - 1], dtype=np.int64)])
    elif str(args.subsample_mode) == "arclength":
        local = _arclength_downsample_indices(points[dedupe_idx], target_spacing=float(args.target_spacing))
    else:
        raise ValueError(f"unsupported subsample_mode={args.subsample_mode!r}")
    return dedupe_idx, dedupe_idx[local]


def _build_one(path: Path, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = _load_npz(path)
    t_pose = np.asarray(data["t_pose"], dtype=np.float64)
    tcp_pos = np.asarray(data["tcp_pos"], dtype=np.float64)
    tcp_quat = np.asarray(data["tcp_quat"], dtype=np.float64)
    ee_pos = np.asarray(data["ee_pos"], dtype=np.float64)
    ee_quat = np.asarray(data["ee_quat"], dtype=np.float64)

    dedupe_idx, selected = _selected_indices(tcp_pos, args)
    t_sel = t_pose[selected]
    arrays: dict[str, np.ndarray] = {
        "t": t_sel.astype(np.float64),
        "source_pose_index": selected.astype(np.int64),
        "tcp_pos": tcp_pos[selected].astype(np.float64),
        "tcp_quat": tcp_quat[selected].astype(np.float64),
        "ee_pos": ee_pos[selected].astype(np.float64),
        "ee_quat": ee_quat[selected].astype(np.float64),
    }

    for time_key, value_key, out_key in (
        ("t_joint", "q", "q"),
        ("t_joint", "dq", "dq"),
        ("t_joint", "effort", "effort"),
        ("t_ee_vel", "tcp_linear_vel", "tcp_linear_vel"),
        ("t_ee_vel", "tcp_angular_vel", "tcp_angular_vel"),
        ("t_ee_vel", "ee_linear_vel", "ee_linear_vel"),
        ("t_ee_vel", "ee_angular_vel", "ee_angular_vel"),
    ):
        nearest = _optional_nearest(data, time_key, value_key, t_sel)
        if nearest is not None:
            arrays[out_key] = np.asarray(nearest, dtype=np.float64)

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

    meta: dict[str, Any] = {
        "source_crop": str(path),
        "source_crop_metadata": json.loads(str(data["crop_metadata_json"])) if "crop_metadata_json" in data else None,
        "original_count": int(tcp_pos.shape[0]),
        "deduped_count": int(dedupe_idx.shape[0]),
        "selected_count": int(selected.shape[0]),
        "subsample_mode": str(args.subsample_mode),
        "target_spacing_m": float(args.target_spacing),
        "index_stride": int(args.index_stride),
        "raw_stats": _segment_stats(tcp_pos),
        "selected_stats": _segment_stats(arrays["tcp_pos"]),
        "tcp_rpy_wrap_stats": _rpy_wrap_stats(tcp_rpy),
    }
    return arrays, meta


def _task_crops(task: Path) -> list[Path]:
    crops = sorted((task / "processed" / "crops").glob("*.npz"))
    if not crops:
        raise FileNotFoundError(f"No crop npz files found under {task / 'processed' / 'crops'}")
    return crops


def _update_task_json(
    task: Path,
    out: Path,
    args: argparse.Namespace,
    *,
    total_count: int,
    figure_path: Path | None,
) -> None:
    task_json = task / "task.json"
    if not task_json.exists():
        return
    data = json.loads(task_json.read_text())
    rel = str(out.relative_to(task))
    entry = {
        "name": out.stem,
        "path": rel,
        "x_space": str(args.x_space),
        "subsample_mode": str(args.subsample_mode),
        "target_spacing_m": float(args.target_spacing),
        "point_count": int(total_count),
    }
    if figure_path is not None:
        entry["figure"] = str(figure_path.relative_to(task)) if figure_path.is_relative_to(task) else str(figure_path)
    train_sets = list(data.get("train_sets", []))
    train_sets = [e for e in train_sets if e.get("path") != rel and e.get("name") != out.stem]
    train_sets.append(entry)
    data["train_sets"] = train_sets
    task_json.write_text(json.dumps(data, indent=2) + "\n")


def _plot_train_set_figure(
    out: Path,
    figure_path: Path,
    *,
    line_width: float,
    point_size: float,
    orientation_axis: str,
    arrow_every: int,
    arrow_length: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(out, allow_pickle=True) as data:
        points = np.asarray(data["tcp_pos"], dtype=np.float64)
        quat = np.asarray(data["tcp_quat"], dtype=np.float64)
        traj_id = np.asarray(data["traj_id"], dtype=np.int64)
        meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}

    dirs = _orientation_axes(quat, str(orientation_axis))
    limits = _axis_limits(points)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig = plt.figure(figsize=(8.2, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    for tid in np.unique(traj_id):
        idx = np.flatnonzero(traj_id == tid)
        pts = points[idx]
        c = colors[int(tid) % len(colors)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=c, lw=float(line_width), alpha=0.75)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=c,
            s=float(point_size),
            alpha=0.95,
            depthshade=True,
            label=f"traj {int(tid)} ({pts.shape[0]} pts)",
        )
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="limegreen", edgecolor="black", s=max(45.0, point_size * 2.5))
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color="red", edgecolor="black", s=max(50.0, point_size * 2.8), marker="X")
        arrow_step = max(1, int(arrow_every))
        arrow_idx = idx[::arrow_step]
        if arrow_idx.size and arrow_idx[-1] != idx[-1]:
            arrow_idx = np.concatenate([arrow_idx, idx[-1:]])
        ax.quiver(
            points[arrow_idx, 0],
            points[arrow_idx, 1],
            points[arrow_idx, 2],
            dirs[arrow_idx, 0],
            dirs[arrow_idx, 1],
            dirs[arrow_idx, 2],
            length=float(arrow_length),
            normalize=True,
            color=c,
            linewidth=0.9,
            alpha=0.9,
            arrow_length_ratio=0.35,
        )

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=24, azim=-58)
    ax.set_title(
        f"{out.stem}: subsampled train set "
        f"({points.shape[0]} pts, {meta.get('subsample_mode', 'unknown')}, "
        f"{float(meta.get('target_spacing_m', 0.0)) * 1000.0:.1f} mm, "
        f"+{str(orientation_axis).upper()} axis)"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=int(dpi))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build learning-ready iiwa train sets from cropped demonstrations.")
    parser.add_argument("inputs", nargs="*", type=Path, help="Cropped demonstration npz files.")
    parser.add_argument("--task", type=Path, default=None, help="Task directory; uses processed/crops/*.npz as inputs.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--name", default=None, help="Dataset version name without .npz.")
    parser.add_argument("--subsample-mode", choices=("arclength", "index-stride", "none"), default="arclength")
    parser.add_argument(
        "--target-spacing",
        type=float,
        default=0.008,
        help="TCP arclength spacing in meters. Default 8mm follows SurfNormal/previous iiwa preprocessing.",
    )
    parser.add_argument("--index-stride", type=int, default=25, help="Used only for --subsample-mode index-stride.")
    parser.add_argument("--dedupe-eps", type=float, default=1e-10)
    parser.add_argument(
        "--x-space",
        choices=("tcp_pos", "tcp_pose_rpy", "tcp_pose_rpy_unwrapped"),
        default="tcp_pos",
        help="Representation exposed as x/x_raw for training.",
    )
    parser.add_argument("--normalize", action="store_true", help="Save x as normalized x_space instead of raw x_space.")
    parser.add_argument("--update-task-json", action="store_true", help="Register the generated train set in task.json.")
    parser.add_argument("--figure-out", type=Path, default=None)
    parser.add_argument("--no-save-figure", action="store_true", help="Do not save a train-set density figure.")
    parser.add_argument("--figure-line-width", type=float, default=0.55)
    parser.add_argument("--figure-point-size", type=float, default=16.0)
    parser.add_argument("--figure-orientation-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--figure-arrow-every", type=int, default=12)
    parser.add_argument("--figure-arrow-length", type=float, default=0.035)
    parser.add_argument("--figure-dpi", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = args.task.resolve() if args.task is not None else None
    inputs = _task_crops(task) if task is not None else [p.resolve() for p in args.inputs]
    if not inputs:
        raise ValueError("Provide crop npz files or --task")
    out = args.out or _default_out_path(task, inputs, args)
    figure_out = None if bool(args.no_save_figure) else (args.figure_out or _default_figure_path(out))

    all_arrays: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, Any]] = []
    traj_id_parts: list[np.ndarray] = []
    starts: list[int] = []
    cursor = 0
    for traj_id, path in enumerate(inputs):
        arrays, meta = _build_one(path, args)
        all_arrays.append(arrays)
        metadata.append(meta)
        n = int(arrays["tcp_pos"].shape[0])
        starts.append(cursor)
        traj_id_parts.append(np.full((n,), traj_id, dtype=np.int64))
        cursor += n
        print(
            f"{path.parent.parent.parent.name}/{path.stem}: "
            f"raw={meta['original_count']} deduped={meta['deduped_count']} selected={meta['selected_count']} "
            f"path={meta['selected_stats']['path_length_m']:.3f}m "
            f"median_step={meta['selected_stats']['adjacent_median_m']:.4f}m"
        )

    common_keys = set(all_arrays[0])
    for arrays in all_arrays[1:]:
        common_keys &= set(arrays)
    merged = {name: np.concatenate([a[name] for a in all_arrays], axis=0) for name in sorted(common_keys)}
    traj_id_arr = np.concatenate(traj_id_parts, axis=0)
    traj_start_indices = np.asarray(starts, dtype=np.int64)

    x_raw = merged[str(args.x_space)].astype(np.float64)
    mean = x_raw.mean(axis=0)
    std = np.maximum(x_raw.std(axis=0), 1e-8)
    x = (x_raw - mean) / std if bool(args.normalize) else x_raw.copy()

    meta_json = json.dumps(
        {
            "inputs": [str(p) for p in inputs],
            "task": str(task) if task is not None else None,
            "subsample_mode": str(args.subsample_mode),
            "target_spacing_m": float(args.target_spacing),
            "target_spacing_source": "Default 0.008 m follows SurfNormal median adjacent xyz distance and previous iiwa preprocessing.",
            "index_stride": int(args.index_stride),
            "dedupe_eps": float(args.dedupe_eps),
            "x_space": str(args.x_space),
            "normalized_x": bool(args.normalize),
            "figure": str(figure_out) if figure_out is not None else None,
            "figure_orientation_axis": str(args.figure_orientation_axis),
            "figure_arrow_every": int(args.figure_arrow_every),
            "figure_arrow_length": float(args.figure_arrow_length),
            "rpy_convention": (
                "ZYX Euler RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll). "
                "Wrapped RPY arrays are in [-pi, pi); unwrapped arrays are diagnostic/optional."
            ),
            "trajectories": metadata,
        },
        indent=2,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    save_arrays: dict[str, np.ndarray] = {
        "x": x.astype(np.float32),
        "x_raw": x_raw.astype(np.float32),
        "traj_id": traj_id_arr,
        "traj_start_indices": traj_start_indices,
        "normalization_mean": mean.astype(np.float32),
        "normalization_std": std.astype(np.float32),
        "metadata_json": np.asarray(meta_json),
    }
    for key, value in merged.items():
        if key in {"traj_id", "traj_start_indices", "metadata_json"}:
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in {"f", "c"} and key != "t":
            save_arrays[key] = arr.astype(np.float32)
        else:
            save_arrays[key] = arr
    np.savez_compressed(out, **save_arrays)
    if figure_out is not None:
        _plot_train_set_figure(
            out,
            figure_out,
            line_width=float(args.figure_line_width),
            point_size=float(args.figure_point_size),
            orientation_axis=str(args.figure_orientation_axis),
            arrow_every=int(args.figure_arrow_every),
            arrow_length=float(args.figure_arrow_length),
            dpi=int(args.figure_dpi),
        )
    if bool(args.update_task_json) and task is not None:
        _update_task_json(task, out, args, total_count=int(x_raw.shape[0]), figure_path=figure_out)
    print(f"[saved] {out}")
    if figure_out is not None:
        print(f"[figure] {figure_out}")
    print(f"total selected points: {x_raw.shape[0]}")
    print(f"normalization mean: {mean}")
    print(f"normalization std: {std}")


if __name__ == "__main__":
    main()
