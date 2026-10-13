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

from datasets.real_iiwa.registry import (  # noqa: E402
    REAL_IIWA_DATASETS,
    canonical_real_iiwa_dataset_name,
    real_iiwa_neural_surface_z_normal,
    real_iiwa_world_to_normalized_xyz,
)


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def _stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1) * float(scale)
    if len(v) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def _arc_length_indices(position_world: np.ndarray, threshold_m: float) -> np.ndarray:
    pos = np.asarray(position_world, dtype=np.float64)
    if len(pos) == 0:
        return np.zeros((0,), dtype=np.int64)
    if threshold_m <= 0.0:
        return np.arange(len(pos), dtype=np.int64)
    indices = [0]
    accum = 0.0
    for i in range(1, len(pos)):
        accum += float(np.linalg.norm(pos[i] - pos[i - 1]))
        if accum >= threshold_m:
            indices.append(i)
            accum = 0.0
    if indices[-1] != len(pos) - 1:
        indices.append(len(pos) - 1)
    return np.asarray(indices, dtype=np.int64)


def _surface_errors(dataset: str, position_world: np.ndarray, tool_z: np.ndarray) -> dict[str, np.ndarray]:
    pos = np.asarray(position_world, dtype=np.float64)
    tool = _normalize(tool_z)
    z_pred, normal = real_iiwa_neural_surface_z_normal(dataset, pos[:, 0], pos[:, 1])
    normal = _normalize(normal)
    grad_norm = np.linalg.norm(normal[:, :2] / np.maximum(normal[:, 2:3], 1e-12), axis=1)
    signed = (pos[:, 2] - np.asarray(z_pred, dtype=np.float64)) / np.sqrt(1.0 + grad_norm**2)
    target_sign = float(REAL_IIWA_DATASETS[canonical_real_iiwa_dataset_name(dataset)].get("target_tool_axis_sign", -1.0))
    target_tool_z = target_sign * normal
    dot = np.sum(tool * target_tool_z, axis=1)
    angle_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    true_pos = pos.copy()
    true_pos[:, 2] = np.asarray(z_pred, dtype=np.float64)
    return {
        "position_true_world": true_pos.astype(np.float32),
        "surface_normal": normal.astype(np.float32),
        "target_tool_z": target_tool_z.astype(np.float32),
        "signed_position_error_m": signed.astype(np.float32),
        "position_error_m": np.abs(signed).astype(np.float32),
        "orientation_error_deg": angle_deg.astype(np.float32),
    }


def _csv_arrays(csv_path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.ndim == 0:
        data = data.reshape(1)
    return {str(name): np.asarray(data[name]) for name in data.dtype.names or ()}


def _column_stack(arrays: dict[str, np.ndarray], names: list[str], dtype: Any = np.float32) -> np.ndarray:
    return np.column_stack([np.asarray(arrays[name], dtype=np.float64) for name in names]).astype(dtype)


def _load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_error_hist(
    *,
    position_error_m: np.ndarray,
    signed_position_error_m: np.ndarray,
    orientation_error_deg: np.ndarray,
    output: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    axes[0].hist(np.asarray(position_error_m) * 1000.0, bins=60, color="#3b75af", alpha=0.9)
    axes[0].set_xlabel("abs position error (mm)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Position")
    axes[1].hist(np.asarray(signed_position_error_m) * 1000.0, bins=60, color="#9b6a4d", alpha=0.9)
    axes[1].axvline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel("signed position error (mm)")
    axes[1].set_title("Signed Position")
    axes[2].hist(np.asarray(orientation_error_deg), bins=60, color="#57965c", alpha=0.9)
    axes[2].set_xlabel("tool-z orientation error (deg)")
    axes[2].set_title("Orientation")
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_xy_trajectory(position_world: np.ndarray, output: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = np.asarray(position_world, dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.8), constrained_layout=True)
    ax.plot(pos[:, 0], pos[:, 1], "-", lw=1.2, color="#2f6f9f", alpha=0.9)
    ax.plot(pos[0, 0], pos[0, 1], "o", ms=5, color="#2f6f9f", markeredgecolor="black", label="start")
    ax.plot(pos[-1, 0], pos[-1, 1], "s", ms=5, color="#2f6f9f", markeredgecolor="black", label="end")
    for frac in (0.2, 0.4, 0.6, 0.8):
        j = int(np.clip(round(frac * (len(pos) - 2)), 0, max(0, len(pos) - 2)))
        p0 = pos[j, :2]
        p1 = pos[j + 1, :2]
        if float(np.linalg.norm(p1 - p0)) < 1e-9:
            continue
        ax.annotate("", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#2f6f9f", mutation_scale=12))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_xy_error_maps(
    *,
    position_world: np.ndarray,
    position_error_m: np.ndarray,
    orientation_error_deg: np.ndarray,
    output: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = np.asarray(position_world, dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    sc0 = axes[0].scatter(pos[:, 0], pos[:, 1], c=np.asarray(position_error_m) * 1000.0, s=5, cmap="magma")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("Position Error")
    fig.colorbar(sc0, ax=axes[0], label="mm")
    sc1 = axes[1].scatter(pos[:, 0], pos[:, 1], c=np.asarray(orientation_error_deg), s=5, cmap="viridis")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_title("Tool-Z Error")
    fig.colorbar(sc1, ax=axes[1], label="deg")
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _save_npz(
    *,
    output: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    csv_path: Path,
    metadata_path: Path | None,
    indices: np.ndarray,
    x_full: np.ndarray,
    xyz_norm: np.ndarray,
    xyz_true_norm: np.ndarray,
    tool_z: np.ndarray,
    errors: dict[str, np.ndarray],
    arc_threshold_m: float,
) -> None:
    position_world = _column_stack(arrays, ["ball_x_m", "ball_y_m", "ball_z_m"])
    quat = _column_stack(arrays, ["ball_qx", "ball_qy", "ball_qz", "ball_qw"])
    joint_pos = _column_stack(arrays, [f"joint_{i}_position_rad" for i in range(1, 8)])
    joint_vel = _column_stack(arrays, [f"joint_{i}_velocity_radps" for i in range(1, 8)])
    joint_effort = _column_stack(arrays, [f"joint_{i}_effort_nm" for i in range(1, 8)])
    joint_external_torque = _column_stack(arrays, [f"joint_{i}_external_torque_nm" for i in range(1, 8)])
    gt_pose_toolz = np.concatenate([errors["position_true_world"], errors["target_tool_z"]], axis=1).astype(np.float32)
    gt_pose_toolz_normalized = np.concatenate([xyz_true_norm, errors["target_tool_z"]], axis=1).astype(np.float32)
    x_world = np.concatenate([position_world, tool_z], axis=1).astype(np.float32)
    idx = np.asarray(indices, dtype=np.int64)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x_full[idx].astype(np.float32),
        source_x=x_full.astype(np.float32),
        source_index=idx,
        time=np.asarray(arrays["ros_time_s"], dtype=np.float64)[idx],
        unix_time_ns=np.asarray(arrays["unix_time_ns"], dtype=np.int64)[idx],
        monotonic_time_ns=np.asarray(arrays["monotonic_time_ns"], dtype=np.int64)[idx],
        position_world=position_world[idx].astype(np.float32),
        position_normalized=xyz_norm[idx].astype(np.float32),
        tool_z=tool_z[idx].astype(np.float32),
        ball_orientation_xyzw=quat[idx].astype(np.float32),
        position_true_world=errors["position_true_world"][idx].astype(np.float32),
        position_true_normalized=xyz_true_norm[idx].astype(np.float32),
        surface_normal=errors["surface_normal"][idx].astype(np.float32),
        target_tool_z=errors["target_tool_z"][idx].astype(np.float32),
        position_error_m=errors["position_error_m"][idx].astype(np.float32),
        signed_position_error_m=errors["signed_position_error_m"][idx].astype(np.float32),
        orientation_error_deg=errors["orientation_error_deg"][idx].astype(np.float32),
        x_world=x_world[idx].astype(np.float32),
        gt_pose_toolz=gt_pose_toolz[idx].astype(np.float32),
        gt_pose_toolz_normalized=gt_pose_toolz_normalized[idx].astype(np.float32),
        joint_position_rad=joint_pos[idx].astype(np.float32),
        joint_velocity_radps=joint_vel[idx].astype(np.float32),
        joint_effort_nm=joint_effort[idx].astype(np.float32),
        joint_external_torque_nm=joint_external_torque[idx].astype(np.float32),
        trajectory_offsets=np.asarray([0, len(idx)], dtype=np.int64),
        source_trajectory_offsets=np.asarray([0, len(x_full)], dtype=np.int64),
        arc_length_threshold_m=np.asarray([arc_threshold_m], dtype=np.float32),
        source_csv=np.asarray(str(csv_path)),
        metadata_json=np.asarray(str(metadata_path) if metadata_path is not None else ""),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        representation=np.asarray("[surface-model normalized xyz, recorded tool_z_x, recorded tool_z_y, recorded tool_z_z]"),
    )


def build_passive_demo_dataset(
    *,
    csv_path: Path,
    metadata_path: Path | None,
    dataset: str,
    output_full: Path,
    output_arc: Path,
    summary_json: Path,
    figure_prefix: Path,
    arc_mm: float,
) -> dict[str, Any]:
    arrays = _csv_arrays(csv_path)
    metadata = _load_metadata(metadata_path)
    position_world = _column_stack(arrays, ["ball_x_m", "ball_y_m", "ball_z_m"])
    tool_z = _normalize(_column_stack(arrays, ["tool_z_x", "tool_z_y", "tool_z_z"])).astype(np.float32)
    xyz_norm = real_iiwa_world_to_normalized_xyz(dataset, position_world)
    errors = _surface_errors(dataset, position_world, tool_z)
    xyz_true_norm = real_iiwa_world_to_normalized_xyz(dataset, errors["position_true_world"])
    x_full = np.concatenate([xyz_norm.astype(np.float32), tool_z.astype(np.float32)], axis=1)

    threshold_m = float(arc_mm) / 1000.0
    full_idx = np.arange(len(position_world), dtype=np.int64)
    arc_idx = _arc_length_indices(position_world, threshold_m)

    _save_npz(
        output=output_full,
        arrays=arrays,
        metadata=metadata,
        csv_path=csv_path,
        metadata_path=metadata_path,
        indices=full_idx,
        x_full=x_full,
        xyz_norm=xyz_norm,
        xyz_true_norm=xyz_true_norm,
        tool_z=tool_z,
        errors=errors,
        arc_threshold_m=0.0,
    )
    _save_npz(
        output=output_arc,
        arrays=arrays,
        metadata=metadata,
        csv_path=csv_path,
        metadata_path=metadata_path,
        indices=arc_idx,
        x_full=x_full,
        xyz_norm=xyz_norm,
        xyz_true_norm=xyz_true_norm,
        tool_z=tool_z,
        errors=errors,
        arc_threshold_m=threshold_m,
    )

    step_full = np.linalg.norm(np.diff(position_world.astype(np.float64), axis=0), axis=1)
    step_arc = np.linalg.norm(np.diff(position_world[arc_idx].astype(np.float64), axis=0), axis=1)
    duration_s = float(np.asarray(arrays["ros_time_s"], dtype=np.float64)[-1] - np.asarray(arrays["ros_time_s"], dtype=np.float64)[0])

    hist_full = figure_prefix.with_name(f"{figure_prefix.name}_full_error_hist.png")
    hist_arc = figure_prefix.with_name(f"{figure_prefix.name}_arc{arc_mm:g}mm_error_hist.png")
    xy_arc = figure_prefix.with_name(f"{figure_prefix.name}_arc{arc_mm:g}mm_xy_trajectory.png")
    xy_error = figure_prefix.with_name(f"{figure_prefix.name}_arc{arc_mm:g}mm_xy_error_map.png")
    _plot_error_hist(
        position_error_m=errors["position_error_m"],
        signed_position_error_m=errors["signed_position_error_m"],
        orientation_error_deg=errors["orientation_error_deg"],
        output=hist_full,
        title=f"{csv_path.stem}: full samples",
    )
    _plot_error_hist(
        position_error_m=errors["position_error_m"][arc_idx],
        signed_position_error_m=errors["signed_position_error_m"][arc_idx],
        orientation_error_deg=errors["orientation_error_deg"][arc_idx],
        output=hist_arc,
        title=f"{csv_path.stem}: arc {arc_mm:g} mm",
    )
    _plot_xy_trajectory(position_world[arc_idx], xy_arc, f"{csv_path.stem}: arc {arc_mm:g} mm XY trajectory")
    _plot_xy_error_maps(
        position_world=position_world[arc_idx],
        position_error_m=errors["position_error_m"][arc_idx],
        orientation_error_deg=errors["orientation_error_deg"][arc_idx],
        output=xy_error,
        title=f"{csv_path.stem}: arc {arc_mm:g} mm errors",
    )

    summary = {
        "input_csv": str(csv_path),
        "metadata_json": str(metadata_path) if metadata_path is not None else None,
        "dataset": dataset,
        "full_output": str(output_full),
        "arc_output": str(output_arc),
        "representation": "[surface-model normalized xyz, recorded tool_z_x, recorded tool_z_y, recorded tool_z_z]",
        "source_points": int(len(position_world)),
        "arc_points": int(len(arc_idx)),
        "arc_length_threshold_mm": float(arc_mm),
        "duration_s": duration_s,
        "path_length_m": float(np.sum(step_full)),
        "full_step_mm": _stats(step_full, scale=1000.0),
        "arc_step_mm": _stats(step_arc, scale=1000.0),
        "position_bounds_m": {
            "min": [float(v) for v in np.min(position_world, axis=0)],
            "max": [float(v) for v in np.max(position_world, axis=0)],
        },
        "full_position_abs_error_mm": _stats(errors["position_error_m"], scale=1000.0),
        "full_position_signed_error_mm": _stats(errors["signed_position_error_m"], scale=1000.0),
        "full_orientation_error_deg": _stats(errors["orientation_error_deg"]),
        "arc_position_abs_error_mm": _stats(errors["position_error_m"][arc_idx], scale=1000.0),
        "arc_position_signed_error_mm": _stats(errors["signed_position_error_m"][arc_idx], scale=1000.0),
        "arc_orientation_error_deg": _stats(errors["orientation_error_deg"][arc_idx]),
        "figures": {
            "full_error_hist": str(hist_full),
            "arc_error_hist": str(hist_arc),
            "arc_xy_trajectory": str(xy_arc),
            "arc_xy_error_map": str(xy_error),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-iiwa passive human-demo tool-z dataset from CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset", default="RealKuka")
    parser.add_argument("--task-id", default="RealKuka")
    parser.add_argument("--demo-name", default=None)
    parser.add_argument("--arc-mm", type=float, default=3.0)
    args = parser.parse_args()

    demo_name = str(args.demo_name or args.csv.stem)
    task_root = Path("datasets/real_iiwa/tasks") / str(args.task_id)
    train_root = task_root / "processed" / "train_sets"
    figure_root = task_root / "figures"
    arc_tag = f"arc{float(args.arc_mm):g}mm"
    summary = build_passive_demo_dataset(
        csv_path=args.csv,
        metadata_path=args.metadata,
        dataset=str(args.dataset),
        output_full=train_root / f"{demo_name}_toolz_full_v001.npz",
        output_arc=train_root / f"{demo_name}_toolz_{arc_tag}_v001.npz",
        summary_json=figure_root / f"{demo_name}_toolz_summary.json",
        figure_prefix=figure_root / demo_name,
        arc_mm=float(args.arc_mm),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
