from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from datasets.ros.bag_utils import (
    POSE_HEADER,
    import_rosbag,
    iter_joint_state_rows,
    iter_pose_topic_rows,
    iter_tf_chain_pose_rows,
    write_csv,
)
from datasets.ros.transforms import quat_to_matrix_xyzw


IIWA_TF_CHAIN = (
    ("world", "iiwa_link_0"),
    ("iiwa_link_0", "iiwa_link_1"),
    ("iiwa_link_1", "iiwa_link_2"),
    ("iiwa_link_2", "iiwa_link_3"),
    ("iiwa_link_3", "iiwa_link_4"),
    ("iiwa_link_4", "iiwa_link_5"),
    ("iiwa_link_5", "iiwa_link_6"),
    ("iiwa_link_6", "iiwa_link_7"),
    ("iiwa_link_7", "iiwa_link_ee"),
)


@dataclass(frozen=True)
class IiwaBagTopics:
    pose: str = "/iiwa/ee_info/Pose"
    ee_vel: str = "/iiwa/ee_info/Vel"
    joint_states: str = "/iiwa/joint_states"
    torque_command: str = "/iiwa/TorqueController/command"
    command_status: str = "/iiwa/commanding_status"
    measure: str = "/iiwa/measure"
    plotvar: str = "/iiwa/plotvar"
    tf: str = "/tf"
    tf_static: str = "/tf_static"


@dataclass(frozen=True)
class ExportedFile:
    name: str
    path: str
    rows: int


def export_pose_topic(bag_path: Path, output_dir: Path, topic: str) -> ExportedFile:
    out = output_dir / "iiwa_ee_pose_topic.csv"
    rows = write_csv(out, POSE_HEADER, iter_pose_topic_rows(bag_path, topic))
    return ExportedFile("pose_topic", str(out), rows)


def export_tf_pose(
    bag_path: Path,
    output_dir: Path,
    *,
    tf_topic: str,
    tf_static_topic: str,
    chain: tuple[tuple[str, str], ...] = IIWA_TF_CHAIN,
) -> ExportedFile:
    out = output_dir / "iiwa_ee_pose_from_tf.csv"
    rows = write_csv(out, POSE_HEADER, iter_tf_chain_pose_rows(bag_path, tf_topic, tf_static_topic, chain))
    return ExportedFile("tf_pose", str(out), rows)


def export_joint_states(bag_path: Path, output_dir: Path, topic: str) -> ExportedFile:
    out = output_dir / "iiwa_joint_states.csv"
    rows = iter_joint_state_rows(bag_path, topic)
    first = next(rows, None)
    if first is None:
        raise RuntimeError(f"No messages found on {topic}")
    marker, header = first
    if marker != "__header__":
        raise RuntimeError("Internal error while reading joint state header")
    count = write_csv(out, header, rows)
    return ExportedFile("joint_states", str(out), count)


def _canonicalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    if q.size == 0:
        return q.reshape(0, 4)
    out = q.astype(np.float64, copy=True)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / (norm + 1e-15)
    out[out[:, 3] < 0.0] *= -1.0
    return out


def _matrix(rows: list[list[float]], width: int | None = None) -> np.ndarray:
    if not rows:
        return np.empty((0, int(width or 0)), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _vector_or_nan(values, width: int) -> list[float]:
    vals = [float(v) for v in values]
    if len(vals) == width:
        return vals
    out = [np.nan] * width
    out[: min(width, len(vals))] = vals[:width]
    return out


def _padded_matrix(rows: list[list[float]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    width = max(len(row) for row in rows)
    out = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        if row:
            out[i, : len(row)] = row
    return out


def _nearest_rows(source_t: np.ndarray, source_rows: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    if source_t.size == 0 or query_t.size == 0:
        return np.empty((query_t.shape[0], source_rows.shape[1] if source_rows.ndim == 2 else 0), dtype=np.float64)
    idx = np.searchsorted(source_t, query_t)
    idx0 = np.clip(idx - 1, 0, source_t.shape[0] - 1)
    idx1 = np.clip(idx, 0, source_t.shape[0] - 1)
    choose_prev = np.abs(query_t - source_t[idx0]) <= np.abs(source_t[idx1] - query_t)
    nearest = np.where(choose_prev, idx0, idx1)
    return source_rows[nearest]


def _apply_tcp_offset(
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    tcp_offset_ee: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if ee_pos.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 4), dtype=np.float64)
    tcp_pos = np.empty_like(ee_pos, dtype=np.float64)
    for i, quat in enumerate(ee_quat):
        tcp_pos[i] = ee_pos[i] + quat_to_matrix_xyzw(quat) @ tcp_offset_ee
    return tcp_pos, ee_quat.astype(np.float64, copy=True)


def _apply_tcp_velocity_offset(
    ee_linear_vel: np.ndarray,
    ee_angular_vel: np.ndarray,
    ee_quat_for_vel: np.ndarray,
    tcp_offset_ee: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if ee_linear_vel.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    tcp_linear_vel = np.empty_like(ee_linear_vel, dtype=np.float64)
    for i, quat in enumerate(ee_quat_for_vel):
        r_world = quat_to_matrix_xyzw(quat) @ tcp_offset_ee
        tcp_linear_vel[i] = ee_linear_vel[i] + np.cross(ee_angular_vel[i], r_world)
    return tcp_linear_vel, ee_angular_vel.astype(np.float64, copy=True)


def _write_dataset_metadata(
    output_dir: Path,
    *,
    bag_path: Path,
    topics: IiwaBagTopics,
    tcp_offset_ee: np.ndarray,
    files: list[ExportedFile],
    arrays: dict[str, tuple[int, ...]],
) -> ExportedFile:
    metadata = {
        "source_bag": str(bag_path),
        "format": "npz",
        "robot": "kuka_iiwa",
        "topics": asdict(topics),
        "tcp": {
            "frame": "iiwa_link_ee",
            "offset_xyz_m": [float(v) for v in tcp_offset_ee],
            "orientation": "same_as_iiwa_link_ee",
            "velocity_assumption": "ee twist linear/angular components are expressed in the world/base frame; tcp_linear_vel = ee_linear_vel + cross(ee_angular_vel, R_ee * offset_xyz_m).",
        },
        "skipped_topics": {
            "additional_outputs": "Skipped because it uses the custom iiwa_driver/AdditionalOutputs message.",
        },
        "tf_chain": [list(edge) for edge in IIWA_TF_CHAIN],
        "arrays": {name: list(shape) for name, shape in sorted(arrays.items())},
        "files": [asdict(item) for item in files],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return ExportedFile("metadata", str(metadata_path), 1)


def collect_iiwa_dataset_arrays(
    bag_path: Path,
    topics: IiwaBagTopics | None = None,
    *,
    tcp_offset_ee: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    topics = topics or IiwaBagTopics()
    tcp_offset = np.asarray(tcp_offset_ee, dtype=np.float64).reshape(3)
    rosbag = import_rosbag()

    joint_names: list[str] | None = None
    joint_t, joint_q, joint_dq, joint_effort = [], [], [], []
    pose_t, ee_pos, ee_quat = [], [], []
    ee_vel_t, ee_linear, ee_angular = [], [], []
    torque_t, torque_command = [], []
    status_t, command_status = [], []
    measure_t, measure_linear, measure_angular = [], [], []
    plotvar_t, plotvar = [], []

    standard_topics = [
        topics.pose,
        topics.ee_vel,
        topics.joint_states,
        topics.torque_command,
        topics.command_status,
        topics.measure,
        topics.plotvar,
    ]
    with rosbag.Bag(str(bag_path)) as bag:
        for topic, msg, t in bag.read_messages(topics=standard_topics):
            stamp = float(t.to_sec())
            if topic == topics.joint_states:
                if joint_names is None:
                    joint_names = list(msg.name)
                width = len(joint_names)
                joint_t.append(stamp)
                joint_q.append(_vector_or_nan(msg.position, width))
                joint_dq.append(_vector_or_nan(msg.velocity, width))
                joint_effort.append(_vector_or_nan(msg.effort, width))
            elif topic == topics.pose:
                pose_t.append(stamp)
                ee_pos.append([msg.position.x, msg.position.y, msg.position.z])
                ee_quat.append([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
            elif topic == topics.ee_vel:
                ee_vel_t.append(stamp)
                ee_linear.append([msg.linear.x, msg.linear.y, msg.linear.z])
                ee_angular.append([msg.angular.x, msg.angular.y, msg.angular.z])
            elif topic == topics.torque_command:
                torque_t.append(stamp)
                torque_command.append([float(v) for v in msg.data])
            elif topic == topics.command_status:
                status_t.append(stamp)
                command_status.append(bool(msg.data))
            elif topic == topics.measure:
                measure_t.append(stamp)
                measure_linear.append([msg.linear.x, msg.linear.y, msg.linear.z])
                measure_angular.append([msg.angular.x, msg.angular.y, msg.angular.z])
            elif topic == topics.plotvar:
                plotvar_t.append(stamp)
                plotvar.append([float(v) for v in msg.data])

    tf_pose = np.asarray(list(iter_tf_chain_pose_rows(bag_path, topics.tf, topics.tf_static, IIWA_TF_CHAIN)), dtype=np.float64)
    if tf_pose.size == 0:
        tf_pose = np.empty((0, 8), dtype=np.float64)

    pose_time = np.asarray(pose_t, dtype=np.float64)
    pose_pos = _matrix(ee_pos, 3)
    pose_quat = _canonicalize_quat_xyzw(_matrix(ee_quat, 4))
    tcp_pos, tcp_quat = _apply_tcp_offset(pose_pos, pose_quat, tcp_offset)

    ee_vel_time = np.asarray(ee_vel_t, dtype=np.float64)
    ee_linear_arr = _matrix(ee_linear, 3)
    ee_angular_arr = _matrix(ee_angular, 3)
    quat_for_vel = _nearest_rows(pose_time, pose_quat, ee_vel_time)
    tcp_linear_vel, tcp_angular_vel = _apply_tcp_velocity_offset(
        ee_linear_arr,
        ee_angular_arr,
        quat_for_vel,
        tcp_offset,
    )

    tf_ee_pos = tf_pose[:, 1:4].astype(np.float64, copy=False)
    tf_ee_quat = _canonicalize_quat_xyzw(tf_pose[:, 4:8])
    tf_tcp_pos, tf_tcp_quat = _apply_tcp_offset(tf_ee_pos, tf_ee_quat, tcp_offset)

    arrays: dict[str, np.ndarray] = {
        "tcp_offset_ee": tcp_offset.astype(np.float64, copy=True),
        "joint_names": np.asarray(joint_names or [], dtype=str),
        "t_joint": np.asarray(joint_t, dtype=np.float64),
        "q": _matrix(joint_q, len(joint_names or [])),
        "dq": _matrix(joint_dq, len(joint_names or [])),
        "effort": _matrix(joint_effort, len(joint_names or [])),
        "t_pose": pose_time,
        "ee_pos": pose_pos,
        "ee_quat": pose_quat,
        "tcp_pos": tcp_pos,
        "tcp_quat": tcp_quat,
        "t_ee_vel": ee_vel_time,
        "ee_linear_vel": ee_linear_arr,
        "ee_angular_vel": ee_angular_arr,
        "tcp_linear_vel": tcp_linear_vel,
        "tcp_angular_vel": tcp_angular_vel,
        "t_tf_pose": tf_pose[:, 0].astype(np.float64, copy=False),
        "tf_ee_pos": tf_ee_pos,
        "tf_ee_quat": tf_ee_quat,
        "tf_tcp_pos": tf_tcp_pos,
        "tf_tcp_quat": tf_tcp_quat,
        "t_torque_command": np.asarray(torque_t, dtype=np.float64),
        "torque_command": _padded_matrix(torque_command),
        "t_command_status": np.asarray(status_t, dtype=np.float64),
        "command_status": np.asarray(command_status, dtype=np.bool_),
        "t_measure": np.asarray(measure_t, dtype=np.float64),
        "measure_linear": _matrix(measure_linear, 3),
        "measure_angular": _matrix(measure_angular, 3),
        "t_plotvar": np.asarray(plotvar_t, dtype=np.float64),
        "plotvar": _padded_matrix(plotvar),
    }
    return arrays


def export_iiwa_npz(
    bag_path: Path,
    output_dir: Path,
    *,
    topics: IiwaBagTopics | None = None,
    tcp_offset_ee: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
) -> list[ExportedFile]:
    topics = topics or IiwaBagTopics()
    tcp_offset = np.asarray(tcp_offset_ee, dtype=np.float64).reshape(3)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = collect_iiwa_dataset_arrays(bag_path, topics, tcp_offset_ee=tcp_offset)
    out = output_dir / "iiwa_demonstration.npz"
    np.savez_compressed(out, **arrays)
    files = [ExportedFile("dataset", str(out), int(arrays["t_joint"].shape[0]))]
    files.append(
        _write_dataset_metadata(
            output_dir,
            bag_path=bag_path,
            topics=topics,
            tcp_offset_ee=tcp_offset,
            files=files,
            arrays={name: value.shape for name, value in arrays.items()},
        )
    )
    return files


def load_iiwa_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def export_iiwa_csv(
    bag_path: Path,
    output_dir: Path,
    *,
    mode: str = "all",
    topics: IiwaBagTopics | None = None,
) -> list[ExportedFile]:
    topics = topics or IiwaBagTopics()
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    if mode in ("all", "pose-topic"):
        exported.append(export_pose_topic(bag_path, output_dir, topics.pose))
    if mode in ("all", "tf-pose"):
        exported.append(
            export_tf_pose(
                bag_path,
                output_dir,
                tf_topic=topics.tf,
                tf_static_topic=topics.tf_static,
            )
        )
    if mode in ("all", "joint-states"):
        exported.append(export_joint_states(bag_path, output_dir, topics.joint_states))

    metadata = {
        "source_bag": str(bag_path),
        "mode": mode,
        "robot": "kuka_iiwa",
        "topics": asdict(topics),
        "tf_chain": [list(edge) for edge in IIWA_TF_CHAIN],
        "files": [asdict(item) for item in exported],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    exported.append(ExportedFile("metadata", str(metadata_path), 1))
    return exported
