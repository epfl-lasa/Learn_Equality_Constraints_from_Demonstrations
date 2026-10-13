from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from datasets.ros.transforms import matrix_to_pose_row, ros_transform_to_matrix


POSE_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw"]


def import_rosbag():
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError(
            "Could not import ROS1 rosbag. Run with ROS Noetic's Python, for example "
            "`/usr/bin/python3 ...`, after sourcing ROS if needed."
        ) from exc
    return rosbag


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable[float]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def iter_pose_topic_rows(bag_path: Path, topic: str):
    rosbag = import_rosbag()
    with rosbag.Bag(str(bag_path)) as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            p = msg.position
            q = msg.orientation
            yield (t.to_sec(), p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def iter_joint_state_rows(bag_path: Path, topic: str):
    rosbag = import_rosbag()
    joint_names = None
    with rosbag.Bag(str(bag_path)) as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            if joint_names is None:
                joint_names = list(msg.name)
                yield ("__header__", ["t", *joint_names])
            yield (t.to_sec(), *[float(v) for v in msg.position])


def iter_tf_chain_pose_rows(
    bag_path: Path,
    tf_topic: str,
    tf_static_topic: str,
    chain: tuple[tuple[str, str], ...],
):
    rosbag = import_rosbag()
    static: dict[tuple[str, str], np.ndarray] = {}
    topics = [tf_static_topic, tf_topic]
    with rosbag.Bag(str(bag_path)) as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            transforms = {
                (ts.header.frame_id, ts.child_frame_id): ros_transform_to_matrix(ts.transform)
                for ts in msg.transforms
            }
            if topic == tf_static_topic:
                static.update(transforms)
                continue

            pose = np.eye(4, dtype=np.float64)
            missing = []
            for edge in chain:
                if edge in transforms:
                    edge_pose = transforms[edge]
                elif edge in static:
                    edge_pose = static[edge]
                else:
                    missing.append(edge)
                    continue
                pose = pose @ edge_pose
            if missing:
                names = ", ".join(f"{a}->{b}" for a, b in missing)
                raise RuntimeError(f"Missing TF edges in {tf_topic}: {names}")

            msg_stamp = msg.transforms[0].header.stamp.to_sec() if msg.transforms else t.to_sec()
            yield matrix_to_pose_row(msg_stamp, pose)
