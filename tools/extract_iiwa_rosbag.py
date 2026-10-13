#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ros.iiwa_bag import IiwaBagTopics, export_iiwa_csv, export_iiwa_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract KUKA iiwa ROS1 bag files into learning-ready datasets."
    )
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/iiwa_bags"))
    parser.add_argument(
        "--format",
        choices=("npz", "csv", "both"),
        default="npz",
        help="Output format. npz is the learning-ready default; csv keeps per-topic tables.",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "pose-topic", "tf-pose", "joint-states"),
        default="all",
        help="CSV export mode. Ignored for --format npz.",
    )
    parser.add_argument("--pose-topic", default="/iiwa/ee_info/Pose")
    parser.add_argument("--ee-vel-topic", default="/iiwa/ee_info/Vel")
    parser.add_argument("--joint-topic", default="/iiwa/joint_states")
    parser.add_argument("--torque-command-topic", default="/iiwa/TorqueController/command")
    parser.add_argument("--command-status-topic", default="/iiwa/commanding_status")
    parser.add_argument("--measure-topic", default="/iiwa/measure")
    parser.add_argument("--plotvar-topic", default="/iiwa/plotvar")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-static-topic", default="/tf_static")
    parser.add_argument(
        "--tcp-offset",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Fixed TCP offset from iiwa_link_ee, expressed in the EE frame, in meters.",
    )
    return parser.parse_args()


def output_dir_for_bag(root: Path, bag: Path, total_bags: int) -> Path:
    if total_bags == 1:
        return root
    return root / bag.stem


def main() -> None:
    args = parse_args()
    topics = IiwaBagTopics(
        pose=args.pose_topic,
        ee_vel=args.ee_vel_topic,
        joint_states=args.joint_topic,
        torque_command=args.torque_command_topic,
        command_status=args.command_status_topic,
        measure=args.measure_topic,
        plotvar=args.plotvar_topic,
        tf=args.tf_topic,
        tf_static=args.tf_static_topic,
    )
    for bag in args.bags:
        out_dir = output_dir_for_bag(args.output_dir, bag, len(args.bags))
        exported = []
        if args.format in ("npz", "both"):
            exported.extend(export_iiwa_npz(bag, out_dir, topics=topics, tcp_offset_ee=args.tcp_offset))
        if args.format in ("csv", "both"):
            exported.extend(export_iiwa_csv(bag, out_dir, mode=args.mode, topics=topics))
        for item in exported:
            print(f"Wrote {item.rows} rows: {item.path}")


if __name__ == "__main__":
    main()
