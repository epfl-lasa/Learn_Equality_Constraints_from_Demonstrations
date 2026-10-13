"""Run the publication dataset list with the standard benchmark pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "configs/paper_benchmark.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["main", "real", "core", "geometry"], default="core")
    parser.add_argument("--methods", default="oncl")
    parser.add_argument("--seeds", default=str(manifest["seed"]))
    parser.add_argument("--outdir", default="outputs/bench/paper")
    args, extra = parser.parse_known_args()
    groups = {
        "main": manifest["main_datasets"],
        "real": manifest["real_datasets"],
        "core": manifest["main_datasets"] + manifest["real_datasets"],
        "geometry": manifest["additional_geometry_examples"],
    }
    command = [sys.executable, "-m", "runners.run_benchmark", "--methods", args.methods,
               "--datasets", ",".join(groups[args.group]), "--seeds", args.seeds,
               "--outdir", args.outdir, *extra]
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == "__main__":
    main()
