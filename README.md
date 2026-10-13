# ONCL: Learning Equality Constraints from Trajectory Demonstrations

Baiyu Peng, Albert Árboles, and Aude Billard.

**IEEE Robotics and Automation Letters (RA-L), 2026**
Supported by the euROBIN Grant.

ONCL (Orthonormal Neural Constraint Learning) learns differentiable equality
constraints from feasible demonstrations. This repository contains ONCL, DataAug,
VAE and ECoMaNN, synthetic and recorded robot datasets, training/evaluation,
and 6D SurfNormal / 12D dual-arm planning and simulation.
It does not contain hardware deployment, robot drivers, or video postprocessing.

## Installation

Python **3.10** is recommended. Validation used Python 3.10.18, PyTorch 2.10.0
with CUDA 12.8, and an NVIDIA RTX 3070. On Linux:

```bash
git clone https://github.com/epfl-lasa/Learn_Equality_Constraints_from_Demonstrations.git
cd Learn_Equality_Constraints_from_Demonstrations
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c constraints-tested.txt
```

Run all experiment commands from the repository root. The constraints file records
the direct package versions used for validation; omit `-c constraints-tested.txt`
to use your own compatible versions. Training selects CUDA when available;
append `--override device=cpu` for CPU execution. W&B logging is disabled by default
and is enabled only with `--wandb-enable`.
PyBullet is needed for the simulations; optional video export also uses FFmpeg.
No ROS installation or robot connection is needed to train or evaluate RealKuka.
On servers without a display, set `export MPLBACKEND=Agg` before running experiments.

The modified ECoMaNN dependency is included under `smp_manifold_learning-master/`
and loaded directly by `methods/ecomann.py`. Do not separately install its legacy
`setup.py`, which lists obsolete TensorFlow/SciPy dependencies unrelated to this adapter.
Its license and attribution are retained. See that directory for the original
implementation by Sutanto et al., CoRL 2020 / PMLR 155 (2021).

## Quick start: train and evaluate one task

```bash
python -m runners.run_one --method oncl --dataset 2d_planar_arm_line_n2 --seed 0 --outdir outputs/example_planar_arm
```

This runs the full current configuration, evaluates the learned model, and saves
metrics, a checkpoint and plots. Substitute `dataaug`, `vae`, or `ecomann` to run
a baseline, using a separate output directory. `run_one` refuses to reuse a
nonempty output directory; choose a new directory or use `--retrain` to clear it.
Use `--help` on an entrypoint to inspect its options.

## Reproduce the benchmark

`configs/paper_benchmark.json` defines the ten current synthetic benchmark tasks,
the recorded `RealKuka` task, and three optional geometry examples. The additional
dataset generators and task-specific configurations are also retained.

| Group | Tasks |
| --- | --- |
| `main` | Ten synthetic tasks listed in DATASETS.md |
| `real` | RealKuka human demonstrations |
| `core` (default) | `main` + `real`, eleven tasks |
| `geometry` | Faceted sphere, genus-3 tube and spiky sphere |

```bash
python -m runners.run_paper_benchmark --methods oncl --seeds 0
```

This uses the current dataset/method parameters without reducing training epochs.
Use `--group main`, `--group real`, or `--group geometry` to select a subset.
Multiple seeds and methods are supported, for example:

```bash
python -m runners.run_paper_benchmark --methods oncl,dataaug,vae,ecomann --seeds 0,1,2 --outdir outputs/bench/comparison
python -m runners.run_one --method oncl --dataset RealKuka --seed 0 --outdir outputs/real_kuka
```

The example seeds `0,1,2` are selectable seeds, not a claim that they reproduce
the complete seed set used for every published figure.

Run a specific group or an arbitrary dataset subset:

```bash
python -m runners.run_paper_benchmark --group real --methods oncl,dataaug,vae --seeds 0 --outdir outputs/bench/real_kuka
python -m runners.run_paper_benchmark --group geometry --methods oncl,dataaug,vae --seeds 0 --outdir outputs/bench/geometry
python -m runners.run_benchmark --methods oncl,dataaug,vae --datasets linked_curves_3d,multi_handle_4d --seeds 0 --outdir outputs/bench/subset
```

To resume an existing benchmark output directory, append `--resume`; completed
method/dataset/seed runs are skipped. `--rewrite` reruns selected benchmark runs.
Configuration layers are `configs/base.json`, `configs/methods/<method>.json`,
`configs/datasets/<dataset>.json`, and `configs/method_dataset/<method>_<dataset>.json`.
An explicit CLI seed and `--override key=value` settings take precedence over
the JSON layers. Nested overrides such as `--override projector.steps=120`
are supported. RealKuka always loads the included recorded samples; changing
`n_train` does not generate new recordings.

For the default benchmark output, seed-0 models are saved as:

```text
outputs/bench/paper/oncl/<dataset>/<dataset>_oncl_seed0_model.pt
```

Each run also saves `<dataset>_<method>_seed<seed>_eval.json`. Benchmark-level
outputs are `summary_metrics.json`, `per_run_metrics.jsonl`, `per_case_metrics.csv`
and `leaderboard_mean_by_method.csv`. Evaluation runs automatically after training
and includes projection/manifold distances, ground-truth/learned-surface distances,
and constraint classification metrics where applicable. RealKuka also reports
position and tool-axis orientation errors against the calibrated surface.
This numerical evaluation is separate from downstream planning and simulation.

Pretrained experiment models and old results are not shipped. Train the relevant
dataset after cloning before running checkpoint-dependent planning or simulation.
All generated models, figures, caches and videos are ignored by Git.

Plots and numerical summaries can be generated from the benchmark outputs:

```bash
python -m benchmark.plot_paper_benchmarks --bench paper --methods oncl --no-fill-missing-zero
python -m benchmark.plot_paper_benchmarks --bench comparison --methods oncl,dataaug,vae,ecomann --no-fill-missing-zero
python -m benchmark.plot_paper_benchmarks --bench real_kuka --datasets RealKuka --methods oncl,dataaug,vae --no-fill-missing-zero
```

The plotter's default ten-task list covers synthetic experiments. RealKuka can
be selected explicitly with `--datasets RealKuka`. `--bench` accepts a name under
`outputs/bench/` or an absolute directory. Figures and analysis tables are saved
under `<benchmark directory>/analysis_bars_paper/`. After training multiple matched
seeds, add `--significance` or `--rank-comparison` for comparison tables.

## 6D SurfNormal simulation

After training `6d_surf_normal_traj` with the core benchmark, run the waypoint task.
It plans using the learned constraint and executes the plan with a UR5 in PyBullet:

```bash
python -m simulation.sim_sine_pose_waypoints --gui 0 --save-error-video 0 --ckpt outputs/bench/paper/oncl/6d_surf_normal_traj/6d_surf_normal_traj_oncl_seed0_model.pt --outdir outputs/simulation/surfnormal
```

The historical `sim_sine_pose_*` filenames are retained; they now import the
current SurfNormal planners. `--gui 0` runs headlessly, `--gui 1` renders
an offscreen video, and `--gui 2` opens an interactive window. Error-curve video
export is controlled separately by `--save-error-video`. Run the obstacle task with:

```bash
python -m simulation.sim_sine_pose_obsavoid --gui 0 --save-error-video 0 --ckpt outputs/bench/paper/oncl/6d_surf_normal_traj/6d_surf_normal_traj_oncl_seed0_model.pt --outdir outputs/simulation/surfnormal_obstacle
```

Outputs include planned/executed poses, tracking and constraint-error summaries,
and figures. To train only this task at the standard checkpoint location:

```bash
python -m runners.run_benchmark --methods oncl --datasets 6d_surf_normal_traj --seeds 0 --outdir outputs/bench/paper --resume
```

## 12D dual-arm simulation

First plan using the trained dual-arm model, then execute in PyBullet:

```bash
python -m runners.run_benchmark --methods oncl --datasets 12d_dual_arm_traj --seeds 0 --outdir outputs/bench/paper --resume
python -m planning.plan_dual_arm --ckpt outputs/bench/paper/oncl/12d_dual_arm_traj/12d_dual_arm_traj_oncl_seed0_model.pt --outdir outputs/simulation/dual_arm
python -m simulation.sim_dual_arm_planned_path --gui 0 --plan-outdir outputs/simulation/dual_arm
```

The included UR5 meshes are used by both environments. All generated results
stay under ignored `outputs/` directories.

The planner saves `dual_arm_learned_plans.npz` and planning metrics. The simulator
saves execution and constraint-error summaries. To compare planners on shared
start/goal pairs after training the corresponding methods:

```bash
python -m planning.compare_planning_methods --dataset 12d_dual_arm --bench outputs/bench/comparison --methods oncl,dataaug,vae,ecomann --checkpoint-seed 0 --fixed-pairs 3 --outdir outputs/planning_comparison/dual_arm
```

`--checkpoint-seed` selects the trained model; planner options such as `--seed`
are forwarded to child planning scripts. `--dry-run` prints the generated commands.

## Recorded robot data

`datasets/real_iiwa/` includes only the RealKuka processed train set, task metadata and
calibration/surface models. Raw ROS bag recordings are excluded and ignored by Git.
`RealKuka` loads the 1,283-sample human
demonstration set, with normalized position and recorded tool-z orientation.
The six-dimensional samples contain normalized TCP XYZ followed by the tool
local-z direction in the world frame. They were thinned at approximately 3 mm
spatial arc-length spacing and cropped to remove entry/exit segments. Training
reads `datasets/real_iiwa/tasks/RealKuka/processed/train_sets/human_demo_paper_toolz_arc3mm_cropped_v001.npz`
directly. Keep the task metadata and calibration checkpoint/grid with this NPZ;
they support normalization, reference-surface generation and evaluation.
Oracle demonstrations and other real-robot training datasets are excluded.
See `DATASETS.md` for scope and provenance. Dataset preparation scripts are
optional; raw-input paths must be supplied explicitly. Training does not use
robot network addresses or the original acquisition computer's directories.

## Changes from the first submission

The current snapshot adds complex geometry datasets, updates SurfNormal sampling,
uses current per-task hyperparameters, and supports shared start/goal pairs for
planning comparisons. Evaluation now defaults to near-manifold query samples
from the ground-truth manifold (`eval_chamfer_near_source=gt`), rather than from
training samples. Checkpoints and evaluation files include the seed in their names.
The first-submission history remains available in Git.

Robustness sweeps, temporary tuning/diagnostic entrypoints, historical results,
robot deployment, and camera/video processing are outside this release scope.
The shared data loader still contains inactive optional transform helpers to
preserve the original numerical pipeline; no robustness presets are shipped.

## Repository layout and validation

| Directory | Purpose |
| --- | --- |
| `methods/` | ONCL and baseline implementations |
| `datasets/` | Generators, RealKuka data and simulation assets |
| `configs/`, `runners/` | Parameters and experiment entrypoints |
| `experiments/`, `evaluation/` | Training and numerical evaluation |
| `planning/`, `simulation/` | Learned-constraint planning and PyBullet execution |
| `plotting/`, `benchmark/`, `tools/` | Figures and offline analysis |

Migration validation trained ONCL seed 0 on eleven core tasks plus three geometry
tasks, and DataAug/VAE seed 0 on six tasks added since the first submission.
Model weights and numerical results matched the development project exactly in
the tested environment. ECoMaNN has not undergone the same migration training
comparison. See [VALIDATION_RESULTS.json](VALIDATION_RESULTS.json) and
[MIGRATION_REPORT.md](MIGRATION_REPORT.md) for the exact scope. These checks are
not a complete multi-seed reproduction of all published results.

ECoMaNN attribution and its original MIT license are retained in
[smp_manifold_learning-master/LICENSE](smp_manifold_learning-master/LICENSE).
Texture attribution is in [datasets/assets/textures/README.md](datasets/assets/textures/README.md).

## Citation

```bibtex
@article{peng2026oncl,
  title   = {Learning Equality Constraints from Trajectory Demonstrations},
  author  = {Peng, Baiyu and {\'A}rboles, Albert and Billard, Aude},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```

Machine-readable metadata is provided in [CITATION.cff](CITATION.cff).
