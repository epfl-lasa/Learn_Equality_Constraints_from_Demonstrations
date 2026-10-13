# Publication migration

Source: `LearnEqConstraints` at `c740a3e5d04a766509b486b619e35a55160b5c9e`.
Base: LASA first-submission `main` at `b0456e22dd4f583912a9cefacbeff1fee7431e3f`.
Working branch: `paper-revision`. Publication is prepared as a fast-forward update
to the LASA `main` branch; the first-submission history is retained.

## Included

- Current algorithm implementations, model architectures, normalization, projection,
  evaluation, dataset generators and per-method/per-dataset configurations.
- The ten synthetic tasks in the current `PAPER_DATASETS` list, plus `RealKuka`.
- Additional complex-geometry generators and configurations, including faceted
  sphere, genus-3, spiky sphere, knot/spline and warped-4D variants.
- The RealKuka human-demonstration dataset, task metadata, surface calibration
  and its offline preparation utility. Other real-robot datasets are excluded.
- 6D SurfNormal waypoint/obstacle simulation and 12D dual-arm planning/simulation,
  with their UR5 assets and original controller parameters.
- ONCL, DataAug, VAE and ECoMaNN; the vendored ECoMaNN license is retained.

## Excluded

Hardware drivers and the nested iiwa_ros repository, robot deployment/export tools,
raw ROS bag recordings, camera/video processing, historical outputs, videos, generated caches, legacy
prototypes, robustness presets, tuning sweeps, standalone diagnostic scripts,
and personal remote-Docker configuration.

The existing history of the LASA repository is preserved. Old outputs are deleted
from the publication snapshot, not erased from earlier LASA commits.

## Publication-specific adjustments

- Repaired the two SurfNormal simulator imports after the planning modules were renamed.
- Updated default checkpoint paths and planning-comparison discovery to the current
  dataset/seed directory layout; added an explicit comparison checkpoint seed.
- Added `configs/paper_benchmark.json` and `runners/run_paper_benchmark.py`.
- Replaced outdated README commands and the old main dataset table.
- Made optional raw preprocessing inputs explicit CLI arguments instead of personal paths.
- Added RealKuka plot labels and ignored generated figures/caches.
- Removed trailing blank lines in four data helper files; no numerical changes.

The training code, algorithm parameters and dataset arrays are otherwise copied
from the development snapshot. Acquisition paths in data metadata are retained as
provenance; runtime training loads repository-relative data.

## Validation protocol

The initial ONCL validation trained both checkouts independently with seed 0, full current epoch
counts and the same configured Python environment. No training or evaluation
hyperparameters are reduced for validation. Model tensors and evaluation metrics
are compared directly; wall-clock training time is excluded.

Environment: Python 3.10.18, PyTorch 2.10.0+cu128, NVIDIA GeForce RTX 3070.
The validation subprocesses set OMP/MKL/OpenBLAS threads to 1, use Matplotlib Agg,
disable W&B, and set PYTHONHASHSEED=0. The original checkout was unchanged during
migration validation; subsequent requested data cleanup applies to both checkouts.

Full logs, checkpoints, comparison scripts and generated simulation artifacts are
under ignored `outputs/migration_validation/`. Compact final results are recorded
in `VALIDATION_RESULTS.json`.

All 14 training comparisons pass: model tensors are identical (maximum absolute
difference 0), and every compared numerical evaluation metric is identical.
The default eleven-task benchmark also passes its resume and plot entrypoints.

| Dataset | Full epochs | Seed | Comparison |
| --- | ---: | ---: | --- |
| `12d_dual_arm_traj` | 4000 | 0 | Identical |
| `2d_planar_arm_line_n2` | 2000 | 0 | Identical |
| `3d_faceted_sphere_surface_traj` | 2600 | 0 | Identical |
| `3d_genus3_tube_surface_traj` | 3000 | 0 | Identical |
| `3d_planar_arm_line_n3_traj` | 2000 | 0 | Identical |
| `3d_spatial_arm_ellip_n3_traj` | 1200 | 0 | Identical |
| `3d_spiky_sphere_surface_traj` | 2600 | 0 | Identical |
| `3d_twosphere_traj` | 2000 | 0 | Identical |
| `3d_vz_2d_ellipse_traj` | 2000 | 0 | Identical |
| `6d_spatial_arm_up_n6_py_traj` | 400 | 0 | Identical |
| `6d_surf_normal_traj` | 2200 | 0 | Identical |
| `RealKuka` | 400 | 0 | Identical |
| `linked_curves_3d` | 2000 | 0 | Identical |
| `multi_handle_4d` | 3200 | 0 | Identical |

All 16 originally registered real-data variants loaded successfully before cleanup.
An intermediate cleanup verified eight registered variants and both Oracle files.
The final publication-only cleanup leaves RealKuka as the sole real-robot dataset;
its data loading, normalization, surface evaluation and reference-grid generation
have been checked against the development checkout. The publication benchmark
CLI and the RealKuka benchmark plot command both complete successfully.

SurfNormal waypoint and obstacle execution complete in headless PyBullet. The
waypoint numerical summary matches the source implementation exactly. The source
required old import-name aliases injected in memory; its files were not edited.
Dual-arm planning and execution also complete, and the default path_0 numerical
summary matches the source exactly.

## Decisions to confirm before publication

1. **Final dataset list.** The current plotter defines ten synthetic tasks, while
   the old DATASETS.md listed different environments. The new default uses the
   current ten plus RealKuka. Other geometry configurations are retained as optional
   experiments; confirm which should appear in the final paper's primary command.
2. **Raw recordings (resolved).** At the user's request, the three ROS bags
   (86.47 MiB) are excluded from this checkout and `*.bag` is ignored by Git.
   The separate curved-surface bag remains in the development project; the bags
   belonging to the two removed plane tasks were subsequently removed there too.
   Processed datasets,
   calibration models and metadata remain available for training and evaluation.
3. **Real-data cleanup (resolved).** Both checkouts exclude the two early plane
   tasks and six Oracle subsets. The subsequent publication-only cleanup removes
   all other real-robot datasets, registrations, configurations and dedicated
   processing tools from this checkout. Only RealKuka's 1,283-point human dataset
   and its calibration/evaluation dependencies remain here. The development
   checkout still retains Oracle full/3 mm and the six curved-surface variants.
4. **Dual-arm example quality.** Default path_0 has a large execution error in both
   versions (mean true-constraint position error about 0.183 m, orientation 19.19 deg).
   Path_1 and path_2 execute with about 5.3/4.4 mm position and 0.98/0.78 deg
   orientation constraint error. These are migration-validation cases only;
   README documents the generic planning/execution workflow without selecting
   a validation trajectory. Confirm the paper's actual experimental settings
   before publishing figures.
5. **Embedded optional transforms.** Inactive noise/missing-segment/subsampling
   helpers remain inside the shared data loader to preserve numerical equivalence.
   Dedicated robustness configurations, launchers and plotting tools are excluded.
6. **Other methods.** DataAug and VAE have additionally passed full seed-0
   training comparisons on the six newly added publication tasks listed below.
   ECoMaNN and a complete multi-seed baseline benchmark have not been retrained.
7. **Raw preprocessing completeness.** The source contains the processed training
   sets but not every original external CSV/metadata input referenced by provenance.
   Training and evaluation are self-contained; rebuilding every dataset from the
   original acquisition stage would require those external inputs.

## DataAug and VAE migration validation

At the user's request, additional baseline validation is restricted to publication
benchmark/geometry tasks absent from the LASA first-submission snapshot. SurfNormal
is excluded from this subset because it already existed under its old name;
dual-arm and other original environments are also excluded. Already completed
DataAug comparisons for linked-curves and multi-handle were reused.

Both checkouts independently train seed 0 with their full current configurations.
All twelve comparisons pass: training arrays and model tensors are identical,
and every compared numerical configuration/evaluation value is identical. Timing
and output paths are excluded. No algorithm or parameter changes were necessary.

| Dataset | DataAug full epochs | VAE full epochs | Comparison |
| --- | ---: | ---: | --- |
| `linked_curves_3d` | 1500 | 2400 | Identical |
| `multi_handle_4d` | 1500 | 3000 | Identical |
| `RealKuka` | 2000 | 800 | Identical |
| `3d_faceted_sphere_surface_traj` | 2000 | 800 | Identical |
| `3d_genus3_tube_surface_traj` | 2000 | 800 | Identical |
| `3d_spiky_sphere_surface_traj` | 2000 | 800 | Identical |

Compact results are in `VALIDATION_RESULTS.json`; full logs, scripts and models
are under ignored `outputs/migration_validation/baselines/`. The development
checkout was not modified by this validation. The previous broad validation was
stopped when the user narrowed the scope; its completed results remain archived
in `comparison_before_scope_change.json` under that ignored directory.

## Publication packaging

README now documents installation, single-run and grouped benchmarks, configuration
overrides, automatic evaluation, result paths, plots, and the two simulation workflows.
RA-L 2026 citation metadata is included in `CITATION.cff`. `constraints-tested.txt`
records the tested direct dependency versions. The root dependency list no longer
installs the vendored legacy setup.py with obsolete TensorFlow/SciPy pins; the
ECoMaNN adapter loads the bundled source directly.

Nine CLI help checks, source compilation, RealKuka loading, and the bundled
ECoMaNN import/forward check pass. Pip dependency resolution passes with the
tested constraints in the existing environment; a completely new environment
installation was not performed. This packaging does not change training
algorithms or task parameters. Raw bags, videos, caches, hardware deployment and
old experiment outputs are excluded from the current snapshot.
