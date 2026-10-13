# Datasets

The core benchmark is defined in [configs/paper_benchmark.json](configs/paper_benchmark.json).

| Dataset ID | Dimension | Purpose |
|---|---:|---|
| `2d_planar_arm_line_n2` | 2 | Planar arm line constraint |
| `3d_planar_arm_line_n3_traj` | 3 | Planar arm demonstrations |
| `3d_spatial_arm_ellip_n3_traj` | 3 | Spatial arm ellipse |
| `3d_twosphere_traj` | 3 | Two-sphere surface |
| `3d_vz_2d_ellipse_traj` | 3 | Twisted ellipse |
| `linked_curves_3d` | 3 | Linked curves |
| `multi_handle_4d` | 4 | Multi-handle manifold |
| `6d_spatial_arm_up_n6_py_traj` | 6 | Upward-facing robot end effector |
| `6d_surf_normal_traj` | 6 | Surface position and normal alignment |
| `12d_dual_arm_traj` | 12 | Dual-arm virtual-link constraint |
| `RealKuka` | 6 | Recorded position and tool-z direction |

## Recorded data and provenance

`RealKuka` contains processed human demonstrations recorded with a KUKA iiwa.
Each sample has six components: normalized TCP position (XYZ) and the measured
tool-z unit direction in the world frame. Loading uses repository-relative paths
through `datasets/real_iiwa/registry.py`.

| Path under `datasets/real_iiwa/` | Contents |
| --- | --- |
| `tasks/RealKuka/processed/train_sets/` | Processed training NPZ |
| `tasks/RealKuka/task.json` | Data provenance, preprocessing and calibration references |
| `surface_models/calibrated_tcp1845_neural_1d_surface_model/` | Surface checkpoint, grid and normalization metadata used for evaluation |

The task metadata records the source collection and recording filenames.

The `RealKuka` default is `human_demo_paper_toolz_arc3mm_cropped_v001.npz`,
with 1,283 points. It was subsampled at 3 mm arc length and cropped to remove
entry/exit segments. Training and evaluation use the included processed data.

Synthetic caches are generated locally and ignored by Git.

## Additional geometry

The `geometry` benchmark group contains trajectory variants of the faceted sphere,
genus-3 tube and spiky sphere. Additional generators include double-torus,
spline/knot and warped-4D manifolds; their parameters are in `configs/datasets/`.

## Additional Datasets Provided by the Project

The following generators support experiments beyond the core benchmark.

| Dataset ID                       | Data Dim | Codim | Traj Note                      | Description                                                                       |
| -------------------------------- | -------: | ----: | ------------------------------ | --------------------------------------------------------------------------------- |
| `2d_discontinuous`               | 2        | 1     | no traj variant                | Piecewise/discontinuous sine-like curve                                           |
| `2d_figure_eight`                | 2        | 1     | no traj variant                | 2D figure-eight curve                                                             |
| `2d_hetero_noise`                | 2        | 1     | no traj variant                | Manifold with non-uniform noise level                                             |
| `2d_looped_spiro`                | 2        | 1     | no traj variant                | Multi-loop spirograph-like curve                                                  |
| `2d_noisy_sine`                  | 2        | 1     | no traj variant                | Sine curve with stronger noise                                                    |
| `2d_sharp_star`                  | 2        | 1     | no traj variant                | Star-like closed curve with sharp corners                                         |
| `2d_sine`                        | 2        | 1     | no traj variant                | Clean sine curve                                                                  |
| `2d_square`                      | 2        | 1     | no traj variant                | 2D square boundary manifold                                                       |
| `3d_0z_2d_discontinuous`         | 3        | 2     | derived lift, non-traj         | Lifted 2D discontinuous dataset with `z=0`                                        |
| `3d_0z_2d_ellipse`               | 3        | 2     | derived lift, non-traj         | Lifted 2D ellipse dataset with `z=0`                                              |
| `3d_0z_2d_figure_eight`          | 3        | 2     | derived lift, non-traj         | Lifted 2D figure-eight dataset with `z=0`                                         |
| `3d_0z_2d_hetero_noise`          | 3        | 2     | derived lift, non-traj         | Lifted 2D hetero-noise dataset with `z=0`                                         |
| `3d_0z_2d_looped_spiro`          | 3        | 2     | derived lift, non-traj         | Lifted 2D looped-spiro dataset with `z=0`                                         |
| `3d_0z_2d_noisy_sine`            | 3        | 2     | derived lift, non-traj         | Lifted 2D noisy-sine dataset with `z=0`                                           |
| `3d_0z_2d_planar_arm_line_n2`    | 3        | 2     | derived lift, non-traj         | Lifted 2D planar-arm-line dataset with `z=0`                                      |
| `3d_0z_2d_sharp_star`            | 3        | 2     | derived lift, non-traj         | Lifted 2D sharp-star dataset with `z=0`                                           |
| `3d_0z_2d_sine`                  | 3        | 2     | derived lift, non-traj         | Lifted 2D sine dataset with `z=0`                                                 |
| `3d_0z_2d_sparse_sine`           | 3        | 2     | derived lift, non-traj         | Lifted 2D sparse-sine dataset with `z=0`                                          |
| `3d_0z_2d_square`                | 3        | 2     | derived lift, non-traj         | Lifted 2D square dataset with `z=0`                                               |
| `3d_paraboloid`                  | 3        | 1     | base version                   | 3D paraboloid surface manifold                                                    |
| `3d_paraboloid_traj`             | 3        | 1     | traj version                   | 3D paraboloid surface manifold with trajectory-style sampling                     |
| `3d_planar_arm_line_n3`          | 3        | 1     | base version                   | 3-DoF planar arm line manifold without trajectory grouping                        |
| `3d_saddle_surface`              | 3        | 1     | base version                   | Saddle-type surface in 3D                                                         |
| `3d_saddle_surface_traj`         | 3        | 1     | traj version                   | Saddle-type surface in 3D with trajectory-style sampling                          |
| `3d_spatial_arm_circle_n3`       | 3        | 2     | no traj variant                | 3-DoF spatial arm constrained on a workspace circle                               |
| `3d_spatial_arm_ellip_n3`        | 3        | 2     | base version                   | 3-DoF spatial arm ellipse manifold without trajectory grouping                    |
| `3d_spatial_arm_plane_n3`        | 3        | 1     | base version                   | 3-DoF spatial arm constrained on a workspace plane                                |
| `3d_spatial_arm_plane_n3_traj`   | 3        | 1     | traj version                   | 3-DoF spatial arm plane manifold with trajectory-style sampling                   |
| `3d_sphere_surface`              | 3        | 1     | base version                   | Sphere surface                                                                    |
| `3d_sphere_surface_traj`         | 3        | 1     | traj version                   | Sphere surface with trajectory-style sampling                                     |
| `3d_spiral`                      | 3        | 2     | no traj variant                | 3D helix/spiral curve manifold                                                    |
| `3d_knot_spline_union`           | 3        | 2     | built-in trajectory gaps       | Union of a trefoil-knot curve and waypoint-spline closed curve, with training demonstrations leaving small gap segments |
| `3d_double_torus_surface`        | 3        | 1     | base version                   | Connected genus-2 double-torus surface, defined as the outer boundary of two obliquely fused solid tori |
| `3d_double_torus_surface_traj`   | 3        | 1     | traj version                   | Connected genus-2 oblique double-torus surface with trajectory-style sampling     |
| `3d_faceted_sphere_surface`      | 3        | 1     | base version                   | Closed bumpy sphere-like surface generated from irregular 3D control points, with faceted ridges and dents |
| `3d_faceted_sphere_surface_traj` | 3        | 1     | traj version                   | Trajectory-demonstration variant of the faceted control-point sphere surface       |
| `3d_spiky_sphere_surface`        | 3        | 1     | base version                   | Hedgehog-like closed surface with many sharp radial spikes and deep dents for stress testing limitations |
| `3d_spiky_sphere_surface_traj`   | 3        | 1     | traj version                   | Trajectory-demonstration variant of the spiky surface stress test                  |
| `3d_torus_surface`               | 3        | 1     | base version                   | Torus surface without trajectory grouping                                         |
| `3d_twosphere`                   | 3        | 1     | base version                   | Two-sphere manifold without trajectory grouping                                   |
| `3d_vz_2d_discontinuous`         | 3        | 2     | derived lift, non-traj         | Lifted 2D discontinuous dataset with varying `z(x,y)`                             |
| `3d_vz_2d_ellipse`               | 3        | 2     | base version                   | Lifted 2D ellipse dataset with varying `z(x,y)` without trajectory grouping       |
| `3d_vz_2d_figure_eight`          | 3        | 2     | derived lift, non-traj         | Lifted 2D figure-eight dataset with varying `z(x,y)`                              |
| `3d_vz_2d_hetero_noise`          | 3        | 2     | derived lift, non-traj         | Lifted 2D hetero-noise dataset with varying `z(x,y)`                              |
| `3d_vz_2d_looped_spiro`          | 3        | 2     | derived lift, non-traj         | Lifted 2D looped-spiro dataset with varying `z(x,y)`                              |
| `3d_vz_2d_noisy_sine`            | 3        | 2     | derived lift, non-traj         | Lifted 2D noisy-sine dataset with varying `z(x,y)`                                |
| `3d_vz_2d_planar_arm_line_n2`    | 3        | 2     | derived lift, non-traj         | Lifted 2D planar-arm-line dataset with varying `z(x,y)`                           |
| `3d_vz_2d_sharp_star`            | 3        | 2     | derived lift, non-traj         | Lifted 2D sharp-star dataset with varying `z(x,y)`                                |
| `3d_vz_2d_sine`                  | 3        | 2     | derived lift, non-traj         | Lifted 2D sine dataset with varying `z(x,y)`                                      |
| `3d_vz_2d_sparse_sine`           | 3        | 2     | derived lift, non-traj         | Lifted 2D sparse-sine dataset with varying `z(x,y)`                               |
| `3d_vz_2d_square`                | 3        | 2     | derived lift, non-traj         | Lifted 2D square dataset with varying `z(x,y)`                                    |
| `6d_spatial_arm_up_n6`           | 6        | 2     | base version, pybullet backend | 6-DoF UR5-style arm upward-orientation set using the pybullet backend             |
| `6d_spatial_arm_up_n6_py`        | 6        | 2     | base version, analytic backend | 6-DoF UR5-style arm upward-orientation set using the analytic backend             |
| `6d_surf_normal` | 6        | 3     | base version                   | Scatter-sample version of the sine-surface pose manifold                          |
| `12d_dual_arm`                   | 12       | 10    | base version                   | Scatter-sample version of the dual-arm vertical-ribbon virtual-link pose manifold |

## Derived Naming Rules

- `3d_vz_<base_2d_dataset>`: lift a 2D base dataset to 3D with varying `z(x,y)` (`data_dim=3`, `codim=2`)
- `3d_0z_<base_2d_dataset>`: lift a 2D base dataset to 3D with `z=0` (`data_dim=3`, `codim=2`)
- `_traj` suffix: trajectory-style sampling variant of the same underlying constraint

## Return Format

- `x_train`: training points on the manifold
- `grid`: dense manifold samples used as reference and evaluation sets

## UR5 Dataset Helpers

Files:

- `datasets/ur5_n6_dataset.py`
- `datasets/ur5_pybullet_utils.py`

Key functions:

- `sample_ur5_upward_dataset(...)`: pybullet-based UR5 upward-orientation dataset sampling
- `sample_ur5_upward_dataset_analytic(...)`: analytic/no-pybullet approximation sampler

## Vendored UR5 Assets

Path:

- `datasets/assets/UR5+gripper/`

The directory contains the URDF, meshes and textures used by PyBullet:

- `ur5_gripper.urdf`
- `mesh/`
- `textures/`
