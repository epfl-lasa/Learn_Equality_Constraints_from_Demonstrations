# Evaluation metrics

Training runners call [eval_runner.py](eval_runner.py), which combines projection
and coverage metrics from [evaluator.py](evaluator.py). Results are saved in each
run's `*_eval.json` file and in the benchmark summary tables.

## Evaluation procedure

1. Merge evaluator defaults, the run configuration, method overrides and dataset
   overrides, in that order.
2. Load the packaged `eval_grid` for the three fixed demonstration tasks.
   Other synthetic tasks use their dataset generator, and RealKuka uses the
   included calibrated surface model.
3. Set the residual threshold `eps` to the `zero_eps_quantile` percentile of
   `||F(x_train)||₂`. Both classification and projection use this threshold.
4. Sample query points and project them with the method's projector. Compute
   residuals, distances to the reference manifold and classification statistics.
5. Reuse the projected points for bidirectional nearest-neighbor coverage metrics.

Query sampling uses a fixed RNG seed (`2026`), independently of the training seed.
By default, 75% of queries are perturbed reference-manifold samples and 25% are
uniform samples in the evaluation bounds. `eval_chamfer_near_source=gt` selects
reference samples; `train` selects training samples instead. Bounds come from the
training data with padding, with task-specific handling of joint limits and angles.
The reference grid and sampling settings affect the reported distances and should
be held fixed when comparing methods.
The packaged tasks use the same reference grid for all training seeds;
`eval_gt_n_grid` controls grid generation for the other synthetic tasks.

## Prediction and projection

Let `r(x) = ||F(x)||₂`. A point is classified as on-manifold when `r(x) < eps`.
The geometric threshold `tau` is `eval_tau_ratio` times the mean axis range of the
reference grid in the evaluation space.

| Metric | Definition | Direction |
| --- | --- | --- |
| `pred_on_mean_v` | Mean `0.5 * r(x)²` on training samples | Lower |
| `pred_recall` | Fraction of reference samples classified as on-manifold | Higher |
| `pred_FPrate` | Fraction of query points with reference distance ≥ `tau` classified as on-manifold | Lower |
| `pred_precision` | Fraction of finite projected points with reference distance < `tau` | Higher |
| `proj_manifold_dist` | Mean distance from projected points to the reference manifold | Lower |
| `proj_true_dist` | Mean distance from each projected point to the reference target for its original query | Lower |
| `proj_v_residual` | Mean `r(x)` after projection | Lower |
| `proj_steps` | Mean projector iteration count | Interpret with accuracy |

`pred_precision` measures geometric accuracy of the projected set. It is not
classification precision on the original query set. False-positive rate is `NaN`
when there are no query points outside the geometric threshold.

For generic datasets, reference distances and projection targets use nearest
neighbors in the finite reference grid. SurfNormal and dual-arm pose evaluation
use task-specific analytic targets when workspace embedding is enabled. RealKuka
uses its calibrated surface for manifold distances; its `proj_true_dist` target
still comes from the reference grid. These targets are evaluation references and
need not be exact Euclidean closest-point projections.

## Coverage

| Metric | Definition |
| --- | --- |
| `gt_to_learned_mean` | Mean nearest-neighbor distance from reference samples to projected samples |
| `learned_to_gt_mean` | Mean nearest-neighbor distance from projected samples to reference samples |
| `bidirectional_chamfer` | Sum of the two directional means |
| `gt_to_learned_p95` | 95th percentile of reference-to-projected distances |
| `learned_to_gt_p95` | 95th percentile of projected-to-reference distances |

Lower values indicate closer agreement. Distances are Euclidean, not squared;
`bidirectional_chamfer` is a sum, not the average of the two directions.

## Coordinate spaces and RealKuka errors

Generic tasks use data-space distances. Robot tasks can use a workspace embedding;
pose embeddings combine position and orientation components, so their combined
distances should not be read as position errors in metres.

For RealKuka, Chamfer distances use world-frame XYZ in metres and do not include
orientation. The additional projection metrics separate position and orientation:

- `real_iiwa_proj_position_error_mm_{mean,median,p95,max}`: local surface-distance
  estimate, computed from the height residual divided by `sqrt(1 + ||gradient||²)`,
  in millimetres. `real_iiwa_proj_position_error_m_mean` reports the mean in metres.
- `real_iiwa_proj_orientation_error_deg_{mean,median,p95,max}`: angle between the
  normalized projected tool-z axis and the signed calibrated surface normal.
- `real_iiwa_proj_orientation_axis_l2_{mean,median,p95,max}`: Euclidean distance
  between those two unit axes.

The merged `dist_space` field describes the **projection** evaluation space
(`data_space`, `workspace`, `workspace_analytic` or `real_iiwa_analytic`). In
particular, it does not describe the separate world-XYZ Chamfer calculation for
RealKuka. `eval_eps_used` and `eps_stop` record the same residual threshold;
`n_gt` and `n_learned` record the sample counts used for coverage.

## Default settings

These values are defined in `DEFAULT_EVAL_CFG`; run, method and dataset settings
can override them.

| Setting | Default | Purpose |
| --- | ---: | --- |
| `zero_eps_quantile` | 90.0 | Training residual percentile for `eps` |
| `eval_tau_ratio` | 0.02 | Geometric classification threshold scale |
| `eval_gt_n_grid` | 8128 | Requested reference grid size; actual size depends on the generator |
| `eval_proj_n_points` | 4096 | Number of query points |
| `eval_chamfer_near_source` | `gt` | Source of near-manifold queries |
| `eval_chamfer_near_ratio` | 0.75 | Fraction of near-manifold queries |
| `eval_chamfer_near_noise_std_ratio` | 0.1 | Perturbation scale relative to each axis span |
| `eval_pad_ratio` | 0.6 | Evaluation bounds padding |
| `eval_min_axis_span_ratio` | 0.08 | Minimum span for narrow axes |
