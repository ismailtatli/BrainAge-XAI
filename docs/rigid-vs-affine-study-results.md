# Final rigid-versus-affine registration decision (v6)

**RIGID. Freeze the existing S2/O0 rigid registration method.** Registration tuning is closed. The authoritative path-free specification is [registration.frozen.yaml](../configs/registration.frozen.yaml). Crop, slice and tensor geometry are not frozen; all-subject preprocessing and training were not run.

Freeze existing S2/O0 rigid registration. Affine gives a clear global boundary advantage and improves deterministic MI in all eighteen cases, but introduces substantial, anisotropic scale changes: moving-to-fixed volume factors 1.205–1.701, median 1.430, and ten prespecified deformation-review flags. Shear is modest; scale is the main concern. IOP inferior/internal visibility remains limited. The evidence does not establish that this morphology normalization preserves age-relevant variation. Choose pose normalization that preserves global scale/shape, accepting the documented template correspondence residuals. This is not evidence of superior age prediction or perfect voxelwise alignment.

## Controlled design and execution

The [predefined protocol](rigid-vs-affine-study.md) was written before the real run. Exactly eighteen stored S2/O0 results were reused unchanged, and exactly eighteen new affine registrations ran on the same six subjects per site. No rigid optimization, baseline MI reevaluation, extra real affine repeat, parameter search or nonlinear registration ran. The affine stage starts from the complete final rigid physical mapping and optimizes twelve parameters on the original native image. There is no new translation warm-up. Both methods use the same pinned MNI template, unmasked 50-bin Mattes MI, RANDOM 2% sampling, seed 42, linear interpolation and multiresolution pyramid.

The affine stage uses the unchanged O0 RegularStepGradientDescent configuration: learning rate 2, minimum step 0.001, relaxation 0.5, gradient tolerance 1e-6, 300 iterations per level, Never learning-rate estimation, physical-shift scales with radius 5 and variation 0.01, one ITK thread. All twelve weights are enabled. Per-level physical scales and iteration trajectories are saved. Resampling uses the full final affine directly from native moving intensities to the exact template grid.

## Metrics, convergence and runtime

Full-resolution negative Mattes MI is evaluated without sampling. Lower is better. Paired gain means stored rigid final MI minus affine final MI. The affine initial MI is the stored rigid final MI; no rigid metric is recomputed. Rigid initial-to-final improvement and affine initial-to-final improvement have different starting mappings and must not be equated.

| Method | Success / grid-valid | Full-resolution final MI p10 / median / p90 | Final-level budget hits | Any full-stage level budget hits | Runtime median / p90 / max (s) |
|---|---:|---|---:|---:|---|
| RIGID | 18/18 / 18/18 | -0.521486 / -0.457370 / -0.402397 | 11 | 14 | 35.10 / 37.06 / 37.22 |
| AFFINE | 18/18 / 18/18 | -0.717184 / -0.690845 / -0.647294 | 3 | 13 | 12.66 / 29.98 / 30.05 |

Affine improves final MI in 18/18 pairs. Paired MI gain p10 / median / p90 is **0.181714 / 0.236384 / 0.285564**. The gain is large and is acknowledged; the final choice does not dispute this numerical benefit.

Rigid runtime is historical whole-case time, including its original translation and rigid stages, metric evaluations and resampling. Affine runtime is additional affine optimization, final metric/coverage and resampling, excluding source loading and QC export. Affine case timing also includes trajectory CSV export, whereas the stored rigid timing excludes it. Adding paired historical rigid and new affine times estimates the complete rigid-plus-affine pipeline; it is not valid to compare incremental affine cost with whole rigid cost as alternative pipelines. Host load was uncontrolled.

Estimated complete rigid-plus-affine pipeline runtime p10 / median / p90: 36.72 / 46.71 / 65.59 seconds. Rigid full-stage level stops: {'iteration_budget': 19, 'step_too_small': 35}; affine level stops: {'iteration_budget': 19, 'step_too_small': 35}. Counts cover three full-stage levels per subject. Original translation warm-up stops remain in the reused subject JSON. A step-size stop or budget hit is not an anatomical acceptance test.

## Site comparisons

| Site | Rigid / affine successes | Affine MI gain p10 / median / p90 | Rigid / affine final-level hits | Affine deformation review flags | Rigid historical / affine incremental median (s) |
|---|---:|---|---:|---:|---|
| Guys | 6/6 / 6/6 | 0.177230 / 0.236384 / 0.293600 | 6 / 1 | 4/6 | 35.80 / 14.16 |
| HH | 6/6 / 6/6 | 0.186927 / 0.224937 / 0.288486 | 1 / 0 | 1/6 | 34.29 / 9.70 |
| IOP | 6/6 / 6/6 | 0.182095 / 0.220020 / 0.268959 | 4 / 2 | 5/6 | 35.25 / 21.83 |

| Site | Affine pull determinant median (range) | Moving-to-fixed volume factor median (range) | Polar rotation disagreement median / max (deg) | Common-center pull displacement median / max (mm) |
|---|---|---|---|---|
| Guys | 0.686 (0.644–0.814) | 1.458 (1.229–1.554) | 6.89 / 19.68 | 14.36 / 16.59 |
| HH | 0.755 (0.697–0.830) | 1.325 (1.205–1.434) | 4.20 / 13.84 | 7.28 / 12.53 |
| IOP | 0.666 (0.588–0.727) | 1.504 (1.375–1.701) | 6.70 / 15.26 | 14.99 / 18.29 |

All sites improve numerical MI. Guys and IOP have more scale/volume review flags than HH. These are descriptive site differences from six cases per site, confounded by individual anatomy and acquisition; they are not estimates of a causal scanner or age effect. There is no observed gross-pose or FOV regression, but there is a site-dependent deformation burden and unresolved IOP visibility.

## Transform deformation and scientific interpretation

All eighteen affine transforms are finite, invertible and orientation preserving. All rigid transforms preserve scale and shear. All thirty-six saved/reused case grids match the template, and fixed-brain-ROI FOV coverage is exactly 1.0 for both final methods. Coverage is a geometric availability check, not a correspondence score. No mask entered optimization.

| Affine diagnostic | p10 / median / p90 | Range or interpretation |
|---|---|---|
| Pull determinant / volume factor | 0.634554 / 0.699496 / 0.810407 | 0.587794–0.829897 |
| Cube-root-determinant global pull scale | 0.859307 / 0.887691 / 0.932325 | 0.837674–0.939741 |
| Moving-to-fixed volume factor | 1.233957 / 1.429605 / 1.576168 | 1.204969–1.701278 |
| Maximum absolute principal pull-scale deviation | 0.139694 / 0.185925 / 0.214788 | 0.108135–0.238511 |
| Maximum absolute inter-column cosine | 0.016329 / 0.032002 / 0.054898 | 0.011671–0.085746 |
| Maximum absolute normalized polar shear | 0.008134 / 0.015944 / 0.027456 | 0.005835–0.042828 |
| Absolute polar rotation angle (degrees) | 5.299289 / 11.124980 / 18.297022 | 3.202101–24.571240 |
| Principal pull scales, descending | [0.914, 0.857, 0.785] / [0.952, 0.898, 0.814] / [1.006, 0.939, 0.860] | Overall individual singular-value range 0.761489–1.049731 |
| Centered affine translation XYZ (mm LPS) | [-5.867, -42.271, -12.550] / [-0.828, -24.223, 19.460] / [2.285, -5.953, 34.386] | Center retained from rigid; pull displacement, not a biological measurement |

The stored matrix maps fixed-template LPS into native-moving LPS. Its determinant therefore describes the pull map. The reciprocal determinant describes the moving-to-fixed volume change in the registered geometry. All eighteen registered-volume factors exceed 1, ranging from approximately 20.5% to 70.1% expansion, median 43.0%. This is a mathematical coordinate change, not evidence of biological enlargement or tissue change. Singular values show anisotropic scaling; low nonorthogonality indicates that scale, rather than large shear, dominates the concern.

Prespecified review flags total 10/18: {'unusual_principal_scale': 7, 'unusual_volume_factor': 9}. Triggers were any singular value outside [0.8, 1.25], pull determinant outside [0.7, 1.4], or absolute inter-column cosine above 0.15. No nonorthogonality trigger fired. These provisional engineering limits are not validated biological safety cutoffs. The final decision considers the magnitude and distribution of scaling, including changes below the triggers; it is not an automatic threshold vote.

Global scale/shape normalization can remove variation that a brain-age model might use. This experiment contains no age-prediction or morphology-retention endpoint, so it cannot show whether removing that variation helps or harms the eventual model. The conservative freeze retains six-degree-of-freedom pose normalization without systematic global scale normalization. MI improvement alone cannot override that scientific limitation.

## Visual engineering review

All eighteen individual matched central sagittal/coronal/axial panels and all three site sheets were inspected by Codex image-based engineering review, not an independent human rating. Panels contain MNI, stored rigid S2 and affine, fixed ROI contours, intensity edges and checkerboards. No medical interpretation is made. Central planes and contour overlays cannot prove whole-volume correspondence, and normal individual shape differences contribute to residual contours.

| Site | Boundary/pose comparison | Ventricles and cerebellum | FOV / stretching observations |
|---|---|---|---|
| Guys | Clear affine cortical and superior/inferior boundary advantage in 6/6 | Visible internal and cerebellar alignment generally improves; individual detail differences remain | No obvious gross-pose regression, clipping or localized stretch artifact; substantial global scaling remains a numerical concern |
| HH | Clear affine cortical and superior/inferior boundary advantage in 6/6 | Visible ventricular and cerebellar correspondence generally improves; fine differences remain | No obvious gross-pose regression, clipping or localized stretch artifact |
| IOP | Cortical/superior envelope improves in 6/6 | Low inferior/internal signal limits ventricular and cerebellar assessment in all six, before and after affine | No obvious gross-pose or FOV regression; five deformation-review flags; internal alignment is not confidently accepted |

The selected rigid reference retains the previously documented cortical/superior and inferior/internal correspondence residuals. Freezing it is a method-selection tradeoff in favor of retaining individual global morphology, not a reversal of those observations or a claim of perfect tissue alignment. No participant exclusions are made.

## Frozen registration specification and limits

Freeze S2/O0: GEOMETRY initialization, no metric mask, translation-only warm-up on levels [4,2], full Euler3D stage on [4,2,1], unchanged O0 RegularStepGradientDescent and physical-shift scales, 300 iterations per level, 2% RANDOM sampling with seed 42 and one thread. The pinned template and direct native-to-template interpolation are preserved. The saved S2 real-data repeatability evidence is exact on all eighteen subjects. Affine determinism was checked on synthetic data only; the eighteen-only real affine budget was respected.

The frozen YAML is a path-free scientific specification, separate from existing local command configurations. It deliberately does not alter the historical pilot defaults or local paths. It must be used explicitly when future work integrates the frozen method. Registration tuning is closed; no additional experiment, nonlinear registration or automatic next-step implementation is recommended. Crop/slice/tensor geometry, all-561-subject processing and training remain deferred.

## Artifacts and verification

Private output: `artifacts/registration/rigid_vs_affine_v6/`. It contains eighteen unchanged S2 reference records, eighteen new affine records and trajectory CSVs, a 36-row method comparison, an 18-row paired comparison, aggregate/site summaries, eighteen comparison PNGs and three site sheets. Full affine matrices, polar/SVD diagnostics, translations, optimizer stops and all other requested case-level data are in the ignored JSON/CSV files. No permanent registered NIfTI volumes are written. Visual notes, method selection, quality gates, validation and integrity verification are saved alongside the results.

The original run refuses a nonempty output directory. The `--finalize-only` command in the protocol regenerates numerical tables solely from existing JSON and cannot rerun registration. Do not rerun the completed study.

Validation: **249 tests passed in 22.42 seconds**; Ruff, source compileall and pip check passed. The nine new synthetic tests cover centered rigid-to-affine mapping, invalid transformations, determinant/SVD/scale/shear calculations, physical optimizer scales, fixed-seed determinism, template-grid output, paired aggregation, exactly eighteen affine calls, finalization without registration and prior-output protection.

Integrity verification hashes all 581 raw MRI files and checks size/mtime, the metadata spreadsheet, manifests, local configs, pinned resources and all 1,780 pre-existing protected files including every previous registration artifact. The prior file trees and old registration implementation are unchanged. Nothing is staged. No commit, push, merge, training or all-subject preprocessing occurred.
