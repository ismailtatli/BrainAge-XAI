# Rigid optimizer study: results

The [predefined protocol](rigid-optimizer-study.md) compares exactly O0/O1/O2 on the same 18 subjects (six per site). Corrected condition A supplies the GEOMETRY, unmasked, two-stage formulation. The pinned template, native preprocessing, seed 42, 2% sampling, single ITK thread and pyramids are identical. No per-subject optimizer changes were made.

## Technical validity and deterministic MI

Negative Mattes MI is evaluated at full fixed resolution without sampling. Positive delta is initial minus final. Sampled optimizer metrics are separate diagnostics. Technical validity does not establish anatomical acceptance.

| Optimizer | Technical successes | MI improved / degraded / unchanged | Delta total p10 / median / p90 | Translation / rigid budget hits | Final rigid-level hits | QC concern subjects |
|---|---:|---:|---|---:|---:|---:|
| O0 | 18/18 | 18 / 0 / 0 | 0.053031 / 0.164110 / 0.232686 | 0 / 14 | 11 | 16 |
| O1 | 18/18 | 18 / 0 / 0 | 0.052032 / 0.166669 / 0.236260 | 0 / 0 | 0 | 13 |
| O2 | 18/18 | 18 / 0 / 0 | 0.051370 / 0.164237 / 0.232662 | 4 / 17 | 16 | 17 |

All 54 completed transforms are finite Euler3D transforms without scale/shear. Determinants span 0.9999999999999997 to 1; maximum orthonormality error is 4.44e-16. All completed resampled grids match the template exactly. Translation rotations stay frozen, and each rigid stage starts from its translation result.

Budget flags mean any level within the named stage exhausted the 300-outer-iteration limit. Final rigid-level flags are reported separately. QC concern counts combine engineering warnings (including budget or disagreement), not visual rejection counts.

## Convergence and computational cost

| Optimizer | Budget level stops | Step-too-small stops | Window-convergence stops | Gradient-tolerance stops | Other stops |
|---|---:|---:|---:|---:|---:|
| O0 | 19 | 71 | 0 | 0 | 0 |
| O1 | 0 | 0 | 90 | 0 | 0 |
| O2 | 42 | 37 | 0 | 11 | 0 |

These are all five level stops per completed case. A window/step termination is an optimizer criterion, not proof of a global optimum or anatomical correspondence. Exact descriptions, consumed iterations, actual physical-shift scales and learning rates are retained per case/level and in compact sampled-trajectory CSVs.

| Optimizer | Timing scope | Median seconds | p90 seconds | Maximum seconds |
|---|---|---:|---:|---:|
| O0 | Case | 35.10 | 37.06 | 37.22 |
| O0 | Translation stage | 3.95 | 4.05 | 4.11 |
| O0 | Rigid stage | 25.53 | 27.32 | 27.48 |
| O1 | Case | 33.31 | 35.46 | 36.20 |
| O1 | Translation stage | 5.50 | 5.89 | 5.95 |
| O1 | Rigid stage | 21.89 | 23.27 | 25.25 |
| O2 | Case | 37.09 | 38.16 | 39.18 |
| O2 | Translation stage | 4.12 | 4.33 | 4.89 |
| O2 | Rigid stage | 27.20 | 28.43 | 28.85 |

Case timing includes initialization, metric/coverage evaluations, stages and final resampling/grid checks; source loading and PNG/table exports are excluded. Stage timings include setup and optimizer execution, with execute-only values also retained. Order was O0/O1/O2 per preloaded source. Host load was not controlled; synthetic quality checks also ran during the original comparison. These are descriptive wall-clock measurements. Line search can perform multiple metric evaluations per outer iteration, so equal iteration budgets do not imply equal computational work.

## Site-stratified results

| Site | Optimizer | Successes | MI improved / degraded / unchanged | Delta p10 / median / p90 | Translation / rigid budget hits | Final rigid-level hits | Runtime median / p90 / max (s) | QC concern subjects |
|---|---|---:|---:|---|---:|---:|---|---:|
| Guys | O0 | 6/6 | 6 / 0 / 0 | 0.058317 / 0.190428 / 0.228693 | 0 / 6 | 6 | 35.80 / 37.18 / 37.22 | 6 |
| Guys | O1 | 6/6 | 6 / 0 / 0 | 0.055149 / 0.183576 / 0.226065 | 0 / 0 | 0 | 33.78 / 35.06 / 35.85 | 4 |
| Guys | O2 | 6/6 | 6 / 0 / 0 | 0.058793 / 0.192069 / 0.229088 | 1 / 5 | 5 | 37.10 / 37.89 / 38.10 | 5 |
| HH | O0 | 6/6 | 6 / 0 / 0 | 0.083766 / 0.149996 / 0.164110 | 0 / 4 | 1 | 34.29 / 36.61 / 36.93 | 5 |
| HH | O1 | 6/6 | 6 / 0 / 0 | 0.078339 / 0.149626 / 0.166669 | 0 / 0 | 0 | 28.70 / 34.46 / 35.29 | 4 |
| HH | O2 | 6/6 | 6 / 0 / 0 | 0.082450 / 0.148765 / 0.164237 | 2 / 6 | 5 | 37.09 / 37.40 / 37.44 | 6 |
| IOP | O0 | 6/6 | 6 / 0 / 0 | 0.055862 / 0.202742 / 0.274460 | 0 / 4 | 4 | 35.25 / 36.56 / 37.02 | 5 |
| IOP | O1 | 6/6 | 6 / 0 / 0 | 0.056666 / 0.200606 / 0.279266 | 0 / 0 | 0 | 33.40 / 35.32 / 36.20 | 5 |
| IOP | O2 | 6/6 | 6 / 0 / 0 | 0.056559 / 0.201840 / 0.273904 | 1 / 6 | 6 | 36.85 / 38.74 / 39.18 | 6 |

Per-site stop descriptions/categories, transform/coverage distributions and stage-runtime quantiles are retained in the ignored summary JSON.

## Transform stability and coverage

Rotation/translation components are LPS X/Y/Z. Transforms map fixed template points into moving source coordinates (pull); translations are parameter values at the shared center, not anatomical error.

| Optimizer | Quantity | p10 XYZ | Median XYZ | p90 XYZ |
|---|---|---|---|---|
| O0 | Rotation (degrees) | [-9.444318, -1.860213, -2.247244] | [-5.070708, 0.640196, -0.838522] | [0.392316, 3.717835, 2.742504] |
| O0 | Translation (mm) | [-5.931278, -44.742789, -21.964266] | [-1.970733, -30.034442, 4.921156] | [2.617803, -14.753955, 22.932390] |
| O1 | Rotation (degrees) | [-2.412520, -0.612151, -0.575154] | [-0.348634, 0.250961, -0.020142] | [0.125311, 0.904414, 0.687261] |
| O1 | Translation (mm) | [-8.363864, -46.681489, -21.014174] | [-0.843717, -30.259364, 6.421519] | [2.325787, -16.415746, 23.063307] |
| O2 | Rotation (degrees) | [-9.641065, -1.898692, -1.873580] | [-5.753216, 0.212314, -0.891142] | [-0.618756, 2.457137, 2.992762] |
| O2 | Translation (mm) | [-6.456828, -44.840320, -22.146506] | [-2.250859, -29.982885, 5.063043] | [2.524432, -14.788056, 23.160959] |

| Pair | Site | Rotation p10 / median / p90 (degrees) | Rotation max | Translation p10 / median / p90 (mm) | Translation max | Review flags |
|---|---|---|---:|---|---:|---:|
| O0/O1 | Overall | 1.447110 / 5.021687 / 7.732299 | 10.947519 | 1.710778 / 3.468212 / 5.664615 | 7.071111 | 11 |
| O0/O1 | Guys | 2.976363 / 4.485317 / 7.294532 | 7.651964 | 1.673413 / 2.411218 / 3.713961 | 3.934031 | 2 |
| O0/O1 | HH | 2.333510 / 5.929149 / 9.433634 | 10.947519 | 2.772178 / 4.249097 / 5.989272 | 6.800914 | 4 |
| O0/O1 | IOP | 1.131073 / 5.350915 / 7.472734 | 7.541110 | 2.303775 / 3.526574 / 5.354821 | 7.071111 | 5 |
| O0/O2 | Overall | 0.377239 / 1.186021 / 3.062732 | 4.689603 | 0.139220 / 0.510853 / 1.012423 | 1.374197 | 0 |
| O0/O2 | Guys | 0.622611 / 1.423817 / 2.622382 | 2.877408 | 0.216018 / 0.809760 / 1.171893 | 1.374197 | 0 |
| O0/O2 | HH | 0.251163 / 0.889581 / 1.249938 | 1.311235 | 0.096617 / 0.405658 / 0.937692 | 1.004640 | 0 |
| O0/O2 | IOP | 0.692674 / 1.199792 / 4.092377 | 4.689603 | 0.297424 / 0.414647 / 0.879833 | 1.030583 | 0 |
| O1/O2 | Overall | 1.701537 / 5.709505 / 8.124573 | 10.427822 | 2.129553 / 3.427023 / 5.432679 | 7.604162 | 12 |
| O1/O2 | Guys | 2.998494 / 5.709505 / 8.738350 | 10.427822 | 1.633193 / 2.945331 / 4.050207 | 4.584427 | 4 |
| O1/O2 | HH | 1.932331 / 5.447575 / 8.631761 | 9.899730 | 3.040626 / 4.214281 / 6.163886 | 7.604162 | 4 |
| O1/O2 | IOP | 1.780887 / 4.758923 / 6.595628 | 6.638629 | 2.094856 / 3.317171 / 5.372477 | 7.087176 | 4 |

Pairwise rotation is the relative-matrix geodesic angle. Translation disagreement is pull-location displacement at the common fixed center. Prespecified >5 degree or >5 mm flags trigger review; they do not identify which optimizer is anatomically correct. Euler component differences and full centers/matrices remain in private records.

| Optimizer | Stage | Coverage min | Coverage median | Coverage max |
|---|---|---:|---:|---:|
| O0 | after_translation | 1.000000 | 1.000000 | 1.000000 |
| O0 | final | 1.000000 | 1.000000 | 1.000000 |
| O0 | initial | 1.000000 | 1.000000 | 1.000000 |
| O1 | after_translation | 1.000000 | 1.000000 | 1.000000 |
| O1 | final | 1.000000 | 1.000000 | 1.000000 |
| O1 | initial | 1.000000 | 1.000000 | 1.000000 |
| O2 | after_translation | 1.000000 | 1.000000 | 1.000000 |
| O2 | final | 1.000000 | 1.000000 | 1.000000 |
| O2 | initial | 1.000000 | 1.000000 | 1.000000 |

The MNI brain mask is solely a spatial coverage reference and QC contour. No metric masks are used. Coverage measures source FOV availability; it does not measure anatomical alignment, and no acceptance threshold was assigned.

## Visual engineering review

All 18 subject panels and site/optimizer contact sheets were reviewed at matched sagittal/coronal/axial planes, including fixed internal intensity contours and checkerboards. Cortical, ventricular, cerebellar and superior/inferior correspondence were considered. This is a qualitative engineering screen with no medical interpretation; central planes cannot establish whole-volume anatomical reliability. Later assessments are stored separately in `visual_review.json`; the original case JSON status fields remain frozen at their execution-time values.

| Optimizer | Reviewed subjects | Gross-pose review concerns | Residual-contour review concerns |
|---|---:|---:|---:|
| O0 | 18 | 0 | 18 |
| O1 | 18 | 0 | 18 |
| O2 | 18 | 0 | 18 |

Visible contour differences are review observations, not failed registrations; individual shape may contribute and this screen does not separate shape variation from residual pose error.

O0/O2 retain similar gross pose across the three sites. O1 often retains pitch closer to initialization and changes matched section placement; no uniform visual advantage is established. Large disagreement is a review warning, not proof that an optimizer is anatomically wrong.
Residual cortical, ventricular and inferior/cerebellar contour differences remain. All six IOP cases have dark inferior signal that limits internal/cerebellar assessment. These observations do not distinguish individual shape variation from residual registration error and are not anatomical rejection labels.

## Candidate selection and full-cohort repeat

Retain O0 as the best-supported global reference candidate. All three have 18/18 technical successes and full-MI improvements, and identical full FOV coverage. O0 and O2 are comparatively close in transforms and gross visual pose (no prespecified pairwise review flags). O2 exhausts the final rigid-level budget in 16/18 versus O0 11/18, is slower, and provides no consistent visual benefit. O1 has 90/90 window-convergence stops and a slightly higher median MI improvement, but differs from O0 beyond the prespecified rotation/translation review limits in 11/18 subjects, with no consistent anatomical advantage in matched-plane review. Those disagreements do not prove O1 wrong or O0 correct. The evidence does not establish a superior replacement for O0; selecting O0 for repeat is not anatomical acceptance.

The globally selected candidate was **O0**, selected after all 54 comparison results and visual panels were available. It was rerun for all 18 subjects with the same seed, template, configuration, core code hashes and recorded environment.

Repeat: 18/18 passed; 18/18 transforms and all three deterministic MI values were exactly equal. Maximum absolute transform/MI difference was 0. Stop conditions matched in 18/18 and sampled trajectory CSVs in 18/18. Predeclared equivalence tolerance: absolute 1e-10, relative 0; runtime was excluded.

## Decision

**Recommendation B: Rigid registration remains viable but optimizer behavior or residual alignment still needs focused tuning.**

O0 remains technically valid and useful as the retained reference, but still exhausts the final rigid-level budget in 11/18 subjects. O1 removes budget stops by reaching different solutions without a consistent visual advantage; O2 increases budget exhaustion. Residual contours, optimizer-dependent pose and limited inferior visibility leave anatomical reliability unresolved. Address convergence behavior and remaining alignment/QC uncertainty before expanding validation or freezing model geometry. No next experiment was implemented.

No next experiment was implemented. No final tensor geometry or slice positions were frozen.

## Outputs and verification

Ignored local output: `artifacts/registration/rigid_optimizer_v4/`. It contains 54 comparison subject records/trajectories, a 54-row subject/optimizer table, 54 pairwise transform comparisons, aggregate/site JSON, 18 subject QC panels, three site sheets and three optimizer sheets. The selected full repeat adds 18 subject records/trajectories. No permanent registered NIfTI collection was created. Subject-bearing tables, images, review notes, input hashes and paths remain ignored.

Integrity verification passed for all 581 raw MRI SHA256 hashes and size/mtime records, IXI.xls, pinned templates, local configs and all 1480 protected prior files. Prior file trees are unchanged. 234 passed in 21.54s; Ruff, compileall and pip check passed. No staged files or model weights; no commit, push, merge or training.

The completed comparison was validated and consumed without rerunning O0/O1/O2 or rewriting its results. SHA256 checks protect all 137 completed comparison files, including the 54 original result JSONs, traces, QC panels and summary. Only the selected full repeat was executed. The existing `--repeat` CLI supported this operation, so no resume code change was necessary.

Command used for the separate repeat:

```sh
.venv/bin/python -m brainage_xai.registration.optimizer_study \
  --config configs/registration.local.yaml \
  --baseline artifacts/registration/rigid_pilot_v1 \
  --corrected artifacts/registration/rigid_tuning_v3 \
  --output artifacts/registration/rigid_optimizer_v4 \
  --repeat O0
```

Repeat outputs are separate under `repeat/O0/`. The CLI refuses an existing repeat target, protecting this completed repeat from overwriting. This command records the operation performed; no additional comparison or repeat was launched.
