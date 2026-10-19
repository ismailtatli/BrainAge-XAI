# Controlled rigid tuning: corrected v3 results

The exact baseline v1 cohort was reused: 18 subjects, six each from Guys, HH and IOP. No outcome-based selection, per-subject parameter changes, affine/nonlinear registration, moving mask, skull stripping, training or final geometry selection occurred.

The preliminary v2 experiment is retained privately and excluded. A scientific control check found different MOMENTS initializations for C and D. QC resampling had left an ITK requested subregion on reused source images; later intensity moments used that subregion even though voxel values were unchanged. This was reproduced synthetically and on a pilot image. The corrected implementation reconstructs independent full buffers with unchanged physical grids for initialization, metric evaluation and optimization. All 72 conditions were repeated in v3. A/B and C/D initialization equality is checked; the cohort and optimizer settings were unchanged. Baseline v1 files remain byte-identical.

All tuned conditions use seed 42, one ITK thread, 2% RANDOM sampling, 50-bin negative Mattes MI, linear interpolation and RegularStepGradientDescent. Translation-only Euler3D uses weights 0,0,0,1,1,1 at factors 4,2 and smoothing 2,1 mm; full Euler3D continues with all weights 1 at factors 4,2,1 and smoothing 2,1,0 mm. The shared limit is 300 iterations per level, maximum 1,500 per condition, versus v1 maximum 600.

| Condition | Initialization | Fixed metric mask |
|---|---|---|
| v1 | GEOMETRY | Verified template brain mask |
| A | GEOMETRY | None |
| B | GEOMETRY | Verified template brain mask |
| C | MOMENTS | None |
| D | MOMENTS | Verified template brain mask |

## Deterministic metrics and convergence

Each transform is evaluated at full fixed resolution without stochastic sampling. Positive improvement means MI_initial minus MI_final. Masked and unmasked domains differ: absolute MI or improvement magnitudes across those domains do not rank methods. Optimizer sampled trajectories are stored separately from these post-hoc metrics.

| Condition | Successes | MI improved / degraded | Median within-condition improvement | Budget hit: any / final rigid level | Warning events / subjects |
|---|---:|---:|---:|---:|---:|
| v1 | 18/18 | 15 / 3 | 0.004615 | 10 / 10 | 18 / 15 |
| A | 18/18 | 18 / 0 | 0.164110 | 14 / 11 | 14 / 14 |
| B | 18/18 | 14 / 4 | 0.005868 | 11 / 8 | 15 / 13 |
| C | 18/18 | 17 / 1 | 0.093538 | 15 / 13 | 16 / 15 |
| D | 18/18 | 17 / 1 | 0.007483 | 12 / 11 | 13 / 12 |

Baseline v1 originally reported 14 improvements and four degradations on its reduced evaluation lattice. The table reevaluates its unchanged transforms at full resolution; changes in those counts reflect evaluation resolution, not a rewritten baseline. Only v1 final-level stop descriptions were retained, so its any-level count cannot recover unrecorded earlier stops. Tuned counts include all five stage/level stops. Baseline warning events retain descriptive source non-background FOV flags; those are not brain-loss or anatomical acceptance flags.

| Condition | Recorded iteration-budget level stops | Step-too-small level stops | Other level stops |
|---|---:|---:|---:|
| v1 | 10 | 8 | 0 |
| A | 19 | 71 | 0 |
| B | 13 | 77 | 0 |
| C | 21 | 69 | 0 |
| D | 13 | 77 | 0 |

## Site-stratified main results

| Site | Condition | Successes | MI improved / degraded | Median improvement | Budget any / final | Warning events |
|---|---|---:|---:|---:|---:|---:|
| Guys | v1 | 6/6 | 6 / 0 | 0.004986 | 4 / 4 | 4 |
| Guys | A | 6/6 | 6 / 0 | 0.190428 | 6 / 6 | 6 |
| Guys | B | 6/6 | 5 / 1 | 0.004503 | 3 / 2 | 4 |
| Guys | C | 6/6 | 6 / 0 | 0.174701 | 5 / 5 | 5 |
| Guys | D | 6/6 | 5 / 1 | 0.003115 | 6 / 5 | 7 |
| HH | v1 | 6/6 | 4 / 2 | 0.007662 | 4 / 4 | 9 |
| HH | A | 6/6 | 6 / 0 | 0.149996 | 4 / 1 | 4 |
| HH | B | 6/6 | 4 / 2 | 0.007754 | 4 / 4 | 6 |
| HH | C | 6/6 | 6 / 0 | 0.118554 | 4 / 2 | 4 |
| HH | D | 6/6 | 6 / 0 | 0.013992 | 3 / 3 | 3 |
| IOP | v1 | 6/6 | 5 / 1 | 0.004615 | 2 / 2 | 5 |
| IOP | A | 6/6 | 6 / 0 | 0.202742 | 4 / 4 | 4 |
| IOP | B | 6/6 | 5 / 1 | 0.005760 | 4 / 2 | 5 |
| IOP | C | 6/6 | 5 / 1 | 0.033114 | 6 / 6 | 7 |
| IOP | D | 6/6 | 6 / 0 | 0.006575 | 3 / 3 | 3 |

Per-site transform quantiles and coverage distributions are retained in the private summary JSON.

## Rigid transform distributions

Values are LPS Euler X/Y/Z components. These are **fixed-template to moving-source pull transforms**. Centers differ between initialization methods; translation parameters are not direct anatomical displacements. Full centers and homogeneous matrices are retained per case.

| Condition | Rotation p10 (degrees) | Rotation p50 | Rotation p90 |
|---|---|---|---|
| v1 | [-1.282, -1.431, -0.970] | [-0.547, 0.553, -0.160] | [1.758, 1.780, 1.224] |
| A | [-9.444, -1.860, -2.247] | [-5.071, 0.640, -0.839] | [0.392, 3.718, 2.743] |
| B | [-1.615, -0.833, -1.186] | [-0.645, 0.462, 0.008] | [1.599, 1.375, 1.173] |
| C | [-8.119, -1.825, -2.221] | [-4.891, 0.061, -0.910] | [-0.228, 2.147, 2.919] |
| D | [-2.038, -2.367, -1.837] | [-0.648, 0.322, -1.108] | [1.145, 1.327, 0.588] |

| Condition | Translation p10 (mm) | Translation p50 | Translation p90 |
|---|---|---|---|
| v1 | [-5.880, -46.476, -27.491] | [-0.861, -34.050, -15.003] | [1.985, -20.514, 10.937] |
| A | [-5.931, -44.743, -21.964] | [-1.971, -30.034, 4.921] | [2.618, -14.754, 22.932] |
| B | [-7.094, -46.714, -30.143] | [-0.570, -32.371, -11.858] | [4.416, -13.540, 8.974] |
| C | [-6.591, -46.720, -22.037] | [-1.247, -30.783, 5.276] | [2.365, -16.406, 23.235] |
| D | [-5.214, -39.407, -21.387] | [-0.914, -27.532, -10.818] | [5.542, -13.550, 21.068] |

Across completed cases, determinant range was 0.9999999999999996 to 1.0000000000000004; maximum orthonormality error was 3.33e-16. All 90 derived/comparison grids were valid.

## Initialization and FOV coverage

| Mode | Fixed center of rotation (LPS mm) | Median initial translation (LPS mm) | Median initial MI, unmasked | Median initial MI, masked |
|---|---|---|---:|---:|
| GEOMETRY | [0.000, 18.000, 18.000] | [-0.292, -38.897, -17.766] | -0.317754 | -0.023500 |
| MOMENTS | [0.069, 19.209, 2.150] | [-1.303, -30.743, -16.944] | -0.353994 | -0.032461 |

Initial rotations were zero. Identical initial transforms are required within A/B and C/D. MOMENTS uses full-image intensities, including scalp/neck/background context; it is not assumed superior to GEOMETRY. Per-case centers, rotations, translations, metrics and coverage are available in private records. Compare initial MI only within matching mask domains.

| Condition | Initial ROI coverage min / median / max | After translation | Final ROI coverage min / median / max |
|---|---|---|---|
| v1 | 1.000000 / 1.000000 / 1.000000 | N/A | 1.000000 / 1.000000 / 1.000000 |
| A | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 |
| B | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 |
| C | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 |
| D | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 |

Coverage tests fixed brain ROI voxel centers against the source physical voxel-cell FOV. It measures sample availability, not anatomical correspondence. Full FOV coverage can coexist with visibly wrong pose. No acceptance threshold was assigned. Active optimizer valid-point counts are retained in trajectories; post-hoc MetricEvaluate counts are explicitly unavailable. Native non-background exclusion remains descriptive head/neck/noise context in private case records, not a brain-loss measure.

## Visual review and decision

Unmasked A/C show substantially more consistent gross pose correspondence than baseline v1 in the matched-plane review, but residual cortical, cerebellar and inferior contour differences remain. Masked B/D retain clear superior/inferior placement and axial-section outliers. Neither fixed masking nor MOMENTS is uniformly superior. These are spatial QC observations, not medical interpretations.

**Recommendation B: Rigid registration appears viable but still needs another focused tuning step.**

A improved deterministic MI in 18/18 subjects and C in 17/18, and their gross pose correspondence is encouraging across the three sites. However, A/C hit an iteration limit at some level in 14/18 and 15/18 cases (final rigid level: 11/18 and 13/18), with residual visual contour concerns. B/D retain visible pose outliers despite finite, numerically rigid transforms and generally improved MI. Template-brain FOV coverage is 100% even for visibly mismatched cases, so it cannot confer anatomical acceptance. Focus the next rigid experiment on convergence and pose stability of the unmasked candidates; do not expand registration, freeze geometry, or escalate to affine automatically.

This is an engineering screen of matched central planes, without medical interpretation or formal anatomical acceptance. Central planes cannot exclude off-plane/local mismatch. No next experimental option was implemented.

## Outputs and verification

Private output: `artifacts/registration/rigid_tuning_v3/`. It contains 90 subject-condition records, comparison CSV, aggregate/site JSON, 18 six-row comparison panels, three site contact sheets and five condition contact sheets. There are no new registered NIfTI files. The real first-subject A repeat reproduced the transform, full metrics and sampled trace exactly. An additional C repeat reproduced the full trace and transform exactly after QC resampling, with unchanged MOMENTS initialization. Synthetic repeated runs cover all four conditions and the requested-region regression.

Quality gates: 221 passed; Ruff, compileall and pip check passed. Integrity checks confirmed the entire v1 file tree and protected file hashes unchanged, 581 raw MRI size/mtime records unchanged, all selected source SHA256 hashes unchanged, IXI.xls/templates/local configs unchanged, all tuning artifacts ignored, no staged files and no model-weight files. No commit, push, merge or training occurred.

Repeat into a **fresh** ignored output directory:

```sh
.venv/bin/python -m brainage_xai.registration.tuning \
  --config configs/registration.local.yaml \
  --baseline artifacts/registration/rigid_pilot_v1 \
  --output artifacts/registration/rigid_tuning_v3_repeat
```
