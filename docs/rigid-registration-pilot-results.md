# Initial six-DOF rigid-registration pilot results

## Review decision

Recommend **B: evaluate initialization/masking and optimizer convergence before
proceeding to final tensor-geometry evaluation**. No second configuration, automatic
tuning, affine/nonlinear comparison, whole-cohort registration or final crop/pad
selection was performed. The configured GEOMETRY-centered, masked-Mattes pilot is
an experimental baseline, not an accepted anatomical normalization pipeline.

The completed outputs have verified common geometry, but visual screening of all
three site contact sheets shows residual superior/inferior and other gross reference
correspondence mismatches. Optimizer/metric flags reinforce that concern. No formal
anatomical QC pass has been assigned, and no disease/medical interpretation is made.

## Method and resource

- SimpleITK **2.5.6**; Euler3D, six free parameters, no scale or shear optimization.
- Exact local TemplateFlow **MNI152NLin2009cAsym res-01 T1w**, with matching fixed
  brain mask, shape **193x229x193**, **1 mm**, **RAS+**, origin **[-96,-132,-78] mm**.
- Template and mask checksum pins, resource citations, initialization/optimizer
  rationale, coordinate conventions and reproduction instructions are in
  [the registration methods document](rigid-registration.md).
- Eighteen unique subjects: **six Guys, six HH, six IOP**. Within each site: two
  low-, two middle-, two high-obliquity cases, including center-offset tails.
- Selected maximum-obliquity ranges: Guys **0.00–18.11 degrees**, HH **0.00–11.22**,
  IOP **0.23–8.54**. Selected center-offset norms reach **19.55 / 6.67 / 30.00 mm**
  respectively. Generated subject IDs remain private.
- One configured registration attempt per subject. **18 technical completions,
  zero execution failures**, all still `completed_qc_review_required`.

Every saved output adopts the exact template shape, spacing, orientation, affine,
origin and FOV. All eighteen saved volumes have finite float32 values, and all saved
transform files were re-read and verified rigid. Maximum rotation-orthonormality
error was **3.33e-16**; determinant range **0.9999999999999997–1.0000000000000002**.
Neither consistent geometry nor rigid-matrix validity establishes anatomical quality.

## Optimizer and metrics

Final-level optimizer stops: **10 iteration-budget exhaustion**, **8 step-too-small**.
Total observed iteration counts across pyramid levels: min **102**, median **298.5**,
mean **303.89**, max **499**. The configured cap was 200 per level across three levels.

Metrics use negative Mattes MI (lower is better). "Before" means after the physical
FOV-center initialization and before optimization; comparison uses the same fixed
mask and deterministic evaluation lattice, separately from stochastic optimizer values.

| Metric | Min | Median | Mean | p95 | Max |
|---|---:|---:|---:|---:|---:|
| Before | -0.054319 | -0.024772 | -0.028141 | -0.016398 | -0.015788 |
| After | -0.098914 | -0.030150 | -0.035398 | -0.018245 | -0.016332 |
| Improvement (before minus after) | -0.001793 | 0.004291 | 0.007257 | 0.023183 | 0.044595 |

Four cases had metric degradation; fourteen improved on the common evaluation lattice.
Improvement alone is insufficient evidence of acceptable anatomical correspondence.
The evaluation-lattice voxel count is its size before mask/overlap rejection; valid
moving overlap can differ with the transform.

## Rotations and translations

These are final **fixed/template LPS -> moving/source LPS pull-transform** parameters,
with the recorded fixed rotation center and `ComputeZYX=True`. They are not simply
apparent subject motion in the output display.

| Parameter | Min | Median | Mean | p95 | Max |
|---|---:|---:|---:|---:|---:|
| Rotation X (degrees) | -1.589 | -0.547 | -0.063 | 2.285 | 3.253 |
| Rotation Y (degrees) | -1.632 | 0.553 | 0.303 | 2.006 | 2.106 |
| Rotation Z (degrees) | -1.391 | -0.160 | -0.037 | 1.302 | 1.481 |
| Translation X (mm) | -7.630 | -0.861 | -1.722 | 2.300 | 3.088 |
| Translation Y (mm) | -49.426 | -34.050 | -33.603 | -19.166 | -13.555 |
| Translation Z (mm) | -34.178 | -15.003 | -11.693 | 15.193 | 15.266 |

No case exceeded the provisional 30-degree per-axis rotation or 100-mm translation
norm warning thresholds. These thresholds are engineering descriptors, not acceptance
criteria or biological limits. Initial and final matrices/parameters remain available
per subject, including cases whose metric worsened.

## Baseline versus registered foreground/crop diagnostics

Signed non-background bbox-center offset relative to each image's own FOV center,
in RAS-axis mm:

| Stage / axis | Min | Median | p95 | Max |
|---|---:|---:|---:|---:|
| Baseline R | -6.0 | 0.0 | 1.075 | 1.5 |
| Registered R | -4.5 | -0.25 | 0.65 | 1.5 |
| Baseline A | -6.5 | -0.75 | 0.675 | 4.5 |
| Registered A | 0.0 | 0.0 | 1.5 | 1.5 |
| Baseline S | -19.0 | 0.0 | 25.75 | 30.0 |
| Registered S | -1.5 | 0.0 | 9.0 | 17.5 |

Apparent tightening is confounded by the smaller template FOV. All eighteen baseline
images would lose baseline non-background foreground under a volume-centered
193x229x193 window (axis counts R/A/S: **6/18/18**). The already template-limited
registered volumes lose none under that same-size window, a geometric tautology
rather than proof of source preservation or correct anatomical registration.

Direct mapping of native non-background voxel centers finds **8.27–22.15%** outside
the template FOV after registration, median **13.49%**, mean **13.96%**, p95 **20.43%**.
Five cases exceed the provisional 15% engineering warning threshold. This rule includes
head/scalp/neck/noise: it is **not a measured fraction of brain tissue lost**. Anatomical
and superior/inferior FOV review remains necessary. No actual downstream crop was run.

## Intensity/interpolation audit

The saved output uses **one direct final linear interpolation from native canonical
source intensities**. The 1 mm baseline and center-initialized QC are separate branches
from native data, not inputs to the final saved image. No raw or normalized image
was repeatedly resampled. Registration pyramids/metric interpolation are optimization
operations, not additional cascaded interpolations of the saved image.

Below are medians of the eighteen per-image statistics in unnormalized source units;
they are not median paired changes and should not be compared biologically across sites.

| Statistic | Baseline | Registered |
|---|---:|---:|
| Foreground mean | 521.44 | 547.83 |
| Foreground std | 519.93 | 529.57 |
| Foreground p0.5 | 0.02123 | 0.02992 |
| Foreground p1 | 0.05858 | 0.08199 |
| Foreground p50 | 370.30 | 436.20 |
| Foreground p99 | 1723.51 | 1734.92 |
| Foreground p99.5 | 1801.05 | 1813.05 |
| Zero/background fraction | 0.41147 | 0.24571 |
| Non-background foreground fraction | 0.58853 | 0.75429 |

Changes reflect interpolation **and** different retained fields/foreground populations.
Private summaries include complete percentile distributions and paired statistic changes.
They are not evidence of biological change or an image-quality/brain-age-MAE improvement.

## Flags, QC artifacts and integrity

- 10 iteration-budget warnings, 4 metric-degradation warnings, 5 provisional
  non-background FOV-exclusion warnings; counts overlap.
- Zero execution failures, nonfinite/nonrigid transforms, incorrect output grids,
  large-rotation warnings or large-translation warnings in this pilot.
- Eighteen per-subject five-row QC panels and three supplementary site review
  contact sheets are private under `artifacts/registration/rigid_pilot_v1/qc/`.
  Site sheets overlay the **fixed** brain-mask contour, never a moving brain mask.
- Private `subjects/*.json`, `manifest.csv`, `summary.json`, `pilot_selection.csv`,
  `transforms/*.tfm`, `volumes/*.nii.gz` and `visual_review_notes.json` retain raw
  parameters, flags, transform direction, intensity/FOV metrics and review limitations.
- All 581 raw MRI size/mtime records, eighteen selected raw-source checksums, input
  manifest/config/template checksums, IXI.xls and the old six-subject pilot checksums
  remained unchanged. Only the new eighteen registered pilot volumes were written.
- No model weights or training, affine/nonlinear registration, final tensor choice,
  commits, pushes or merges were performed.

The next scientific review should decide which initialization/masking/convergence
changes to evaluate in a separate pilot, using anatomical overlays and FOV checks
alongside reproducible metrics. This run does not justify proceeding directly to
final tensor geometry or escalating automatically to affine registration.
