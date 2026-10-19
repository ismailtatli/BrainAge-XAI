# Phase 2 experimental rigid-registration pilot

This is a research evaluation of six-DOF alignment, prompted by measured IXI
obliquity/position variability and unsafe centered crops in the geometry audit.
It does not predict brain age, compute BAG, train a model, or diagnose disease.
No final crop/pad or slice geometry is selected. Only eighteen eligible subjects
are registered; the full 561-subject cohort is not registered by this command.
The existing Phase 2 pipeline and six-subject preprocessing pilot remain available.

## Exact local reference and citations

The reference is **TemplateFlow MNI152NLin2009cAsym, resolution 01, T1w**, specifically
`tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz`. The matching fixed-image mask is
`tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz`.

The [TemplateFlow resource repository](https://github.com/templateflow/tpl-MNI152NLin2009cAsym)
and its [template_description.json](https://github.com/templateflow/tpl-MNI152NLin2009cAsym/blob/master/template_description.json)
define the requested 1 mm grid: shape `[193,229,193]`, RAS+ orientation, voxel-center
origin `[-96,-132,-78]` mm, and identity spatial rotation/scaling at 1 mm spacing.
Runtime checks the full affine, mm units, finite/nonempty image, shape, spacing,
orientation, binary mask, and exact SHA-256 checksums. Missing/incompatible resources
fail; another MNI reference is never substituted. The resource is a population
reference with construction/population biases, not a biologically neutral standard.

Pinned T1 SHA-256:

```text
1f27aabea9f7183dc0c69dafa71e1787b921ca778d655d9c1bf302b273d5627a
```

Pinned mask SHA-256:

```text
e40bb1816736504d4c25abc243c1c8503df1d308f64c048318aa4b829e09a7ae
```

The downloaded bytes were independently checked against the upstream git-annex
MD5 keys `6c64ebf985fe30d93e49a4baa41b9278` (T1) and
`dc727a2781ca720df0385f9501ace07d` (mask). SHA-256 pins are enforced thereafter.
Both local files stay under ignored `artifacts/resources/`; no template binary is
part of this repository's tracked source. Consult the resource repository's license
when redistributing or using the template.

Citations:

- Ciric R, Thompson WH, Lorenz R, et al. *TemplateFlow: FAIR-sharing of multi-scale,
  multi-species brain models.* Nature Methods 19, 1568–1571 (2022).
  [doi:10.1038/s41592-022-01681-2](https://doi.org/10.1038/s41592-022-01681-2).
- Fonov VS, Evans AC, McKinstry RC, Almli CR, Collins DL. *Unbiased nonlinear average
  age-appropriate brain templates from birth to adulthood.* NeuroImage 47 Suppl 1,
  S102 (2009). [doi:10.1016/S1053-8119(09)70884-5](https://doi.org/10.1016/S1053-8119(09)70884-5).
- The TemplateFlow metadata also cites Fonov et al., *Unbiased average age-appropriate
  atlases for pediatric studies*, NeuroImage 54, 313–327 (2011).
  [doi:10.1016/j.neuroimage.2010.07.033](https://doi.org/10.1016/j.neuroimage.2010.07.033).

## Setup and explicit resource retrieval

Install the updated project in Python 3.11. SimpleITK has an explicit bounded runtime
dependency `>=2.4,<3`; no optimizer is implemented from scratch.

If the exact resources are not already cached locally, the optional helper is an
**explicit network action**:

```bash
.venv/bin/python scripts/fetch_mni_template.py
```

It downloads only the pinned T1/mask from TemplateFlow's S3 resource paths, checks
SHA-256 before installing files, then verifies metadata. Existing matching files
are reused; mismatched files are rejected. It does not silently overwrite them.
The helper is never called by registration, imports, or tests. Normal runtime is
offline, and synthetic tests require neither network nor the real template.

For a new local configuration:

```bash
cp configs/registration.example.yaml configs/registration.local.yaml
export BRAINAGE_MNI_TEMPLATE="$PWD/artifacts/resources/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
export BRAINAGE_MNI_BRAIN_MASK="$PWD/artifacts/resources/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz"
.venv/bin/brainage-register-pilot --config configs/registration.local.yaml
```

Alternatively supply local relative paths in the ignored YAML. Paths resolve from
that YAML directory. Output defaults to ignored `artifacts/registration/rigid_pilot_v1/`.
A nonempty output directory is refused: choose a new run directory for an intentional
rerun or parameter study. Source MRI, source directories, templates and input manifests
cannot be overwritten or used as output directories. No IXI spreadsheet is written.

## Deterministic challenge-sample selection

Only Phase 1 `brain_age_label_eligible == True` subjects with existing files and
complete successful geometry diagnostics are considered. Eligibility is not recomputed.

Within each of Guys, HH and IOP:

1. Sort by maximum obliquity, then subject ID for ties.
2. Partition that ordered list into three contiguous thirds (low/middle/high).
3. Within each third select the largest Euclidean norm of physical bbox-center offset
   (ID breaks ties), then a distinct subject closest to the representative angle:
   lowest angle in the low third, median in the middle third, highest in the high third.
4. Reject insufficient site populations or duplicate subjects.

This yields exactly six subjects per site, eighteen total. Selection is independent
of manifest row order and does not use outcomes or optimize brain-age accuracy.
The generated selection manifest contains private IDs/paths and selection reasons;
no IDs are embedded in code or tracked documentation.

## Registration method and configuration rationale

Only SimpleITK `Euler3DTransform` parameters are optimized: three rotations and
three translations. `ComputeZYX=True` fixes rotation matrix order `Rz @ Ry @ Rx`.
No similarity scale, affine transform optimizer, shear parameter or deformable field
exists in this implementation.

[Nibabel RAS coordinates are converted to ITK LPS physical coordinates](https://simpleitk.readthedocs.io/en/master/registrationOverview.html)
by flipping physical X/Y signs, with the matching origin and direction matrix.
Array transposition only adapts Python x/y/z storage to SimpleITK z/y/x storage;
it does not resample. Native canonical voxel-to-world geometry, including tiny
header non-orthogonality, is retained rather than silently forced to an orthogonal
direction. This is existing acquisition geometry, not an optimized affine deformation.
The final rigid rotation is separately checked for orthonormality, singular values
all one, determinant +1, finite parameters, and exactly six free parameters.

Initialization uses `CenteredTransformInitializer(..., GEOMETRY)` to align **physical
field-of-view centers** and place the rotation pivot at the fixed FOV center. This
is a scanner-intensity-independent start when a trustworthy moving brain mask is
unavailable; it is not a claim that FOV centers are anatomically equivalent. A long
neck or uneven source FOV can make this initialization imperfect. Initial/final
parameters, centers, fixed parameters, homogeneous matrices, angles and translations
are all saved.

Baseline parameters, all explicit in the YAML/provenance:

| Parameter | Baseline | Rationale |
|---|---|---|
| Metric | Negative Mattes mutual information, 50 bins | T1 contrast can vary by site; lower metric is better |
| Metric sampling | RANDOM, 2%, seed 42 | Bound computation with reproducible stochastic samples |
| Fixed mask | Exact TemplateFlow brain mask | Reduce domination by template background/non-brain regions |
| Moving mask | None | Current non-background rule is not a brain mask; no skull stripping |
| Interpolator | Linear | Continuous T1 intensities |
| Pyramid | Shrink `[4,2,1]`, Gaussian sigma `[2,1,0]` mm | Coarse-to-fine capture and refinement in physical units |
| Optimizer | Regular-step gradient descent | Established SimpleITK implementation |
| Learning rate / min step | 2.0 / 0.001 | Physical-shift scaling makes angular/mm parameters comparable |
| Iterations | Up to 200 per level | Bounded pilot computation; exhaustion is flagged |
| Relaxation / gradient tolerance | 0.5 / 1e-6 | Explicit step reduction and gradient stopping controls |
| Optimizer scales | `SetOptimizerScalesFromPhysicalShift` | Rotations and translations have different units |
| Threads | 1 globally and per method | Remove thread-reduction variation alongside seeded sampling |
| Evaluation lattice | Fixed grid shrunk by 2 | Comparable deterministic full-lattice metric before/after |

Only the requested method family is supported; alternative initialization/optimizer
families are not automatically tried. Fixing seed and one thread provides repeatability
within the same inputs/configuration, library versions and platform, not guaranteed
bitwise reproducibility across different platforms/library releases.

SimpleITK estimates the **fixed/template LPS -> moving/source LPS pull transform**.
Reported Euler angles and translations use that convention; their signs are not
simply the apparent motion of the subject in an output figure. Euler translation
parameters depend on the recorded rotation center; the full homogeneous matrix is
also provided. See the [SimpleITK registration overview](https://simpleitk.readthedocs.io/en/master/registrationOverview.html)
for transform conventions, metric sign, physical scales, masks and reproducibility.

## Comparable metric evaluation and failure/outlier handling

Optimization values across pyramid levels/samples are not treated as directly
comparable. Before and after metrics are separately evaluated with the same metric,
mask and deterministic fixed evaluation lattice, against native moving intensities.
`metric_improvement = before - after` is positive for improvement in negative MI.
MetricEvaluate uses all evaluation-lattice points, not random optimizer samples.
This comparison still does not prove good anatomy: overlap and interpolated intensity
relationships change with the transform. Even an already-correct synthetic identity
can show a small MI degradation after multiresolution optimization; this is flagged,
not automatically reverted or declared invalid/valid solely by metric value.

Every subject retains explicit status, optimizer stop text/iterations, transforms,
outputs and warnings/errors. A technical completion is always
`completed_qc_review_required`, with `anatomical_qc_status=not_reviewed`.
Failures remain in manifests/provenance and do not stop subsequent subjects where safe.
A run with failures exits nonzero. Nothing is automatically excluded/deleted.

Provisional engineering warning limits (configurable, never biological thresholds):
rotation magnitude on any Euler axis >30 degrees, translation-vector norm >100 mm,
and native non-background source fraction outside the template FOV >15%.
Metric degradation and iteration-budget exhaustion are also warnings. Nonfinite/nonrigid
transforms or incorrect output geometry are explicit failures. Registration success
does not suppress these warnings.

Clipping is measured by mapping native source non-background voxel centers into
continuous template indices and counting those outside its voxel-cell FOV
`[-0.5, size-0.5)`. This counts head/neck/scalp/noise as well as intracranial tissue,
so it must never be described as a fraction of brain/anatomy lost. It is a conservative
FOV warning requiring visual review, not an automatic rejection rule.

## Output grid, interpolation and baseline comparison

Final output is sampled **directly from native canonical intensities onto the exact
selected template grid**: shape 193x229x193, 1 mm, RAS+, the template affine/origin/FOV.
It is not stretched to an array shape. Outside-source sampling is explicitly zero;
finite float32 intensities retain their source scale, without clipping/z-scoring here.

Each saved registration output has **one final linear interpolation from native data**.
Canonicalization is a permutation/flip with affine metadata changes and is not an
interpolation. Gaussian pyramids and metric sampling are optimization operations,
not repeated resampling of the final image. Baseline RAS+ -> 1 mm resampling and
initial-transform QC on the template are separate one-interpolation branches from
native data; neither is fed into the final registration image. No normalized image
is repeatedly resampled.

Per-subject intensity audits compare native, baseline, initialized-template and final
foreground mean/std, robust percentiles, zero fraction and foreground fraction.
Foreground is the same threshold-based non-background rule, not a moving brain mask.
Changes combine interpolation, sampled FOV and mask-population effects; do not
interpret them biologically. Baseline/final bbox and center-offset summaries and
native-FOV-exclusion fractions support crop feasibility review, but a consistent
output array size can hide clipped source tissue. No crop is implemented here.

## QC and artifacts

Private output layout:

- `pilot_selection.csv`: exactly eighteen selected IDs, sites, geometry and reasons.
- `subjects/*.json`, `manifest.csv`: full provenance, flags and failures.
- `volumes/*.nii.gz`: successful raw-scale registered intensities on the template grid.
- `transforms/*.tfm`: six-DOF fixed-to-moving Euler transforms in LPS coordinates.
- `qc/*.png`: orthogonal template, current baseline at its own FOV center,
  center-initialized moving image, final image, and fixed/final checkerboard.
- `summary.json`: aggregate metrics, rotations/translations, geometry, intensity and
  foreground comparisons, stop reasons, warnings and template/version provenance.

Panels label L/R neurological convention, anterior/posterior and superior/inferior.
Baseline has its own physical FOV center and is not overlaid as if already aligned;
initialized/final/checkerboard share the template grid/center. Full sagittal/coronal
fields and the displayed source-FOV-exclusion fraction help expose clipping.
Each panel's contrast window is independent. All figures are registration QC only,
with no diagnostic interpretation. QC-generation failures are explicit warnings.

Raw MRI, Phase 1 manifest, IXI.xls and the previous six-subject pilot are immutable.
Outputs and local configuration remain ignored. No model weights are created.

## Scientific boundaries and review decision

Orientation canonicalization changes axis convention. Resampling changes grid/spacing.
Rigid registration estimates translation/rotation for gross anatomical reference
correspondence; it does **not remove anatomical variability**. Affine registration
would also optimize scale/shear, and nonlinear registration would deform anatomy;
neither is implemented or evaluated in this task. An MNI template's construction
with nonlinear averaging does not make this subject-to-template method nonlinear.

After reviewing this pilot, choose A (proceed to final tensor-geometry evaluation),
B (evaluate tuning/initialization/masking), or C (experimentally compare affine).
Do not automatically tune, change masks, register the entire cohort, or freeze
crop/pad/slice positions based solely on successful computation or lower MI.
There is no brain-age MAE result because no model is trained.

See [the initial measured pilot results and recommendation](rigid-registration-pilot-results.md)
for quantitative outcomes, flags, interpolation audit and the decision to evaluate tuning.
