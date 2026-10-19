# Phase 2: MRI preprocessing and subject preparation

This is research infrastructure. It performs no diagnosis, age prediction, BAG
calculation, attribution, or model training. Source MRI and demographic files are
immutable. Generated NIfTI, QC previews, and participant manifests remain private.

## Input contract and configuration

Copy `configs/preprocessing.example.yaml` to the ignored
`configs/preprocessing.local.yaml`. Paths resolve against the YAML directory;
`~` and environment variables are supported, with unset variables rejected.
The default manifest is the local Phase 1 `reports/generated/ixi_audit/manifest.csv`.
No eligibility rule is recomputed: only explicit true flags enter Phase 2.
The manifest must have `subject_id`, `t1_path`, `age`, `acquisition_site`, and
`brain_age_label_eligible`. Malformed flags, duplicate eligible IDs, unsafe IDs,
invalid eligible ages, absent paths, nonexistent NIfTI files, or one source file
assigned to multiple eligible IDs fail before any preprocessing. Missing site
is represented as `unknown`; it is never inferred again. Relative T1 paths resolve
against the manifest's directory.

The example config sets an exploratory 1 mm isotropic spacing, 0.5/99.5 foreground
clipping percentiles, and an absolute background threshold of zero. None is a
biological or diagnostic threshold. Spacing is required in configuration, not fixed
inside the algorithms. Ratios, seed, predefined age-bin width, and QC settings are
also configurable. No final tensor shape or final research resolution is frozen.

## Spatial operations

`orientation.py` validates non-empty 3D geometry, finite/full-rank affine, positive
spacing, and coded orientation. Missing qform/sform is rejected rather than silently
using a fallback. Known non-mm spatial units require explicit conversion; unknown
units are assumed mm with a provenance warning. The IXI pilot's source units are mm.

Canonicalization uses `nibabel.as_closest_canonical`: axis permutations/flips update
the affine together with the image data. Nearest RAS+ axis codes do not remove
obliquity. This is neither anatomical alignment between subjects nor nonlinear
registration to a template.

`resampling.py` builds an axis-aligned RAS+ physical grid covering the source voxel
centers with nibabel's `vox2out_vox`. `resample_from_to` then uses **linear**
interpolation of float32 continuous T1 intensities with zero outside the source FOV.
Integer input is converted before interpolation to prevent quantization. Output
voxel spacing and affine describe the new physical grid; there is no stretching to
an arbitrary array shape. Resampling does not create new anatomical information.
There is no anti-aliasing filter beyond interpolation: coarser target resolutions
would require a separate scientific review of downsampling and smoothing.

The spatial audit predicts the *same* canonical/resampled grid directly from header
shape, orientation transform, and affine. It never materializes a voxel array or
writes preprocessed MRI. Tests compare predictions with actual oblique resampling
and forbid array-proxy access during the audit. Header-only success is not renewed
validation of voxel integrity: the Phase 1 audit supplies that validation, and the
full preprocessing path checks intensities again.

## Per-subject intensity normalization

`intensity.py` defines foreground as `abs(intensity) > background_threshold`, clips
foreground at configurable percentiles, then computes foreground mean/population
standard deviation and z-scores those clipped values. Background remains exactly
zero. Every subject fits its own statistics; no cohort or test-subject statistics
are used. The input array is not mutated. NaN/Inf, empty foreground, and constant or
numerically degenerate foreground fail explicitly rather than returning NaNs.

This baseline mask is **not skull stripping, tissue segmentation, or a brain mask**.
It can include scalp and acquisition noise, particularly after interpolation.
Review the pilot before choosing a foreground threshold, masking strategy, or a
final interpolation/resolution policy. Z-scored foreground may legitimately contain
zeros; do not later infer a brain mask from normalized nonzero values alone.

## Commands and outputs

From an installed Python 3.11 environment:

```bash
cp configs/preprocessing.example.yaml configs/preprocessing.local.yaml
brainage-preprocess-audit --config configs/preprocessing.local.yaml
brainage-preprocess --config configs/preprocessing.local.yaml --limit 6
brainage-split --config configs/preprocessing.local.yaml
```

`--limit` selects a deterministic pilot by cycling through sorted sites and sorted
subject IDs, so six subjects include two from each IXI site when available. It limits
work after eligibility filtering and never changes label eligibility. Without a
limit, `brainage-preprocess` processes the whole eligible manifest; **do not run that
until the pilot and spatial audit are reviewed**. Existing processed NIfTI files are
protected by default; `--overwrite` explicitly replaces derived outputs only.

Default output root: `artifacts/preprocessed/phase2/` (ignored).

- `spatial_audit/summary.json`: aggregate shape/spacing/orientation distributions and ranges.
- `spatial_audit/manifest.csv`, `spatial_audit/subjects/*.json`: private header provenance.
- `preprocessing/summary.json`, `preprocessing/manifest.csv`, `preprocessing/subjects/*.json`:
  private pilot/preprocessing records, including failures and normalization statistics.
- `volumes/*.nii.gz`: float32 derived NIfTI with the new physical affine and coded forms.
- `qc/*.png`: canonical-versus-processed orthogonal previews.
- `splits/split_manifest.csv`: subject ID, age, site, split, and source path; private.
- `splits/summary.json`: split counts and aggregate ages/sites, seed, ratios and algorithm.

Source shape/spacing/orientation, source/canonical/resampled affine, target shape and
spacing, normalization parameters, output/QC paths, warnings, and errors are recorded
per subject. Run summaries record Python/package versions and the effective config.
Files are handled one subject at a time. MRI saves use temporary files and atomic
replacement; record/report writes are not transactional. Failed subjects remain in
provenance and processing continues. A run with processing failures exits nonzero.
QC failures are warnings with no QC path, while a valid processed volume remains a
success; inspect `qc_success_count` as well as processing status.

Reports at fixed paths are replaced on rerun. Per-subject files for earlier,
different subsets may remain, so the **current manifest** determines the current
run population; directory file counts are not a substitute. Keep separate output
roots for alternative parameter studies. Output inside Git must be ignored, and
source directories/manifest input directories cannot serve as output directories.

QC rows use planes through the same source-center world coordinate and physical
voxel aspect. Source oblique planes can differ from the axis-aligned processed
planes. Each panel uses its own display percentiles; this is visual geometry and
artifact inspection, not intensity calibration or medical interpretation.

## Subject-level splits and leakage

`splits.py` sorts eligible subject IDs, seeds NumPy's generator, and forms strata
from acquisition site and fixed-width age bins (`floor(age / age_bin_width)`).
Largest-remainder rounding sets exact global train/validation/test capacities
(approximately 70/15/15). Each stratum first allocates `floor(n_stratum * ratio)`
shuffled subjects to each split. Remaining subjects enter a single seeded pool and
are assigned by remaining global capacities. This preserves a proportional core in
large strata while safely handling singleton/small strata without duplication or
exceptions. Very small strata cannot be represented in every split; preservation
is approximate, and summary ages/sites must be reviewed. No demographic values are
imputed to fill strata. Sorting makes results independent of manifest row order.

Every eligible subject appears exactly once and no subject crosses splits. Splits
are assigned **before future slice extraction**; all slices/repeats of a subject
must inherit its split. This Phase 2 manifest already enforces one T1 per eligible
subject. Future fitted cohort transforms, bias correction, feature selection, and
models must use training data only. Per-subject normalization is independent and
can be performed before or after split assignment without learning cohort statistics.
Processing failures do not silently alter the planned split cohort: downstream
training preparation must join preprocessing status, investigate failures, and
explicitly version any revised cohort rather than silently dropping failed subjects.

## Pending scientific decisions

Review all spatial distributions and pilot previews before choosing crop/pad size.
No crop/pad or fixed tensor shape is implemented. Evaluate foreground/background
handling, acquisition noise, obliquity/interpolation effects, downsampling policy,
and possible future masking/registration in a separate review. No final anatomical
QC decision is automated by successful preprocessing.

## Validation

```bash
python -m pytest -q
ruff check .
python -m compileall -q src
python -m pip check
```

The test suite uses synthetic temporary NIfTI/data only, including axis-permuted
representations, oblique geometry, finite/outlier/degenerate intensities, rare split
strata, eligibility failures, provenance, QC, and deterministic subsets.

## Initial local validation snapshot

Using the existing Phase 1 eligible population, exploratory 1 mm isotropic spacing,
linear interpolation, example intensity parameters, and split seed 42:

- Header-only audit: 561 eligible subjects, 561 successes, zero failures.
- Source orientation: PSR for all 561; canonical orientation: RAS for all 561.
- Source units: mm for all 561. Exact spacing triples vary slightly in stored precision;
  rounded to four decimals, 557 are `(0.9375, 0.9375, 1.2000)` and four are
  `(0.9766, 0.9766, 1.2000)` mm in the original voxel-axis order.
- Exact original spacing ranges, axis by axis: approximately
  `[0.937486, 0.937486, 1.199993]` to `[0.976563, 0.976563, 1.200008]` mm.
- Canonical shapes: `150x256x256` (489), `146x256x256` (68), `130x256x256` (2),
  and `140x256x256` (2).
- Predicted resampled shapes: 489 distinct; axis-wise minima `[156, 241, 241]`,
  maxima `[244, 314, 302]`. These are per-axis bounds, not one observed shape.
  Most frequent: `179x241x243` (14), `180x241x241` (8), `182x241x245` (5).
- Oblique/sheared source geometry warnings: 555 subjects. This contributes to output
  shape diversity and is not by itself a preprocessing failure.
- Full-volume pilot: exactly six subjects, two per acquisition site; six successful
  derived NIfTI volumes, six QC comparisons, zero preprocessing/QC failures. All six
  have oblique/sheared geometry warnings. Other subjects have no processed volumes.

Splits cover all 561 eligible subjects exactly once. Ages below are in years and
rounded for readability; machine-readable private summaries retain full precision.

| Split | Subjects | Age mean | Median | Min | Max | Guys | HH | IOP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 393 | 48.48 | 48.25 | 19.98 | 86.20 | 221 | 126 | 46 |
| Validation | 84 | 48.31 | 46.04 | 22.53 | 81.94 | 45 | 28 | 11 |
| Test | 84 | 49.58 | 50.21 | 20.70 | 86.32 | 47 | 26 | 11 |

This aggregate snapshot contains no subject identifiers or participant paths.
The full shape distributions and per-subject provenance stay under the ignored
output root. These results support review, not a claim that a final tensor size,
foreground mask, or anatomical alignment policy has been selected. In normalized
QC panels, background value zero can appear gray under the panel's display window;
voxel value zero is preserved numerically and is not a tissue label.

## Quantitative geometry and non-background foreground diagnostics

Use the same local configuration and target spacing:

```bash
brainage-geometry-audit --config configs/preprocessing.local.yaml
```

This command canonicalizes and resamples **one eligible subject at a time in
memory**, computes measurements, and releases the arrays. It saves **no MRI volumes**,
performs no normalization, and does not crop or pad any subject. The foreground is
measured on resampled intensities before clipping/z-scoring, using the *same shared
function* as normalization: `abs(intensity) > background_threshold`. No brain mask
or skull stripping is performed: skull, scalp, other head tissue, neck and low-level
acquisition noise can contribute. This conservative baseline foreground must not be
interpreted as an anatomical brain extent.

Private outputs under `artifacts/preprocessed/phase2/geometry_diagnostics/`:

- `subjects/*.json` and `manifest.csv`: every subject's geometry, foreground extents,
  position measurements, status, warnings and errors, including failures.
- `summary.json`: full-population and site-specific quantitative summaries and
  diagnostic tensor candidates. Successful geometry and foreground measurements
  have separate denominator counts when a later foreground/resampling step fails.
- `pilot_diagnostics.json`: a supplementary diagnostic record for the existing pilot
  subject IDs, without modifying its NIfTI, QC, manifest, summary or provenance files.
  Site counts and the number of diagnosed pilot subjects are explicit.

The header-only `brainage-preprocess-audit` also adds quantitative affine summaries
under `affine_geometry`. It remains header-only. The old combined oblique/sheared
warning is replaced for new runs by separate `oblique_source_resampled_to_axis_aligned_grid`
and `non_orthogonal_source_affine` warnings. Historical pilot records retain their
original warning text; their diagnostic supplement provides the new measurements.

### Obliquity versus non-orthogonality

Obliquity angles use `nibabel.affines.obliquity`, converted from radians to **degrees**.
Its output corresponds to the closest voxel-axis direction for each world R/A/S
axis. Separately named source-voxel-axis angles are also reported, so original axis
permutations cannot silently change the interpretation. Maximum angle histograms,
min/median/mean/p90/p95/p99/max, per-axis statistics, and strict `>` threshold counts
at 0.1/0.5/1/3/5/10 degrees are QC descriptors, not biological thresholds.

For affine spatial columns `A`, let `U` be each column divided by its Euclidean
length. The non-orthogonality metric is the largest absolute off-diagonal entry of
`U.T @ U`, i.e. the largest absolute cosine between distinct voxel-axis directions.
A rotation with orthogonal axes can be highly oblique with **zero shear metric**.
Reflections and anisotropic voxel scaling are not themselves shear. The determinant
sign and affine-derived voxel sizes are retained independently.

`has_measurable_shear` means the metric is **greater than 1e-5**. This dimensionless
numerical tolerance corresponds to about 0.000573 degrees of departure from a right
angle for small deviations and avoids labeling ordinary float32 header noise as
meaningful shear. It is a numerical/QC choice, not proof of a physical scanner defect.
"Effectively axis aligned" means non-sheared and maximum obliquity <= 0.1 degrees.
The other disjoint categories are non-sheared oblique and measurable non-orthogonality.
These tolerances and histogram definitions are saved in the summaries.

### Foreground bounds, position, and tensor candidates

Bounding-box starts are **inclusive**, ends **exclusive**, and sizes `end - start`.
Physical extent is `size * resampled_voxel_spacing`: the coverage of whole voxel
cells, not the center-to-center span `(size - 1) * spacing`. Bbox center is
`(start + end - 1) / 2`, not center of mass or anatomical head/brain center. Volume
center is `(full_shape - 1) / 2`. Signed offsets are bbox center minus volume center,
with physical offsets along resampled RAS axes in mm. Absolute voxel center positions
also vary when full shapes differ, even if signed offsets are small.

Summaries include min/p50/p90/p95/p99/max and mean, using NumPy linear percentiles.
For candidate 95/99/100% coverage, dimensions are rounded upward to multiples of
both 8 and 16. A shared rank in independently sorted axis sizes is increased until
the requested **joint three-axis empirical coverage** is attained. This avoids
mislabeling independent 95th-percentile dimensions as 95% joint coverage. No optimal
or final shape is claimed, and marginal extrema need not belong to the same subject.

Every candidate reports:

- `bbox_larger_than_candidate_axis_counts` and `_any_count`: subjects that cannot
  fit by size, even with individually chosen translations.
- `bbox_joint_coverage_count/fraction`: simultaneous three-axis fit by size.
- `center_crop_foreground_loss_axis_counts` and `_any_count`: subjects whose baseline
  foreground would actually intersect a boundary of a common **volume-centered**
  crop/pad window, including positional offsets. The diagnostic window starts at
  `floor((full_shape - candidate_shape) / 2)`; negative starts represent padding.
- `center_crop_joint_coverage_count`: subjects with no predicted foreground clipping.

Only boundary intersections are counted, not the number of foreground voxels lost.
No pixels are cropped by this audit, and a size that contains every non-background
bbox still does not establish anatomical correspondence between subjects. Candidate
sizes are diagnostics of the current population and parameters, not selected model
input dimensions or fitted cohort intensity statistics.

### Canonicalization, resampling, and registration are different operations

RAS+ canonicalization only reorders/flips axes while preserving physical coordinates.
Isotropic resampling changes sampling and interpolation on those physical coordinates.
**RAS+ canonicalization and 1 mm resampling do not guarantee anatomical voxel-wise
alignment between subjects.**

Rigid registration estimates relative rotation and translation to a shared reference
without scale/shear deformation. Affine registration additionally permits global
scale and shear; nonlinear registration permits spatially varying deformation.
None is implemented here. Obliquity alone does not prove anatomical misalignment:
scan prescription, head pose and scanner coordinates can all contribute, and broad
non-background bounds may be dominated by noise or non-brain tissue. The measurements
inform whether common-template rigid registration and landmark/overlay QC should be
evaluated before model-space geometry is frozen; they cannot replace anatomical QC.

See [the initial measured diagnostics and recommendation](geometry-diagnostics.md) for
aggregate obliquity, bounds, position, candidate coverage, and site results.
