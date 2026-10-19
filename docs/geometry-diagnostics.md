# Phase 2 geometry diagnostics: initial local results

This snapshot uses the existing 561 Phase 1 eligible subjects, the current RAS+
canonicalization, 1 mm isotropic linear resampling, and the unchanged baseline
non-background rule `abs(intensity) > 0`. All 561 succeeded with zero failures.
No MRI volumes were saved by this audit. The existing six-subject pilot (two per
site), including NIfTI/QC/provenance files, was preserved byte-for-byte and given a
separate diagnostic metadata supplement.

These are aggregate research QC observations, not medical interpretations.
Per-subject records remain private under the ignored
`artifacts/preprocessed/phase2/geometry_diagnostics/` output directory. See
[preprocessing methods and conventions](preprocessing.md#quantitative-geometry-and-non-background-foreground-diagnostics)
for formulas, numerical tolerances, coordinate conventions, and candidate policy.

## Obliquity and non-orthogonality

Maximum obliquity per subject, in degrees:

| Min | Median | Mean | p90 | p95 | p99 | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 4.33 | 5.13 | 9.73 | 13.28 | 13.80 | 18.11 |

| Strictly greater than (degrees) | Subjects |
|---:|---:|
| 0.1 | 554 |
| 0.5 | 543 |
| 1 | 512 |
| 3 | 363 |
| 5 | 247 |
| 10 | 53 |

At the documented 1e-5 absolute axis-cosine tolerance, eight affines have measurable
non-orthogonality, 546 are non-sheared oblique, and seven are effectively axis-aligned.
The eight are all from IOP. Median shear metric is 2.98e-9; maximum is only
1.6827e-5 (about 0.000964 degrees departure from a right angle). These extremely
small departures above the numerical tolerance are not evidence of important
physical shear, scanner defects, or a need for affine deformation. Obliquity and
position/FOV variability are the more substantial measurements here.

## Non-background foreground bounds

Axes below follow the resampled RAS directions. Sizes are voxel counts and also
whole-cell extents in mm at the audited 1 mm spacing. They are not brain dimensions:
skull/scalp/head tissue, neck and acquisition noise can contribute.

| Axis | Min | p50 | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| R | 155 | 191 | 207 | 212 | 222.4 | 236 |
| A | 227 | 249 | 270 | 273 | 279.0 | 300 |
| S | 178 | 248 | 272 | 282 | 287.4 | 300 |

Foreground fraction of the full resampled grid:

| Min | Median | Mean | p90 | p95 | p99 | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3230 | 0.6058 | 0.6032 | 0.7484 | 0.7812 | 0.8336 | 0.9519 |

## Position variability

Signed bbox-center offset from volume center (mm; equal numerically to voxels at
1 mm spacing). These are bbox centers, not anatomical centers or centers of mass.

| Axis | Min | p50 | Mean | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| R | -6.0 | 0.0 | -0.209 | 0.5 | 0.5 | 1.5 | 3.0 |
| A | -7.0 | 0.0 | -0.176 | 0.5 | 3.0 | 8.5 | 13.5 |
| S | -19.0 | 0.0 | 0.012 | 1.5 | 1.5 | 15.8 | 30.0 |

Absolute bbox-center voxel coordinates:

| Axis | Min | p50 | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| R | 77.0 | 96.5 | 105.5 | 108.5 | 113.7 | 122.5 |
| A | 117.0 | 126.0 | 141.0 | 148.0 | 156.1 | 165.5 |
| S | 110.0 | 125.5 | 140.0 | 142.5 | 146.4 | 150.5 |

## Diagnostic tensor candidates, not selected sizes

The coverage target is joint bbox-size fit; rounding can exceed the nominal target.
These diagnostics do not crop any image. R/A/S counts overlap, so their sum need
not equal the number of affected subjects. A centered crop can clip a bbox that fits
by size because of its position.

| Target / multiple | Diagnostic shape | Size-fit subjects | Too large R/A/S (any) | Centered-crop loss R/A/S (any) |
|---|---|---:|---|---|
| 95% / 8 | 216x280x288 | 537 / 561 | 19 / 3 / 5 (24) | 22 / 14 / 24 (51) |
| 95% / 16 | 224x288x288 | 552 / 561 | 4 / 1 / 5 (9) | 5 / 2 / 24 (28) |
| 99% / 8 | 232x280x296 | 557 / 561 | 1 / 3 / 1 (4) | 1 / 14 / 6 (19) |
| 99% / 16 | 224x288x304 | 556 / 561 | 4 / 1 / 0 (5) | 5 / 2 / 0 (7) |
| 100% / 8 | 240x304x304 | 561 / 561 | 0 / 0 / 0 (0) | 0 / 1 / 0 (1) |
| 100% / 16 | 240x304x304 | 561 / 561 | 0 / 0 / 0 (0) | 0 / 1 / 0 (1) |

Even the 100% size-fit candidate loses baseline foreground in one subject under
volume-centered cropping. A larger window could be constructed for containment,
but that would not establish anatomical correspondence.

## Site associations

Site is filename-derived acquisition site, not an inferred hardware model. These
are observational associations; they do not establish scanner/site causation.

| Site | Subjects | Obliquity min / median / mean / p90 / p95 / p99 / max (degrees) |
|---|---:|---|
| Guys | 313 | 0.00 / 4.70 / 5.78 / 12.66 / 13.50 / 14.27 / 18.11 |
| HH | 180 | 0.00 / 5.36 / 5.20 / 8.25 / 8.63 / 9.07 / 11.22 |
| IOP | 68 | 0.23 / 1.43 / 1.99 / 4.01 / 5.44 / 7.68 / 8.54 |

Ranges below are per-axis minima to per-axis maxima; extrema can come from different
subjects, so the displayed vectors are not necessarily observed shapes.

| Site | Resampled shape range | Foreground bbox-size range |
|---|---|---|
| Guys | [156,241,241] to [244,314,302] | [155,227,221] to [236,300,300] |
| HH | [180,241,241] to [227,284,287] | [179,239,239] to [224,280,278] |
| IOP | [176,241,241] to [220,257,264] | [174,235,178] to [206,248,256] |

Signed center offsets in mm:

| Site | Mean R/A/S | Median R/A/S | p95 R/A/S | Min R/A/S | Max R/A/S |
|---|---|---|---|---|---|
| Guys | [-0.276,0.288,-0.752] | [0,0,0] | [0.7,4.9,1.0] | [-3,-7,-19] | [3,13.5,3.5] |
| HH | [-0.083,-0.881,0.514] | [0,-0.5,0.5] | [0,0,1.5] | [-0.5,-6.5,-1] | [0.5,0,2] |
| IOP | [-0.228,-0.449,2.206] | [0,-0.5,0] | [1,0.825,17.65] | [-6,-3,-8.5] | [2.5,2,30] |

Full per-site percentile summaries are retained in the private JSON. Guys has the
broadest output-grid/obliquity range, whereas IOP has the largest positive superior
bbox-center tail despite lower median obliquity. This reinforces that obliquity and
non-background position are distinct diagnostics.

## Recommendation: evaluate rigid registration before freezing geometry

Recommend **B: evaluate subject-to-common-template rigid registration and anatomical
overlay/landmark QC before choosing final model-space geometry**. Do not implement
registration or freeze any crop/pad target from this snapshot.

Evidence includes maximum obliquity up to 18.11 degrees, site-dependent geometry,
signed foreground-center offsets reaching -19/+30 mm on the superior axis, and
predicted centered-crop losses in 19 subjects for the 99%/8 size candidate and one
subject even for a 100%-by-size candidate. Size-only containment is therefore not a
sufficient justification for a common centered crop.

This does not prove rigid registration is necessary or sufficient. Broad baseline
foreground includes non-brain structures/noise, and these measurements do not assess
anatomical voxel-wise correspondence. Individually foreground-centered crop/pad or
larger windows might avoid these particular losses, but would need separate QC and
cannot by themselves establish alignment. The tiny non-orthogonality measurements
do not justify affine or nonlinear registration. No final geometry is selected.
