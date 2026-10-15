# Audit semantics and limitations

The manifest has one row for each valid normalized subject ID in the union of MRI
and demographic records. Unparseable MRI names are extra rows without subject IDs;
unparseable demographic IDs remain in `metadata_records.csv`, counted separately.
IDs normalize integer, zero-padded, and IXI-prefixed forms to `IXI###` (at least three
digits). Fractional, zero, negative, or ambiguous identifiers are rejected.

A match means both an identified MRI and demographic record exist. It says nothing
about uniqueness, age completeness, or image quality. `t1_paths` preserves every
scan, and `metadata_rows` points to original spreadsheet rows (header on row 1).
Every source column is retained as `source:<column>` in `metadata_records.csv`,
alongside normalized fields. The spreadsheet is never written to. Duplicate rows
are never dropped, and no ancillary values are combined into a synthetic row.
Resolved subject labels are separate derived values with source row provenance.
Duplicate MRI subjects similarly have no scalar `t1_path`.

Resolution statuses are `unique`, `duplicate_label_consistent`,
`duplicate_label_ambiguous`, `missing_age`, and `no_metadata`. Duplicate labels
compare normalized age, sex, DOB and study date exactly (no age tolerance).
Disagreement in any available label or invalid non-missing identity date makes
duplicates ambiguous. Invalid duplicate ages are also ambiguous; any missing age
prevents eligibility, even when another row has an age. Conflicts take priority
over `missing_age`, and all applicable exclusions remain recorded.

Missing sex/dates are absence of evidence, not disagreement. A canonical value
may only be taken from available agreeing source values; wholly missing fields
remain null. No age or date is inferred from other fields. Complete ISO dates and
spreadsheet datetime cells normalize to calendar dates; other string formats need
an explicit `date_format` with year/month/day. Numeric date serials and ambiguous
date strings are flagged invalid rather than assigned an epoch or locale.
Sex normalization only trims/case-normalizes strings and equates numeric forms
such as `1` and `1.0`; it does not interpret coding.

`label_conflict_fields` identifies conflicting labels; `ancillary_conflict_fields`
logs disagreements in all non-target source columns, including missing-versus-present
values and zero-coded values. These ancillary fields do not affect label resolution.
Metadata site/scanner values are exposed only when every row agrees, including
missingness. Filename `acquisition_site` recognizes Guys/HH/IOP only in the expected
ID-site-numeric acquisition-T1 structure; unknown or malformed sites remain null.
Scanner model/hardware is never inferred from these tokens.

Eligibility requires exactly one T1, successful MRI validation, demographic match,
valid ages in every contributing row, unique/consistent labels, and no invalid or
conflicting identity metadata. Absent validation is ineligible, not assumed valid.
All applicable `exclusion_reasons` are retained. Eligibility is a research curation
flag and does not establish a diagnosis, model validity, or leakage-free splits.

Age statistics and sex/site/scanner distributions use matched subjects with exactly
one demographic record, regardless of MRI validity. Valid ages must be numeric,
finite, and positive; this is a basic sanity check, not a cohort age threshold.
Missing and invalid ages are distinct. Duplicate metadata subjects are excluded
from demographic summary statistics. Global record-level age counts and
subject-level duplicate/missing counts are also reported explicitly.
These existing demographic summary denominators are preserved for comparability;
`valid_age_count` is not the newly derived eligible cohort size.
`brain_age_label_eligible_count` and `brain_age_label_ineligible_count` cover the union
of identified MRI/demographic subjects. Malformed MRI names and invalid metadata IDs
are recorded separately and cannot be eligible. Resolution counts use this same
subject union. Exclusion counts overlap because one subject can fail several criteria.
`acquisition_site_distribution` counts discovered MRI files, including ineligible
subjects, with unknown sites reported separately. It is distinct from the existing
metadata-based `site_distribution`.
Sex codes are preserved without assuming their meaning. Optional metadata fields
are not inferred from filename/site or scanner specifications.

Validation reports shape, dimensionality, spacing, affine sanity, coded-orientation
warnings, all finite/non-finite voxel counts, and finite-voxel min/max/mean/population
standard deviation (including background). A 3D structural T1 is expected.
The affine must be finite, homogeneous, and have a full-rank spatial transform;
spacing must be finite and positive. No clinical thresholds are introduced.
These checks do not establish tissue quality, absence of motion, acquisition
suitability, or pathology. No registration, segmentation, or clinical interpretation
is performed. Corrupt files are reported and processing continues.

Each file's scaled voxel array is materialized once through nibabel's data proxy,
avoiding repeated compressed-plane reads/seeks for `.nii.gz`. Statistics accumulate
over planes of this in-memory array using the existing float64 calculation. The
volume retains its native/scaling-derived dtype, preserving precision rather than
rounding float64 data or scaled values to float32. Only one MRI volume plus
plane-sized temporary arrays is held at a time, and the array is released when
validation returns. The audit retains only validation results, never voxel arrays.
Full validation scans all voxels; memory is bounded by one image's dimensions and
dtype, not by a fixed byte ceiling. Progress is flushed to stderr before each file;
the final stdout message includes total files, failures, and the output directory.
A malformed huge header can still cause costly work; only audit trusted local data.

The output order is deterministic for unchanged input files and environment.
Audit provenance records Python/package versions and resolved input paths/column
configuration, without wall-clock timestamps. It is not a cryptographic input
snapshot. Store your own private dataset version/checksums if needed for a study.
The source data is only opened for reading. Outputs overwrite fixed report names;
interruption during export can leave partial outputs, so check successful CLI exit.

References: [IXI dataset](https://brain-development.org/ixi-dataset/) and
[nibabel proxy slicing](https://nipy.org/nibabel/images_and_memory.html).
