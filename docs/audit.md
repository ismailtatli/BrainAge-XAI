# Audit semantics and limitations

The manifest has one row for each valid normalized subject ID in the union of MRI
and demographic records. Unparseable MRI names are extra rows without subject IDs;
unparseable demographic IDs remain in `metadata_records.csv`, counted separately.
IDs normalize integer, zero-padded, and IXI-prefixed forms to `IXI###` (at least three
digits). Fractional, zero, negative, or ambiguous identifiers are rejected.

A match means both an identified MRI and demographic record exist. It says nothing
about uniqueness, age completeness, or image quality. `t1_paths` preserves every
scan, and `metadata_rows` points to original spreadsheet rows (header on row 1).
Duplicate metadata subjects have empty scalar demographic fields in the manifest;
all original normalized records remain available. No conflicting age is selected.
Duplicate MRI subjects similarly have no scalar `t1_path`.

Age statistics and sex/site/scanner distributions use matched subjects with exactly
one demographic record, regardless of MRI validity. Valid ages must be numeric,
finite, and positive; this is a basic sanity check, not a cohort age threshold.
Missing and invalid ages are distinct. Duplicate metadata subjects are excluded
from demographic summary statistics. Global record-level age counts and
subject-level duplicate/missing counts are also reported explicitly.
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

Voxel values are read as scaled float64 2D planes through nibabel's data proxy.
Statistics accumulate over those planes, without caching the full volume. For gzip
inputs repeated slices can cost extra decompression time; plane memory is bounded
by image dimensions, not by a fixed byte ceiling. Full validation scans all voxels.
A malformed huge header can still cause costly work; only audit trusted local data.

The output order is deterministic for unchanged input files and environment.
Audit provenance records Python/package versions and resolved input paths/column
configuration, without wall-clock timestamps. It is not a cryptographic input
snapshot. Store your own private dataset version/checksums if needed for a study.
The source data is only opened for reading. Outputs overwrite fixed report names;
interruption during export can leave partial outputs, so check successful CLI exit.

References: [IXI dataset](https://brain-development.org/ixi-dataset/) and
[nibabel proxy slicing](https://nipy.org/nibabel/images_and_memory.html).
