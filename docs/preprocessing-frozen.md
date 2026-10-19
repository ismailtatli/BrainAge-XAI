# Frozen Phase 2 preprocessing

The authoritative [preprocessing specification](../configs/preprocessing.frozen.yaml) freezes the **full MNI152NLin2009cAsym resolution-01 template grid: 193 × 229 × 193, 1 mm isotropic, RAS+**. It records the physical affine, pinned template checksum, software versions and input membership checksums. There is no crop or pad. Model slices are not selected. This is research infrastructure, not a clinical tool.

## Cohort and leakage control

Read `brain_age_label_eligible == True` from the existing Phase 1 manifest; do not derive new eligibility. Exactly 561 subjects are expected. Consume the saved subject-level split without regeneration: train 393, validation 84, test 84, seed 42. The split manifest and its original summary are pinned by SHA-256. Verify unique subject identifiers, exact eligible/split set equality, matching source/site/age and fixed split counts before processing. No slice-level split is used.

## Frozen processing sequence

Validate the immutable native T1 and its source hash, then apply the existing closest-canonical orientation conversion without intensity interpolation. Call the unchanged S2/O0 implementation named and hashed in [registration.frozen.yaml](../configs/registration.frozen.yaml): GEOMETRY initialization, translation-only warm-up, full Euler3D rigid optimization, unmasked Mattes MI, RANDOM 2% sampling, seed 42, RegularStepGradientDescent and physical-shift scales. The 300-iteration-per-level budget and all other registration settings remain frozen. No affine or nonlinear registration is permitted.

Resample native canonical intensities directly to the template grid once with linear interpolation, float32 output and zero fill. Optimizer pyramid construction and metric interpolation are internal registration operations; they do not introduce a saved intermediate resampling. The template brain mask is a coverage/display reference only.

After registration, apply the existing validated `normalize_intensity` function independently to each subject. Foreground is absolute intensity greater than zero, including non-brain tissues; this is not skull stripping. Estimate 0.5th/99.5th percentiles on that subject's foreground, clip, and z-score using its clipped foreground mean and population standard deviation. Background stays exactly zero. No cohort-level normalization is fitted, so validation/test subjects cannot influence training normalization parameters. The normalization implementation is pinned by checksum.

Save float32 NIfTI with the full template affine, 1 mm spacing, millimeter units and MNI-coded qform/sform. Record source/output hashes, rigid transform, metric/coverage diagnostics, normalization statistics, split/site, runtime, warnings and explicit failure stage locally. Failure never changes membership.

## Execution and safe resume

```sh
.venv/bin/python -m brainage_xai.preprocessing.final --workers 4
```

The default output is ignored `artifacts/preprocessed/final_v1/`, containing volumes, subject provenance, cohort/manifest CSVs, failure CSV, aggregate summaries and QC. Four independent processes each run the frozen one-thread registration; no cross-subject optimization or normalization is introduced. The output directory is exclusively locked against concurrent writers.

```sh
.venv/bin/python -m brainage_xai.preprocessing.final --workers 4 --resume
```

A completed case is reused only after checking source/configuration/output hashes, the full saved array, finite/nonempty foreground, normalization statistics, exact template grid and coded orientation, background count and rigid-transform validity. A nonempty output directory requires `--resume`. Invalid/incomplete cases stop resume unless regeneration is explicit:

```sh
.venv/bin/python -m brainage_xai.preprocessing.final --workers 4 --resume --regenerate-invalid
```

Regeneration preserves old invalid files under an ignored recovery directory. Valid finished subjects are always skipped, including with this flag. Changed frozen inputs or source hashes prevent resume rather than silently accepting a different run. Verification without optimization is available:

```sh
.venv/bin/python -m brainage_xai.preprocessing.final --resume --verify-only
```

## QC and scientific limits

Every saved subject is reloaded and checked numerically. Summaries include success/failure, full-grid geometry, finite standardized foreground, preserved background, rigid determinant/orthogonality, coverage and warnings by split/site. Registration budget hits and contour differences are review signals, not automatic exclusions. FOV coverage is not anatomical correspondence. The documented rigid-versus-affine decision deliberately retains individual global morphology and accepts remaining template-correspondence differences.

A manageable visual sample uses six subjects per site, covering low/median/high original obliquity and previously challenging registration cases. Matched sagittal/coronal/axial template overlays and checkerboards are engineering QC views, not model slice selection or medical interpretation. Detailed participant notes remain ignored. No training is run by either command.
