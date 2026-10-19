# Final registration decision protocol (v6)

This is the final Phase 2 registration experiment. Reuse the exact saved eighteen-subject S2/O0 cohort (six each Guys, HH, IOP) and final rigid transforms from v4. No rigid optimization or baseline MI reevaluation is permitted. Existing S2 repeatability evidence is retained. All new artifacts belong in the ignored `artifacts/registration/rigid_vs_affine_v6/`; a nonempty output directory is rejected. Earlier experiments, source images, metadata, templates and local configurations are read-only.

## One predefined affine configuration

Each final centered Euler3D mapping is converted to an affine by copying its center, matrix and translation. This preserves every physical point mapping. Run exactly one twelve-parameter affine optimization per subject, on the original native canonical moving image and existing MNI152NLin2009cAsym resolution-01 template. There is no new translation warm-up or rigid stage. The affine stage uses:

- Mattes MI, 50 histogram bins, RANDOM sampling 0.02, seed 42, one ITK thread.
- Linear interpolation, pyramid shrink factors `[4, 2, 1]`, smoothing `[2, 1, 0]` mm.
- O0 RegularStepGradientDescent: learning rate 2, minimum step 0.001, relaxation 0.5, gradient tolerance 1e-6, 300 outer iterations per level, learning-rate estimation Never, maximum physical step argument 0.
- All twelve optimizer weights enabled; `SetOptimizerScalesFromPhysicalShift(5, 0.01)`.
- No metric masks, skull stripping, nonlinear mapping or parameter search.

The affine initialization already contains the complete rigid mapping. Native intensities are resampled directly onto the template once for QC, without resampling a rigid intermediate. The template brain mask is used only for display and FOV coverage. Final full-resolution unsampled negative Mattes MI is evaluated separately. A positive paired MI gain is stored rigid final MI minus affine final MI; lower negative MI is better. Affine starting MI is the stored rigid final MI.

Record optimizer events, per-level stops, iteration counts, physical-shift scales, budget hits, runtimes, exact output grid and brain-ROI voxel-cell FOV coverage. Rigid runtime is historical; affine runtime is incremental. Their sum estimates the future rigid-plus-affine pipeline cost, with uncontrolled host-load differences.

## Deformation diagnostics and review policy

Transforms map fixed-template LPS points into native-moving LPS coordinates (a pull map). Record the full 3×3 linear matrix, centered translation, determinant, singular values, deviations from one, cube-root-determinant global pull scale, normalized column Gram matrix, polar stretch and normalized off-diagonal polar shear. The proper polar rotation is recorded separately; Euler angles of a nonorthogonal matrix are not treated as rigid angles. The positive determinant is the pull volume factor; its reciprocal is the moving-to-fixed volume factor. Neither is interpreted biologically.

Nonfinite, near-singular (minimum singular value below 1e-8), or orientation-reversing transforms fail validation. Before real outcomes, the following provisional engineering review triggers are fixed: any principal pull scale outside `[0.8, 1.25]`, determinant outside `[0.7, 1.4]`, or absolute inter-column cosine above `0.15`. These are visibility prompts, not validated scientific acceptance thresholds. Changes below these limits may still be concerning for preserving morphology. Scale and shear can alter age-relevant information; an MI gain does not establish scientific suitability.

## Visual review and final choice

Review all eighteen matched sagittal/coronal/axial panels, showing MNI, rigid and affine with template brain contours, template intensity edges and checkerboards. Review cortical, ventricular where visible, cerebellar and superior/inferior correspondence, gross pose, clipping/FOV and obvious stretching. Summarize each site separately. Center slices cannot establish correspondence over the whole brain; FOV coverage is not anatomical correctness.

Choose **AFFINE** only for clear, consistent spatial-correspondence improvement across sites without concerning scale/shear or technical regression. Otherwise choose **RIGID** and freeze existing S2. Close registration tuning with exactly one method; do not propose another tuning experiment. Freeze registration alone. Crop/slice/tensor geometry, all-subject preprocessing and training remain deferred.

## Commands

```sh
.venv/bin/python -m brainage_xai.registration.affine_study \
  --output artifacts/registration/rigid_vs_affine_v6
```

Do not rerun after completion. Derived numerical tables can be regenerated without registration:

```sh
.venv/bin/python -m brainage_xai.registration.affine_study \
  --output artifacts/registration/rigid_vs_affine_v6 --finalize-only
```

Tests use synthetic temporary images only.
