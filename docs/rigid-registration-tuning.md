# Controlled rigid tuning experiment

This experiment reuses the exact saved 18-subject v1 cohort (six per site), without
selection based on outcomes. Baseline files are read only, hashed before/after,
and never rewritten. It remains an Euler3D experiment; there is no moving mask,
skull stripping, affine/nonlinear registration, model training, or final geometry
selection.

Run from the repository root with the existing local resource configuration:

```sh
.venv/bin/python -m brainage_xai.registration.tuning \
  --config configs/registration.local.yaml \
  --baseline artifacts/registration/rigid_pilot_v1 \
  --output artifacts/registration/rigid_tuning_v3
```

Outputs must be empty and Git ignored. A completed directory cannot be reused.
The CLI uses the template/source settings in the existing config, explicitly
replacing the v1 iteration budget with 300 iterations per level and full-resolution
post-hoc evaluation. It does not edit the config. The resulting parameters are
recorded in `provenance.json`.

| Condition | Initialization | Fixed metric mask |
|---|---|---|
| A | GEOMETRY | None |
| B | GEOMETRY | Verified MNI brain mask |
| C | MOMENTS | None |
| D | MOMENTS | Verified MNI brain mask |

All four use native canonical moving data, the same verified MNI template,
linear interpolation, 50-bin Mattes MI, RANDOM sampling at 2%, explicit seed 42,
one ITK thread, physical-shift optimizer scales and RegularStepGradientDescent.
The shared learning rate is 2, minimum step .001, relaxation .5 and gradient
magnitude tolerance 1e-6. Stage 1 uses shrink factors 4,2, smoothing 2,1 mm and
weights 0,0,0,1,1,1. Stage 2 continues from that transform with factors 4,2,1,
smoothing 2,1,0 mm and all six weights 1. Each level permits 300 iterations:
a bounded maximum of 1,500 per condition versus v1's maximum 600. No per-subject
budget changes or optimization retries occur. One first-subject A repeat is solely
a reproducibility test and does not replace the original result.

The fixed mask is binary and must match size, spacing, origin and direction.
GEOMETRY aligns physical FOV centers; MOMENTS uses intensity moments and may be
biased by scalp, neck and background. Initial centers, rotations, translations,
full metrics and ROI coverage are retained for both modes.

`MetricEvaluate` separately evaluates the full fixed resolution without sampling
at initialization, after translation, and after rigid optimization. Lower negative
Mattes MI is better; `within_condition_MI_change = MI_initial - MI_final`, so
positive values indicate improvement. Masked and unmasked absolute MI magnitudes
represent different domains and must not be used to rank conditions. Baseline
transforms are reevaluated at full resolution in the new output, because original
v1 metrics used a reduced lattice. The baseline has no translation-only stage.
Sampled optimization trajectories, valid-point counts, per-level stop descriptions
and per-stage iteration counts are recorded independently. `MetricEvaluate` does
not retain its metric object for a reliable valid-point-count query, so that count
is explicitly null in post-hoc evaluations.

Transforms map **fixed-template LPS to moving-source LPS**, as required for pull
resampling. Euler translation parameters are not direct anatomical displacement.
Coverage maps each fixed brain ROI voxel center through that transform into the
moving continuous-index domain, using voxel-cell bounds [-.5, size-.5). It reports
inside/outside counts and the in-bounds fraction at all three stages. This measures
FOV availability, not alignment; no coverage acceptance threshold is invented.
Native non-background exclusion remains descriptive head/neck/noise context,
never a brain-loss or subject-brain-mask measure.

Registered images are resampled directly from the native canonical source in
memory. Only central sagittal/coronal/axial planes are retained for contact sheets;
no 72-volume NIfTI collection is written. Every comparison uses matched template
positions and fixed ROI contours. Panels and contact sheets support assessment of
rotation, translation, superior/inferior clipping, cerebellar, cortical and left/right
mismatch. Technical success never confers anatomical acceptance.

Private outputs include per-case JSON, a subject-by-condition CSV including v1,
per-subject comparison PNGs, site/condition contact sheets, repeatability evidence,
and aggregate/site-stratified metrics, stop counts, transform quantiles and coverage.
Participant identifiers, source paths and images stay under ignored artifacts.

Synthetic tests exercise both initializers, mask compatibility, weighted stage
transition, reproducibility, deterministic full metrics, pull coverage, rigidity,
comparison rows and output safety. No participant data are used in tests.

API references: [SimpleITK optimizer weights and registration methods](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ImageRegistrationMethod.html),
[registration overview and reproducibility](https://simpleitk.readthedocs.io/en/master/registrationOverview.html).

The preliminary v2 run is retained privately and excluded from scientific comparison.
A consistency check found that ITK QC resampling left a requested subregion on
reused source images; subsequent MOMENTS initialization then used that subregion
instead of the full native image. Source voxel values were unchanged. The corrected
runner reconstructs independent full-buffer images with identical origin, spacing
and direction for initialization, metric evaluation and each optimization stage.
A synthetic regression test reproduces the preceding-resampling scenario, and final
report generation rejects differences between A/B or C/D initialization transforms.
All four conditions are repeated in v3, with v1 still preserved byte-for-byte.

Corrected cohort results and the single scientific recommendation are in
[rigid-registration-tuning-results.md](rigid-registration-tuning-results.md).
An additional real-subject C repeat after QC resampling reproduced the full trace
and transform exactly and preserved full-image MOMENTS initialization.
