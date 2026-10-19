# Phase 2 rigid optimizer study: predefined protocol

The exact 18 corrected-v3 subjects and unmasked GEOMETRY formulation are held fixed.
Only optimizer/convergence configuration varies. Prior v1, preliminary tuning and
corrected v3 outputs are read only. The local config is read for paths; corrected
v3 provenance supplies shared registration parameters. All three optimizers are
rerun on all subjects, with O0 checked against saved corrected-A results.

## O0 audit, before optimizer changes

Corrected A uses SimpleITK `RegularStepGradientDescent` with learning rate 2,
minimum step .001, relaxation .5, gradient magnitude tolerance 1e-6, and 300 outer
iterations per pyramid level. Its five-argument API call leaves learning-rate
estimation at `Never` and maximum physical step at 0 (no explicit cap). It does
not use windowed convergence: convergence minimum/window are not applicable.
Scales use `SetOptimizerScalesFromPhysicalShift` with default central radius 5
and small parameter variation .01. Both stages use this exact optimizer.

Translation stage: Euler3D weights [0,0,0,1,1,1], factors [4,2], physical smoothing
[2,1] mm. Rigid stage: Euler3D weights [1,1,1,1,1,1], factors [4,2,1], smoothing
[2,1,0] mm. Maximum outer iterations: 600 + 900 = 1,500. No scaling or shear.
O0 explicitly spells out the API defaults rather than changing their meaning.

## Three configurations, fixed before real v4 outcomes

| Setting | O0 | O1 | O2 |
|---|---|---|---|
| Family | RegularStepGradientDescent | GradientDescentLineSearch | RegularStepGradientDescent |
| Learning rate | 2 | 1 | 1 |
| Outer iterations per level | 300 | 300 | 300 |
| Minimum step | .001 | N/A | .0001 |
| Relaxation | .5 | N/A | .5 |
| Gradient magnitude tolerance | 1e-6 | N/A | 1e-6 |
| Convergence minimum | N/A | 1e-6 | N/A |
| Convergence window | N/A | 10 | N/A |
| Search lower / upper | N/A | 0 / 2 | N/A |
| Search epsilon | N/A | .01 | N/A |
| Maximum inner search iterations | N/A | 20 | N/A |
| Learning-rate estimation | Never | Once | Once |
| Maximum physical-step setting | 0 | 1 mm | 1 mm |
| Scale estimator | Physical shift (radius 5, variation .01) | Same | Same |

O1 retains documented convergence/search tolerances and narrows the default upper
search multiplier from 5 to 2 as a conservative global choice. O2 combines a
smaller configured rate, physically estimated rate and tighter terminal step.
These are optimizer configurations, not isolated single-hyperparameter effects;
no claims attribute an effect to just one parameter. No subject-specific tuning
or increases to the 300-iteration budget are allowed.

The physical-step parameter governs learning-rate estimation; it is not a hard
trust region on every line-search trial. O1 can evaluate the metric multiple times
inside each outer iteration. Its 300-iteration bound is therefore not equivalent
to 300 metric calls or equal computational cost. Actual runtime, level stops,
estimated scales and learning rates are recorded alongside outer event counts.

## Fixed scientific factors

The pinned TemplateFlow MNI T1, native canonical source values/grid, linear
interpolator, 50-bin Mattes MI, RANDOM 2% sampling, seed 42, one ITK thread,
initialization, stage pyramids and weights are identical. There are no fixed or
moving metric masks. The verified MNI brain mask is used solely for spatial
coverage and QC contours. Images are reconstructed into independent full buffers
to prevent the prior requested-region state problem; voxel values and physical
grids remain unchanged. Callback references are removed after each stage.

Negative Mattes MI is evaluated separately at full template resolution without
sampling at initialization, after translation, and after full rigid optimization.
`delta_translation = initial - after_translation`, `delta_rigid = after_translation
- final`, `delta_total = initial - final`; positive means improvement. These
values are distinct from sampled optimizer trajectories. Trajectory CSVs retain
outer iterations, levels, sampled metric, valid-point count and actual rate.
Compact per-stage/per-level first/best/final, best position and worsening-step
counts are included. Worsening steps are counted within a level only; sampled
values from different pyramid levels must not be treated as the same objective.

All transforms use fixed-template LPS to moving-source LPS pull mapping. Relative
rotation is the geodesic angle between rotation matrices; translation differences
are measured at the common fixed center. Euler-component deltas are also saved.
Prespecified engineering review flags are relative rotation >5 degrees or common
center pull-location disagreement >5 mm. These are provisional review triggers,
not anatomical acceptance thresholds; disagreement can mean one optimizer has
corrected another's poor solution. No subjects are excluded by these flags.

Template-brain in-bounds fraction counts fixed ROI centers in the moving physical
voxel-cell domain [-.5, size-.5). It measures FOV sample availability, not correct
anatomical alignment. There is no coverage acceptance threshold.

## Outputs, runtime and repeat

Output must be new and Git ignored. No permanent registered NIfTI collection is
created. Subject panels show fixed/O0/O1/O2, fixed brain contours, internal fixed
intensity edges and paired checkerboards at matched sagittal/coronal/axial planes.
Site and optimizer contact sheets retain only three planes per image. These
positions are diagnostic and do not freeze model slices or tensor geometry.

Case runtime includes initialization, three post-hoc metric/coverage evaluations,
both stages and final grid/finite checking/resampling; it excludes source loading,
PNG writing and table export. Stage runtime includes independent-buffer setup and
optimizer execution. Execute-only timings are also retained. Order is O0/O1/O2,
images are preloaded per subject, and one registration executes at a time. Runtime
is secondary to QC and is not an equal-work comparison between optimizer families.
Wall-clock measurements are descriptive: host load was not controlled, and
synthetic quality checks also ran while the comparison was in progress.

After reviewing technical validity, convergence, full MI, transform disagreement,
coverage and visual QC, select one global candidate and repeat it for all 18
subjects in the same environment. Repeated transforms, all three full MI values,
initialization, stage stops, sampled trajectories and weights are compared.
Exact equality is reported where attained; predeclared numerical equivalence is
absolute 1e-10 for transforms/full MI, with no relative tolerance. Runtime is not
expected to reproduce and is excluded from equality checks.

```sh
.venv/bin/python -m brainage_xai.registration.optimizer_study \
  --config configs/registration.local.yaml \
  --baseline artifacts/registration/rigid_pilot_v1 \
  --corrected artifacts/registration/rigid_tuning_v3 \
  --output artifacts/registration/rigid_optimizer_v4
```

The same CLI with `--repeat O1` (or the globally selected O0/O2) repeats the chosen
candidate into a new nested directory. It never overwrites prior case results.

API references: [SimpleITK registration methods](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ImageRegistrationMethod.html),
[registration overview](https://simpleitk.readthedocs.io/en/master/registrationOverview.html).
