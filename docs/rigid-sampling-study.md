# Final rigid sampling experiment

This is the last rigid tuning experiment. The selected O0 configuration and exact
saved 18-subject cohort (six Guys, six HH, six IOP) remain fixed. S2 reuses original
O0 result JSONs, deterministic metrics, transforms and historical runtime from
`rigid_optimizer_v4`. S2 is never optimized or evaluated with MI again. Its stored
transform is applied only to reconstruct diagnostic planes for visual comparison.

Only S5 (5%) and S10 (10%) execute: 36 new registrations in total. Existing
S2 full-cohort repeat evidence is retained from v4. S5/S10 receive synthetic
fixed-seed repeat tests; each is run once on the real cohort, with no additional
real-data repeat. The registration engine is reused unchanged. Sampling remains RANDOM with seed 42;
all other registration parameters are copied from corrected-A/O0 provenance.
GEOMETRY initialization, unmasked 50-bin Mattes MI, native canonical preprocessing,
pinned MNI template, linear interpolation, single ITK thread, Euler3D two-stage
structure, physical-shift scales and 300 iterations per level remain identical.
Translation weights are [0,0,0,1,1,1], with factors [4,2] and smoothing [2,1] mm.
Rigid weights are all one, factors [4,2,1], smoothing [2,1,0] mm. RegularStepGradientDescent
uses rate 2, minimum step .001, relaxation .5, gradient tolerance 1e-6, learning-rate
estimation Never, and maximum physical-step setting 0. No metric masks are used.

Full-resolution post-hoc negative Mattes MI is evaluated without sampling at all
three stages for new registrations. Positive improvement means initial minus
final; sampled optimizer traces remain separate. S2 values come unchanged from
its saved results. Registration validity, coverage at all stages, level stops,
iterations, actual scales/rates, runtimes and pairwise relative rotations/common
center pull-displacement differences are recorded. The existing >5 degree or
>5 mm engineering review triggers are retained; they are not anatomical rejection
thresholds. FOV coverage is sample availability, not anatomical correspondence.

Each subject panel shows fixed/S2/S5/S10 at matched sagittal, coronal and axial
planes, with fixed brain contours, fixed intensity contours and checkerboards.
Three site sheets support site review. Only QC planes are retained in memory;
there is no permanent registered NIfTI collection. Display planes do not freeze
model geometry. Case time excludes source loading and PNG/table export. S2 time
is historical; host load is uncontrolled, so these are descriptive cost estimates.

Selection must consider technical validity, visual correspondence (gross pose,
cortex, ventricles where visible, cerebellum, superior/inferior extent and FOV),
transform stability, site consistency and runtime alongside deterministic MI.
A numerical stopping criterion or MI gain alone cannot establish reliability.
Absent a clear benefit from denser sampling, retain S2 rather than tune further.
The final decision must distinguish sufficiently reliable rigid registration from
unresolved unacceptable correspondence. No held-out or affine experiment runs
as part of this study; no final tensor geometry is selected.

```sh
.venv/bin/python -m brainage_xai.registration.sampling_study \
  --output artifacts/registration/rigid_sampling_v5
```

The command refuses a nonempty output directory and checks saved O0 environment,
core hashes, configuration, cohort/order and full-cohort repeat evidence. It never
retries or adds percentages. `--finalize-only` aggregates existing completed
sampling records without registration; it does not overwrite subject results or
prior experiments. Participant-bearing tables, images and provenance stay ignored.
