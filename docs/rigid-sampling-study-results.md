# Final rigid sampling study: results

The [predefined sampling protocol](rigid-sampling-study.md) holds the selected O0 formulation fixed on the exact saved 18 subjects, six per site. S2 reuses existing 2% results without optimization or MI reevaluation. Only S5 (5%) and S10 (10%) ran: 36 new registrations. Native preprocessing, template, GEOMETRY initialization, two-stage Euler3D structure, unmasked Mattes MI, interpolation, seed 42, physical-shift scales, optimizer parameters and 300-iteration-per-level budget remained fixed.

## Numerical results

Positive deterministic improvement is initial minus final negative Mattes MI, evaluated without sampling at full template resolution. Sampled optimizer trajectories remain separate.

| Sampling | Successes | MI improved / degraded / unchanged | Delta p10 / median / p90 | Translation / rigid any-level budget hits | Final rigid-level budget hits |
|---|---:|---:|---|---:|---:|
| S2 | 18/18 | 18 / 0 / 0 | 0.053031 / 0.164110 / 0.232686 | 0 / 14 | 11 |
| S5 | 18/18 | 18 / 0 / 0 | 0.055121 / 0.168703 / 0.233250 | 1 / 18 | 11 |
| S10 | 18/18 | 18 / 0 / 0 | 0.054377 / 0.163958 / 0.232755 | 0 / 16 | 8 |

| Sampling | MI initial p10 / median / p90 | MI after translation p10 / median / p90 | MI final p10 / median / p90 |
|---|---|---|---|
| S2 | -0.385267 / -0.317754 / -0.193057 | -0.511246 / -0.447708 / -0.406832 | -0.521486 / -0.457370 / -0.402397 |
| S5 | -0.385267 / -0.317754 / -0.193057 | -0.511410 / -0.447930 / -0.406874 | -0.521121 / -0.461548 / -0.408457 |
| S10 | -0.385267 / -0.317754 / -0.193057 | -0.511328 / -0.445232 / -0.406554 | -0.520786 / -0.457089 / -0.405243 |

| Sampling | Budget / step / gradient level stops | Case runtime median / p90 / max (s) | Translation runtime median / p90 / max (s) | Rigid runtime median / p90 / max (s) |
|---|---|---|---|---|
| S2 | 19 / 71 / 0 | 35.10 / 37.06 / 37.22 | 3.95 / 4.05 / 4.11 | 25.53 / 27.32 / 27.48 |
| S5 | 27 / 63 / 0 | 63.17 / 68.90 / 73.25 | 4.18 / 4.53 / 9.48 | 53.26 / 58.49 / 59.66 |
| S10 | 30 / 58 / 2 | 102.33 / 123.17 / 126.63 | 4.46 / 4.68 / 9.71 | 92.33 / 113.24 / 116.47 |

Stop counts cover all five levels. A budget hit or small-step termination is not an anatomical pass/fail result. S2 runtimes are the original v4 measurements; new case times include initialization, deterministic metric/coverage checks, both stages and final resampling, excluding source loading and PNG/table export. Host load was uncontrolled. These are descriptive costs, not a synchronized benchmark.

| New sampling | Lower final deterministic MI than S2 | Final-MI gain over S2 p10 / median / p90 |
|---|---:|---|
| S5 | 10/18 | -0.002248 / 0.000298 / 0.005159 |
| S10 | 10/18 | -0.001341 / 0.000462 / 0.003604 |

## Site checks

| Site | Sampling | Successes | MI improved / degraded | Delta median | Translation / rigid budget hits | Final rigid-level hits | Runtime median / p90 / max (s) | Visual residual concerns / gross-pose concerns |
|---|---|---:|---:|---:|---:|---:|---|---|
| Guys | S2 | 6/6 | 6 / 0 | 0.190428 | 0 / 6 | 6 | 35.80 / 37.18 / 37.22 | 6 / 0 |
| Guys | S5 | 6/6 | 6 / 0 | 0.191766 | 0 / 6 | 5 | 65.95 / 70.76 / 73.25 | 6 / 0 |
| Guys | S10 | 6/6 | 6 / 0 | 0.191916 | 0 / 5 | 2 | 108.04 / 123.20 / 126.63 | 6 / 0 |
| HH | S2 | 6/6 | 6 / 0 | 0.149996 | 0 / 4 | 1 | 34.29 / 36.61 / 36.93 | 6 / 0 |
| HH | S5 | 6/6 | 6 / 0 | 0.152503 | 1 / 6 | 1 | 43.22 / 59.21 / 67.41 | 6 / 0 |
| HH | S10 | 6/6 | 6 / 0 | 0.151542 | 0 / 6 | 2 | 72.64 / 121.68 / 122.56 | 6 / 0 |
| IOP | S2 | 6/6 | 6 / 0 | 0.202742 | 0 / 4 | 4 | 35.25 / 36.56 / 37.02 | 6 / 0 |
| IOP | S5 | 6/6 | 6 / 0 | 0.201226 | 0 / 6 | 5 | 63.17 / 69.03 / 69.34 | 6 / 0 |
| IOP | S10 | 6/6 | 6 / 0 | 0.201001 | 0 / 5 | 4 | 114.43 / 123.22 / 123.34 | 6 / 0 |

| Pair | Site | Rotation p10 / median / p90 (degrees) | Rotation max | Common-center translation p10 / median / p90 (mm) | Translation max | Review flags |
|---|---|---|---:|---|---:|---:|
| S2/S10 | Overall | 1.087730 / 5.257489 / 7.303254 | 9.463620 | 0.630101 / 1.659742 / 3.018761 | 4.786648 | 10 |
| S2/S10 | Guys | 3.487656 / 5.699084 / 6.516978 | 7.121924 | 1.149918 / 2.170374 / 2.695548 | 2.809703 | 4 |
| S2/S10 | HH | 0.516561 / 1.872918 / 7.361667 | 9.463620 | 0.399734 / 0.929353 / 3.318827 | 4.786648 | 2 |
| S2/S10 | IOP | 3.912296 / 5.331834 / 7.302729 | 7.726357 | 1.250638 / 1.697892 / 3.037894 | 3.506561 | 4 |
| S2/S5 | Overall | 1.003066 / 3.780371 / 6.906715 | 7.978287 | 0.590196 / 1.752916 / 3.472029 | 4.355259 | 3 |
| S2/S5 | Guys | 2.054997 / 4.144648 / 6.239909 | 7.604387 | 0.980197 / 1.703167 / 3.062894 | 3.968929 | 1 |
| S2/S5 | HH | 0.247228 / 1.660547 / 5.980163 | 7.978287 | 0.144473 / 2.207915 / 3.479529 | 4.355259 | 1 |
| S2/S5 | IOP | 3.283174 / 4.158300 / 5.764709 | 6.607713 | 1.068687 / 1.389645 / 2.734011 | 3.259072 | 1 |
| S5/S10 | Overall | 0.484774 / 1.108102 / 3.929360 | 4.834328 | 0.099157 / 0.518553 / 1.982269 | 2.939937 | 0 |
| S5/S10 | Guys | 0.554964 / 1.384433 / 4.268460 | 4.834328 | 0.086118 / 0.697131 / 1.755167 | 1.809339 | 0 |
| S5/S10 | HH | 0.448887 / 1.287333 / 2.775749 | 3.907675 | 0.412882 / 1.510402 / 2.662855 | 2.939937 | 0 |
| S5/S10 | IOP | 0.461996 / 1.108102 / 2.844876 | 3.979957 | 0.112438 / 0.273939 / 0.666100 | 0.953056 | 0 |

Relative rotation is the geodesic angle between rigid matrices; translation disagreement is pull displacement at the common fixed center. The prespecified >5 degree or >5 mm triggers are review flags, not evidence that one result is anatomically wrong.

## Transform validity and coverage

| Sampling | Quantity | p10 XYZ | Median XYZ | p90 XYZ |
|---|---|---|---|---|
| S2 | Rotation degrees | [-9.444, -1.860, -2.247] | [-5.071, 0.640, -0.839] | [0.392, 3.718, 2.743] |
| S2 | Translation mm | [-5.931, -44.743, -21.964] | [-1.971, -30.034, 4.921] | [2.618, -14.754, 22.932] |
| S5 | Rotation degrees | [-13.696, -1.632, -3.656] | [-8.517, -0.095, -1.280] | [-1.571, 3.250, 2.977] |
| S5 | Translation mm | [-4.624, -43.291, -22.072] | [-1.153, -29.149, 5.018] | [2.958, -13.750, 22.278] |
| S10 | Rotation degrees | [-14.042, -1.529, -3.969] | [-9.766, -0.068, -1.135] | [-3.199, 3.869, 3.403] |
| S10 | Translation mm | [-5.012, -43.162, -22.114] | [-1.189, -27.910, 4.917] | [2.359, -13.399, 22.777] |

All 54 completed transforms are finite rigid Euler3D transforms, without scale/shear. Determinants range from 0.9999999999999997 to 1; maximum orthonormality error is 3.33e-16. All completed output grids match the pinned template.

| Sampling | Stage | Coverage min / median / max |
|---|---|---|
| S2 | initial | 1.000000 / 1.000000 / 1.000000 |
| S2 | after_translation | 1.000000 / 1.000000 / 1.000000 |
| S2 | final | 1.000000 / 1.000000 / 1.000000 |
| S5 | initial | 1.000000 / 1.000000 / 1.000000 |
| S5 | after_translation | 1.000000 / 1.000000 / 1.000000 |
| S5 | final | 1.000000 / 1.000000 / 1.000000 |
| S10 | initial | 1.000000 / 1.000000 / 1.000000 |
| S10 | after_translation | 1.000000 / 1.000000 / 1.000000 |
| S10 | final | 1.000000 / 1.000000 / 1.000000 |

Coverage uses the MNI brain ROI solely as a spatial reference, without metric masking. It measures source FOV availability, not anatomical correspondence. Complete coverage cannot establish acceptance. Per-case clipping/FOV observations remain in private visual notes.

## Visual engineering review

Engineering review of matched central sagittal/coronal/axial fixed-contour and checkerboard views. Review includes gross pose, cortex, visible ventricles, cerebellum, superior/inferior extent and clipping/FOV; no medical interpretation. Individual shape differences may contribute to residual contours. Central views cannot prove whole-volume correspondence.

All 18 panels and the Guys/HH/IOP sheets were reviewed. Gross pose remains broadly plausible and no obvious clipping/FOV loss is seen, but acceptable anatomical correspondence for freezing is not established. Residual cortical/superior, ventricular/internal and inferior/cerebellar differences persist across densities.
IOP inferior signal limits internal/cerebellar review. A QC-only fixed-ROI display-window check on stored S2 did not remove that limitation. Visible residuals may include individual shape differences; no participant is excluded and no medical interpretation is made.
No consistent visual improvement of S5 or S10 over S2 was established within any site. Denser transforms are closer to each other than to S2, but agreement and MI improvements do not independently confer anatomical acceptance.

## Final decision

**C: Residual anatomical correspondence remains unacceptable. Stop rigid tuning and recommend one small controlled rigid-vs-affine comparison.**

Do not freeze any tested rigid configuration for preprocessing. All settings are technically valid, but technical validity and full FOV coverage do not establish acceptable tissue correspondence. Higher sampling changes pose and convergence without a consistent visual improvement around cortical/superior, ventricular/internal and inferior/cerebellar contours. Inferior visibility remains limited in IOP. S5 leaves the final-level budget-hit count at 11/18; S10 reduces it to 8/18 but increases any-level rigid budget hits from 14/18 to 16/18. Median case cost rises from 35.10 seconds for historical S2 to 63.17/102.33 seconds for S5/S10. Each denser setting improves final deterministic MI over S2 in only 10/18 cases, with small median paired gains. S5/S10 agree more closely with each other, but that consistency does not resolve the remaining anatomical QC concerns. Retain S2 as the rigid reference, stop rigid hyperparameter tuning, and recommend one small controlled rigid-versus-affine comparison before deciding preprocessing geometry. This is an engineering acceptance decision from the available matched-plane review, not a claim that affine is proven superior or that every contour difference is registration error.

This completes the last rigid tuning experiment. No next experiment was implemented. No tensor geometry, slice indices, model training, BAG or XAI work was performed.

## Outputs and verification

Private output: `artifacts/registration/rigid_sampling_v5/`. It contains 18 S2 reference records with original source hashes, 36 new registration records/traces, a 54-row comparison table, 54 pairwise comparisons, aggregate/site summaries, 18 comparison panels and three site sheets. S2 transforms were resampled solely for QC; its stored metrics, stages, transforms and historical runtime were reused exactly. There are no permanent registered NIfTI volumes. One supplementary stored-S2 display-window check is saved separately under `qc_display_checks/`; it involved no registration or MI evaluation. Visual assessments are retained separately in `visual_review.json`.

Integrity passed for all 581 raw MRI hashes and size/mtime records, IXI.xls, pinned templates, local configs and all 1658 prior protected files, including v4 comparison/repeat and all earlier experiments. Prior file trees are unchanged. 240 passed in 23.53s; Ruff, compileall and pip check passed. No staged participant artifacts or local configs, no model weights, no commit, push, merge or training.

The run used the module command in the protocol. Its output directory must be empty. It cannot rerun S2 or add percentages. `--finalize-only` regenerates tables from existing sampling records without registration.
