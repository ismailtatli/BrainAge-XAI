# Final Phase 2 preprocessing results

Registration remains frozen at S2/O0 rigid. The [frozen preprocessing specification](../configs/preprocessing.frozen.yaml) fixes the full MNI152NLin2009cAsym grid, 193×229×193 at 1 mm, RAS+. There is no crop/pad, affine/nonlinear registration, split regeneration, model slice selection or training. See [processing and resume documentation](preprocessing-frozen.md).

Processed **561/561**; successful **561**; failed **0**. Every subject remains in the existing cohort and saved split. Failures are explicit in the local manifest and failure CSV; membership is never changed automatically.

| Group | Expected | Successful | Failed |
|---|---:|---:|---:|
| train | 393 | 393 | 0 |
| validation | 84 | 84 | 0 |
| test | 84 | 84 | 0 |
| Guys | 313 | 313 | 0 |
| HH | 180 | 180 | 0 |
| IOP | 68 | 68 | 0 |

## Numerical QC

All **561 outputs** have shape **193×229×193** and spacing **1×1×1 mm**. Every successful output was saved and reloaded to verify finite values, exact template affine/orientation/spacing, float32 data, millimeter units, MNI-coded qform/sform, nonempty standardized foreground, preserved background, source/output hashes and finite normalization provenance. Rigid transforms were reconstructed and checked for determinant +1, orthogonality and unit singular values. Maximum determinant deviation from +1 is 5.55e-16.

Normalization statistics are finite and background is preserved for **561/561**. Maximum absolute normalized foreground mean is 1.12e-9; maximum deviation of foreground standard deviation from one is 4.57e-9. Minimum foreground count is 4,092,494 voxels. Parameters are fitted independently within each registered subject using existing 0.5/99.5 foreground percentile clipping and z-scoring. No training/validation/test pooled fitting occurs; zero background is retained. Split/eligibility manifests are checksum pinned, unique and disjoint, and unchanged.

Lowest final fixed-brain-ROI in-bounds fraction: **99.884%**. This is source FOV availability, not anatomical correctness.

| Registration warning | Subject count |
|---|---:|
| Any-level rigid iteration budget hit | 464 |
| Any-level translation warm-up budget hit | 28 |
| Full-resolution MI degradation from initialization | 13 |
| Fixed brain ROI has samples outside source FOV | 1 |

Warning categories overlap. Budget hits and small-step stops do not certify or invalidate anatomical correspondence; they remain visible review signals. No subject is automatically excluded.

Production registration was compared with 18 already completed S2 references: exact transforms, full-resolution metrics/coverage and optimizer stops/iteration counts = True. This comparison only reads prior artifacts; it does not rerun a tuning experiment.

## Runtime and visual QC

Production wall time: 7009.01 seconds (116.82 minutes), including preflight, processing, save/reload QC and checkpoints. Four independent single-threaded workers were used. This changes throughput without changing the frozen per-subject algorithm. Per-subject runtime is stored locally; concurrent runtimes are not added to estimate wall time. Later verification/QC/test time is separate.

All eighteen main sample panels, fourteen additional warning-focused panels (31 unique subjects) and three site sheets were inspected. No new critical gross processing failure, normalization collapse or obvious central-plane brain clipping was observed. Frozen rigid registration retains global cortical/superior and inferior/internal correspondence residuals; this is not a claim of voxelwise template acceptance. All six main IOP cases retain limited inferior/internal contrast, limiting ventricular/cerebellar assessment. MI degradation and the small fixed-ROI FOV shortfall remain explicit warnings; no retries, parameter changes or automatic subject exclusions were performed. Exact-zero background can appear gray under the display window because foreground z-scores include negative values. Numerical zero-background verification passes. Review is engineering QC, not medical interpretation.

Reviewed 31 unique subjects; new critical processing concerns observed: 0. The main sample contains six per site, low/median/high original obliquity and known challenging cases. Fourteen additional warning-focused panels are identified locally. Matched template-center sagittal/coronal/axial contour and checkerboard views are engineering QC, not medical interpretation or model slice selection. This is Codex image-based review, not an independent human rating. Central-plane review cannot certify whole-volume tissue correspondence. Remaining rigid correspondence limitations and site-specific visibility limitations are retained in the QC notes.

## Verification and location

**255 tests passed in 25.97 seconds** at production verification; Ruff, source compileall and pip check passed. Integrity passed: all 581 raw MRI hashes and size/mtime, IXI.xls, templates, manifests, local configs and 1,865 prior protected artifacts are unchanged. All generated NIfTIs are Git-ignored. No model weights or training were produced. Git staging, commit and push are separate release actions; they do not rerun processing.

The actual verify-only resume pass revalidated and reused all 561 outputs with zero new registrations. Volume size/mtime and all subject provenance hashes remained unchanged. Output hashes are unique, and eligible cohort, output and manifest subject sets match exactly, with no missing split members.

Ignored local output: `artifacts/preprocessed/final_v1/` (volumes, provenance, QC, cohort and manifest, summary, failure CSV, leakage verification, registration reproduction, integrity and validation reports). Validated resume skips completed valid subjects; explicit regeneration preserves invalid files in recovery. Source and frozen input changes block resume.

## Completion

Phase 2 preprocessing is complete, scientifically accepted and frozen. Registration remains permanently S2/O0 rigid; the full preprocessing grid is fixed by the public specification. No critical QC blocker remains.

No model training was started. Model slice/input selection remains deferred; preprocessing completion does not authorize automatic training.
