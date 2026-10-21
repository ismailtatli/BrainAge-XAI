# Phase 3 scientific ML core results

Phase 3 is scientifically accepted and frozen with multi-view model B. Research infrastructure only. Phase 1/2 inputs, subject membership, the selected checkpoint and model selection remain frozen. Git finalization does not authorize retraining or another locked-test evaluation.

## Dataset and representation

561 IXI subjects: train 393, validation 84, locked test 84. No split regeneration or slice-level splitting. Site counts: Guys 313, HH 180, IOP 68.

Exact Phase 2 grid: MNI152NLin2009cAsym, 193×229×193 at 1 mm RAS+. Existing S2/O0 rigid only; no preprocessing change, crop/pad or new registration.

Model A uses central axial index 96 (z=18 mm). Model B uses the public brain-mask nonzero extent only: round-to-even(min + fraction × (max−min)), fractions 0.20/0.35/0.50/0.65/0.80, independently along each axis. Zero-based indices:

- axial: [37, 60, 83, 106, 129]; RAS axis mm: [-41.0, -18.0, 5.0, 28.0, 51.0].
- coronal: [60, 87, 114, 142, 169]; RAS axis mm: [-72.0, -45.0, -18.0, 10.0, 37.0].
- sagittal: [53, 74, 96, 118, 139]; RAS axis mm: [-43.0, -22.0, 0.0, 22.0, 43.0].

Mask extent xyz: [24, 24, 6] to [168, 205, 160]. Mask SHA-256: `e40bb1816736504d4c25abc243c1c8503df1d308f64c048318aa4b829e09a7ae`.

Each full plane is transposed and flipped vertically for consistent display, clipped from fixed normalized-MRI range [−3,3] into [0,1], resized bilinearly to 224×224 with antialiasing and align_corners=False, replicated to RGB and ImageNet channel-normalized. There is no fitted cross-subject input transform or augmentation.

## Models and fixed training

Both use torchvision ResNet18 IMAGENET1K_V1 weights (SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`). No random-initialization substitution. Model B shares one backbone across all 15 slices, averages five slice features per view, then concatenates three 512-dimensional view means. Model A uses its single 512-feature vector. Both heads are Linear(128), ReLU, Linear(1), plus training mean age. Fine-tune layer4 and the head only; all BatchNorm statistics/parameters are frozen. The frozen prefix is cached as float32 without quantization; no test features are cached until selection and calibration are complete.

SmoothL1 beta=1.0; AdamW lr=0.0003, weight decay=0.0001; subject batch=4; maximum 40 epochs, validation-MAE patience 8, minimum improvement 0.001 years. No scheduler or search. Gradient norm capped at 5. Python/NumPy/PyTorch seed 42; deterministic algorithms, zero DataLoader workers and seeded shuffle. Cross-hardware bitwise identity is not promised.

| Model | Parameters / trainable | Epochs / best | Device | Training seconds |
|---|---:|---:|---|---:|
| A | 11242305 / 8454401 | 24 / 16 | mps | 96.39 |
| B | 11373377 / 8585473 | 28 / 20 | mps | 530.60 |

Software: nibabel 5.4.2, numpy 2.4.6, pandas 2.3.3, torch 2.10.0, torchvision 0.25.0; Python 3.11.17.

Training runtimes exclude input verification, slice/prefix caching and post-lock reporting. Per-epoch histories, attempt records and complete provenance are private artifacts.

## Validation-only selection

| Metric | A | B |
|---|---:|---:|
| mae | 7.45882 | 5.42819 |
| rmse | 10.14028 | 7.09331 |
| r2 | 0.59891 | 0.80374 |
| mean_error | -1.11914 | -0.81701 |
| median_absolute_error | 5.57415 | 4.79124 |
| prediction_age_correlation | 0.79319 | 0.90095 |
| residual_age_slope | -0.24453 | -0.12226 |
| residual_age_correlation | -0.38848 | -0.27781 |

Selected **B** by minimum validation MAE among nonpathological candidates. Predefined pathology checks reject nonpositive age correlation/R², prediction SD below one year or ages outside 0–120. No test result entered selection. Architecture/checkpoint/correction/reference were frozen before the exclusive test claim.

## Locked test

Exactly one selected-model evaluation on 84 subjects. MAE **6.09402 years**, RMSE **7.56254**, R² **0.79620**. Deterministic percentile bootstrap 95% CI for mean absolute error: **[5.15361, 7.10058] years**, 10,000 subject resamples, seed 2026; this does not include retraining or external-domain uncertainty.

- mean_error: -0.663377.
- median_absolute_error: 5.380411.
- prediction_age_correlation: 0.893933.
- residual_age_slope: -0.233737.
- residual_age_correlation: -0.519756.

Learning curves and predicted-vs-age, residual-vs-age and absolute-error plots are stored privately under the run's `plots/` directory.

## Locked-test subgroups

Age bins were fixed before training, left closed and right open. Sex codes are reported as supplied; a sex codebook was not inferred.

| Group | Category | n | MAE (years) | Mean error |
|---|---|---:|---:|---:|
| acquisition_site | Guys | 47 | 5.67517 | -1.07417 |
| acquisition_site | HH | 26 | 6.58570 | 0.90827 |
| acquisition_site | IOP | 11 | 6.72154 | -2.62296 |
| age_bin | [18, 30) | 15 | 6.86280 | 5.77989 |
| age_bin | [30, 45) | 18 | 5.30838 | 1.94980 |
| age_bin | [45, 60) | 24 | 5.97176 | -3.02260 |
| age_bin | [60, 75) | 25 | 5.82649 | -3.22222 |
| age_bin | [75, 100) | 2 | 12.21023 | -12.21023 |
| sex_code | 1.0 | 39 | 6.05887 | -0.33648 |
| sex_code | 2.0 | 45 | 6.12449 | -0.94669 |

Small subgroup estimates are unstable and descriptive; no subgroup optimization or formal fairness claim is made. Acquisition-site/scanner, age and sex distributions can confound MRI predictions; no independent site-held-out evaluation was performed.

## BAG, validation-only correction and reference

Raw BAG = raw predicted age − chronological age. Fit only the selected model's 84 validation predictions by ordinary least squares: raw BAG = intercept + beta × age.

Frozen intercept = **5.089670155 years**; beta = **-0.122256331**. corrected BAG = raw BAG − (intercept + beta × chronological age); corrected brain age = chronological age + corrected BAG. Test subjects are absent from fit/reference IDs, and their coefficients were never refitted. All raw prediction metrics above stay unchanged.

| Split | BAG | Mean | SD | Age correlation |
|---|---|---:|---:|---:|
| validation | corrected_BAG | -0.000000 | 6.809382 | 0.000000 |
| validation | raw_BAG | -0.817011 | 7.088416 | -0.277813 |
| test | corrected_BAG | 0.307872 | 6.741604 | -0.278676 |
| test | raw_BAG | -0.663377 | 7.578634 | -0.519756 |

Validation corrected BAG has zero mean and age-linear association by construction, not independent validation. Correction uses chronological age and must not be marketed as improved age prediction. Positive corrected BAG means structurally older-looking relative to the learned reference; negative means younger-looking. No diagnostic/clinical threshold.

The frozen validation corrected-BAG empirical midrank distribution supplies a **research-model reference** percentile. The validation raw-residual 2.5/97.5 percentiles supply **empirical model uncertainty** bounds: raw prediction minus upper/lower residual quantile. These are marginal empirical reference intervals, not clinical confidence intervals, disease probabilities or calibrated conditional coverage guarantees. Using the same validation subjects for model choice and reference estimation can be optimistic.

## XAI

Grad-CAM for the selected model only: 6 deterministic test examples, 90 slice maps and 18 view panels. Examples are the union of age and absolute-error rank quantiles 0/0.5/1, with ID tie-breaking. This post-lock error-based example selection changes no metric or model. Per-slice arrays, exact input indices, overlays and checkpoint provenance remain private. The shared layer4 7×7 positive Grad-CAM grid is upsampled to 224×224 and independently max-normalized per slice; colors are not comparable magnitudes.

Regions receiving relatively greater attribution for the model's brain-age prediction.

Grad-CAM is a coarse, gradient-based attribution rather than pathology localization or causal evidence. It does not establish tissue-level biological specificity; no lesion, disease, diagnostic or causal neuroanatomical interpretation is justified.

## Limitations and integrity

The IXI cohort is small (561), single-dataset and subject to site/scanner confounding. Residual rigid-registration correspondence and the Phase 2 IOP contrast/visibility limitation remain. This is 2.5D rather than full volumetric modeling, with ImageNet transfer and fixed slice/intensity windows. There is no independent external cohort, external-validity demonstration or clinical validation. BAG is not a diagnosis. Locked-test findings do not authorize retraining or changing the selected method.

Integrity verification rehashes all 581 raw MRI files, IXI.xls, templates, Phase 2 final_v1 files and frozen Phase 2 source/configs and checks their sizes/mtimes. Subject membership is unique and disjoint. Checkpoints, inputs, predictions and participant provenance remain Git-ignored under `artifacts/modeling/phase3/run_v2/`. Scientific verification preceded Git staging, commit and push; Git finalization does not rerun the accepted workflow.

## Post-lock diagnostic notes

There are 18 standalone heatmap panels in addition to the MRI overlays. 8 of 90 positive Grad-CAM maps are zero. These are retained without replacing examples; positive-only Grad-CAM suppresses negative contributions, so a blank map does not establish absence of model dependence or anatomical signal.

After validation-only correction, locked-test BAG–age correlation remains **-0.279** (-0.278676 before rounding). Correction reduces the observed age association but does not guarantee age-independence. It does not fully generalize the validation age trend; reference percentiles are not age/site-specific norms. No test refitting, model change or additional training was performed. The 75–100-year test group has only two subjects and cannot establish older-age performance.

MRI is not skull stripped. Zero positive Grad-CAM maps and nonbrain/background attribution are known XAI limitations. Nonbrain tissue, background/FOV and site cues may contribute to predictions and attributions. Attributions are not isolated biological effects. BAG and Grad-CAM are research-only and non-diagnostic.

The predefined 0.001-year minimum improvement governs both checkpoint replacement and patience. Sub-threshold numerical MAE decreases do not replace a saved checkpoint.

Selected checkpoint SHA-256: `23a57975e318647636c12dd23ceb087e826d2c3ea7f2e89ec4fe9be30922f51d`.

## Engineering verification

Final verification: **284 tests passed in 29.26 seconds**; `ruff check .`, source compileall and pip check passed. Both production models completed on their first attempt without a technical retry. Total model-training runtime was 626.99 seconds (10.45 minutes), excluding preparation/caching and reporting.

The repository/leakage audit found no tracked MRI, participant exports, generated artifacts, local configs, personal home paths, binaries, model weights or detected secrets. All 581 raw MRI files, metadata, templates and all 1,186 final_v1 files remain unchanged. Core data/model/statistics/workflow sources match their pretraining hashes; post-lock changes only add saved-map rendering and reporting. These scientific checks preceded Git finalization; no merge or additional training is authorized.

The saved diagnostic plots and a mapped axial Grad-CAM panel were inspected for legibility and slice labeling. Some coarse attribution extends into nonbrain/background regions. This reinforces the limits on brain-specific interpretation and the possibility of nonbrain/FOV/site cues; no post-test model change was made.
