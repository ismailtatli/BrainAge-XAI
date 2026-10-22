# BrainAge-XAI — final academic project report

## 1. Abstract

BrainAge-XAI integrates an audited IXI cohort, frozen rigid MRI preprocessing,
subject-level 2.5D regression, locked evaluation, validation-only BAG correction,
research reference context and Grad-CAM in a local application. The selected
multi-view ResNet18 achieved validation MAE 5.428 years and locked-test MAE 6.094
years (95% bootstrap CI 5.154–7.101), RMSE 7.563 and R² 0.796. Corrected test BAG
retains age correlation −0.279. A shared CLI/Streamlit inference service supplies
individual PDF/JSON reports. These are research-model outputs, not diagnoses.

## 2. Research objective

Estimate chronological age from structural T1 MRI, characterize model-dependent
Brain Age Gap and expose attribution with explicit leakage, privacy and
interpretation limits. Cognitive-aging associations and clinical utility remain
future questions; neither is established by age prediction performance alone.

## 3. IXI dataset and cohort curation

The accepted local audit found 581 T1 NIfTIs. Conservative image/metadata matching,
validation, age checks and duplicate-label handling retained 561 eligible subjects:
Guys 313, HH 180 and IOP 68. Invalid or ambiguous labels were not imputed. Raw data
and spreadsheet stayed immutable. See [Phase 1 audit semantics](audit.md) and the
[official dataset source](https://brain-development.org/ixi-dataset/). The local
cohort is not distributed or a promise of counts for another acquisition/download.

## 4. Leakage-safe subject split

Saved membership is immutable: train 393, validation 84, test 84, seed 42. Subjects
occur exactly once, with no cross-split overlap. Slices/views inherit their subject's
membership. No slice-level split or post-result membership regeneration occurred.
Normalization is subject-local, not a fit to pooled train/validation/test data.

## 5. MRI preprocessing

The accepted [specification](../configs/preprocessing.frozen.yaml) preserves the full
MNI152NLin2009cAsym resolution-01 grid, 193×229×193 at 1 mm RAS+. Orientation
conversion has no interpolation. Following rigid alignment, native intensities
are interpolated once to the template, float32, with zero fill. Existing foreground
0.5/99.5 percentile clipping/z-scoring preserves exact-zero background. There is no
crop/pad, affine/nonlinear registration or skull stripping. All 561 outputs passed
numerical save/reload checks with zero failures. Numerical QC is not proof of
voxelwise anatomical correspondence.

## 6. Registration study and final rigid decision

Isolated 18-subject studies examined initialization/masking, optimizer settings,
sampling density and a single rigid-versus-affine comparison. Their measured
protocols/results are retained in the registration documents, including the
[final decision](rigid-vs-affine-study-results.md). S2/O0 rigid is permanently
frozen; it uses GEOMETRY initialization, translation warm-up, full Euler3D,
unmasked Mattes MI at 2% sampling/seed 42, RegularStepGradientDescent and physical-
shift scales. The engineering decision considered spatial correspondence and the
morphological consequences of affine scale/shear rather than MI alone. This
integration starts no registration study and changes no scientific parameters.

## 7. Final 2.5D architecture

The final model B shares ImageNet-pretrained ResNet18 across three views, five
fixed slices each. A public-mask-only 20/35/50/65/80% extent rule gives axial
37/60/83/106/129, coronal 60/87/114/142/169, sagittal 53/74/96/118/139. Full planes
resize consistently to 224×224; fixed MRI [−3,3] values map to [0,1], repeat to RGB
and receive ImageNet channel normalization. Per-view mean features are concatenated
into a small 128-unit ReLU regression head with training-mean offset. The axial
baseline A uses only central index 96. No third architecture was trained.

## 8. Training protocol

Both models used the same subjects, seed 42, SmoothL1 beta 1 and AdamW (lr 0.0003,
weight decay 0.0001), batch four, 40-epoch cap, patience eight and 0.001-year
checkpoint/stopping improvement criterion. Layer4/head were fine-tuned; BatchNorm
statistics/parameters stayed fixed. MPS training completed on the first attempt
for both models, taking 96.39 seconds for A and 530.60 seconds for B, excluding
verification/caching/reporting. A stopped after 24 epochs, checkpoint 16; B after
28, checkpoint 20. No broad search or new training occurred during integration.

## 9. Validation model selection

Validation MAE was 7.459 years for A and 5.428 for B. Predefined collapse checks
preceded MAE selection. B's validation RMSE was 7.093 and R² 0.804. Architecture,
checkpoint and validation-derived calibration/reference artifacts were frozen
before the test gate. Tiny sub-0.001-year improvements do not replace a checkpoint
under the recorded fixed policy.

## 10. Locked test methodology

Only the selected checkpoint was evaluated once on the 84-subject test set. A
persistent exclusive claim prevents silently repeating the pass, including after
a crash. Test outcomes did not choose architecture, slices, epochs, optimizer,
checkpoint or model. Deterministic 95% percentile bootstrap CI for overall MAE uses
10,000 subject resamples, seed 2026. No post-test retraining or correction refit occurred.

## 11. Quantitative results

| Locked-test metric | Measured result |
|---|---:|
| MAE | 6.094 years |
| RMSE | 7.563 years |
| R² | 0.796 |
| MAE 95% bootstrap CI | 5.154–7.101 years |
| Mean error | −0.663 years |
| Median absolute error | 5.380 years |
| Prediction–age correlation | 0.894 |
| Residual–age slope / correlation | −0.234 / −0.520 |

The CI is for mean test error; it is not individual empirical uncertainty or
external-domain uncertainty. Full metrics/learning curves are documented in
[Phase 3 results](phase3-scientific-ml-results.md); participant plots remain private.

## 12. BAG and age-bias correction

Raw BAG = raw prediction − chronological age. On validation only, fit raw BAG =
intercept + beta × age. Frozen intercept is 5.089670154648087 and beta is
−0.12225633106796398. Corrected BAG subtracts that trend; corrected brain age =
age + corrected BAG. The correction requires age and does not improve independent
age prediction; raw metrics are unchanged. Test raw BAG is −0.663 ± 7.579 years;
corrected BAG is 0.308 ± 6.742. Corrected test BAG–age correlation remains **−0.279**:
correction reduces but does not guarantee age-independence. Zero validation
mean/linear correlation is a construction property, not independent validation.

## 13. Reference and uncertainty

The frozen validation corrected-BAG empirical distribution gives a midrank
research-model reference percentile. Raw validation-residual 2.5/97.5 quantiles
provide an empirical reference interval around the raw prediction. These are
marginal empirical model uncertainty/reference quantities, not clinical confidence
intervals, disease probabilities or guaranteed conditional coverage. Reference
construction sharing validation data with model choice can be optimistic.

## 14. XAI methodology

Accepted regression Grad-CAM uses shared layer4 features for each input slice.
Positive 7×7 maps are upsampled to 224×224 and independently normalized. The study's
six deterministic examples were the union of age/error rank quantiles 0/0.5/1;
post-lock error-based illustration did not change metrics/models. Eight of 90
positive maps were zero and retained. Nonbrain/background attribution was observed.
New inference produces all 15 maps; representative UI/PDF views always use each
view's middle fixed input slice, never the most attractive attribution.

“Grad-CAM indicates regions receiving relatively greater attribution for the model's prediction.”

Positive clipping suppresses negative contributions; blank maps do not establish
absence of model dependence. Grad-CAM is coarse attribution, not pathology
localization, causal evidence or a validated biological explanation.

## 15. Site, age and sex analyses

Raw test site MAE: Guys n=47, 5.675; HH n=26, 6.586; IOP n=11, 6.722 years.
Age-bin MAE/n: [18,30) 6.863/15; [30,45) 5.308/18; [45,60) 5.972/24;
[60,75) 5.826/25; [75,100) 12.210/2. Sex codes 1/2 had MAE 6.059/6.124,
n=39/45; coding was not interpreted without a codebook. Full mean errors appear
in the [model card](model-card.md). Small groups are descriptive, especially n=2;
no clinical/fairness conclusion or site-held-out generalization was established.

## 16. Application workflow

A shared inference service validates one MRI and age, verifies pinned artifacts,
processes a temporary copy with unchanged frozen functions, predicts with model B,
applies frozen correction/reference, computes mapped Grad-CAM and generates a
structured result plus PDF/JSON. No IXI filename assumptions enter new-MRI inference.
Streamlit is loopback-local, with status/QC, distinct raw/corrected outputs,
reference context, all-slice gallery and explicit downloads. The CLI uses exactly
the same service. Technical QC failure produces no fabricated result. Outside-
range age is allowed with an extrapolation warning; no clinical cutoffs are added.
Temporary MRI copies are cleaned on normal success and exceptions. Reports use
anonymous analysis IDs; MRI-derived images/outputs remain sensitive.

One existing **training** subject was used only for an end-to-end technical smoke:
raw MRI through frozen processing, inference, all 15 maps, PDF and JSON passed in
39.12 seconds; the normalized voxel array matched its accepted Phase 2 output
exactly and the temporary upload was removed. This is not an added scientific
evaluation; its predictions/identity and participant-derived artifacts stay private.
The accepted locked test and study results were not rerun or updated.

## 17. Limitations

The 561-subject single dataset is small; there is no independent external cohort
or clinical validation. Site/scanner, age and sex distributions can confound MRI
predictions. Rigid residual correspondence and IOP contrast/visibility limitations
persist. The model is 2.5D, uses ImageNet transfer and fixed windows/slices, and
includes nonbrain/FOV cues. Corrected BAG remains age-associated on test. Reference
coverage is not a clinical guarantee. XAI has zero/background maps and no demonstrated
causal/biological specificity. Technical application completeness does not resolve
these scientific limitations or grant clinical validity.

## 18. Ethical and research-only scope

BrainAge-XAI is a research prototype. BAG is not a diagnosis. No normal/abnormal,
healthy/unhealthy or disease-risk thresholds, probabilities or treatment advice
are supplied. Positive/negative BAG denotes older-/younger-looking relative to
the learned research reference. Values rounding to zero at report precision get
numerical/reference wording, not a clinical normal range. No source MRI, identities,
weights or participant exports are published. Follow data permission, privacy and
sharing obligations; derived reports are sensitive even without names.

## 19. Conclusion

The accepted scientific core is integrated into one reproducible local inference,
attribution and reporting application, retaining immutable study decisions and
explicit interpretation boundaries. Evidence supports the measured IXI prediction
results and the technical workflow; it does not establish diagnostic utility.

## 20. Future work — not completed

External validation; OASIS/cognitive-aging outcome analysis; improved independently
validated bias correction; segmentation/brain masking; stronger XAI validation;
and volumetric architectures may be considered in separately authorized research.
They are not implemented or evidence for the current model. No release/tag is
published by this integration.
