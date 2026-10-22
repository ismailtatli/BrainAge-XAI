# Model card — BrainAge-XAI multi-view B

## Status and intended use

Accepted, frozen academic research model for estimating chronological age from
T1-weighted structural MRI and describing model-dependent BAG/reference position.
BrainAge-XAI is a research prototype. BAG and Grad-CAM are non-diagnostic.
No external clinical validity has been established. This is not a medical device
or a validated biomarker of individual disease or cognitive decline.

Out-of-scope uses include diagnosis, screening, treatment decisions, disease-risk
probabilities, healthy/unhealthy classification, causal neuroanatomical claims,
and unqualified application to other populations/scanners. New MRI inference is
supported technically; generalization outside IXI has not been established.

## Dataset and leakage controls

Accepted local Phase 1: 581 T1 files, 561 eligible labeled subjects. Retained cohort:
Guys 313, HH 180, IOP 68. Immutable subject split: 393 train, 84 validation, 84 test.
No subject overlap or slice-level split. No participant data, predictions or weights
are distributed with this repository. IXI access and local artifact setup are
separate prerequisites; see [reproducibility](reproducibility.md).

Observed chronological-age ranges: training **19.980835–86.198494 years**;
validation reference **22.532512–81.941136 years**. These are measured ranges,
not validity thresholds. Outside either range, technically valid inference is
allowed with an explicit extrapolation warning.

## Input and preprocessing

One readable, finite, nonempty 3D `.nii`/`.nii.gz` T1 MRI with nondegenerate affine
and coded orientation; chronological age is required only for BAG/reference output.
Filename, site, sex and other optional metadata are not model inputs. T1 modality
cannot be established automatically. No assumptions depend on IXI filename tokens.

The frozen Phase 2 pipeline performs orientation-only closest-canonical conversion,
GEOMETRY-initialized S2/O0 rigid registration, translation warm-up/full Euler3D,
unmasked Mattes MI at 2% sampling/seed 42, fixed RegularStepGradientDescent with
physical-shift scales and the accepted multiresolution settings. Native intensities
are resampled once with linear interpolation onto MNI152NLin2009cAsym's exact
193×229×193, 1 mm RAS+ grid. Foreground-local 0.5/99.5 percentile clipping/z-scoring
preserves zero background. No affine/nonlinear registration, crop/pad or skull stripping.
Numerical QC/FOV coverage does not certify anatomical correspondence.

## Architecture and training

One ImageNet IMAGENET1K_V1 ResNet18 backbone is shared across exactly 15 fixed slices.
The public mask extent at 20/35/50/65/80% determines indices once:
axial 37/60/83/106/129; coronal 60/87/114/142/169; sagittal 53/74/96/118/139.
Full planes are resized bilinearly/antialiased to 224×224, standardized MRI values
are clipped to [−3,3] and mapped to [0,1], then repeated to RGB and ImageNet-normalized.
Five slice features are averaged per view; three view vectors are concatenated,
then a 128-unit ReLU regression head predicts age with a training-mean offset.

Only layer4/head were fine-tuned; BatchNorm statistics/parameters were frozen.
SmoothL1 beta 1, AdamW lr 0.0003, weight decay 0.0001, batch four subjects,
40-epoch cap, patience eight and 0.001-year checkpoint/stopping improvement criterion.
Seed 42; MPS; no hyperparameter search. B completed 28 epochs, selecting epoch 20.
There are 11,373,377 parameters, 8,585,473 trainable. Architecture/checkpoint selection
preceded test evaluation. Checkpoint SHA-256:
`23a57975e318647636c12dd23ceb087e826d2c3ea7f2e89ec4fe9be30922f51d`.

## Measured evaluation

Validation MAE: axial A **7.459 years**, multi-view B **5.428 years**. B was selected
by validation MAE after predefined pathological-generalization checks. Only B was
evaluated once on the locked 84-subject test. Raw test metrics:

| Metric | Value |
|---|---:|
| MAE | 6.094 years |
| RMSE | 7.563 years |
| R² | 0.796 |
| Overall MAE 95% bootstrap CI | 5.154–7.101 years |
| Mean error | −0.663 years |
| Median absolute error | 5.380 years |
| Prediction–age correlation | 0.894 |
| Residual–age slope | −0.234 |
| Residual–age correlation | −0.520 |

CI uses 10,000 deterministic subject bootstrap resamples, seed 2026. It concerns
mean test error, not a single individual's uncertainty or retraining/external-domain variation.

| Test subgroup | n | MAE | Mean error |
|---|---:|---:|---:|
| Guys | 47 | 5.675 | −1.074 |
| HH | 26 | 6.586 | 0.908 |
| IOP | 11 | 6.722 | −2.623 |
| Age [18,30) | 15 | 6.863 | 5.780 |
| Age [30,45) | 18 | 5.308 | 1.950 |
| Age [45,60) | 24 | 5.972 | −3.023 |
| Age [60,75) | 25 | 5.826 | −3.222 |
| Age [75,100) | 2 | 12.210 | −12.210 |
| Sex code 1 | 39 | 6.059 | −0.336 |
| Sex code 2 | 45 | 6.124 | −0.947 |

Sex coding is reported without inferring a codebook. Groups are descriptive and
small; the two oldest subjects cannot establish older-age performance. Scanner/site
confounding remains; no site-held-out or external-cohort evaluation was completed.

## BAG and research reference

Raw BAG = raw prediction − age. Fit only 84 selected-model validation predictions:
raw BAG = intercept + beta × age. The runtime uses full-precision frozen artifacts:
intercept **5.089670154648087**, beta **−0.12225633106796398**. Corrected BAG subtracts
that age trend; corrected brain age = age + corrected BAG. This uses chronological
age and is not an improved independent age predictor. Raw performance metrics stay unchanged.

Test raw BAG is −0.663 ± 7.579 years; corrected BAG is **0.308 ± 6.742 years**.
Corrected test BAG–age correlation remains **−0.279**. Correction reduces but does
not guarantee age-independence. Zero validation mean/linear correlation occurs by
construction. No test refitting occurred.

Corrected-BAG empirical midrank percentiles use the frozen validation distribution.
Empirical uncertainty intervals invert the raw validation-residual 2.5/97.5
quantiles around the raw prediction. Labels are **research-model reference** and
**empirical model uncertainty**. They are not clinical confidence intervals,
age/site-specific norms or disease probabilities; conditional coverage is not
established. Selection/reference estimation sharing validation data can be optimistic.

## XAI and ethical limitations

Accepted Grad-CAM targets the shared layer4 output for each of the 15 input slices.
Positive 7×7 maps are upsampled and independently max-normalized. Slice indices,
view/input offsets and checkpoint provenance are preserved. Representative UI/PDF
views use each view's fixed middle slice, not attractive attributions.

“Grad-CAM indicates regions receiving relatively greater attribution for the model's prediction.”

Zero maps and nonbrain/background attribution are known limitations. In the accepted
six-example XAI set, eight of 90 positive maps were zero; examples were not replaced.
Positive clipping omits negative contributions. Colors are not comparable salience
magnitudes across slices. Attributions are not lesions, pathology localization,
causal effects or diagnostic findings.

The small IXI cohort, 2.5D rather than volumetric representation, ImageNet transfer,
residual rigid correspondence and documented IOP contrast/visibility limitation
restrict interpretation. Nonbrain, FOV and site cues can contribute. No independent
external cohort, cognitive-outcome study or clinical validation was completed.
MRI-derived reports/images are sensitive even without participant names. The local
app cleans temporary MRI copies on normal success/error and persists outputs only
on explicit export. See [privacy/cleanup details](demo.md#privacy-and-cleanup).
