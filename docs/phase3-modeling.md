# Gated Phase 3 modeling

Phase 3 is scientifically accepted and frozen with multi-view model B. Its
[accepted results](phase3-scientific-ml-results.md), checkpoint, selection and
validation-only calibration are closed. The commands below document the original
workflow; finalization must not rerun training or the consumed locked-test gate.

Phase 2 is immutable. Only accepted `artifacts/preprocessed/final_v1/` volumes and
their existing 393/84/84 subject membership enter modeling. Phase 3 never calls
registration, preprocessing or split generation. Install the optional modeling
dependencies with `python -m pip install -e '.[dev,modeling]'`. The fixed compatible
[PyTorch/torchvision versions](https://pytorch.org/get-started/previous-versions/)
are recorded in `pyproject.toml`; frozen Phase 2 package versions must be retained.

The [fixed policy](../configs/modeling.frozen.yaml) precedes training and test access.
ImageNet [ResNet18 IMAGENET1K_V1](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet18.html)
weights must be explicitly downloaded to ignored modeling resources and verified.
Production preparation fails when weights are missing or unverified. Synthetic
unit tests use random temporary initialization solely to test mechanics offline.

## Input and models

Model A uses central axial index 96. Model B uses exactly five slices per axial,
coronal and sagittal view. Along each axis, find the public MNI brain-mask nonzero
minimum and maximum; select round-to-even(min + fraction × (max−min)) at fractions
0.20, 0.35, 0.50, 0.65 and 0.80. The mask is the existing checksum-pinned public
resource, not participant data. The resulting zero-based indices are axial
37/60/83/106/129, coronal 60/87/114/142/169 and sagittal 53/74/96/118/139.
These correspond to RAS axis millimeters −41/−18/5/28/51,
−72/−45/−18/10/37 and −43/−22/0/22/43, respectively.

Each full plane is transposed/flipped consistently and resized to 224×224 using
bilinear antialiased interpolation, without crop/pad. Fixed normalized intensity
range [−3,3] maps to [0,1], then repeats into RGB and receives ImageNet channel
normalization. Exact-zero Phase 2 background maps to gray 0.5. No participant-
specific slice choice, fitted cohort transform, test-derived choice or augmentation.

Both models use one pretrained ResNet18 backbone. Only layer4 and a small
128-unit regression head are optimized; BatchNorm parameters/statistics stay fixed.
Model B averages five slice features within each view and concatenates three
view representations. Both output scalar chronological age with an offset equal
to training mean age. A float32 cache of the nontrainable prefix avoids repeating
the same frozen computation. Test features are constructed only after selection.

## Workflow and gates

Invoke the module from the repository root, with a new ignored output directory:

```sh
.venv/bin/python -m brainage_xai.modeling.workflow prepare --output artifacts/modeling/phase3/new_run --device cpu
.venv/bin/python -m brainage_xai.modeling.workflow train --output artifacts/modeling/phase3/new_run --device mps
.venv/bin/python -m brainage_xai.modeling.workflow test --output artifacts/modeling/phase3/new_run --device mps
.venv/bin/python -m brainage_xai.modeling.workflow report --output artifacts/modeling/phase3/new_run --device mps
```

`run` executes the gates consecutively. An existing run is never overwritten by
preparation. Once selected, further training is forbidden. Both models use identical
subjects, loss/optimizer/epoch/stopping policy and deterministic seeds. Select by
validation MAE after predefined collapse checks. Freeze the checkpoint, validation-
only age-bias fit and empirical reference before claiming test evaluation.

An exclusive, persistent test claim prevents a second evaluation, including after
a crash. A failed claimed test pass requires scientific review; it cannot silently
restart. Post-test reporting consumes saved predictions and never tunes a model.
Grad-CAM intentionally uses the frozen selected model to attribute predefined
test examples; it is separate from the single aggregate test evaluation.

The same policy permits only one technical training retry per model, recorded
before each attempt. No retry may change hyperparameters or architecture. MPS is
used where available; an unsupported MPS operation may require a recorded CPU
fallback. No silent fallback to CPU or random pretrained weights.

Python, NumPy and PyTorch seeds are fixed, DataLoader workers are zero, shuffling
uses a seeded generator and deterministic algorithms are required. As documented
by [PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html), identical
results across hardware, platforms and releases are not guaranteed.

## Calibration, uncertainty and interpretation

Raw BAG is predicted age minus chronological age. Fit raw BAG against age only
on selected-model validation predictions; freeze its intercept and slope. Subtract
that fitted age trend from test BAG without refitting. This requires chronological
age and is not a new independent age predictor. Raw accuracy metrics stay raw.

Corrected test BAG retains age correlation **-0.279**. Correction reduces the
observed association but does not guarantee age-independence, including on new
subjects. The frozen parameters must not be refitted using test outcomes.

Validation corrected-BAG midrank percentiles are a **research-model reference**.
Validation raw-residual 2.5/97.5 quantiles provide **empirical model uncertainty**
intervals. They are neither clinical confidence intervals nor disease probabilities.
The shared validation set used for selection and reference estimation can be optimistic.

Grad-CAM uses the selected model's shared layer4 output, separately mapped to each
actual input slice. Examples are the union of age/error rank quantiles 0/0.5/1,
with subject-ID tie breaking, chosen after metrics lock. Display maps are positive,
independently normalized and spatially coarse.

Zero positive Grad-CAM maps and attribution extending into background/nonbrain
regions are known limitations. Retain these outputs; do not substitute attractive
examples or treat a blank map as absence of model dependence.

Required interpretation wording:

“Regions receiving relatively greater attribution for the model's brain-age prediction.”

BAG and Grad-CAM are research-only and non-diagnostic. BAG is not a diagnosis.
Grad-CAM is not pathology localization or causal evidence.
IXI size, site/scanner confounding, rigid correspondence, IOP contrast limitations,
2.5D representation, single-dataset validity and lack of external/clinical validation
must accompany results. Participant outputs, weights and plots stay ignored.
