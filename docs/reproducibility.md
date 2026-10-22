# Reproducibility and local setup

The public repository contains scientific code, synthetic tests, aggregate results
and public-safe configurations. It does **not** contain IXI images/metadata,
participant manifests/predictions, templates or trained checkpoints. Reproducing
the measured IXI study requires separately obtained data and local artifacts;
cloning the repository alone cannot reproduce those results or run real inference.

## Environment

Use Python **3.11.17**. Exact patch/package checks protect the accepted runtime;
newer versions are not silently substituted. Package version stays 0.1.0 because
it is recorded in the frozen Phase 2 specification; the local integration version
is 0.1.0-local. No release/tag is created by this integration.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c configs/runtime-constraints.txt -e '.[dev,modeling,app,app-test]'
python -m pip check
```

Ensure `python --version` actually reports 3.11.17; an arbitrary `python3.11`
installation may have a different patch version. The constraints file records the
verified numerical/modeling/UI/PDF/testing packages. Provenance also records
installed versions. Cross-platform/backend bitwise model identity is not promised;
scientific runtime checks reject unverified software versions. MPS is selected by
`device: auto` when available, otherwise CPU with an explicit result warning.
`device: mps` fails if unavailable. No device choice changes the checkpoint.

## Data access and research reproduction

Obtain T1 images and the demographics spreadsheet separately from the
[official IXI dataset page](https://brain-development.org/ixi-dataset/), follow its
access/license/attribution conditions and keep the files private. The official
page specifies CC BY-SA 3.0 and requests source acknowledgement. This repository
neither redistributes the data nor confers new permissions. Three acquisition
sites introduce scanner/site confounding. Project counts below are measurements
from the accepted local study, not promised download counts.

1. Configure existing local MRI/spreadsheet paths using
   `configs/ixi.example.yaml` → ignored `configs/ixi.local.yaml`.
2. Phase 1: `brainage-audit --config configs/ixi.local.yaml`. See
   [audit semantics](audit.md). The accepted local audit discovered 581 T1 files
   and 561 label-eligible subjects; it did not impute invalid/ambiguous labels.
3. Phase 2's [historical workflow](preprocessing.md) audits geometry and creates
   subject splits. **For this accepted study, consume existing membership; never
   regenerate its 393/84/84 split.** Existing [frozen preprocessing](preprocessing-frozen.md)
   uses S2/O0 rigid, the full MNI grid and subject-local normalization. Template
   setup is explicit: `.venv/bin/python scripts/fetch_mni_template.py`.
4. The accepted full dataset is ignored `artifacts/preprocessed/final_v1/`.
   Its [results](preprocessing-final-results.md) document 561 successes and zero
   failures. Do not rerun/alter it for application integration.
5. [Phase 3 protocol](phase3-modeling.md) documents the two predefined models,
   validation-only selection/calibration and consumed single-use test gate.
   [Accepted results](phase3-scientific-ml-results.md) contain the scientific
   evidence. Training/evaluation commands there are historical reproduction
   instructions; the accepted run must not be retrained or re-evaluated.
6. For **new-MRI inference**, install the accepted local runtime artifacts and
   follow [demo/CLI setup](demo.md). It requires no cohort spreadsheet/identifiers.
   A new independent reproduction is not the accepted frozen run and would require
   explicit research authorization, appropriate artifact ownership and all leakage
   gates; this integration does not start one.

## Artifact verification

`configs/inference.frozen.yaml` records accepted artifact/config/source hashes,
aggregate observed training/reference age ranges and performance context.
Its own digest is anchored in the inference loader. Local paths are set only in
ignored `configs/inference.local.yaml`. Checkpoint, correction, reference, slices,
template and mask hashes are verified before inference. There is no fallback or
implicit model/template/dataset download. Absence of the private accepted weights
is a real runtime prerequisite, not a reason to retrain automatically.

An individual inference returns anonymous analysis-ID provenance with input hash,
scientific/integration source hashes, package versions, device, configs and
reference/checkpoint hashes. It excludes participant identities, source filename,
local absolute paths and calibration subject-ID lists. Report JSON/PDF describe
research-model reference context and empirical model uncertainty; neither is a
clinical interval or diagnosis.

## Validation and privacy

```sh
.venv/bin/python -m pytest -q
ruff check .
.venv/bin/python -m compileall -q src app
.venv/bin/python -m pip check
```

Tests use synthetic temporary MRI/reference/checkpoint fixtures and do not download
weights or evaluate actual participants. Application tests require `app` and
`app-test` extras. The isolated-environment smoke, local-server health check and
one training-subject end-to-end smoke are engineering verification, not additional
study performance measurements. Their detailed outputs stay ignored under
`artifacts/integration/`.

The accepted checkpoint, test-prediction hash, calibration/reference hashes and
single test claim are protected. Raw MRI, IXI.xls, templates, final_v1 and Phase 3
artifacts are verified unchanged. Generated outputs must be Git-ignored. Never
commit participant data, model weights, derived private images or local configs.
Do not replace synthetic fixtures with real MRI. No software license or DOI is
introduced here; third-party/data terms remain applicable.
