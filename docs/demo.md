# Local research demonstration

BrainAge-XAI is a research prototype, not a diagnostic application. The application
uses the accepted multi-view B checkpoint and frozen preprocessing, registration,
BAG correction and validation reference. It never trains, selects a model or
repeats the locked test. No accounts, database or cloud upload are implemented.

## Setup

Use Python **3.11.17** and the [reproducibility guide](reproducibility.md). From the
repository root, activate the environment and install the constrained extras:

```sh
python -m pip install -c configs/runtime-constraints.txt -e '.[dev,modeling,app,app-test]'
cp configs/inference.example.yaml configs/inference.local.yaml
```

Set the paths in the ignored local config. Paths resolve relative to the YAML
file. Required local resources are:

- Accepted B checkpoint, SHA-256 `23a57975e318647636c12dd23ceb087e826d2c3ea7f2e89ec4fe9be30922f51d`.
- Accepted `bias_correction.json`, `reference.json` and `slices.json` from the
  private Phase 3 run. Their checksums are pinned in `configs/inference.frozen.yaml`.
- The pinned MNI152NLin2009cAsym 1 mm T1 template and binary brain mask.
- Original frozen public configs and scientific source files, unchanged.

These resources are not distributed in Git. There is no automatic download or
fallback to a new/untrained checkpoint. Missing or different resources produce a
setup error. Template retrieval remains the explicit, checksum-verified
`scripts/fetch_mni_template.py` setup command. No IXI filenames or full cohort
manifest are required for single-subject application use. Aggregate training and
validation age ranges are in the public runtime contract.

## Streamlit

```sh
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Open `http://127.0.0.1:8501`. The checked-in Streamlit configuration binds to
loopback, disables usage telemetry and static file serving, and sets a 512 MiB
upload cap. Do not expose this unauthenticated research prototype to a network.
For another local config, set `BRAINAGE_INFERENCE_CONFIG` before launch.

1. Supply one T1-weighted `.nii` or `.nii.gz` and chronological age in years.
2. Run the analysis; status messages show validation, rigid registration,
   normalization, model inference, Grad-CAM and reporting.
3. Inspect the raw and corrected estimates, BAG/reference context and QC warnings.
4. Inspect the representative three views and expand the complete 15-slice gallery.
5. Explicitly download the PDF, JSON or complete XAI package if required.
6. Clear the analysis when finished.

Chronological age is required for BAG and reference reporting; it is **not an MRI
model input**. Filename, participant identity, site and sex do not affect prediction.
T1 modality and population similarity cannot be verified automatically. Inputs
outside observed age ranges are allowed but prominently flagged as extrapolative.
The nonnegative-age rule and byte/voxel caps are data/resource checks, not clinical
acceptance thresholds. The default voxel cap is 64 million, checked before reading
image data; the byte and voxel caps can be configured locally.

## CLI and programmatic use

```sh
brainage-infer --input /path/to/new-t1.nii.gz --age 50 \
  --config configs/inference.local.yaml \
  --output artifacts/inference/results/new-analysis
```

The output must be new; existing analyses are never overwritten. Paths inside a
Git repository must be ignored. Protected scientific artifact/source directories
and the input MRI tree cannot be output/temp destinations. Omit `--output` to
create an anonymous analysis-ID directory under configured `output_dir`.
Progress goes to stderr; stdout contains an analysis-ID/output JSON receipt.
Failures return code 2 and never fabricate a completed result.

```python
from brainage_xai.inference.pipeline import infer_file, export_bundle

bundle = infer_file('/path/to/new-t1.nii.gz', 50.0, 'configs/inference.local.yaml')
result = bundle.result.as_dict()
# Export is optional; inference itself retains only result/derived images in memory.
export_bundle(bundle, 'artifacts/inference/results/new-analysis')
```

Both interfaces call the same inference service. Output includes `result.json`,
`report.pdf`, `provenance.json`, 15 labeled XAI PNGs, raw Grad-CAM arrays and export
checksums. JSON and PDF agree on raw/corrected quantities and reference context.
Representative panels always use each view's middle fixed slice; selection is
independent of attribution magnitude. No participant filename or source NIfTI is
embedded. MRI-derived images remain sensitive even without identifiers.

## Interpretation and limits

Raw BAG is prediction minus age. Corrected BAG subtracts the frozen validation
age trend; corrected brain age is age plus corrected BAG, **not a medical
measurement**. Positive/negative BAG means older-/younger-looking relative to the
learned research reference; there are no healthy/unhealthy or disease-risk cutoffs.

The validation reference percentile is not a clinical norm. The empirical interval
uses validation residual quantiles and does not guarantee conditional coverage.
The 95% bootstrap CI for **overall test MAE** is a different, cohort-level quantity.
Corrected test BAG retains age association **r = -0.279**; correction does not
guarantee age-independence. Zero maps and nonbrain/background attribution are known
Grad-CAM limitations, not evidence of absence/presence of disease.

## Privacy and cleanup

MRI bytes are copied to a private temporary directory with a generated name.
Normal success and exception paths remove that directory before returning. No
uploaded NIfTI or processed NIfTI is exported. Streamlit clears the upload widget
after processing; result/XAI/download bytes remain only in its session memory
until cleared/disconnected. CLI outputs are intentionally persisted at the
explicit local destination. The app has no persistent patient database or upload
cache. Font caches contain no MRI data. Abrupt process termination/power loss
cannot guarantee Python cleanup; inspect/remove orphaned `brainage-*` folders in
the configured temporary directory after such a failure. Do not store uploads in
cloud-synchronized directories. Downloads/derived images need the same privacy
care as other research participant artifacts.
