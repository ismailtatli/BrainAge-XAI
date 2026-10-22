# Application integration validation

Validated locally on 2026-10-22. These are engineering checks, not new scientific
experiments or performance estimates. The accepted model, preprocessing,
registration, split, correction and locked evaluation remain unchanged.

| Check | Result |
|---|---|
| Full synthetic/regression suite | 315 tests passed |
| Isolated environment application suite | 31 tests passed |
| Isolated environment install/CLI help/pip check | Passed |
| Ruff | Passed |
| compileall (`src app`) | Passed |
| Main-environment pip check | Passed |
| Streamlit AppTest startup/result/download/error behavior | Passed |
| Loopback server `/_stcore/health` | HTTP 200, `ok`; server stopped afterward |
| Local documentation file links | Passed |
| One TRAIN-subject raw-to-report smoke | Passed, 39.12 seconds |
| Frozen Phase 2 normalized voxel parity for smoke | Exact match |
| Smoke outputs | JSON, three-page PDF, 15 mapped Grad-CAM images/arrays |
| Temporary MRI cleanup | Passed; no source/processed NIfTI exported |
| Frozen input integrity | 1,800 protected files, including 581 raw MRIs, unchanged |
| Accepted Phase 3 artifacts/resources | 1,212 files unchanged |
| Locked-test gate | Still exactly one evaluation |

The isolated environment was created separately and installed from the public
constrained extras; it did not reuse the main environment's site-packages.
Synthetic fixtures contain no real participants or downloaded weights.

The real smoke used one deterministic training subject only, never a test subject.
It verified raw input through unchanged processing, MNI geometry, accepted model,
frozen BAG/reference, all-slice Grad-CAM, PDF/JSON and cleanup. Its subject identity,
individual estimates and participant-derived outputs are private and ignored.
No smoke predictions entered scientific metrics. Subsequent visual review used
saved normalized training inputs and attribution arrays only, without another
registration/model pass. Image-title bounds and PDF image aspect ratios were fixed;
raw/corrected quantities and raw-prediction-based empirical interval labels were
reviewed, as was removal of stale results after an incomplete new submission.

Final privacy checks cover tracked and candidate public files: no MRI, participant
exports/manifests, generated participant XAI, weights, local configs, personal home
paths or detected secrets. Scientific source/config diffs are empty. New inference
outputs remain under ignored artifacts. Repository finalization stages and commits
only public-safe application/source/tests/configs/documentation on the feature
branch; no merge/release/tag is created. The public repository still requires
separately provided accepted runtime artifacts for real use. The release notes are
a draft, not a published release or clinical approval.

The final repository audit repeated all 315 tests (15.86 seconds), Ruff, compileall
and pip check successfully. All 51 local documentation links and 12 external
references passed; 26 shell examples passed syntax checks and 15 CLI entry/module
help checks passed without executing scientific workflows. Streamlit again returned
HTTP 200 from its loopback health endpoint and was stopped. SHA-256 verification
confirmed the 1,800 protected inputs and 1,212 accepted Phase 3 files unchanged;
the locked-test gate remained at exactly one evaluation. Accepted aggregate
numbers, raw/corrected terminology and research-only interpretation were checked
against the frozen results.

Detailed machine-readable test, integrity and smoke records are kept locally under
`artifacts/integration/final_application_v1/` and are not distributed.
