# Release notes — draft, not published

Status: local academic research application prepared for final review. Repository
finalization commits and pushes public-safe files on the feature branch only;
no release, tag or merge is created.

## Included

Public source, synthetic tests, example/frozen configs, constrained dependencies,
local Streamlit app, shared CLI, PDF/JSON/XAI reporting and academic documentation.
Accepted Phase 1/2/3 decisions and results remain frozen. See the
[final report](final-project-report.md), [model card](model-card.md) and
[setup guide](reproducibility.md).

## Required local resources

Exact accepted checkpoint and validation correction/reference/slices, plus pinned
MNI T1/mask. These are not bundled, downloaded silently or replaced if absent.
Raw IXI access is separately required for research reproduction, not single-MRI
application operation. Source/config/version checksum mismatches fail closed.

## Deliberately excluded

Raw MRI, IXI.xls/participant metadata, subject manifests/predictions, weights,
participant XAI, local paths/configs and secrets. The release is not clinically
validated and must not be represented as diagnostic. Clinical deployment and
external validation are out of scope. Software/data redistribution terms must be
reviewed separately; no new software license, affiliation, DOI or publication
metadata is invented. CITATION.cff is deferred because verified author/publication
metadata is insufficient. Existing third-party/data obligations remain applicable.

## Versions and review gates

The package stays 0.1.0 for frozen scientific environment compatibility; application
integration metadata is 0.1.0-local. Final synthetic checks, clean-environment
smoke, local server startup, training-subject technical smoke and privacy/integrity
results must accompany review. The technical smoke is not additional performance
validation. Known BAG age dependence and zero/background Grad-CAM limitations must
remain prominent. Publishing/tagging requires a separate explicit instruction.
