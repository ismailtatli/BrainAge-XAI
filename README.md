# BrainAge-XAI

**Explainable Deep Learning for Brain Age Gap Estimation and Cognitive Aging Analysis
from Structural MRI**

BrainAge-XAI is a university / TÜBİTAK-oriented research project investigating whether
T1-weighted structural MRI can support reliable apparent brain age estimation and
transparent analysis of cognitive aging associations.

The research question is: how accurately can a model estimate structural brain age,
how stable are its attributions, and how does bias-corrected Brain Age Gap relate to
cognitive-aging measures in independent research analyses?

## Current scope and status

Phase 1 provides the Python project scaffold and local IXI dataset audit:
configuration, discovery, demographic matching, NIfTI validation, reproducible
CSV/JSON/Markdown outputs, and central-slice inspection. Synthetic tests require no
real MRI data. The completed local IXI audit found 561 eligible research subjects.

Phase 2 is complete and scientifically accepted: **561/561 subjects processed,
zero failures**, with the saved train/validation/test split of **393/84/84**.
Registration is permanently frozen as **S2/O0 rigid**. Every output uses the exact
**193×229×193, 1 mm MNI template grid**, with per-subject intensity normalization
and no crop/pad. See the [frozen preprocessing protocol](docs/preprocessing-frozen.md),
[public specification](configs/preprocessing.frozen.yaml) and
[final results and QC limitations](docs/preprocessing-final-results.md).
No critical QC blocker remains. Model slice/input selection remains deferred.

No training, dataset download, disease classification, OASIS integration, final BAG
interpretation, or final web application is implemented. Future work includes brain-age
regression, bias correction, attribution validation, healthy reference distributions,
and research reporting.

## Scientific boundaries

Apparent brain age is a model-estimated structural brain age, not literal biological
age. BAG alone is never a diagnosis. An attribution heatmap does not identify a
lesion or disease. This project supplies no clinical thresholds. Population research
associations do not establish individual causal explanations.

Later splits must be subject-level, including repeat scans. Fit preprocessing,
normalization, feature selection, and age-bias correction on training data only;
keep validation and test subjects independent. An audit match or valid image is not
an automatic declaration of eligibility for modeling.

## Architecture

```text
configs/                 Example YAML; ignored local configuration
src/brainage_xai/
  config/settings.py     Validated paths and metadata column selection
  data/discovery.py      NIfTI discovery and IXI ID parsing
  data/metadata.py       Spreadsheet parsing and age validation
  data/resolution.py     Conservative duplicate-label resolution and conflict logging
  data/matching.py       Subject-level matching and duplicate flags
  data/validation.py     Header and single-volume voxel checks
  data/audit.py          CLI, summaries and exports
  data/preparation.py    Phase 1 eligible-manifest validation and pilot selection
  data/splits.py         Deterministic subject-level train/validation/test splits
  preprocessing/         Orientation, resampling, intensity, pipeline and QC
  registration/          Frozen rigid implementation and historical comparisons
  visualization/slices.py  Orientation-aware central slices
tests/                   Synthetic fixtures and regression tests
docs/                    Audit semantics and limitations
reports/                 Policy for private generated reports
notebooks/               Reserved for later research exploration
scripts/                 Explicit checksum-pinned template setup
```

`utils/` will be added only when shared functionality is needed.
CLI entry points live beside their implementations; wrapper scripts are unnecessary.

## Installation (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

If using Conda instead, activate your Python 3.11 environment before creating the
virtual environment. Runtime dependencies support Phases 1 and 2. `xlrd` supports
legacy `.xls` spreadsheets; `openpyxl` supports `.xlsx`. SciPy is an explicit
dependency for spatial interpolation through nibabel. There is no deep-learning
framework dependency. Dependency ranges are declared in `pyproject.toml`; each audit
records the installed versions. For an exact environment record, keep
`python -m pip freeze` output locally with your research artifacts.

## Dataset policy

**The IXI dataset is not distributed with this repository.** Supply existing data
locally; these commands do not download it. MRI, participant spreadsheets, trained
weights, secrets, and large generated artifacts must never be committed. Existing
biomedical `.gitignore` protections are retained and local configuration/spreadsheet
protections are extended. Store custom output paths outside version-controlled data;
Git ignore rules cannot protect arbitrary locations you select.

The [official IXI dataset page](https://brain-development.org/ixi-dataset/) describes
T1 imaging and demographics, three acquisition sites, and its attribution/license
requirements. `acquisition_site` is parsed from the Guys/HH/IOP filename token.
Metadata site/scanner values are exported only from configured spreadsheet columns;
scanner hardware/model is never inferred from filenames or numeric acquisition tokens.

## Configure local IXI paths

From the repository root:

```bash
cp configs/ixi.example.yaml configs/ixi.local.yaml
export IXI_T1_DIR='/path/to/local/IXI-T1'
export IXI_DEMOGRAPHICS_PATH='/path/to/local/IXI.xls'
```

Replace the example paths with your own existing locations. Alternatively, edit
`configs/ixi.local.yaml` to contain local paths; that file is ignored by Git.
Relative paths resolve from the YAML file's directory, not the shell working
directory. Environment variables and `~` expand; unset variables cause explicit
errors. The default output is `reports/generated/ixi_audit/`, ignored by Git.
Output must be outside the MRI source tree and cannot contain the input spreadsheet.

Default columns are `IXI_ID`, `AGE`, and optional `SEX`, on the first worksheet.
Optional `dob_column` and `study_date_column` default to `DOB` and `STUDY_DATE`,
so the existing local IXI configuration uses these fields without modification.
Set `sex_column` to the actual spreadsheet header (for example `SEX_ID (1=m, 2=f)`).
Adjust `metadata.sheet_name` or column names for your spreadsheet. Set optional
`site_column` / `scanner_column` only if those fields actually exist. Sex values
are normalized for representation; numeric coding is not interpreted without a codebook.
Dates accept complete ISO dates and spreadsheet datetimes. Other string formats
require an explicit `metadata.date_format` containing `%Y`, `%m`, and `%d`.
Missing dates remain missing; invalid dates are flagged rather than guessed.

## Run the audit and inspect MRI

```bash
brainage-audit --config configs/ixi.local.yaml
brainage-slices /path/to/local/IXI001-Guys-0001-T1.nii.gz \
  --output reports/generated/central_slices.png
```

The audit writes `manifest.csv`, `metadata_records.csv`, `mri_validation.csv`,
`summary.json`, and `summary.md`. Re-running overwrites these named outputs.
They contain participant identifiers and local paths: keep them private.
During validation, each filename and its position in the total are printed to stderr
and flushed immediately. The completion message reports file/failure counts and the
resolved output directory.

All `.nii` / `.nii.gz` candidates below the configured T1 directory are discovered,
including malformed filenames. Matching is subject-level; duplicate scans and
metadata are flagged and retained, rather than silently choosing one record.
Missing or invalid ages are not imputed. See [audit semantics](docs/audit.md) for
count denominators, quality checks, and limitations.

Every original metadata row and column remains available under `source:*` columns
in `metadata_records.csv`, alongside normalized values and `metadata_row` provenance.
The source spreadsheet is never modified. Duplicate labels resolve only when all
available normalized AGE/SEX/DOB/STUDY_DATE values agree. Any conflicting label or
invalid identity date is excluded rather than guessed. Every duplicate row must
have a valid age for eligibility. Missing sex/dates supply no conflicting evidence;
only observed, agreeing values can be exposed as canonical labels.
Ancillary differences are logged in `ancillary_conflict_fields`; those values are
never merged into a new combined row.

`brain_age_label_eligible` requires exactly one validated T1 MRI, a demographic
match, a valid age, and a unique or consistent duplicate label with no ambiguous
identity metadata. `exclusion_reasons` records every applicable failure. This is
a dataset-curation flag, not a medical classification or a train/test split.
Summary outputs include resolution, eligibility, exclusion, and filename-site counts;
all participant-bearing outputs remain local and ignored by Git.

Slice inspection uses nibabel orientation utilities and loads only the three
required planes. Axes follow the nearest RAS+ orientation with neurological display
convention (right increases to the viewer's right). Oblique scans are not resampled;
these are inspection images, not clinical anatomical views. Images with no coded
orientation are rejected for labeled visualization.

## Tests and static checks

```bash
python -m pytest
python -m ruff check .
python -m compileall -q src
brainage-audit --help
brainage-slices --help
```

Tests generate temporary synthetic MRI and metadata, covering parsing, configuration,
duplicates, unmatched subjects, missing/invalid ages, corrupt/non-finite images,
orientation, and repeatable exports. No actual IXI participant data is required.

## Phase 2: completed preprocessing

The accepted production run is complete under ignored
`artifacts/preprocessed/final_v1/`. Its frozen protocol consumes the existing
eligible cohort and saved split without regeneration, applies S2/O0 rigid
registration and resamples native intensities once onto the full MNI grid.
All generated volumes, participant provenance and QC images remain private.
No training is performed. Verification of a local completed run is available with:

```bash
.venv/bin/python -m brainage_xai.preprocessing.final --resume --verify-only
```

The following audit/pilot commands document the earlier development workflow;
they do not replace the [frozen production protocol](docs/preprocessing-frozen.md).

```bash
cp configs/preprocessing.example.yaml configs/preprocessing.local.yaml
brainage-preprocess-audit --config configs/preprocessing.local.yaml
brainage-geometry-audit --config configs/preprocessing.local.yaml
brainage-preprocess --config configs/preprocessing.local.yaml --limit 6
brainage-split --config configs/preprocessing.local.yaml
```

Use the existing local Phase 1 manifest. Only subjects already marked eligible
enter Phase 2. The spatial audit reads headers for all eligible subjects, without
writing full processed volumes. The six-subject pilot cycles across sites and saves
NIfTI, provenance, and orthogonal QC under ignored `artifacts/preprocessed/phase2/`.
These pilot outputs are separate from the accepted production outputs.

Raw MRI remains immutable. Canonical RAS+ orientation is not nonlinear registration;
resampling creates no new anatomical information. Intensity normalization is
per-subject foreground percentile clipping/z-scoring, with zero background. The
foreground threshold is not skull stripping. Subject splits precede future slice
extraction and must remain disjoint to prevent leakage. This is research
infrastructure, with no diagnosis, prediction, BAG calculation, or model training.

See [the Phase 2 document](docs/preprocessing.md) for algorithms, output semantics,
limitations and historical geometry diagnostics. The frozen full-grid production
specification supersedes its earlier pending geometry choices.
The geometry audit separates obliquity from affine non-orthogonality and measures
non-background extents/positions across sites without saving processed MRI volumes.
Its 95/99/100% tensor-size candidates are diagnostics only, including predicted
foreground losses under centered cropping. Canonical RAS+ orientation and 1 mm
sampling do not guarantee anatomical voxel-wise alignment between subjects.

## Registration decision and experiment history

The [final rigid-versus-affine decision](docs/rigid-vs-affine-study-results.md)
retains S2/O0 rigid registration. Higher sampling and affine registration were
evaluated; registration tuning is closed. The
[frozen registration specification](configs/registration.frozen.yaml) records the
unchanged algorithm and its source checksums. The preprocessing specification
separately freezes the full output grid. Earlier experiment protocols below
describe their original scope, rather than additional work to run.

The [registration evaluation](docs/rigid-registration.md) adds an isolated
18-subject SimpleITK Euler3D pilot against the checksum-pinned local TemplateFlow
MNI152NLin2009cAsym 1 mm T1 reference. It compares the existing baseline with
rotation/translation-only alignment, saves private quantitative/visual QC, and
performs one final interpolation directly from native intensities. It does not
register all subjects, train a model, or freeze tensor geometry. Template retrieval
is an explicit setup action; normal runtime and synthetic tests need no network.

The controlled 18-subject rigid tuning experiment is documented in
[rigid-registration-tuning.md](docs/rigid-registration-tuning.md). It preserves
baseline v1 and compares initialization and fixed-mask use with a shared two-stage
Euler3D protocol; generated comparisons remain private under ignored artifacts.

The corrected experiment results and recommendation are in
[rigid-registration-tuning-results.md](docs/rigid-registration-tuning-results.md).

The completed [optimizer-only study](docs/rigid-optimizer-study.md) held corrected
condition A fixed and compared three predefined global optimizer configurations
on the same 18 subjects, followed by an 18-subject repeat of the selected candidate.
