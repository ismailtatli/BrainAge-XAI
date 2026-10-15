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

Phase 1 implements the Python project scaffold and local IXI dataset audit:
configuration, discovery, demographic matching, NIfTI validation, reproducible
CSV/JSON/Markdown outputs, and central-slice inspection. Synthetic tests require no
real MRI data. Real-dataset results remain to be reviewed after local configuration.

No training, dataset download, disease classification, OASIS integration, final BAG
interpretation, or final web application is implemented. Future work includes MRI
preprocessing, subject-level splitting, regression, bias correction, attribution
validation, healthy reference distributions, and research reporting.

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
  visualization/slices.py  Orientation-aware central slices
tests/                   Synthetic fixtures and regression tests
docs/                    Audit semantics and limitations
reports/                 Policy for private generated reports
notebooks/               Reserved for later research exploration
scripts/                 Reserved for future orchestration scripts
```

`preprocessing/` and `utils/` will be added when they contain real functionality.
CLI entry points live beside their implementations; wrapper scripts are unnecessary.

## Installation (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

If using Conda instead, activate your Python 3.11 environment before creating the
virtual environment. Runtime dependencies are limited to Phase 1. `xlrd` supports
legacy `.xls` spreadsheets; `openpyxl` supports `.xlsx`. There is no deep-learning
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
