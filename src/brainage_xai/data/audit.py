"""Reproducible local audit and label-curation outputs without source mutation."""

import argparse
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from brainage_xai.config import AuditConfig, load_config

from .discovery import discover_t1_files, parse_acquisition_site
from .matching import build_manifest
from .metadata import load_metadata
from .validation import validate_nifti


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.dropna().value_counts().sort_index().items()}


def summarize(
    manifest: pd.DataFrame, metadata: pd.DataFrame, checks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Describe counts with explicit denominators; ambiguous records are excluded from stats."""
    subjects = manifest.loc[manifest.subject_id.notna()]
    matched = subjects.loc[subjects.matching_status == "matched"]
    unique = matched.loc[matched.metadata_count == 1]
    ages = unique.age.dropna().astype(float)
    eligible = subjects.brain_age_label_eligible.astype(bool)
    return {
        "discovered_mri_count": len(checks),
        "identified_subject_count": len(subjects),
        "matched_subject_count": len(matched),
        "unmatched_subject_count": int((subjects.matching_status != "matched").sum()),
        "matching_status_counts": _counts(manifest.matching_status),
        "metadata_record_count": len(metadata),
        "invalid_metadata_id_count": int(metadata.subject_id.isna().sum()),
        "metadata_age_status_counts": _counts(metadata.age_status),
        "valid_age_count": len(ages),
        "missing_age_count": int((unique.age_status == "missing").sum()),
        "invalid_age_count": int((unique.age_status == "invalid").sum()),
        "age_statistics": {
            key: float(getattr(ages, key)()) if len(ages) else None
            for key in ("min", "max", "mean", "median")
        },
        "statistics_population": "matched subjects with exactly one demographic record; "
        "age statistics use valid ages only; MRI quality is not an exclusion",
        "sex_distribution": _counts(unique.sex),
        "site_distribution": _counts(unique.site),
        "scanner_distribution": _counts(unique.scanner),
        "duplicate_mri_subject_count": int((subjects.mri_count > 1).sum()),
        "duplicate_metadata_subject_count": int((subjects.metadata_count > 1).sum()),
        "brain_age_label_eligible_count": int(eligible.sum()),
        "brain_age_label_ineligible_count": int((~eligible).sum()),
        "metadata_resolution_status_counts": _counts(subjects.metadata_resolution_status),
        "duplicate_label_consistent_subject_count": int(
            subjects.metadata_resolution_status.eq("duplicate_label_consistent").sum()
        ),
        "duplicate_label_ambiguous_subject_count": int(
            subjects.metadata_resolution_status.eq("duplicate_label_ambiguous").sum()
        ),
        "eligibility_exclusion_reason_counts": dict(
            sorted(
                Counter(
                    reason for reasons in subjects.exclusion_reasons for reason in reasons
                ).items()
            )
        ),
        "ancillary_conflict_subject_count": sum(
            bool(x) for x in subjects.ancillary_conflict_fields
        ),
        "acquisition_site_distribution": dict(
            sorted(
                Counter(
                    site
                    for c in checks
                    if (site := parse_acquisition_site(c["t1_path"])) is not None
                ).items()
            )
        ),
        "unknown_acquisition_site_mri_count": sum(
            parse_acquisition_site(c["t1_path"]) is None for c in checks
        ),
        "subjects_with_missing_age_count": sum("missing_age" in x for x in subjects.issues),
        "shape_distribution": dict(
            sorted(
                Counter(
                    "x".join(map(str, c["shape"])) for c in checks if c["shape"] is not None
                ).items()
            )
        ),
        "validation_failure_count": sum(not c["valid"] for c in checks),
        "validation_warning_counts": dict(
            sorted(Counter(warning for c in checks for warning in c["warnings"]).items())
        ),
        "validation_failures": [
            {"t1_path": c["t1_path"], "errors": c["errors"]} for c in checks if not c["valid"]
        ],
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    export = frame.copy()
    for column in export:
        export[column] = export[column].map(
            lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (list, dict)) else x
        )
    export.to_csv(path, index=False)


def run_audit(config: AuditConfig) -> dict[str, Any]:
    """Write deterministic CSV/JSON/Markdown results in the validated local output directory."""
    paths = discover_t1_files(config.t1_dir)
    metadata = load_metadata(config.demographics_path, **config.metadata)
    checks = []
    for index, path in enumerate(paths, start=1):
        print(f"Validating MRI {index}/{len(paths)}: {path.name}", file=sys.stderr, flush=True)
        checks.append(validate_nifti(path))
    manifest = build_manifest(paths, metadata, checks)
    summary = summarize(manifest, metadata, checks)
    summary["provenance"] = {
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("brainage-xai", "numpy", "pandas", "nibabel", "PyYAML", "openpyxl", "xlrd")
        },
        "t1_dir": str(config.t1_dir),
        "demographics_path": str(config.demographics_path),
        "metadata_config": config.metadata,
    }
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(manifest, output / "manifest.csv")
    _write_csv(metadata, output / "metadata_records.csv")
    _write_csv(
        pd.DataFrame(
            checks,
            columns=list(checks[0])
            if checks
            else ["t1_path", "valid", "errors", "warnings", "shape", "ndim"],
        ),
        output / "mri_validation.csv",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# IXI research dataset audit",
        "",
        "Research inspection only; no diagnosis.",
        "",
        "Counts and distributions:",
        "",
    ]
    for key, value in summary.items():
        if key not in {"provenance", "validation_failures"}:
            lines.append(f"- **{key}**: {json.dumps(value, sort_keys=True)}")
    lines.extend(
        [
            "",
            "See mri_validation.csv for failures/warnings and metadata_records.csv "
            "for all ages, duplicate records, and invalid IDs.",
            "",
            "Outputs contain participant information and local paths; keep them private.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local IXI T1 MRI and demographic metadata")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        summary = run_audit(config)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    print(
        f"Audited {summary['discovered_mri_count']} MRI files; "
        f"{summary['validation_failure_count']} validation failures. "
        f"Output directory: {config.output_dir}"
    )


if __name__ == "__main__":
    main()
