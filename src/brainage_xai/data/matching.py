"""One row per identified subject, preserving duplicate ambiguity instead of choosing a scan."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .discovery import parse_acquisition_site, parse_subject_id
from .resolution import resolve_metadata


def build_manifest(
    paths: list[Path], metadata: pd.DataFrame, validation: list[dict[str, Any]] | None = None
) -> pd.DataFrame:
    """Aggregate subjects with paths/record counts; never resolve duplicates silently."""
    scans: dict[str, list[str]] = defaultdict(list)
    invalid = []
    for path in paths:
        subject_id = parse_subject_id(path)
        if subject_id is None:
            invalid.append(str(path))
        else:
            scans[subject_id].append(str(path))
    checks = {v["t1_path"]: v for v in validation or []}
    records = []
    ids = set(scans) | set(metadata["subject_id"].dropna())
    for subject_id in sorted(ids):
        rows = metadata.loc[metadata["subject_id"] == subject_id]
        files = sorted(scans.get(subject_id, []))
        flags = []
        status = (
            "matched"
            if files and len(rows)
            else ("mri_without_demographics" if files else "demographics_without_mri")
        )
        if len(files) > 1:
            flags.append("duplicate_mri")
        if len(rows) > 1:
            flags.append("duplicate_subject_id")
        if (rows["age_status"] == "missing").any():
            flags.append("missing_age")
        if (rows["age_status"] == "invalid").any():
            flags.append("invalid_age")
        if any(not checks[f]["valid"] for f in files if f in checks):
            flags.append("mri_validation_failure")
        if any(
            any(e.startswith("unreadable_mri") for e in checks[f]["errors"])
            for f in files
            if f in checks
        ):
            flags.append("unreadable_mri")
        resolved = resolve_metadata(rows)
        sites = sorted({site for f in files if (site := parse_acquisition_site(f)) is not None})
        exclusions = []
        if not files:
            exclusions.append("no_mri")
        elif len(files) > 1:
            exclusions.append("duplicate_mri")
        if not len(rows):
            exclusions.append("no_demographics")
        else:
            if rows.age_status.eq("missing").any():
                exclusions.append("missing_age")
            if rows.age_status.eq("invalid").any():
                exclusions.append("invalid_age")
            if resolved["metadata_resolution_status"] == "duplicate_label_ambiguous":
                exclusions.append("ambiguous_metadata_label")
            if resolved["invalid_identity_metadata"]:
                exclusions.append("invalid_identity_metadata")
        if any(f not in checks for f in files):
            exclusions.append("mri_not_validated")
        if any(not checks[f]["valid"] for f in files if f in checks):
            exclusions.append("mri_validation_failure")
        records.append(
            {
                "subject_id": subject_id,
                "t1_path": files[0] if len(files) == 1 else None,
                "t1_paths": files,
                "mri_count": len(files),
                "metadata_count": len(rows),
                "metadata_rows": rows["metadata_row"].tolist(),
                **resolved,
                "acquisition_site": sites[0] if len(sites) == 1 else None,
                "acquisition_sites": sites,
                "brain_age_label_eligible": not exclusions,
                "exclusion_reasons": exclusions,
                "matching_status": status,
                "issues": flags,
            }
        )
    for path in sorted(invalid):
        records.append(
            {
                "subject_id": None,
                "t1_path": path,
                "t1_paths": [path],
                "mri_count": 1,
                "metadata_count": 0,
                "metadata_rows": [],
                "age": None,
                "age_status": None,
                "sex": None,
                "site": None,
                "scanner": None,
                "dob": None,
                "study_date": None,
                "metadata_resolution_status": "no_metadata",
                "label_conflict_fields": [],
                "ancillary_conflict_fields": [],
                "invalid_identity_metadata": False,
                "acquisition_site": parse_acquisition_site(path),
                "acquisition_sites": [],
                "brain_age_label_eligible": False,
                "exclusion_reasons": ["invalid_subject_id", "no_demographics"],
                "matching_status": "invalid_mri_filename",
                "issues": ["invalid_subject_id"],
            }
        )
    columns = [
        "subject_id",
        "t1_path",
        "t1_paths",
        "mri_count",
        "metadata_count",
        "metadata_rows",
        "age",
        "age_status",
        "sex",
        "site",
        "scanner",
        "matching_status",
        "issues",
        "dob",
        "study_date",
        "metadata_resolution_status",
        "label_conflict_fields",
        "ancillary_conflict_fields",
        "invalid_identity_metadata",
        "acquisition_site",
        "acquisition_sites",
        "brain_age_label_eligible",
        "exclusion_reasons",
    ]
    return pd.DataFrame(records, columns=columns)
