"""One row per identified subject, preserving duplicate ambiguity instead of choosing a scan."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .discovery import parse_subject_id


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
        row = rows.iloc[0] if len(rows) == 1 else None
        records.append(
            {
                "subject_id": subject_id,
                "t1_path": files[0] if len(files) == 1 else None,
                "t1_paths": files,
                "mri_count": len(files),
                "metadata_count": len(rows),
                "metadata_rows": rows["metadata_row"].tolist(),
                **{
                    key: row[key] if row is not None else None
                    for key in ("age", "age_status", "sex", "site", "scanner")
                },
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
    ]
    return pd.DataFrame(records, columns=columns)
