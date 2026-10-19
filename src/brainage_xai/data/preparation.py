"""Validate and select existing Phase 1 eligible subjects without recomputing eligibility."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"subject_id", "t1_path", "age", "acquisition_site", "brain_age_label_eligible"}


def eligible_subjects(manifest_path: Path) -> pd.DataFrame:
    """Read strict eligibility flags; resolve relative MRI paths against the manifest directory."""
    manifest_path = Path(manifest_path).resolve()
    frame = pd.read_csv(
        manifest_path, dtype={"subject_id": str, "t1_path": str, "acquisition_site": str}
    )
    if missing := REQUIRED_COLUMNS - set(frame):
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    flags = frame.brain_age_label_eligible.astype(str).str.strip().str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Manifest eligibility must contain only explicit True/False values")
    selected = frame.loc[flags == "true"].copy()
    if (
        selected.subject_id.isna().any()
        or not selected.subject_id.map(
            lambda value: bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value))
        ).all()
    ):
        raise ValueError("Eligible subject IDs must be non-empty safe filename identifiers")
    if selected.subject_id.duplicated().any():
        raise ValueError("Duplicate eligible subject IDs")
    age = pd.to_numeric(selected.age, errors="coerce")
    if not (np.isfinite(age) & (age > 0)).all():
        raise ValueError("Eligible subjects must have valid ages; repair the Phase 1 manifest")
    selected["age"] = age.astype(float)
    if selected.t1_path.isna().any() or selected.t1_path.str.strip().eq("").any():
        raise ValueError("Missing T1 path for eligible subject")
    paths = []
    for value in selected.t1_path:
        path = Path(value).expanduser()
        path = (manifest_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
            raise ValueError(f"Eligible T1 path must be an existing NIfTI file: {path}")
        paths.append(str(path))
    if len(paths) != len(set(paths)):
        raise ValueError("One T1 source path is assigned to multiple eligible subjects")
    selected["t1_path"] = paths
    # Missing site is explicit unknown; no site or eligibility is inferred here.
    selected["acquisition_site"] = selected.acquisition_site.fillna("unknown").str.strip()
    selected.loc[selected.acquisition_site.eq(""), "acquisition_site"] = "unknown"
    return selected.sort_values("subject_id").reset_index(drop=True)


def select_pilot(subjects: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Deterministically round-robin sites and sorted subject IDs; never alter eligibility."""
    if limit is None:
        return subjects.copy()
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    groups = [
        list(group.index)
        for _, group in subjects.sort_values("subject_id").groupby("acquisition_site", sort=True)
    ]
    indices = []
    offset = 0
    while len(indices) < min(limit, len(subjects)):
        for group in groups:
            if offset < len(group):
                indices.append(group[offset])
                if len(indices) == limit:
                    break
        offset += 1
    return subjects.loc[indices].reset_index(drop=True)
