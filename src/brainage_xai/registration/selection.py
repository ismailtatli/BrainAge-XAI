"""Deterministic eighteen-subject site/geometry challenge sample, with no duplicated IDs."""

import json

import numpy as np
import pandas as pd


def select_registration_pilot(eligible: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """Six per site: two low, two median, two high obliquity, prioritizing offset tails.

    Each angle-sorted site is partitioned into contiguous thirds. Within each third,
    choose the largest absolute physical center-offset norm first, then the subject
    nearest that third's low/median/high representative angle (ID breaks ties).
    Small thirds fill from angle-nearest remaining subjects; never duplicate a subject.
    """
    columns = {
        "subject_id",
        "max_obliquity_deg",
        "foreground_center_offset_mm",
        "diagnostic_status",
    }
    if missing := columns - set(geometry):
        raise ValueError(f"Missing geometry selection columns: {sorted(missing)}")
    if eligible.subject_id.duplicated().any() or geometry.subject_id.duplicated().any():
        raise ValueError("Pilot input subject IDs must be unique")
    if not eligible.brain_age_label_eligible.astype(str).str.lower().eq("true").all():
        raise ValueError("Pilot requires only Phase 1 eligible subjects")
    fields = ["subject_id", "max_obliquity_deg", "foreground_center_offset_mm", "diagnostic_status"]
    joined = eligible.merge(geometry[fields], on="subject_id", how="left", validate="one_to_one")
    if joined.diagnostic_status.isna().any() or not joined.diagnostic_status.eq("success").all():
        raise ValueError("Complete successful geometry diagnostics required for pilot selection")
    angles = pd.to_numeric(joined.max_obliquity_deg, errors="coerce")
    if not np.isfinite(angles).all():
        raise ValueError("Invalid obliquity values")

    def norm(value):
        vector = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("Invalid center offset vector")
        return float(np.linalg.norm(vector))

    joined = joined.assign(
        max_obliquity_deg=angles, center_offset_norm_mm=joined.foreground_center_offset_mm.map(norm)
    )
    result = []
    for site in ("Guys", "HH", "IOP"):
        group = (
            joined.loc[joined.acquisition_site == site]
            .sort_values(["max_obliquity_deg", "subject_id"])
            .reset_index(drop=True)
        )
        if len(group) < 6:
            raise ValueError(f"Need at least six eligible diagnostic subjects at {site}")
        used = set()
        for third, label in zip(
            np.array_split(np.arange(len(group)), 3), ("low", "median", "high")
        ):
            pool = group.loc[third]
            representative = float(
                pool.max_obliquity_deg.min()
                if label == "low"
                else pool.max_obliquity_deg.max()
                if label == "high"
                else pool.max_obliquity_deg.median()
            )
            tail = pool.sort_values(
                ["center_offset_norm_mm", "subject_id"], ascending=[False, True]
            ).iloc[0]
            candidates = pool.loc[~pool.subject_id.isin({tail.subject_id})].copy()
            if candidates.empty:
                candidates = group.loc[~group.subject_id.isin(used | {tail.subject_id})].copy()
            candidates["angle_distance"] = (candidates.max_obliquity_deg - representative).abs()
            angle = candidates.sort_values(["angle_distance", "subject_id"]).iloc[0]
            for row, reason in ((tail, "offset_tail"), (angle, "angle_representative")):
                record = row.to_dict()
                record["selection_stratum"] = label
                record["selection_reason"] = reason
                result.append(record)
                used.add(row.subject_id)
    frame = pd.DataFrame(result)
    if frame.subject_id.duplicated().any() or len(frame) != 18:
        raise ValueError("Pilot selection produced duplicate subjects or unexpected coverage")
    return frame.drop(columns=["angle_distance"], errors="ignore")
