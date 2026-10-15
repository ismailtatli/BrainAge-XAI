"""Conservative subject labels derived without altering or merging source records."""

from typing import Any

import pandas as pd

LABEL_FIELDS = ("age", "sex", "dob", "study_date")


def resolve_metadata(rows: pd.DataFrame) -> dict[str, Any]:
    """Compare available labels exactly; missing ages and invalid identity dates are unsafe.

    Missing sex/dates supply no evidence of agreement or disagreement. A canonical
    value must occur in a source row. Invalid non-missing dates cannot be ignored.
    Ancillary differences include missing-versus-present values and never affect labels.
    """
    result: dict[str, Any] = {
        **dict.fromkeys((*LABEL_FIELDS, "age_status", "site", "scanner")),
        "metadata_resolution_status": "no_metadata",
        "label_conflict_fields": [],
        "ancillary_conflict_fields": [],
        "invalid_identity_metadata": False,
    }
    if rows.empty:
        return result
    conflicts = [field for field in LABEL_FIELDS if rows[field].nunique(dropna=True) > 1]
    invalid_dates = [
        field for field in ("dob", "study_date") if rows[f"{field}_status"].eq("invalid").any()
    ]
    result["invalid_identity_metadata"] = bool(invalid_dates)
    result["label_conflict_fields"] = sorted(set(conflicts + invalid_dates))
    missing_age = rows.age_status.eq("missing").any()
    invalid_age = rows.age_status.eq("invalid").any()
    if len(rows) > 1:
        ancillary = rows.attrs.get("ancillary_columns", [])
        result["ancillary_conflict_fields"] = sorted(
            column.removeprefix("source:")
            for column in ancillary
            if rows[column].nunique(dropna=False) > 1
        )
    if len(rows) > 1 and (conflicts or invalid_dates or invalid_age):
        status = "duplicate_label_ambiguous"
    elif missing_age:
        status = "missing_age"
    elif len(rows) > 1:
        status = "duplicate_label_consistent"
    else:
        status = "unique"
    result["metadata_resolution_status"] = status
    if status != "duplicate_label_ambiguous":
        for field in LABEL_FIELDS:
            available = rows[field].dropna()
            result[field] = available.iloc[0] if len(available) else None
        if missing_age or invalid_age:
            result["age"] = None
        result["age_status"] = "missing" if missing_age else "invalid" if invalid_age else "valid"
    # Ancillary fields are exposed only if the complete column agrees, including missingness.
    for field in ("site", "scanner"):
        if rows[field].nunique(dropna=False) == 1:
            result[field] = rows[field].iloc[0]
    return result
