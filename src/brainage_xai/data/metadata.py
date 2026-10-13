"""Spreadsheet parsing that preserves missing/invalid ages and duplicate records."""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_subject_id(value: Any) -> str | None:
    """Normalize integer Excel IDs or IXI-prefixed IDs; reject fractional/ambiguous IDs."""
    if pd.isna(value) or isinstance(value, (bool, np.bool_)):
        return None
    text = str(value).strip().upper()
    match = re.fullmatch(r"(?:IXI[\s_-]*)?(\d+)(?:\.0+)?", text)
    if match is None or int(match[1]) <= 0:
        return None
    return f"IXI{int(match[1]):03d}"


def normalize_metadata(
    frame: pd.DataFrame,
    *,
    id_column: str = "IXI_ID",
    age_column: str = "AGE",
    sex_column: str | None = "SEX",
    site_column: str | None = None,
    scanner_column: str | None = None,
) -> pd.DataFrame:
    """Retain every row; age must be finite and positive (no clinical thresholds)."""
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Spreadsheet contains duplicate column names")
    for column in (id_column, age_column, site_column, scanner_column):
        if column is not None and column not in frame:
            raise ValueError(f"Missing configured metadata column: {column}")
    result = pd.DataFrame(index=frame.index)
    result["metadata_row"] = np.arange(2, len(frame) + 2)
    result["raw_subject_id"] = frame[id_column]
    result["subject_id"] = frame[id_column].map(normalize_subject_id)
    raw_age = frame[age_column]
    result["raw_age"] = raw_age
    missing = raw_age.isna() | raw_age.astype(str).str.strip().eq("")
    age = pd.to_numeric(raw_age, errors="coerce")
    valid = np.isfinite(age) & (age > 0) & ~raw_age.map(lambda x: isinstance(x, (bool, np.bool_)))
    result["age"] = age.where(valid, np.nan)
    result["age_status"] = np.where(missing, "missing", np.where(valid, "valid", "invalid"))
    for field, column in (("sex", sex_column), ("site", site_column), ("scanner", scanner_column)):
        result[field] = (
            frame[column].map(lambda x: None if pd.isna(x) else (str(x).strip() or None))
            if column is not None and column in frame
            else None
        )
    return result.reset_index(drop=True)


def load_metadata(path: Path, *, sheet_name: str | int = 0, **columns: Any) -> pd.DataFrame:
    """Support legacy .xls (xlrd) and .xlsx (openpyxl); never impute ages."""
    path = Path(path)
    if path.suffix.lower() not in {".xls", ".xlsx"}:
        raise ValueError("Expected an .xls or .xlsx demographic spreadsheet")
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    return normalize_metadata(pd.read_excel(path, sheet_name=sheet_name, engine=engine), **columns)
