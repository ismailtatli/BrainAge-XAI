"""Configuration paths are resolved relative to the YAML file, never the working directory."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AuditConfig:
    t1_dir: Path
    demographics_path: Path
    output_dir: Path
    metadata: dict[str, Any]


def load_config(path: Path) -> AuditConfig:
    """Load configuration and reject missing inputs or unsafe output locations."""
    path = Path(path).resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(raw) - {"t1_dir", "demographics_path", "output_dir", "metadata"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(map(str, unknown))}")

    def resolve(key: str) -> Path:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty path string")
        value = os.path.expandvars(value)
        if re.search(r"\$(?:\w+|\{[^}]+\})", value):
            raise ValueError(f"{key} contains an unset environment variable")
        result = Path(value).expanduser()
        return (path.parent / result).resolve() if not result.is_absolute() else result.resolve()

    t1, demographics, output = (resolve(k) for k in ("t1_dir", "demographics_path", "output_dir"))
    if not t1.is_dir():
        raise ValueError(f"T1 directory does not exist: {t1}")
    if not demographics.is_file() or demographics.suffix.lower() not in {".xls", ".xlsx"}:
        raise ValueError("demographics_path must be an existing .xls or .xlsx spreadsheet")
    if output == t1 or t1 in output.parents:
        raise ValueError("output_dir must be outside the source MRI directory")
    if output.exists() and not output.is_dir():
        raise ValueError("output_dir must be a directory")
    if output == demographics or output in demographics.parents:
        raise ValueError("output_dir must not contain the source spreadsheet")
    metadata = raw.get("metadata", {})
    allowed = {
        "sheet_name",
        "id_column",
        "age_column",
        "sex_column",
        "site_column",
        "scanner_column",
        "dob_column",
        "study_date_column",
        "date_format",
    }
    if not isinstance(metadata, dict) or set(metadata) - allowed:
        raise ValueError("metadata must be a mapping with supported column/sheet keys")
    for key, value in metadata.items():
        if key == "sheet_name":
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError("sheet_name must be a sheet name or non-negative index")
            if isinstance(value, int) and value < 0:
                raise ValueError("sheet_name must be non-negative")
        elif value is None and key in {
            "sex_column",
            "site_column",
            "scanner_column",
            "dob_column",
            "study_date_column",
            "date_format",
        }:
            continue
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty column name")
        elif key == "date_format" and not all(token in value for token in ("%Y", "%m", "%d")):
            raise ValueError("date_format must include %Y, %m and %d")
    return AuditConfig(t1, demographics, output, metadata)
