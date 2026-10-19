"""Read-only per-subject geometry/foreground diagnostics; never saves MRI volumes."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from brainage_xai.config.preprocessing import (
    PreprocessingConfig,
    load_preprocessing_config,
    validate_output_directory,
)
from brainage_xai.data.preparation import eligible_subjects, select_pilot

from .geometry import foreground_extent, geometry_summary
from .orientation import canonicalize
from .pipeline import inspect_subject
from .resampling import resample_t1


def diagnose_subject(subject: dict, config: PreprocessingConfig) -> dict[str, Any]:
    """Read one MRI, reproduce the existing resampling and baseline mask, then release arrays."""
    record = inspect_subject(subject, config)
    record["diagnostic_status"] = "failed"
    record["normalization_method"] = "not_run_foreground_extent_audit"
    record["output_path"] = None
    if record["preprocessing_status"] != "success":
        return record
    try:
        canonical = canonicalize(nib.load(subject["t1_path"], mmap=True))
        resampled = resample_t1(canonical, config.target_spacing_mm)
        record.update(
            foreground_extent(
                np.asarray(resampled.dataobj),
                nib.affines.voxel_sizes(resampled.affine),
                background_threshold=config.background_threshold,
            )
        )
        record["diagnostic_status"] = "success"
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    return record


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_diagnostics(config: PreprocessingConfig, *, limit: int | None = None) -> dict[str, Any]:
    """Write private diagnostics and a pilot supplement, preserving existing pilot files."""
    subjects = eligible_subjects(config.manifest_path)
    validate_output_directory(
        config.output_dir, [config.manifest_path, *(Path(p) for p in subjects.t1_path)]
    )
    selected = select_pilot(subjects, limit)
    output = config.output_dir / "geometry_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    per_subject = output / "subjects"
    per_subject.mkdir(exist_ok=True)
    records = []
    for index, subject in enumerate(selected.to_dict("records"), start=1):
        print(
            f"Diagnosing MRI {index}/{len(selected)}: {subject['subject_id']}",
            file=sys.stderr,
            flush=True,
        )
        record = diagnose_subject(subject, config)
        _write_json(per_subject / f"{subject['subject_id']}.json", record)
        records.append(record)
    summary = geometry_summary(records)
    summary["eligible_subjects_seen"] = len(subjects)
    summary["selected_subject_count"] = len(selected)
    summary["site_summaries"] = {
        site: geometry_summary([r for r in records if r["acquisition_site"] == site])
        for site in sorted({r["acquisition_site"] for r in records})
    }
    summary["warning_counts"] = dict(
        sorted(Counter(w for r in records for w in r["warnings"]).items())
    )
    summary["error_type_counts"] = dict(
        sorted(Counter(error.split(":", 1)[0] for r in records for error in r["errors"]).items())
    )
    summary["provenance"] = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("brainage-xai", "nibabel", "numpy", "scipy", "pandas")
        },
        "manifest_path": str(config.manifest_path),
        "manifest_sha256": hashlib.sha256(config.manifest_path.read_bytes()).hexdigest(),
        "target_spacing_mm": list(config.target_spacing_mm),
        "interpolation": "linear",
        "background_threshold": config.background_threshold,
        "limit": limit,
        "foreground_stage": "resampled intensities before clipping/zscore; same baseline mask",
        "percentile_method": "numpy linear",
        "center_crop_policy": "start=floor((full_shape-candidate_shape)/2); end exclusive",
        "candidate_policy": "per-axis sorted-size rank grown to meet empirical joint coverage",
        "normalization_performed": False,
        "volumes_saved": 0,
        "final_tensor_shape": None,
    }
    # Supplement the *existing* pilot rather than reselecting/reprocessing or overwriting its files.
    pilot_manifest = config.output_dir / "preprocessing/manifest.csv"
    if pilot_manifest.is_file():
        pilot = pd.read_csv(pilot_manifest, dtype={"subject_id": str, "acquisition_site": str})
        ids = set(pilot.subject_id)
        pilot_records = [r for r in records if r["subject_id"] in ids]
        pilot_summary = {
            "existing_pilot_subject_count": len(pilot),
            "diagnosed_pilot_subject_count": len(pilot_records),
            "site_counts": {
                str(k): int(v) for k, v in pilot.acquisition_site.value_counts().items()
            },
            "existing_pilot_outputs_modified": False,
            "geometry": geometry_summary(pilot_records),
            "subjects": pilot_records,
        }
        _write_json(output / "pilot_diagnostics.json", pilot_summary)
        summary["pilot_supplement"] = {
            k: v for k, v in pilot_summary.items() if k not in {"subjects", "geometry"}
        }
    frame = pd.DataFrame(records)
    for column in frame:
        frame[column] = frame[column].map(
            lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
        )
    frame.to_csv(output / "manifest.csv", index=False)
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantitative affine and foreground extent diagnostics"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Optional deterministic site-balanced subset")
    args = parser.parse_args()
    try:
        config = load_preprocessing_config(args.config)
        summary = run_diagnostics(config, limit=args.limit)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Diagnostics failed: {exc}\n")
    print(
        f"Eligible: {summary['eligible_subjects_seen']}; selected: {summary['subject_count']}; "
        f"successes: {summary['success_count']}; failures: {summary['failure_count']}. "
        f"Output directory: {config.output_dir / 'geometry_diagnostics'}; MRI volumes saved: 0"
    )
    if summary["failure_count"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
