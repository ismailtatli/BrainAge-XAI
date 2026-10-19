"""Header-only spatial audits and sequential single-subject preprocessing with provenance."""

import argparse
import importlib.metadata
import json
import platform
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
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

from .geometry import affine_geometry, affine_summary
from .intensity import normalize_intensity
from .orientation import canonicalize, inspect_orientation
from .qc import save_comparison
from .resampling import output_grid, resample_t1


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _record(subject: dict, config: PreprocessingConfig) -> dict[str, Any]:
    return {
        "subject_id": subject["subject_id"],
        "source_t1_path": subject["t1_path"],
        "acquisition_site": subject["acquisition_site"],
        "source_shape": None,
        "source_spacing": None,
        "source_orientation": None,
        "canonical_shape": None,
        "canonical_orientation": None,
        "resampled_shape": None,
        "target_spacing_mm": list(config.target_spacing_mm),
        "normalization_method": "foreground_percentile_clip_zscore",
        "lower_percentile": config.lower_percentile,
        "upper_percentile": config.upper_percentile,
        "background_threshold": config.background_threshold,
        "interpolation": "linear",
        "output_path": None,
        "qc_path": None,
        "preprocessing_status": "failed",
        "warnings": [],
        "errors": [],
    }


def inspect_subject(subject: dict, config: PreprocessingConfig) -> dict[str, Any]:
    """Predict canonical/resampled geometry without reading intensity arrays."""
    record = _record(subject, config)
    record["normalization_method"] = "not_run_spatial_audit"
    try:
        image = nib.load(subject["t1_path"], mmap=True)
        record.update(inspect_orientation(image))
        record.update(affine_geometry(image.affine))
        if record["source_spatial_units"] == "unknown":
            record["warnings"].append("unknown_spatial_units_assumed_mm")
        shape, affine = output_grid(
            record["canonical_shape"],
            np.asarray(record["canonical_affine"]),
            config.target_spacing_mm,
        )
        record.update(
            resampled_shape=list(shape),
            resampled_affine=affine.tolist(),
            preprocessing_status="success",
        )
        if record["is_oblique"]:
            record["warnings"].append("oblique_source_resampled_to_axis_aligned_grid")
        if record["has_measurable_shear"]:
            record["warnings"].append("non_orthogonal_source_affine")
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    return record


def preprocess_subject(
    subject: dict, config: PreprocessingConfig, *, overwrite: bool = False
) -> dict[str, Any]:
    """Process one source MRI; failures remain explicit and never escape into other subjects."""
    record = inspect_subject(subject, config)
    record["normalization_method"] = "foreground_percentile_clip_zscore"
    if record["preprocessing_status"] != "success":
        return record
    record["preprocessing_status"] = "failed"
    destination = config.output_dir / "volumes" / f"{subject['subject_id']}.nii.gz"
    temporary = None
    try:
        source = Path(subject["t1_path"])
        validate_output_directory(destination.parent, [source, config.manifest_path])
        if destination.resolve() == source.resolve():
            raise ValueError("Output must never overwrite source MRI")
        if destination.exists() and not overwrite:
            raise FileExistsError(
                "Processed MRI already exists; use --overwrite for derived outputs"
            )
        canonical = canonicalize(nib.load(source, mmap=True))
        resampled = resample_t1(canonical, config.target_spacing_mm)
        normalized, statistics = normalize_intensity(
            np.asarray(resampled.dataobj),
            lower_percentile=config.lower_percentile,
            upper_percentile=config.upper_percentile,
            background_threshold=config.background_threshold,
        )
        record.update(statistics)
        header = resampled.header.copy()
        header.set_data_dtype(np.float32)
        header.set_slope_inter(1.0, 0.0)
        processed = nib.Nifti1Image(normalized, resampled.affine, header)
        # These are derived physical coordinates, not template-aligned anatomy.
        processed.set_sform(resampled.affine, code=2)
        processed.set_qform(resampled.affine, code=2)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".nii.gz", delete=False
        ) as f:
            temporary = Path(f.name)
        nib.save(processed, temporary)
        temporary.replace(destination)
        temporary = None
        record["output_path"] = str(destination)
        if config.qc_enabled:
            qc_path = config.output_dir / "qc" / f"{subject['subject_id']}.png"
            try:
                save_comparison(canonical, processed, qc_path, dpi=config.qc_dpi)
                record["qc_path"] = str(qc_path)
            except Exception as exc:
                record["warnings"].append(f"qc_failed: {type(exc).__name__}: {exc}")
        record["preprocessing_status"] = "success"
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record


def spatial_summary(records: list[dict]) -> dict[str, Any]:
    """Aggregate shapes/orientations/spacings without subject IDs or source paths."""
    successful = [r for r in records if r["preprocessing_status"] == "success"]
    summary: dict[str, Any] = {
        "subject_count": len(records),
        "success_count": len(successful),
        "failure_count": len(records) - len(successful),
    }
    for field in (
        "source_shape",
        "canonical_shape",
        "resampled_shape",
        "source_orientation",
        "canonical_orientation",
        "source_spacing",
    ):
        counts = Counter("x".join(str(v) for v in r[field]) for r in successful)
        summary[f"{field}_distribution"] = dict(sorted(counts.items()))
    for field in ("source_spacing", "canonical_shape", "resampled_shape"):
        values = np.asarray([r[field] for r in successful])
        summary[f"{field}_range"] = (
            {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}
            if len(values)
            else None
        )
    summary["warning_counts"] = dict(
        sorted(Counter(warning for r in records for warning in r["warnings"]).items())
    )
    summary["qc_success_count"] = sum(r["qc_path"] is not None for r in records)
    summary["affine_geometry"] = affine_summary(records)
    return summary


def run_preprocessing(
    config: PreprocessingConfig,
    *,
    audit_only: bool = False,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Use existing eligible labels, optionally selecting a deterministic pilot across sites."""
    subjects = eligible_subjects(config.manifest_path)
    validate_output_directory(
        config.output_dir, [config.manifest_path, *(Path(p) for p in subjects.t1_path)]
    )
    selected = select_pilot(subjects, limit)
    report_dir = config.output_dir / ("spatial_audit" if audit_only else "preprocessing")
    report_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir = report_dir / "subjects"
    provenance_dir.mkdir(exist_ok=True)
    records = []
    for index, subject in enumerate(selected.to_dict("records"), start=1):
        print(
            f"{'Inspecting' if audit_only else 'Preprocessing'} MRI {index}/{len(selected)}: "
            f"{subject['subject_id']}",
            file=sys.stderr,
            flush=True,
        )
        record = (
            inspect_subject(subject, config)
            if audit_only
            else preprocess_subject(subject, config, overwrite=overwrite)
        )
        _json(provenance_dir / f"{subject['subject_id']}.json", record)
        records.append(record)
    summary = spatial_summary(records)
    summary.update(
        eligible_subjects_seen=len(subjects),
        selected_subject_count=len(selected),
        audit_only=audit_only,
        target_spacing_mm=list(config.target_spacing_mm),
        fixed_tensor_shape=None,
    )
    summary["provenance"] = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("brainage-xai", "numpy", "nibabel", "scipy", "pandas", "matplotlib")
        },
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "pilot_selection": "round_robin_sorted_sites_then_sorted_subject_ids",
        "limit": limit,
    }
    _json(report_dir / "summary.json", summary)
    # Lists/dicts in CSV are JSON, not Python repr, for unambiguous downstream parsing.
    frame = pd.DataFrame(records)
    for column in frame:
        frame[column] = frame[column].map(
            lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
        )
    frame.to_csv(report_dir / "manifest.csv", index=False)
    return summary


def _main(audit_only: bool) -> None:
    parser = argparse.ArgumentParser(
        description="Header-only spatial audit"
        if audit_only
        else "Per-subject research MRI preprocessing"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, help="Positive deterministic pilot size, balanced by site"
    )
    if not audit_only:
        parser.add_argument("--overwrite", action="store_true", help="Replace derived outputs only")
    args = parser.parse_args()
    try:
        config = load_preprocessing_config(args.config)
        summary = run_preprocessing(
            config,
            audit_only=audit_only,
            limit=args.limit,
            overwrite=getattr(args, "overwrite", False),
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Preprocessing failed: {exc}\n")
    print(
        f"Eligible: {summary['eligible_subjects_seen']}; selected: {summary['subject_count']}; "
        f"successes: {summary['success_count']}; failures: {summary['failure_count']}. "
        f"Output directory: {config.output_dir}"
    )
    if summary["failure_count"]:
        parser.exit(1)


def main() -> None:
    _main(False)


def audit_main() -> None:
    _main(True)
