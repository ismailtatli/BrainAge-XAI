"""Private eighteen-subject rigid pilot, baseline comparison, provenance, and explicit warnings."""

import argparse
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
import SimpleITK as sitk

from brainage_xai.config.preprocessing import validate_output_directory
from brainage_xai.data.preparation import eligible_subjects
from brainage_xai.preprocessing.geometry import describe, foreground_extent
from brainage_xai.preprocessing.intensity import foreground_mask
from brainage_xai.preprocessing.orientation import canonicalize, inspect_orientation
from brainage_xai.preprocessing.resampling import resample_t1

from .qc import registration_qc
from .rigid import from_sitk, register_rigid, resample_to_fixed, source_foreground_clipping, to_sitk
from .selection import select_registration_pilot
from .settings import RegistrationConfig, RigidParameters, load_registration_config
from .template import TemplateBundle, load_template, sha256


def intensity_statistics(data: np.ndarray, threshold: float) -> dict[str, Any]:
    mask = foreground_mask(data, background_threshold=threshold)
    values = data[mask].astype(np.float64)
    return {
        "foreground_mean": float(values.mean()),
        "foreground_std": float(values.std()),
        "foreground_percentiles": dict(
            zip(
                ("p0_5", "p1", "p50", "p99", "p99_5"),
                np.percentile(values, [0.5, 1, 50, 99, 99.5]).tolist(),
            )
        ),
        "zero_fraction": float(np.count_nonzero(data == 0) / data.size),
        "foreground_fraction": float(mask.mean()),
    }


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _euler(info: dict) -> sitk.Euler3DTransform:
    tx = sitk.Euler3DTransform()
    tx.SetFixedParameters(info["fixed_parameters"])
    tx.SetParameters(info["parameters"])
    return tx


def qc_warning_flags(
    trace: dict, clipping_fraction: float, parameters: RigidParameters
) -> list[str]:
    """Provisional engineering flags; never anatomical pass/fail or deletion rules."""
    flags = []
    if trace["metric_improvement"] < 0:
        flags.append("metric_degradation_on_common_evaluation_lattice")
    if (
        max(abs(v) for v in trace["final_transform"]["rotation_deg_lps_xyz"])
        > parameters.rotation_warning_deg
    ):
        flags.append("provisional_large_rotation")
    if (
        np.linalg.norm(trace["final_transform"]["translation_mm_lps_xyz"])
        > parameters.translation_warning_mm
    ):
        flags.append("provisional_large_translation")
    if clipping_fraction > parameters.clipping_warning_fraction:
        flags.append("provisional_source_nonbackground_fov_clipping_not_brain_loss")
    if "maxim" in trace["optimizer_stop_condition"].lower():
        flags.append("optimizer_iteration_budget_exhausted")
    return flags


def register_subject(
    subject: dict,
    config: RegistrationConfig,
    bundle: TemplateBundle,
    fixed: sitk.Image,
    fixed_mask: sitk.Image | None,
) -> dict[str, Any]:
    record = {
        "subject_id": subject["subject_id"],
        "source_t1_path": subject["t1_path"],
        "acquisition_site": subject["acquisition_site"],
        "selection_stratum": subject["selection_stratum"],
        "selection_reason": subject["selection_reason"],
        "registration_success": False,
        "registration_status": "failed",
        "anatomical_qc_status": "not_reviewed",
        "warnings": [],
        "errors": [],
        "output_path": None,
        "transform_path": None,
        "qc_path": None,
        "registration_parameters": asdict(config.parameters),
        "template": bundle.provenance,
        "final_interpolation_steps_from_native": 1,
        "baseline_comparison_interpolation_steps": 1,
        "initialized_qc_interpolation_steps_from_native": 1,
        "interpolation_chain": "native -> template once; separate baseline and QC branches",
        "source_sha256": None,
    }
    temporary = None
    try:
        validate_output_directory(
            config.output_dir,
            [Path(subject["t1_path"]), config.manifest_path, config.template_path],
        )
        record["source_sha256"] = sha256(Path(subject["t1_path"]))
        source = nib.load(subject["t1_path"])
        record.update(inspect_orientation(source))
        canonical = canonicalize(source)
        # A field used for reference frames, not a promise that source anatomy is aligned.
        native_data = canonical.get_fdata(dtype=np.float32, caching="unchanged")
        record["intensity_native"] = intensity_statistics(native_data, config.background_threshold)
        moving = to_sitk(canonical)
        trace = {}
        try:
            final, trace = register_rigid(
                fixed, moving, config.parameters, fixed_mask, diagnostics=trace
            )
        finally:
            record.update(trace)
        record["initialization_clipping"] = source_foreground_clipping(
            native_data,
            canonical.affine,
            bundle.image.shape,
            bundle.image.affine,
            _euler(trace["initial_transform"]),
            config.background_threshold,
        )
        record["final_clipping"] = source_foreground_clipping(
            native_data,
            canonical.affine,
            bundle.image.shape,
            bundle.image.affine,
            final,
            config.background_threshold,
        )
        registered_sitk = resample_to_fixed(moving, fixed, final)
        if (
            registered_sitk.GetSize() != fixed.GetSize()
            or not np.allclose(registered_sitk.GetSpacing(), fixed.GetSpacing(), rtol=0, atol=1e-6)
            or not np.allclose(registered_sitk.GetOrigin(), fixed.GetOrigin(), rtol=0, atol=1e-6)
            or not np.allclose(
                registered_sitk.GetDirection(), fixed.GetDirection(), rtol=0, atol=1e-6
            )
        ):
            raise ValueError("Unexpected output geometry: template grid not inherited")
        registered = from_sitk(registered_sitk, bundle.image)
        data = np.asarray(registered.dataobj)
        if not np.isfinite(data).all():
            raise ValueError("Non-finite registered output")
        record["output_shape"] = list(registered.shape)
        record["output_spacing_mm"] = nib.affines.voxel_sizes(registered.affine).tolist()
        record["output_orientation"] = list(nib.aff2axcodes(registered.affine))
        record["output_affine_ras"] = registered.affine.tolist()
        record["output_geometry_verified"] = True
        record["foreground_after"] = foreground_extent(
            data, record["output_spacing_mm"], background_threshold=config.background_threshold
        )
        record["intensity_registered"] = intensity_statistics(data, config.background_threshold)
        baseline = resample_t1(canonical, (1.0, 1.0, 1.0))
        record["foreground_baseline"] = foreground_extent(
            np.asarray(baseline.dataobj),
            (1.0, 1.0, 1.0),
            background_threshold=config.background_threshold,
        )
        record["intensity_baseline"] = intensity_statistics(
            np.asarray(baseline.dataobj), config.background_threshold
        )
        initialized = from_sitk(
            resample_to_fixed(moving, fixed, _euler(trace["initial_transform"])), bundle.image
        )
        record["intensity_initialized_template"] = intensity_statistics(
            np.asarray(initialized.dataobj), config.background_threshold
        )
        warnings = record["warnings"]
        warnings.extend(
            qc_warning_flags(
                trace, record["final_clipping"]["outside_template_fov_fraction"], config.parameters
            )
        )
        for directory in ("volumes", "transforms", "qc"):
            (config.output_dir / directory).mkdir(exist_ok=True)
        output = config.output_dir / "volumes" / f"{subject['subject_id']}.nii.gz"
        transform = config.output_dir / "transforms" / f"{subject['subject_id']}.tfm"
        if output.exists() or transform.exists():
            raise FileExistsError(
                "Pilot output exists; choose a new output_dir; never overwrite source or pilot"
            )
        if output.resolve() == Path(subject["t1_path"]).resolve():
            raise ValueError("Cannot overwrite source MRI")
        with tempfile.NamedTemporaryFile(
            dir=output.parent, suffix=".nii.gz", delete=False
        ) as stream:
            temporary = Path(stream.name)
        nib.save(registered, temporary)
        temporary.replace(output)
        temporary = None
        sitk.WriteTransform(final, str(transform))
        record["output_path"] = str(output)
        record["transform_path"] = str(transform)
        qc = config.output_dir / "qc" / f"{subject['subject_id']}.png"
        try:
            registration_qc(
                bundle.image,
                baseline,
                initialized,
                registered,
                qc,
                dpi=config.qc_dpi,
                checkerboard_pixels=config.checkerboard_pixels,
                clipping_fraction=record["final_clipping"]["outside_template_fov_fraction"],
            )
            record["qc_path"] = str(qc)
        except Exception as exc:
            warnings.append(f"qc_generation_failed: {type(exc).__name__}: {exc}")
        record["registration_success"] = True
        record["registration_status"] = "completed_qc_review_required"
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record


def _crop_coverage(extents: list[dict], shape: list[int]) -> dict[str, Any]:
    """Counterfactual centered crop/pad at a stated size; never modifies image data."""
    flags = []
    for extent in extents:
        start = (np.array(extent["resampled_shape"]) - shape) // 2
        end = start + shape
        flags.append(
            (np.array(extent["foreground_bbox_start"]) < start)
            | (np.array(extent["foreground_bbox_end_exclusive"]) > end)
        )
    array = np.asarray(flags, dtype=bool).reshape(-1, 3)
    return {
        "subject_count": len(extents),
        "foreground_loss_axis_counts": array.sum(axis=0).tolist(),
        "foreground_loss_any_count": int(array.any(axis=1).sum()),
    }


def pilot_summary(records: list[dict]) -> dict[str, Any]:
    completed = [r for r in records if r["registration_success"]]
    summary = {
        "subject_count": len(records),
        "success_count": len(completed),
        "failure_count": len(records) - len(completed),
        "anatomical_qc_review_required_count": len(completed),
        "site_counts": dict(sorted(Counter(r["acquisition_site"] for r in records).items())),
        "warning_counts": dict(sorted(Counter(w for r in records for w in r["warnings"]).items())),
        "optimizer_stop_counts": dict(
            sorted(
                Counter(r.get("optimizer_stop_condition", "not_completed") for r in records).items()
            )
        ),
        "output_shape_distribution": dict(
            Counter("x".join(map(str, r["output_shape"])) for r in completed)
        ),
        "output_geometry_verified_count": sum(
            r.get("output_geometry_verified", False) for r in records
        ),
        "qc_output_count": sum(r["qc_path"] is not None for r in records),
        "final_tensor_shape": None,
        "final_interpolation_steps_from_native": 1,
    }
    stop_categories = Counter()
    for record in records:
        text = record.get("optimizer_stop_condition", "").lower()
        category = (
            "iteration_budget"
            if "maximum" in text
            else "step_too_small"
            if "step too small" in text
            else "gradient_tolerance"
            if "gradient magnitude tolerance" in text
            else "other_or_failed"
        )
        stop_categories[category] += 1
    summary["optimizer_stop_category_counts"] = dict(sorted(stop_categories.items()))
    if completed:
        shape = completed[0]["output_shape"]
        summary["template_size_centered_crop_diagnostic"] = {
            "target_shape": shape,
            "not_a_selected_tensor_shape": True,
            "baseline": _crop_coverage([r["foreground_baseline"] for r in completed], shape),
            "registered": _crop_coverage([r["foreground_after"] for r in completed], shape),
            "caution": "Template FOV limits output; inspect source clipping separately",
        }
    for field in ("metric_before", "metric_after", "metric_improvement", "iteration_count"):
        summary[f"{field}_summary"] = describe(np.array([r[field] for r in completed]))
    for field in ("rotation_deg_lps_xyz", "translation_mm_lps_xyz"):
        summary[f"{field}_summary"] = describe(
            np.array([r["final_transform"][field] for r in completed])
        )
    for stage in ("foreground_baseline", "foreground_after"):
        for field in ("foreground_center_offset_mm", "foreground_bbox_size", "foreground_fraction"):
            summary[f"{stage}_{field}_summary"] = describe(
                np.array([r[stage][field] for r in completed])
            )
    summary["native_nonbackground_outside_template_fraction_summary"] = describe(
        np.array([r["final_clipping"]["outside_template_fov_fraction"] for r in completed])
    )
    for stage in (
        "intensity_native",
        "intensity_baseline",
        "intensity_initialized_template",
        "intensity_registered",
    ):
        summary[stage] = {
            field: describe(np.array([r[stage][field] for r in completed]))
            for field in (
                "foreground_mean",
                "foreground_std",
                "zero_fraction",
                "foreground_fraction",
            )
        }
        summary[stage]["foreground_percentiles"] = {
            key: describe(np.array([r[stage]["foreground_percentiles"][key] for r in completed]))
            for key in ("p0_5", "p1", "p50", "p99", "p99_5")
        }
    summary["interpolation_audit_registered_minus_baseline"] = {
        field: describe(
            np.array(
                [
                    r["intensity_registered"][field] - r["intensity_baseline"][field]
                    for r in completed
                ]
            )
        )
        for field in ("foreground_mean", "foreground_std", "zero_fraction", "foreground_fraction")
    }
    return summary


def run_registration(config: RegistrationConfig) -> dict[str, Any]:
    config.parameters.validate()
    subjects = eligible_subjects(config.manifest_path)
    geometry = pd.read_csv(config.geometry_manifest_path, dtype={"subject_id": str})
    pilot = select_registration_pilot(subjects, geometry)
    validate_output_directory(
        config.output_dir,
        [
            config.manifest_path,
            config.geometry_manifest_path,
            config.template_path,
            *([config.brain_mask_path] if config.brain_mask_path else []),
            *(Path(p) for p in subjects.t1_path),
        ],
    )
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError("Registration output_dir must be empty; use a new run directory")
    bundle = load_template(
        config.template_path, config.brain_mask_path, use_mask=config.parameters.use_fixed_mask
    )
    fixed = to_sitk(bundle.image)
    mask = sitk.Cast(to_sitk(bundle.mask), sitk.sitkUInt8) if bundle.mask is not None else None
    config.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir = config.output_dir / "subjects"
    provenance_dir.mkdir(exist_ok=True)
    pilot[
        [
            "subject_id",
            "age",
            "acquisition_site",
            "t1_path",
            "max_obliquity_deg",
            "foreground_center_offset_mm",
            "center_offset_norm_mm",
            "selection_stratum",
            "selection_reason",
        ]
    ].to_csv(config.output_dir / "pilot_selection.csv", index=False)
    records = []
    for index, subject in enumerate(pilot.to_dict("records"), start=1):
        print(
            f"Rigid pilot {index}/{len(pilot)} "
            f"({subject['acquisition_site']}, {subject['selection_stratum']})",
            file=sys.stderr,
            flush=True,
        )
        record = register_subject(subject, config, bundle, fixed, mask)
        _json(provenance_dir / f"{subject['subject_id']}.json", record)
        records.append(record)
    frame = pd.DataFrame(records)
    for column in frame:
        frame[column] = frame[column].map(
            lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
        )
    frame.to_csv(config.output_dir / "manifest.csv", index=False)
    summary = pilot_summary(records)
    summary["template"] = bundle.provenance
    summary["provenance"] = {
        "simpleitk_version": sitk.Version_VersionString(),
        "python": platform.python_version(),
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()
        },
        "source_manifest_sha256": sha256(config.manifest_path),
        "geometry_manifest_sha256": sha256(config.geometry_manifest_path),
        "selection": "site angle thirds: offset-tail + angle representative per third",
    }
    _json(config.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental 18-subject six-DOF rigid registration pilot"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_registration_config(args.config)
        summary = run_registration(config)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Registration pilot failed: {exc}\n")
    print(
        f"Rigid pilot: {summary['subject_count']} subjects; {summary['success_count']} completed; "
        f"{summary['failure_count']} failures; completed cases require anatomical QC. "
        f"Output: {config.output_dir}"
    )
    if summary["failure_count"]:
        parser.exit(1)
