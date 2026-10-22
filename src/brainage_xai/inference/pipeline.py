"""One new MRI, frozen scientific functions and ephemeral, identity-free input handling."""

import hashlib
import importlib.metadata
import io
import json
import math
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import yaml

from brainage_xai.data.validation import validate_nifti
from brainage_xai.modeling.data import create_inputs, imagenet_channels, slice_selection
from brainage_xai.modeling.models import grad_cam, load_checkpoint, seed_everything
from brainage_xai.preprocessing.intensity import normalize_intensity
from brainage_xai.preprocessing.orientation import canonicalize, inspect_orientation
from brainage_xai.registration import optimizers
from brainage_xai.registration.rigid import from_sitk, resample_to_fixed, to_sitk, transform_record
from brainage_xai.registration.settings import RigidParameters
from brainage_xai.registration.template import (
    TEMPLATE_AFFINE,
    load_template,
    sha256,
    validate_grid,
)

from . import APPLICATION_VERSION
from .config import CONTRACT_SHA256, ROOT, load_runtime_config, safe_destination
from .reports import AnalysisBundle, pdf_report, render_xai
from .result import (
    LIMITATIONS,
    XAI_WORDING,
    InferenceResult,
    age_warnings,
    bag_values,
    interpretation,
    validate_age,
)

_INFERENCE_LOCK = threading.RLock()


class InferenceFailure(ValueError):
    """Technical failure: no prediction/report should be displayed as a successful result."""


def validate_input(path, max_voxels):
    image = nib.load(path)
    if image.ndim != 3 or any(n <= 0 for n in image.shape):
        raise ValueError("A nonempty image with exactly three spatial dimensions is required")
    if math.prod(image.shape) > max_voxels:
        raise ValueError("Image exceeds configured technical memory/voxel limit")
    if image.get_data_dtype().kind not in "iuf":
        raise ValueError("Real scalar MRI intensities are required")
    orientation = inspect_orientation(image)
    validation = validate_nifti(Path(path))
    if not validation["valid"] or validation["intensity_std"] in (None, 0):
        raise ValueError("MRI must be readable, finite, nonempty and have nonconstant intensities")
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(data).all() or not np.any(data):
        raise ValueError("MRI has empty or nonfinite float32 data")
    return image, {k: v for k, v in validation.items() if k != "t1_path"}, orientation


def preprocess(image, config, progress):
    """Compose the unchanged Phase 2 production functions, with one final interpolation."""
    progress("Rigid registration to the MNI template")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    bundle = load_template(config.paths["template"], config.paths["template_mask"])
    fixed, mask = to_sitk(bundle.image), sitk.Cast(to_sitk(bundle.mask), sitk.sitkUInt8)
    moving = to_sitk(canonicalize(image))
    specification = yaml.safe_load((ROOT / "configs/registration.frozen.yaml").read_text())
    parameters = RigidParameters(**specification["registration_parameters"])
    transform, registration, _ = optimizers.register(fixed, moving, mask, parameters, "O0")
    transform_record(transform, parameters.rigid_tolerance)
    if not np.isfinite(
        [registration[k] for k in ("MI_initial", "MI_after_translation", "MI_final")]
    ).all():
        raise ValueError("Nonfinite registration metric")
    progress("Resampling once and applying frozen intensity normalization")
    registered = from_sitk(resample_to_fixed(moving, fixed, transform), bundle.image)
    data = registered.get_fdata(dtype=np.float32, caching="unchanged")
    normalized, statistics = normalize_intensity(data)
    foreground = data != 0
    if (
        not foreground.any()
        or not np.isfinite(normalized).all()
        or not np.all(normalized[~foreground] == 0)
        or abs(normalized[foreground].mean(dtype=np.float64)) > 1e-4
        or abs(normalized[foreground].std(dtype=np.float64) - 1) > 1e-4
    ):
        raise ValueError("Frozen normalization QC failed")
    header = registered.header.copy()
    header.set_slope_inter(1.0, 0.0)
    processed = nib.Nifti1Image(normalized, registered.affine, header)
    processed.set_qform(TEMPLATE_AFFINE, code=4)
    processed.set_sform(TEMPLATE_AFFINE, code=4)
    processed.header.set_xyzt_units("mm")
    validate_grid(processed)
    coverage = registration["evaluations"][-1]["template_brain_in_bounds_fraction"]
    if not np.isfinite(coverage) or coverage <= 0:
        raise ValueError("Template brain has no valid source FOV coverage")
    warnings = []
    for stage in ("translation", "rigid"):
        if registration[stage + "_stage"]["budget_hit"]:
            warnings.append(f"Registration {stage} iteration budget reached; review alignment.")
    if registration["delta_total"] < 0:
        warnings.append("Full-resolution registration MI declined; inspect alignment.")
    info = registration["final_transform"]
    if max(abs(v) for v in info["rotation_deg_lps_xyz"]) > parameters.rotation_warning_deg:
        warnings.append(
            "Frozen engineering QC flags a provisional large rotation; inspect alignment."
        )
    if np.linalg.norm(info["translation_mm_lps_xyz"]) > parameters.translation_warning_mm:
        warnings.append("Frozen engineering QC flags a large pull translation; inspect alignment.")
    if registration["evaluations"][-1]["template_brain_outside_count"]:
        warnings.append("Some template brain samples lie outside input FOV; inspect clipping.")
    selection = slice_selection(np.asanyarray(bundle.mask.dataobj) > 0, bundle.mask.affine)
    if any(selection[k] != config.slices[k] for k in selection):
        raise ValueError("Frozen public-mask slice selection mismatch")
    preprocessing_qc = {
        "status": "passed",
        "shape": list(processed.shape),
        "spacing_mm": [1.0, 1.0, 1.0],
        "orientation": "RAS",
        "affine_ras": processed.affine.tolist(),
        "normalization": statistics,
        "background_preserved": True,
        "final_interpolations": 1,
    }
    registration_qc = {
        "status": "passed_with_warnings" if warnings else "passed",
        "method": "S2/O0 rigid",
        "diagnostics": registration,
        "template_brain_inbounds_fraction": float(coverage),
        "scope": "Numerical/engineering QC; not anatomical correctness certification",
    }
    return normalized, preprocessing_qc, registration_qc, warnings


def choose_device(requested):
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS is unavailable; select CPU explicitly or use auto")
    return (
        ("mps" if torch.backends.mps.is_available() else "cpu")
        if requested == "auto"
        else requested
    )


def infer_upload(payload, name, age, config_path, progress=lambda message: None):
    return _analyze(io.BytesIO(payload), name, age, config_path, progress)


def infer_file(path, age, config_path, progress=lambda message: None):
    path = Path(path)
    with path.open("rb") as source:
        return _analyze(source, path.name, age, config_path, progress, path.resolve())


def _analyze(source, name, age, config_path, progress, original=None):
    age = validate_age(age)
    lower = name.lower()
    suffix = ".nii.gz" if lower.endswith(".nii.gz") else ".nii" if lower.endswith(".nii") else None
    if suffix is None:
        raise InferenceFailure("Supported MRI extensions are .nii and .nii.gz")
    with _INFERENCE_LOCK:
        config = load_runtime_config(config_path)
        safe_destination(config.temporary_dir, [original] if original else [])
        config.temporary_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        started = time.perf_counter()
        analysis_id = str(uuid.uuid4())
        stage = "input validation"
        try:
            with tempfile.TemporaryDirectory(prefix="brainage-", dir=config.temporary_dir) as temp:
                path = Path(temp) / ("upload" + suffix)
                copied = 0
                with path.open("xb") as destination:
                    while chunk := source.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > config.max_upload_bytes:
                            raise ValueError("Upload exceeds configured technical byte limit")
                        destination.write(chunk)
                progress("Validating the MRI and physical geometry")
                image, validation, orientation = validate_input(path, config.max_voxels)
                input_digest = sha256(path)
                stage = "preprocessing/registration"
                seed_everything(42)
                volume, prep_qc, reg_qc, registration_warnings = preprocess(image, config, progress)
                prep_qc["normalized_array_sha256"] = hashlib.sha256(
                    np.ascontiguousarray(volume).tobytes()
                ).hexdigest()
                prep_qc["input"] = validation
                device = choose_device(config.device)
                stage = "accepted model inference"
                progress("Predicting with the frozen multi-view model")
                gray = create_inputs(volume, config.slices)[1:]
                rgb = imagenet_channels(gray)[None].to(device)
                model, meta = load_checkpoint(config.paths["checkpoint"], device)
                if model.kind != "B" or meta["kind"] != "B":
                    raise ValueError("Only the accepted multi-view model B is permitted")
                with torch.inference_mode():
                    prediction = float(model(rgb).cpu()[0])
                if not np.isfinite(prediction):
                    raise ValueError("Nonfinite model prediction")
                values = bag_values(age, prediction, config, analysis_id)
                stage = "Grad-CAM/report generation"
                progress("Generating all 15 Grad-CAM maps and the individual report")
                maps, xai_prediction = grad_cam(model, rgb.detach().clone())
                if (
                    not np.isfinite(maps).all()
                    or maps.shape != (15, 224, 224)
                    or not np.isclose(xai_prediction[0], prediction, rtol=0, atol=0.02)
                ):
                    raise ValueError("Grad-CAM output/prediction consistency failure")
                files, mapping = render_xai(gray.numpy(), maps, config.slices["indices"])
                warnings = age_warnings(age, config.contract["age_ranges_years"])
                warnings += registration_warnings + LIMITATIONS
                if orientation["source_spatial_units"] == "unknown":
                    warnings.append(
                        "Input spatial units are unspecified; the frozen pipeline "
                        "assumes millimeters."
                    )
                if device == "cpu" and config.device == "auto":
                    warnings.append(
                        "MPS is unavailable; inference used CPU with the same checkpoint."
                    )
                zero_maps = int((maps.max((1, 2)) == 0).sum())
                if zero_maps:
                    warnings.append(
                        f"{zero_maps}/15 positive Grad-CAM maps are zero; "
                        "negative contributions are not shown."
                    )
                provenance = {
                    "input_sha256": input_digest,
                    "config_sha256": config.config_sha256,
                    "contract_sha256": CONTRACT_SHA256,
                    "checkpoint_sha256": sha256(config.paths["checkpoint"]),
                    "model": "B",
                    "checkpoint_epoch": int(meta["epoch"]),
                    "device": device,
                    "preprocessing_spec_sha256": config.contract["spec_sha256"][
                        "preprocessing.frozen.yaml"
                    ],
                    "registration_spec_sha256": config.contract["spec_sha256"][
                        "registration.frozen.yaml"
                    ],
                    "template_sha256": sha256(config.paths["template"]),
                    "template_mask_sha256": sha256(config.paths["template_mask"]),
                    "reference_sha256": sha256(config.paths["reference"]),
                    "correction_sha256": sha256(config.paths["correction"]),
                    "slices_sha256": sha256(config.paths["slices"]),
                    "correction_fit_split": "validation",
                    "correction_fit_count": 84,
                    "correction_intercept": config.correction["intercept"],
                    "correction_beta": config.correction["beta"],
                    "scientific_implementation_sha256": config.contract["source_sha256"],
                    "integration_implementation_sha256": {
                        p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
                    },
                    "software": {
                        k: importlib.metadata.version(k)
                        for k in (
                            "brainage-xai",
                            "torch",
                            "torchvision",
                            "numpy",
                            "nibabel",
                            "SimpleITK",
                            "reportlab",
                        )
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                    "upload_retention": "Temporary MRI removed before returning; no source "
                    "MRI is exported. Result/XAI derivatives remain sensitive.",
                }
                result = InferenceResult(
                    schema_version=1,
                    analysis_id=analysis_id,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    application_version=APPLICATION_VERSION,
                    chronological_age=age,
                    raw_predicted_brain_age=prediction,
                    **values,
                    preprocessing_qc=prep_qc,
                    registration_qc=reg_qc,
                    provenance=provenance,
                    observed_age_ranges_years=config.contract["age_ranges_years"],
                    warnings=list(dict.fromkeys(warnings)),
                    xai={
                        "method": "accepted Grad-CAM",
                        "interpretation": XAI_WORDING,
                        "slices": mapping,
                        "raw_arrays": "xai/gradcam.npy",
                        "checkpoint_sha256": provenance["checkpoint_sha256"],
                        "coarse_grid": [7, 7],
                        "zero_positive_maps": zero_maps,
                        "normalization": "Independent positive max normalization per slice",
                    },
                    interpretation=interpretation(values["corrected_BAG"]),
                )
                files["report.pdf"] = pdf_report(result, files)
                files["result.json"] = result.to_json().encode()
                files["provenance.json"] = json.dumps(
                    provenance, indent=2, allow_nan=False
                ).encode()
                bundle = AnalysisBundle(result, files)
            # MRI temp cleanup occurs before the bundle is returned, including on exceptions.
            progress("Analysis complete; temporary MRI removed")
            return bundle
        except Exception as exc:
            raise InferenceFailure(f"Technical failure during {stage}: {exc}") from exc


def export_bundle(bundle, directory, inputs=()):
    directory = safe_destination(directory, inputs)
    if directory.exists():
        raise FileExistsError("Output already exists; previous analyses are never overwritten")
    directory.mkdir(parents=True, mode=0o700)
    try:
        for name, data in bundle.files.items():
            destination = directory / name
            if not destination.resolve().is_relative_to(directory):
                raise ValueError("Unsafe output reference")
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with destination.open("xb") as stream:
                stream.write(data)
            destination.chmod(0o600)
        hashes = {n: hashlib.sha256(b).hexdigest() for n, b in bundle.files.items()}
        (directory / "export_checksums.json").write_text(json.dumps(hashes, indent=2))
    except Exception:
        shutil.rmtree(directory)
        raise
    return directory
