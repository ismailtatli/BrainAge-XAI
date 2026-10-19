"""Frozen full-template-grid preprocessing with subject-level splits and validated resume."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import sys
import tempfile
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

from brainage_xai.config.preprocessing import validate_output_directory
from brainage_xai.data.preparation import eligible_subjects
from brainage_xai.data.validation import validate_nifti
from brainage_xai.registration import optimizers
from brainage_xai.registration.optimizer_study import core_hashes
from brainage_xai.registration.pipeline import _euler
from brainage_xai.registration.rigid import from_sitk, resample_to_fixed, to_sitk, transform_record
from brainage_xai.registration.settings import RigidParameters
from brainage_xai.registration.template import (
    MASK_SHA256,
    T1_SHA256,
    TEMPLATE_AFFINE,
    TEMPLATE_SHAPE,
    load_template,
    sha256,
    validate_grid,
)

from .intensity import normalize_intensity
from .orientation import canonicalize

_CONTEXT = None


def atomic_json(path, value):
    """Replace a derived checkpoint only after a complete finite JSON serialization."""
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_membership(cohort, membership, expected_counts):
    """Consume existing labels only; no split generation or eligibility fitting."""
    if cohort.subject_id.duplicated().any() or membership.subject_id.duplicated().any():
        raise ValueError("Duplicate subject in eligibility/split manifest")
    if set(cohort.subject_id) != set(membership.subject_id):
        raise ValueError("Missing or extra split members")
    if membership.split.value_counts().to_dict() != expected_counts:
        raise ValueError("Frozen split counts differ")
    joined = cohort.merge(
        membership, on="subject_id", suffixes=("", "_split"), validate="one_to_one"
    )
    for name in ("t1_path", "acquisition_site"):
        if not joined[name].eq(joined[name + "_split"]).all():
            raise ValueError("Frozen split source/site mismatch")
    if not np.allclose(joined.age, joined.age_split, rtol=0, atol=1e-10):
        raise ValueError("Frozen split age mismatch")
    return joined[[*cohort.columns, "split"]].sort_values("subject_id").reset_index(drop=True)


def load_spec(path):
    """Validate frozen scientific parameters, implementations, resources and software."""
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[3]
    spec = yaml.safe_load(path.read_text())
    if spec["geometry"] != {
        "shape_xyz": list(TEMPLATE_SHAPE),
        "spacing_mm": [1, 1, 1],
        "orientation": "RAS",
        "affine_ras": TEMPLATE_AFFINE.tolist(),
        "crop": False,
        "pad": False,
        "model_slices_selected": False,
    }:
        raise ValueError("Frozen full-template geometry changed")
    files = {
        "cohort": (spec["cohort"]["manifest"], spec["cohort"]["manifest_sha256"]),
        "membership": (spec["split"]["membership"], spec["split"]["membership_sha256"]),
        "split_summary": (spec["split"]["summary"], spec["split"]["summary_sha256"]),
        "registration": (
            spec["registration"]["specification"],
            spec["registration"]["specification_sha256"],
        ),
        "template": (spec["template"]["t1"], T1_SHA256),
        "mask": (spec["template"]["qc_brain_mask"], MASK_SHA256),
    }
    paths = {key: root / value[0] for key, value in files.items()}
    if any(sha256(paths[key]) != value[1] for key, value in files.items()):
        raise ValueError("Frozen cohort/split/registration/template checksum differs")
    rigid_spec = yaml.safe_load(paths["registration"].read_text())
    expected = RigidParameters(
        iterations_per_level=300, evaluation_shrink_factor=1, use_fixed_mask=False
    )
    if (
        rigid_spec["method"] != "RIGID"
        or rigid_spec["registration_parameters"] != json.loads(json.dumps(asdict(expected)))
        or rigid_spec["optimizer_configuration"] != optimizers.optimizer_configuration("O0")
        or rigid_spec["implementation"]["core_sha256"] != core_hashes()
        or spec["registration"]["affine"]
        or spec["registration"]["nonlinear"]
    ):
        raise ValueError("Frozen S2/O0 registration changed")
    intensity = spec["intensity"]
    if (
        intensity["method"] != "foreground_percentile_clip_zscore"
        or [intensity[k] for k in ("lower_percentile", "upper_percentile", "background_threshold")]
        != [0.5, 99.5, 0.0]
        or intensity["implementation_sha256"] != sha256(Path(__file__).with_name("intensity.py"))
    ):
        raise ValueError("Validated per-subject normalization changed")
    if spec["software"]["python"] != platform.python_version() or any(
        importlib.metadata.version(name) != version
        for name, version in spec["software"]["packages"].items()
    ):
        raise ValueError("Frozen software versions differ")
    cohort = eligible_subjects(paths["cohort"])
    if len(cohort) != spec["cohort"]["expected_count"] or len(cohort) != 561:
        raise ValueError("Require exactly 561 existing eligible subjects")
    membership = pd.read_csv(paths["membership"])
    cohort = validate_membership(cohort, membership, spec["split"]["expected_counts"])
    return spec, paths, cohort, expected


def validate_finished(subject, output, source_hash, fingerprint):
    """Read and verify the complete saved volume and provenance before permitting reuse."""
    sid = subject["subject_id"]
    path = output / "volumes" / f"{sid}.nii.gz"
    record = json.loads((output / "provenance" / f"{sid}.json").read_text())
    if (
        record["status"] != "success"
        or record["subject_id"] != sid
        or record["split"] != subject["split"]
        or record["site"] != subject["acquisition_site"]
        or record["source_sha256"] != source_hash
        or record["run_fingerprint"] != fingerprint
        or sha256(path) != record["output_sha256"]
    ):
        raise ValueError("Saved subject provenance/hash/configuration mismatch")
    image = nib.load(path)
    validate_grid(image)
    for getter in (image.get_qform, image.get_sform):
        matrix, code = getter(coded=True)
        if code != 4 or not np.allclose(matrix, TEMPLATE_AFFINE, rtol=0, atol=1e-5):
            raise ValueError("Output requires coded MNI template qform and sform")
    if image.get_data_dtype() != np.dtype("float32"):
        raise ValueError("Output must be float32")
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    values = data[data != 0]
    if not len(values) or not np.isfinite(data).all():
        raise ValueError("Empty/nonfinite output")
    if (
        abs(float(values.mean(dtype=np.float64))) > 1e-4
        or abs(float(values.std(dtype=np.float64)) - 1) > 1e-4
    ):
        raise ValueError("Saved normalized foreground is not standardized")
    if np.count_nonzero(data == 0) < record["normalization"]["background_voxel_count"]:
        raise ValueError("Saved background count differs")
    for key, value in record["normalization"].items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError("Nonfinite normalization provenance: " + key)
    tx = transform_record(_euler(record["registration"]["final_transform"]))
    if not np.allclose(
        tx["matrix_lps"],
        record["registration"]["final_transform"]["matrix_lps"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Inconsistent rigid transform provenance")
    return record


def initialize_worker(template_path, mask_path, parameters):
    global _CONTEXT
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    bundle = load_template(template_path, mask_path)
    _CONTEXT = (
        bundle,
        to_sitk(bundle.image),
        sitk.Cast(to_sitk(bundle.mask), sitk.sitkUInt8),
        parameters,
    )


def process_subject(subject, output, source_hash, fingerprint):
    """One native subject at a time per worker; no modifications to registration/normalization."""
    start = time.perf_counter()
    sid = subject["subject_id"]
    destination = output / "volumes" / f"{sid}.nii.gz"
    if destination.exists() or (output / "provenance" / f"{sid}.json").exists():
        raise FileExistsError("Existing outputs must be validated/handled by resume first")
    record = {
        "subject_id": sid,
        "site": subject["acquisition_site"],
        "split": subject["split"],
        "source_t1_path": subject["t1_path"],
        "source_sha256": source_hash,
        "run_fingerprint": fingerprint,
        "status": "failed",
        "stage": "source_validation",
        "warnings": [],
        "exception": None,
        "qc_state": "not_validated",
    }
    temporary = None
    try:
        if sha256(Path(subject["t1_path"])) != source_hash:
            raise ValueError("Source hash changed")
        record["source_validation"] = validate_nifti(Path(subject["t1_path"]))
        if not record["source_validation"]["valid"]:
            raise ValueError(str(record["source_validation"]["errors"]))
        record["warnings"].extend(record["source_validation"]["warnings"])
        record["stage"] = "canonicalization"
        canonical = canonicalize(nib.load(subject["t1_path"]))
        moving = to_sitk(canonical)
        bundle, fixed, mask, parameters = _CONTEXT
        record["stage"] = "rigid_registration"
        final, registration, rows = optimizers.register(fixed, moving, mask, parameters, "O0")
        record["registration"] = registration
        record["stage"] = "resampling"
        registered = from_sitk(resample_to_fixed(moving, fixed, final), bundle.image)
        data = registered.get_fdata(dtype=np.float32, caching="unchanged")
        record["stage"] = "normalization"
        normalized, statistics = normalize_intensity(data)
        foreground = data != 0
        if not np.all(normalized[~foreground] == 0):
            raise ValueError("Normalization modified background")
        statistics.update(
            normalized_foreground_mean=float(normalized[foreground].mean(dtype=np.float64)),
            normalized_foreground_std=float(normalized[foreground].std(dtype=np.float64)),
            background_voxel_count=int((~foreground).sum()),
            background_preserved=True,
        )
        record["normalization"] = statistics
        header = registered.header.copy()
        header.set_slope_inter(1.0, 0.0)
        processed = nib.Nifti1Image(normalized, registered.affine, header)
        processed.set_qform(TEMPLATE_AFFINE, code=4)
        processed.set_sform(TEMPLATE_AFFINE, code=4)
        processed.header.set_xyzt_units("mm")
        processed.header["cal_min"] = float(normalized.min())
        processed.header["cal_max"] = float(normalized.max())
        record["stage"] = "save"
        with tempfile.NamedTemporaryFile(
            dir=output / "work", suffix=".nii.gz", delete=False
        ) as stream:
            temporary = Path(stream.name)
        nib.save(processed, temporary)
        temporary.replace(destination)
        temporary = None
        record.update(
            output_path=str(destination),
            output_sha256=sha256(destination),
            output_shape=list(processed.shape),
            output_spacing_mm=[1.0] * 3,
            output_orientation="RAS",
            output_affine_ras=processed.affine.tolist(),
        )
        for name in ("translation", "rigid"):
            if registration[name + "_stage"]["budget_hit"]:
                record["warnings"].append(name + "_iteration_budget")
        if registration["delta_total"] < 0:
            record["warnings"].append("deterministic_MI_degradation")
        info = registration["final_transform"]
        if max(abs(v) for v in info["rotation_deg_lps_xyz"]) > parameters.rotation_warning_deg:
            record["warnings"].append("provisional_large_rotation")
        if np.linalg.norm(info["translation_mm_lps_xyz"]) > parameters.translation_warning_mm:
            record["warnings"].append("provisional_large_pull_translation")
        if registration["evaluations"][-1]["template_brain_outside_count"]:
            record["warnings"].append("template_ROI_has_outside_FOV_samples_not_anatomical_error")
        record.update(status="success", stage="output_verification", qc_state="numerical_pass")
        atomic_json(output / "provenance" / f"{sid}.json", record)
        validate_finished(subject, output, source_hash, fingerprint)
        record["stage"] = "complete"
    except Exception as exc:
        record.update(
            status="failed",
            exception=f"{type(exc).__name__}: {exc}",
            qc_state="failed_at_" + record["stage"],
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    record["runtime_seconds"] = time.perf_counter() - start
    atomic_json(output / "provenance" / f"{sid}.json", record)
    return record


def summary_records(records, expected):
    good = [r for r in records if r["status"] == "success"]
    result = {
        "expected_count": expected,
        "processed_count": len(records),
        "success_count": len(good),
        "failure_count": len(records) - len(good),
        "pending_count": expected - len(records),
        "shape_distribution": dict(Counter(str(r["output_shape"]) for r in good)),
        "spacing_distribution": dict(Counter(str(r["output_spacing_mm"]) for r in good)),
        "warning_counts": dict(Counter(w for r in records for w in r["warnings"])),
        "normalization": {
            "finite_statistics_count": sum(
                all(
                    np.isfinite(v)
                    for v in r["normalization"].values()
                    if isinstance(v, (int, float))
                )
                for r in good
            ),
            "background_preserved_count": sum(
                r["normalization"]["background_preserved"] for r in good
            ),
            "foreground_count_min": min(
                (r["normalization"]["foreground_voxel_count"] for r in good), default=None
            ),
            "max_absolute_foreground_mean": max(
                (abs(r["normalization"]["normalized_foreground_mean"]) for r in good), default=None
            ),
            "max_foreground_std_error": max(
                (abs(r["normalization"]["normalized_foreground_std"] - 1) for r in good),
                default=None,
            ),
        },
        "final_roi_coverage_min": min(
            (
                r["registration"]["evaluations"][-1]["template_brain_in_bounds_fraction"]
                for r in good
            ),
            default=None,
        ),
    }
    for field, scopes in (
        ("split", ("train", "validation", "test")),
        ("site", ("Guys", "HH", "IOP")),
    ):
        result["by_" + field] = {
            scope: {
                "expected": sum(s[field] == scope for s in records),
                "success": sum(r[field] == scope for r in good),
                "failure": sum(r[field] == scope and r["status"] != "success" for r in records),
                "warning_counts": dict(
                    Counter(w for r in records if r[field] == scope for w in r["warnings"])
                ),
                "grid_and_normalization_pass_count": sum(r[field] == scope for r in good),
                "max_absolute_foreground_mean": max(
                    (
                        abs(r["normalization"]["normalized_foreground_mean"])
                        for r in good
                        if r[field] == scope
                    ),
                    default=None,
                ),
                "max_foreground_std_error": max(
                    (
                        abs(r["normalization"]["normalized_foreground_std"] - 1)
                        for r in good
                        if r[field] == scope
                    ),
                    default=None,
                ),
                "rigid_determinant_max_error": max(
                    (
                        abs(r["registration"]["final_transform"]["rotation_determinant"] - 1)
                        for r in good
                        if r[field] == scope
                    ),
                    default=None,
                ),
                "final_roi_coverage_min": min(
                    (
                        r["registration"]["evaluations"][-1]["template_brain_in_bounds_fraction"]
                        for r in good
                        if r[field] == scope
                    ),
                    default=None,
                ),
            }
            for scope in scopes
        }
    return result


def checkpoint(output, records, expected, state):
    rows = [
        {
            k: r.get(k)
            for k in (
                "subject_id",
                "site",
                "split",
                "status",
                "stage",
                "exception",
                "output_path",
                "source_sha256",
                "output_sha256",
                "runtime_seconds",
                "qc_state",
            )
        }
        | {"warnings": json.dumps(r["warnings"])}
        for r in records
    ]
    frame = pd.DataFrame(rows)
    temporary = output / "work" / "manifest.csv"
    frame.to_csv(temporary, index=False)
    temporary.replace(output / "manifest.csv")
    failures = frame.loc[frame.status != "success"] if len(frame) else frame
    failures.to_csv(output / "failures.csv", index=False)
    summary = summary_records(records, expected)
    summary.update(
        run=state, elapsed_current_session_seconds=time.perf_counter() - state["session_start"]
    )
    atomic_json(output / "summary.json", summary)
    return summary


def run(spec_path, output, *, workers=4, resume=False, regenerate_invalid=False, verify_only=False):
    started = time.perf_counter()
    spec, paths, cohort, parameters = load_spec(spec_path)
    output = Path(output).resolve()
    validate_output_directory(output, [*paths.values(), *(Path(p) for p in cohort.t1_path)])
    if workers < 1 or workers > 4:
        raise ValueError("Use one to four independent single-threaded workers")
    if (regenerate_invalid or verify_only) and not resume:
        raise ValueError("Regeneration/verification require explicit --resume")
    if verify_only and regenerate_invalid:
        raise ValueError("Verification cannot regenerate or move outputs")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Existing final output requires --resume; no implicit overwrite")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("Another final preprocessing process holds the output lock") from None
    try:
        for name in ("volumes", "provenance", "qc", "work"):
            (output / name).mkdir(exist_ok=True)
        subjects = cohort.to_dict("records")
        source_hashes = {s["subject_id"]: sha256(Path(s["t1_path"])) for s in subjects}
        inputs = {str(path): sha256(path) for path in paths.values()}
        inputs[str(Path(spec_path).resolve())] = sha256(Path(spec_path))
        inputs[str(Path(__file__).resolve())] = sha256(Path(__file__))
        fingerprint = hashlib.sha256(
            json.dumps({"inputs": inputs, "sources": source_hashes}, sort_keys=True).encode()
        ).hexdigest()
        provenance_path = output / "run_provenance.json"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            if provenance["fingerprint"] != fingerprint:
                raise ValueError("Resume inputs/software/specification changed")
        else:
            provenance = {
                "fingerprint": fingerprint,
                "input_sha256": inputs,
                "source_sha256": source_hashes,
                "specification": spec,
                "sessions": [],
                "registration_implementation": "unchanged optimizers.register O0",
            }
            atomic_json(provenance_path, provenance)
            cohort.to_csv(output / "cohort.csv", index=False)
        known = {s["subject_id"] for s in subjects}
        unexpected = {p.name[:-7] for p in (output / "volumes").glob("*.nii.gz")} - known
        unexpected |= {p.stem for p in (output / "provenance").glob("*.json")} - known
        if unexpected:
            raise ValueError("Unexpected output subject files: " + str(sorted(unexpected)))
        records, pending = [], []
        for index, subject in enumerate(subjects, 1):
            sid = subject["subject_id"]
            volume = output / "volumes" / f"{sid}.nii.gz"
            record_path = output / "provenance" / f"{sid}.json"
            if volume.exists() or record_path.exists():
                try:
                    records.append(
                        validate_finished(subject, output, source_hashes[sid], fingerprint)
                    )
                    print(f"Validated resume {index}/561: {sid}", file=sys.stderr, flush=True)
                    continue
                except Exception as exc:
                    if not regenerate_invalid:
                        raise ValueError(
                            f"Invalid/incomplete output {sid}: {exc}; "
                            "use --regenerate-invalid explicitly"
                        ) from exc
                    backup = output / "recovery" / uuid.uuid4().hex / sid
                    backup.mkdir(parents=True)
                    for path in (volume, record_path):
                        if path.exists():
                            path.replace(backup / path.name)
            pending.append(subject)
        if verify_only and pending:
            raise ValueError("Verification found missing outputs")
        reused = len(records)
        state = {
            "session_start": started,
            "worker_count": workers,
            "reused_valid_count": reused,
            "new_subject_count": len(pending),
            "verify_only": verify_only,
        }
        if pending:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(paths["template"], paths["mask"], parameters),
            ) as pool:
                futures = {
                    pool.submit(
                        process_subject, s, output, source_hashes[s["subject_id"]], fingerprint
                    ): s
                    for s in pending
                }
                for future in as_completed(futures):
                    subject = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {
                            "subject_id": subject["subject_id"],
                            "site": subject["acquisition_site"],
                            "split": subject["split"],
                            "status": "failed",
                            "stage": "worker",
                            "exception": f"{type(exc).__name__}: {exc}",
                            "qc_state": "worker_failure",
                            "warnings": [],
                        }
                        atomic_json(output / "provenance" / f"{subject['subject_id']}.json", record)
                    records.append(record)
                    print(
                        f"Final preprocessing {len(records)}/561: {subject['subject_id']} "
                        f"{record['status']} ({record.get('runtime_seconds', 0):.1f}s)",
                        file=sys.stderr,
                        flush=True,
                    )
                    if len(records) % 10 == 0 or len(records) == len(subjects):
                        checkpoint(output, records, len(subjects), state)
        provenance["sessions"].append(
            {k: v for k, v in state.items() if k != "session_start"}
            | {"elapsed_seconds": time.perf_counter() - started}
        )
        atomic_json(provenance_path, provenance)
        state["cumulative_session_seconds"] = sum(
            s["elapsed_seconds"] for s in provenance["sessions"]
        )
        summary = checkpoint(output, records, len(subjects), state)
        if any(sha256(Path(path)) != h for path, h in inputs.items()):
            raise ValueError("Frozen input changed during preprocessing")
        return summary
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/preprocessing.frozen.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/preprocessed/final_v1"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--regenerate-invalid", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.spec,
        args.output,
        workers=args.workers,
        resume=args.resume,
        regenerate_invalid=args.regenerate_invalid,
        verify_only=args.verify_only,
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("expected_count", "success_count", "failure_count", "pending_count")
            }
        )
    )
    if summary["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
