"""Final rigid-versus-affine decision: reuse eighteen S2 cases, run eighteen affine stages."""

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from brainage_xai.config.preprocessing import validate_output_directory
from brainage_xai.preprocessing.orientation import canonicalize

from .affine import REVIEW_LIMITS, affine_coverage, affine_record, optimize_affine
from .optimizer_qc import _panel
from .optimizer_study import core_hashes, environment, load_inputs
from .optimizers import optimizer_configuration, quantiles, stop_category
from .pipeline import _euler, _json
from .rigid import resample_to_fixed, to_sitk
from .sampling_study import load_baseline, reference_record, require_new_output
from .template import sha256
from .tuning import full_metric
from .tuning_qc import planes

SITES = ("Guys", "HH", "IOP")


def run_affine_case(subject, reference, fixed, moving, mask, p, output):
    """No rigid optimization and no baseline metric reevaluation; native source sampled once."""
    start = time.perf_counter()
    r = {
        "subject_id": subject["subject_id"],
        "site": subject["acquisition_site"],
        "method": "AFFINE",
        "reused_baseline": False,
        "success": False,
        "output_geometry_valid": False,
        "fixed_metric_mask_used": False,
        "moving_metric_mask_used": False,
        "warnings": [],
        "MI_initial": reference["MI_final"],
        "initial_MI_origin": "stored S2 final MI; same mapping; no rigid reevaluation",
        "rigid_baseline_result_sha256": reference["baseline_result_sha256"],
        "anatomical_qc_status": "not_reviewed",
    }
    view = None
    try:
        tx, stage, rows = optimize_affine(fixed, moving, _euler(reference["final_transform"]), p)
        r.update(
            affine_stage=stage,
            initial_transform=stage["input_transform"],
            final_transform=affine_record(tx),
        )
        r["evaluations"] = [
            {
                "MI": reference["MI_final"],
                **{
                    k: v
                    for k, v in reference["evaluations"][-1].items()
                    if k.startswith("template_")
                },
                "origin": "stored rigid baseline",
            },
            dict(
                full_metric(fixed, moving, tx, None, p), **affine_coverage(fixed, moving, mask, tx)
            ),
        ]
        r["MI_final"] = r["evaluations"][-1]["MI"]
        r["delta_total"] = r["MI_initial"] - r["MI_final"]
        image = resample_to_fixed(moving, fixed, tx)
        r["output_geometry_valid"] = all(
            getattr(image, k)() == getattr(fixed, k)()
            for k in ("GetSize", "GetOrigin", "GetSpacing", "GetDirection")
        )
        if (
            not r["output_geometry_valid"]
            or not np.isfinite(sitk.GetArrayViewFromImage(image)).all()
        ):
            raise ValueError("Invalid affine output grid/intensities")
        r["warnings"].extend(r["final_transform"]["review_warnings"])
        if r["delta_total"] < 0:
            r["warnings"].append("deterministic_MI_degradation")
        if stage["budget_hit"]:
            r["warnings"].append("affine_iteration_budget")
        if r["evaluations"][-1]["template_brain_outside_count"]:
            r["warnings"].append("template_ROI_has_outside_FOV_samples_not_anatomical_error")
        r["trajectory_csv"] = str(output / "traces" / f"{r['subject_id']}_AFFINE.csv")
        pd.DataFrame(rows).to_csv(r["trajectory_csv"], index=False)
        r["optimizer_execute_seconds"] = stage["optimizer_execute_seconds"]
        r["success"] = True
        r["runtime_seconds"] = time.perf_counter() - start
        r["estimated_rigid_plus_affine_seconds"] = (
            reference["runtime_seconds"] + r["runtime_seconds"]
        )
        view = planes(image)
    except (RuntimeError, ValueError) as exc:
        r["runtime_seconds"] = time.perf_counter() - start
        r["warnings"].append("execution_failure: " + str(exc))
        print(f"{r['subject_id']} affine failed: {exc}", file=sys.stderr, flush=True)
    return r, view


def summarize(records):
    """Require complete paired design; report failures rather than dropping subjects."""
    if len(records) != 36 or len({(r["subject_id"], r["method"]) for r in records}) != 36:
        raise ValueError("Require eighteen unique rigid/affine pairs")
    grouped = {}
    for r in records:
        grouped.setdefault(r["subject_id"], {})[r["method"]] = r
    if len(grouped) != 18 or any(set(g) != {"RIGID", "AFFINE"} for g in grouped.values()):
        raise ValueError("Invalid comparison design")
    pairs = []
    for sid, cases in grouped.items():
        rigid, affine = cases["RIGID"], cases["AFFINE"]
        if rigid["site"] != affine["site"]:
            raise ValueError("Paired site mismatch")
        if affine["MI_initial"] != rigid["MI_final"]:
            raise ValueError("Affine must start at stored rigid final metric")
        if rigid["success"] and affine["success"]:
            a, b = rigid["final_transform"], affine["final_transform"]
            if not np.allclose(
                a["matrix_lps"], affine["initial_transform"]["matrix_lps"], atol=1e-10, rtol=0
            ):
                raise ValueError("Affine initialization differs from rigid final mapping")
            relative = np.asarray(a["matrix_lps"])[:3, :3].T @ np.asarray(
                b["polar_rotation_matrix"]
            )
            angle = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
            delta = np.asarray(b["translation_mm_lps_xyz"]) - a["translation_mm_lps_xyz"]
            pairs.append(
                {
                    "subject_id": sid,
                    "site": rigid["site"],
                    "MI_rigid": rigid["MI_final"],
                    "MI_affine": affine["MI_final"],
                    "MI_gain": rigid["MI_final"] - affine["MI_final"],
                    "polar_rotation_disagreement_deg": angle,
                    "common_center_pull_displacement_mm": float(np.linalg.norm(delta)),
                    "affine_runtime_seconds": affine["runtime_seconds"],
                    "rigid_historical_runtime_seconds": rigid["runtime_seconds"],
                    "estimated_pipeline_runtime_seconds": affine[
                        "estimated_rigid_plus_affine_seconds"
                    ],
                    "affine_determinant": b["determinant"],
                    "affine_singular_values": b["singular_values"],
                    "affine_global_pull_scale": b["global_isotropic_pull_scale"],
                    "affine_max_absolute_column_cosine": b["max_absolute_column_cosine"],
                    "affine_review_warnings": b["review_warnings"],
                }
            )

    def aggregate(rows):
        good = [r for r in rows if r["success"]]
        stages = [r["affine_stage"] if r["method"] == "AFFINE" else r["rigid_stage"] for r in good]
        stops = [stop for stage in stages for stop in stage["level_stops"]]
        result = {
            "count": len(rows),
            "success_count": len(good),
            "failure_count": len(rows) - len(good),
            "output_geometry_valid_count": sum(r["output_geometry_valid"] for r in rows),
            "MI_final": quantiles([r["MI_final"] for r in good]),
            "runtime_seconds": quantiles([r["runtime_seconds"] for r in good]),
            "final_level_budget_hit_count": sum(s["final_level_budget_hit"] for s in stages),
            "any_level_budget_hit_count": sum(s["budget_hit"] for s in stages),
            "iterations": quantiles([s["iterations"] for s in stages]),
            "stop_categories": dict(Counter(map(stop_category, stops))),
            "stop_conditions": dict(Counter(stops)),
            "template_brain_in_bounds_fraction": quantiles(
                [r["evaluations"][-1]["template_brain_in_bounds_fraction"] for r in good]
            ),
        }
        if rows and rows[0]["method"] == "AFFINE":
            for field in (
                "determinant",
                "singular_values",
                "global_isotropic_pull_scale",
                "max_absolute_principal_scale_deviation",
                "max_absolute_column_cosine",
                "max_absolute_normalized_polar_shear",
                "inverse_moving_to_fixed_volume_factor",
                "translation_mm_lps_xyz",
                "polar_rotation_angle_deg",
            ):
                result[field] = quantiles([r["final_transform"][field] for r in good])
            result["deformation_review_count"] = sum(
                bool(r["final_transform"]["review_warnings"]) for r in good
            )
            result["deformation_warning_counts"] = dict(
                Counter(w for r in good for w in r["final_transform"]["review_warnings"])
            )
        return result

    summary = {
        method: {
            "overall": aggregate([r for r in records if r["method"] == method]),
            "by_site": {
                site: aggregate([r for r in records if r["method"] == method and r["site"] == site])
                for site in SITES
            },
        }
        for method in ("RIGID", "AFFINE")
    }
    summary["paired"] = {}
    for scope in ("overall", *SITES):
        rows = [r for r in pairs if scope == "overall" or r["site"] == scope]
        summary["paired"][scope] = {
            "complete_pair_count": len(rows),
            "affine_lower_MI_count": sum(r["MI_gain"] > 0 for r in rows),
            **{
                key: quantiles([r[key] for r in rows])
                for key in (
                    "MI_gain",
                    "polar_rotation_disagreement_deg",
                    "common_center_pull_displacement_mm",
                    "estimated_pipeline_runtime_seconds",
                )
            },
        }
    return summary, pairs


def finalize(output):
    records = [json.loads(p.read_text()) for p in sorted((output / "subjects").glob("*.json"))]
    summary, pairs = summarize(records)
    rows = []
    for r in records:
        row = {
            k: r.get(k)
            for k in (
                "subject_id",
                "site",
                "method",
                "success",
                "reused_baseline",
                "MI_initial",
                "MI_final",
                "delta_total",
                "runtime_seconds",
                "output_geometry_valid",
                "warnings",
            )
        }
        if r["success"]:
            stage = r["affine_stage"] if r["method"] == "AFFINE" else r["rigid_stage"]
            row.update(
                {
                    k: stage[k]
                    for k in ("iterations", "stop", "budget_hit", "final_level_budget_hit")
                }
            )
            row["template_brain_in_bounds_fraction"] = r["evaluations"][-1][
                "template_brain_in_bounds_fraction"
            ]
            row["transform_diagnostics"] = r["final_transform"]
        rows.append(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
        )
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(
        [{k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in pairs]
    ).to_csv(output / "paired_comparison.csv", index=False)
    _json(output / "summary.json", summary)
    return summary


def run(config, baseline, pilot, corrected, output):
    baseline, pilot, corrected, output = [
        Path(p).resolve() for p in (baseline, pilot, corrected, output)
    ]
    require_new_output(output)
    cohort, p, bundle, fixed, mask, inputs = load_inputs(config, pilot, corrected, output)
    old = load_baseline(baseline, cohort, p)
    previous = output.parent
    inputs += [path for path in previous.rglob("*") if path.is_file()]
    validate_output_directory(output, inputs)
    hashes = {str(path): sha256(path) for path in inputs}
    for folder in ("subjects", "traces", "qc"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output / "selection.csv", index=False)
    _json(
        output / "provenance.json",
        {
            "shared_parameters": asdict(p),
            "optimizer_configuration": optimizer_configuration("O0"),
            "affine_stages_per_subject": 1,
            "affine_parameter_count": 12,
            "new_registration_count": 18,
            "rigid_registration_count": 0,
            "initialization": (
                "existing final S2/O0 pull transform converted to affine; no new warm-up"
            ),
            "core_sha256": core_hashes(),
            "study_source_sha256": {
                name: sha256(Path(__file__).with_name(name))
                for name in ("affine.py", "affine_study.py")
            },
            "template": bundle.provenance,
            "environment": environment(),
            "input_sha256": hashes,
            "deformation_review_limits": REVIEW_LIMITS,
            "mask_role": "FOV coverage/display only; no metric masks",
            "runtime_note": (
                "rigid historical; affine incremental incl final MI/resampling, "
                "excluding IO/QC; total is estimate"
            ),
            "decision_policy": (
                "clear consistent visual benefit across sites without concerning deformation; "
                "otherwise RIGID; no more tuning"
            ),
        },
    )
    fixed_planes, mask_planes = planes(fixed), planes(mask)
    panels = []
    for n, subject in enumerate(cohort.to_dict("records"), 1):
        sid = subject["subject_id"]
        stored, path = old[sid]
        reference = reference_record(stored, path)
        reference.update(method="RIGID", qc_path=str(output / "qc" / f"{sid}_comparison.png"))
        moving = to_sitk(canonicalize(nib.load(subject["t1_path"])))
        _json(output / "subjects" / f"{sid}_RIGID.json", reference)
        rigid_view = planes(resample_to_fixed(moving, fixed, _euler(stored["final_transform"])))
        print(
            f"Affine study {n}/18 {sid}: one affine stage; S2 reused", file=sys.stderr, flush=True
        )
        result, affine_view = run_affine_case(subject, reference, fixed, moving, mask, p, output)
        result["qc_path"] = reference["qc_path"]
        _json(output / "subjects" / f"{sid}_AFFINE.json", result)
        _panel(
            [
                ("MNI template", fixed_planes),
                ("rigid S2", rigid_view),
                ("rigid S2 / fixed checkerboard", rigid_view),
                ("affine", affine_view),
                ("affine / fixed checkerboard", affine_view),
            ],
            fixed_planes,
            mask_planes,
            Path(reference["qc_path"]),
            sid + " — rigid versus affine",
            True,
        )
        panels.append((sid, subject["acquisition_site"], rigid_view, affine_view))
        print(f"Completed {n}/18: success={result['success']}", file=sys.stderr, flush=True)
    for site in SITES:
        rows = [("MNI template", fixed_planes)]
        for sid, s, rv, av in panels:
            if site == s:
                rows.extend([(sid + " rigid", rv), (sid + " affine", av)])
        _panel(
            rows,
            fixed_planes,
            mask_planes,
            output / "qc" / f"site_{site}.png",
            site + " comparison",
        )
    summary = finalize(output)
    if any(sha256(Path(path)) != h for path, h in hashes.items()):
        raise ValueError("Protected source/template/prior artifact changed")
    print(f"Completed 18 affine registrations; 18 rigid results reused: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/registration.local.yaml"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("artifacts/registration/rigid_optimizer_v4")
    )
    parser.add_argument("--pilot", type=Path, default=Path("artifacts/registration/rigid_pilot_v1"))
    parser.add_argument(
        "--corrected", type=Path, default=Path("artifacts/registration/rigid_tuning_v3")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        finalize(args.output.resolve())
    else:
        run(args.config, args.baseline, args.pilot, args.corrected, args.output)


if __name__ == "__main__":
    main()
