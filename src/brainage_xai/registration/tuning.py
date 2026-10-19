"""Controlled two-factor, two-stage Euler3D experiment; baseline is read-only."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from brainage_xai.config.preprocessing import validate_output_directory
from brainage_xai.data.preparation import eligible_subjects
from brainage_xai.preprocessing.orientation import canonicalize

from .pipeline import _euler, _json
from .rigid import resample_to_fixed, source_foreground_clipping, to_sitk, transform_record
from .settings import load_registration_config
from .template import load_template, sha256

CONDITIONS = {
    "A": ("GEOMETRY", False),
    "B": ("GEOMETRY", True),
    "C": ("MOMENTS", False),
    "D": ("MOMENTS", True),
}


def fresh_image(image):
    """Reset ITK requested-region/pipeline state while preserving every voxel and physical grid.

    A shallow Image copy shares underlying ITK state. Resampling can leave a
    requested subregion which ImageMomentsCalculator subsequently uses. Deep
    buffer reconstruction prevents earlier conditions/QC from affecting MOMENTS.
    """
    result = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
    result.CopyInformation(image)
    return result


def check_mask(fixed, mask):
    if mask is None or any(
        not np.allclose(getattr(fixed, name)(), getattr(mask, name)(), rtol=0, atol=1e-6)
        for name in ("GetSize", "GetOrigin", "GetSpacing", "GetDirection")
    ):
        raise ValueError("Fixed brain mask must match template physical grid")
    values = sitk.GetArrayViewFromImage(mask)
    if not np.isin(values, [0, 1]).all() or not np.any(values):
        raise ValueError("Fixed brain mask must be nonempty and binary")


def initialize_mode(fixed, moving, mode):
    if mode not in {"GEOMETRY", "MOMENTS"}:
        raise ValueError("Unsupported initialization")
    fixed, moving = fresh_image(fixed), fresh_image(moving)
    tx = sitk.Euler3DTransform()
    tx.SetComputeZYX(True)
    return sitk.Euler3DTransform(
        sitk.CenteredTransformInitializer(
            fixed, moving, tx, getattr(sitk.CenteredTransformInitializerFilter, mode)
        )
    )


def full_metric(fixed, moving, transform, mask, parameters):
    fixed, moving = fresh_image(fixed), fresh_image(moving)
    mask = fresh_image(mask) if mask is not None else None
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(1)
    method.SetMetricAsMattesMutualInformation(parameters.histogram_bins)
    method.SetMetricSamplingStrategy(method.NONE)
    method.SetInterpolator(sitk.sitkLinear)
    if mask is not None:
        check_mask(fixed, mask)
        method.SetMetricFixedMask(mask)
    method.SetInitialTransform(transform)
    value = float(method.MetricEvaluate(fixed, moving))
    if not np.isfinite(value):
        raise ValueError("Nonfinite full-resolution metric")
    # MetricEvaluate does not retain an optimizer metric object for this getter.
    return {
        "MI": value,
        "metric_valid_points": None,
        "valid_points_note": "MetricEvaluate does not expose retained valid-point count",
        "evaluation": "full resolution; no sampling; negative Mattes MI; lower better",
    }


def brain_coverage(fixed, moving, mask, transform):
    """Fixed ROI centers -> moving continuous index; voxel-cell domain [-.5,size-.5)."""
    check_mask(fixed, mask)
    tx = np.asarray(transform_record(transform)["matrix_lps"])
    fmat = np.asarray(fixed.GetDirection()).reshape(3, 3) @ np.diag(fixed.GetSpacing())
    mmat = np.asarray(moving.GetDirection()).reshape(3, 3) @ np.diag(moving.GetSpacing())
    mapping = np.linalg.solve(mmat, tx[:3, :3] @ fmat)
    offset = np.linalg.solve(mmat, tx[:3, :3] @ fixed.GetOrigin() + tx[:3, 3] - moving.GetOrigin())
    data = sitk.GetArrayViewFromImage(mask)
    total = inside = 0
    for z in range(data.shape[0]):
        y, x = np.nonzero(data[z])
        indices = mapping @ np.vstack((x, y, np.full(len(x), z))) + offset[:, None]
        valid = (indices >= -0.5) & (indices < np.array(moving.GetSize())[:, None] - 0.5)
        total += len(x)
        inside += int(valid.all(axis=0).sum())
    return {
        "template_brain_in_bounds_fraction": inside / total,
        "template_brain_inside_count": inside,
        "template_brain_outside_count": total - inside,
        "definition": "FOV coverage of fixed ROI centers, not anatomical correctness",
    }


def optimize_stage(fixed, moving, transform, mask, p, translation_only):
    fixed, moving = fresh_image(fixed), fresh_image(moving)
    mask = fresh_image(mask) if mask is not None else None
    input_info = transform_record(transform)
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(1)
    method.SetMetricAsMattesMutualInformation(p.histogram_bins)
    method.SetMetricSamplingStrategy(getattr(method, p.sampling_strategy))
    if p.sampling_strategy != "NONE":
        method.SetMetricSamplingPercentage(p.sampling_fraction, p.sampling_seed)
    if mask is not None:
        method.SetMetricFixedMask(mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        p.learning_rate,
        p.min_step,
        p.iterations_per_level,
        p.relaxation_factor,
        p.gradient_tolerance,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    weights = [0, 0, 0, 1, 1, 1] if translation_only else [1] * 6
    method.SetOptimizerWeights(weights)
    # Coarse translation stage uses the same first two levels in all four conditions.
    factors = (
        p.shrink_factors[:-1]
        if translation_only and len(p.shrink_factors) > 1
        else p.shrink_factors
    )
    sigmas = p.smoothing_sigmas_mm[: len(factors)]
    method.SetShrinkFactorsPerLevel(factors)
    method.SetSmoothingSigmasPerLevel(sigmas)
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    final = sitk.Euler3DTransform(transform)
    method.SetInitialTransform(final, inPlace=True)
    trajectory, stops = [], []

    def iteration():
        trajectory.append(
            {
                "level": int(method.GetCurrentLevel()),
                "iteration": int(method.GetOptimizerIteration()),
                "sampled_metric": float(method.GetMetricValue()),
                "valid_points": int(method.GetMetricNumberOfValidPoints()),
            }
        )

    def level():
        if method.GetCurrentLevel() > 0:
            stops.append(method.GetOptimizerStopConditionDescription())

    method.AddCommand(sitk.sitkIterationEvent, iteration)
    method.AddCommand(sitk.sitkMultiResolutionIterationEvent, level)
    try:
        method.Execute(fixed, moving)
    finally:
        # Callbacks close over method: break the Python/C++ reference cycle so
        # completed stages cannot retain image buffers across subjects.
        method.RemoveAllCommands()
    stops.append(method.GetOptimizerStopConditionDescription())
    transform_record(final)
    if translation_only and not np.array_equal(
        final.GetParameters()[:3], transform.GetParameters()[:3]
    ):
        raise ValueError("Translation-only stage changed rotation")
    return final, {
        "iterations": len(trajectory),
        "stop_condition": stops[-1],
        "level_stop_conditions": stops,
        "trajectory": trajectory,
        "weights": weights,
        "shrink_factors": list(factors),
        "smoothing_sigmas_mm": list(sigmas),
        "input_transform": input_info,
        "output_transform": transform_record(final),
    }


def tune(fixed, moving, brain_mask, mode, masked, parameters):
    parameters.validate()
    check_mask(fixed, brain_mask)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    mask = brain_mask if masked else None
    initial = initialize_mode(fixed, moving, mode)
    initial_info = transform_record(initial)
    evaluations = []

    def evaluate(tx):
        return dict(
            full_metric(fixed, moving, tx, mask, parameters),
            **brain_coverage(fixed, moving, brain_mask, tx),
        )

    evaluations.append(evaluate(initial))
    translated, stage1 = optimize_stage(fixed, moving, initial, mask, parameters, True)
    evaluations.append(evaluate(translated))
    final, stage2 = optimize_stage(fixed, moving, translated, mask, parameters, False)
    evaluations.append(evaluate(final))
    return final, {
        "initial_transform": initial_info,
        "final_transform": transform_record(final),
        "evaluations": evaluations,
        "translation_stage": stage1,
        "rigid_stage": stage2,
        "MI_initial": evaluations[0]["MI"],
        "MI_after_translation": evaluations[1]["MI"],
        "MI_final": evaluations[2]["MI"],
        "within_condition_MI_change": evaluations[0]["MI"] - evaluations[2]["MI"],
    }


def comparison_row(record):
    row = {
        k: record.get(k)
        for k in (
            "subject_id",
            "site",
            "condition",
            "initialization_mode",
            "fixed_mask_used",
            "success",
            "MI_initial",
            "MI_after_translation",
            "MI_final",
            "within_condition_MI_change",
            "output_geometry_valid",
            "technical_warnings",
            "qc_path",
        )
    }
    row["technical_execution_success"] = record.get("success", False)
    row["anatomical_qc_status"] = record.get("anatomical_qc_status", "not_reviewed")
    for name in ("translation_stage", "rigid_stage"):
        for field in ("iterations", "stop_condition"):
            row[f"{name}_{field}"] = record.get(name, {}).get(field)
    row.update(record.get("final_transform", {}))
    for label, index in [("initial", 0), ("after_translation", 1), ("final", -1)]:
        ev = record.get("evaluations", [])
        row[f"template_brain_in_bounds_fraction_{label}"] = (
            ev[index]["template_brain_in_bounds_fraction"] if ev and ev[index] else None
        )
    return {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}


def aggregate(records):
    good = [r for r in records if r["success"]]
    changes = [r["within_condition_MI_change"] for r in good]
    stops = [
        s
        for r in good
        for stage in ("translation_stage", "rigid_stage")
        for s in r.get(stage, {}).get("level_stop_conditions", [])
    ]

    def quant(values):
        return (
            dict(zip(("p10", "p50", "p90"), np.percentile(values, [10, 50, 90], axis=0).tolist()))
            if values
            else {}
        )

    return {
        "execution_success_count": len(good),
        "execution_failure_count": len(records) - len(good),
        "budget_exhausted_subject_count": sum(
            any(
                "maximum" in s.lower()
                for stage in ("translation_stage", "rigid_stage")
                for s in r.get(stage, {}).get("level_stop_conditions", [])
            )
            for r in good
        ),
        "stop_condition_counts": dict(Counter(stops)),
        "stop_category_counts": dict(
            Counter(
                "iteration_budget"
                if "maximum" in s.lower()
                else "step_too_small"
                if "step too small" in s.lower()
                else "gradient_tolerance"
                if "gradient magnitude tolerance" in s.lower()
                else "other"
                for s in stops
            )
        ),
        "final_rigid_level_budget_exhausted_subject_count": sum(
            "maximum" in r.get("rigid_stage", {}).get("stop_condition", "").lower() for r in good
        ),
        "MI_improvement_count": sum(x > 0 for x in changes),
        "MI_degradation_count": sum(x < 0 for x in changes),
        "median_within_condition_MI_improvement": float(np.median(changes)) if changes else None,
        "rotation_deg_lps_xyz": quant([r["final_transform"]["rotation_deg_lps_xyz"] for r in good]),
        "translation_mm_lps_xyz": quant(
            [r["final_transform"]["translation_mm_lps_xyz"] for r in good]
        ),
        "template_brain_in_bounds_fraction": quant(
            [r["evaluations"][-1]["template_brain_in_bounds_fraction"] for r in good]
        ),
        "technical_warning_count": sum(len(r["technical_warnings"]) for r in records),
        "technical_warning_subject_count": sum(bool(r["technical_warnings"]) for r in records),
    }


def run(config_path, baseline, output):
    from .tuning_qc import contact_sheets, planes, subject_panel

    config = load_registration_config(config_path)
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    cohort = pd.read_csv(baseline / "pilot_selection.csv")
    if (
        len(cohort) != 18
        or cohort.subject_id.nunique() != 18
        or cohort.acquisition_site.value_counts().to_dict() != {"Guys": 6, "HH": 6, "IOP": 6}
    ):
        raise ValueError("Exact baseline eighteen-subject cohort is required")
    eligible = eligible_subjects(config.manifest_path).set_index("subject_id")
    for s in cohort.to_dict("records"):
        if (
            s["subject_id"] not in eligible.index
            or Path(s["t1_path"]).resolve()
            != Path(eligible.loc[s["subject_id"], "t1_path"]).resolve()
        ):
            raise ValueError("Baseline cohort no longer matches eligible source manifest")
    protected = [p for p in baseline.rglob("*") if p.is_file()]
    inputs = (
        [Path(s) for s in cohort.t1_path]
        + protected
        + [config.template_path, config.brain_mask_path, config_path]
    )
    validate_output_directory(output, inputs)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Tuning output must be empty; baseline cannot be overwritten")
    before = {str(p): sha256(p) for p in inputs}
    bundle = load_template(config.template_path, config.brain_mask_path)
    fixed, brain = to_sitk(bundle.image), sitk.Cast(to_sitk(bundle.mask), sitk.sitkUInt8)
    check_mask(fixed, brain)
    p = replace(config.parameters, iterations_per_level=300, evaluation_shrink_factor=1)
    output.mkdir(parents=True, exist_ok=True)
    (output / "subjects").mkdir()
    (output / "qc").mkdir()
    _json(
        output / "provenance.json",
        {
            "parameters": asdict(p),
            "conditions": CONDITIONS,
            "template": bundle.provenance,
            "input_sha256": before,
            "SimpleITK": sitk.Version_VersionString(),
            "budget": "300/level: translation 4,2; rigid 4,2,1; maximum 1500 iterations/condition",
            "MI_comparison": "within-condition only; masked/unmasked domains differ",
            "transform_convention": "fixed-template LPS -> moving-source LPS pull mapping",
        },
    )
    records, panels = [], []
    fixed_planes, mask_planes = planes(fixed), planes(brain)
    for n, subject in enumerate(cohort.to_dict("records"), 1):
        sid, site = subject["subject_id"], subject["acquisition_site"]
        source = canonicalize(nib.load(subject["t1_path"]))
        moving = to_sitk(source)
        old = json.loads((baseline / "subjects" / f"{sid}.json").read_text())
        if sha256(Path(subject["t1_path"])) != old["source_sha256"]:
            raise ValueError("Source differs from preserved baseline")
        initial, final = _euler(old["initial_transform"]), _euler(old["final_transform"])
        ev = [
            dict(full_metric(fixed, moving, t, brain, p), **brain_coverage(fixed, moving, brain, t))
            for t in (initial, final)
        ]
        base = {
            "subject_id": sid,
            "site": site,
            "condition": "v1",
            "initialization_mode": "GEOMETRY",
            "fixed_mask_used": True,
            "success": True,
            "MI_initial": ev[0]["MI"],
            "MI_after_translation": None,
            "MI_final": ev[1]["MI"],
            "within_condition_MI_change": ev[0]["MI"] - ev[1]["MI"],
            "evaluations": [ev[0], None, ev[1]],
            "initial_transform": old["initial_transform"],
            "final_transform": old["final_transform"],
            "technical_warnings": old.get("warnings", []),
            "output_geometry_valid": True,
            "rigid_stage": {
                "iterations": old["iteration_count"],
                "stop_condition": old["optimizer_stop_condition"],
                "level_stop_conditions": [old["optimizer_stop_condition"]],
            },
            "baseline_note": (
                "Original transforms reevaluated full resolution; original files unchanged"
            ),
        }
        views = {"Fixed": fixed_planes, "v1": planes(resample_to_fixed(moving, fixed, final))}
        for condition, (mode, masked) in CONDITIONS.items():
            print(
                f"Tuning subject {n}/18 {sid}, condition {condition} ({mode}, mask={masked})",
                file=sys.stderr,
                flush=True,
            )
            record = {
                "subject_id": sid,
                "site": site,
                "condition": condition,
                "initialization_mode": mode,
                "fixed_mask_used": masked,
                "success": False,
                "output_geometry_valid": False,
                "technical_warnings": [],
                "anatomical_qc_status": "not_reviewed",
            }
            try:
                tx, trace = tune(fixed, moving, brain, mode, masked, p)
                record.update(trace)
                image = resample_to_fixed(moving, fixed, tx)
                record["output_geometry_valid"] = all(
                    getattr(image, k)() == getattr(fixed, k)()
                    for k in ("GetSize", "GetOrigin", "GetSpacing", "GetDirection")
                )
                if (
                    not record["output_geometry_valid"]
                    or not np.isfinite(sitk.GetArrayViewFromImage(image)).all()
                ):
                    raise ValueError("Invalid resampled geometry/intensity")
                record["success"] = True
                record["source_nonbackground_context"] = source_foreground_clipping(
                    np.asarray(source.dataobj),
                    source.affine,
                    bundle.image.shape,
                    bundle.image.affine,
                    tx,
                )
                if trace["within_condition_MI_change"] < 0:
                    record["technical_warnings"].append("deterministic_MI_degradation")
                if any(
                    "maximum" in s.lower()
                    for stage in ("translation_stage", "rigid_stage")
                    for s in trace[stage]["level_stop_conditions"]
                ):
                    record["technical_warnings"].append("iteration_budget_exhausted")
                if (
                    max(abs(x) for x in trace["final_transform"]["rotation_deg_lps_xyz"])
                    > p.rotation_warning_deg
                ):
                    record["technical_warnings"].append("provisional_large_rotation")
                if (
                    np.linalg.norm(trace["final_transform"]["translation_mm_lps_xyz"])
                    > p.translation_warning_mm
                ):
                    record["technical_warnings"].append("provisional_large_translation")
                views[condition] = planes(image)
                if n == 1 and condition == "A":
                    repeated, repeated_trace = tune(fixed, moving, brain, mode, masked, p)
                    repeat = {
                        "condition": condition,
                        "subject_id": sid,
                        "identical_transform": tx.GetParameters() == repeated.GetParameters(),
                        "identical_trace": trace == repeated_trace,
                    }
                    _json(output / "repeatability.json", repeat)
                    if not all(repeat[k] for k in ("identical_transform", "identical_trace")):
                        raise ValueError("Real-subject repeatability failed")
            except (RuntimeError, ValueError) as exc:
                record["success"] = False
                record["technical_warnings"].append(str(exc))
            record["qc_path"] = str(output / "qc" / f"{sid}_comparison.png")
            _json(output / "subjects" / f"{sid}_{condition}.json", record)
            records.append(record)
        base["qc_path"] = str(output / "qc" / f"{sid}_comparison.png")
        records.append(base)
        _json(output / "subjects" / f"{sid}_v1.json", base)
        subject_panel(views, mask_planes, Path(base["qc_path"]), sid)
        panels.append((sid, site, views))
        pd.DataFrame([comparison_row(r) for r in records]).to_csv(
            output / "comparison.csv", index=False
        )
    contact_sheets(panels, mask_planes, output / "qc")
    summary = {
        c: {
            "overall": aggregate([r for r in records if r["condition"] == c]),
            "by_site": {
                site: aggregate([r for r in records if r["condition"] == c and r["site"] == site])
                for site in ("Guys", "HH", "IOP")
            },
        }
        for c in ("v1", *CONDITIONS)
    }
    _json(output / "summary.json", summary)
    if any(sha256(Path(path)) != digest for path, digest in before.items()):
        raise ValueError("Protected source or baseline changed")
    print(f"Completed controlled tuning: 18 subjects, 72 tuned attempts. Outputs: {output}")
    return finalize_tables(output)


def finalize_tables(output):
    """Rebuild derived tables from private case records without rerunning registration."""
    output = Path(output)
    records = [json.loads(p.read_text()) for p in sorted((output / "subjects").glob("*.json"))]
    if len(records) != 90:
        raise ValueError("Expected eighteen subjects times five conditions")
    for sid in {r["subject_id"] for r in records}:
        cases = {r["condition"]: r for r in records if r["subject_id"] == sid}
        for a, b in (("A", "B"), ("C", "D")):
            if (
                cases[a].get("initial_transform") is not None
                and cases[b].get("initial_transform") is not None
                and cases[a]["initial_transform"] != cases[b]["initial_transform"]
            ):
                raise ValueError("Fixed-mask factor changed initialization; experiment invalid")
    for r in records:
        if r["condition"] == "v1":
            r["technical_warnings"] = [
                w
                for w in r["technical_warnings"]
                if "metric_degradation" not in w and w != "deterministic_MI_degradation"
            ]
            if r["within_condition_MI_change"] < 0:
                r["technical_warnings"].append("deterministic_MI_degradation")
            r["anatomical_qc_status"] = "not_reviewed"
            _json(output / "subjects" / f"{r['subject_id']}_v1.json", r)
    pd.DataFrame([comparison_row(r) for r in records]).to_csv(
        output / "comparison.csv", index=False
    )
    summary = {
        c: {
            "overall": aggregate([r for r in records if r["condition"] == c]),
            "by_site": {
                site: aggregate([r for r in records if r["condition"] == c and r["site"] == site])
                for site in ("Guys", "HH", "IOP")
            },
        }
        for c in ("v1", *CONDITIONS)
    }

    def quant(values):
        return dict(
            zip(("p10", "p50", "p90"), np.percentile(values, [10, 50, 90], axis=0).tolist())
        )

    initialization = {}
    for condition in CONDITIONS:
        cases = [r for r in records if r["condition"] == condition and r["success"]]
        if not cases:
            continue
        initialization[condition] = {
            "mode": cases[0]["initialization_mode"],
            "fixed_mask_used": cases[0]["fixed_mask_used"],
            "center_mm_lps": quant([r["initial_transform"]["center_mm_lps"] for r in cases]),
            "rotation_deg_lps_xyz": quant(
                [r["initial_transform"]["rotation_deg_lps_xyz"] for r in cases]
            ),
            "translation_mm_lps_xyz": quant(
                [r["initial_transform"]["translation_mm_lps_xyz"] for r in cases]
            ),
            "MI_initial_within_mask_domain": quant([r["MI_initial"] for r in cases]),
            "template_brain_in_bounds_fraction_initial": quant(
                [r["evaluations"][0]["template_brain_in_bounds_fraction"] for r in cases]
            ),
        }
    summary["initialization_analysis"] = initialization
    summary["interpretation"] = {
        "MI": "positive within_condition_MI_change is improvement; no cross-mask magnitude ranking",
        "coverage": "FOV availability, not anatomical correctness; no acceptance threshold",
        "baseline_stops": "v1 retains only final-level stop description; earlier stops unavailable",
        "anatomical_qc": "technical success does not confer anatomical acceptance",
    }
    _json(output / "summary.json", summary)
    provenance = json.loads((output / "provenance.json").read_text())
    provenance["common_optimizer_parameters"] = {
        k: v
        for k, v in provenance["parameters"].items()
        if k not in ("initialization", "use_fixed_mask")
    }
    provenance["parameter_note"] = (
        "parameters contains original config defaults with tuning budget/evaluation overrides; "
        "conditions and each case specify actual initialization and fixed-mask use"
    )
    _json(output / "provenance.json", provenance)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.baseline, args.output)


if __name__ == "__main__":
    main()
