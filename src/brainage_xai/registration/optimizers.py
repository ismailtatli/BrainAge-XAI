"""Predefined optimizer-only Euler3D comparison, with isolated image state."""

import time
from collections import Counter
from copy import deepcopy

import numpy as np
import SimpleITK as sitk

from .rigid import transform_record
from .tuning import brain_coverage, fresh_image, full_metric, initialize_mode

# Defined before inspecting v4 outcomes. Iteration limits are outer iterations per level.
SPECS = {
    "O0": {
        "family": "RegularStepGradientDescent",
        "kwargs": {
            "learningRate": 2.0,
            "minStep": 0.001,
            "numberOfIterations": 300,
            "relaxationFactor": 0.5,
            "gradientMagnitudeTolerance": 1e-6,
            "estimateLearningRate": "Never",
            "maximumStepSizeInPhysicalUnits": 0.0,
        },
    },
    "O1": {
        "family": "GradientDescentLineSearch",
        "kwargs": {
            "learningRate": 1.0,
            "numberOfIterations": 300,
            "convergenceMinimumValue": 1e-6,
            "convergenceWindowSize": 10,
            "lineSearchLowerLimit": 0.0,
            "lineSearchUpperLimit": 2.0,
            "lineSearchEpsilon": 0.01,
            "lineSearchMaximumIterations": 20,
            "estimateLearningRate": "Once",
            "maximumStepSizeInPhysicalUnits": 1.0,
        },
    },
    "O2": {
        "family": "RegularStepGradientDescent",
        "kwargs": {
            "learningRate": 1.0,
            "minStep": 0.0001,
            "numberOfIterations": 300,
            "relaxationFactor": 0.5,
            "gradientMagnitudeTolerance": 1e-6,
            "estimateLearningRate": "Once",
            "maximumStepSizeInPhysicalUnits": 1.0,
        },
    },
}
SCALE_POLICY = {
    "method": "SetOptimizerScalesFromPhysicalShift",
    "centralRegionRadius": 5,
    "smallParameterVariation": 0.01,
}
DISAGREEMENT_ROTATION_DEG = 5.0
DISAGREEMENT_TRANSLATION_MM = 5.0


def optimizer_configuration(key, budget=300):
    if key not in SPECS or isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("Expected O0/O1/O2 and a positive integer iteration limit")
    spec = deepcopy(SPECS[key])
    spec["kwargs"]["numberOfIterations"] = budget
    spec["scales"] = SCALE_POLICY
    return spec


def configure_optimizer(method, key, budget=300):
    spec = optimizer_configuration(key, budget)
    kwargs = spec["kwargs"].copy()
    kwargs["estimateLearningRate"] = getattr(method, kwargs["estimateLearningRate"])
    getattr(method, "SetOptimizerAs" + spec["family"])(**kwargs)
    method.SetOptimizerScalesFromPhysicalShift(5, 0.01)
    return spec


def budget_hit(description):
    return "maximum number of iterations" in description.lower()


def stop_category(description):
    text = description.lower()
    if budget_hit(description):
        return "iteration_budget"
    if "step too small" in text:
        return "step_too_small"
    if "convergence checker passed" in text:
        return "window_convergence"
    if "gradient magnitude tolerance" in text:
        return "gradient_tolerance"
    return "other"


def trajectory_summary(rows):
    if not rows:
        return {}
    values = np.array([r["sampled_metric"] for r in rows])
    best = int(np.argmin(values))
    return {
        "first_metric": float(values[0]),
        "best_metric": float(values[best]),
        "final_metric": float(values[-1]),
        "best_position": {k: rows[best][k] for k in ("level", "iteration")},
        "worsening_steps_within_levels": sum(
            b["sampled_metric"] > a["sampled_metric"]
            for a, b in zip(rows, rows[1:])
            if a["level"] == b["level"]
        ),
        "note": "Sampled objectives differ between pyramid levels; use full MI deltas",
    }


def stage(fixed, moving, transform, p, key, translation_only):
    started = time.perf_counter()
    fixed, moving = fresh_image(fixed), fresh_image(moving)
    incoming = transform_record(transform)
    final = sitk.Euler3DTransform(transform)
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(1)
    method.SetMetricAsMattesMutualInformation(p.histogram_bins)
    method.SetMetricSamplingStrategy(getattr(method, p.sampling_strategy))
    if p.sampling_strategy != "NONE":
        method.SetMetricSamplingPercentage(p.sampling_fraction, p.sampling_seed)
    method.SetInterpolator(sitk.sitkLinear)
    configuration = configure_optimizer(method, key, p.iterations_per_level)
    weights = [0, 0, 0, 1, 1, 1] if translation_only else [1] * 6
    method.SetOptimizerWeights(weights)
    factors = (
        p.shrink_factors[:-1]
        if translation_only and len(p.shrink_factors) > 1
        else p.shrink_factors
    )
    sigmas = p.smoothing_sigmas_mm[: len(factors)]
    method.SetShrinkFactorsPerLevel(factors)
    method.SetSmoothingSigmasPerLevel(sigmas)
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(final, inPlace=True)
    rows, stops, scales = [], [], {}

    def iteration():
        level = int(method.GetCurrentLevel())
        values = list(method.GetOptimizerScales())
        scales[str(level)] = values
        value = float(method.GetMetricValue())
        if not np.isfinite(value):
            raise ValueError("Nonfinite sampled optimizer metric")
        rows.append(
            {
                "level": level,
                "iteration": int(method.GetOptimizerIteration()),
                "sampled_metric": value,
                "learning_rate": float(method.GetOptimizerLearningRate()),
                "valid_points": int(method.GetMetricNumberOfValidPoints()),
            }
        )

    def level():
        if method.GetCurrentLevel() > 0:
            stops.append(method.GetOptimizerStopConditionDescription())

    method.AddCommand(sitk.sitkIterationEvent, iteration)
    method.AddCommand(sitk.sitkMultiResolutionIterationEvent, level)
    execute_start = time.perf_counter()
    try:
        method.Execute(fixed, moving)
    finally:
        method.RemoveAllCommands()
    execute_seconds = time.perf_counter() - execute_start
    stops.append(method.GetOptimizerStopConditionDescription())
    info = transform_record(final, p.rigid_tolerance)
    if translation_only and info["parameters"][:3] != incoming["parameters"][:3]:
        raise ValueError("Translation-only optimization changed rotation")
    summaries = [
        {
            **trajectory_summary([r for r in rows if r["level"] == i]),
            "level": i,
            "stop": stop,
            "budget_hit": budget_hit(stop),
            "iterations": sum(r["level"] == i for r in rows),
            "physical_shift_scales": scales.get(str(i), []),
        }
        for i, stop in enumerate(stops)
    ]
    return (
        final,
        {
            "iterations": len(rows),
            "stop": stops[-1],
            "level_stops": stops,
            "budget_hit": any(map(budget_hit, stops)),
            "final_level_budget_hit": budget_hit(stops[-1]),
            "runtime_seconds": time.perf_counter() - started,
            "optimizer_execute_seconds": execute_seconds,
            "configuration": configuration,
            "weights": weights,
            "shrink_factors": list(factors),
            "smoothing_sigmas_mm": list(sigmas),
            "input_transform": incoming,
            "output_transform": info,
            "trajectory_summary": trajectory_summary(rows),
            "level_summaries": summaries,
            "final_optimizer_metric": float(method.GetMetricValue()),
            "final_learning_rate": rows[-1]["learning_rate"] if rows else None,
            "final_physical_shift_scales": scales.get(str(len(stops) - 1), []),
        },
        rows,
    )


def register(fixed, moving, coverage_mask, p, key):
    if p.initialization != "GEOMETRY" or p.use_fixed_mask:
        raise ValueError("Optimizer study requires GEOMETRY and no metric masks")
    p.validate()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    started = time.perf_counter()
    initial = initialize_mode(fixed, moving, "GEOMETRY")
    initial_info = transform_record(initial)

    def evaluate(tx):
        return dict(
            full_metric(fixed, moving, tx, None, p),
            **brain_coverage(fixed, moving, coverage_mask, tx),
        )

    evaluations = [evaluate(initial)]
    translated, translation, trace1 = stage(fixed, moving, initial, p, key, True)
    evaluations.append(evaluate(translated))
    final, rigid, trace2 = stage(fixed, moving, translated, p, key, False)
    evaluations.append(evaluate(final))
    if translation["output_transform"] != rigid["input_transform"]:
        raise ValueError("Two-stage transform continuity violated")
    a, b, c = [e["MI"] for e in evaluations]
    return (
        final,
        {
            "initial_transform": initial_info,
            "final_transform": transform_record(final),
            "translation_stage": translation,
            "rigid_stage": rigid,
            "evaluations": evaluations,
            "MI_initial": a,
            "MI_after_translation": b,
            "MI_final": c,
            "delta_translation": a - b,
            "delta_rigid": b - c,
            "delta_total": a - c,
            "runtime_seconds": time.perf_counter() - started,
            "optimizer_execute_seconds": translation["optimizer_execute_seconds"]
            + rigid["optimizer_execute_seconds"],
        },
        [
            dict(r, stage=name)
            for name, trace in [("translation", trace1), ("rigid", trace2)]
            for r in trace
        ],
    )


def disagreement(a, b):
    """Relative rigid rotation and pull displacement at the common fixed center."""
    ra = np.asarray(a["matrix_lps"])[:3, :3]
    rb = np.asarray(b["matrix_lps"])[:3, :3]
    relative = ra.T @ rb
    sine = (
        np.linalg.norm(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
        / 2
    )
    angle = float(np.degrees(np.arctan2(sine, (np.trace(relative) - 1) / 2)))
    center = np.array(a["center_mm_lps"])
    if not np.allclose(center, b["center_mm_lps"], rtol=0, atol=1e-10):
        raise ValueError("Euler translation differences require a common center")
    delta = np.array(b["translation_mm_lps_xyz"]) - a["translation_mm_lps_xyz"]
    distance = float(np.linalg.norm(delta))
    return {
        "rotation_geodesic_deg": angle,
        "delta_euler_deg_xyz": (
            np.array(b["rotation_deg_lps_xyz"]) - a["rotation_deg_lps_xyz"]
        ).tolist(),
        "delta_translation_mm_xyz": delta.tolist(),
        "translation_norm_mm": distance,
        "common_center_mm_lps": center.tolist(),
        "provisional_review_warning": angle > DISAGREEMENT_ROTATION_DEG
        or distance > DISAGREEMENT_TRANSLATION_MM,
        "convention": "fixed LPS -> moving LPS pull; disagreement is not anatomical error",
    }


def quantiles(values):
    if not values:
        return {}
    a = np.asarray(values)
    return {
        **dict(zip(("p10", "median", "p90"), np.percentile(a, [10, 50, 90], axis=0).tolist())),
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
    }


def aggregate(records):
    good = [r for r in records if r["success"]]
    stops = [
        x
        for r in good
        for stage in ("translation_stage", "rigid_stage")
        for x in r[stage]["level_stops"]
    ]
    return {
        "technical_success_count": len(good),
        "execution_failure_count": len(records) - len(good),
        "MI_improved_count": sum(r["delta_total"] > 0 for r in good),
        "MI_degraded_count": sum(r["delta_total"] < 0 for r in good),
        "MI_unchanged_count": sum(r["delta_total"] == 0 for r in good),
        "translation_budget_hit_count": sum(r["translation_stage"]["budget_hit"] for r in good),
        "rigid_budget_hit_count": sum(r["rigid_stage"]["budget_hit"] for r in good),
        "final_rigid_level_budget_hit_count": sum(
            r["rigid_stage"]["final_level_budget_hit"] for r in good
        ),
        "stop_conditions": dict(Counter(stops)),
        "stop_categories": dict(Counter(map(stop_category, stops))),
        "delta_total": quantiles([r["delta_total"] for r in good]),
        "rotation_deg_lps_xyz": quantiles(
            [r["final_transform"]["rotation_deg_lps_xyz"] for r in good]
        ),
        "translation_mm_lps_xyz": quantiles(
            [r["final_transform"]["translation_mm_lps_xyz"] for r in good]
        ),
        "template_brain_in_bounds_fraction": {
            label: quantiles(
                [r["evaluations"][i]["template_brain_in_bounds_fraction"] for r in good]
            )
            for label, i in [("initial", 0), ("after_translation", 1), ("final", 2)]
        },
        "runtime_seconds": quantiles([r["runtime_seconds"] for r in good]),
        "optimizer_execute_seconds": quantiles([r["optimizer_execute_seconds"] for r in good]),
        "stage_runtime_seconds": {
            stage: quantiles([r[stage]["runtime_seconds"] for r in good])
            for stage in ("translation_stage", "rigid_stage")
        },
        "QC_concern_subject_count": sum(bool(r["warnings"]) for r in records),
        "warning_counts": dict(Counter(w for r in records for w in r["warnings"])),
    }
