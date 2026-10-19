"""One rigid-initialized affine stage, with explicit pull-space deformation diagnostics."""

import time

import numpy as np
import SimpleITK as sitk

from .optimizers import budget_hit, configure_optimizer, trajectory_summary
from .tuning import check_mask, fresh_image

# Engineering review triggers, declared before v6 outcomes; not biological thresholds.
REVIEW_LIMITS = {
    "principal_scale_min": 0.8,
    "principal_scale_max": 1.25,
    "pull_volume_factor_min": 0.7,
    "pull_volume_factor_max": 1.4,
    "max_absolute_column_cosine": 0.15,
}


def rigid_to_affine(rigid):
    """Preserve the full centered fixed-LPS -> moving-LPS mapping exactly."""
    rigid = sitk.Euler3DTransform(rigid)
    affine = sitk.AffineTransform(3)
    affine.SetCenter(rigid.GetCenter())
    affine.SetMatrix(rigid.GetMatrix())
    affine.SetTranslation(rigid.GetTranslation())
    return affine


def affine_record(transform):
    """Polar/SVD diagnostics; scales describe a pull map, never biological change."""
    tx = sitk.AffineTransform(transform)
    a = np.asarray(tx.GetMatrix()).reshape(3, 3)
    center = np.asarray(tx.GetCenter())
    translation = np.asarray(tx.GetTranslation())
    if not all(np.isfinite(x).all() for x in (a, center, translation)):
        raise ValueError("Nonfinite affine transform")
    u, singular, vt = np.linalg.svd(a)
    det = float(np.linalg.det(a))
    if singular[-1] < 1e-8 or det <= 0:
        raise ValueError("Affine must be nonsingular and orientation preserving")
    rotation = u @ vt
    stretch = vt.T @ np.diag(singular) @ vt
    norms = np.linalg.norm(a, axis=0)
    gram = (a / norms).T @ (a / norms)
    offdiag = gram - np.eye(3)
    shear = stretch / np.sqrt(np.outer(np.diag(stretch), np.diag(stretch))) - np.eye(3)
    max_cosine = float(np.max(np.abs(offdiag)))
    warnings = []
    if (
        singular.min() < REVIEW_LIMITS["principal_scale_min"]
        or singular.max() > REVIEW_LIMITS["principal_scale_max"]
    ):
        warnings.append("unusual_principal_scale")
    if (
        not REVIEW_LIMITS["pull_volume_factor_min"]
        <= det
        <= REVIEW_LIMITS["pull_volume_factor_max"]
    ):
        warnings.append("unusual_volume_factor")
    if max_cosine > REVIEW_LIMITS["max_absolute_column_cosine"]:
        warnings.append("unusual_nonorthogonality")
    matrix = np.eye(4)
    matrix[:3, :3] = a
    matrix[:3, 3] = center + translation - a @ center
    # A proper polar rotation is meaningful; Euler angles of the affine matrix are not.
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    return {
        "parameters": list(tx.GetParameters()),
        "fixed_parameters": list(tx.GetFixedParameters()),
        "center_mm_lps": center.tolist(),
        "translation_mm_lps_xyz": translation.tolist(),
        "matrix_lps": matrix.tolist(),
        "linear_matrix_lps": a.tolist(),
        "determinant": det,
        "singular_values": singular.tolist(),
        "principal_scale_deviation_from_unit": (singular - 1).tolist(),
        "max_absolute_principal_scale_deviation": float(np.max(np.abs(singular - 1))),
        "global_isotropic_pull_scale": det ** (1 / 3),
        "pull_volume_change_factor": det,
        "inverse_moving_to_fixed_volume_factor": 1 / det,
        "axis_column_norms": norms.tolist(),
        "column_cosine_matrix": gram.tolist(),
        "max_absolute_column_cosine": max_cosine,
        "polar_stretch_matrix": stretch.tolist(),
        "normalized_polar_shear_matrix": shear.tolist(),
        "max_absolute_normalized_polar_shear": float(np.max(np.abs(shear))),
        "polar_rotation_matrix": rotation.tolist(),
        "polar_rotation_angle_deg": float(angle),
        "condition_number": float(singular[0] / singular[-1]),
        "valid": True,
        "review_warnings": warnings,
        "convention": "fixed-template LPS -> native-moving LPS pull; no biological interpretation",
    }


def affine_coverage(fixed, moving, mask, transform):
    """Same voxel-cell coverage definition as rigid; support general affine maps."""
    check_mask(fixed, mask)
    tx = np.asarray(affine_record(transform)["matrix_lps"])
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


def optimize_affine(fixed, moving, rigid, p):
    """Exactly one 12-parameter Execute, with the unchanged O0 optimizer/pyramid."""
    p.validate()
    if p.use_fixed_mask or p.sampling_fraction != 0.02 or p.sampling_seed != 42:
        raise ValueError("Require unmasked seeded 2% baseline configuration")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    start = time.perf_counter()
    fixed, moving = fresh_image(fixed), fresh_image(moving)
    final = rigid_to_affine(rigid)
    incoming = affine_record(final)
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(1)
    method.SetMetricAsMattesMutualInformation(p.histogram_bins)
    method.SetMetricSamplingStrategy(getattr(method, p.sampling_strategy))
    method.SetMetricSamplingPercentage(p.sampling_fraction, p.sampling_seed)
    method.SetInterpolator(sitk.sitkLinear)
    configuration = configure_optimizer(method, "O0", p.iterations_per_level)
    method.SetOptimizerWeights([1] * 12)
    method.SetShrinkFactorsPerLevel(p.shrink_factors)
    method.SetSmoothingSigmasPerLevel(p.smoothing_sigmas_mm)
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(final, inPlace=True)
    rows, stops, scales = [], [], {}

    def iteration():
        level = int(method.GetCurrentLevel())
        values = list(method.GetOptimizerScales())
        if len(values) != 12 or not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Invalid affine physical-shift scales")
        scales[str(level)] = values
        value = float(method.GetMetricValue())
        if not np.isfinite(value):
            raise ValueError("Nonfinite sampled affine MI")
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
    info = {
        "configuration": configuration,
        "weights": [1] * 12,
        "shrink_factors": list(p.shrink_factors),
        "smoothing_sigmas_mm": list(p.smoothing_sigmas_mm),
        "input_transform": incoming,
        "output_transform": affine_record(final),
        "iterations": len(rows),
        "stop": stops[-1],
        "level_stops": stops,
        "budget_hit": any(map(budget_hit, stops)),
        "final_level_budget_hit": budget_hit(stops[-1]),
        "level_summaries": [
            {
                "level": i,
                "stop": stop,
                "iterations": sum(r["level"] == i for r in rows),
                "physical_shift_scales": scales.get(str(i), []),
            }
            for i, stop in enumerate(stops)
        ],
        "trajectory_summary": trajectory_summary(rows),
        "runtime_seconds": time.perf_counter() - start,
        "optimizer_execute_seconds": execute_seconds,
    }
    return final, info, rows
