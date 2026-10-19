"""SimpleITK Euler3D registration in LPS physical coordinates, with one final interpolation."""

from dataclasses import asdict
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .settings import RigidParameters

LPS_FROM_RAS = np.diag([-1.0, -1.0, 1.0])


def to_sitk(image: nib.spatialimages.SpatialImage) -> sitk.Image:
    """Preserve voxel-to-physical mapping, including tiny source header non-orthogonality."""
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    if image.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("Registration requires finite 3D intensities")
    spatial = LPS_FROM_RAS @ image.affine[:3, :3]
    spacing = nib.affines.voxel_sizes(image.affine)
    result = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
    result.SetSpacing(tuple(float(v) for v in spacing))
    result.SetOrigin(tuple(LPS_FROM_RAS @ image.affine[:3, 3]))
    result.SetDirection(tuple((spatial / spacing).ravel()))
    return result


def from_sitk(image: sitk.Image, reference: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    """Build a float32 derived NIfTI on the exact template affine/grid."""
    data = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
    if tuple(data.shape) != reference.shape:
        raise ValueError("Unexpected registered output shape")
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    result = nib.Nifti1Image(data.astype(np.float32), reference.affine, header)
    result.set_sform(reference.affine, code=4)
    result.set_qform(reference.affine, code=4)
    result.header["cal_min"] = float(data.min())
    result.header["cal_max"] = float(data.max())
    return result


def transform_record(transform: sitk.Euler3DTransform, tolerance: float = 1e-6) -> dict[str, Any]:
    """Assert six parameters, orthonormal rotation, positive unit determinant and no scale/shear."""
    if transform.GetName() != "Euler3DTransform" or transform.GetNumberOfParameters() != 6:
        raise ValueError("Only a six-parameter Euler3D rigid transform is allowed")
    matrix = np.asarray(transform.GetMatrix()).reshape(3, 3)
    parameters = np.asarray(transform.GetParameters())
    orthogonality = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    determinant = float(np.linalg.det(matrix))
    singular = np.linalg.svd(matrix, compute_uv=False)
    if (
        not np.isfinite(parameters).all()
        or not np.isfinite(matrix).all()
        or not np.isfinite(transform.GetFixedParameters()).all()
        or orthogonality > tolerance
        or abs(determinant - 1) > tolerance
        or not np.allclose(singular, 1, rtol=0, atol=tolerance)
    ):
        raise ValueError("Non-finite or non-rigid registration transform")
    center = np.asarray(transform.GetCenter())
    translation = np.asarray(transform.GetTranslation())
    homogeneous = np.eye(4)
    homogeneous[:3, :3] = matrix
    homogeneous[:3, 3] = center + translation - matrix @ center
    return {
        "parameters": parameters.tolist(),
        "fixed_parameters": list(transform.GetFixedParameters()),
        "rotation_deg_lps_xyz": np.degrees(parameters[:3]).tolist(),
        "translation_mm_lps_xyz": translation.tolist(),
        "center_mm_lps": center.tolist(),
        "matrix_lps": homogeneous.tolist(),
        "rotation_determinant": determinant,
        "orthonormality_error": orthogonality,
        "rotation_singular_values": singular.tolist(),
        "compute_zyx": bool(transform.GetComputeZYX()),
        "rigid_tolerance": tolerance,
        "direction": "fixed/template LPS -> moving/source LPS (resampling pull transform)",
    }


def initialize(fixed: sitk.Image, moving: sitk.Image) -> sitk.Euler3DTransform:
    """Align physical FOV centers as an optimization start, not anatomical correspondence."""
    euler = sitk.Euler3DTransform()
    euler.SetComputeZYX(True)
    return sitk.Euler3DTransform(
        sitk.CenteredTransformInitializer(
            fixed, moving, euler, sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    )


def metric_evaluation(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform: sitk.Euler3DTransform,
    mask: sitk.Image | None,
    parameters: RigidParameters,
) -> float:
    """Comparable negative Mattes MI on a fixed deterministic evaluation lattice, before/after."""
    factor = parameters.evaluation_shrink_factor
    evaluation_fixed = sitk.Shrink(fixed, [factor] * 3) if factor > 1 else fixed
    evaluation_mask = sitk.Shrink(mask, [factor] * 3) if mask is not None and factor > 1 else mask
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(1)
    method.SetMetricAsMattesMutualInformation(parameters.histogram_bins)
    method.SetMetricSamplingStrategy(method.NONE)
    if evaluation_mask is not None:
        method.SetMetricFixedMask(evaluation_mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetInitialTransform(transform)
    value = float(method.MetricEvaluate(evaluation_fixed, moving))
    if not np.isfinite(value):
        raise ValueError("Non-finite registration evaluation metric")
    return value


def register_rigid(
    fixed: sitk.Image,
    moving: sitk.Image,
    parameters: RigidParameters,
    mask: sitk.Image | None = None,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[sitk.Euler3DTransform, dict[str, Any]]:
    """Execute one configured multiresolution attempt; never retry with different initialization."""
    parameters.validate()
    if parameters.use_fixed_mask and mask is None:
        raise ValueError("Configured fixed-image brain mask is required")
    trace = {} if diagnostics is None else diagnostics
    # All ITK filters and metric reductions are single threaded for fixed-seed reproducibility.
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    initial = initialize(fixed, moving)
    initial_info = transform_record(initial, parameters.rigid_tolerance)
    trace.update(initial_transform=initial_info, parameters=asdict(parameters))
    before = metric_evaluation(fixed, moving, initial, mask, parameters)
    trace["metric_before"] = before
    final = sitk.Euler3DTransform(initial)
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(1)
    method.SetMetricAsMattesMutualInformation(parameters.histogram_bins)
    method.SetMetricSamplingStrategy(getattr(method, parameters.sampling_strategy))
    if parameters.sampling_strategy != "NONE":
        method.SetMetricSamplingPercentage(parameters.sampling_fraction, parameters.sampling_seed)
    if mask is not None:
        method.SetMetricFixedMask(mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=parameters.learning_rate,
        minStep=parameters.min_step,
        numberOfIterations=parameters.iterations_per_level,
        relaxationFactor=parameters.relaxation_factor,
        gradientMagnitudeTolerance=parameters.gradient_tolerance,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(parameters.shrink_factors)
    method.SetSmoothingSigmasPerLevel(parameters.smoothing_sigmas_mm)
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(final, inPlace=True)
    counts = [0] * len(parameters.shrink_factors)

    def iteration():
        counts[method.GetCurrentLevel()] += 1

    method.AddCommand(sitk.sitkIterationEvent, iteration)
    try:
        method.Execute(fixed, moving)
    except RuntimeError:
        trace["optimizer_stop_condition"] = method.GetOptimizerStopConditionDescription()
        trace["iterations_per_level_actual"] = counts
        trace["iteration_count"] = sum(counts)
        raise
    final_info = transform_record(final, parameters.rigid_tolerance)
    trace.update(
        final_transform=final_info,
        optimizer_stop_condition=method.GetOptimizerStopConditionDescription(),
    )
    after = metric_evaluation(fixed, moving, final, mask, parameters)
    optimizer_value = float(method.GetMetricValue())
    if not np.isfinite(optimizer_value):
        raise ValueError("Non-finite optimizer metric")
    trace.update(
        {
            "initial_transform": initial_info,
            "final_transform": final_info,
            "metric_before": before,
            "metric_after": after,
            "metric_improvement": before - after,
            "metric_definition": "negative Mattes MI; lower better; same evaluation lattice/mask",
            "metric_evaluation_voxel_count": int(
                np.prod(sitk.Shrink(fixed, [parameters.evaluation_shrink_factor] * 3).GetSize())
            ),
            "optimizer_metric_final": optimizer_value,
            "optimizer_stop_condition": method.GetOptimizerStopConditionDescription(),
            "iteration_count": sum(counts),
            "iterations_per_level_actual": counts,
            "optimizer_final_iteration_index": int(method.GetOptimizerIteration()),
            "parameters": asdict(parameters),
            "transform_type": "Euler3DTransform",
            "optimizer_scales_policy": "SetOptimizerScalesFromPhysicalShift",
        }
    )
    return final, trace


def resample_to_fixed(
    moving: sitk.Image, fixed: sitk.Image, transform: sitk.Euler3DTransform
) -> sitk.Image:
    """One direct interpolation from the native canonical source onto the template grid."""
    return sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def source_foreground_clipping(
    data: np.ndarray,
    source_affine: np.ndarray,
    template_shape: tuple[int, ...],
    template_affine: np.ndarray,
    transform: sitk.Euler3DTransform,
    threshold: float = 0,
) -> dict[str, Any]:
    """Count native non-background voxel centers outside template voxel-cell FOV, in planes.

    This measures head/neck/noise FOV exclusion, not brain/anatomical loss. It does
    not derive a moving brain mask or modify source voxel intensities.
    """
    info = transform_record(transform)
    conversion = np.eye(4)
    conversion[:3, :3] = LPS_FROM_RAS
    fixed_to_moving_ras = conversion @ np.array(info["matrix_lps"]) @ conversion
    mapping = np.linalg.inv(template_affine) @ np.linalg.inv(fixed_to_moving_ras) @ source_affine
    total = outside = 0
    for z in range(data.shape[2]):
        x, y = np.nonzero((data[:, :, z] > threshold) | (data[:, :, z] < -threshold))
        if not len(x):
            continue
        xyz = mapping[:3, :3] @ np.vstack([x, y, np.full(len(x), z)]) + mapping[:3, 3, None]
        inside = ((xyz >= -0.5) & (xyz < np.asarray(template_shape)[:, None] - 0.5)).all(axis=0)
        total += len(x)
        outside += int((~inside).sum())
    if not total:
        raise ValueError("Empty source non-background foreground")
    return {
        "source_nonbackground_voxel_count": total,
        "outside_template_fov_count": outside,
        "outside_template_fov_fraction": outside / total,
        "definition": "native non-background centers outside template FOV; not brain loss",
    }
