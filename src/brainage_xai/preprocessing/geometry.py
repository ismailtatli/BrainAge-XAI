"""Quantitative affine geometry, non-background extents, and diagnostic crop coverage."""

from typing import Any

import nibabel as nib
import numpy as np

from .intensity import foreground_mask

# QC/numerical descriptors, never biological thresholds.
SHEAR_COSINE_TOLERANCE = 1e-5
AXIS_ALIGNMENT_TOLERANCE_DEG = 0.1
OBLIQUITY_THRESHOLDS_DEG = (0.1, 0.5, 1, 3, 5, 10)
PERCENTILES = {"min": 0, "p50": 50, "p90": 90, "p95": 95, "p99": 99, "max": 100}


def affine_geometry(affine: np.ndarray) -> dict[str, Any]:
    """Separate angular obliquity from normalized-column non-orthogonality.

    nibabel obliquity returns radians indexed by world axes; reports convert to degrees.
    Source-voxel-axis angles are also provided explicitly to avoid a permutation ambiguity.
    Shear metric is max absolute off-diagonal cosine of normalized affine columns.
    """
    affine = np.asarray(affine, dtype=float)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], [0, 0, 0, 1])
        or np.linalg.matrix_rank(affine[:3, :3]) != 3
    ):
        raise ValueError("Geometry diagnostics require a finite, nonsingular homogeneous affine")
    spatial = affine[:3, :3]
    spacing = nib.affines.voxel_sizes(affine)
    directions = spatial / spacing
    gram = directions.T @ directions
    off_diagonal = gram - np.diag(np.diag(gram))
    shear_metric = float(np.max(np.abs(off_diagonal)))
    with np.errstate(invalid="ignore"):
        # nibabel normalizes directions internally; exact/near axis-alignment may round above 1.
        angles = np.degrees(nib.affines.obliquity(affine))
    if not np.isfinite(angles).all():
        angles = np.degrees(np.arccos(np.clip(np.abs(directions).max(axis=1), 0, 1)))
    voxel_angles = np.degrees(np.arccos(np.clip(np.abs(directions).max(axis=0), 0, 1)))
    maximum = float(angles.max())
    shear = shear_metric > SHEAR_COSINE_TOLERANCE
    return {
        "obliquity_deg": angles.tolist(),
        "obliquity_axis_order": "world_RAS_nibabel",
        "source_voxel_axis_obliquity_deg": voxel_angles.tolist(),
        "max_obliquity_deg": maximum,
        "has_measurable_shear": bool(shear),
        "shear_metric_max_abs_axis_cosine": shear_metric,
        "shear_cosine_tolerance": SHEAR_COSINE_TOLERANCE,
        "axis_alignment_tolerance_deg": AXIS_ALIGNMENT_TOLERANCE_DEG,
        "is_oblique": maximum > AXIS_ALIGNMENT_TOLERANCE_DEG,
        "effectively_axis_aligned": not shear and maximum <= AXIS_ALIGNMENT_TOLERANCE_DEG,
        "affine_determinant_sign": int(np.sign(np.linalg.det(spatial))),
        "affine_voxel_sizes_mm": spacing.tolist(),
    }


def foreground_extent(
    data: np.ndarray, spacing_mm: np.ndarray, *, background_threshold: float = 0.0
) -> dict[str, Any]:
    """Bounds are inclusive start/exclusive end; physical size covers whole voxel cells.

    Coordinates/offsets follow resampled RAS voxel axes. Bbox center, not center of
    mass or anatomical center, is compared with (full_shape - 1) / 2.
    """
    data = np.asarray(data)
    spacing = np.asarray(spacing_mm, dtype=float)
    if data.ndim != 3 or spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Foreground extent requires 3D data and positive finite voxel sizes")
    mask = foreground_mask(data, background_threshold=background_threshold)
    starts, ends = [], []
    for axis in range(3):
        occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        starts.append(int(occupied[0]))
        ends.append(int(occupied[-1]) + 1)
    start, end = np.array(starts), np.array(ends)
    size = end - start
    center = (start + end - 1) / 2
    offset = center - (np.asarray(data.shape) - 1) / 2
    count = int(np.count_nonzero(mask))
    return {
        "resampled_shape": list(data.shape),
        "foreground_bbox_start": starts,
        "foreground_bbox_end_exclusive": ends,
        "foreground_bbox_size": size.tolist(),
        "foreground_voxel_count": count,
        "foreground_fraction": count / data.size,
        "foreground_extent_mm": (size * spacing).tolist(),
        "foreground_center_voxel": center.tolist(),
        "foreground_center_offset_voxel": offset.tolist(),
        "foreground_center_offset_mm": (offset * spacing).tolist(),
        "foreground_rule": "abs(intensity) > background_threshold; non-background, not brain mask",
        "background_threshold": float(background_threshold),
    }


def describe(values: np.ndarray) -> dict[str, Any]:
    """Scalar or per-axis sample summaries; numpy's linear percentile convention."""
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {**dict.fromkeys(PERCENTILES), "median": None, "mean": None}
    if not np.isfinite(values).all():
        raise ValueError("Cannot summarize non-finite diagnostics")

    def native(value: np.ndarray) -> Any:
        return value.tolist()

    result = {name: native(np.percentile(values, q, axis=0)) for name, q in PERCENTILES.items()}
    result["median"] = result["p50"]
    result["mean"] = native(values.mean(axis=0))
    return result


def candidate_tensor_sizes(records: list[dict]) -> list[dict[str, Any]]:
    """Diagnostic sizes with guaranteed empirical *joint bbox-size* coverage, not alignment.

    Grow a shared rank in per-axis sorted sizes until rounded dimensions contain the
    requested fraction jointly. This avoids claiming 95% joint coverage from three
    independent 95th percentiles. No image is cropped or padded.
    """
    if not records:
        return []
    sizes = np.asarray([r["foreground_bbox_size"] for r in records], dtype=int)
    full = np.asarray([r["resampled_shape"] for r in records], dtype=int)
    starts = np.asarray([r["foreground_bbox_start"] for r in records], dtype=int)
    ends = np.asarray([r["foreground_bbox_end_exclusive"] for r in records], dtype=int)
    if (
        sizes.shape != (len(records), 3)
        or np.any(sizes <= 0)
        or np.any(full <= 0)
        or np.any(starts < 0)
        or np.any(ends > full)
        or not np.array_equal(ends - starts, sizes)
    ):
        raise ValueError("Invalid foreground bounding-box records")
    ordered = np.sort(sizes, axis=0)
    candidates = []
    n = len(sizes)
    for target in (0.95, 0.99, 1.0):
        for multiple in (8, 16):
            required = int(np.ceil(target * n))
            for rank in range(int(np.ceil(target * (n - 1))), n):
                shape = ((ordered[rank] + multiple - 1) // multiple) * multiple
                too_large = sizes > shape
                fitting = ~too_large.any(axis=1)
                if int(fitting.sum()) >= required:
                    break
            # Standard deterministic volume-centered crop/pad, with lower-index tie breaking.
            crop_start = (full - shape) // 2
            crop_end = crop_start + shape
            center_loss = (starts < crop_start) | (ends > crop_end)
            candidates.append(
                {
                    "target_joint_bbox_coverage": target,
                    "rounding_multiple": multiple,
                    "diagnostic_shape": shape.tolist(),
                    "bbox_joint_coverage_count": int(fitting.sum()),
                    "bbox_joint_coverage_fraction": float(fitting.mean()),
                    "bbox_larger_than_candidate_axis_counts": too_large.sum(axis=0).tolist(),
                    "bbox_larger_than_candidate_any_count": int(too_large.any(axis=1).sum()),
                    "center_crop_foreground_loss_axis_counts": center_loss.sum(axis=0).tolist(),
                    "center_crop_foreground_loss_any_count": int(center_loss.any(axis=1).sum()),
                    "center_crop_joint_coverage_count": int((~center_loss.any(axis=1)).sum()),
                    "candidate_is_final": False,
                }
            )
    return candidates


def affine_summary(records: list[dict]) -> dict[str, Any]:
    """Aggregate header-only affine diagnostics without inspecting foreground arrays."""
    geometry = [r for r in records if r.get("max_obliquity_deg") is not None]
    angles = np.asarray([r["max_obliquity_deg"] for r in geometry])
    bins = [0, 0.1, 0.5, 1, 3, 5, 10, 15, 20, 30, 45, 90]
    counts, _ = np.histogram(angles, bins=bins)
    summary = {
        "geometry_subject_count": len(geometry),
        "max_obliquity_deg_summary": describe(angles),
        "obliquity_deg_per_axis_summary": describe(
            np.asarray([r["obliquity_deg"] for r in geometry])
        ),
        "max_obliquity_deg_distribution": {
            "bin_edges_deg": bins,
            "counts": counts.tolist(),
            "intervals": "[lower, upper), final upper inclusive",
        },
        "obliquity_threshold_counts": {
            str(q): int((angles > q).sum()) for q in OBLIQUITY_THRESHOLDS_DEG
        },
        "sheared_affine_count": sum(r["has_measurable_shear"] for r in geometry),
        "non_sheared_oblique_count": sum(
            not r["has_measurable_shear"] and r["is_oblique"] for r in geometry
        ),
        "effectively_axis_aligned_count": sum(r["effectively_axis_aligned"] for r in geometry),
        "shear_metric_summary": describe(
            np.asarray([r["shear_metric_max_abs_axis_cosine"] for r in geometry])
        ),
        "shear_cosine_tolerance": SHEAR_COSINE_TOLERANCE,
        "axis_alignment_tolerance_deg": AXIS_ALIGNMENT_TOLERANCE_DEG,
    }
    return summary


def geometry_summary(records: list[dict]) -> dict[str, Any]:
    """Summarize available affine and foreground observations; denominators are explicit."""
    foreground = [r for r in records if r.get("foreground_bbox_size") is not None]
    spatial = [r for r in records if r.get("resampled_shape") is not None]
    summary = {
        **affine_summary(records),
        "subject_count": len(records),
        "foreground_subject_count": len(foreground),
        "resampled_shape_subject_count": len(spatial),
        "success_count": sum(r.get("diagnostic_status") == "success" for r in records),
        "failure_count": sum(r.get("diagnostic_status") == "failed" for r in records),
        "diagnostic_tensor_candidates": candidate_tensor_sizes(foreground),
    }
    for field in (
        "resampled_shape",
        "foreground_bbox_size",
        "foreground_fraction",
        "foreground_extent_mm",
        "foreground_center_voxel",
        "foreground_center_offset_voxel",
        "foreground_center_offset_mm",
    ):
        population = spatial if field == "resampled_shape" else foreground
        summary[f"{field}_summary"] = describe(np.asarray([r[field] for r in population]))
    return summary
