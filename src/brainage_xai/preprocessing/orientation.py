"""Affine-aware RAS+ orientation and header-only spatial inspection."""

from typing import Any

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform


def inspect_orientation(image: nib.spatialimages.SpatialImage) -> dict[str, Any]:
    """Inspect a 3D header without reading voxel data; reject untrustworthy geometry."""
    if image.ndim != 3 or any(size <= 0 for size in image.shape):
        raise ValueError("Preprocessing requires a non-empty 3D MRI")
    affine = image.affine
    if (
        affine is None
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], [0, 0, 0, 1])
        or np.linalg.matrix_rank(affine[:3, :3]) != 3
    ):
        raise ValueError("Invalid MRI affine")
    spacing = np.asarray(image.header.get_zooms(), dtype=float)
    if not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Invalid MRI voxel spacing")
    if not int(image.header["qform_code"]) and not int(image.header["sform_code"]):
        raise ValueError("No coded orientation; preprocessing cannot trust fallback geometry")
    units = image.header.get_xyzt_units()[0]
    if units not in {"mm", "unknown"}:
        raise ValueError(
            "Non-millimeter spatial units need explicit conversion before preprocessing"
        )
    transform = ornt_transform(io_orientation(affine), axcodes2ornt(("R", "A", "S")))
    shape = tuple(int(image.shape[i]) for i in np.argsort(transform[:, 0]))
    canonical_affine = affine @ inv_ornt_aff(transform, image.shape)
    return {
        "source_shape": list(image.shape),
        "source_spacing": [float(v) for v in spacing],
        "source_orientation": list(nib.aff2axcodes(affine)),
        "source_affine": affine.tolist(),
        "source_spatial_units": units,
        "canonical_shape": list(shape),
        "canonical_orientation": list(nib.aff2axcodes(canonical_affine)),
        "canonical_affine": canonical_affine.tolist(),
    }


def canonicalize(image: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    """Reorder/flip data with the corresponding affine update; no registration."""
    inspect_orientation(image)
    return nib.as_closest_canonical(image)
