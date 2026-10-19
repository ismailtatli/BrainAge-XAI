"""Continuous T1 resampling onto an axis-aligned grid covering source voxel centers."""

from collections.abc import Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from nibabel.spaces import vox2out_vox


def validate_spacing(spacing: Sequence[float]) -> tuple[float, float, float]:
    """Require positive, finite, isotropic millimeter spacing without choosing a resolution."""
    values = np.asarray(spacing, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("target_spacing_mm must contain three positive finite values")
    if not np.allclose(values, values[0], rtol=0, atol=1e-8):
        raise ValueError("target_spacing_mm must be isotropic")
    return tuple(float(v) for v in values)


def output_grid(
    shape: Sequence[int], affine: np.ndarray, target_spacing_mm: Sequence[float]
) -> tuple[tuple[int, ...], np.ndarray]:
    """Predict the exact resampling grid using geometry alone (no voxel reads)."""
    spacing = validate_spacing(target_spacing_mm)
    result_shape, result_affine = vox2out_vox((tuple(shape), affine), spacing)
    return tuple(int(v) for v in result_shape), result_affine


def resample_t1(
    image: nib.spatialimages.SpatialImage, target_spacing_mm: Sequence[float]
) -> nib.Nifti1Image:
    """Use linear interpolation in physical coordinates, with zero outside the source FOV."""
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(data).all():
        raise ValueError("Non-finite MRI intensities before resampling")
    # Floating input is essential: integer ndimage output would quantize interpolation.
    floating = nib.Nifti1Image(data, image.affine, image.header.copy())
    floating.set_data_dtype(np.float32)
    grid = output_grid(image.shape, image.affine, target_spacing_mm)
    return resample_from_to(floating, grid, order=1, mode="constant", cval=0.0)
