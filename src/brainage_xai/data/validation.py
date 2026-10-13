"""Header sanity checks and intensity validation with one volume read per file."""

from itertools import product
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def validate_nifti(path: Path) -> dict[str, Any]:
    """Read one scaled volume, inspect every voxel, and return only scalar/list results."""
    result: dict[str, Any] = {
        "t1_path": str(path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "shape": None,
        "ndim": None,
        "voxel_spacing": None,
        "affine_valid": False,
        "finite_count": 0,
        "nonfinite_count": 0,
        "intensity_min": None,
        "intensity_max": None,
        "intensity_mean": None,
        "intensity_std": None,
    }
    try:
        img = nib.load(path, mmap=True)
        shape = img.shape
        result.update(
            shape=list(shape),
            ndim=len(shape),
            voxel_spacing=[float(v) for v in img.header.get_zooms()],
        )
        if len(shape) != 3:
            result["errors"].append("expected_3d_t1")
        if any(s <= 0 for s in shape):
            result["errors"].append("empty_dimension")
        spacing = np.asarray(img.header.get_zooms(), dtype=float)
        if not np.all(np.isfinite(spacing) & (spacing > 0)):
            result["errors"].append("invalid_voxel_spacing")
        affine = img.affine
        sane = (
            affine is not None
            and affine.shape == (4, 4)
            and np.isfinite(affine).all()
            and np.allclose(affine[3], [0, 0, 0, 1])
            and np.linalg.matrix_rank(affine[:3, :3]) == 3
        )
        result["affine_valid"] = bool(sane)
        if not sane:
            result["errors"].append("invalid_affine")
        if not int(img.header["qform_code"]) and not int(img.header["sform_code"]):
            result["warnings"].append("no_coded_orientation_using_fallback_affine")
        # Materialize the proxy once: repeated proxy slices can repeatedly seek/decompress gzip.
        # Keep its native/scaling-derived dtype rather than rounding scaled values to float32.
        # Only this volume and float64 plane-sized statistics buffers remain in memory.
        data = np.asanyarray(img.dataobj)
        count, mean, m2 = 0, 0.0, 0.0
        for tail in product(*(range(s) for s in shape[2:])):
            block = np.asarray(data[(slice(None), slice(None)) + tail], dtype=np.float64)
            mask = np.isfinite(block)
            result["nonfinite_count"] += int(block.size - mask.sum())
            values = block[mask]
            n = values.size
            if not n:
                continue
            bmean = float(values.mean())
            delta = bmean - mean
            new_count = count + n
            m2 += float(np.sum((values - bmean) ** 2)) + delta**2 * count * n / new_count
            mean += delta * n / new_count
            count = new_count
            lo, hi = float(values.min()), float(values.max())
            result["intensity_min"] = (
                lo if result["intensity_min"] is None else min(lo, result["intensity_min"])
            )
            result["intensity_max"] = (
                hi if result["intensity_max"] is None else max(hi, result["intensity_max"])
            )
        result["finite_count"] = int(count)
        if count:
            result.update(intensity_mean=mean, intensity_std=float(np.sqrt(m2 / count)))
        if result["nonfinite_count"]:
            result["errors"].append("nonfinite_voxels")
        if not count:
            result["errors"].append("no_finite_voxels")
        elif result["intensity_min"] == result["intensity_max"]:
            result["warnings"].append("constant_intensity")
    except Exception as exc:
        # Corrupt inputs should not prevent remaining subjects from being audited.
        result["errors"].append(f"unreadable_mri: {type(exc).__name__}: {exc}")
    result["valid"] = not result["errors"]
    return result
