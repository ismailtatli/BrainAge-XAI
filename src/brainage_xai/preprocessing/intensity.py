"""Foreground-only robust normalization, independently fitted to each MRI."""

from typing import Any

import numpy as np


def foreground_mask(data: np.ndarray, *, background_threshold: float = 0.0) -> np.ndarray:
    """Shared non-background rule; includes skull/scalp/noise and is not a brain mask."""
    if not np.isfinite(background_threshold) or background_threshold < 0:
        raise ValueError("background_threshold must be non-negative and finite")
    data = np.asarray(data)
    if not np.isfinite(data).all():
        raise ValueError("Non-finite MRI intensities during foreground detection")
    # Comparisons avoid signed-integer abs overflow (e.g. int16 -32768).
    mask = (data > background_threshold) | (data < -background_threshold)
    if not mask.any():
        raise ValueError("Empty foreground")
    return mask


def normalize_intensity(
    data: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    background_threshold: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip and z-score non-background voxels; reject degenerate or non-finite inputs.

    Foreground is abs(intensity) > threshold, not a brain/skull-stripping mask.
    Background remains zero. Source data is never modified.
    """
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")
    data = np.asarray(data)
    mask = foreground_mask(data, background_threshold=background_threshold)
    foreground = data[mask].astype(np.float64)
    low, high = np.percentile(foreground, [lower_percentile, upper_percentile])
    clipped = np.clip(foreground, low, high)
    mean, std = float(clipped.mean()), float(clipped.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Constant or numerically degenerate foreground")
    output = np.zeros(data.shape, dtype=np.float32)
    output[mask] = (clipped - mean) / std
    if not np.isfinite(output).all():
        raise ValueError("Non-finite normalized intensities")
    return output, {
        "normalization_method": "foreground_percentile_clip_zscore",
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "background_threshold": float(background_threshold),
        "foreground_voxel_count": int(mask.sum()),
        "clip_low": float(low),
        "clip_high": float(high),
        "foreground_clipped_mean": mean,
        "foreground_clipped_std": std,
    }
