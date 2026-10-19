"""Explicit experimental rigid-registration parameters; no silent runtime downloads."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from brainage_xai.config.preprocessing import validate_output_directory

from .template import TEMPLATE_ID


@dataclass(frozen=True)
class RigidParameters:
    histogram_bins: int = 50
    sampling_strategy: str = "RANDOM"
    sampling_fraction: float = 0.02
    sampling_seed: int = 42
    shrink_factors: tuple[int, ...] = (4, 2, 1)
    smoothing_sigmas_mm: tuple[float, ...] = (2.0, 1.0, 0.0)
    learning_rate: float = 2.0
    min_step: float = 0.001
    iterations_per_level: int = 200
    relaxation_factor: float = 0.5
    gradient_tolerance: float = 1e-6
    evaluation_shrink_factor: int = 2
    use_fixed_mask: bool = True
    initialization: str = "GEOMETRY"
    threads: int = 1
    rotation_warning_deg: float = 30.0
    translation_warning_mm: float = 100.0
    clipping_warning_fraction: float = 0.15
    rigid_tolerance: float = 1e-6

    def validate(self) -> None:
        integers = {
            "histogram_bins": (self.histogram_bins, 8),
            "sampling_seed": (self.sampling_seed, 1),
            "iterations_per_level": (self.iterations_per_level, 1),
            "evaluation_shrink_factor": (self.evaluation_shrink_factor, 1),
        }
        for name, (value, minimum) in integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.sampling_strategy not in {"RANDOM", "REGULAR", "NONE"}:
            raise ValueError("Unsupported metric sampling strategy")
        if self.initialization != "GEOMETRY" or self.threads != 1 or isinstance(self.threads, bool):
            raise ValueError("This experiment requires GEOMETRY initialization and one thread")
        if not isinstance(self.use_fixed_mask, bool):
            raise ValueError("use_fixed_mask must be boolean")
        if (
            not self.shrink_factors
            or len(self.shrink_factors) != len(self.smoothing_sigmas_mm)
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in self.shrink_factors
            )
            or any(a <= b for a, b in zip(self.shrink_factors, self.shrink_factors[1:]))
            or self.shrink_factors[-1] != 1
        ):
            raise ValueError("Shrink factors must decrease to 1 and match smoothing levels")
        for value in self.smoothing_sigmas_mm:
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError("Smoothing sigmas must be finite nonnegative mm values")
        if any(a < b for a, b in zip(self.smoothing_sigmas_mm, self.smoothing_sigmas_mm[1:])):
            raise ValueError("Smoothing sigmas must be non-increasing")
        for name in (
            "sampling_fraction",
            "learning_rate",
            "min_step",
            "relaxation_factor",
            "gradient_tolerance",
            "rotation_warning_deg",
            "translation_warning_mm",
            "clipping_warning_fraction",
            "rigid_tolerance",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if (
            self.sampling_fraction > 1
            or not self.relaxation_factor < 1
            or self.clipping_warning_fraction > 1
        ):
            raise ValueError("Invalid sampling, relaxation, or clipping fraction")
        if self.min_step >= self.learning_rate:
            raise ValueError("min_step must be smaller than learning_rate")


@dataclass(frozen=True)
class RegistrationConfig:
    manifest_path: Path
    geometry_manifest_path: Path
    output_dir: Path
    template_path: Path
    brain_mask_path: Path | None
    parameters: RigidParameters
    subjects_per_site: int = 6
    background_threshold: float = 0.0
    qc_dpi: int = 110
    checkerboard_pixels: int = 12


def load_registration_config(path: Path) -> RegistrationConfig:
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid registration YAML: {exc}") from exc
    allowed = {
        "manifest_path",
        "geometry_manifest_path",
        "output_dir",
        "template",
        "registration",
        "subjects_per_site",
        "background_threshold",
        "qc_dpi",
        "checkerboard_pixels",
    }
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("Invalid registration configuration keys")

    def resolve(value: object, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a path string")
        value = os.path.expandvars(value)
        if re.search(r"\$(?:\w+|\{[^}]+\})", value):
            raise ValueError(f"Unset environment variable in {name}")
        result = Path(value).expanduser()
        return (path.parent / result).resolve() if not result.is_absolute() else result.resolve()

    manifest = resolve(raw.get("manifest_path"), "manifest_path")
    geometry = resolve(raw.get("geometry_manifest_path"), "geometry_manifest_path")
    output = resolve(raw.get("output_dir"), "output_dir")
    if not manifest.is_file() or not geometry.is_file():
        raise ValueError("Existing Phase 1 and geometry diagnostic manifests are required")
    template = raw.get("template")
    if not isinstance(template, dict) or set(template) - {
        "id",
        "resolution",
        "t1_path",
        "brain_mask_path",
    }:
        raise ValueError("Invalid template configuration")
    if template.get("id") != TEMPLATE_ID or str(template.get("resolution")).zfill(2) != "01":
        raise ValueError("Only MNI152NLin2009cAsym resolution 01 is supported")
    t1 = resolve(template.get("t1_path"), "template.t1_path")
    mask = (
        resolve(template["brain_mask_path"], "template.brain_mask_path")
        if template.get("brain_mask_path")
        else None
    )
    parameters = raw.get("registration", {})
    if not isinstance(parameters, dict) or set(parameters) - set(
        RigidParameters.__dataclass_fields__
    ):
        raise ValueError("Unsupported registration parameters")
    for key in ("shrink_factors", "smoothing_sigmas_mm"):
        if key in parameters:
            if not isinstance(parameters[key], list):
                raise ValueError(f"{key} must be a list")
            parameters[key] = tuple(parameters[key])
    params = RigidParameters(**parameters)
    params.validate()
    n = raw.get("subjects_per_site", 6)
    dpi = raw.get("qc_dpi", 110)
    tiles = raw.get("checkerboard_pixels", 12)
    for value, name, minval in (
        (n, "subjects_per_site", 6),
        (dpi, "qc_dpi", 50),
        (tiles, "checkerboard_pixels", 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minval:
            raise ValueError(f"{name} must be an integer >= {minval}")
    if n != 6:
        raise ValueError("This experimental command is limited to six subjects per site")
    threshold = raw.get("background_threshold", 0.0)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("background_threshold must be finite and nonnegative")
    validate_output_directory(output, [manifest, geometry, t1, *([mask] if mask else [])])
    return RegistrationConfig(
        manifest, geometry, output, t1, mask, params, n, float(threshold), dpi, tiles
    )
