"""Strict Phase 2 parameters and private-output path checks."""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from brainage_xai.preprocessing.resampling import validate_spacing


@dataclass(frozen=True)
class PreprocessingConfig:
    manifest_path: Path
    output_dir: Path
    target_spacing_mm: tuple[float, float, float]
    lower_percentile: float
    upper_percentile: float
    background_threshold: float
    split_seed: int
    split_ratios: tuple[float, float, float]
    age_bin_width: float
    qc_enabled: bool
    qc_dpi: int


def validate_output_directory(output: Path, inputs: list[Path]) -> None:
    """Require ignored output inside Git repositories and separation from input locations."""
    output = Path(output).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("output_dir must be a directory")
    for source in inputs:
        source = Path(source).resolve()
        if output == source or output in source.parents:
            raise ValueError("Output directory must not contain an input file")
        if source.parent == output or source.parent in output.parents:
            raise ValueError("Output directory must be outside source input directories")
    for root in (output, *output.parents):
        if (root / ".git").exists():
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "check-ignore",
                    "-q",
                    "--",
                    str(output / ".brainage-output-check"),
                ],
                check=False,
            )
            if result.returncode != 0:
                raise ValueError("Output inside a Git repository must be ignored (use artifacts/)")
            break


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    """Resolve environment/relative paths and validate all numerical parameters explicitly."""
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    allowed = {"manifest_path", "output_dir", "target_spacing_mm", "intensity", "split", "qc"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("Phase 2 config must be a mapping with supported keys")

    def resolve(key: str) -> Path:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty path")
        value = os.path.expandvars(value)
        if re.search(r"\$(?:\w+|\{[^}]+\})", value):
            raise ValueError(f"Unset environment variable in {key}")
        result = Path(value).expanduser()
        return (path.parent / result).resolve() if not result.is_absolute() else result.resolve()

    def section(key: str, keys: set[str]) -> dict:
        result = raw.get(key, {})
        if not isinstance(result, dict) or set(result) - keys:
            raise ValueError(f"Invalid {key} configuration")
        return result

    def number(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
        return float(value)

    manifest, output = resolve("manifest_path"), resolve("output_dir")
    if not manifest.is_file():
        raise ValueError("manifest_path must be an existing Phase 1 CSV")
    validate_output_directory(output, [manifest])
    spacing = raw.get("target_spacing_mm")
    if not isinstance(spacing, list) or any(isinstance(v, bool) for v in spacing):
        raise ValueError("target_spacing_mm must be a list of three positive numbers")
    spacing = validate_spacing([number(v, "target_spacing_mm") for v in spacing])
    intensity = section(
        "intensity", {"lower_percentile", "upper_percentile", "background_threshold"}
    )
    low = number(intensity.get("lower_percentile", 0.5), "lower_percentile")
    high = number(intensity.get("upper_percentile", 99.5), "upper_percentile")
    background = number(intensity.get("background_threshold", 0.0), "background_threshold")
    if not 0 <= low < high <= 100 or background < 0:
        raise ValueError("Invalid clipping percentiles or background threshold")
    split = section("split", {"seed", "ratios", "age_bin_width"})
    seed = split.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split.seed must be a non-negative integer")
    ratios = split.get("ratios", [0.7, 0.15, 0.15])
    if not isinstance(ratios, list) or len(ratios) != 3:
        raise ValueError("split.ratios requires train, validation, test values")
    ratios = tuple(number(v, "split.ratios") for v in ratios)
    if any(v <= 0 for v in ratios) or not np.isclose(sum(ratios), 1, atol=1e-8, rtol=0):
        raise ValueError("Split ratios must be positive and sum to one")
    width = number(split.get("age_bin_width", 10), "age_bin_width")
    if width <= 0:
        raise ValueError("age_bin_width must be positive")
    qc = section("qc", {"enabled", "dpi"})
    enabled, dpi = qc.get("enabled", True), qc.get("dpi", 120)
    if (
        not isinstance(enabled, bool)
        or isinstance(dpi, bool)
        or not isinstance(dpi, int)
        or dpi < 50
    ):
        raise ValueError("qc.enabled must be boolean and qc.dpi an integer >= 50")
    return PreprocessingConfig(
        manifest, output, spacing, low, high, background, seed, ratios, width, enabled, dpi
    )
