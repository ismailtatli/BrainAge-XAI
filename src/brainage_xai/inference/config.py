"""Fail-closed local runtime configuration; no downloads or alternative checkpoints."""

import importlib.metadata
import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from brainage_xai.config.preprocessing import validate_output_directory
from brainage_xai.registration.template import MASK_SHA256, T1_SHA256, sha256

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_SHA256 = "146b98ab66a4f951dcd31b2c2f913aa7908c73fde8d3af9febd74e2f50d8736a"


class RuntimeUnavailable(ValueError):
    """A required accepted local runtime resource is missing or differs."""


@dataclass(frozen=True)
class RuntimeConfig:
    paths: dict
    contract: dict
    correction: dict
    reference: dict
    slices: dict
    temporary_dir: Path
    output_dir: Path
    device: str
    max_upload_bytes: int
    max_voxels: int
    config_sha256: str


def safe_destination(path, inputs=()):
    path = Path(path).resolve()
    protected = [
        ROOT / "artifacts" / n for n in ("preprocessed", "modeling", "resources", "registration")
    ]
    protected += [
        ROOT / n
        for n in ("src", "tests", "configs", "docs", ".git", ".venv", "data", "reports", "app")
    ]
    if any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise ValueError("Output/temp directory overlaps protected project inputs")
    validate_output_directory(path, list(inputs))
    return path


def load_runtime_config(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeUnavailable(
            "Local inference config missing; copy inference.example.yaml "
            "to inference.local.yaml and supply the accepted artifacts"
        )
    raw = yaml.safe_load(path.read_text())
    file_keys = {
        "contract",
        "checkpoint",
        "correction",
        "reference",
        "slices",
        "template",
        "template_mask",
    }
    allowed = file_keys | {
        "schema_version",
        "temporary_dir",
        "output_dir",
        "device",
        "max_upload_bytes",
        "max_voxels",
    }
    if not isinstance(raw, dict) or set(raw) != allowed or raw["schema_version"] != 1:
        raise RuntimeUnavailable("Invalid inference configuration keys/schema")

    def resolve(key):
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            raise RuntimeUnavailable(f"Expected a local path for {key}")
        value = os.path.expandvars(value)
        if re.search(r"\$(?:\w+|\{[^}]+\})", value):
            raise RuntimeUnavailable(f"Unset environment variable for {key}")
        p = Path(value).expanduser()
        return (path.parent / p).resolve() if not p.is_absolute() else p.resolve()

    paths = {key: resolve(key) for key in file_keys}
    for key, p in paths.items():
        if not p.is_file():
            raise RuntimeUnavailable(f"Required accepted artifact unavailable: {key}; no fallback")
    if sha256(paths["contract"]) != CONTRACT_SHA256:
        raise RuntimeUnavailable("Accepted runtime contract checksum mismatch")
    contract = yaml.safe_load(paths["contract"].read_text())
    for key, expected in {
        **contract["artifact_sha256"],
        "template": T1_SHA256,
        "template_mask": MASK_SHA256,
    }.items():
        if sha256(paths[key]) != expected:
            raise RuntimeUnavailable(f"Accepted artifact checksum mismatch: {key}; no fallback")
    for name, expected in contract["spec_sha256"].items():
        if sha256(ROOT / "configs" / name) != expected:
            raise RuntimeUnavailable(f"Frozen scientific config differs: {name}")
    for name, expected in contract["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeUnavailable(f"Frozen scientific implementation differs: {name}")
    spec = yaml.safe_load((ROOT / "configs/preprocessing.frozen.yaml").read_text())
    versions = {**spec["software"]["packages"], **contract["software"]}
    if platform.python_version() != contract["python"] or any(
        importlib.metadata.version(k) != v for k, v in versions.items()
    ):
        raise RuntimeUnavailable(
            "Frozen Python/scientific package versions differ; see setup guide"
        )
    correction, reference, slices = [
        json.loads(paths[k].read_text()) for k in ("correction", "reference", "slices")
    ]
    if (
        correction["fit_split"] != "validation"
        or reference["fit_split"] != "validation"
        or set(correction["fit_subjects"]) != set(reference["fit_subjects"])
        or len(set(reference["fit_subjects"])) != 84
    ):
        raise RuntimeUnavailable("Invalid validation-only calibration provenance")
    if raw["device"] not in ("auto", "cpu", "mps"):
        raise RuntimeUnavailable("device must be auto, cpu or mps")
    for key in ("max_upload_bytes", "max_voxels"):
        if type(raw[key]) is not int or raw[key] <= 0:
            raise RuntimeUnavailable(f"{key} must be a positive technical resource bound")
    return RuntimeConfig(
        paths,
        contract,
        correction,
        reference,
        slices,
        safe_destination(resolve("temporary_dir"), paths.values()),
        safe_destination(resolve("output_dir"), paths.values()),
        raw["device"],
        raw["max_upload_bytes"],
        raw["max_voxels"],
        sha256(path),
    )
