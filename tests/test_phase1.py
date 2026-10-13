"""Synthetic-only regression tests: no real participant data is required."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from brainage_xai.config import load_config
from brainage_xai.data.audit import run_audit
from brainage_xai.data.discovery import discover_t1_files, parse_subject_id
from brainage_xai.data.matching import build_manifest
from brainage_xai.data.metadata import load_metadata, normalize_metadata, normalize_subject_id
from brainage_xai.data.validation import validate_nifti
from brainage_xai.visualization.slices import central_slices, plot_central_slices


@pytest.mark.parametrize(
    "name,expected",
    [
        ("IXI002-Guys-0828-T1.nii.gz", "IXI002"),
        ("IXI2-HH-1-T1.nii", "IXI002"),
        ("ixi123-iop-4-t1.NII.GZ", "IXI123"),
        ("IXI002-T2.nii", None),
        ("bad.nii", None),
        ("IXI000-T1.nii", None),
        ("IXI002-T1.nii.gz.bak", None),
    ],
)
def test_filename(name, expected):
    assert parse_subject_id(name) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (2, "IXI002"),
        (2.0, "IXI002"),
        (" 002 ", "IXI002"),
        ("ixi002", "IXI002"),
        ("IXI-002", "IXI002"),
        (None, None),
        (np.nan, None),
        (2.5, None),
        (True, None),
        ("IXI002-Guys", None),
        (0, None),
        ("2e0", None),
    ],
)
def test_metadata_id(value, expected):
    assert normalize_subject_id(value) == expected


def metadata():
    return normalize_metadata(
        pd.DataFrame(
            {
                "IXI_ID": [1, "001", 2, 3, 4, "broken"],
                "AGE": [20, 21, None, "bad", 45, 50],
                "SEX": [1, 2, 1, 2, 1, 1],
            }
        )
    )


def test_matching_duplicates_and_missing_age(tmp_path):
    paths = [
        tmp_path / name
        for name in [
            "IXI001-a-T1.nii",
            "IXI001-b-T1.nii.gz",
            "IXI002-T1.nii",
            "IXI005-T1.nii",
            "bad.nii",
        ]
    ]
    manifest = build_manifest(paths, metadata())
    subjects = manifest.dropna(subset=["subject_id"]).set_index("subject_id")
    assert {"duplicate_mri", "duplicate_subject_id"} <= set(subjects.loc["IXI001", "issues"])
    assert pd.isna(subjects.loc["IXI001", "age"])
    assert subjects.loc["IXI001", "metadata_rows"] == [2, 3]
    assert "missing_age" in subjects.loc["IXI002", "issues"]
    assert subjects.loc["IXI003", "matching_status"] == "demographics_without_mri"
    assert subjects.loc["IXI005", "matching_status"] == "mri_without_demographics"
    assert "invalid_mri_filename" in manifest.matching_status.tolist()


@pytest.mark.parametrize(
    "age,status",
    [
        (None, "missing"),
        (" ", "missing"),
        ("bad", "invalid"),
        (-1, "invalid"),
        (0, "invalid"),
        (np.inf, "invalid"),
        (True, "invalid"),
        ("42.5", "valid"),
    ],
)
def test_age_validation(age, status):
    normalized = normalize_metadata(pd.DataFrame({"IXI_ID": [1], "AGE": [age]}))
    assert normalized.iloc[0].age_status == status
    assert (pd.isna(normalized.iloc[0].age)) == (status != "valid")


def test_spreadsheet(tmp_path):
    path = tmp_path / "demo.xlsx"
    pd.DataFrame({" IXI_ID ": ["002"], "AGE": [30], "SEX": [2], "SITE": ["HH"]}).to_excel(
        path, index=False
    )
    result = load_metadata(path, site_column="SITE")
    assert result.iloc[0].subject_id == "IXI002"
    assert result.iloc[0].site == "HH"
    with pytest.raises(ValueError, match="Missing configured"):
        load_metadata(path, scanner_column="SCANNER")
    with pytest.raises(ValueError, match="Missing configured"):
        normalize_metadata(pd.DataFrame({"ID": [1]}))


def setup_config(tmp_path):
    t1 = tmp_path / "mri"
    t1.mkdir()
    sheet = tmp_path / "demographics.xlsx"
    pd.DataFrame({"IXI_ID": [1], "AGE": [30]}).to_excel(sheet, index=False)
    config_path = tmp_path / "config.yaml"
    raw = {"t1_dir": "mri", "demographics_path": "demographics.xlsx", "output_dir": "audit"}
    config_path.write_text(yaml.safe_dump(raw))
    return config_path, raw


def test_configuration_paths_and_environment(tmp_path, monkeypatch):
    path, raw = setup_config(tmp_path)
    config = load_config(path)
    assert config.t1_dir == tmp_path / "mri"
    monkeypatch.setenv("TEST_IXI_DIR", str(config.t1_dir))
    raw["t1_dir"] = "${TEST_IXI_DIR}"
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).t1_dir == config.t1_dir
    monkeypatch.delenv("TEST_IXI_DIR")
    with pytest.raises(ValueError, match="unset"):
        load_config(path)


@pytest.mark.parametrize(
    "updates",
    [
        {"t1_dir": "missing"},
        {"t1_dir": ""},
        {"demographics_path": "missing.xlsx"},
        {"output_dir": "mri/output"},
        {"output_dir": "."},
        {"output_dir": "demographics.xlsx"},
        {"metadata": []},
        {"metadata": {"age_column": None}},
        {"metadata": {"sheet_name": -1}},
        {"unexpected": 1},
    ],
)
def test_invalid_configuration(tmp_path, updates):
    path, raw = setup_config(tmp_path)
    raw.update(updates)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


def save_image(path, data=None, affine=None):
    if data is None:
        data = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
    if affine is None:
        affine = np.diag([1.0, 2.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), path)


def test_validation_and_discovery(tmp_path):
    good = tmp_path / "IXI001-T1.nii.gz"
    bad = tmp_path / "IXI002-T1.nii"
    save_image(good)
    bad.write_bytes(b"not a NIfTI")
    result = validate_nifti(good)
    assert result["valid"]
    assert result["shape"] == [4, 5, 6]
    assert result["voxel_spacing"] == [1, 2, 3]
    assert result["finite_count"] == 120
    assert result["intensity_mean"] == pytest.approx(59.5)
    assert result["intensity_std"] == pytest.approx(np.arange(120).std())
    assert "unreadable_mri" in validate_nifti(bad)["errors"][0]
    assert discover_t1_files(tmp_path) == [good, bad]


def test_nonfinite_and_dimensionality(tmp_path):
    path = tmp_path / "IXI001-T1.nii"
    data = np.ones((3, 4, 5), dtype=np.float32)
    data[0, 0, 0] = np.nan
    data[1, 0, 0] = np.inf
    save_image(path, data)
    result = validate_nifti(path)
    assert result["nonfinite_count"] == 2
    assert "nonfinite_voxels" in result["errors"]
    save_image(path, np.ones((3, 4, 5, 2), dtype=np.float32))
    assert "expected_3d_t1" in validate_nifti(path)["errors"]


@pytest.mark.parametrize(
    "affine",
    [
        np.diag([-1.0, 2.0, -3.0, 1.0]),
        np.array([[0, -2, 0, 0], [0, 0, 3, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=float),
    ],
)
def test_orientation(tmp_path, affine):
    path = tmp_path / "IXI001-T1.nii"
    save_image(path, affine=affine)
    planes, spacing = central_slices(path)
    canonical = nib.as_closest_canonical(nib.load(path)).get_fdata()
    for axis, plane in enumerate(planes):
        assert np.array_equal(plane, np.take(canonical, canonical.shape[axis] // 2, axis=axis).T)
    assert sorted(spacing) == [1, 2, 3]


def test_plot(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    path = tmp_path / "IXI001-T1.nii.gz"
    save_image(path)
    output = tmp_path / "slices.png"
    plot_central_slices(path, output)
    assert output.stat().st_size > 0


def test_end_to_end_audit_reproducible(tmp_path):
    path, _ = setup_config(tmp_path)
    config = load_config(path)
    good = config.t1_dir / "IXI001-T1.nii.gz"
    save_image(good)
    source_before = good.read_bytes()
    (config.t1_dir / "IXI002-T1.nii").write_bytes(b"corrupt")
    summary = run_audit(config)
    assert summary["discovered_mri_count"] == 2
    assert summary["matched_subject_count"] == 1
    assert summary["validation_failure_count"] == 1
    assert summary["age_statistics"]["mean"] == 30
    assert summary["valid_age_count"] == 1
    assert set(p.name for p in config.output_dir.iterdir()) == {
        "manifest.csv",
        "metadata_records.csv",
        "mri_validation.csv",
        "summary.json",
        "summary.md",
    }
    assert json.loads((config.output_dir / "summary.json").read_text()) == summary
    first = {p.name: p.read_bytes() for p in config.output_dir.iterdir()}
    run_audit(config)
    assert first == {p.name: p.read_bytes() for p in config.output_dir.iterdir()}
    assert good.read_bytes() == source_before


def test_empty_audit(tmp_path):
    path, _ = setup_config(tmp_path)
    summary = run_audit(load_config(path))
    assert summary["discovered_mri_count"] == 0
    assert summary["age_statistics"]["mean"] is None
    assert summary["unmatched_subject_count"] == 1


def test_malformed_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("metadata: [unterminated")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(path)


def test_legacy_xls_engine(tmp_path, monkeypatch):
    """Verify legacy reader dispatch without committing a spreadsheet fixture."""
    calls = []

    def reader(path, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"IXI_ID": [1], "AGE": [25]})

    monkeypatch.setattr(pd, "read_excel", reader)
    result = load_metadata(tmp_path / "synthetic.xls", sheet_name="Participants")
    assert calls == [{"sheet_name": "Participants", "engine": "xlrd"}]
    assert result.iloc[0].age == 25


def test_missing_orientation_rejected_for_plot(tmp_path):
    path = tmp_path / "IXI001-T1.nii"
    image = nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.float32), np.eye(4))
    image.set_qform(None, code=0)
    image.set_sform(None, code=0)
    nib.save(image, path)
    assert "no_coded_orientation_using_fallback_affine" in validate_nifti(path)["warnings"]
    with pytest.raises(ValueError, match="No coded orientation"):
        central_slices(path)


def test_truncated_payload(tmp_path):
    path = tmp_path / "IXI001-T1.nii"
    save_image(path)
    path.write_bytes(path.read_bytes()[:360])
    result = validate_nifti(path)
    assert not result["valid"]
    assert any(error.startswith("unreadable_mri") for error in result["errors"])


def test_completely_empty_inputs(tmp_path):
    path, _ = setup_config(tmp_path)
    config = load_config(path)
    pd.DataFrame({"IXI_ID": [], "AGE": []}).to_excel(config.demographics_path, index=False)
    summary = run_audit(config)
    assert summary["identified_subject_count"] == 0
    assert summary["matched_subject_count"] == 0
    assert summary["valid_age_count"] == 0


def test_duplicate_and_invalid_age_summary(tmp_path):
    path, _ = setup_config(tmp_path)
    config = load_config(path)
    pd.DataFrame(
        {"IXI_ID": [1, "001", 2, 3, "bad"], "AGE": [20, 21, None, "invalid", 40]}
    ).to_excel(config.demographics_path, index=False)
    for subject in (1, 2, 3):
        save_image(config.t1_dir / f"IXI{subject:03d}-T1.nii")
    summary = run_audit(config)
    assert summary["matched_subject_count"] == 3
    assert summary["duplicate_metadata_subject_count"] == 1
    assert summary["valid_age_count"] == 0
    assert summary["missing_age_count"] == 1
    assert summary["invalid_age_count"] == 1
    assert summary["invalid_metadata_id_count"] == 1
    assert summary["age_statistics"]["mean"] is None
