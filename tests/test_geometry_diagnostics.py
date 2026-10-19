"""Synthetic geometry and extent diagnostics; no participant fixtures."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from brainage_xai.config.preprocessing import load_preprocessing_config
from brainage_xai.preprocessing.diagnostics import diagnose_subject, run_diagnostics
from brainage_xai.preprocessing.geometry import (
    SHEAR_COSINE_TOLERANCE,
    affine_geometry,
    candidate_tensor_sizes,
    describe,
    foreground_extent,
    geometry_summary,
)
from brainage_xai.preprocessing.intensity import foreground_mask, normalize_intensity
from brainage_xai.preprocessing.pipeline import inspect_subject


def rotation(degrees):
    radians = np.radians(degrees)
    affine = np.eye(4)
    affine[:2, :2] = [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    return affine


@pytest.mark.parametrize("affine", [np.eye(4), np.diag([-1, 2, -3, 1])])
def test_axis_aligned_affine(affine):
    record = affine_geometry(affine)
    assert record["max_obliquity_deg"] == pytest.approx(0, abs=1e-6)
    assert record["effectively_axis_aligned"]
    assert not record["has_measurable_shear"]
    assert not record["is_oblique"]
    assert record["affine_determinant_sign"] == int(np.sign(np.linalg.det(affine[:3, :3])))


def test_orthogonal_rotation_is_degrees_not_shear():
    affine = rotation(20)
    affine[:3, :3] = affine[:3, :3] @ np.diag([1, 2, 3])
    record = affine_geometry(affine)
    np.testing.assert_allclose(record["obliquity_deg"], [20, 20, 0], atol=1e-6)
    assert record["max_obliquity_deg"] == pytest.approx(20)
    assert record["is_oblique"]
    assert not record["has_measurable_shear"]
    assert record["shear_metric_max_abs_axis_cosine"] < 1e-12
    np.testing.assert_allclose(record["affine_voxel_sizes_mm"], [1, 2, 3])


def test_real_shear_and_noise_tolerance():
    affine = np.eye(4)
    affine[0, 1] = 0.2
    record = affine_geometry(affine)
    assert record["has_measurable_shear"]
    assert record["shear_metric_max_abs_axis_cosine"] == pytest.approx(0.2 / np.sqrt(1.04))
    affine[0, 1] = 1e-7
    record = affine_geometry(affine)
    assert not record["has_measurable_shear"]
    assert record["shear_metric_max_abs_axis_cosine"] < SHEAR_COSINE_TOLERANCE
    assert record["effectively_axis_aligned"]


@pytest.mark.parametrize("affine", [np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3)])
def test_invalid_affines_rejected(affine):
    with pytest.raises(ValueError):
        affine_geometry(affine)


def test_foreground_bbox_extent_and_center_offsets():
    data = np.zeros((10, 12, 14), dtype=np.float32)
    data[2:5, 4:8, 6:11] = 7
    record = foreground_extent(data, [1, 2, 3])
    assert record["foreground_bbox_start"] == [2, 4, 6]
    assert record["foreground_bbox_end_exclusive"] == [5, 8, 11]
    assert record["foreground_bbox_size"] == [3, 4, 5]
    assert record["foreground_extent_mm"] == [3, 8, 15]
    assert record["foreground_voxel_count"] == 60
    assert record["foreground_fraction"] == pytest.approx(60 / data.size)
    assert record["foreground_center_voxel"] == [3, 5.5, 8]
    assert record["foreground_center_offset_voxel"] == [-1.5, 0, 1.5]
    assert record["foreground_center_offset_mm"] == [-1.5, 0, 4.5]


def test_same_foreground_rule_as_normalizer_and_no_mutation():
    data = np.zeros((4, 5, 6), dtype=np.float32)
    data[1:3, 1:4, 1:5] = np.arange(24).reshape(2, 3, 4) - 12
    original = data.copy()
    _, stats = normalize_intensity(data, background_threshold=2)
    record = foreground_extent(data, [1, 1, 1], background_threshold=2)
    assert record["foreground_voxel_count"] == stats["foreground_voxel_count"]
    assert record["foreground_voxel_count"] == int(
        foreground_mask(data, background_threshold=2).sum()
    )
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("kind", ["empty", "nan", "inf", "negative_threshold"])
def test_bad_foreground_fails_safely(kind):
    data = np.ones((4, 5, 6), dtype=np.float32)
    if kind == "empty":
        data[:] = 0
    elif kind in {"nan", "inf"}:
        data[0, 0, 0] = np.nan if kind == "nan" else np.inf
    with pytest.raises(ValueError):
        foreground_extent(
            data, [1, 1, 1], background_threshold=-1 if kind == "negative_threshold" else 0
        )


def test_percentile_scalar_and_axis_summaries():
    values = np.arange(101, dtype=float)
    result = describe(values)
    assert result == {
        "min": 0.0,
        "p50": 50.0,
        "median": 50.0,
        "mean": 50.0,
        "p90": 90.0,
        "p95": 95.0,
        "p99": 99.0,
        "max": 100.0,
    }
    axis = describe(np.stack([values, values + 5, values + 10], axis=1))
    assert axis["p95"] == [95, 100, 105]
    assert describe(np.array([]))["median"] is None


def bbox_record(size, *, full=(64, 64, 64), start=None):
    if start is None:
        start = ((np.array(full) - size) // 2).tolist()
    return {
        "foreground_bbox_size": list(size),
        "resampled_shape": list(full),
        "foreground_bbox_start": list(start),
        "foreground_bbox_end_exclusive": (np.array(start) + size).tolist(),
    }


def test_candidate_joint_coverage_and_actual_center_crop_counts():
    records = [bbox_record([10, 10, 10]) for _ in range(100)]
    # Distinct outliers on each axis: three individual p95s do not guarantee joint 95% coverage.
    for index in range(12):
        size = [10, 10, 10]
        size[index // 4] = 40
        records[index] = bbox_record(size)
    candidates = candidate_tensor_sizes(records)
    assert len(candidates) == 6
    for candidate in candidates:
        shape = np.array(candidate["diagnostic_shape"])
        assert (shape % candidate["rounding_multiple"] == 0).all()
        assert candidate["bbox_joint_coverage_count"] >= np.ceil(
            candidate["target_joint_bbox_coverage"] * len(records)
        )
        assert not candidate["candidate_is_final"]
    assert all(
        c["bbox_larger_than_candidate_any_count"] == 0
        for c in candidates
        if c["target_joint_bbox_coverage"] == 1
    )
    # Box fits by size, but is near an edge: centered crop would remove foreground.
    off_center = candidate_tensor_sizes(
        [bbox_record([10, 10, 10], full=(100, 100, 100), start=[0, 45, 45])]
    )
    for candidate in off_center:
        assert candidate["bbox_joint_coverage_count"] == 1
        assert candidate["center_crop_foreground_loss_axis_counts"] == [1, 0, 0]
        assert candidate["center_crop_foreground_loss_any_count"] == 1


def summary_record(site, angle, shear=False):
    record = affine_geometry(rotation(angle))
    record.update(foreground_extent(np.arange(1, 121).reshape(4, 5, 6), [1, 1, 1]))
    record.update(acquisition_site=site, diagnostic_status="success", has_measurable_shear=shear)
    if shear:
        record["effectively_axis_aligned"] = False
    return record


def test_obliquity_counts_and_summary_denominators():
    records = [summary_record("Guys", 0), summary_record("HH", 8), summary_record("IOP", 20, True)]
    summary = geometry_summary(records)
    assert summary["subject_count"] == summary["geometry_subject_count"] == 3
    assert summary["sheared_affine_count"] == 1
    assert summary["non_sheared_oblique_count"] == 1
    assert summary["effectively_axis_aligned_count"] == 1
    assert summary["obliquity_threshold_counts"]["5"] == 2
    assert summary["obliquity_threshold_counts"]["10"] == 1
    assert sum(summary["max_obliquity_deg_distribution"]["counts"]) == 3


@pytest.fixture
def diagnostic_config(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for index, site in enumerate(("Guys", "HH", "IOP", "Guys", "HH", "IOP")):
        path = raw / f"synthetic{index}.nii.gz"
        data = np.zeros((7, 8, 9), dtype=np.float32)
        data[1:6, 2:7, 2:8] = np.arange(150).reshape(5, 5, 6) + 1
        image = nib.Nifti1Image(data, rotation(index * 3))
        image.header.set_xyzt_units("mm")
        nib.save(image, path)
        rows.append(
            {
                "subject_id": f"synthetic{index}",
                "t1_path": str(path),
                "age": 20 + index,
                "acquisition_site": site,
                "brain_age_label_eligible": True,
            }
        )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    pd.DataFrame(rows).to_csv(inputs / "manifest.csv", index=False)
    config_path = tmp_path / "preprocess.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "manifest_path": "inputs/manifest.csv",
                "output_dir": "artifacts",
                "target_spacing_mm": [1, 1, 1],
            }
        )
    )
    return load_preprocessing_config(config_path), rows


def test_runner_site_summaries_pilot_supplement_and_no_volume_writes(diagnostic_config):
    config, rows = diagnostic_config
    pilot = config.output_dir / "preprocessing"
    pilot.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(pilot / "manifest.csv", index=False)
    existing = (pilot / "manifest.csv").read_bytes()
    sources = {r["t1_path"]: Path(r["t1_path"]).read_bytes() for r in rows}
    summary = run_diagnostics(config)
    assert summary["success_count"] == 6
    assert summary["failure_count"] == 0
    assert set(summary["site_summaries"]) == {"Guys", "HH", "IOP"}
    assert all(s["subject_count"] == 2 for s in summary["site_summaries"].values())
    supplement = json.loads(
        (config.output_dir / "geometry_diagnostics/pilot_diagnostics.json").read_text()
    )
    assert supplement["diagnosed_pilot_subject_count"] == 6
    assert supplement["site_counts"] == {"Guys": 2, "HH": 2, "IOP": 2}
    assert (pilot / "manifest.csv").read_bytes() == existing
    assert not list(config.output_dir.rglob("*.nii*"))
    for path, original in sources.items():
        assert Path(path).read_bytes() == original


def test_failure_recording_and_partial_run(diagnostic_config):
    config, rows = diagnostic_config
    Path(rows[0]["t1_path"]).write_bytes(b"corrupt synthetic MRI")
    result = run_diagnostics(config, limit=3)
    assert result["subject_count"] == 3
    assert result["failure_count"] == 1
    assert result["success_count"] == 2
    records = pd.read_csv(config.output_dir / "geometry_diagnostics/manifest.csv")
    assert len(records) == 3
    assert records.output_path.isna().all()


def test_empty_foreground_retains_geometry_failure(diagnostic_config):
    config, rows = diagnostic_config
    subject = rows[0]
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4)), subject["t1_path"])
    result = diagnose_subject(subject, config)
    assert result["diagnostic_status"] == "failed"
    assert result["max_obliquity_deg"] == 0
    assert result["errors"]


def test_header_geometry_still_never_reads_voxels(diagnostic_config, monkeypatch):
    config, rows = diagnostic_config

    def forbid(*args, **kwargs):
        raise AssertionError("Must not read voxels for affine diagnostics")

    monkeypatch.setattr(nib.arrayproxy.ArrayProxy, "__array__", forbid)
    result = inspect_subject(rows[0], config)
    assert result["preprocessing_status"] == "success"
    assert result["max_obliquity_deg"] == 0
    assert not result["has_measurable_shear"]


def test_distinct_obliquity_and_shear_warnings(diagnostic_config):
    config, rows = diagnostic_config
    path = Path(rows[0]["t1_path"])
    data = np.ones((4, 5, 6), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, rotation(20)), path)
    result = inspect_subject(rows[0], config)
    assert "oblique_source_resampled_to_axis_aligned_grid" in result["warnings"]
    assert "non_orthogonal_source_affine" not in result["warnings"]
    affine = np.eye(4)
    affine[0, 1] = 0.2
    nib.save(nib.Nifti1Image(data, affine), path)
    result = inspect_subject(rows[0], config)
    assert "non_orthogonal_source_affine" in result["warnings"]
    assert not any("oblique_or_sheared" in warning for warning in result["warnings"])


def test_geometry_and_foreground_failure_denominators():
    success = summary_record("Guys", 0)
    failed = {
        **affine_geometry(rotation(10)),
        "diagnostic_status": "failed",
        "resampled_shape": [10, 20, 30],
    }
    summary = geometry_summary([success, failed])
    assert summary["geometry_subject_count"] == 2
    assert summary["foreground_subject_count"] == 1
    assert summary["resampled_shape_subject_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["resampled_shape_summary"]["max"] == [10, 20, 30]
