"""Synthetic-only Phase 2 tests: spatial consistency, subject isolation, and safe failures."""

from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml
from nibabel.orientations import apply_orientation, inv_ornt_aff

from brainage_xai.config.preprocessing import load_preprocessing_config, validate_output_directory
from brainage_xai.data.preparation import eligible_subjects, select_pilot
from brainage_xai.data.splits import run_split, split_subjects, split_summary
from brainage_xai.preprocessing.intensity import normalize_intensity
from brainage_xai.preprocessing.orientation import canonicalize, inspect_orientation
from brainage_xai.preprocessing.pipeline import inspect_subject, run_preprocessing
from brainage_xai.preprocessing.qc import save_comparison
from brainage_xai.preprocessing.resampling import output_grid, resample_t1


@pytest.fixture
def phase2(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for index in range(9):
        path = raw / f"subject{index}.nii.gz"
        data = np.zeros((7, 8, 9), dtype=np.float32)
        data[1:-1, 1:-1, 1:-1] = np.arange(210).reshape(5, 6, 7) + 1
        nib.save(nib.Nifti1Image(data, np.diag([1.5, 1.5, 2.0, 1])), path)
        rows.append(
            {
                "subject_id": f"subject{index}",
                "t1_path": str(path),
                "age": 20 + 5 * index,
                "acquisition_site": ("Guys", "HH", "IOP")[index % 3],
                "brain_age_label_eligible": index != 8,
            }
        )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest = inputs / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "manifest_path": "inputs/manifest.csv",
                "output_dir": "artifacts",
                "target_spacing_mm": [1, 1, 1],
                "qc": {"enabled": True, "dpi": 60},
            }
        )
    )
    return config_path, load_preprocessing_config(config_path)


def test_eligible_filter_and_pilot_sites(phase2):
    _, config = phase2
    frame = eligible_subjects(config.manifest_path)
    assert len(frame) == 8
    assert "subject8" not in set(frame.subject_id)
    selected = select_pilot(frame, 6)
    assert len(selected) == 6
    assert selected.acquisition_site.value_counts().to_dict() == {"Guys": 2, "HH": 2, "IOP": 2}
    pd.testing.assert_frame_equal(selected, select_pilot(frame.sample(frac=1, random_state=8), 6))
    assert len(select_pilot(frame, 100)) == 8
    assert select_pilot(frame.iloc[:0], 6).empty
    with pytest.raises(ValueError, match="positive"):
        select_pilot(frame, 0)


@pytest.mark.parametrize(
    "kind", ["column", "duplicate", "path", "age", "flag", "reused_path", "id"]
)
def test_manifest_validation(phase2, kind):
    _, config = phase2
    frame = pd.read_csv(config.manifest_path)
    if kind == "column":
        frame = frame.drop(columns=["age"])
    elif kind == "duplicate":
        frame.loc[1, "subject_id"] = frame.loc[0, "subject_id"]
    elif kind == "path":
        frame.loc[0, "t1_path"] = None
    elif kind == "age":
        frame.loc[0, "age"] = -1
    elif kind == "flag":
        frame["brain_age_label_eligible"] = "possibly"
    elif kind == "reused_path":
        frame.loc[1, "t1_path"] = frame.loc[0, "t1_path"]
    else:
        frame.loc[0, "subject_id"] = "../unsafe"
    frame.to_csv(config.manifest_path, index=False)
    with pytest.raises(ValueError):
        eligible_subjects(config.manifest_path)


def test_ineligible_missing_paths_do_not_enter_preprocessing(phase2):
    _, config = phase2
    frame = pd.read_csv(config.manifest_path)
    frame.loc[8, "t1_path"] = None
    frame.to_csv(config.manifest_path, index=False)
    assert len(eligible_subjects(config.manifest_path)) == 8


@pytest.mark.parametrize(
    "transform", [np.array([[0, -1], [1, 1], [2, -1]]), np.array([[2, -1], [0, 1], [1, -1]])]
)
def test_canonical_spatial_consistency(transform):
    data = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    affine[:3, 3] = [10, -20, 30]
    changed = apply_orientation(data, transform)
    changed_affine = affine @ inv_ornt_aff(transform, data.shape)
    image = nib.Nifti1Image(changed, changed_affine)
    result = canonicalize(image)
    np.testing.assert_array_equal(result.dataobj, data)
    np.testing.assert_allclose(result.affine, affine)
    metadata = inspect_orientation(image)
    assert metadata["canonical_shape"] == list(data.shape)
    assert metadata["canonical_orientation"] == ["R", "A", "S"]


def test_resampling_spacing_and_continuous_intensity():
    data = np.indices((4, 5, 6))[0].astype(np.int16) * 10
    image = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
    result = resample_t1(image, (1, 1, 1))
    assert result.shape == (7, 9, 11)
    np.testing.assert_allclose(nib.affines.voxel_sizes(result.affine), [1, 1, 1])
    assert result.get_fdata()[1, 2, 2] == pytest.approx(5)  # linear, not nearest integer voxel
    assert result.get_data_dtype() == np.dtype("float32")
    expected_shape, expected_affine = output_grid(image.shape, image.affine, (1, 1, 1))
    assert result.shape == expected_shape
    np.testing.assert_allclose(result.affine, expected_affine)


def test_oblique_header_prediction_matches_actual_resampling():
    angle = np.deg2rad(15)
    affine = np.eye(4)
    affine[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    image = nib.Nifti1Image(np.ones((5, 6, 7), dtype=np.float32), affine)
    canonical = canonicalize(image)
    meta = inspect_orientation(image)
    shape, predicted_affine = output_grid(
        meta["canonical_shape"], np.array(meta["canonical_affine"]), (2, 2, 2)
    )
    actual = resample_t1(canonical, (2, 2, 2))
    assert actual.shape == shape
    np.testing.assert_allclose(actual.affine, predicted_affine)


def test_intensity_background_outliers_and_immutable_input():
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[1:-1, 1:-1, 1:-1] = np.arange(512).reshape(8, 8, 8) + 1
    data[1, 1, 1] = 1e10
    before = data.copy()
    result, stats = normalize_intensity(data, lower_percentile=1, upper_percentile=99)
    mask = data != 0
    assert np.all(result[~mask] == 0)
    assert result[mask].mean() == pytest.approx(0, abs=1e-6)
    assert result[mask].std() == pytest.approx(1, abs=1e-6)
    assert stats["clip_high"] < 1e5
    assert result.max() < 3
    np.testing.assert_array_equal(data, before)
    assert stats["foreground_voxel_count"] == 512


def test_normalization_is_per_subject():
    data = np.arange(1, 121, dtype=np.float32).reshape(4, 5, 6)
    first, _ = normalize_intensity(data)
    second, _ = normalize_intensity(data * 5 + 100)
    np.testing.assert_allclose(first, second, atol=1e-6)


@pytest.mark.parametrize("kind", ["constant", "empty", "nan", "inf"])
def test_degenerate_intensity_fails(kind):
    data = np.ones((4, 5, 6), dtype=np.float32)
    if kind == "empty":
        data[:] = 0
    elif kind in ("nan", "inf"):
        data[0, 0, 0] = np.nan if kind == "nan" else np.inf
    with pytest.raises(ValueError):
        normalize_intensity(data)


@pytest.mark.parametrize(
    "parameters",
    [
        {"lower_percentile": 99, "upper_percentile": 2},
        {"background_threshold": -1},
        {"upper_percentile": 101},
    ],
)
def test_invalid_normalization_parameters(parameters):
    with pytest.raises(ValueError):
        normalize_intensity(np.arange(120).reshape(4, 5, 6), **parameters)


def split_frame(n=300):
    return pd.DataFrame(
        {
            "subject_id": [f"s{i:04d}" for i in range(n)],
            "age": [20 + (i % 6) * 10 for i in range(n)],
            "acquisition_site": [("A", "B", "C")[(i // 6) % 3] for i in range(n)],
            "brain_age_label_eligible": True,
            "t1_path": "unused_for_pure_split",
        }
    )


def test_deterministic_split_no_leakage_coverage_and_stratification():
    frame = split_frame(600)
    result = split_subjects(frame, seed=3)
    repeat = split_subjects(frame.sample(frac=1, random_state=4), seed=3)
    pd.testing.assert_frame_equal(result, repeat)
    assert result.subject_id.is_unique
    assert set(result.subject_id) == set(frame.subject_id)
    assert result.split.value_counts().to_dict() == {"train": 420, "validation": 90, "test": 90}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert set(result.loc[result.split == left, "subject_id"]).isdisjoint(
            result.loc[result.split == right, "subject_id"]
        )
    for _, group in result.groupby("split"):
        assert set(group.acquisition_site) == {"A", "B", "C"}
        assert abs(group.age.mean() - frame.age.mean()) < 3
        assert (group.acquisition_site.value_counts() / len(group) - 1 / 3).abs().max() < 0.05
    assert not result.split.equals(split_subjects(frame, seed=99).split)
    assert sum(item["count"] for item in split_summary(result).values()) == len(frame)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 19])
def test_tiny_strata_fallback(n):
    frame = split_frame(n)
    frame["acquisition_site"] = [f"rare{i}" for i in range(n)]
    result = split_subjects(frame)
    assert len(result) == n
    assert result.subject_id.is_unique
    assert result.split.notna().all()
    assert result.split.isin(["train", "validation", "test"]).all()


def test_split_rejects_ineligible_and_duplicate_subjects():
    frame = split_frame(10)
    frame.loc[0, "brain_age_label_eligible"] = False
    with pytest.raises(ValueError):
        split_subjects(frame)
    frame.loc[0, "brain_age_label_eligible"] = True
    frame.loc[0, "subject_id"] = frame.loc[1, "subject_id"]
    with pytest.raises(ValueError):
        split_subjects(frame)


def test_header_audit_never_reads_voxels_or_writes_volumes(phase2, monkeypatch):
    _, config = phase2

    def forbid(*args, **kwargs):
        raise AssertionError("Header audit must not read voxel arrays")

    monkeypatch.setattr(nib.arrayproxy.ArrayProxy, "__array__", forbid)
    monkeypatch.setattr(nib.arrayproxy.ArrayProxy, "__getitem__", forbid)
    summary = run_preprocessing(config, audit_only=True)
    assert summary["eligible_subjects_seen"] == summary["success_count"] == 8
    assert summary["failure_count"] == 0
    assert summary["canonical_orientation_distribution"] == {"RxAxS": 8}
    assert not list(config.output_dir.rglob("*.nii*"))
    assert (config.output_dir / "spatial_audit/subjects/subject0.json").exists()


def test_pipeline_pilot_provenance_failures_and_source_immutability(phase2, monkeypatch):
    import json

    monkeypatch.setenv("MPLBACKEND", "Agg")
    _, config = phase2
    subjects = eligible_subjects(config.manifest_path)
    pilot = select_pilot(subjects, 3)
    before = {p: Path(p).read_bytes() for p in pilot.t1_path}
    # Corrupt one eligible input: failure must be recorded, others must continue.
    broken = Path(pilot.iloc[0].t1_path)
    broken.write_bytes(b"corrupt synthetic file")
    summary = run_preprocessing(config, limit=3)
    assert summary["eligible_subjects_seen"] == 8
    assert summary["selected_subject_count"] == 3
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["qc_success_count"] == 2
    provenance = list((config.output_dir / "preprocessing/subjects").glob("*.json"))
    assert len(provenance) == 3
    for path in provenance:
        record = json.loads(path.read_text())
        assert {
            "subject_id",
            "source_t1_path",
            "source_shape",
            "source_spacing",
            "source_orientation",
            "canonical_orientation",
            "resampled_shape",
            "target_spacing_mm",
            "normalization_method",
            "lower_percentile",
            "upper_percentile",
            "output_path",
            "preprocessing_status",
            "warnings",
            "errors",
        } <= set(record)
        if record["preprocessing_status"] == "success":
            output = nib.load(record["output_path"])
            assert nib.aff2axcodes(output.affine) == ("R", "A", "S")
            assert output.get_data_dtype() == np.dtype("float32")
            np.testing.assert_allclose(nib.affines.voxel_sizes(output.affine), [1, 1, 1])
            assert Path(record["qc_path"]).is_file()
        else:
            assert record["errors"]
            assert record["output_path"] is None
    for path, content in before.items():
        if Path(path) != broken:
            assert Path(path).read_bytes() == content


def test_full_pipeline_shape_matches_audit_and_protects_existing_outputs(phase2):
    _, config = phase2
    config = replace(config, qc_enabled=False)
    summary = run_preprocessing(config, limit=1)
    assert summary["success_count"] == 1
    original = select_pilot(eligible_subjects(config.manifest_path), 1).iloc[0].to_dict()
    prediction = inspect_subject(original, config)
    path = next((config.output_dir / "volumes").glob("*.nii.gz"))
    assert list(nib.load(path).shape) == prediction["resampled_shape"]
    assert run_preprocessing(config, limit=1)["failure_count"] == 1
    assert run_preprocessing(config, limit=1, overwrite=True)["success_count"] == 1


def test_qc_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    data = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
    image = nib.Nifti1Image(data, np.eye(4))
    output = tmp_path / "qc.png"
    save_comparison(image, image, output, dpi=60)
    assert output.stat().st_size > 0


def test_split_output_is_local_aggregate_and_repeatable(phase2):
    config_path, config = phase2
    summary = run_split(config_path)
    assert summary["eligible_subject_count"] == 8
    file = config.output_dir / "splits/split_manifest.csv"
    original = file.read_bytes()
    run_split(config_path)
    assert file.read_bytes() == original
    assert len(pd.read_csv(file)) == 8
    assert "subject_id" not in str(summary)


@pytest.mark.parametrize(
    "updates",
    [
        {"target_spacing_mm": [1, 2, 1]},
        {"target_spacing_mm": [0, 0, 0]},
        {"target_spacing_mm": [1, 1]},
        {"target_spacing_mm": [True, 1, 1]},
        {"manifest_path": "missing.csv"},
        {"intensity": {"lower_percentile": 100}},
        {"split": {"ratios": [0.8, 0.2, 0.1]}},
        {"split": {"seed": True}},
        {"split": {"age_bin_width": 0}},
        {"qc": {"enabled": "yes"}},
        {"qc": {"dpi": 10}},
        {"unexpected": 1},
    ],
)
def test_phase2_configuration_validation(phase2, updates):
    path, _ = phase2
    raw = yaml.safe_load(path.read_text())
    raw.update(updates)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, TypeError)):
        load_preprocessing_config(path)


def test_output_source_separation_and_tracked_directory_rejection(phase2):
    _, config = phase2
    source = Path(eligible_subjects(config.manifest_path).iloc[0].t1_path)
    with pytest.raises(ValueError):
        validate_output_directory(source.parent / "derived", [source])
    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="ignored"):
        validate_output_directory(repo / "src/derived", [])
    validate_output_directory(repo / "artifacts/preprocessed", [])


def test_intensity_signed_integer_foreground_and_small_scale():
    data = np.array([-32768, 1, 10, 100, 0], dtype=np.int16)
    normalized, stats = normalize_intensity(data)
    assert stats["foreground_voxel_count"] == 4
    assert normalized[-1] == 0
    base = np.arange(1, 121, dtype=np.float64).reshape(4, 5, 6)
    np.testing.assert_allclose(
        normalize_intensity(base)[0], normalize_intensity(base * 1e-12)[0], atol=1e-6
    )


def test_spatial_metadata_rejects_unknown_orientation_and_non_mm_units():
    image = nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("meter")
    with pytest.raises(ValueError, match="millimeter"):
        inspect_orientation(image)
    image.header.set_xyzt_units("mm")
    image.set_qform(None, code=0)
    image.set_sform(None, code=0)
    with pytest.raises(ValueError, match="coded orientation"):
        inspect_orientation(image)


def test_nonfinite_resampling_and_qc_failure_recording(phase2, monkeypatch):
    from brainage_xai.preprocessing.pipeline import preprocess_subject

    _, config = phase2
    subject = select_pilot(eligible_subjects(config.manifest_path), 1).iloc[0].to_dict()

    def fail_qc(*args, **kwargs):
        raise OSError("Synthetic QC failure")

    monkeypatch.setattr("brainage_xai.preprocessing.pipeline.save_comparison", fail_qc)
    record = preprocess_subject(subject, config)
    assert record["preprocessing_status"] == "success"
    assert any(warning.startswith("qc_failed") for warning in record["warnings"])
    image = nib.Nifti1Image(np.full((4, 5, 6), np.nan, dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="Non-finite"):
        resample_t1(image, (1, 1, 1))


def test_non_integer_limit_and_non_numeric_spacing_rejected(phase2):
    path, config = phase2
    with pytest.raises(ValueError, match="integer"):
        select_pilot(eligible_subjects(config.manifest_path), 1.5)
    raw = yaml.safe_load(path.read_text())
    raw["target_spacing_mm"] = ["1", "1", "1"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="numeric"):
        load_preprocessing_config(path)
