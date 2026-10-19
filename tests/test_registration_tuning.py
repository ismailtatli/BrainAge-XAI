"""Synthetic-only controlled rigid tuning and artifact-safety checks."""

from dataclasses import replace

import numpy as np
import pytest
import SimpleITK as sitk

from brainage_xai.config.preprocessing import validate_output_directory
from brainage_xai.registration.rigid import transform_record
from brainage_xai.registration.settings import RigidParameters
from brainage_xai.registration.tuning import (
    CONDITIONS,
    aggregate,
    brain_coverage,
    check_mask,
    comparison_row,
    full_metric,
    initialize_mode,
    tune,
)


@pytest.fixture
def images():
    z, y, x = np.indices((24, 26, 28))
    a = 100 * np.exp(-((x - 12) ** 2 / 25 + (y - 11) ** 2 / 36 + (z - 10) ** 2 / 20))
    a += 40 * np.exp(-((x - 19) ** 2 + (y - 18) ** 2 + (z - 16) ** 2) / 8)
    fixed = sitk.GetImageFromArray(a.astype("float32"))
    mask = sitk.Cast(fixed > 3, sitk.sitkUInt8)
    moving = sitk.Image(fixed)
    moving.SetOrigin((2.0, -1.0, 3.0))
    return fixed, moving, mask


@pytest.fixture
def params():
    return replace(
        RigidParameters(),
        iterations_per_level=8,
        shrink_factors=(2, 1),
        smoothing_sigmas_mm=(1.0, 0.0),
        sampling_fraction=0.4,
    )


@pytest.mark.parametrize("mode", ["GEOMETRY", "MOMENTS"])
def test_initialization(images, mode):
    fixed, moving, mask = images
    tx = initialize_mode(fixed, moving, mode)
    assert np.allclose(tx.GetTranslation(), (2, -1, 3))
    assert np.allclose(tx.GetParameters()[:3], 0)
    assert brain_coverage(fixed, moving, mask, tx)["template_brain_in_bounds_fraction"] == 1
    assert np.isclose(transform_record(tx)["rotation_determinant"], 1)


def test_coverage_pull_and_outside(images):
    fixed, moving, mask = images
    tx = initialize_mode(fixed, moving, "GEOMETRY")
    tx.SetTranslation((200, 0, 0))
    result = brain_coverage(fixed, moving, mask, tx)
    assert result["template_brain_in_bounds_fraction"] == 0
    assert result["template_brain_outside_count"] > 0


def test_mask_physical_compatibility(images):
    fixed, _, mask = images
    check_mask(fixed, mask)
    for method, value in [
        ("SetOrigin", (1, 0, 0)),
        ("SetSpacing", (2, 1, 1)),
        ("SetDirection", (-1, 0, 0, 0, 1, 0, 0, 0, 1)),
    ]:
        wrong = sitk.Image(mask)
        getattr(wrong, method)(value)
        with pytest.raises(ValueError, match="physical grid"):
            check_mask(fixed, wrong)
    with pytest.raises(ValueError):
        check_mask(fixed, sitk.Image([2, 2, 2], sitk.sitkUInt8))


@pytest.mark.parametrize("condition", list(CONDITIONS))
def test_two_stage_repeatability_and_full_metric(images, params, condition):
    fixed, moving, mask = images
    mode, masked = CONDITIONS[condition]
    source = sitk.GetArrayFromImage(moving)
    tx, r = tune(fixed, moving, mask, mode, masked, params)
    repeat, rr = tune(fixed, moving, mask, mode, masked, params)
    assert r == rr
    assert tx.GetParameters() == repeat.GetParameters()
    assert np.array_equal(source, sitk.GetArrayFromImage(moving))
    first, second = r["translation_stage"], r["rigid_stage"]
    assert first["weights"] == [0, 0, 0, 1, 1, 1]
    assert second["weights"] == [1] * 6
    assert first["input_transform"]["parameters"][:3] == first["output_transform"]["parameters"][:3]
    assert first["output_transform"] == second["input_transform"]
    assert first["trajectory"] and second["trajectory"]
    assert len(first["level_stop_conditions"]) == len(first["shrink_factors"])
    info = transform_record(tx)
    assert info["orthonormality_error"] < 1e-6
    assert np.isclose(info["rotation_determinant"], 1)
    actual = full_metric(fixed, moving, tx, mask if masked else None, params)
    assert actual["MI"] == r["MI_final"]
    assert (
        full_metric(fixed, moving, tx, mask if masked else None, replace(params, sampling_seed=19))
        == actual
    )
    r.update(
        subject_id="synthetic",
        site="synthetic",
        condition=condition,
        success=True,
        technical_warnings=[],
        output_geometry_valid=True,
    )
    row = comparison_row(r)
    assert row["translation_stage_iterations"] == first["iterations"]
    assert row["MI_after_translation"] == r["MI_after_translation"]
    assert aggregate([r])["execution_success_count"] == 1


def test_source_and_unignored_output_rejected(tmp_path):
    source = tmp_path / "source" / "synthetic.nii.gz"
    source.parent.mkdir()
    source.write_bytes(b"synthetic")
    with pytest.raises(ValueError):
        validate_output_directory(source.parent / "output", [source])
    with pytest.raises(ValueError):
        validate_output_directory(tmp_path, [source])
    from pathlib import Path

    with pytest.raises(ValueError):
        validate_output_directory(Path.cwd() / "unsafe_tuning_output", [source])


def test_moments_can_differ_from_geometry(images):
    fixed, moving, _ = images
    data = sitk.GetArrayFromImage(moving)
    data[1:5, 1:5, 1:5] += 1000  # Synthetic high-intensity neck/scalp-like bias.
    biased = sitk.GetImageFromArray(data)
    biased.CopyInformation(moving)
    geometry = initialize_mode(fixed, biased, "GEOMETRY")
    moments = initialize_mode(fixed, biased, "MOMENTS")
    assert not np.allclose(geometry.GetTranslation(), moments.GetTranslation())


def test_partial_coverage_uses_physical_grid_and_pull_direction():
    fixed = sitk.Image([6, 6, 6], sitk.sitkFloat32)
    fixed.SetSpacing((2, 3, 4))
    fixed.SetOrigin((7, -11, 5))
    moving = sitk.Image(fixed)
    mask = sitk.Image([6, 6, 6], sitk.sitkUInt8) + 1
    mask.CopyInformation(fixed)
    tx = sitk.Euler3DTransform()
    tx.SetTranslation((6, 0, 0))  # Three source voxels, not six.
    result = brain_coverage(fixed, moving, mask, tx)
    assert result["template_brain_inside_count"] == 3 * 6 * 6
    assert result["template_brain_in_bounds_fraction"] == 0.5


def test_baseline_table_has_no_translation_stage():
    ev = {"template_brain_in_bounds_fraction": 0.99}
    record = {"subject_id": "synthetic", "evaluations": [ev, None, ev]}
    row = comparison_row(record)
    assert row["MI_after_translation"] is None
    assert row["template_brain_in_bounds_fraction_after_translation"] is None
    assert row["translation_stage_iterations"] is None


def test_private_summary_and_initialization_analysis(tmp_path):
    import json

    from brainage_xai.registration.tuning import finalize_tables

    output = tmp_path / "private"
    (output / "subjects").mkdir(parents=True)
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "parameters": {
                    "initialization": "GEOMETRY",
                    "use_fixed_mask": True,
                    "sampling_seed": 42,
                }
            }
        )
    )
    transform = transform_record(sitk.Euler3DTransform())
    for subject in range(18):
        for condition in ("v1", *CONDITIONS):
            mode, masked = CONDITIONS.get(condition, ("GEOMETRY", True))
            record = {
                "subject_id": f"synthetic{subject}",
                "condition": condition,
                "site": ("Guys", "HH", "IOP")[subject // 6],
                "success": True,
                "initialization_mode": mode,
                "fixed_mask_used": masked,
                "technical_warnings": ["metric_degradation_on_common_evaluation_lattice"]
                if condition == "v1"
                else [],
                "MI_initial": -0.1,
                "MI_final": -0.2,
                "within_condition_MI_change": 0.1,
                "evaluations": [{"template_brain_in_bounds_fraction": 1}] * 3,
                "initial_transform": transform,
                "final_transform": transform,
            }
            (output / "subjects" / f"synthetic{subject}_{condition}.json").write_text(
                json.dumps(record)
            )
    summary = finalize_tables(output)
    assert summary["A"]["overall"]["execution_success_count"] == 18
    assert summary["D"]["by_site"]["IOP"]["MI_improvement_count"] == 6
    assert summary["v1"]["overall"]["technical_warning_count"] == 0
    assert summary["initialization_analysis"]["C"]["mode"] == "MOMENTS"
    assert len((output / "comparison.csv").read_text().splitlines()) == 91
    assert not list(output.rglob("*.nii*"))
    baseline = output / "subjects" / "synthetic0_v1.json"
    record = json.loads(baseline.read_text())
    record.update(MI_final=-0.05, within_condition_MI_change=-0.05)
    baseline.write_text(json.dumps(record))
    first = finalize_tables(output)
    second = finalize_tables(output)
    assert first["v1"]["overall"]["technical_warning_count"] == 1
    assert first == second
    invalid = output / "subjects" / "synthetic0_C.json"
    record = json.loads(invalid.read_text())
    record["initial_transform"]["parameters"][3] += 1
    invalid.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="experiment invalid"):
        finalize_tables(output)


@pytest.mark.parametrize("masked", [False, True])
def test_full_metric_prefers_known_phantom_correspondence(images, params, masked):
    fixed, moving, mask = images
    correct = initialize_mode(fixed, moving, "GEOMETRY")
    wrong = sitk.Euler3DTransform(correct)
    wrong.SetTranslation((7, 4, -1))
    roi = mask if masked else None
    assert (
        full_metric(fixed, moving, correct, roi, params)["MI"]
        < full_metric(fixed, moving, wrong, roi, params)["MI"]
    )


def test_rotated_coverage_matches_independent_point_queries():
    fixed = sitk.Image([5, 6, 7], sitk.sitkFloat32)
    fixed.SetOrigin((-3, 4, -5))
    fixed.SetSpacing((2, 3, 4))
    moving = sitk.Image(fixed)
    mask = sitk.Image([5, 6, 7], sitk.sitkUInt8) + 1
    mask.CopyInformation(fixed)
    tx = sitk.Euler3DTransform()
    tx.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint((2, 2.5, 3)))
    tx.SetRotation(0.2, -0.3, 0.1)
    tx.SetTranslation((2, -3, 4))
    expected = 0
    for index in np.ndindex(fixed.GetSize()):
        point = tx.TransformPoint(fixed.TransformIndexToPhysicalPoint(index))
        continuous = moving.TransformPhysicalPointToContinuousIndex(point)
        expected += all(-0.5 <= v < n - 0.5 for v, n in zip(continuous, moving.GetSize()))
    assert brain_coverage(fixed, moving, mask, tx)["template_brain_inside_count"] == expected


def test_moments_uses_full_source_after_qc_requested_subregion():
    fixed = sitk.GetImageFromArray(np.ones((8, 9, 10), np.float32))
    fixed.SetOrigin((5, 5, 5))
    data = np.ones((24, 26, 28), np.float32)
    data[15:20, 16:21, 17:22] = 40
    moving = sitk.GetImageFromArray(data)
    expected = initialize_mode(fixed, moving, "MOMENTS")
    # ITK resampling leaves the source requested-region set to its required subset.
    sitk.Resample(moving, fixed, sitk.Euler3DTransform(), sitk.sitkLinear)
    actual = initialize_mode(fixed, moving, "MOMENTS")
    assert actual.GetParameters() == expected.GetParameters()
    assert actual.GetFixedParameters() == expected.GetFixedParameters()
    assert np.array_equal(sitk.GetArrayFromImage(moving), data)
