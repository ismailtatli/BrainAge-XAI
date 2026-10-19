"""Synthetic-only tests for rigid transforms, local resources, selection and private QC."""

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import yaml

from brainage_xai.registration.pipeline import register_subject, run_registration
from brainage_xai.registration.rigid import (
    from_sitk,
    register_rigid,
    resample_to_fixed,
    source_foreground_clipping,
    to_sitk,
    transform_record,
)
from brainage_xai.registration.selection import select_registration_pilot
from brainage_xai.registration.settings import (
    RegistrationConfig,
    RigidParameters,
    load_registration_config,
)
from brainage_xai.registration.template import (
    TEMPLATE_AFFINE,
    TEMPLATE_ID,
    TEMPLATE_SHAPE,
    TemplateBundle,
    load_template,
    sha256,
    validate_grid,
)


@pytest.fixture(scope="module")
def phantom():
    xyz = np.indices((40, 44, 36)).astype(float)
    data = np.zeros((40, 44, 36))
    for center, sigma, amplitude in [
        ((16, 19, 15), (6, 8, 7), 100),
        ((25, 28, 20), (3, 4, 3), 80),
        ((12, 30, 11), (2, 3, 2), 60),
        ((27, 12, 26), (3, 2, 3), 90),
    ]:
        data += amplitude * np.exp(
            -sum(((xyz[i] - center[i]) / sigma[i]) ** 2 for i in range(3)) / 2
        )
    image = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    image.header.set_xyzt_units("mm")
    return image


@pytest.fixture
def parameters():
    return replace(
        RigidParameters(),
        sampling_strategy="NONE",
        use_fixed_mask=False,
        shrink_factors=(2, 1),
        smoothing_sigmas_mm=(1.0, 0.0),
        evaluation_shrink_factor=1,
    )


def known_transform(fixed, rotation_deg=0, translation=(0, 0, 0)):
    tx = sitk.Euler3DTransform()
    tx.SetComputeZYX(True)
    tx.SetCenter(
        fixed.TransformContinuousIndexToPhysicalPoint(tuple((np.array(fixed.GetSize()) - 1) / 2))
    )
    tx.SetRotation(0, 0, np.radians(rotation_deg))
    tx.SetTranslation(translation)
    return tx


@pytest.mark.parametrize(
    "rotation,translation", [(0, (0, 0, 0)), (0, (2, -1.5, 1)), (5, (0, 0, 0))]
)
def test_identity_translation_rotation_recovery(phantom, parameters, rotation, translation):
    fixed = to_sitk(phantom)
    truth = known_transform(fixed, rotation, translation)
    moving = sitk.Resample(fixed, fixed, truth.GetInverse(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    final, info = register_rigid(fixed, moving, parameters)
    np.testing.assert_allclose(final.GetTranslation(), translation, atol=0.3)
    np.testing.assert_allclose(np.degrees(final.GetParameters()[:3]), [0, 0, rotation], atol=0.8)
    if rotation or any(translation):
        assert info["metric_after"] < info["metric_before"]
    else:
        # Multi-resolution MI need not improve an already-correct identity alignment.
        assert abs(info["metric_after"] - info["metric_before"]) < 0.03
    assert info["iteration_count"] >= 0
    assert info["initial_transform"]["direction"].startswith("fixed/template LPS")
    output = resample_to_fixed(moving, fixed, final)
    assert output.GetSize() == fixed.GetSize()
    assert output.GetOrigin() == fixed.GetOrigin()
    assert output.GetDirection() == fixed.GetDirection()


def test_seeded_registration_is_deterministic(phantom, parameters):
    fixed = to_sitk(phantom)
    truth = known_transform(fixed, 4, (1, 2, -1))
    moving = sitk.Resample(fixed, fixed, truth.GetInverse(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    params = replace(
        parameters, sampling_strategy="RANDOM", sampling_fraction=0.2, iterations_per_level=80
    )
    first, one = register_rigid(fixed, moving, params)
    second, two = register_rigid(fixed, moving, params)
    np.testing.assert_array_equal(first.GetParameters(), second.GetParameters())
    assert one["metric_after"] == two["metric_after"]
    assert one["iterations_per_level_actual"] == two["iterations_per_level_actual"]


def test_rigid_sanity_no_scale_or_shear():
    tx = sitk.Euler3DTransform()
    tx.SetComputeZYX(True)
    tx.SetCenter((10.0, -20.0, 30.0))
    tx.SetRotation(0.2, -0.1, 0.3)
    tx.SetTranslation((2.0, 3.0, 4.0))
    info = transform_record(tx)
    matrix = np.array(info["matrix_lps"])
    np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1)
    np.testing.assert_allclose(info["rotation_singular_values"], [1, 1, 1], atol=1e-12)
    point = np.array([11.0, 20.0, 30.0])
    np.testing.assert_allclose((matrix @ np.r_[point, 1])[:3], tx.TransformPoint(tuple(point)))
    with pytest.raises(ValueError):
        transform_record(sitk.AffineTransform(3))
    tx.SetTranslation((np.nan, 0, 0))
    with pytest.raises(ValueError):
        transform_record(tx)


def test_ras_lps_conversion_preserves_physical_points_including_tiny_shear(phantom):
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    affine[:3, 3] = [10, -20, 30]
    affine[0, 1] = 1e-5
    image = nib.Nifti1Image(np.asarray(phantom.dataobj), affine)
    converted = to_sitk(image)
    index = (3, 5, 7)
    ras = nib.affines.apply_affine(affine, index)
    np.testing.assert_allclose(
        converted.TransformIndexToPhysicalPoint(index), ras * [-1, -1, 1], atol=1e-10
    )
    np.testing.assert_allclose(sitk.GetArrayFromImage(converted).transpose(2, 1, 0), image.dataobj)


def test_template_grid_validation_synthetic_full_shape_and_missing_resources(tmp_path, monkeypatch):
    # Same expected geometry, synthetic constant intensities; never a participant fixture.
    image = nib.Nifti1Image(np.ones(TEMPLATE_SHAPE, dtype=np.float32), TEMPLATE_AFFINE)
    image.header.set_xyzt_units("mm")
    validate_grid(image)
    bad = nib.Nifti1Image(np.ones((10, 11, 12), dtype=np.float32), np.eye(4))
    bad.header.set_xyzt_units("mm")
    with pytest.raises(ValueError, match="incompatible"):
        validate_grid(bad)
    with pytest.raises(ValueError, match="Missing local"):
        load_template(tmp_path / "missing.nii.gz", None, use_mask=False)
    path = tmp_path / "synthetic_template.nii.gz"
    nib.save(image, path)
    with pytest.raises(ValueError, match="checksum"):
        load_template(path, None, use_mask=False)
    monkeypatch.setattr("brainage_xai.registration.template.T1_SHA256", sha256(path))
    bundle = load_template(path, None, use_mask=False)
    assert bundle.provenance["template"]["shape"] == list(TEMPLATE_SHAPE)
    assert bundle.provenance["template"]["sha256"] == sha256(path)
    with pytest.raises(ValueError, match="mask"):
        load_template(path, None, use_mask=True)


def selection_inputs(n=12):
    eligible = []
    geometry = []
    for site in ("Guys", "HH", "IOP"):
        for i in range(n):
            sid = f"{site}_{i:03d}"
            eligible.append(
                {
                    "subject_id": sid,
                    "age": 30 + i,
                    "acquisition_site": site,
                    "brain_age_label_eligible": True,
                    "t1_path": f"{sid}.nii.gz",
                }
            )
            geometry.append(
                {
                    "subject_id": sid,
                    "max_obliquity_deg": float(i),
                    "foreground_center_offset_mm": json.dumps([0, 0, 20 if i == 6 else 0]),
                    "diagnostic_status": "success",
                }
            )
    return pd.DataFrame(eligible), pd.DataFrame(geometry)


def test_pilot_site_angles_offset_tails_unique_deterministic():
    eligible, geometry = selection_inputs()
    pilot = select_registration_pilot(eligible, geometry)
    assert len(pilot) == 18 and pilot.subject_id.is_unique
    assert pilot.acquisition_site.value_counts().to_dict() == {"Guys": 6, "HH": 6, "IOP": 6}
    assert pilot.groupby(["acquisition_site", "selection_stratum"]).size().eq(2).all()
    assert len(pilot.loc[pilot.center_offset_norm_mm == 20]) == 3
    pd.testing.assert_frame_equal(
        pilot,
        select_registration_pilot(
            eligible.sample(frac=1, random_state=4), geometry.sample(frac=1, random_state=5)
        ),
    )
    for _, group in pilot.groupby("acquisition_site"):
        assert group.max_obliquity_deg.min() <= 1
        assert group.max_obliquity_deg.max() == 11


def test_selection_missing_geometry_and_small_sites_fail():
    eligible, geometry = selection_inputs(6)
    with pytest.raises(ValueError):
        select_registration_pilot(eligible, geometry.iloc[1:])
    with pytest.raises(ValueError):
        select_registration_pilot(eligible.iloc[1:], geometry)
    with pytest.raises(ValueError):
        select_registration_pilot(pd.concat([eligible, eligible.iloc[:1]]), geometry)


@pytest.fixture
def synthetic_run(tmp_path, phantom, parameters):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest_dir = tmp_path / "inputs"
    manifest_dir.mkdir()
    eligible, geometry = selection_inputs(6)
    for index, row in eligible.iterrows():
        path = raw / f"{row.subject_id}.nii.gz"
        nib.save(phantom, path)
        eligible.loc[index, "t1_path"] = str(path)
    manifest = manifest_dir / "manifest.csv"
    eligible.to_csv(manifest, index=False)
    geom = manifest_dir / "geometry.csv"
    geometry.to_csv(geom, index=False)
    template = tmp_path / "template"
    template.mkdir()
    fixed_path = template / "fixed.nii.gz"
    nib.save(phantom, fixed_path)
    config = RegistrationConfig(
        manifest,
        geom,
        tmp_path / "artifacts",
        fixed_path,
        None,
        parameters,
        qc_dpi=50,
        checkerboard_pixels=8,
    )
    bundle = TemplateBundle(
        phantom,
        None,
        {"id": TEMPLATE_ID, "resolution": "01", "template": {"sha256": sha256(fixed_path)}},
    )
    subject = select_registration_pilot(eligible, geometry).iloc[0].to_dict()
    return config, bundle, subject


def test_subject_output_provenance_intensity_qc_and_no_source_overwrite(synthetic_run):
    config, bundle, subject = synthetic_run
    config.output_dir.mkdir()
    before = Path(subject["t1_path"]).read_bytes()
    record = register_subject(subject, config, bundle, to_sitk(bundle.image), None)
    assert record["registration_success"], record["errors"]
    assert record["registration_status"] == "completed_qc_review_required"
    assert record["anatomical_qc_status"] == "not_reviewed"
    assert record["output_geometry_verified"]
    if record["metric_improvement"] < 0:
        assert "metric_degradation_on_common_evaluation_lattice" in record["warnings"]
    assert record["final_interpolation_steps_from_native"] == 1
    assert record["foreground_after"]["resampled_shape"] == list(bundle.image.shape)
    output = nib.load(record["output_path"])
    assert output.shape == bundle.image.shape
    np.testing.assert_allclose(output.affine, bundle.image.affine)
    assert Path(record["qc_path"]).stat().st_size > 0
    from brainage_xai.registration.pipeline import pilot_summary

    summary = pilot_summary([record])
    assert (
        summary["template_size_centered_crop_diagnostic"]["registered"]["foreground_loss_any_count"]
        == 0
    )
    assert "p99" in summary["intensity_registered"]["foreground_percentiles"]
    assert Path(subject["t1_path"]).read_bytes() == before
    unsafe = replace(config, output_dir=Path(subject["t1_path"]).parent)
    failed = register_subject(subject, unsafe, bundle, to_sitk(bundle.image), None)
    assert not failed["registration_success"]
    assert Path(subject["t1_path"]).read_bytes() == before


def test_subject_optimizer_failure_is_recorded(synthetic_run, monkeypatch):
    config, bundle, subject = synthetic_run
    config.output_dir.mkdir()

    def fail(*args, **kwargs):
        kwargs["diagnostics"]["initial_transform"] = {"synthetic": "initial"}
        raise RuntimeError("Synthetic optimizer failure")

    monkeypatch.setattr("brainage_xai.registration.pipeline.register_rigid", fail)
    record = register_subject(subject, config, bundle, to_sitk(bundle.image), None)
    assert not record["registration_success"]
    assert record["initial_transform"] == {"synthetic": "initial"}
    assert "optimizer failure" in record["errors"][0]
    assert record["output_path"] is None


def test_driver_keeps_all_eighteen_failures_and_private_provenance(synthetic_run, monkeypatch):
    config, bundle, _ = synthetic_run
    monkeypatch.setattr("brainage_xai.registration.pipeline.load_template", lambda *a, **k: bundle)

    def failure(subject, *args):
        return {
            "subject_id": subject["subject_id"],
            "acquisition_site": subject["acquisition_site"],
            "registration_success": False,
            "registration_status": "failed",
            "warnings": [],
            "errors": ["synthetic failure"],
            "qc_path": None,
        }

    monkeypatch.setattr("brainage_xai.registration.pipeline.register_subject", failure)
    summary = run_registration(config)
    assert summary["subject_count"] == summary["failure_count"] == 18
    assert len(list((config.output_dir / "subjects").glob("*.json"))) == 18
    assert len(pd.read_csv(config.output_dir / "manifest.csv")) == 18
    with pytest.raises(FileExistsError):
        run_registration(config)


def test_clipping_count_and_template_output(phantom):
    identity = sitk.Euler3DTransform()
    record = source_foreground_clipping(
        np.ones((4, 5, 6)), np.eye(4), (4, 5, 6), np.eye(4), identity
    )
    assert record["outside_template_fov_fraction"] == 0
    shifted = sitk.Euler3DTransform()
    shifted.SetTranslation((100.0, 0, 0))
    assert (
        source_foreground_clipping(np.ones((4, 5, 6)), np.eye(4), (4, 5, 6), np.eye(4), shifted)[
            "outside_template_fov_fraction"
        ]
        == 1
    )
    converted = from_sitk(to_sitk(phantom), phantom)
    np.testing.assert_allclose(converted.affine, phantom.affine)


@pytest.mark.parametrize(
    "updates",
    [
        {"threads": 2},
        {"sampling_seed": 0},
        {"sampling_fraction": 2},
        {"initialization": "MOMENTS"},
        {"shrink_factors": (1, 2)},
        {"smoothing_sigmas_mm": (2, -1, 0)},
        {"histogram_bins": True},
        {"iterations_per_level": 0},
        {"rigid_tolerance": -1},
    ],
)
def test_invalid_parameters(updates):
    with pytest.raises(ValueError):
        replace(RigidParameters(), **updates).validate()


def test_registration_configuration_validation(synthetic_run, tmp_path):
    config, _, _ = synthetic_run
    raw = {
        "manifest_path": str(config.manifest_path),
        "geometry_manifest_path": str(config.geometry_manifest_path),
        "output_dir": str(config.output_dir),
        "template": {"id": TEMPLATE_ID, "resolution": "01", "t1_path": str(config.template_path)},
        "registration": {"use_fixed_mask": False},
    }
    path = tmp_path / "registration.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_registration_config(path).subjects_per_site == 6
    raw["template"]["id"] = "wrong_template"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_registration_config(path)


def test_engineering_outlier_flags_and_centered_crop_counts():
    from brainage_xai.registration.pipeline import _crop_coverage, qc_warning_flags

    parameters = RigidParameters()
    trace = {
        "metric_improvement": -1.0,
        "optimizer_stop_condition": "Maximum number of iterations exceeded",
        "final_transform": {
            "rotation_deg_lps_xyz": [40, 0, 0],
            "translation_mm_lps_xyz": [150, 0, 0],
        },
    }
    flags = qc_warning_flags(trace, 0.3, parameters)
    assert len(flags) == 5
    assert "provisional_large_rotation" in flags
    assert "provisional_large_translation" in flags
    assert "provisional_source_nonbackground_fov_clipping_not_brain_loss" in flags
    extents = [
        {
            "resampled_shape": [40, 40, 40],
            "foreground_bbox_start": [0, 10, 10],
            "foreground_bbox_end_exclusive": [10, 20, 20],
        }
    ]
    assert _crop_coverage(extents, [20, 40, 40])["foreground_loss_axis_counts"] == [1, 0, 0]


def test_synthetic_masked_metric_and_grid_error_recording(
    phantom, parameters, synthetic_run, monkeypatch
):
    fixed = to_sitk(phantom)
    mask = sitk.Cast(fixed > 10, sitk.sitkUInt8)
    final, info = register_rigid(
        fixed, fixed, replace(parameters, use_fixed_mask=True, iterations_per_level=20), mask
    )
    assert np.isfinite(info["metric_after"])
    assert transform_record(final)["rotation_determinant"] == pytest.approx(1)
    config, bundle, subject = synthetic_run
    config.output_dir.mkdir()
    bad = sitk.Image([2, 3, 4], sitk.sitkFloat32)
    monkeypatch.setattr("brainage_xai.registration.pipeline.resample_to_fixed", lambda *a: bad)
    record = register_subject(subject, config, bundle, to_sitk(bundle.image), None)
    assert not record["registration_success"]
    assert "Unexpected output geometry" in record["errors"][0]
    assert record["output_path"] is None
