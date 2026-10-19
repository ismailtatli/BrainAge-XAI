"""Synthetic full-grid production, split and safe-resume regression tests."""

import json
from concurrent.futures import Future
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import yaml

from brainage_xai.preprocessing import final
from brainage_xai.preprocessing.final_qc import select_qc
from brainage_xai.registration.rigid import to_sitk, transform_record
from brainage_xai.registration.settings import RigidParameters
from brainage_xai.registration.template import TEMPLATE_AFFINE, TEMPLATE_SHAPE, sha256


def membership():
    cohort = pd.DataFrame(
        [
            dict(
                subject_id=f"synthetic{i}",
                t1_path=f"synthetic{i}.nii.gz",
                age=30 + i,
                acquisition_site="Guys",
                brain_age_label_eligible=True,
            )
            for i in range(3)
        ]
    )
    splits = cohort.drop(columns="brain_age_label_eligible").assign(
        split=["train", "validation", "test"]
    )
    return cohort, splits


def test_existing_subject_membership_is_consumed_without_split_fitting():
    cohort, splits = membership()
    expected = dict(train=1, validation=1, test=1)
    combined = final.validate_membership(cohort, splits, expected)
    assert combined.split.tolist() == splits.split.tolist()
    for broken in [
        pd.concat([splits, splits.iloc[:1]]),
        splits.iloc[:-1],
        splits.assign(acquisition_site="HH"),
        splits.assign(split="train"),
    ]:
        with pytest.raises(ValueError):
            final.validate_membership(cohort, broken, expected)


@pytest.fixture
def volume(tmp_path, monkeypatch):
    data = np.zeros(TEMPLATE_SHAPE, dtype=np.float32)
    data[80:112, 100:132, 80:112] = np.random.default_rng(42).normal(100, 15, (32, 32, 32))
    image = nib.Nifti1Image(data, TEMPLATE_AFFINE)
    image.header.set_xyzt_units("mm")
    (tmp_path / "inputs").mkdir()
    source = tmp_path / "inputs" / "synthetic.nii.gz"
    nib.save(image, source)
    fixed = to_sitk(image)
    p = RigidParameters(iterations_per_level=300, evaluation_shrink_factor=1, use_fixed_mask=False)
    monkeypatch.setattr(final, "_CONTEXT", (SimpleNamespace(image=image), fixed, None, p))
    calls = []

    def frozen_registration(f, m, mask, parameters, key):
        calls.append(key)
        assert parameters == p and key == "O0"
        tx = sitk.Euler3DTransform()
        tx.SetComputeZYX(True)
        info = transform_record(tx)
        coverage = dict(template_brain_in_bounds_fraction=1.0, template_brain_outside_count=0)
        return (
            tx,
            {
                "final_transform": info,
                "evaluations": [coverage] * 3,
                "translation_stage": {"budget_hit": False},
                "rigid_stage": {"budget_hit": False},
                "delta_total": 0.2,
            },
            [],
        )

    monkeypatch.setattr(final.optimizers, "register", frozen_registration)
    subject = dict(
        subject_id="synthetic", t1_path=str(source), acquisition_site="Guys", split="train"
    )
    output = tmp_path / "new"
    for name in ("volumes", "provenance", "work"):
        (output / name).mkdir(parents=True)
    return subject, source, output, calls, p


def test_full_grid_normalization_validation_and_finished_outputs_protected(volume):
    subject, source, output, calls, _ = volume
    before = source.read_bytes()
    record = final.process_subject(subject, output, sha256(source), "synthetic-fingerprint")
    assert record["status"] == "success", record
    assert calls == ["O0"] and source.read_bytes() == before
    image = nib.load(record["output_path"])
    assert image.shape == TEMPLATE_SHAPE
    np.testing.assert_allclose(image.affine, TEMPLATE_AFFINE, rtol=0, atol=1e-6)
    assert image.header.get_zooms() == (1, 1, 1)
    assert record["normalization"]["background_preserved"]
    assert record["normalization"]["background_voxel_count"] > 0
    assert abs(record["normalization"]["normalized_foreground_mean"]) < 1e-6
    assert abs(record["normalization"]["normalized_foreground_std"] - 1) < 1e-6
    final.validate_finished(subject, output, sha256(source), "synthetic-fingerprint")
    saved = (output / "provenance/synthetic.json").read_bytes()
    with pytest.raises(FileExistsError):
        final.process_subject(subject, output, sha256(source), "synthetic-fingerprint")
    assert saved == (output / "provenance/synthetic.json").read_bytes()
    assert calls == ["O0"]
    with pytest.raises(ValueError):
        final.validate_finished(subject, output, sha256(source), "different-config")
    Path = type(output)
    Path(record["output_path"]).write_bytes(b"corrupt synthetic output")
    with pytest.raises(ValueError):
        final.validate_finished(subject, output, sha256(source), "synthetic-fingerprint")


def test_failure_stage_is_explicit_and_membership_retained(volume, monkeypatch):
    subject, source, output, _, _ = volume

    def broken_normalization(*args):
        raise ValueError("synthetic normalization failure")

    monkeypatch.setattr(final, "normalize_intensity", broken_normalization)
    r = final.process_subject(subject, output, sha256(source), "fingerprint")
    assert r["status"] == "failed" and r["stage"] == "normalization"
    assert "synthetic normalization failure" in r["exception"]
    assert (
        json.loads((output / "provenance/synthetic.json").read_text())["subject_id"] == "synthetic"
    )
    summary = final.summary_records([r], 1)
    assert summary["failure_count"] == 1 and summary["by_split"]["train"]["expected"] == 1


def test_safe_resume_skips_valid_and_regenerates_only_explicit_invalid(volume, monkeypatch):
    subject, source, _, calls, p = volume
    output = source.parent.parent / "run"
    cohort = pd.DataFrame([dict(subject, age=30, brain_age_label_eligible=True)])
    spec_path = source.parent / "spec.yaml"
    spec_path.write_text("synthetic: true")
    monkeypatch.setattr(
        final, "load_spec", lambda _: ({}, {"template": source, "mask": source}, cohort, p)
    )

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future

    monkeypatch.setattr(final, "ProcessPoolExecutor", Pool)
    first = final.run(spec_path, output)
    assert first["success_count"] == 1 and calls == ["O0"]
    with pytest.raises(ValueError, match="resume"):
        final.run(spec_path, output)
    repeat = final.run(spec_path, output, resume=True)
    assert repeat["run"]["reused_valid_count"] == 1 and calls == ["O0"]
    path = output / "volumes/synthetic.nii.gz"
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="regenerate-invalid"):
        final.run(spec_path, output, resume=True)
    assert path.read_bytes() == b"broken" and calls == ["O0"]
    recovered = final.run(spec_path, output, resume=True, regenerate_invalid=True)
    assert recovered["success_count"] == 1 and calls == ["O0", "O0"]
    assert len(list((output / "recovery").glob("*/*/*.nii.gz"))) == 1
    final.run(spec_path, output, resume=True, verify_only=True)
    assert calls == ["O0", "O0"]
    source.write_bytes(b"modified synthetic source")
    with pytest.raises(ValueError, match="inputs"):
        final.run(spec_path, output, resume=True)


def test_qc_sample_has_obliquity_extremes_and_challenging_examples():
    cohort = pd.DataFrame(
        [
            dict(subject_id=f"synthetic{site}{i}", acquisition_site=site)
            for site in ["Guys", "HH", "IOP"]
            for i in range(10)
        ]
    )
    geometry = cohort.assign(max_obliquity_deg=[*range(10)] * 3)
    challenges = {
        site: [f"synthetic{site}2", f"synthetic{site}3"] for site in ["Guys", "HH", "IOP"]
    }
    sample = select_qc(cohort, geometry, challenges)
    assert len(sample) == sample.subject_id.nunique() == 18
    for site, group in sample.groupby("acquisition_site"):
        assert len(group) == 6
        assert {0, 5, 9}.issubset(set(group.max_obliquity_deg))
        assert set(challenges[site]).issubset(set(group.subject_id))


def test_frozen_spec_is_participant_free_and_full_template_grid():
    from pathlib import Path

    text = Path("configs/preprocessing.frozen.yaml").read_text()
    spec = yaml.safe_load(text)
    assert "/Users/" not in text
    assert spec["geometry"]["shape_xyz"] == list(TEMPLATE_SHAPE)
    assert not spec["geometry"]["crop"] and not spec["geometry"]["pad"]
    assert not spec["registration"]["affine"] and not spec["registration"]["nonlinear"]
    assert spec["split"]["expected_counts"] == dict(train=393, validation=84, test=84)
    assert spec["intensity"]["fit_scope"].startswith("per subject")
    assert spec["cohort"]["recompute_eligibility"] is False
