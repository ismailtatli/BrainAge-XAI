"""Synthetic affine mapping, diagnostics and bounded experiment regression checks."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from brainage_xai.registration import affine, affine_study, optimizers
from brainage_xai.registration.optimizer_study import run_case
from brainage_xai.registration.pipeline import _euler
from brainage_xai.registration.rigid import resample_to_fixed
from brainage_xai.registration.settings import RigidParameters
from brainage_xai.registration.tuning import brain_coverage, full_metric


def test_centered_rigid_to_affine_preserves_points():
    rigid = sitk.Euler3DTransform()
    rigid.SetComputeZYX(True)
    rigid.SetCenter((11, -13, 4))
    rigid.SetRotation(0.12, -0.08, 0.2)
    rigid.SetTranslation((3, 5, -7))
    tx = affine.rigid_to_affine(rigid)
    for point in ((0, 0, 0), (11, -13, 4), (81, 44, -50), (-70, -80, 72)):
        np.testing.assert_allclose(
            tx.TransformPoint(point), rigid.TransformPoint(point), atol=1e-12
        )
    assert tx.GetNumberOfParameters() == 12
    assert tx.GetCenter() == rigid.GetCenter()
    assert affine.affine_record(tx)["determinant"] == pytest.approx(1)


def test_known_scale_and_shear_diagnostics():
    rigid = sitk.Euler3DTransform()
    rigid.SetRotation(0.1, -0.2, 0.3)
    rotation = np.asarray(rigid.GetMatrix()).reshape(3, 3)
    tx = sitk.AffineTransform(3)
    tx.SetMatrix((rotation @ np.diag([0.9, 1.1, 1.2])).ravel())
    r = affine.affine_record(tx)
    np.testing.assert_allclose(r["singular_values"], [1.2, 1.1, 0.9])
    assert r["determinant"] == pytest.approx(0.9 * 1.1 * 1.2)
    assert r["global_isotropic_pull_scale"] ** 3 == pytest.approx(r["determinant"])
    assert r["inverse_moving_to_fixed_volume_factor"] == pytest.approx(1 / r["determinant"])
    assert r["max_absolute_column_cosine"] < 1e-12
    np.testing.assert_allclose(r["polar_rotation_matrix"], rotation, atol=1e-12)
    tx.SetMatrix((1, 0.3, 0, 0, 1, 0, 0, 0, 1))
    r = affine.affine_record(tx)
    assert r["determinant"] == pytest.approx(1)
    assert r["max_absolute_column_cosine"] > 0.15
    assert r["max_absolute_normalized_polar_shear"] > 0.1
    assert "unusual_nonorthogonality" in r["review_warnings"]
    tx.SetMatrix(np.diag([0.7, 1.0, 1.0]).ravel())
    assert "unusual_principal_scale" in affine.affine_record(tx)["review_warnings"]


@pytest.mark.parametrize(
    "matrix", [np.diag([0, 1, 1]), np.diag([-1, 1, 1]), np.diag([np.nan, 1, 1])]
)
def test_invalid_affine_rejected(matrix):
    tx = sitk.AffineTransform(3)
    tx.SetMatrix(matrix.astype(float).ravel())
    with pytest.raises(ValueError):
        affine.affine_record(tx)


@pytest.fixture
def synthetic(tmp_path):
    z, y, x = np.indices((28, 30, 32))
    data = 100 * np.exp(-((x - 14) ** 2 / 30 + (y - 13) ** 2 / 40 + (z - 12) ** 2 / 25))
    data += 40 * np.exp(-((x - 22) ** 2 + (y - 20) ** 2 + (z - 18) ** 2) / 10)
    fixed = sitk.GetImageFromArray(data.astype(np.float32))
    moving = sitk.GetImageFromArray(np.roll(data, 2, axis=2).astype(np.float32))
    moving.SetOrigin((2, -1, 3))
    mask = sitk.Cast(fixed > 3, sitk.sitkUInt8)
    p = replace(
        RigidParameters(),
        iterations_per_level=8,
        shrink_factors=(2, 1),
        smoothing_sigmas_mm=(1.0, 0.0),
        evaluation_shrink_factor=1,
        use_fixed_mask=False,
    )
    (tmp_path / "traces").mkdir()
    rigid, _ = run_case(
        {"subject_id": "synthetic", "acquisition_site": "Guys"},
        "O0",
        fixed,
        moving,
        mask,
        p,
        tmp_path,
    )
    assert rigid["success"]
    rigid.update(method="RIGID", baseline_result_sha256="synthetic")
    case, view = affine_study.run_affine_case(
        {"subject_id": "synthetic", "acquisition_site": "Guys"},
        rigid,
        fixed,
        moving,
        mask,
        p,
        tmp_path,
    )
    assert case["success"], case["warnings"]
    return fixed, moving, mask, p, rigid, case, view


def test_determinism_physical_scales_and_template_grid(synthetic):
    f, m, mask, p, rigid, case, _ = synthetic
    before = sitk.GetArrayFromImage(m)
    tx = _euler(rigid["final_transform"])
    a, info, rowsa = affine.optimize_affine(f, m, tx, p)
    b, _, rowsb = affine.optimize_affine(f, m, tx, p)
    assert a.GetParameters() == b.GetParameters() == tuple(case["final_transform"]["parameters"])
    assert rowsa == rowsb
    assert info["configuration"] == optimizers.optimizer_configuration("O0", 8)
    assert info["weights"] == [1] * 12
    for level in info["level_summaries"]:
        assert len(level["physical_shift_scales"]) == 12
        assert min(level["physical_shift_scales"]) > 0
    image = resample_to_fixed(m, f, a)
    for field in ("GetSize", "GetDirection", "GetOrigin", "GetSpacing"):
        assert getattr(image, field)() == getattr(f, field)()
    np.testing.assert_array_equal(before, sitk.GetArrayFromImage(m))
    converted = affine.rigid_to_affine(tx)
    assert full_metric(f, m, converted, None, p)["MI"] == pytest.approx(
        rigid["MI_final"], abs=1e-12
    )
    assert affine.affine_coverage(f, m, mask, converted) == brain_coverage(f, m, mask, tx)
    assert full_metric(f, m, a, None, p)["MI"] == case["MI_final"]


def paired_records(rigid, case):
    records = []
    for n in range(18):
        for method, source in (("RIGID", rigid), ("AFFINE", case)):
            r = deepcopy(source)
            r.update(subject_id=f"synthetic{n}", site=["Guys", "HH", "IOP"][n // 6], method=method)
            records.append(r)
    return records


def test_aggregation_and_pair_validation(synthetic):
    *_, rigid, case, _ = synthetic
    records = paired_records(rigid, case)
    summary, pairs = affine_study.summarize(records)
    assert len(pairs) == 18
    assert summary["AFFINE"]["overall"]["success_count"] == 18
    assert summary["RIGID"]["by_site"]["HH"]["count"] == 6
    assert summary["paired"]["overall"]["MI_gain"]["median"] == case["delta_total"]
    with pytest.raises(ValueError):
        affine_study.summarize(records[:-1])
    records[1]["initial_transform"]["matrix_lps"][0][3] += 2
    with pytest.raises(ValueError, match="initialization"):
        affine_study.summarize(records)


def test_previous_output_protected(tmp_path, monkeypatch):
    old = tmp_path / "prior"
    old.mkdir()
    (old / "subject.json").write_text("unchanged")
    monkeypatch.setattr(
        affine_study, "load_inputs", lambda *a: pytest.fail("must fail before load")
    )
    with pytest.raises(ValueError, match="empty"):
        affine_study.run("config", "baseline", "pilot", "corrected", old)
    assert (old / "subject.json").read_text() == "unchanged"


def test_runner_calls_only_eighteen_affine_cases(synthetic, tmp_path, monkeypatch):
    f, m, mask, p, rigid, case, view = synthetic
    cohort = pd.DataFrame(
        [
            dict(
                subject_id=f"synthetic{n}",
                acquisition_site=["Guys", "HH", "IOP"][n // 6],
                t1_path="unused",
            )
            for n in range(18)
        ]
    )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    originals = {}
    for s in cohort.to_dict("records"):
        r = deepcopy(rigid)
        r.update(subject_id=s["subject_id"], site=s["acquisition_site"])
        path = baseline / (s["subject_id"] + ".json")
        path.write_text(json.dumps(r))
        originals[s["subject_id"]] = (r, path)
    saved = {path: path.read_bytes() for _, path in originals.values()}
    monkeypatch.setattr(
        affine_study,
        "load_inputs",
        lambda *a: (cohort, p, SimpleNamespace(provenance={}), f, mask, []),
    )
    monkeypatch.setattr(affine_study, "load_baseline", lambda *a: originals)
    monkeypatch.setattr(affine_study.nib, "load", lambda *a: None)
    monkeypatch.setattr(affine_study, "canonicalize", lambda *a: None)
    monkeypatch.setattr(affine_study, "to_sitk", lambda *a: m)
    monkeypatch.setattr(affine_study, "_panel", lambda *a: None)
    monkeypatch.setattr(optimizers, "register", lambda *a: pytest.fail("no rigid optimization"))
    calls = []

    def only_affine(subject, reference, *args):
        calls.append(subject["subject_id"])
        r = deepcopy(case)
        r.update(subject_id=subject["subject_id"], site=subject["acquisition_site"])
        return r, view

    monkeypatch.setattr(affine_study, "run_affine_case", only_affine)
    output = tmp_path / "new"
    affine_study.run("config", baseline, "pilot", "corrected", output)
    assert len(calls) == len(set(calls)) == 18
    assert all(path.read_bytes() == data for path, data in saved.items())
    assert len(pd.read_csv(output / "comparison.csv")) == 36
    assert len(list((output / "subjects").glob("*_AFFINE.json"))) == 18
    # Finalization reads saved JSON; it cannot launch either registration method.
    affine_study.finalize(output)
    assert len(calls) == 18
