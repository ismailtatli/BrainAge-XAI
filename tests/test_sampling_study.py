"""Synthetic checks for the fixed O0 sampling-only experiment."""

import json
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from brainage_xai.registration import optimizers
from brainage_xai.registration import sampling_study as study
from brainage_xai.registration.optimizer_study import run_case
from brainage_xai.registration.settings import RigidParameters


@pytest.fixture
def phantom(tmp_path):
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
    r, view = run_case(
        {"subject_id": "synthetic", "acquisition_site": "Guys"},
        "O0",
        fixed,
        moving,
        mask,
        p,
        tmp_path,
    )
    assert r["success"]
    return fixed, moving, mask, p, r, view


def test_only_sampling_changes():
    base = RigidParameters(
        iterations_per_level=300, evaluation_shrink_factor=1, use_fixed_mask=False
    )
    before = asdict(base)
    for key, fraction in study.SAMPLING.items():
        actual = asdict(study.sampling_parameters(base, key))
        assert actual.pop("sampling_fraction") == fraction
        expected = before.copy()
        expected.pop("sampling_fraction")
        assert actual == expected
    assert asdict(base) == before and study.NEW_CONDITIONS == ("S5", "S10")
    assert optimizers.optimizer_configuration("O0")["kwargs"]["learningRate"] == 2
    assert optimizers.optimizer_configuration("O0")["kwargs"]["estimateLearningRate"] == "Never"
    with pytest.raises(ValueError):
        study.sampling_parameters(base, "S20")
    with pytest.raises(ValueError):
        study.sampling_parameters(replace(base, sampling_fraction=0.05), "S10")


@pytest.mark.parametrize("key", ["S5", "S10"])
def test_fixed_seed_determinism(phantom, key):
    f, m, mask, p, _, _ = phantom
    params = study.sampling_parameters(p, key)
    a, ra, rowsa = optimizers.register(f, m, mask, params, "O0")
    b, rb, rowsb = optimizers.register(f, m, mask, params, "O0")
    assert a.GetParameters() == b.GetParameters()
    assert rowsa == rowsb and ra["MI_final"] == rb["MI_final"]
    assert params.sampling_seed == 42
    assert ra["translation_stage"]["configuration"] == optimizers.optimizer_configuration("O0", 8)


def test_baseline_reuse_and_aggregation(phantom, tmp_path):
    _, _, _, _, original, _ = phantom
    source = tmp_path / "baseline.json"
    source.write_text(json.dumps(original))
    saved = source.read_bytes()
    reference = study.reference_record(original, source)
    assert source.read_bytes() == saved
    assert reference["reused_baseline"] and reference["MI_final"] == original["MI_final"]
    records = []
    for n in range(18):
        for key, fraction in study.SAMPLING.items():
            r = deepcopy(reference)
            r.update(
                subject_id=f"synthetic{n}",
                site=["Guys", "HH", "IOP"][n // 6],
                condition=key,
                sampling_fraction=fraction,
                reused_baseline=key == "S2",
            )
            records.append(r)
    summary, pairs = study.summarize(records)
    assert len(pairs) == 54
    assert summary["S2"]["overall"]["technical_success_count"] == 18
    assert summary["S10"]["by_site"]["IOP"]["technical_success_count"] == 6
    assert np.isclose(
        summary["transform_disagreement"]["S2/S5"]["overall"]["rotation_geodesic_deg"]["max"],
        0,
        atol=1e-12,
        rtol=0,
    )
    assert summary["baseline_paired_MI"]["S5"]["overall"]["lower_final_MI_than_S2"] == 0
    assert source.read_bytes() == saved
    with pytest.raises(ValueError):
        study.summarize(records[:-1])


def test_no_overwrite_or_registration_on_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    old = output / "prior.json"
    old.write_text("preserve")
    monkeypatch.setattr(
        study, "load_inputs", lambda *args: pytest.fail("must stop before input/registration")
    )
    with pytest.raises(ValueError, match="empty"):
        study.run(
            tmp_path / "config",
            tmp_path / "base",
            tmp_path / "pilot",
            tmp_path / "corrected",
            output,
        )
    assert old.read_text() == "preserve"


def test_run_executes_only_36_new_cases(phantom, tmp_path, monkeypatch):
    f, m, mask, p, record, view = phantom
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    cohort = pd.DataFrame(
        [
            {
                "subject_id": f"synthetic{n}",
                "acquisition_site": ["Guys", "HH", "IOP"][n // 6],
                "t1_path": str(tmp_path / f"synthetic{n}.nii.gz"),
            }
            for n in range(18)
        ]
    )
    originals = {}
    for subject in cohort.to_dict("records"):
        r = deepcopy(record)
        r.update(subject_id=subject["subject_id"], site=subject["acquisition_site"])
        path = baseline / (subject["subject_id"] + ".json")
        path.write_text(json.dumps(r))
        originals[subject["subject_id"]] = (r, path)
    hashes = {path: path.read_bytes() for _, path in originals.values()}
    from types import SimpleNamespace

    monkeypatch.setattr(
        study, "load_inputs", lambda *args: (cohort, p, SimpleNamespace(provenance={}), f, mask, [])
    )
    monkeypatch.setattr(study, "load_baseline", lambda *args: originals)
    monkeypatch.setattr(study.nib, "load", lambda *args: None)
    monkeypatch.setattr(study, "canonicalize", lambda *args: None)
    monkeypatch.setattr(study, "to_sitk", lambda *args: m)
    monkeypatch.setattr(study, "_panel", lambda *args: None)
    calls = []

    def new_case(subject, key, fixed, moving, coverage, params, directory):
        calls.append((subject["subject_id"], params.sampling_fraction, key))
        return deepcopy(originals[subject["subject_id"]][0]), view

    monkeypatch.setattr(study, "run_case", new_case)
    study.run(
        tmp_path / "config", baseline, tmp_path / "pilot", tmp_path / "corrected", tmp_path / "new"
    )
    assert len(calls) == 36
    assert {fraction for _, fraction, _ in calls} == {0.05, 0.10}
    assert all(key == "O0" for _, _, key in calls)
    assert all(path.read_bytes() == data for path, data in hashes.items())
    assert len(pd.read_csv(tmp_path / "new/comparison.csv")) == 54
