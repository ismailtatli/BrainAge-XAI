"""Synthetic-only optimizer comparison, continuity, diagnostics and output safety."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from brainage_xai.config.preprocessing import validate_output_directory
from brainage_xai.registration import optimizer_study as study
from brainage_xai.registration import optimizers as opt
from brainage_xai.registration.rigid import transform_record
from brainage_xai.registration.settings import RigidParameters
from brainage_xai.registration.tuning import tune


@pytest.fixture
def phantom():
    z, y, x = np.indices((26, 28, 30))
    data = 100 * np.exp(-((x - 13) ** 2 / 30 + (y - 12) ** 2 / 40 + (z - 11) ** 2 / 25))
    data += 40 * np.exp(-((x - 21) ** 2 + (y - 19) ** 2 + (z - 17) ** 2) / 10)
    fixed = sitk.GetImageFromArray(data.astype(np.float32))
    moving = sitk.GetImageFromArray(np.roll(data, 2, axis=2).astype(np.float32))
    moving.SetOrigin((2, -1, 3))
    mask = sitk.Cast(fixed > 3, sitk.sitkUInt8)
    params = replace(
        RigidParameters(),
        iterations_per_level=8,
        shrink_factors=(2, 1),
        smoothing_sigmas_mm=(1.0, 0.0),
        sampling_fraction=0.4,
        evaluation_shrink_factor=1,
        use_fixed_mask=False,
    )
    return fixed, moving, mask, params


@pytest.mark.parametrize("key", opt.SPECS)
def test_explicit_configuration(key):
    class Recorder:
        Never, Once = 0, 1

        def __getattr__(self, name):
            def record(*args, **kwargs):
                self.calls[name] = (args, kwargs)

            return record

    method = Recorder()
    method.calls = {}
    config = opt.configure_optimizer(method, key)
    kwargs = method.calls["SetOptimizerAs" + opt.SPECS[key]["family"]][1]
    assert kwargs["numberOfIterations"] == 300
    assert kwargs["estimateLearningRate"] == (0 if key == "O0" else 1)
    assert method.calls["SetOptimizerScalesFromPhysicalShift"][0] == (5, 0.01)
    assert config == opt.optimizer_configuration(key)
    if key == "O1":
        assert kwargs["convergenceWindowSize"] == 10
        assert kwargs["lineSearchMaximumIterations"] == 20
        assert kwargs["lineSearchUpperLimit"] == 2
    if key == "O2":
        assert kwargs["minStep"] == 0.0001
        assert kwargs["maximumStepSizeInPhysicalUnits"] == 1
    with pytest.raises(ValueError):
        opt.optimizer_configuration(key, 0)


@pytest.mark.parametrize("key", opt.SPECS)
def test_registration_repeatability_freezing_continuity(phantom, key):
    f, m, mask, p = phantom
    before = sitk.GetArrayFromImage(m)
    tx, r, rows = opt.register(f, m, mask, p, key)
    tx2, r2, rows2 = opt.register(f, m, mask, p, key)
    assert tx.GetParameters() == tx2.GetParameters()
    assert rows == rows2
    assert np.array_equal(sitk.GetArrayFromImage(m), before)
    assert r["initial_transform"] == r2["initial_transform"]
    first, second = r["translation_stage"], r["rigid_stage"]
    assert first["weights"] == [0, 0, 0, 1, 1, 1]
    assert first["input_transform"]["parameters"][:3] == first["output_transform"]["parameters"][:3]
    assert first["output_transform"] == second["input_transform"]
    for stage in (first, second):
        assert stage["runtime_seconds"] >= stage["optimizer_execute_seconds"] > 0
        assert stage["stop"] and stage["level_stops"]
        assert stage["trajectory_summary"]["best_position"]
        assert len(stage["level_summaries"]) == len(stage["shrink_factors"])
        assert stage["final_physical_shift_scales"]
        assert all(v > 0 for v in stage["final_physical_shift_scales"])
        assert stage["configuration"]["scales"] == opt.SCALE_POLICY
    assert r["MI_final"] == r2["MI_final"]
    assert r["delta_total"] == r["MI_initial"] - r["MI_final"]
    assert np.isclose(r["delta_translation"] + r["delta_rigid"], r["delta_total"])
    assert np.isclose(r["final_transform"]["rotation_determinant"], 1)
    assert r["final_transform"]["orthonormality_error"] < 1e-10
    if key == "O0":
        _, old = tune(f, m, mask, "GEOMETRY", False, p)
        assert study.reproduce_o0(r, old) == {
            "exact_transforms_and_MI": True,
            "equivalent_at_1e_10": True,
            "stop_conditions_identical": True,
            "iterations_identical": True,
        }


def test_budget_stops_and_trajectory():
    for text in (
        "RegularStepGradientDescentOptimizerv4: Maximum number of iterations (300) exceeded.",
        "GradientDescentLineSearchOptimizerv4: Maximum number of iterations (300) exceeded.",
    ):
        assert opt.budget_hit(text)
        assert opt.stop_category(text) == "iteration_budget"
    assert not opt.budget_hit("line search maximum evaluations reached")
    assert opt.stop_category("Convergence checker passed at iteration 25.") == "window_convergence"
    assert opt.stop_category("Step too small after 40 iterations") == "step_too_small"
    rows = [
        dict(level=level, iteration=i, sampled_metric=v)
        for level, i, v in [(0, 0, -0.1), (0, 1, -0.2), (1, 0, -0.05), (1, 1, -0.04)]
    ]
    summary = opt.trajectory_summary(rows)
    assert summary["worsening_steps_within_levels"] == 1
    assert summary["best_position"] == {"level": 0, "iteration": 1}
    assert summary["first_metric"] == -0.1
    assert summary["best_metric"] == -0.2
    assert summary["final_metric"] == -0.04


def test_transform_disagreement():
    a = sitk.Euler3DTransform()
    b = sitk.Euler3DTransform()
    b.SetRotation(0, 0, np.deg2rad(6))
    b.SetTranslation((3, 4, 0))
    diff = opt.disagreement(transform_record(a), transform_record(b))
    assert np.isclose(diff["rotation_geodesic_deg"], 6)
    assert diff["translation_norm_mm"] == 5
    assert diff["provisional_review_warning"]
    assert not opt.disagreement(transform_record(a), transform_record(a))[
        "provisional_review_warning"
    ]
    b.SetCenter((1, 0, 0))
    with pytest.raises(ValueError, match="common center"):
        opt.disagreement(transform_record(a), transform_record(b))


def test_private_result_tables_runtime_and_provenance(phantom, tmp_path):
    f, m, mask, p = phantom
    root = tmp_path / "experiment"
    for directory in ("subjects", "traces", "qc"):
        (root / directory).mkdir(parents=True)
    subject = {"subject_id": "synthetic", "acquisition_site": "Guys"}
    record, view = study.run_case(subject, "O0", f, m, mask, p, root)
    assert record["success"] and record["output_geometry_valid"]
    assert len(view) == 3
    assert record["runtime_seconds"] > 0
    assert not record["fixed_metric_mask_used"] and not record["moving_metric_mask_used"]
    assert study.environment()["ITK_threads"] == 1
    assert "optimizers.py" in study.core_hashes()
    for n in range(18):
        for key in opt.SPECS:
            r = deepcopy(record)
            r.update(subject_id=f"synthetic{n}", site=("Guys", "HH", "IOP")[n // 6], optimizer=key)
            (root / "subjects" / f"synthetic{n}_{key}.json").write_text(json.dumps(r))
    summary = study.write_tables(root)
    assert summary["O0"]["overall"]["technical_success_count"] == 18
    assert summary["O1"]["by_site"]["HH"]["technical_success_count"] == 6
    assert summary["O2"]["overall"]["runtime_seconds"]["max"] > 0
    assert len(pd.read_csv(root / "comparison.csv")) == 54
    assert len(pd.read_csv(root / "transform_disagreement.csv")) == 54
    assert study.write_tables(root) == summary
    assert not list(root.rglob("*.nii*"))


def test_source_protection_and_mask_rejection(phantom, tmp_path):
    source = tmp_path / "source" / "synthetic.nii.gz"
    source.parent.mkdir()
    source.write_bytes(b"synthetic source")
    for target in (source.parent / "output", tmp_path, Path.cwd() / "unignored_optimizer_output"):
        with pytest.raises(ValueError):
            validate_output_directory(target, [source])
    assert source.read_bytes() == b"synthetic source"
    f, m, mask, p = phantom
    with pytest.raises(ValueError, match="no metric masks"):
        opt.register(f, m, mask, replace(p, use_fixed_mask=True), "O1")


def test_coverage_reference_does_not_change_optimizer(phantom):
    f, m, mask, p = phantom
    alternative = sitk.Image(mask.GetSize(), sitk.sitkUInt8) + 1
    alternative.CopyInformation(mask)
    a, ra, rowsa = opt.register(f, m, mask, p, "O1")
    b, rb, rowsb = opt.register(f, m, alternative, p, "O1")
    assert a.GetParameters() == b.GetParameters()
    assert rowsa == rowsb
    assert ra["MI_initial"] == rb["MI_initial"]
    assert ra["MI_final"] == rb["MI_final"]


def test_existing_experiment_is_not_overwritten(monkeypatch, tmp_path):
    output = tmp_path / "private"
    output.mkdir()
    prior = output / "existing.json"
    prior.write_text("preserve prior result")
    monkeypatch.setattr(study, "load_inputs", lambda *args: (None,) * 6)
    with pytest.raises(ValueError, match="must be empty"):
        study.run(tmp_path / "config", tmp_path / "baseline", tmp_path / "corrected", output)
    assert prior.read_text() == "preserve prior result"


def test_failed_case_records_error_without_source_mutation(phantom, monkeypatch, tmp_path):
    f, m, mask, p = phantom
    before = sitk.GetArrayFromImage(m)

    def fail(*args):
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(opt, "register", fail)
    r, view = study.run_case(
        {"subject_id": "synthetic", "acquisition_site": "HH"}, "O2", f, m, mask, p, tmp_path
    )
    assert not r["success"] and view is None
    assert r["runtime_seconds"] >= 0
    assert "synthetic optimizer failure" in r["warnings"][0]
    assert np.array_equal(sitk.GetArrayFromImage(m), before)
