"""Phase 3 uses temporary synthetic data; no real participants or weight downloads."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from brainage_xai.modeling.data import (
    COUNTS,
    create_inputs,
    imagenet_channels,
    slice_selection,
    validate_splits,
)
from brainage_xai.modeling.models import (
    BrainAgeModel,
    grad_cam,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)
from brainage_xai.modeling.reporting import example_selection
from brainage_xai.modeling.statistics import (
    apply_bias,
    bootstrap_mae,
    fit_bias,
    fit_reference,
    metrics,
    reference_position,
    select_model,
    subgroups,
)
from brainage_xai.modeling.workflow import claim_test, prepare, provenance
from brainage_xai.registration.template import sha256


def frame(split="validation"):
    return pd.DataFrame(
        {
            "subject_id": [f"synthetic-{i}" for i in range(8)],
            "split": split,
            "age": np.arange(20, 60, 5),
            "predicted_age": np.arange(20, 60, 5) * 0.8 + 7,
            "acquisition_site": ["Guys", "HH", "IOP", "Guys"] * 2,
            "sex": [1, 2] * 4,
        }
    )


def test_frozen_counts_and_subject_membership():
    f = pd.DataFrame(
        {
            "subject_id": range(561),
            "age": 40,
            "split": sum(([s] * n for s, n in COUNTS.items()), []),
        }
    )
    assert len(validate_splits(f)) == 561
    for a in COUNTS:
        for b in COUNTS:
            if a != b:
                assert not set(f[f.split.eq(a)].subject_id) & set(f[f.split.eq(b)].subject_id)
    f.loc[0, "subject_id"] = f.loc[1, "subject_id"]
    with pytest.raises(ValueError, match="exactly once"):
        validate_splits(f)


def test_wrong_split_counts_rejected():
    f = pd.DataFrame({"subject_id": ["synthetic"], "age": [40], "split": ["test"]})
    with pytest.raises(ValueError, match="counts"):
        validate_splits(f)


def selection():
    mask = np.zeros((21, 23, 25), bool)
    mask[2:19, 3:20, 4:23] = True
    return slice_selection(mask, np.eye(4))


def test_template_only_slice_coordinates():
    s = selection()
    assert s["indices"]["axial"] == [8, 10, 13, 16, 18]
    assert s["indices"]["sagittal"] == [5, 8, 10, 12, 15]
    assert s["baseline_axial_index"] == 12
    assert s == selection()
    assert s["coordinates_mm"]["axial"] == s["indices"]["axial"]


def test_empty_mask_rejected():
    with pytest.raises(ValueError):
        slice_selection(np.zeros((10, 10, 10)), np.eye(4))


def test_input_determinism_channels_and_shape():
    seed_everything(42)
    volume = np.random.default_rng(42).normal(size=(21, 23, 25)).astype("float32")
    a, b = create_inputs(volume, selection()), create_inputs(volume, selection())
    assert a.shape == (16, 1, 224, 224)
    assert torch.equal(a, b)
    assert 0 <= a.min() <= a.max() <= 1
    rgb = imagenet_channels(a)
    assert rgb.shape == (16, 3, 224, 224)
    for i, (mean, std) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], strict=True)):
        assert torch.allclose(rgb[:, i] * std + mean, a[:, 0], atol=1e-6)


@pytest.mark.parametrize("kind,slices", [("A", 1), ("B", 15)])
def test_model_shapes_shared_backbone_and_checkpoint(tmp_path, kind, slices):
    seed_everything(42)
    model = BrainAgeModel(kind, 40).eval()
    x = torch.randn(1, slices, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1,)
    assert len([n for n, _ in model.named_modules() if n == "backbone"]) == 1
    model.train()
    assert not model.backbone.bn1.training
    assert not model.backbone.layer3[0].conv1.weight.requires_grad
    assert model.backbone.layer4[0].conv1.weight.requires_grad
    path = tmp_path / "synthetic.pt"
    save_checkpoint(path, model, {"kind": kind, "mean_age": 40.0})
    loaded, _ = load_checkpoint(path)
    with torch.no_grad():
        assert torch.equal(loaded(x), y)


def test_metrics_known_values():
    m = metrics([20, 30, 40], [22, 28, 40])
    assert m["mae"] == pytest.approx(4 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(8 / 3))
    assert m["r2"] == pytest.approx(0.96)
    assert m["mean_error"] == 0
    assert m["median_absolute_error"] == 2


def test_metric_nonfinite_rejected():
    with pytest.raises(ValueError):
        metrics([20], [np.nan])


def test_selection_uses_validation_mae_only():
    a, b = frame(), frame()
    b.predicted_age = b.age + 0.2
    assert select_model({"A": a, "B": b})["selected"] == "B"
    b["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_model({"A": a, "B": b})


def test_pathological_candidate_rejected():
    a, b = frame(), frame()
    b.predicted_age = 40
    r = select_model({"A": a, "B": b})
    assert r["selected"] == "A" and r["candidates"]["B"]["pathological"]


def test_validation_only_bias_fit_sign_and_application():
    f = frame()
    fit = fit_bias(f)
    assert fit["intercept"] == pytest.approx(7)
    assert fit["beta"] == pytest.approx(-0.2)
    corrected = apply_bias(f, fit)
    assert np.max(np.abs(corrected.corrected_BAG)) < 1e-12
    assert corrected.raw_BAG.iloc[0] > 0
    assert corrected.raw_BAG.iloc[-1] < 0
    assert np.allclose(corrected.corrected_brain_age, corrected.age + corrected.corrected_BAG)
    with pytest.raises(ValueError, match="validation"):
        fit_bias(frame("test"))
    with pytest.raises(ValueError, match="Test subject"):
        apply_bias(frame("test"), fit)


def test_bootstrap_fixed_seed():
    assert bootstrap_mae([1, 2, 3], [2, 2, 5]) == bootstrap_mae([1, 2, 3], [2, 2, 5])
    assert bootstrap_mae([1, 2], [1, 2]) == [0, 0]


def test_validation_reference_midrank_and_fit_isolation():
    f = frame()
    f["corrected_BAG"] = np.arange(8)
    f["raw_BAG"] = f.predicted_age - f.age
    ref = fit_reference(f)
    assert reference_position([-1, 3.5, 8], ref).tolist() == [0, 50, 100]
    assert reference_position([0], ref).tolist() == [6.25]
    assert ref["uncertainty"] == "empirical model uncertainty"
    with pytest.raises(ValueError):
        fit_reference(f.assign(split="test"))


@pytest.mark.parametrize("kind,slices", [("A", 1), ("B", 15)])
def test_gradcam_maps_to_each_slice(kind, slices):
    seed_everything(42)
    m = BrainAgeModel(kind, 40)
    maps, predicted = grad_cam(m, torch.randn(1, slices, 3, 64, 64))
    assert maps.shape == (slices, 64, 64) and predicted.shape == (1,)
    assert np.isfinite(maps).all() and maps.min() >= 0 and maps.max() <= 1.00001


def test_deterministic_xai_example_rule():
    f = frame("test")
    assert example_selection(f) == example_selection(f.sample(frac=1, random_state=3))
    assert 3 <= len(example_selection(f)) <= 6


def test_subgroup_counts_preserved():
    result = subgroups(frame("test"))
    for groups in result.values():
        assert sum(g["n"] for g in groups.values()) == 8
    assert result["acquisition_site"]["IOP"]["n"] == 2


def test_test_gate_exclusive_claim_and_checkpoint_integrity(tmp_path):
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic")
    (tmp_path / "selection.json").write_text(
        json.dumps({"checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint)})
    )
    state = {"stage": "selected_and_calibrated", "test_evaluations": 0}
    (tmp_path / "state.json").write_text(json.dumps(state))
    for name in ("bias_correction.json", "reference.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "model_lock.json").write_text(
        json.dumps(
            {
                n: sha256(tmp_path / n)
                for n in ("selection.json", "bias_correction.json", "reference.json")
            }
        )
    )
    claim_test(tmp_path)
    with pytest.raises(ValueError):
        claim_test(tmp_path)
    (tmp_path / "state.json").write_text(json.dumps(state))
    with pytest.raises(FileExistsError):
        claim_test(tmp_path)


def test_test_gate_requires_selection(tmp_path):
    with pytest.raises(FileNotFoundError):
        claim_test(tmp_path)


def test_prior_artifacts_never_overwritten(tmp_path):
    p = tmp_path / "existing.txt"
    p.write_text("protected")
    with pytest.raises(FileExistsError):
        prepare(tmp_path, {})
    assert p.read_text() == "protected"


def test_provenance_generation_synthetic_weights(tmp_path, monkeypatch):
    from brainage_xai.modeling import workflow

    path = tmp_path / "synthetic.pt"
    path.write_bytes(b"synthetic weights, not pretrained")
    monkeypatch.setattr(workflow, "WEIGHTS", path)
    prov = provenance({"seed": 42})
    assert prov["pretrained_sha256"] == sha256(path)
    assert prov["source_sha256"]["models.py"]
    assert prov["policy"]["seed"] == 42


def test_fixed_training_policy():
    from brainage_xai.modeling.workflow import ROOT

    p = yaml.safe_load((ROOT / "configs/modeling.frozen.yaml").read_text())
    assert p["models"] == ["A", "B"] and p["seed"] == 42
    assert len(p["slice_fractions"]) == 5
    assert p["technical_retries_per_model"] == 1


def test_training_pass_never_requires_test_inputs(tmp_path, monkeypatch):
    from torchvision.models import resnet18

    from brainage_xai.modeling import workflow

    seed_everything(42)
    weight_file = tmp_path / "synthetic_initialization.pt"
    torch.save(resnet18(weights=None).state_dict(), weight_file)
    monkeypatch.setattr(workflow, "WEIGHTS", weight_file)
    train, val = frame("train"), frame("validation")
    train.subject_id = "train-" + train.subject_id
    val.subject_id = "validation-" + val.subject_id
    cohort = pd.concat([train, val], ignore_index=True)
    (tmp_path / "features").mkdir()
    for sid in cohort.subject_id:
        np.save(
            tmp_path / "features" / f"{sid}.npy",
            np.random.default_rng(42).normal(size=(16, 256, 2, 2)).astype("float32"),
        )
    policy = {
        "seed": 42,
        "subject_batch_size": 4,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "smooth_l1_beta": 1.0,
        "maximum_epochs": 1,
        "early_stopping_patience": 8,
        "minimum_mae_improvement": 0.001,
    }
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    result = workflow.train_one(tmp_path, "A", policy, cohort, "cpu", 1)
    assert result["epochs_completed"] == 1
    predictions = pd.read_csv(tmp_path / "A/validation.csv")
    assert set(predictions.split) == {"validation"}
    _, meta = load_checkpoint(tmp_path / result["checkpoint"])
    assert set(meta["training_subjects"]).isdisjoint(meta["validation_subjects"])
    assert not any("test" in p.name for p in tmp_path.rglob("*"))


def test_pretrained_unavailable_never_silently_random(tmp_path, monkeypatch):
    from brainage_xai.modeling import workflow

    monkeypatch.setattr(workflow, "WEIGHTS", tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="random fallback forbidden"):
        prepare(tmp_path / "new_run", {})


def test_fixed_prefix_equivalent_to_direct_forward():
    seed_everything(42)
    model = BrainAgeModel("A", 40).eval()
    x = torch.randn(2, 1, 3, 64, 64)
    with torch.no_grad():
        direct = model(x)
        maps = model.prefix(x.flatten(0, 1))[:, None]
        assert torch.equal(direct, model.from_maps(maps))


def test_selection_rejects_different_validation_members():
    b = frame()
    b.loc[0, "subject_id"] = "other-synthetic"
    with pytest.raises(ValueError, match="identical validation"):
        select_model({"A": frame(), "B": b})


def test_training_closed_after_selection(tmp_path):
    from brainage_xai.modeling.workflow import train

    (tmp_path / "state.json").write_text(json.dumps({"stage": "selected_and_calibrated"}))
    with pytest.raises(ValueError, match="closed after selection"):
        train(tmp_path, {}, "cpu")


def test_saved_heatmap_rendering_never_evaluates_model(tmp_path, monkeypatch):
    from brainage_xai.modeling import reporting

    def forbidden(*args, **kwargs):
        pytest.fail("Rendering saved heatmaps must not evaluate or load a model")

    monkeypatch.setattr(reporting, "load_checkpoint", forbidden)
    directory = tmp_path / "xai"
    directory.mkdir()
    (directory / "provenance.json").write_text(
        json.dumps({"subjects": ["synthetic"], "views": ["axial"]})
    )
    (tmp_path / "slices.json").write_text(json.dumps({"baseline_axial_index": 4}))
    np.save(directory / "synthetic_gradcam.npy", np.zeros((1, 8, 8), dtype="float32"))
    result = reporting.standalone_heatmaps(tmp_path)
    assert result["heatmap_panels"] == 1 and result["zero_positive_maps"] == 1
    assert (directory / "synthetic_axial_heatmaps.png").is_file()
    assert result["attribution_source_sha256"]["synthetic_gradcam.npy"]
