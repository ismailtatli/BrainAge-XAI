"""Synthetic application tests: no real MRI, accepted checkpoint or participant artifacts."""

import io
import json
import platform
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from pypdf import PdfReader

from brainage_xai.inference import cli, config, pipeline
from brainage_xai.inference.config import CONTRACT_SHA256, ROOT, RuntimeUnavailable
from brainage_xai.inference.result import (
    AGE_WARNING,
    InferenceResult,
    age_warnings,
    bag_values,
    interpretation,
    validate_age,
)
from brainage_xai.registration.template import sha256


def nifti_bytes(tmp_path, *, shape=(12, 13, 14), nonfinite=False, constant=False):
    values = (
        np.ones(shape, dtype=np.float32)
        if constant
        else np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    )
    if nonfinite:
        values.flat[0] = np.nan
    image = nib.Nifti1Image(values, np.eye(4))
    image.header.set_xyzt_units("mm")
    p = tmp_path / "synthetic.nii.gz"
    nib.save(image, p)
    return p


def fake_runtime(tmp):
    tmp.mkdir(exist_ok=True, parents=True)
    resource = tmp / "opaque-resource.bin"
    resource.write_bytes(b"synthetic resource")
    slices = {
        "baseline_axial_index": 96,
        "indices": {
            "axial": [37, 60, 83, 106, 129],
            "coronal": [60, 87, 114, 142, 169],
            "sagittal": [53, 74, 96, 118, 139],
        },
    }
    return SimpleNamespace(
        paths={
            k: resource
            for k in [
                "checkpoint",
                "template",
                "template_mask",
                "reference",
                "correction",
                "slices",
            ]
        },
        slices=slices,
        device="cpu",
        temporary_dir=tmp / "scratch",
        output_dir=tmp / "output",
        max_voxels=1000000,
        max_upload_bytes=1000000,
        config_sha256="0" * 64,
        correction={
            "intercept": 5.0,
            "beta": -0.1,
            "fit_split": "validation",
            "fit_subjects": ["validation-only"],
        },
        reference={"corrected_bag": [-10.0, 0.0, 10.0], "raw_residual_quantiles": [-8.0, 9.0]},
        contract={
            "spec_sha256": {
                "preprocessing.frozen.yaml": "1" * 64,
                "registration.frozen.yaml": "2" * 64,
            },
            "source_sha256": {},
            "age_ranges_years": {"train": [20.0, 85.0], "validation": [22.0, 81.0]},
        },
    )


def patch_pipeline(mp, tmp):
    cfg = fake_runtime(tmp)
    mp.setattr(pipeline, "load_runtime_config", lambda _: cfg)

    def processed(image, config, progress):
        a = np.zeros((193, 229, 193), dtype=np.float32)
        a[40:150, 50:180, 30:150] = 0.5
        return a, {"status": "passed"}, {"status": "passed"}, []

    mp.setattr(pipeline, "preprocess", processed)

    class FakeModel(torch.nn.Module):
        kind = "B"

        def forward(self, x):
            assert not torch.is_grad_enabled()
            assert x.shape == (1, 15, 3, 224, 224)
            return torch.tensor([58.0], device=x.device)

    mp.setattr(pipeline, "load_checkpoint", lambda *args: (FakeModel(), {"kind": "B", "epoch": 20}))
    mp.setattr(
        pipeline,
        "grad_cam",
        lambda model, x: (np.zeros((15, 224, 224), dtype=np.float32), np.array([58.0])),
    )
    return cfg


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synthetic-analysis")
    source = nifti_bytes(tmp)
    with pytest.MonkeyPatch.context() as mp:
        cfg = patch_pipeline(mp, tmp / "runtime")
        b = pipeline.infer_upload(source.read_bytes(), "DO-NOT-RETAIN-ID.nii.gz", 50.0, "unused")
        assert not list(cfg.temporary_dir.iterdir())
        return b


@pytest.mark.parametrize("shape", [(5, 6), (5, 6, 7, 1)])
def test_exactly_3d_required(tmp_path, shape):
    p = nifti_bytes(tmp_path, shape=shape)
    with pytest.raises(ValueError, match="three spatial"):
        pipeline.validate_input(p, 100000)


@pytest.mark.parametrize("failure", ["nonfinite", "constant", "empty", "corrupt", "voxel_limit"])
def test_input_failures(tmp_path, failure):
    p = nifti_bytes(tmp_path, nonfinite=failure == "nonfinite", constant=failure == "constant")
    if failure == "empty":
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4)), p)
    if failure == "corrupt":
        p.write_bytes(b"not NIfTI")
    with pytest.raises(Exception):
        pipeline.validate_input(p, 1 if failure == "voxel_limit" else 100000)


def test_invalid_coded_geometry(tmp_path):
    p = nifti_bytes(tmp_path)
    image = nib.load(p)
    image.set_qform(None, code=0)
    image.set_sform(None, code=0)
    nib.save(image, p)
    with pytest.raises(ValueError, match="coded orientation"):
        pipeline.validate_input(p, 100000)


@pytest.mark.parametrize("age", [float("nan"), float("inf"), -1, True, None])
def test_invalid_age(age):
    with pytest.raises(ValueError):
        validate_age(age)


def test_extrapolation_allowed_with_exact_warning(tmp_path):
    c = fake_runtime(tmp_path)
    assert validate_age(0) == 0
    assert age_warnings(100, c.contract["age_ranges_years"]) == [AGE_WARNING]
    assert age_warnings(21, c.contract["age_ranges_years"]) == [AGE_WARNING]
    assert not age_warnings(40, c.contract["age_ranges_years"])


def test_bag_and_reference_use_frozen_values(tmp_path):
    c = fake_runtime(tmp_path)
    r = bag_values(50, 58, c, "anonymous-uuid")
    assert r["raw_BAG"] == 8 and r["corrected_BAG"] == 8
    assert r["corrected_brain_age"] == 58
    assert r["reference_percentile"] == pytest.approx(200 / 3)
    assert r["empirical_uncertainty"]["lower_years"] == 49
    assert r["empirical_uncertainty"]["upper_years"] == 66
    nonzero_trend = bag_values(60, 58, c, "anonymous-uuid")
    assert nonzero_trend["raw_BAG"] == -2
    assert nonzero_trend["corrected_BAG"] == -1
    assert nonzero_trend["corrected_brain_age"] == 59


def test_interpretation_sign_and_no_thresholds():
    assert "older-looking" in interpretation(1.0)
    assert "younger-looking" in interpretation(-1.0)
    assert "displayed precision" in interpretation(0.001)
    assert "displayed precision" in interpretation(-0.001)
    assert "no clinical normal range" in interpretation(0)
    for x in [-2.0, 0.0, 2.0]:
        assert "healthy" not in interpretation(x) and "abnormal" not in interpretation(x)


def test_cleanup_on_preprocessing_error(tmp_path, monkeypatch):
    p = nifti_bytes(tmp_path)
    cfg = patch_pipeline(monkeypatch, tmp_path / "runtime")

    def fail(*args):
        raise ValueError("synthetic QC failure")

    monkeypatch.setattr(pipeline, "preprocess", fail)
    with pytest.raises(pipeline.InferenceFailure, match="synthetic QC failure"):
        pipeline.infer_upload(p.read_bytes(), "arbitrary.nii.gz", 50.0, "unused")
    assert not list(cfg.temporary_dir.iterdir())


def test_upload_size_and_extension_fail_closed(tmp_path, monkeypatch):
    cfg = patch_pipeline(monkeypatch, tmp_path / "runtime")
    cfg.max_upload_bytes = 10
    with pytest.raises(pipeline.InferenceFailure, match="byte limit"):
        pipeline.infer_upload(b"0" * 20, "scan.nii", 50.0, "unused")
    assert not list(cfg.temporary_dir.iterdir())
    with pytest.raises(pipeline.InferenceFailure, match="extensions"):
        pipeline.infer_upload(b"0", "scan.gz", 50.0, "unused")


def test_checkpoint_failure_never_falls_back(tmp_path, monkeypatch):
    source_dir = tmp_path / "raw-input"
    source_dir.mkdir()
    source = nifti_bytes(source_dir)
    cfg = patch_pipeline(monkeypatch, tmp_path / "runtime")

    def missing(*args):
        raise FileNotFoundError("required checkpoint missing")

    monkeypatch.setattr(pipeline, "load_checkpoint", missing)
    with pytest.raises(pipeline.InferenceFailure, match="checkpoint missing"):
        pipeline.infer_file(source, 50.0, "unused")
    assert not list(cfg.temporary_dir.iterdir())


def test_result_json_schema_and_provenance(bundle):
    d = json.loads(bundle.files["result.json"])
    assert d["schema_version"] == 1 and d["chronological_age"] == 50
    assert d["raw_predicted_brain_age"] == 58 and d["raw_BAG"] == 8
    assert (
        d["provenance"]["model"] == "B" and d["provenance"]["correction_fit_split"] == "validation"
    )
    assert "fit_subjects" not in bundle.files["result.json"].decode()
    assert "DO-NOT-RETAIN-ID" not in bundle.result.to_json()
    invalid = asdict(bundle.result)
    invalid["raw_BAG"] = -8
    with pytest.raises(ValueError, match="sign"):
        InferenceResult(**invalid)


def test_xai_mapping_all_fifteen_and_representatives(bundle):
    records = bundle.result.xai["slices"]
    assert len(records) == 15 and len({r["input_offset"] for r in records}) == 15
    assert len([r for r in records if r["representative"]]) == 3
    assert [r["mni_slice_index"] for r in records[:5]] == [37, 60, 83, 106, 129]
    assert np.load(io.BytesIO(bundle.files["xai/gradcam.npy"])).shape == (15, 224, 224)


def test_pdf_contains_results_disclaimer_and_no_identifier(bundle):
    pdf = PdfReader(io.BytesIO(bundle.files["report.pdf"]))
    text = "\n".join(page.extract_text() for page in pdf.pages)
    assert len(pdf.pages) >= 2
    for phrase in [
        "58.00",
        "+8.00",
        "research prototype",
        "not a diagnosis",
        "6.094",
        "5.154",
        "7.101",
    ]:
        assert phrase in text
    assert "DO-NOT-RETAIN-ID" not in text and "/Users/" not in text
    assert not pdf.attachments


def test_exports_private_and_never_overwritten(bundle, tmp_path):
    destination = tmp_path / "analysis"
    pipeline.export_bundle(bundle, destination)
    assert (destination / "report.pdf").is_file()
    assert not list(destination.rglob("*.nii*"))
    with pytest.raises(FileExistsError):
        pipeline.export_bundle(bundle, destination)
    assert (destination / "result.json").stat().st_mode & 0o077 == 0


def test_cli_uses_shared_pipeline(bundle, tmp_path, monkeypatch, capsys):
    source = tmp_path / "input" / "arbitrary.nii"
    source.parent.mkdir()
    source.write_bytes(b"synthetic")
    calls = []
    monkeypatch.setattr(
        cli, "load_runtime_config", lambda _: SimpleNamespace(output_dir=tmp_path / "output")
    )

    def run(*args, **kwargs):
        calls.append(args)
        return bundle

    monkeypatch.setattr(cli, "infer_file", run)
    assert (
        cli.main(
            [
                "--input",
                str(source),
                "--age",
                "50",
                "--config",
                "unused",
                "--output",
                str(tmp_path / "result"),
            ]
        )
        == 0
    )
    assert len(calls) == 1 and calls[0][1] == 50
    assert json.loads(capsys.readouterr().out)["analysis_id"] == bundle.result.analysis_id


def test_frozen_public_contract_and_privacy_rules():
    assert sha256(ROOT / "configs/inference.frozen.yaml") == CONTRACT_SHA256
    for p in ["artifacts/inference/analysis/result.json", "configs/inference.local.yaml"]:
        import subprocess

        assert subprocess.run(["git", "check-ignore", "-q", p]).returncode == 0


@pytest.fixture
def synthetic_config(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    spec = root / "configs/preprocessing.frozen.yaml"
    spec.write_text("software: {packages: {}}\n")
    paths = {}
    for key in ["checkpoint", "template", "template_mask", "correction", "reference", "slices"]:
        paths[key] = root / (key + ".bin")
        paths[key].write_bytes(b"synthetic")
    ids = [f"synthetic-validation-{i}" for i in range(84)]
    paths["correction"].write_text(json.dumps({"fit_split": "validation", "fit_subjects": ids}))
    paths["reference"].write_text(json.dumps({"fit_split": "validation", "fit_subjects": ids}))
    paths["slices"].write_text("{}")
    contract = {
        "artifact_sha256": {
            k: sha256(v) for k, v in paths.items() if k not in ["template", "template_mask"]
        },
        "spec_sha256": {spec.name: sha256(spec)},
        "source_sha256": {},
        "software": {},
        "python": platform.python_version(),
    }
    cp = root / "contract.yaml"
    cp.write_text(yaml.safe_dump(contract))
    monkeypatch.setattr(config, "ROOT", root)
    monkeypatch.setattr(config, "CONTRACT_SHA256", sha256(cp))
    monkeypatch.setattr(config, "T1_SHA256", sha256(paths["template"]))
    monkeypatch.setattr(config, "MASK_SHA256", sha256(paths["template_mask"]))
    raw = {
        "schema_version": 1,
        **{k: str(v) for k, v in paths.items()},
        "contract": str(cp),
        "temporary_dir": str(tmp_path / "scratch"),
        "output_dir": str(tmp_path / "outputs"),
        "device": "cpu",
        "max_upload_bytes": 1000000,
        "max_voxels": 1000000,
    }
    p = root / "runtime.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p, paths


def test_configuration_missing_and_checksum_fail_closed(synthetic_config):
    p, paths = synthetic_config
    assert config.load_runtime_config(p).device == "cpu"
    paths["checkpoint"].write_bytes(b"different")
    with pytest.raises(RuntimeUnavailable, match="checksum mismatch"):
        config.load_runtime_config(p)
    paths["checkpoint"].unlink()
    with pytest.raises(RuntimeUnavailable, match="unavailable"):
        config.load_runtime_config(p)


def test_config_rejects_protected_output_and_contract_tampering(synthetic_config):
    p, paths = synthetic_config
    raw = yaml.safe_load(p.read_text())
    raw["output_dir"] = str(config.ROOT / "configs" / "unsafe")
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="protected"):
        config.load_runtime_config(p)
    Path(raw["contract"]).write_text("tampered")
    with pytest.raises(RuntimeUnavailable, match="contract checksum"):
        config.load_runtime_config(p)


def test_streamlit_startup_without_runtime_is_clear(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BRAINAGE_INFERENCE_CONFIG", str(tmp_path / "missing.yaml"))
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert any("Local runtime is not ready" in w.value for w in app.warning)
    assert next(b for b in app.button if "Run BrainAge-XAI" in b.label).disabled


def test_streamlit_result_and_downloads(bundle, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BRAINAGE_INFERENCE_CONFIG", str(tmp_path / "missing.yaml"))
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"))
    app.session_state["analysis_bundle"] = bundle
    app.run(timeout=30)
    assert not app.exception
    assert any(m.label == "Corrected BAG" for m in app.metric)
    assert len(app.get("download_button")) == 3
    assert any("Attribution gallery" in t.label for t in app.tabs)


def test_frozen_preprocessing_bridge_to_model_inputs(tmp_path, monkeypatch):
    import SimpleITK as sitk

    from brainage_xai.modeling.data import create_inputs, slice_selection
    from brainage_xai.preprocessing.intensity import normalize_intensity
    from brainage_xai.registration.template import TEMPLATE_AFFINE, TemplateBundle

    shape = (193, 229, 193)
    data = np.zeros(shape, dtype=np.float32)
    data[50:80, 60:90, 50:80] = np.random.default_rng(42).normal(100, 10, (30, 30, 30))
    image = nib.Nifti1Image(data, TEMPLATE_AFFINE)
    image.header.set_xyzt_units("mm")
    mask = np.zeros(shape, dtype=np.uint8)
    mask[24:169, 24:206, 6:161] = 1
    mask_image = nib.Nifti1Image(mask, TEMPLATE_AFFINE)
    mask_image.header.set_xyzt_units("mm")
    template = TemplateBundle(image, mask_image, {})
    cfg = fake_runtime(tmp_path)
    cfg.slices = slice_selection(mask > 0, TEMPLATE_AFFINE)
    monkeypatch.setattr(pipeline, "load_template", lambda *args: template)

    def identity(fixed, moving, coverage, parameters, key):
        assert key == "O0" and parameters.sampling_fraction == 0.02
        assert parameters.sampling_seed == 42 and parameters.iterations_per_level == 300
        assert not parameters.use_fixed_mask
        tx = sitk.Euler3DTransform()
        tx.SetComputeZYX(True)
        record = {
            "MI_initial": -1.0,
            "MI_after_translation": -1.0,
            "MI_final": -1.0,
            "delta_total": 0.0,
            "evaluations": [
                {"template_brain_in_bounds_fraction": 1.0, "template_brain_outside_count": 0}
            ],
            "translation_stage": {"budget_hit": False},
            "rigid_stage": {"budget_hit": False},
        }
        record["final_transform"] = pipeline.transform_record(tx)
        return tx, record, []

    monkeypatch.setattr(pipeline.optimizers, "register", identity)
    actual, qc, reg, warnings = pipeline.preprocess(image, cfg, lambda _: None)
    expected, _ = normalize_intensity(data)
    assert np.array_equal(actual, expected)
    assert qc["shape"] == list(shape) and qc["final_interpolations"] == 1
    assert reg["method"] == "S2/O0 rigid" and not warnings
    assert create_inputs(actual, cfg.slices)[1:].shape == (15, 1, 224, 224)


def test_invalid_new_submission_clears_previous_result(bundle, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    runtime = fake_runtime(tmp_path / "runtime")
    monkeypatch.setattr(config, "load_runtime_config", lambda _: runtime)
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"))
    app.session_state["analysis_bundle"] = bundle
    app.run(timeout=30)
    submit = next(b for b in app.button if "Run BrainAge-XAI" in b.label)
    submit.click().run(timeout=30)
    assert not app.exception
    assert "analysis_bundle" not in app.session_state
    assert any("Provide one MRI" in warning.value for warning in app.warning)
