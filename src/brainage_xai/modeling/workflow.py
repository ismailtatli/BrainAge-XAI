"""Immutable-input workflow: prepare -> train A/B -> select/calibrate -> test once -> XAI."""

import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from brainage_xai.preprocessing.final import atomic_json
from brainage_xai.registration.template import sha256

from .data import (
    FeatureDataset,
    create_inputs,
    imagenet_channels,
    slice_selection,
    verify_frozen,
)
from .models import BrainAgeModel, load_checkpoint, save_checkpoint, seed_everything
from .statistics import (
    apply_bias,
    bag_summary,
    bootstrap_mae,
    fit_bias,
    fit_reference,
    metrics,
    reference_position,
    select_model,
    subgroups,
)

ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "artifacts/preprocessed/final_v1"
DEFAULT_OUTPUT = ROOT / "artifacts/modeling/phase3/run_v1"
WEIGHTS = ROOT / "artifacts/modeling/phase3/resources/checkpoints/resnet18-f37072fd.pth"


def log(message, **kwargs):
    print(message, flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def provenance(policy):
    files = sorted(Path(__file__).parent.glob("*.py"))
    return {
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "source_sha256": {p.name: sha256(p) for p in files},
        "policy": policy,
        "policy_sha256": sha256(ROOT / "configs/modeling.frozen.yaml"),
        "pretrained_sha256": sha256(WEIGHTS),
        "pretrained_source": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "software": {
            n: importlib.metadata.version(n)
            for n in ("torch", "torchvision", "numpy", "nibabel", "pandas")
        },
        "python": platform.python_version(),
        "machine": platform.machine(),
        "deterministic_algorithms": True,
        "data_loader_workers": 0,
        "reproducibility_limit": "No promise of bitwise identity across hardware/backends/releases",
    }


def protected_snapshot():
    old = read_json(FINAL / "integrity_verification.json")["baseline_sha256"]
    paths = {Path(p) for p in old["raw_hashes"]}
    paths.update(
        Path(p)
        for p in old["protected"]
        if p.endswith((".xls", ".xlsx")) or "/artifacts/resources/" in p
    )
    paths.update(p for p in FINAL.rglob("*") if p.is_file())
    tracked = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "configs",
                "src/brainage_xai/preprocessing",
                "src/brainage_xai/registration",
            ],
            cwd=ROOT,
        )
        .decode()
        .splitlines()
    )
    paths.update(ROOT / n for n in tracked if "modeling" not in n)
    result = {}
    for p in sorted(paths):
        digest = sha256(p)
        expected = old["raw_hashes"].get(str(p), old["protected"].get(str(p)))
        if expected is not None and expected != digest:
            raise ValueError("Protected Phase 2/raw/template input differs: " + p.name)
        s = p.stat()
        result[str(p)] = {"sha256": digest, "size": s.st_size, "mtime_ns": s.st_mtime_ns}
    if len(old["raw_hashes"]) != 581:
        raise ValueError("Raw input inventory differs")
    return result


def verify_integrity(baseline):
    changes = []
    for name, old in baseline.items():
        p = Path(name)
        if not p.is_file():
            changes.append(name)
            continue
        s = p.stat()
        if [s.st_size, s.st_mtime_ns, sha256(p)] != [old["size"], old["mtime_ns"], old["sha256"]]:
            changes.append(name)
    current = {str(p) for p in FINAL.rglob("*") if p.is_file()}
    expected = {n for n in baseline if Path(n).is_relative_to(FINAL)}
    if current != expected:
        changes.append("final_v1 file inventory")
    if changes:
        raise ValueError(f"Immutable input changes: {changes[:3]}")
    return {
        "passed": True,
        "protected_files": len(baseline),
        "raw_mri_count": 581,
        "final_v1_unchanged": True,
        "metadata_templates_configs_unchanged": True,
    }


def prepare(output, policy):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Never overwrite an existing modeling run")
    if not WEIGHTS.is_file() or not sha256(WEIGHTS).startswith("f37072fd"):
        raise FileNotFoundError("Verified ImageNet weights unavailable; random fallback forbidden")
    output.mkdir(parents=True)
    started = time.perf_counter()
    atomic_json(output / "policy.json", policy)
    atomic_json(output / "provenance.json", provenance(policy))
    log("Snapshotting immutable raw, template, config and final_v1 inputs")
    atomic_json(output / "integrity_baseline.json", protected_snapshot())
    cohort = verify_frozen(ROOT, FINAL, log)
    cohort.to_csv(output / "cohort.csv", index=False)
    spec = yaml.safe_load((ROOT / "configs/preprocessing.frozen.yaml").read_text())
    mask_image = nib.load(ROOT / spec["template"]["qc_brain_mask"])
    selection = slice_selection(np.asanyarray(mask_image.dataobj) > 0, mask_image.affine)
    if selection["baseline_axial_index"] != policy["baseline_axial_index"]:
        raise ValueError("Baseline central slice policy differs")
    selection["template_mask_sha256"] = sha256(ROOT / spec["template"]["qc_brain_mask"])
    atomic_json(output / "slices.json", selection)
    atomic_json(
        output / "prepared_inputs.json",
        {n: sha256(output / n) for n in ("cohort.csv", "slices.json", "policy.json")},
    )
    atomic_json(
        output / "state.json",
        {
            "stage": "prepared",
            "test_evaluations": 0,
            "prepare_seconds": time.perf_counter() - started,
        },
    )
    log("Frozen inputs verified; immutable subject membership and slice rule saved")


def cache_features(output, cohort, model, device):
    """Only fixed, nontrainable ImageNet prefix; training never sees test examples."""
    gray_dir, feature_dir = output / "inputs", output / "features"
    gray_dir.mkdir(exist_ok=True)
    feature_dir.mkdir(exist_ok=True)
    selection = read_json(output / "slices.json")
    model.eval()
    for i, row in enumerate(cohort.itertuples(), 1):
        feature_file = feature_dir / f"{row.subject_id}.npy"
        gray_file = gray_dir / f"{row.subject_id}.npy"
        if feature_file.exists() and gray_file.exists():
            continue
        image = nib.load(FINAL / "volumes" / f"{row.subject_id}.nii.gz")
        gray = create_inputs(image.get_fdata(dtype=np.float32, caching="unchanged"), selection)
        np.save(gray_file, gray.numpy(), allow_pickle=False)
        maps = []
        with torch.no_grad():
            for batch in gray.split(4):
                maps.append(model.prefix(imagenet_channels(batch).to(device)).cpu())
        np.save(feature_file, torch.cat(maps).numpy(), allow_pickle=False)
        if i % 20 == 0 or i == len(cohort):
            log(f"Cached fixed inputs/prefix {i}/{len(cohort)} ({set(cohort.split)})")


def loader(frame, output, kind, policy, shuffle):
    return DataLoader(
        FeatureDataset(frame, output / "features", kind),
        batch_size=policy["subject_batch_size"],
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(policy["seed"]),
        num_workers=0,
        drop_last=False,
    )


def predict(model, batches, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for maps, _ in batches:
            predictions.extend(model.from_maps(maps.to(device)).cpu().numpy().tolist())
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite model predictions")
    return np.array(predictions)


def prediction_frame(frame, predictions):
    f = frame[["subject_id", "split", "age", "acquisition_site", "sex"]].copy()
    f["predicted_age"] = predictions
    return f


def train_one(output, kind, policy, cohort, device, attempt):
    started = time.perf_counter()
    seed_everything(policy["seed"])
    train = cohort[cohort.split.eq("train")].reset_index(drop=True)
    validation = cohort[cohort.split.eq("validation")].reset_index(drop=True)
    pretrained = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = BrainAgeModel(kind, train.age.mean(), pretrained).to(device)
    train_loader = loader(train, output, kind, policy, True)
    validation_loader = loader(validation, output, kind, policy, False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=policy["learning_rate"],
        weight_decay=policy["weight_decay"],
    )
    loss_fn = nn.SmoothL1Loss(beta=policy["smooth_l1_beta"])
    best, stale, history = float("inf"), 0, []
    model_dir = output / kind
    model_dir.mkdir(exist_ok=True)
    checkpoint = model_dir / f"best_attempt_{attempt}.pt"
    for epoch in range(1, policy["maximum_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, abs_sum, count = 0.0, 0.0, 0
        for maps, ages in train_loader:
            maps, ages = maps.to(device), ages.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model.from_maps(maps)
            loss = loss_fn(predicted, ages)
            if not torch.isfinite(loss):
                raise FloatingPointError("Obvious numerical divergence: nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            loss_sum += loss.item() * len(ages)
            abs_sum += (predicted.detach() - ages).abs().sum().item()
            count += len(ages)
        val_predictions = predict(model, validation_loader, device)
        val_mae = metrics(validation.age, val_predictions)["mae"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / count,
                "train_mae": abs_sum / count,
                "validation_mae": val_mae,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
        log(
            f"Model {kind} epoch {epoch}: train MAE {abs_sum / count:.3f}, "
            f"validation MAE {val_mae:.3f} ({history[-1]['epoch_seconds']:.1f}s)"
        )
        if val_mae < best - policy["minimum_mae_improvement"]:
            best, stale = val_mae, 0
            metadata = {
                "kind": kind,
                "mean_age": float(train.age.mean()),
                "epoch": epoch,
                "validation_mae": best,
                "policy_sha256": sha256(output / "policy.json"),
                "training_subjects": sorted(train.subject_id.tolist()),
                "validation_subjects": sorted(validation.subject_id.tolist()),
                "pretrained_sha256": sha256(WEIGHTS),
                "device": device,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
            save_checkpoint(checkpoint, model, metadata)
        else:
            stale += 1
        atomic_json(model_dir / f"history_attempt_{attempt}.json", history)
        if stale >= policy["early_stopping_patience"]:
            break
    model, metadata = load_checkpoint(checkpoint, device)
    val_predictions = predict(model, validation_loader, device)
    prediction_frame(validation, val_predictions).to_csv(model_dir / "validation.csv", index=False)
    result = {
        "checkpoint": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": sha256(checkpoint),
        "best_epoch": metadata["epoch"],
        "epochs_completed": len(history),
        "runtime_seconds": time.perf_counter() - started,
        "device": device,
        "attempt": attempt,
        "metrics": metrics(validation.age, val_predictions),
        "parameters": metadata["parameter_count"],
        "trainable_parameters": metadata["trainable_parameters"],
    }
    atomic_json(model_dir / "result.json", result)
    return result


def train(output, policy, device):
    state = read_json(output / "state.json")
    if state["stage"] not in ("prepared", "training") or (output / "selection.json").exists():
        raise ValueError("Training is closed after selection")
    if (
        sha256(ROOT / "configs/modeling.frozen.yaml")
        != read_json(output / "provenance.json")["policy_sha256"]
    ):
        raise ValueError("Training policy changed")
    if sha256(WEIGHTS) != read_json(output / "provenance.json")["pretrained_sha256"]:
        raise ValueError("Verified pretrained weights changed")
    prepared = read_json(output / "prepared_inputs.json")
    if any(sha256(output / n) != digest for n, digest in prepared.items()):
        raise ValueError("Prepared subject membership/input representation changed")
    current_provenance = provenance(policy)
    training_provenance = output / "training_provenance.json"
    if training_provenance.exists():
        if read_json(training_provenance) != current_provenance:
            raise ValueError("Cannot resume training after source/environment changes")
    else:
        atomic_json(training_provenance, current_provenance)
    cohort = pd.read_csv(output / "cohort.csv")
    train_val = cohort[cohort.split.ne("test")]
    seed_everything(policy["seed"])
    prefix = BrainAgeModel("A", 0, torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    cache_features(output, train_val, prefix.to(device), device)
    del prefix
    state["stage"] = "training"
    atomic_json(output / "state.json", state)
    for kind in ("A", "B"):
        if (output / kind / "result.json").exists():
            continue
        attempts_path = output / f"{kind}_attempts.json"
        attempts = read_json(attempts_path) if attempts_path.exists() else []
        while len(attempts) < 2:
            attempts.append({"attempt": len(attempts) + 1, "device": device, "status": "started"})
            atomic_json(attempts_path, attempts)
            try:
                train_one(output, kind, policy, cohort, device, len(attempts))
                attempts[-1]["status"] = "complete"
                atomic_json(attempts_path, attempts)
                break
            except (RuntimeError, FloatingPointError) as error:
                attempts[-1].update(status="failed", error=str(error))
                atomic_json(attempts_path, attempts)
                unsupported = "MPS" in str(error) and any(
                    w in str(error).lower()
                    for w in ("not implemented", "not supported", "unsupported")
                )
                if unsupported:
                    device = "cpu"
                elif not isinstance(error, FloatingPointError):
                    raise
                if len(attempts) == 2:
                    raise
                log(f"One documented technical retry for {kind}: {error}")
        else:
            raise RuntimeError("Technical retry allowance exhausted")
    validation = {k: pd.read_csv(output / k / "validation.csv") for k in ("A", "B")}
    decision = select_model(validation)
    kind = decision["selected"]
    result = read_json(output / kind / "result.json")
    decision.update(
        checkpoint=result["checkpoint"],
        checkpoint_sha256=result["checkpoint_sha256"],
        selection_split="validation",
        test_evaluations_at_selection=0,
    )
    atomic_json(output / "selection.json", decision)
    fit = fit_bias(validation[kind])
    fit["validation_predictions_sha256"] = sha256(output / kind / "validation.csv")
    atomic_json(output / "bias_correction.json", fit)
    calibrated = apply_bias(validation[kind], fit)
    calibrated.to_csv(output / "validation_calibrated.csv", index=False)
    atomic_json(output / "reference.json", fit_reference(calibrated))
    lock_files = [
        "selection.json",
        "bias_correction.json",
        "reference.json",
        "policy.json",
        result["checkpoint"],
        f"{kind}/validation.csv",
    ]
    atomic_json(output / "model_lock.json", {n: sha256(output / n) for n in lock_files})
    state["stage"] = "selected_and_calibrated"
    atomic_json(output / "state.json", state)
    log(f"Selected {kind} using validation only; checkpoint, correction and reference frozen")


def claim_test(output):
    decision = read_json(output / "selection.json")
    state = read_json(output / "state.json")
    if (
        state["stage"] != "selected_and_calibrated"
        or state["test_evaluations"] != 0
        or sha256(output / decision["checkpoint"]) != decision["checkpoint_sha256"]
    ):
        raise ValueError("Locked test requires selected, unchanged checkpoint and unused test gate")
    for n in ("bias_correction.json", "reference.json"):
        if not (output / n).is_file():
            raise ValueError("Validation-only calibration must be frozen before test")
    locked = read_json(output / "model_lock.json")
    if any(sha256(output / n) != digest for n, digest in locked.items()):
        raise ValueError("Frozen selection/checkpoint/calibration/reference changed")
    # Exclusive durable claim: a crash cannot silently cause a second test evaluation.
    with (output / "test_evaluation_claim.json").open("x") as stream:
        json.dump(
            {
                "checkpoint_sha256": decision["checkpoint_sha256"],
                "correction_sha256": sha256(output / "bias_correction.json"),
                "reference_sha256": sha256(output / "reference.json"),
                "test_evaluations": 1,
            },
            stream,
            indent=2,
        )
    state.update(stage="test_claimed", test_evaluations=1)
    atomic_json(output / "state.json", state)
    return decision


def evaluate_locked_test(output, policy, device):
    # Cache construction uses no learned/test-fitted quantities and no test outcomes.
    decision = read_json(output / "selection.json")
    cohort = pd.read_csv(output / "cohort.csv")
    test = cohort[cohort.split.eq("test")].reset_index(drop=True)
    seed_everything(policy["seed"])
    prefix = BrainAgeModel("A", 0, torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    cache_features(output, test, prefix.to(device), device)
    del prefix
    decision = claim_test(output)
    model, _ = load_checkpoint(output / decision["checkpoint"], device)
    predictions = predict(model, loader(test, output, decision["selected"], policy, False), device)
    raw = prediction_frame(test, predictions)
    raw.to_csv(output / "test_predictions.csv", index=False)
    calibrated = apply_bias(raw, read_json(output / "bias_correction.json"))
    reference = read_json(output / "reference.json")
    calibrated["research_reference_percentile"] = reference_position(
        calibrated.corrected_BAG, reference
    )
    lo, hi = reference["raw_residual_quantiles"]
    calibrated["empirical_age_lower"] = calibrated.predicted_age - hi
    calibrated["empirical_age_upper"] = calibrated.predicted_age - lo
    calibrated.to_csv(output / "test_calibrated.csv", index=False)
    results = {
        "raw_metrics": metrics(raw.age, raw.predicted_age),
        "mae_bootstrap_95_ci": bootstrap_mae(
            raw.age, raw.predicted_age, policy["bootstrap_seed"], policy["bootstrap_replicates"]
        ),
        "bootstrap_seed": policy["bootstrap_seed"],
        "subgroups": subgroups(raw, policy["age_bin_edges"]),
        "bag_validation": bag_summary(pd.read_csv(output / "validation_calibrated.csv")),
        "bag_test": bag_summary(calibrated),
        "test_evaluations": 1,
        "checkpoint_sha256": decision["checkpoint_sha256"],
        "test_predictions_sha256": sha256(output / "test_predictions.csv"),
    }
    atomic_json(output / "test_results.json", results)
    state = read_json(output / "state.json")
    state["stage"] = "test_complete"
    atomic_json(output / "state.json", state)
    log(f"Locked test evaluation complete, once: MAE {results['raw_metrics']['mae']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "test", "report", "run"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/modeling"):
        raise ValueError("Model outputs must stay under ignored artifacts/modeling")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this execution context; no silent CPU fallback")
    os.environ.setdefault("PYTHONHASHSEED", "42")
    policy = yaml.safe_load((ROOT / "configs/modeling.frozen.yaml").read_text())
    seed_everything(policy["seed"])
    if args.stage in ("prepare", "run"):
        prepare(output, policy)
    if args.stage in ("train", "run"):
        train(output, policy, args.device)
    if args.stage in ("test", "run"):
        evaluate_locked_test(output, policy, args.device)
    if args.stage in ("report", "run"):
        from .reporting import finish

        finish(output, policy, args.device)


if __name__ == "__main__":
    main()
