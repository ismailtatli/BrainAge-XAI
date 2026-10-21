"""Post-lock plots, mapped Grad-CAM and aggregate-only public reporting."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from brainage_xai.preprocessing.final import atomic_json
from brainage_xai.registration.template import sha256

from .data import imagenet_channels
from .models import grad_cam, load_checkpoint
from .workflow import ROOT, log, read_json, verify_integrity

ATTRIBUTION_WORDING = (
    "Regions receiving relatively greater attribution for the model's brain-age prediction."
)


def example_selection(frame):
    """Predefined union of nearest age/error quantile ranks, ties resolved by ID."""
    selected = set()
    f = frame.assign(absolute_error=(frame.predicted_age - frame.age).abs())
    for column in ("age", "absolute_error"):
        ordered = f.sort_values([column, "subject_id"]).reset_index(drop=True)
        for fraction in (0, 0.5, 1):
            selected.add(ordered.iloc[int(np.rint(fraction * (len(ordered) - 1)))].subject_id)
    return sorted(selected)


def plots(output):
    directory = output / "plots"
    directory.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, kind in zip(axes, ("A", "B"), strict=True):
        result = read_json(output / kind / "result.json")
        history = read_json(output / kind / f"history_attempt_{result['attempt']}.json")
        ax.plot([r["epoch"] for r in history], [r["train_mae"] for r in history], label="train")
        ax.plot(
            [r["epoch"] for r in history],
            [r["validation_mae"] for r in history],
            label="validation",
        )
        ax.axvline(result["best_epoch"], color="black", linestyle=":")
        ax.set(xlabel="Epoch", ylabel="MAE (years)", title=f"Model {kind}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=150)
    plt.close(fig)
    test = pd.read_csv(output / "test_predictions.csv")
    residual = test.predicted_age - test.age
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(test.age, test.predicted_age, s=15)
    axes[0].plot([18, 90], [18, 90], "k--")
    axes[0].set(xlabel="Chronological age", ylabel="Raw predicted age")
    axes[1].scatter(test.age, residual, s=15)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(xlabel="Chronological age", ylabel="Raw prediction error (years)")
    axes[2].hist(residual.abs(), bins=12)
    axes[2].set(xlabel="Absolute prediction error (years)", ylabel="Subjects")
    fig.tight_layout()
    fig.savefig(directory / "locked_test_diagnostics.png", dpi=150)
    plt.close(fig)


def xai(output, device):
    selection = read_json(output / "selection.json")
    checkpoint = output / selection["checkpoint"]
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    model, _ = load_checkpoint(checkpoint, device)
    frame = pd.read_csv(output / "test_predictions.csv")
    subjects = example_selection(frame)
    slices = read_json(output / "slices.json")
    directory = output / "xai"
    directory.mkdir(exist_ok=True)
    if (directory / "provenance.json").exists():
        return read_json(directory / "provenance.json")
    views = ["axial"] if selection["selected"] == "A" else ["axial", "coronal", "sagittal"]
    slice_count = 1 if len(views) == 1 else 5
    maps_count = panels = 0
    for sid in subjects:
        gray = torch.from_numpy(np.load(output / "inputs" / f"{sid}.npy"))
        gray = gray[:1] if selection["selected"] == "A" else gray[1:]
        maps, predicted = grad_cam(model, imagenet_channels(gray)[None].to(device))
        saved_prediction = float(frame.loc[frame.subject_id.eq(sid), "predicted_age"].iloc[0])
        if not np.isclose(predicted[0], saved_prediction, rtol=0, atol=0.02):
            raise ValueError(
                "Grad-CAM/full-input prediction differs from frozen cached-map inference"
            )
        np.save(directory / f"{sid}_gradcam.npy", maps, allow_pickle=False)
        metadata = {
            "subject_id": sid,
            "model": selection["selected"],
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "input_sha256": sha256(output / "inputs" / f"{sid}.npy"),
            "prediction": float(predicted[0]),
            "interpretation": ATTRIBUTION_WORDING,
            "selection_rule": "age and absolute-error rank quantiles 0,0.5,1 union",
            "slices": {},
            "all_zero_positive_cams": int(np.sum(maps.max((1, 2)) == 0)),
        }
        for view_index, view in enumerate(views):
            offset = view_index * slice_count
            indices = (
                [slices["baseline_axial_index"]] if len(views) == 1 else slices["indices"][view]
            )
            metadata["slices"][view] = indices
            fig, axes = plt.subplots(2, slice_count, figsize=(3 * slice_count, 6), squeeze=False)
            for j, index in enumerate(indices):
                plane = gray[offset + j, 0].numpy()
                axes[0, j].imshow(plane, cmap="gray", vmin=0, vmax=1)
                axes[1, j].imshow(plane, cmap="gray", vmin=0, vmax=1)
                axes[1, j].imshow(maps[offset + j], cmap="inferno", alpha=0.5, vmin=0, vmax=1)
                axes[0, j].set_title(f"{view}, MNI index {index}")
                for ax in axes[:, j]:
                    ax.axis("off")
            fig.suptitle(ATTRIBUTION_WORDING, fontsize=9)
            fig.tight_layout()
            fig.savefig(directory / f"{sid}_{view}.png", dpi=130)
            plt.close(fig)
            panels += 1
        atomic_json(directory / f"{sid}.json", metadata)
        maps_count += len(maps)
        log(f"Grad-CAM completed for deterministic example ({len(maps)} slices)")
    report = {
        "subject_count": len(subjects),
        "slice_maps": maps_count,
        "overlay_panels": panels,
        "views": views,
        "subjects": subjects,
        "interpretation": ATTRIBUTION_WORDING,
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "layer": "shared ResNet18 layer4 output; 7x7 feature grid upsampled bilinearly",
    }
    atomic_json(directory / "provenance.json", report)
    return report


def metric_table(candidates):
    columns = [
        "mae",
        "rmse",
        "r2",
        "mean_error",
        "median_absolute_error",
        "prediction_age_correlation",
        "residual_age_slope",
        "residual_age_correlation",
    ]
    lines = ["| Metric | A | B |", "|---|---:|---:|"]
    for column in columns:
        a, b = [candidates[k]["metrics"][column] for k in ("A", "B")]
        lines.append(f"| {column} | {a:.5f} | {b:.5f} |")
    return "\n".join(lines)


def standalone_heatmaps(output):
    """Render existing attribution arrays only; no model evaluation or example reselection."""
    directory = output / "xai"
    report = read_json(directory / "provenance.json")
    slices = read_json(output / "slices.json")
    count, zero_maps, sources = 0, 0, {}
    for sid in report["subjects"]:
        source = directory / f"{sid}_gradcam.npy"
        maps = np.load(source, allow_pickle=False)
        sources[source.name] = sha256(source)
        zero_maps += int(np.sum(maps.max((1, 2)) == 0))
        for vi, view in enumerate(report["views"]):
            indices = (
                [slices["baseline_axial_index"]]
                if len(report["views"]) == 1
                else slices["indices"][view]
            )
            fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3), squeeze=False)
            for j, index in enumerate(indices):
                axes[0, j].imshow(maps[vi * len(indices) + j], cmap="inferno", vmin=0, vmax=1)
                axes[0, j].set_title(f"{view}, MNI index {index}")
                axes[0, j].axis("off")
            fig.tight_layout()
            fig.savefig(directory / f"{sid}_{view}_heatmaps.png", dpi=130)
            plt.close(fig)
            count += 1
    summary = {
        "heatmap_panels": count,
        "zero_positive_maps": zero_maps,
        "attribution_source_sha256": sources,
        "renderer_sha256": sha256(Path(__file__)),
    }
    atomic_json(directory / "heatmaps.json", summary)
    return summary


def document(output, policy, xai_report):
    decision = read_json(output / "selection.json")
    results = read_json(output / "test_results.json")
    fit = read_json(output / "bias_correction.json")
    slices = read_json(output / "slices.json")
    prov = read_json(output / "provenance.json")
    training = {k: read_json(output / k / "result.json") for k in ("A", "B")}
    cohort = pd.read_csv(output / "cohort.csv")
    m = results["raw_metrics"]
    lo, hi = results["mae_bootstrap_95_ci"]
    lines = [
        "# Phase 3 scientific ML core results",
        "",
        "Research infrastructure only. Phase 1/2 inputs and subject membership remain frozen.",
        "",
        "## Dataset and representation",
        "",
        "561 IXI subjects: train 393, validation 84, locked test 84. No split regeneration or "
        "slice-level splitting. Site counts: "
        + ", ".join(f"{k} {v}" for k, v in cohort.acquisition_site.value_counts().items())
        + ".",
        "",
        "Exact Phase 2 grid: MNI152NLin2009cAsym, 193×229×193 at 1 mm RAS+. "
        "Existing S2/O0 rigid only; no preprocessing change, crop/pad or new registration.",
        "",
        "Model A uses central axial index 96 (z=18 mm). Model B uses the public brain-mask "
        "nonzero extent only: round-to-even(min + fraction × (max−min)), fractions "
        "0.20/0.35/0.50/0.65/0.80, independently along each axis. Zero-based indices:",
        "",
    ]
    for v in ("axial", "coronal", "sagittal"):
        lines.append(f"- {v}: {slices['indices'][v]}; RAS axis mm: {slices['coordinates_mm'][v]}.")
    lines += [
        "",
        "Mask extent xyz: "
        + str(slices["extent_min_xyz"])
        + " to "
        + str(slices["extent_max_xyz"])
        + ". Mask SHA-256: `"
        + slices["template_mask_sha256"]
        + "`.",
        "",
        "Each full plane is transposed and flipped vertically for consistent display, clipped "
        "from fixed normalized-MRI range [−3,3] into [0,1], resized bilinearly to 224×224 with "
        "antialiasing and align_corners=False, replicated to RGB and ImageNet channel-normalized. "
        "There is no fitted cross-subject input transform or augmentation.",
        "",
        "## Models and fixed training",
        "",
        "Both use torchvision ResNet18 IMAGENET1K_V1 weights (SHA-256 `"
        + prov["pretrained_sha256"]
        + "`). No random-initialization substitution. "
        "Model B shares one backbone across all 15 slices, averages five slice features per view, "
        "then concatenates three 512-dimensional view means. Model A uses its single 512-feature "
        "vector. Both heads are Linear(128), ReLU, Linear(1), plus training mean age. "
        "Fine-tune layer4 and the head only; all BatchNorm statistics/parameters are frozen. "
        "The frozen prefix is cached as float32 without quantization; no test features are cached "
        "until selection and calibration are complete.",
        "",
        f"SmoothL1 beta={policy['smooth_l1_beta']}; AdamW lr={policy['learning_rate']}, "
        f"weight decay={policy['weight_decay']}; subject batch={policy['subject_batch_size']}; "
        f"maximum {policy['maximum_epochs']} epochs, validation-MAE patience "
        f"{policy['early_stopping_patience']}, minimum improvement "
        f"{policy['minimum_mae_improvement']} years. No scheduler or search. "
        "Gradient norm capped at 5. Python/NumPy/PyTorch seed 42; deterministic algorithms, "
        "zero DataLoader workers and seeded shuffle. "
        "Cross-hardware bitwise identity is not promised.",
        "",
        "| Model | Parameters / trainable | Epochs / best | Device | Training seconds |",
        "|---|---:|---:|---|---:|",
    ]
    for k, t in training.items():
        lines.append(
            f"| {k} | {t['parameters']} / {t['trainable_parameters']} | "
            f"{t['epochs_completed']} / {t['best_epoch']} | {t['device']} | "
            f"{t['runtime_seconds']:.2f} |"
        )
    lines += [
        "",
        "Software: "
        + ", ".join(f"{k} {v}" for k, v in prov["software"].items())
        + "; Python "
        + prov["python"]
        + ".",
        "",
        "Training runtimes exclude input verification, slice/prefix caching "
        "and post-lock reporting. "
        "Per-epoch histories, attempt records and complete provenance are private artifacts.",
        "",
        "## Validation-only selection",
        "",
        metric_table(decision["candidates"]),
        "",
        f"Selected **{decision['selected']}** by minimum validation MAE among nonpathological "
        "candidates. Predefined pathology checks reject nonpositive age correlation/R², "
        "prediction SD below one year or ages outside 0–120. No test result entered selection. "
        "Architecture/checkpoint/correction/reference were frozen before the exclusive test claim.",
        "",
        "## Locked test",
        "",
        f"Exactly one selected-model evaluation on 84 subjects. MAE **{m['mae']:.5f} years**, "
        f"RMSE **{m['rmse']:.5f}**, R² **{m['r2']:.5f}**. Deterministic percentile bootstrap "
        f"95% CI for mean absolute error: **[{lo:.5f}, {hi:.5f}] years**, 10,000 subject "
        "resamples, seed 2026; this does not include retraining or external-domain uncertainty.",
        "",
    ]
    for key in (
        "mean_error",
        "median_absolute_error",
        "prediction_age_correlation",
        "residual_age_slope",
        "residual_age_correlation",
    ):
        lines.append(f"- {key}: {m[key]:.6f}.")
    lines += [
        "",
        "Learning curves and predicted-vs-age, residual-vs-age and absolute-error plots "
        "are stored privately under the run's `plots/` directory.",
        "",
        "## Locked-test subgroups",
        "",
        "Age bins were fixed before training, left closed "
        "and right open. Sex codes are reported as supplied; a sex codebook was not inferred.",
        "",
        "| Group | Category | n | MAE (years) | Mean error |",
        "|---|---|---:|---:|---:|",
    ]
    for group, categories in results["subgroups"].items():
        for name, values in categories.items():
            lines.append(
                f"| {group} | {name} | {values['n']} | {values['mae']:.5f} | "
                f"{values['mean_error']:.5f} |"
            )
    lines += [
        "",
        "Small subgroup estimates are unstable and descriptive; no subgroup optimization "
        "or formal fairness claim is made. Acquisition-site/scanner, age and sex distributions "
        "can confound MRI predictions; no independent site-held-out evaluation was performed.",
        "",
        "## BAG, validation-only correction and reference",
        "",
        "Raw BAG = raw predicted age − chronological age. Fit only the selected model's 84 "
        "validation predictions by ordinary least squares: raw BAG = intercept + beta × age.",
        "",
        f"Frozen intercept = **{fit['intercept']:.9f} years**; beta = **{fit['beta']:.9f}**. "
        "corrected BAG = raw BAG − (intercept + beta × chronological age); corrected brain age "
        "= chronological age + corrected BAG. Test subjects are absent from fit/reference IDs, "
        "and their coefficients were never refitted. "
        "All raw prediction metrics above stay unchanged.",
        "",
        "| Split | BAG | Mean | SD | Age correlation |",
        "|---|---|---:|---:|---:|",
    ]
    for split, key in (("validation", "bag_validation"), ("test", "bag_test")):
        for name, values in results[key].items():
            lines.append(
                f"| {split} | {name} | {values['mean']:.6f} | {values['sd']:.6f} | "
                f"{values['age_correlation']:.6f} |"
            )
    lines += [
        "",
        "Validation corrected BAG has zero mean and age-linear association by construction, "
        "not independent validation. Correction uses chronological age and must not be marketed as "
        "improved age prediction. Positive corrected BAG means structurally older-looking relative "
        "to the learned reference; negative means younger-looking. "
        "No diagnostic/clinical threshold.",
        "",
        "The frozen validation corrected-BAG empirical midrank distribution supplies a "
        "**research-model reference** percentile. The validation raw-residual 2.5/97.5 percentiles "
        "supply **empirical model uncertainty** bounds: raw prediction minus upper/lower residual "
        "quantile. These are marginal empirical reference intervals, not clinical confidence "
        "intervals, disease probabilities or calibrated conditional coverage guarantees. "
        "Using the same validation subjects for model choice and reference estimation "
        "can be optimistic.",
        "",
        "## XAI",
        "",
        f"Grad-CAM for the selected model only: {xai_report['subject_count']} deterministic test "
        f"examples, {xai_report['slice_maps']} slice maps and {xai_report['overlay_panels']} "
        "view panels. Examples are the union of age and absolute-error rank quantiles "
        "0/0.5/1, with ID tie-breaking. This post-lock error-based example selection changes "
        "no metric or model. Per-slice arrays, exact input indices, overlays and checkpoint "
        "provenance remain private. The shared layer4 7×7 positive Grad-CAM grid is upsampled "
        "to 224×224 and independently max-normalized per slice; "
        "colors are not comparable magnitudes.",
        "",
        ATTRIBUTION_WORDING,
        "",
        "Grad-CAM is a coarse, gradient-based attribution rather than pathology localization or "
        "causal evidence. It does not establish tissue-level biological specificity; no lesion, "
        "disease, diagnostic or causal neuroanatomical interpretation is justified.",
        "",
        "## Limitations and integrity",
        "",
        "The IXI cohort is small (561), single-dataset and subject to site/scanner confounding. "
        "Residual rigid-registration correspondence and the Phase 2 IOP contrast/visibility "
        "limitation remain. This is 2.5D rather than full volumetric modeling, with ImageNet "
        "transfer and fixed slice/intensity windows. There is no independent external cohort, "
        "external-validity demonstration or clinical validation. BAG is not a diagnosis. "
        "Locked-test findings do not authorize retraining or changing the selected method.",
        "",
        "Integrity verification rehashes all 581 raw MRI files, IXI.xls, templates, "
        "Phase 2 final_v1 "
        "files and frozen Phase 2 source/configs and checks their sizes/mtimes. Subject membership "
        "is unique and disjoint. Checkpoints, inputs, predictions and participant provenance "
        f"remain Git-ignored under `{output.relative_to(ROOT)}/`. No commit/push/merge.",
        "",
    ]
    heatmaps = read_json(output / "xai/heatmaps.json")
    lines += [
        "## Post-lock diagnostic notes",
        "",
        f"There are {heatmaps['heatmap_panels']} standalone heatmap panels in addition to "
        f"the MRI overlays. {heatmaps['zero_positive_maps']} of {xai_report['slice_maps']} "
        "positive Grad-CAM maps are zero. These are retained without replacing examples; "
        "positive-only Grad-CAM suppresses negative contributions, so a blank map does not "
        "establish absence of model dependence or anatomical signal.",
        "",
        f"After validation-only correction, locked-test BAG–age correlation remains "
        f"{results['bag_test']['corrected_BAG']['age_correlation']:.6f}. Correction does not "
        "fully generalize the validation age trend; reference percentiles are "
        "not age/site-specific "
        "norms. No test refitting, model change or additional training was performed. The "
        "75–100-year test group has only two subjects and cannot establish older-age performance.",
        "",
        "MRI is not skull stripped. Nonbrain tissue, background/FOV and site cues may "
        "contribute to predictions and attributions. "
        "Attributions are not isolated biological effects.",
        "",
        "The predefined 0.001-year minimum improvement governs both checkpoint replacement "
        "and patience. Sub-threshold numerical MAE decreases do not replace a saved checkpoint.",
        "",
        "Selected checkpoint SHA-256: `" + decision["checkpoint_sha256"] + "`.",
        "",
    ]
    path = ROOT / "docs/phase3-scientific-ml-results.md"
    path.write_text("\n".join(lines))
    return path


def finish(output, policy, device):
    if read_json(output / "state.json")["stage"] not in ("test_complete", "complete"):
        raise ValueError("Reporting requires completed locked test")
    plots(output)
    xai_report = xai(output, device)
    standalone_heatmaps(output)
    integrity = verify_integrity(read_json(output / "integrity_baseline.json"))
    atomic_json(output / "integrity_verification.json", integrity)
    path = document(output, policy, xai_report)
    state = read_json(output / "state.json")
    state["stage"] = "complete"
    atomic_json(output / "state.json", state)
    log(f"Scientific results document: {path.relative_to(ROOT)}")
