"""Numerical metrics, validation-only calibration and reference context."""

import numpy as np
import pandas as pd


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def metrics(ages, predictions):
    a, p = np.asarray(ages, float), np.asarray(predictions, float)
    if a.shape != p.shape or a.ndim != 1 or not len(a) or not np.isfinite([a, p]).all():
        raise ValueError("Require paired finite subject-level predictions")
    e = p - a
    ss = np.sum((a - a.mean()) ** 2)
    return {
        "n": len(a),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "r2": float(1 - np.sum(e**2) / ss) if ss > 0 else None,
        "mean_error": float(e.mean()),
        "median_absolute_error": float(np.median(np.abs(e))),
        "prediction_age_correlation": correlation(a, p),
        "residual_age_slope": float(np.polyfit(a, e, 1)[0]) if ss > 0 else None,
        "residual_age_correlation": correlation(a, e),
    }


def select_model(validation):
    """Only validation predictions accepted; predefined collapse checks precede MAE."""
    if set(validation) != {"A", "B"}:
        raise ValueError("Require both predefined validation candidates")
    ids = [
        f.sort_values("subject_id")[["subject_id", "age"]].reset_index(drop=True)
        for f in validation.values()
    ]
    if not ids[0].equals(ids[1]):
        raise ValueError("Candidates must use identical validation subjects and ages")
    candidates = {}
    for kind, frame in validation.items():
        if kind not in ("A", "B") or not frame.split.eq("validation").all():
            raise ValueError("Selection accepts validation data for A/B only")
        m = metrics(frame.age, frame.predicted_age)
        pathological = (
            m["prediction_age_correlation"] is None
            or m["prediction_age_correlation"] <= 0
            or frame.predicted_age.std(ddof=0) < 1.0
            or frame.predicted_age.min() < 0
            or frame.predicted_age.max() > 120
            or m["r2"] is None
            or m["r2"] <= 0
        )
        candidates[kind] = {"metrics": m, "pathological": bool(pathological)}
    valid = [k for k in sorted(candidates) if not candidates[k]["pathological"]]
    if not valid:
        raise ValueError("Both candidates show pathological validation generalization")
    selected = min(valid, key=lambda k: candidates[k]["metrics"]["mae"])
    return {
        "selected": selected,
        "candidates": candidates,
        "criterion": "minimum validation MAE among nonpathological candidates; ties A",
    }


def bootstrap_mae(age, prediction, seed=2026, replicates=10000):
    e = np.abs(np.asarray(prediction) - np.asarray(age))
    rng = np.random.default_rng(seed)
    means = e[rng.integers(0, len(e), size=(replicates, len(e)))].mean(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def fit_bias(validation):
    if not validation.split.eq("validation").all() or validation.subject_id.duplicated().any():
        raise ValueError("Bias correction fit requires unique validation subjects only")
    age = validation.age.to_numpy(float)
    bag = validation.predicted_age.to_numpy(float) - age
    if len(age) < 3 or np.std(age) < 1e-12:
        raise ValueError("Insufficient validation ages")
    intercept, beta = np.linalg.lstsq(np.column_stack([np.ones(len(age)), age]), bag, rcond=None)[0]
    return {
        "intercept": float(intercept),
        "beta": float(beta),
        "fit_split": "validation",
        "fit_subjects": sorted(validation.subject_id.tolist()),
    }


def apply_bias(frame, fit):
    if fit["fit_split"] != "validation":
        raise ValueError("Invalid correction fit provenance")
    if frame.split.eq("test").any() and set(frame.subject_id) & set(fit["fit_subjects"]):
        raise ValueError("Test subject present in calibration fit")
    f = frame.copy()
    f["raw_BAG"] = f.predicted_age - f.age
    f["corrected_BAG"] = f.raw_BAG - (fit["intercept"] + fit["beta"] * f.age)
    f["corrected_brain_age"] = f.age + f.corrected_BAG
    return f


def bag_summary(frame):
    return {
        name: {
            "mean": float(frame[name].mean()),
            "sd": float(frame[name].std(ddof=1)),
            "age_correlation": correlation(frame.age, frame[name]),
        }
        for name in ("raw_BAG", "corrected_BAG")
    }


def fit_reference(validation):
    if not validation.split.eq("validation").all():
        raise ValueError("Reference distribution requires validation only")
    return {
        "label": "research-model reference",
        "uncertainty": "empirical model uncertainty",
        "fit_split": "validation",
        "fit_subjects": sorted(validation.subject_id.tolist()),
        "corrected_bag": sorted(validation.corrected_BAG.tolist()),
        "raw_residual_quantiles": np.quantile(validation.raw_BAG, [0.025, 0.975]).tolist(),
    }


def reference_position(values, reference):
    samples = np.asarray(reference["corrected_bag"])
    v = np.asarray(values)
    return (
        100 * ((samples[:, None] < v).sum(0) + 0.5 * (samples[:, None] == v).sum(0)) / len(samples)
    )


def subgroups(frame, edges=(18, 30, 45, 60, 75, 100)):
    f = frame.copy()
    f["age_bin"] = pd.cut(f.age, edges, right=False).astype(str)
    f["sex_code"] = f.sex.fillna("missing").astype(str)
    result = {}
    for field in ("acquisition_site", "age_bin", "sex_code"):
        result[field] = {
            str(k): {
                "n": len(g),
                "mae": float(np.abs(g.predicted_age - g.age).mean()),
                "mean_error": float((g.predicted_age - g.age).mean()),
            }
            for k, g in f.groupby(field, observed=True)
        }
    return result
