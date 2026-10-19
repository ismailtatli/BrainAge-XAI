"""Seeded subject splits with site/age-bin stratification and exact global quotas."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from brainage_xai.config.preprocessing import load_preprocessing_config, validate_output_directory

from .preparation import eligible_subjects

SPLIT_NAMES = ("train", "validation", "test")


def split_subjects(
    subjects: pd.DataFrame,
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    age_bin_width: float = 10.0,
) -> pd.DataFrame:
    """Allocate strata proportionally, then pool remainders to fill largest-remainder quotas.

    Small strata may not appear in every split. They enter the same seeded remainder
    pool rather than triggering brittle stratified-library exceptions or duplication.
    """
    if subjects.subject_id.duplicated().any():
        raise ValueError("Duplicate subjects cannot be split safely")
    if not subjects.brain_age_label_eligible.astype(str).str.lower().eq("true").all():
        raise ValueError("Splits require only Phase 1 eligible subjects")
    ratios = np.asarray(ratios, dtype=float)
    if (
        ratios.shape != (3,)
        or not np.isfinite(ratios).all()
        or (ratios <= 0).any()
        or not np.isclose(ratios.sum(), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError("Three positive split ratios must sum to one")
    if not np.isfinite(age_bin_width) or age_bin_width <= 0:
        raise ValueError("age_bin_width must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    result = subjects.sort_values("subject_id").reset_index(drop=True).copy()
    ages = pd.to_numeric(result.age, errors="coerce")
    if not (np.isfinite(ages) & (ages > 0)).all():
        raise ValueError("Splits require valid ages")
    rng = np.random.default_rng(seed)
    exact = len(result) * ratios
    quotas = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - quotas), kind="stable")[: len(result) - quotas.sum()]:
        quotas[i] += 1
    remaining = quotas.copy()
    labels = np.full(len(result), "", dtype=object)
    strata = result.assign(_age_bin=np.floor(ages / age_bin_width).astype(int))
    pool = []
    for _, group in strata.groupby(["acquisition_site", "_age_bin"], sort=True, dropna=False):
        indices = rng.permutation(group.index.to_numpy())
        base = np.floor(len(indices) * ratios).astype(int)
        start = 0
        for i, size in enumerate(base):
            labels[indices[start : start + size]] = SPLIT_NAMES[i]
            remaining[i] -= size
            start += size
        pool.extend(indices[start:])
    # Global largest-remainder capacities ensure exact overall ratios even for singleton strata.
    for index in rng.permutation(pool):
        candidates = np.flatnonzero(remaining > 0)
        probabilities = remaining[candidates] / remaining[candidates].sum()
        selected = int(rng.choice(candidates, p=probabilities))
        labels[index] = SPLIT_NAMES[selected]
        remaining[selected] -= 1
    if remaining.any() or np.any(labels == ""):
        raise RuntimeError("Split allocation failed complete coverage")
    result["split"] = labels
    return result


def split_summary(frame: pd.DataFrame) -> dict:
    """Aggregate statistics only; subject IDs and paths never appear in this summary."""
    summary = {}
    for name in SPLIT_NAMES:
        group = frame.loc[frame.split == name]
        summary[name] = {
            "count": len(group),
            "age": {
                key: float(getattr(group.age, key)()) if len(group) else None
                for key in ("mean", "median", "min", "max")
            },
            "site_counts": {
                str(k): int(v)
                for k, v in group.acquisition_site.value_counts().sort_index().items()
            },
        }
    return summary


def run_split(config_path: Path) -> dict:
    config = load_preprocessing_config(config_path)
    subjects = eligible_subjects(config.manifest_path)
    validate_output_directory(
        config.output_dir, [config.manifest_path, *(Path(p) for p in subjects.t1_path)]
    )
    frame = split_subjects(
        subjects,
        seed=config.split_seed,
        ratios=config.split_ratios,
        age_bin_width=config.age_bin_width,
    )
    output = config.output_dir / "splits"
    output.mkdir(parents=True, exist_ok=True)
    frame[["subject_id", "age", "acquisition_site", "split", "t1_path"]].to_csv(
        output / "split_manifest.csv", index=False
    )
    summary = {
        "eligible_subject_count": len(frame),
        "seed": config.split_seed,
        "ratios": list(config.split_ratios),
        "age_bin_width": config.age_bin_width,
        "algorithm": "site_age_bin_floor_allocation_then_seeded_global_remainder_pool",
        "splits": split_summary(frame),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic subject-level splits")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = run_split(args.config)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Split failed: {exc}\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
