"""Manageable obliquity/challenging-case QC sample; no model slice selection."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from brainage_xai.registration.optimizer_qc import _panel
from brainage_xai.registration.rigid import to_sitk
from brainage_xai.registration.template import load_template, sha256
from brainage_xai.registration.tuning_qc import planes

from .final import atomic_json, load_spec


def select_qc(cohort, geometry, challenging):
    """Six per site: prior challenging cases plus low/median/high obliquity strata."""
    if geometry.subject_id.duplicated().any():
        raise ValueError("Duplicate geometry subject")
    frame = cohort.merge(
        geometry[["subject_id", "max_obliquity_deg"]], on="subject_id", validate="one_to_one"
    )
    if len(frame) != len(cohort) or not np.isfinite(frame.max_obliquity_deg).all():
        raise ValueError("Missing original-obliquity diagnostic")
    chosen = []
    for site, group in frame.groupby("acquisition_site", sort=True):
        ordered = group.sort_values(["max_obliquity_deg", "subject_id"]).reset_index(drop=True)
        candidates = []
        for sid in challenging.get(site, [])[:2]:
            if sid not in set(ordered.subject_id):
                raise ValueError("Challenging QC reference outside eligible cohort")
            candidates.append((sid, "previous_registration_challenging"))
        for index, reason in [
            (0, "low_original_obliquity"),
            (len(ordered) // 2, "median_original_obliquity"),
            (len(ordered) - 1, "high_original_obliquity"),
        ]:
            candidates.append((ordered.iloc[index].subject_id, reason))
        candidates += [(row.subject_id, "obliquity_sample_fill") for row in ordered.itertuples()]
        seen = set()
        for sid, reason in candidates:
            if sid in seen:
                continue
            row = ordered.loc[ordered.subject_id == sid].iloc[0].to_dict()
            row["qc_selection_reason"] = reason
            chosen.append(row)
            seen.add(sid)
            if len(seen) == 6:
                break
        if len(seen) != 6:
            raise ValueError("Need six distinct QC subjects per site")
    return pd.DataFrame(chosen)


def generate(spec_path, output, geometry, previous):
    spec, paths, cohort, _ = load_spec(spec_path)
    challenging = {}
    for site in ("Guys", "HH", "IOP"):
        rows = [json.loads(p.read_text()) for p in previous.glob("subjects/*_RIGID.json")]
        rows = [r for r in rows if r["site"] == site]
        rows.sort(
            key=lambda r: max(abs(v) for v in r["final_transform"]["rotation_deg_lps_xyz"]),
            reverse=True,
        )
        challenging[site] = [r["subject_id"] for r in rows[:2]]
    selected = select_qc(cohort, pd.read_csv(geometry), challenging)
    selected.to_csv(output / "qc" / "selection.csv", index=False)
    bundle = load_template(paths["template"], paths["mask"])
    fixed, masks = planes(to_sitk(bundle.image)), planes(to_sitk(bundle.mask))
    panels = []
    for s in selected.to_dict("records"):
        path = output / "volumes" / f"{s['subject_id']}.nii.gz"
        record = json.loads((output / "provenance" / f"{s['subject_id']}.json").read_text())
        view = planes(to_sitk(nib.load(path))) if record["status"] == "success" else None
        _panel(
            [
                ("MNI template", fixed),
                ("Final normalized rigid", view),
                ("Final / fixed checkerboard", view),
            ],
            fixed,
            masks,
            output / "qc" / f"{s['subject_id']}.png",
            f"{s['subject_id']} {s['acquisition_site']} — original obliquity "
            f"{s['max_obliquity_deg']:.2f} degrees",
            True,
        )
        panels.append((s["subject_id"], s["acquisition_site"], view))
    for site in ("Guys", "HH", "IOP"):
        _panel(
            [("MNI template", fixed)] + [(sid, view) for sid, s, view in panels if s == site],
            fixed,
            masks,
            output / "qc" / f"site_{site}.png",
            site + " final preprocessing",
        )
    atomic_json(
        output / "qc" / "provenance.json",
        {
            "selection_count": len(selected),
            "per_site": 6,
            "geometry_diagnostics_sha256": sha256(geometry),
            "previous_experiment": str(previous),
            "scope": "Engineering QC only; no medical interpretation; no model slices selected",
            "review_status": "pending image-based review",
            "specification": spec["geometry"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/preprocessed/final_v1"))
    args = parser.parse_args()
    generate(
        Path("configs/preprocessing.frozen.yaml"),
        args.output,
        Path("artifacts/preprocessed/phase2/geometry_diagnostics/manifest.csv"),
        Path("artifacts/registration/rigid_vs_affine_v6"),
    )


if __name__ == "__main__":
    main()
