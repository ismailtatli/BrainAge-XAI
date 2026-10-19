"""Last rigid sampling comparison: reuse S2 and execute only S5/S10."""

import argparse
import itertools
import json
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import nibabel as nib
import pandas as pd

from brainage_xai.config.preprocessing import validate_output_directory
from brainage_xai.preprocessing.orientation import canonicalize

from . import optimizers
from .optimizer_qc import _panel
from .optimizer_study import core_hashes, environment, load_inputs, result_row, run_case
from .pipeline import _euler, _json
from .rigid import resample_to_fixed, to_sitk
from .template import sha256
from .tuning_qc import planes

SAMPLING = {"S2": 0.02, "S5": 0.05, "S10": 0.10}
NEW_CONDITIONS = ("S5", "S10")


def sampling_parameters(base, condition):
    """Only the predefined fraction changes; the S2 baseline object is immutable."""
    if condition not in SAMPLING or base.sampling_fraction != 0.02:
        raise ValueError("Require S2 baseline and exactly S2/S5/S10")
    p = replace(base, sampling_fraction=SAMPLING[condition])
    p.validate()
    return p


def require_new_output(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; never overwrite a previous experiment")


def load_baseline(directory, cohort, p):
    provenance = json.loads((directory / "provenance.json").read_text())
    if (
        provenance["shared_parameters"] != json.loads(json.dumps(asdict(p)))
        or provenance["optimizer_configurations"]["O0"] != optimizers.SPECS["O0"]
        or provenance["core_sha256"] != core_hashes()
        or provenance["environment"] != environment()
    ):
        raise ValueError("O0 scientific configuration/implementation/environment changed")
    old_cohort = pd.read_csv(directory / "selection.csv")
    if not old_cohort.equals(cohort):
        raise ValueError("Must use exact saved O0 cohort/order")
    selected = json.loads((directory / "candidate_selection.json").read_text())
    repeated = json.loads((directory / "repeatability.json").read_text())
    if selected["optimizer"] != "O0" or repeated["optimizer"] != "O0" or not repeated["all_passed"]:
        raise ValueError("Require the selected, reproducible O0 baseline")
    if len(repeated["subjects"]) != 18:
        raise ValueError("Require complete eighteen-subject repeat evidence")
    records = {}
    for subject in cohort.to_dict("records"):
        sid = subject["subject_id"]
        path = directory / "subjects" / f"{sid}_O0.json"
        r = json.loads(path.read_text())
        if not r["success"] or not r["output_geometry_valid"] or r["optimizer"] != "O0":
            raise ValueError("Invalid stored O0 baseline")
        if r["fixed_metric_mask_used"] or r["moving_metric_mask_used"]:
            raise ValueError("Metric masks are prohibited")
        for stage in ["translation_stage", "rigid_stage"]:
            if r[stage]["configuration"] != optimizers.optimizer_configuration("O0"):
                raise ValueError("Stored O0 optimizer differs")
        records[sid] = (r, path)
    return records


def reference_record(record, path):
    r = deepcopy(record)
    r.update(
        condition="S2",
        sampling_fraction=0.02,
        reused_baseline=True,
        baseline_result_path=str(path),
        baseline_result_sha256=sha256(path),
        runtime_origin="historical O0 measurement; no registration rerun",
    )
    # Old optimizer-family disagreement flags are outside this experiment.
    r["warnings"] = [w for w in r["warnings"] if not w.startswith("optimizer_disagreement/")]
    return r


def summarize(records):
    """Pure aggregation: no image access or registration and no prior artifact writes."""
    if len(records) != 54 or len({(r["subject_id"], r["condition"]) for r in records}) != 54:
        raise ValueError("Expected eighteen subjects x S2/S5/S10")
    subjects = {r["subject_id"] for r in records}
    if len(subjects) != 18 or {r["condition"] for r in records} != set(SAMPLING):
        raise ValueError("Invalid cohort or sampling conditions")
    pairs = []
    for sid in sorted(subjects):
        cases = {r["condition"]: r for r in records if r["subject_id"] == sid}
        good = [r for r in cases.values() if r["success"]]
        if any(r["initial_transform"] != good[0]["initial_transform"] for r in good):
            raise ValueError("Sampling changed initialization")
        if good and max(r["MI_initial"] for r in good) - min(r["MI_initial"] for r in good) > 1e-10:
            raise ValueError("Sampling changed full-resolution initial MI")
        for a, b in itertools.combinations(SAMPLING, 2):
            if cases[a]["success"] and cases[b]["success"]:
                pairs.append(
                    dict(
                        optimizers.disagreement(
                            cases[a]["final_transform"], cases[b]["final_transform"]
                        ),
                        subject_id=sid,
                        site=cases[a]["site"],
                        pair=a + "/" + b,
                    )
                )
    summary = {
        key: {
            "overall": optimizers.aggregate([r for r in records if r["condition"] == key]),
            "by_site": {
                site: optimizers.aggregate(
                    [r for r in records if r["condition"] == key and r["site"] == site]
                )
                for site in ["Guys", "HH", "IOP"]
            },
        }
        for key in SAMPLING
    }

    def pair_summary(rows):
        return {
            "count": len(rows),
            "rotation_geodesic_deg": optimizers.quantiles(
                [r["rotation_geodesic_deg"] for r in rows]
            ),
            "translation_norm_mm": optimizers.quantiles([r["translation_norm_mm"] for r in rows]),
            "review_warning_count": sum(r["provisional_review_warning"] for r in rows),
        }

    summary["transform_disagreement"] = {
        pair: {
            "overall": pair_summary([r for r in pairs if r["pair"] == pair]),
            "by_site": {
                site: pair_summary([r for r in pairs if r["pair"] == pair and r["site"] == site])
                for site in ["Guys", "HH", "IOP"]
            },
        }
        for pair in ["S2/S5", "S2/S10", "S5/S10"]
    }
    base = {r["subject_id"]: r for r in records if r["condition"] == "S2"}
    summary["baseline_paired_MI"] = {
        key: {
            scope: {
                "count": len(rows),
                "lower_final_MI_than_S2": sum(
                    r["MI_final"] < base[r["subject_id"]]["MI_final"] for r in rows
                ),
                "delta_final_MI_vs_S2": optimizers.quantiles(
                    [base[r["subject_id"]]["MI_final"] - r["MI_final"] for r in rows]
                ),
            }
            for scope, rows in [
                ("overall", [r for r in records if r["condition"] == key and r["success"]]),
                *[
                    (
                        site,
                        [
                            r
                            for r in records
                            if r["condition"] == key and r["site"] == site and r["success"]
                        ],
                    )
                    for site in ["Guys", "HH", "IOP"]
                ],
            ]
        }
        for key in NEW_CONDITIONS
    }
    return summary, pairs


def finalize(output):
    records = [json.loads(p.read_text()) for p in sorted((output / "subjects").glob("*.json"))]
    summary, pairs = summarize(records)
    rows = []
    for r in records:
        row = result_row(r)
        row.update(
            condition=r["condition"],
            sampling_fraction=r["sampling_fraction"],
            reused_baseline=r["reused_baseline"],
            runtime_origin=r["runtime_origin"],
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(
        [{k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in pairs]
    ).to_csv(output / "transform_disagreement.csv", index=False)
    _json(output / "summary.json", summary)
    return summary


def run(config, baseline, pilot, corrected, output):
    baseline, pilot, corrected, output = map(
        lambda p: Path(p).resolve(), [baseline, pilot, corrected, output]
    )
    require_new_output(output)
    cohort, p, bundle, fixed, mask, inputs = load_inputs(config, pilot, corrected, output)
    old = load_baseline(baseline, cohort, p)
    inputs += [path for path in baseline.rglob("*") if path.is_file()]
    validate_output_directory(output, inputs)
    hashes = {str(path): sha256(path) for path in inputs}
    for name in ["subjects", "qc", "S5/traces", "S10/traces"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output / "selection.csv", index=False)
    _json(
        output / "provenance.json",
        {
            "parameters_by_condition": {
                key: asdict(sampling_parameters(p, key)) for key in SAMPLING
            },
            "optimizer_configuration": optimizers.optimizer_configuration("O0"),
            "core_sha256": core_hashes(),
            "study_sha256": sha256(Path(__file__)),
            "template": bundle.provenance,
            "environment": environment(),
            "input_sha256": hashes,
            "baseline_reused_without_optimization_or_MI_reevaluation": True,
            "new_registration_conditions": list(NEW_CONDITIONS),
            "new_registration_count": 36,
            "mask_role": "coverage and QC only; no optimization mask",
            "runtime_note": (
                "S2 historical v4 case time; new cases measured identically, host load uncontrolled"
            ),
            "positive_delta": "MI_initial minus MI_final; lower negative Mattes MI is better",
            "disagreement_review_limits": {
                "rotation_deg": 5.0,
                "common_center_translation_mm": 5.0,
            },
        },
    )
    fixed_planes, mask_planes = planes(fixed), planes(mask)
    panels = []
    for n, subject in enumerate(cohort.to_dict("records"), 1):
        sid = subject["subject_id"]
        moving = to_sitk(canonicalize(nib.load(subject["t1_path"])))
        stored, path = old[sid]
        reference = reference_record(stored, path)
        reference["qc_path"] = str(output / "qc" / f"{sid}_sampling.png")
        _json(output / "subjects" / f"{sid}_S2.json", reference)
        # Reconstruct only diagnostic planes from the stored pull transform; no optimizer/MI call.
        views = {"S2": planes(resample_to_fixed(moving, fixed, _euler(stored["final_transform"])))}
        for key in NEW_CONDITIONS:
            print(f"Sampling study {n}/18 {sid}: {key}", file=sys.stderr, flush=True)
            params = sampling_parameters(p, key)
            r, view = run_case(subject, "O0", fixed, moving, mask, params, output / key)
            r.update(
                condition=key,
                sampling_fraction=params.sampling_fraction,
                reused_baseline=False,
                runtime_origin="new registration; excludes source IO and QC/table export",
                registration_parameters=asdict(params),
                qc_path=reference["qc_path"],
            )
            if r["success"] and (
                r["initial_transform"] != reference["initial_transform"]
                or abs(r["MI_initial"] - reference["MI_initial"]) > 1e-10
            ):
                raise ValueError("Non-sampling factor changed")
            _json(output / "subjects" / f"{sid}_{key}.json", r)
            views[key] = view
        rows = [("Fixed MNI", fixed_planes)]
        for key in SAMPLING:
            rows.extend([(key, views.get(key)), (key + " / fixed checkerboard", views.get(key))])
        _panel(
            rows,
            fixed_planes,
            mask_planes,
            Path(reference["qc_path"]),
            sid + " — sampling comparison",
            True,
        )
        panels.append((sid, subject["acquisition_site"], views))
    for site in ["Guys", "HH", "IOP"]:
        rows = [("Fixed MNI", fixed_planes)] + [
            (sid + " " + key, views[key])
            for sid, s, views in panels
            if s == site
            for key in SAMPLING
        ]
        _panel(
            rows,
            fixed_planes,
            mask_planes,
            output / "qc" / f"site_{site}.png",
            site + " sampling comparison",
        )
    summary = finalize(output)
    if any(sha256(Path(path)) != h for path, h in hashes.items()):
        raise ValueError("Prior inputs/artifacts changed")
    print(f"Completed 36 new registrations; S2 reused: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/registration.local.yaml"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("artifacts/registration/rigid_optimizer_v4")
    )
    parser.add_argument("--pilot", type=Path, default=Path("artifacts/registration/rigid_pilot_v1"))
    parser.add_argument(
        "--corrected", type=Path, default=Path("artifacts/registration/rigid_tuning_v3")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        finalize(args.output.resolve())
    else:
        run(args.config, args.baseline, args.pilot, args.corrected, args.output)


if __name__ == "__main__":
    main()
