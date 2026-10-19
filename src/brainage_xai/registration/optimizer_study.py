"""Private 18-subject O0/O1/O2 comparison and full-cohort selected-candidate repeat."""

import argparse
import itertools
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from brainage_xai.config.preprocessing import validate_output_directory
from brainage_xai.data.preparation import eligible_subjects
from brainage_xai.preprocessing.orientation import canonicalize

from . import optimizers
from .optimizer_qc import contact_sheets, subject_panel
from .pipeline import _json
from .rigid import resample_to_fixed, to_sitk
from .settings import RigidParameters, load_registration_config
from .template import load_template, sha256
from .tuning_qc import planes

ATOL = 1e-10


def environment():
    return {
        "python": platform.python_version(),
        "SimpleITK": sitk.Version_VersionString(),
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
        "ITK_threads": 1,
    }


def core_hashes():
    directory = Path(__file__).parent
    return {
        name: sha256(directory / name)
        for name in ("optimizers.py", "rigid.py", "tuning.py", "settings.py", "template.py")
    }


def load_inputs(config_path, baseline, corrected, output):
    config = load_registration_config(config_path)
    cohort = pd.read_csv(baseline / "pilot_selection.csv")
    if (
        len(cohort) != 18
        or cohort.subject_id.nunique() != 18
        or cohort.acquisition_site.value_counts().to_dict() != {"Guys": 6, "HH": 6, "IOP": 6}
    ):
        raise ValueError("Exact v1 eighteen-subject cohort required")
    eligible = eligible_subjects(config.manifest_path).set_index("subject_id")
    for s in cohort.to_dict("records"):
        if (
            s["subject_id"] not in eligible.index
            or Path(s["t1_path"]).resolve()
            != Path(eligible.loc[s["subject_id"], "t1_path"]).resolve()
            or s["acquisition_site"] != eligible.loc[s["subject_id"], "acquisition_site"]
        ):
            raise ValueError("Preserved cohort differs from eligible manifest")
        v1 = json.loads((baseline / "subjects" / f"{s['subject_id']}.json").read_text())
        a = json.loads((corrected / "subjects" / f"{s['subject_id']}_A.json").read_text())
        if (
            not a["success"]
            or a["fixed_mask_used"]
            or a["initialization_mode"] != "GEOMETRY"
            or sha256(Path(s["t1_path"])) != v1["source_sha256"]
        ):
            raise ValueError("Corrected-A/source provenance mismatch")
    raw = json.loads((corrected / "provenance.json").read_text())["parameters"]
    for name in ("shrink_factors", "smoothing_sigmas_mm"):
        raw[name] = tuple(raw[name])
    p = replace(RigidParameters(**raw), use_fixed_mask=False)
    p.validate()
    expected = RigidParameters(
        iterations_per_level=300, evaluation_shrink_factor=1, use_fixed_mask=False
    )
    if asdict(p) != asdict(expected):
        raise ValueError("Corrected-A shared parameters changed; do not silently reinterpret O0")
    inputs = [Path(s) for s in cohort.t1_path] + [
        config.manifest_path,
        config.geometry_manifest_path,
        config.template_path,
        config.brain_mask_path,
        Path(config_path).resolve(),
    ]
    protected = [
        path
        for directory in (baseline, corrected)
        for path in directory.rglob("*")
        if path.is_file()
    ]
    inputs += protected
    validate_output_directory(output, inputs)
    bundle = load_template(config.template_path, config.brain_mask_path)
    fixed = to_sitk(bundle.image)
    mask = sitk.Cast(to_sitk(bundle.mask), sitk.sitkUInt8)
    return cohort, p, bundle, fixed, mask, inputs


def result_row(r):
    row = {
        k: r.get(k)
        for k in (
            "subject_id",
            "site",
            "optimizer",
            "initialization",
            "success",
            "MI_initial",
            "MI_after_translation",
            "MI_final",
            "delta_translation",
            "delta_rigid",
            "delta_total",
            "runtime_seconds",
            "optimizer_execute_seconds",
            "warnings",
            "qc_path",
            "output_geometry_valid",
            "anatomical_qc_status",
            "trajectory_csv",
        )
    }
    for label, stage in [("translation", "translation_stage"), ("rigid", "rigid_stage")]:
        for field in (
            "iterations",
            "stop",
            "budget_hit",
            "runtime_seconds",
            "final_level_budget_hit",
        ):
            row[label + "_" + field] = r.get(stage, {}).get(field)
    for label, i in [("initial", 0), ("after_translation", 1), ("final", 2)]:
        row["template_brain_in_bounds_fraction_" + label] = r.get("evaluations", [{}, {}, {}])[
            i
        ].get("template_brain_in_bounds_fraction")
    for k in (
        "rotation_deg_lps_xyz",
        "translation_mm_lps_xyz",
        "rotation_determinant",
        "orthonormality_error",
    ):
        row[k] = r.get("final_transform", {}).get(k)
    return {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}


def run_case(subject, key, fixed, moving, mask, p, directory):
    sid = subject["subject_id"]
    start = time.perf_counter()
    record = {
        "subject_id": sid,
        "site": subject["acquisition_site"],
        "optimizer": key,
        "initialization": "GEOMETRY",
        "fixed_metric_mask_used": False,
        "moving_metric_mask_used": False,
        "success": False,
        "output_geometry_valid": False,
        "anatomical_qc_status": "not_reviewed",
        "warnings": [],
        "qc_path": str(directory / "qc" / f"{sid}_optimizers.png"),
    }
    view = None
    try:
        final, trace, rows = optimizers.register(fixed, moving, mask, p, key)
        record.update(trace)
        record["core_registration_seconds"] = trace["runtime_seconds"]
        image = resample_to_fixed(moving, fixed, final)
        record["output_geometry_valid"] = all(
            getattr(image, k)() == getattr(fixed, k)()
            for k in ("GetSize", "GetOrigin", "GetSpacing", "GetDirection")
        )
        if (
            not record["output_geometry_valid"]
            or not np.isfinite(sitk.GetArrayViewFromImage(image)).all()
        ):
            raise ValueError("Invalid registered grid or intensities")
        record["runtime_seconds"] = time.perf_counter() - start
        record["success"] = True
        if record["delta_total"] < 0:
            record["warnings"].append("deterministic_MI_degradation")
        for stage in ("translation", "rigid"):
            if record[stage + "_stage"]["budget_hit"]:
                record["warnings"].append(stage + "_iteration_budget")
        if (
            max(abs(v) for v in record["final_transform"]["rotation_deg_lps_xyz"])
            > p.rotation_warning_deg
        ):
            record["warnings"].append("provisional_large_rotation")
        if (
            np.linalg.norm(record["final_transform"]["translation_mm_lps_xyz"])
            > p.translation_warning_mm
        ):
            record["warnings"].append("provisional_large_pull_translation")
        if record["evaluations"][-1]["template_brain_outside_count"]:
            record["warnings"].append("template_ROI_has_outside_FOV_samples_not_anatomical_error")
        record["trajectory_csv"] = str(directory / "traces" / f"{sid}_{key}.csv")
        pd.DataFrame(rows).to_csv(record["trajectory_csv"], index=False)
        view = planes(image)
    except (RuntimeError, ValueError) as exc:
        record["success"] = False
        record["runtime_seconds"] = time.perf_counter() - start
        record["warnings"].append("execution_failure: " + str(exc))
        print(f"{sid} {key} failed: {exc}", file=sys.stderr, flush=True)
    return record, view


def reproduce_o0(record, reference):
    exact = True
    equivalent = True
    for key in ("initial_transform", "final_transform"):
        for field in ("parameters", "fixed_parameters", "matrix_lps"):
            a, b = record[key][field], reference[key][field]
            exact &= a == b
            equivalent &= bool(np.allclose(a, b, rtol=0, atol=ATOL))
    for key in ("MI_initial", "MI_after_translation", "MI_final"):
        exact &= record[key] == reference[key]
        equivalent &= abs(record[key] - reference[key]) <= ATOL
    stops = all(
        record[s]["level_stops"] == reference[s]["level_stop_conditions"]
        for s in ("translation_stage", "rigid_stage")
    )
    iterations = all(
        record[s]["iterations"] == reference[s]["iterations"]
        for s in ("translation_stage", "rigid_stage")
    )
    return {
        "exact_transforms_and_MI": bool(exact),
        "equivalent_at_1e_10": bool(equivalent),
        "stop_conditions_identical": stops,
        "iterations_identical": iterations,
    }


def write_tables(output):
    records = [json.loads(p.read_text()) for p in sorted((output / "subjects").glob("*.json"))]
    if len(records) != 54 or len({(r["subject_id"], r["optimizer"]) for r in records}) != 54:
        raise ValueError("Expected exactly 18 x 3 results")
    pairs = []
    for r in records:
        r["warnings"] = [w for w in r["warnings"] if not w.startswith("optimizer_disagreement/")]
    for sid in sorted({r["subject_id"] for r in records}):
        cases = {r["optimizer"]: r for r in records if r["subject_id"] == sid}
        initial = [r["initial_transform"] for r in cases.values() if r["success"]]
        if any(x != initial[0] for x in initial):
            raise ValueError("Optimizer choice changed initialization")
        mi = [r["MI_initial"] for r in cases.values() if r["success"]]
        if mi and max(mi) - min(mi) > ATOL:
            raise ValueError("Optimizer choice changed initial metric")
        for a, b in itertools.combinations(optimizers.SPECS, 2):
            if not cases[a]["success"] or not cases[b]["success"]:
                continue
            pair = dict(
                optimizers.disagreement(cases[a]["final_transform"], cases[b]["final_transform"]),
                subject_id=sid,
                site=cases[a]["site"],
                pair=a + "/" + b,
            )
            pairs.append(pair)
            if pair["provisional_review_warning"]:
                for key in (a, b):
                    cases[key]["warnings"].append("optimizer_disagreement/" + a + "/" + b)
    for r in records:
        _json(output / "subjects" / f"{r['subject_id']}_{r['optimizer']}.json", r)
    pd.DataFrame([result_row(r) for r in records]).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(
        [{k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in pairs]
    ).to_csv(output / "transform_disagreement.csv", index=False)
    summary = {
        key: {
            "overall": optimizers.aggregate([r for r in records if r["optimizer"] == key]),
            "by_site": {
                site: optimizers.aggregate(
                    [r for r in records if r["optimizer"] == key and r["site"] == site]
                )
                for site in ("Guys", "HH", "IOP")
            },
        }
        for key in optimizers.SPECS
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
                for site in ("Guys", "HH", "IOP")
            },
        }
        for pair in ("O0/O1", "O0/O2", "O1/O2")
    }
    _json(output / "summary.json", summary)
    return summary


def run(config_path, baseline, corrected, output):
    baseline, corrected, output = map(lambda p: Path(p).resolve(), (baseline, corrected, output))
    cohort, p, bundle, fixed, mask, inputs = load_inputs(config_path, baseline, corrected, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("New ignored output directory must be empty; prior results are read only")
    before = {str(p): sha256(p) for p in inputs}
    for name in ("subjects", "traces", "qc"):
        (output / name).mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output / "selection.csv", index=False)
    _json(
        output / "provenance.json",
        {
            "shared_parameters": asdict(p),
            "optimizer_configurations": optimizers.SPECS,
            "template": bundle.provenance,
            "environment": environment(),
            "core_sha256": core_hashes(),
            "input_sha256": before,
            "mask_role": "coverage/QC only; no fixed or moving metric mask",
            "absolute_equivalence_tolerance": ATOL,
            "disagreement_review_thresholds": {
                "rotation_deg": 5.0,
                "common_center_pull_displacement_mm": 5.0,
            },
            "runtime_definition": (
                "case: initialization/evaluations/coverage/stages/resampling; "
                "excludes source IO, PNG/table export"
            ),
            "stage_runtime_definition": (
                "buffer setup, optimizer setup, execute and trace collection; "
                "execute-only also retained"
            ),
        },
    )
    panels = []
    fixed_planes = planes(fixed)
    mask_planes = planes(mask)
    for n, subject in enumerate(cohort.to_dict("records"), 1):
        sid = subject["subject_id"]
        moving = to_sitk(canonicalize(nib.load(subject["t1_path"])))
        views = {"Fixed": fixed_planes}
        for key in optimizers.SPECS:
            print(f"Optimizer study {n}/18 {sid}: {key}", file=sys.stderr, flush=True)
            r, view = run_case(subject, key, fixed, moving, mask, p, output)
            if key == "O0" and r["success"]:
                reference = json.loads((corrected / "subjects" / f"{sid}_A.json").read_text())
                r["corrected_A_reproduction"] = reproduce_o0(r, reference)
                if not all(r["corrected_A_reproduction"].values()):
                    _json(output / "subjects" / f"{sid}_{key}.json", r)
                    raise ValueError("O0 failed corrected-A reproduction; stop experiment")
            _json(output / "subjects" / f"{sid}_{key}.json", r)
            if view is not None:
                views[key] = view
        subject_panel(sid, views, mask_planes, output / "qc" / f"{sid}_optimizers.png")
        panels.append((sid, subject["acquisition_site"], views))
    contact_sheets(panels, mask_planes, output / "qc")
    summary = write_tables(output)
    if any(sha256(Path(path)) != h for path, h in before.items()):
        raise ValueError("Prior inputs/results changed")
    print(f"Completed 18-subject optimizer comparison: {output}")
    return summary


def repeat(config_path, baseline, corrected, output, key):
    baseline, corrected, output = map(lambda p: Path(p).resolve(), (baseline, corrected, output))
    cohort, p, bundle, fixed, mask, inputs = load_inputs(config_path, baseline, corrected, output)
    provenance = json.loads((output / "provenance.json").read_text())
    if (
        environment() != provenance["environment"]
        or core_hashes() != provenance["core_sha256"]
        or json.loads(json.dumps(asdict(p))) != provenance["shared_parameters"]
    ):
        raise ValueError("Repeat environment/implementation/shared configuration changed")
    if any(sha256(Path(path)) != h for path, h in provenance["input_sha256"].items()):
        raise ValueError("Repeat inputs differ from original experiment")
    if optimizers.SPECS != provenance["optimizer_configurations"]:
        raise ValueError("Repeat optimizer configuration changed")
    target = output / "repeat" / key
    validate_output_directory(target, [Path(s) for s in cohort.t1_path] + inputs)
    if target.exists():
        raise ValueError("Repeat must not overwrite an earlier repeat")
    for name in ("subjects", "traces", "qc"):
        (target / name).mkdir(parents=True, exist_ok=True)
    reports = []
    for n, subject in enumerate(cohort.to_dict("records"), 1):
        sid = subject["subject_id"]
        print(f"Repeating full candidate {key}: {n}/18 {sid}", file=sys.stderr, flush=True)
        moving = to_sitk(canonicalize(nib.load(subject["t1_path"])))
        r, _ = run_case(subject, key, fixed, moving, mask, p, target)
        _json(target / "subjects" / f"{sid}_{key}.json", r)
        original = json.loads((output / "subjects" / f"{sid}_{key}.json").read_text())
        if not r["success"] or not original["success"]:
            reports.append({"subject_id": sid, "passed": False, "reason": "execution failure"})
            continue
        exact = True
        equivalent = True
        max_delta = 0.0
        for field in ("initial_transform", "final_transform"):
            for attr in ("parameters", "fixed_parameters", "matrix_lps"):
                a, b = r[field][attr], original[field][attr]
                exact &= a == b
                max_delta = max(max_delta, float(np.max(np.abs(np.array(a) - b))))
                equivalent &= bool(np.allclose(a, b, rtol=0, atol=ATOL))
        for field in ("MI_initial", "MI_after_translation", "MI_final"):
            exact &= r[field] == original[field]
            equivalent &= abs(r[field] - original[field]) <= ATOL
            max_delta = max(max_delta, abs(r[field] - original[field]))
        same_stops = all(
            r[s]["level_stops"] == original[s]["level_stops"]
            for s in ("translation_stage", "rigid_stage")
        )
        same_trace = sha256(Path(r["trajectory_csv"])) == sha256(Path(original["trajectory_csv"]))
        reports.append(
            {
                "subject_id": sid,
                "exact_transform_and_MI": bool(exact),
                "equivalent_transform_and_MI": bool(equivalent),
                "max_absolute_difference": max_delta,
                "identical_stop_conditions": same_stops,
                "identical_sampled_trajectory_CSV": same_trace,
                "passed": bool(equivalent and same_stops),
                "runtime_seconds": r["runtime_seconds"],
            }
        )
    report = {
        "optimizer": key,
        "subjects": reports,
        "absolute_tolerance": ATOL,
        "relative_tolerance": 0.0,
        "all_passed": all(r["passed"] for r in reports),
        "environment": environment(),
    }
    _json(output / "repeatability.json", report)
    if not report["all_passed"]:
        raise ValueError("Full-cohort repeatability failed")
    print(f"Full {key} repeat passed for all eighteen subjects")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "baseline", "corrected", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repeat", choices=list(optimizers.SPECS))
    args = parser.parse_args()
    if args.repeat:
        repeat(args.config, args.baseline, args.corrected, args.output, args.repeat)
    else:
        run(args.config, args.baseline, args.corrected, args.output)


if __name__ == "__main__":
    main()
