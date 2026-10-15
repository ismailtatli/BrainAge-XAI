"""Synthetic curation tests; source rows and MRI validation remain independent."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from brainage_xai.data.audit import summarize
from brainage_xai.data.discovery import parse_acquisition_site
from brainage_xai.data.matching import build_manifest
from brainage_xai.data.metadata import normalize_date, normalize_metadata


def records(**overrides):
    values = {
        "IXI_ID": [901, 901],
        "AGE": [42.25, 42.25],
        "SEX": [1, 1],
        "DOB": ["1960-01-02", datetime(1960, 1, 2)],
        "STUDY_DATE": [datetime(2002, 4, 2), "2002-04-02"],
        "HEIGHT": [170, 170],
    }
    values.update(overrides)
    return pd.DataFrame(values)


def manifest(frame, *, mri=True, valid=True, validated=True, duplicate_mri=False):
    paths = [Path("IXI901-Guys-1234-T1.nii.gz")] if mri else []
    if duplicate_mri:
        paths.append(Path("IXI901-HH-1235-T1.nii.gz"))
    checks = [{"t1_path": str(path), "valid": valid, "errors": []} for path in paths]
    return build_manifest(paths, normalize_metadata(frame), checks if validated else None)


def test_exact_duplicate():
    frame = records(DOB=["1960-01-02"] * 2, STUDY_DATE=["2002-04-02"] * 2)
    original = frame.copy(deep=True)
    result = manifest(frame).iloc[0]
    assert result.metadata_resolution_status == "duplicate_label_consistent"
    assert result.brain_age_label_eligible
    assert result.age == 42.25
    assert result.sex == "1"
    assert result.dob == "1960-01-02"
    assert result.study_date == "2002-04-02"
    assert result.metadata_rows == [2, 3]
    assert result.ancillary_conflict_fields == []
    pd.testing.assert_frame_equal(frame, original)


def test_ancillary_conflict_does_not_merge_or_change_label():
    frame = records(HEIGHT=[170, 0], WEIGHT=[None, 70], ETHNIC_ID=[1, 0])
    normalized = normalize_metadata(frame)
    assert list(normalized["source:HEIGHT"]) == [170, 0]
    assert len(normalized) == len(frame)
    row = manifest(frame).iloc[0]
    assert row.metadata_resolution_status == "duplicate_label_consistent"
    assert row.brain_age_label_eligible
    assert row.ancillary_conflict_fields == ["ETHNIC_ID", "HEIGHT", "WEIGHT"]


@pytest.mark.parametrize(
    "column,values,field",
    [
        ("AGE", [42.25, 43.25], "age"),
        ("SEX", [1, 2], "sex"),
        ("DOB", ["1960-01-02", "1961-01-02"], "dob"),
        ("STUDY_DATE", ["2002-04-02", "2003-04-02"], "study_date"),
    ],
)
def test_conflicting_labels(column, values, field):
    result = manifest(records(**{column: values})).iloc[0]
    assert result.metadata_resolution_status == "duplicate_label_ambiguous"
    assert not result.brain_age_label_eligible
    assert result.label_conflict_fields == [field]
    assert "ambiguous_metadata_label" in result.exclusion_reasons
    assert pd.isna(result.age)


@pytest.mark.parametrize("ages", [[None, None], [42.25, None]])
def test_missing_age_duplicates_are_not_imputed(ages):
    result = manifest(records(AGE=ages)).iloc[0]
    assert result.metadata_resolution_status == "missing_age"
    assert not result.brain_age_label_eligible
    assert pd.isna(result.age)
    assert "missing_age" in result.exclusion_reasons


def test_missing_dates_are_not_invented():
    result = manifest(records(DOB=[None, None], STUDY_DATE=[None, None])).iloc[0]
    assert result.dob is None
    assert result.study_date is None
    assert result.brain_age_label_eligible
    partial = manifest(records(STUDY_DATE=[None, "2002-04-02"])).iloc[0]
    assert partial.study_date == "2002-04-02"
    assert partial.brain_age_label_eligible


@pytest.mark.parametrize(
    "frame,options,reason",
    [
        (records().iloc[:0], {}, "no_demographics"),
        (records(), {"mri": False}, "no_mri"),
        (records(), {"valid": False}, "mri_validation_failure"),
        (records(), {"validated": False}, "mri_not_validated"),
        (records(), {"duplicate_mri": True}, "duplicate_mri"),
        (records(AGE=["broken", "broken"]), {}, "invalid_age"),
        (records(DOB=["nonsense", "nonsense"]), {}, "invalid_identity_metadata"),
    ],
)
def test_eligibility_exclusions(frame, options, reason):
    result = manifest(frame, **options).iloc[0]
    assert not result.brain_age_label_eligible
    assert reason in result.exclusion_reasons


def test_unique_valid_label_and_invalid_identity_date():
    row = manifest(records().iloc[:1]).iloc[0]
    assert row.metadata_resolution_status == "unique"
    assert row.brain_age_label_eligible
    row = manifest(records(DOB=["bad", "bad"]).iloc[:1]).iloc[0]
    assert not row.brain_age_label_eligible
    assert "invalid_identity_metadata" in row.exclusion_reasons


@pytest.mark.parametrize(
    "value,expected,status",
    [
        ("1960-02-29", "1960-02-29", "valid"),
        (datetime(2000, 3, 2), "2000-03-02", "valid"),
        ("2000-03-02T12:30:00", "2000-03-02", "valid"),
        (None, None, "missing"),
        (" ", None, "missing"),
        (pd.NaT, None, "missing"),
        ("1961-02-29", None, "invalid"),
        ("02/03/1960", None, "invalid"),
        ("1960", None, "invalid"),
        (22000, None, "invalid"),
    ],
)
def test_date_normalization(value, expected, status):
    assert normalize_date(value) == (expected, status)


def test_explicit_date_format_and_configurable_columns():
    frame = pd.DataFrame(
        {"ID": [901], "TARGET": [40], "BIRTH": ["02/03/1960"], "SCAN_DATE": ["04/05/2000"]}
    )
    result = normalize_metadata(
        frame,
        id_column="ID",
        age_column="TARGET",
        dob_column="BIRTH",
        study_date_column="SCAN_DATE",
        date_format="%d/%m/%Y",
    )
    assert result.iloc[0].dob == "1960-03-02"
    assert result.iloc[0].study_date == "2000-05-04"
    assert normalize_date("1960", "%Y") == (None, "invalid")


@pytest.mark.parametrize(
    "name,site",
    [
        ("IXI901-Guys-0828-T1.nii.gz", "Guys"),
        ("IXI902-HH-1211-T1.nii", "HH"),
        ("ixi903-iop-0873-t1.nii.gz", "IOP"),
        ("bad.nii.gz", None),
        ("IXI901-NewSite-0828-T1.nii.gz", None),
        ("IXI901-Guys-T1.nii", None),
        ("IXI901-Guys-1234-T2.nii", None),
        ("IXI000-Guys-0828-T1.nii.gz", None),
    ],
)
def test_filename_site(name, site):
    assert parse_acquisition_site(name) == site


def test_site_does_not_infer_scanner():
    row = manifest(records()).iloc[0]
    assert row.acquisition_site == "Guys"
    assert row.scanner is None


def test_generated_subject_ids_use_same_resolution_policy():
    for subject in (700, 803, 1090):
        frame = normalize_metadata(records(IXI_ID=[subject, subject], SEX=[1.0, "1"]))
        path = Path(f"IXI{subject:03d}-IOP-3456-T1.nii.gz")
        row = build_manifest(
            [path], frame, [{"t1_path": str(path), "valid": True, "errors": []}]
        ).iloc[0]
        assert row.metadata_resolution_status == "duplicate_label_consistent"
        assert row.brain_age_label_eligible


def test_resolution_summary_counts():
    frame = normalize_metadata(
        pd.DataFrame(
            records().to_dict("records")
            + records(IXI_ID=[902, 902], AGE=[30, 40]).to_dict("records")
            + records(IXI_ID=[903, 903], AGE=[None, None]).to_dict("records")
        )
    )
    paths = [Path(f"IXI{i}-HH-1234-T1.nii.gz") for i in (901, 902, 903)]
    checks = [
        {"t1_path": str(p), "valid": True, "errors": [], "warnings": [], "shape": [3, 4, 5]}
        for p in paths
    ]
    result = summarize(build_manifest(paths, frame, checks), frame, checks)
    assert result["brain_age_label_eligible_count"] == 1
    assert result["brain_age_label_ineligible_count"] == 2
    assert result["duplicate_label_consistent_subject_count"] == 1
    assert result["duplicate_label_ambiguous_subject_count"] == 1
    assert result["eligibility_exclusion_reason_counts"] == {
        "ambiguous_metadata_label": 1,
        "missing_age": 1,
    }
    assert result["acquisition_site_distribution"] == {"HH": 3}


def test_source_records_export_keeps_all_values_and_provenance(tmp_path):
    from brainage_xai.data.audit import _write_csv
    from brainage_xai.data.metadata import load_metadata

    original = records(HEIGHT=[170, 0], WEIGHT=[60, 0])
    spreadsheet = tmp_path / "source.xlsx"
    original.to_excel(spreadsheet, index=False)
    before = spreadsheet.read_bytes()
    normalized = load_metadata(spreadsheet)
    output = tmp_path / "metadata_records.csv"
    _write_csv(normalized, output)
    exported = pd.read_csv(output)
    assert exported.metadata_row.tolist() == [2, 3]
    assert len(exported) == len(original)
    for column in original:
        assert f"source:{column}" in exported
    assert exported["source:HEIGHT"].tolist() == [170, 0]
    assert exported["source:WEIGHT"].tolist() == [60, 0]
    assert spreadsheet.read_bytes() == before


@pytest.mark.parametrize("value", ["%Y", "%m/%d", 12])
def test_date_format_configuration_validation(tmp_path, value):
    import yaml

    from brainage_xai.config import load_config

    (tmp_path / "mri").mkdir()
    spreadsheet = tmp_path / "demo.xlsx"
    records().to_excel(spreadsheet, index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "t1_dir": "mri",
                "demographics_path": "demo.xlsx",
                "output_dir": "out",
                "metadata": {"date_format": value},
            }
        )
    )
    with pytest.raises(ValueError):
        load_config(config)
