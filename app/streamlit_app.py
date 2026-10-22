"""Local academic demo; scientific logic lives in brainage_xai.inference."""

import os
from pathlib import Path

import streamlit as st

from brainage_xai.inference.config import ROOT, load_runtime_config
from brainage_xai.inference.pipeline import infer_upload
from brainage_xai.inference.result import AGE_WARNING, LIMITATIONS, RESEARCH_DISCLAIMER, XAI_WORDING


def display_result(bundle):
    result = bundle.result
    st.divider()
    st.subheader("Research-model result")
    st.caption(f"Analysis {result.analysis_id} · {result.timestamp_utc}")
    if AGE_WARNING in result.warnings:
        st.warning(AGE_WARNING)
    a, b, c = st.columns(3)
    a.metric("Chronological age", f"{result.chronological_age:.2f} years")
    b.metric("Raw predicted brain age", f"{result.raw_predicted_brain_age:.2f} years")
    c.metric("Corrected brain age", f"{result.corrected_brain_age:.2f} years")
    st.caption("Corrected brain age is a bias-adjusted model estimate, not a medical measurement.")
    left, right = st.columns([1, 2])
    left.metric("Corrected BAG", f"{result.corrected_BAG:+.2f} years")
    left.caption(f"Raw BAG: {result.raw_BAG:+.2f} years")
    right.write(result.interpretation)
    right.write(f"Research-model reference percentile: **{result.reference_percentile:.1f}**")
    right.progress(min(1.0, max(0.0, result.reference_percentile / 100)))
    right.caption("Validation reference position; not a clinical normal range or disease risk.")
    interval = result.empirical_uncertainty
    st.info(
        f"Empirical model uncertainty/reference interval: "
        f"{interval['lower_years']:.2f}–{interval['upper_years']:.2f} years. "
        "Based on raw predicted age and validation residuals; not a clinical confidence interval."
    )
    status, gallery, context, provenance = st.tabs(
        ["Preprocessing & QC", "Attribution gallery", "Research limits", "Provenance"]
    )
    with status:
        st.write("Input/preprocessing: **" + result.preprocessing_qc["status"] + "**")
        st.write("Rigid registration: **" + result.registration_qc["status"] + "**")
        st.write("193 × 229 × 193 · 1 mm · MNI152NLin2009cAsym · S2/O0 rigid")
        st.caption("Numerical QC does not certify anatomical alignment. Inspect the views below.")
        with st.expander("Warnings and QC details"):
            for warning in result.warnings:
                st.write("• " + warning)
            st.json(
                {"preprocessing": result.preprocessing_qc, "registration": result.registration_qc}
            )
    with gallery:
        st.write(XAI_WORDING)
        st.caption(
            "Zero maps and background/nonbrain attribution may occur. Attribution is "
            "not causal or diagnostic; colors are normalized independently per slice."
        )
        columns = st.columns(3)
        representatives = [r for r in result.xai["slices"] if r["representative"]]
        for column, record in zip(columns, representatives, strict=True):
            column.image(
                bundle.files[record["image"]],
                caption=f"{record['view'].title()} · MNI index {record['mni_slice_index']}",
            )
        with st.expander("Inspect all 15 fixed input slices"):
            for view in ("axial", "coronal", "sagittal"):
                st.markdown(f"**{view.title()}**")
                columns = st.columns(5)
                records = [r for r in result.xai["slices"] if r["view"] == view]
                for column, record in zip(columns, records, strict=True):
                    column.image(
                        bundle.files[record["image"]], caption=f"MNI {record['mni_slice_index']}"
                    )
    with context:
        st.write(RESEARCH_DISCLAIMER)
        st.write(
            "Measured IXI locked-test MAE **6.094 years**; overall MAE bootstrap "
            "95% CI **5.154–7.101 years**. These are cohort-level performance estimates."
        )
        for limitation in LIMITATIONS:
            st.write("• " + limitation)
    with provenance:
        st.json(result.provenance)
    st.subheader("Download this analysis")
    st.caption(
        "Reports contain MRI-derived images and sensitive model estimates. Downloads "
        "are explicit; the app does not save them permanently."
    )
    a, b, c = st.columns(3)
    prefix = "brainage-" + result.analysis_id
    a.download_button("PDF report", bundle.files["report.pdf"], prefix + ".pdf", "application/pdf")
    b.download_button(
        "Result JSON", bundle.files["result.json"], prefix + ".json", "application/json"
    )
    c.download_button(
        "Complete XAI package", bundle.zip_bytes(), prefix + ".zip", "application/zip"
    )


def main():
    st.set_page_config(page_title="BrainAge-XAI · Research", page_icon="🧠", layout="wide")
    st.markdown(
        "<style>.block-container{max-width:1180px;padding-top:2rem}"
        "[data-testid=stMetric]{padding:1rem;border:1px solid #dce4eb;"
        "border-radius:8px;background:white} footer{visibility:hidden}</style>",
        unsafe_allow_html=True,
    )
    st.caption("STRUCTURAL MRI · ACADEMIC RESEARCH PROTOTYPE")
    st.title("BrainAge-XAI")
    st.write("A transparent, local workflow for research brain-age estimates and attribution.")
    st.info(RESEARCH_DISCLAIMER)
    config_path = Path(
        os.environ.get("BRAINAGE_INFERENCE_CONFIG", str(ROOT / "configs/inference.local.yaml"))
    )
    ready, config = True, None
    try:
        config = load_runtime_config(config_path)
    except (ValueError, OSError, KeyError) as exc:
        ready = False
        st.warning(
            "Local runtime is not ready. Supply the accepted checkpoint, template "
            "and validation reference artifacts using the setup guide."
        )
        with st.expander("Local setup details"):
            st.code(str(exc))
            st.write("See docs/demo.md. No model or dataset is downloaded automatically.")
    with st.sidebar:
        st.markdown("### Research context")
        st.write("Frozen 2.5D multi-view ResNet18")
        st.write("IXI · 393 train / 84 validation / 84 test")
        st.write("Validation MAE 5.428 years")
        st.write("Locked-test MAE 6.094 years")
        st.divider()
        st.caption(
            "Local processing. No accounts, database or cloud upload. Temporary MRI "
            "copies are removed after success or failure. Derived results remain "
            "in this browser session until cleared."
        )
        if st.button("Clear analysis and uploaded data"):
            st.session_state.pop("analysis_bundle", None)
            st.session_state["upload_generation"] = st.session_state.get("upload_generation", 0) + 1
            st.rerun()
    generation = st.session_state.get("upload_generation", 0)
    st.subheader("1 · Provide an MRI and chronological age")
    with st.form("analysis", clear_on_submit=True):
        upload = st.file_uploader(
            "T1-weighted structural MRI (.nii or .nii.gz)",
            type=["nii", "gz"],
            key=f"upload_{generation}",
            help="One 3D MRI. Filename/site/sex are not model inputs. "
            "Modality cannot be verified automatically.",
        )
        age = st.number_input(
            "Chronological age (years)",
            min_value=0.0,
            value=None,
            step=0.1,
            placeholder="Enter age for BAG calculation",
        )
        if config:
            ranges = config.contract["age_ranges_years"]
            st.caption(
                f"Observed training ages {ranges['train'][0]:.2f}–{ranges['train'][1]:.2f}; "
                f"validation reference ages {ranges['validation'][0]:.2f}–"
                f"{ranges['validation'][1]:.2f}. Outside-range inference is extrapolative."
            )
        submitted = st.form_submit_button(
            "2 · Run BrainAge-XAI analysis", disabled=not ready, type="primary"
        )
    if submitted:
        st.session_state.pop("analysis_bundle", None)
        if upload is None or age is None:
            st.warning("Provide one MRI and chronological age before analysis.")
        else:
            try:
                with st.status("Analyzing MRI locally", expanded=True) as status:
                    bundle = infer_upload(
                        upload.getvalue(), upload.name, age, config_path, progress=st.write
                    )
                    st.session_state["analysis_bundle"] = bundle
                    status.update(
                        label="Analysis complete · temporary MRI removed", state="complete"
                    )
            except (ValueError, OSError, RuntimeError) as exc:
                st.session_state["analysis_error"] = str(exc)
            st.session_state["upload_generation"] = generation + 1
            st.rerun()
    if "analysis_error" in st.session_state:
        st.error(st.session_state.pop("analysis_error"))
        st.caption("No result was produced. Temporary processing files were cleaned.")
    if "analysis_bundle" in st.session_state:
        display_result(st.session_state["analysis_bundle"])


if __name__ == "__main__":
    main()
