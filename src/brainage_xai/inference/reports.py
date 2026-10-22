"""Local report/XAI rendering; identifiers and source MRI files are never embedded."""

import io
import threading
import zipfile
from dataclasses import dataclass
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as RasterImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .result import LIMITATIONS, XAI_WORDING, InferenceResult

_RENDER_LOCK = threading.RLock()


@dataclass
class AnalysisBundle:
    result: InferenceResult
    files: dict[str, bytes]

    def zip_bytes(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in self.files.items():
                archive.writestr(name, data)
        return buffer.getvalue()


def render_xai(gray, maps, indices):
    """All fifteen fixed slices; representative images always use each view's middle slice."""
    files, records = {}, []
    with _RENDER_LOCK:
        for vi, view in enumerate(("axial", "coronal", "sagittal")):
            for j, index in enumerate(indices[view]):
                offset = vi * 5 + j
                image = gray[offset, 0]
                cam = maps[offset]
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
                axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
                axes[1].imshow(cam, cmap="inferno", vmin=0, vmax=1, alpha=0.5)
                axes[0].set_title(f"{view} · MNI index {index}")
                axes[1].set_title("Grad-CAM overlay")
                for axis in axes:
                    axis.axis("off")
                fig.tight_layout(pad=0.3)
                data = io.BytesIO()
                fig.savefig(data, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)
                name = f"xai/{view}-{index:03d}.png"
                files[name] = data.getvalue()
                records.append(
                    {
                        "view": view,
                        "mni_slice_index": int(index),
                        "input_offset": offset,
                        "image": name,
                        "representative": j == 2,
                        "zero_positive_map": bool(cam.max() == 0),
                    }
                )
    raw = io.BytesIO()
    np.save(raw, maps, allow_pickle=False)
    files["xai/gradcam.npy"] = raw.getvalue()
    return files, records


def pdf_report(result, images):
    """A research-only PDF with the same numerical result and three fixed representative views."""
    with _RENDER_LOCK:
        from pathlib import Path

        font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        if "BrainAgeSans" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("BrainAgeSans", str(font)))
        styles = getSampleStyleSheet()
        for style in styles.byName.values():
            style.fontName = "BrainAgeSans"
        styles.add(
            ParagraphStyle(
                "SmallResearch",
                fontName="BrainAgeSans",
                fontSize=8,
                leading=11,
                textColor=colors.HexColor("#475569"),
            )
        )
        body = []

        def paragraph(text, style="BodyText"):
            body.extend([Paragraph(escape(str(text)), styles[style]), Spacer(1, 7)])

        paragraph("BrainAge-XAI", "Title")
        paragraph("Individual research-model report", "Heading2")
        paragraph(result.disclaimer)
        paragraph(
            f"Analysis {result.analysis_id} | {result.timestamp_utc} | "
            f"Application {result.application_version}",
            "SmallResearch",
        )
        paragraph("Input and preprocessing", "Heading2")
        paragraph(
            f"Input validation and preprocessing: {result.preprocessing_qc['status']}. "
            f"Rigid registration: {result.registration_qc['status']}. "
            "Exact MNI grid: 193 × 229 × 193 at 1 mm; no crop/pad."
        )
        entries = [
            ("Chronological age", f"{result.chronological_age:.2f} years"),
            ("Raw predicted brain age", f"{result.raw_predicted_brain_age:.2f} years"),
            ("Corrected brain age (model estimate)", f"{result.corrected_brain_age:.2f} years"),
            ("Raw BAG", f"{result.raw_BAG:+.2f} years"),
            ("Corrected BAG", f"{result.corrected_BAG:+.2f} years"),
            ("Research-model reference percentile", f"{result.reference_percentile:.1f}"),
        ]
        table = Table(
            [
                [Paragraph(k, styles["BodyText"]), Paragraph(v, styles["BodyText"])]
                for k, v in entries
            ],
            colWidths=[3.55 * inch, 2.4 * inch],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        body.extend([table, Spacer(1, 12)])
        paragraph(result.interpretation)
        interval = result.empirical_uncertainty
        paragraph(
            f"Empirical model uncertainty/reference interval based on raw predicted age: "
            f"{interval['lower_years']:.2f}–{interval['upper_years']:.2f} years. "
            + interval["scope"]
        )
        paragraph("Warnings and limitations", "Heading2")
        for warning in dict.fromkeys(result.warnings + LIMITATIONS):
            paragraph(warning, "SmallResearch")
        paragraph("Representative attributions", "Heading2")
        paragraph(XAI_WORDING)
        for record in result.xai["slices"]:
            if record["representative"]:
                png = images[record["image"]]
                with RasterImage.open(io.BytesIO(png)) as raster:
                    width, height = raster.size
                body.append(
                    Image(io.BytesIO(png), width=5.5 * inch, height=5.5 * inch * height / width)
                )
                body.append(Spacer(1, 6))
        paragraph("Measured model performance context", "Heading2")
        paragraph(
            "IXI locked test (84 subjects): raw MAE 6.094 years; 95% bootstrap CI "
            "for overall mean absolute error 5.154–7.101 years. RMSE 7.563; R² 0.796. "
            "This cohort-level CI is not an individual uncertainty interval."
        )
        paragraph("Provenance", "Heading2")
        for key in (
            "checkpoint_sha256",
            "contract_sha256",
            "config_sha256",
            "reference_sha256",
            "correction_sha256",
        ):
            paragraph(f"{key}: {result.provenance[key]}", "SmallResearch")
        paragraph(
            "No source filename, participant identity or source NIfTI is embedded. "
            "XAI panels are MRI-derived and sensitive; share only with appropriate permission.",
            "SmallResearch",
        )
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            title="BrainAge-XAI research report",
            author="BrainAge-XAI",
            subject="Non-diagnostic individual research-model result",
            leftMargin=45,
            rightMargin=45,
            topMargin=40,
            bottomMargin=40,
        )

        def footer(canvas, document):
            canvas.setFont("BrainAgeSans", 8)
            canvas.drawString(45, 22, "Research only · BAG is not a diagnosis")
            canvas.drawRightString(550, 22, str(document.page))

        doc.build(body, onFirstPage=footer, onLaterPages=footer)
        return buffer.getvalue()
