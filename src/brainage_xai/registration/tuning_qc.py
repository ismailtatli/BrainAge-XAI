"""Matched template-grid QC using only three retained planes per registered image."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .qc import _window


def planes(image):
    data = sitk.GetArrayViewFromImage(image).transpose(2, 1, 0)
    return [np.take(data, data.shape[axis] // 2, axis=axis).T.copy() for axis in range(3)]


def draw(ax, plane, mask, label, axis):
    ax.imshow(_window(plane), origin="lower", cmap="gray", vmin=0, vmax=1)
    ax.contour(mask, levels=[0.5], colors=["lime"], linewidths=0.4)
    ax.set_title(label, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("P → A" if axis == 0 else "L → R (neurological)", fontsize=7)
    ax.set_ylabel("I → S" if axis != 2 else "P → A", fontsize=7)


def subject_panel(views, masks, path, subject):
    import matplotlib.pyplot as plt

    labels = {
        "Fixed": "Fixed MNI",
        "v1": "Baseline v1: GEOMETRY + mask",
        "A": "A: GEOMETRY, unmasked",
        "B": "B: GEOMETRY, fixed brain mask",
        "C": "C: MOMENTS, unmasked",
        "D": "D: MOMENTS, fixed brain mask",
    }
    fig, axes = plt.subplots(6, 3, figsize=(12, 19))
    try:
        for row, key in enumerate(labels):
            for axis in range(3):
                if key in views:
                    draw(
                        axes[row, axis],
                        views[key][axis],
                        masks[axis],
                        labels[key] + " / " + ("Sagittal", "Coronal", "Axial")[axis],
                        axis,
                    )
                else:
                    axes[row, axis].text(0.2, 0.5, "Execution failed")
        fig.suptitle(
            f"{subject}: matched template-center planes; green = fixed brain ROI\n"
            "Fixed-template LPS → moving-source LPS pull transform; anatomical QC pending"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)


def contact_sheets(panels, masks, output):
    import matplotlib.pyplot as plt

    # Sagittal/coronal expose S/I boundaries; axial exposes L/R and cortical mismatch.
    def save(rows, path):
        fig, axes = plt.subplots(len(rows), 3, figsize=(10, 2.8 * len(rows)), squeeze=False)
        try:
            for i, (label, view) in enumerate(rows):
                for axis in range(3):
                    draw(axes[i, axis], view[axis], masks[axis], label, axis)
            fig.suptitle("Matched template-center planes; green=fixed ROI; visual QC only")
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            fig.savefig(path, dpi=85)
        finally:
            plt.close(fig)

    for site in ("Guys", "HH", "IOP"):
        rows = [
            (sid + " " + c, views[c])
            for sid, s, views in panels
            if s == site
            for c in ("Fixed", "v1", "A", "B", "C", "D")
            if c in views
        ]
        save(rows, Path(output) / f"site_{site}.png")
    for c in ("v1", "A", "B", "C", "D"):
        rows = [("Fixed MNI", panels[0][2]["Fixed"])] + [
            (sid + " " + site + " " + c, views[c]) for sid, site, views in panels if c in views
        ]
        save(rows, Path(output) / f"condition_{c}.png")
