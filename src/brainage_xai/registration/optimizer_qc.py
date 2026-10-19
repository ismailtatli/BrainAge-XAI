"""Optimizer QC: fixed contours, internal intensity edges, and matched checkerboards."""

from pathlib import Path

import numpy as np

from .qc import _window
from .tuning_qc import draw


def _panel(rows, fixed, masks, path, title, checkerboards=False):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 2.9 * len(rows)), squeeze=False)
    try:
        for row, (label, view) in enumerate(rows):
            for axis in range(3):
                if view is None:
                    axes[row, axis].text(0.1, 0.5, label + " failed")
                    axes[row, axis].axis("off")
                    continue
                plane = view[axis]
                if checkerboards and "checkerboard" in label:
                    a, b = _window(fixed[axis]), _window(plane)
                    y, x = np.indices(a.shape)
                    plane = np.where(((x // 10 + y // 10) % 2) == 0, a, b)
                draw(axes[row, axis], plane, masks[axis], label, axis)
                if "checkerboard" not in label:
                    axes[row, axis].contour(
                        _window(fixed[axis]),
                        levels=[0.2, 0.5, 0.8],
                        colors=["cyan"],
                        linewidths=0.25,
                        alpha=0.5,
                    )
        fig.suptitle(
            title + "\nGreen=fixed brain ROI; cyan=fixed intensity edges; visual QC only\n"
            "Fixed-template LPS → moving-source LPS pull mapping",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95 if len(rows) < 10 else 0.98))
        fig.savefig(path, dpi=95 if len(rows) < 10 else 85)
    finally:
        plt.close(fig)


def subject_panel(subject, views, masks, path):
    rows = [("Fixed MNI", views["Fixed"])]
    for key in ("O0", "O1", "O2"):
        rows.extend([(key, views.get(key)), (key + " / fixed checkerboard", views.get(key))])
    _panel(rows, views["Fixed"], masks, path, subject + " — optimizer comparison", True)


def contact_sheets(panels, masks, output):
    output = Path(output)
    fixed = panels[0][2]["Fixed"]
    for site in ("Guys", "HH", "IOP"):
        rows = [("Fixed MNI", fixed)] + [
            (sid + " " + key, views.get(key))
            for sid, s, views in panels
            if s == site
            for key in ("O0", "O1", "O2")
        ]
        _panel(rows, fixed, masks, output / f"site_{site}.png", site + " optimizer comparison")
    for key in ("O0", "O1", "O2"):
        rows = [("Fixed MNI", fixed)] + [
            (sid + " " + site + " " + key, views.get(key)) for sid, site, views in panels
        ]
        _panel(rows, fixed, masks, output / f"optimizer_{key}.png", key + " all subjects")
