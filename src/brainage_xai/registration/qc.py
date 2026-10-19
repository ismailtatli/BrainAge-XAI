"""Private multi-planar rigid-registration QC, with explicit orientation and FOV labels."""

from pathlib import Path

import nibabel as nib
import numpy as np


def _window(plane: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(plane, [1, 99])
    return np.clip((plane - lo) / max(hi - lo, np.finfo(float).eps), 0, 1)


def registration_qc(
    fixed: nib.spatialimages.SpatialImage,
    baseline: nib.spatialimages.SpatialImage,
    initialized: nib.spatialimages.SpatialImage,
    registered: nib.spatialimages.SpatialImage,
    path: Path,
    *,
    dpi: int = 110,
    checkerboard_pixels: int = 12,
    clipping_fraction: float = 0.0,
) -> None:
    """Baseline shows its own FOV center; initialized/final/checkerboard share template center.

    Full sagittal/coronal fields expose superior/inferior boundaries. Individual
    windows aid visual spatial comparison, not quantitative intensity calibration.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 3, figsize=(13, 18))
    labels = [
        "Fixed MNI template",
        "Current baseline (own FOV center)",
        "Moving before optimization (center-initialized)",
        "After rigid registration",
        "Fixed / after checkerboard",
    ]
    fixed_planes = []
    registered_planes = []
    try:
        for row, image in enumerate((fixed, baseline, initialized, registered)):
            if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
                raise ValueError("Registration QC requires RAS+ images")
            spacing = nib.affines.voxel_sizes(image.affine)
            for axis, name in enumerate(("Sagittal", "Coronal", "Axial")):
                plane = np.take(np.asarray(image.dataobj), image.shape[axis] // 2, axis=axis).T
                axes[row, axis].imshow(
                    _window(plane),
                    origin="lower",
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    aspect=spacing[[i for i in range(3) if i != axis][1]]
                    / spacing[[i for i in range(3) if i != axis][0]],
                )
                axes[row, axis].set_title(f"{labels[row]}\n{name}", fontsize=10)
                if row == 0:
                    fixed_planes.append(_window(plane))
                if row == 3:
                    registered_planes.append(_window(plane))
        for axis in range(3):
            a, b = fixed_planes[axis], registered_planes[axis]
            if a.shape != b.shape:
                raise ValueError("QC checkerboard requires matching template grids")
            y, x = np.indices(a.shape)
            check = ((y // checkerboard_pixels + x // checkerboard_pixels) % 2) == 0
            axes[4, axis].imshow(np.where(check, a, b), origin="lower", cmap="gray", vmin=0, vmax=1)
            axes[4, axis].set_title(labels[4], fontsize=10)
        for row in range(5):
            for axis in range(3):
                ax = axes[row, axis]
                ax.set_xlabel("Posterior ←  → Anterior" if axis == 0 else "L ←  → R (neurological)")
                ax.set_ylabel("Inferior ←  → Superior" if axis != 2 else "Posterior ←  → Anterior")
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(
            "Registration QC only — no medical interpretation\n"
            f"Native non-background source outside template FOV: {clipping_fraction:.1%} "
            "(includes head/neck/noise; not brain loss)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
