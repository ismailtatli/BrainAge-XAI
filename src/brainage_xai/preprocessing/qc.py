"""Private orthogonal QC comparisons in physical space; no medical interpretation."""

from pathlib import Path

import nibabel as nib
import numpy as np


def save_comparison(
    canonical: nib.spatialimages.SpatialImage,
    processed: nib.spatialimages.SpatialImage,
    output: Path,
    *,
    dpi: int = 120,
) -> None:
    """Compare planes through the same source-center world coordinate, with physical aspect."""
    import matplotlib.pyplot as plt

    center_voxel = (np.asarray(canonical.shape) - 1) / 2
    world_center = nib.affines.apply_affine(canonical.affine, center_voxel)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    try:
        for row, (image, title) in enumerate(
            ((canonical, "Canonical source"), (processed, "Resampled / normalized"))
        ):
            if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
                raise ValueError("QC comparison expects nearest RAS+ axes")
            index = np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine), world_center))
            index = np.clip(index.astype(int), 0, np.asarray(image.shape) - 1)
            spacing = nib.affines.voxel_sizes(image.affine)
            for axis, name in enumerate(("Sagittal", "Coronal", "Axial")):
                selection = [slice(None)] * 3
                selection[axis] = int(index[axis])
                plane = np.asarray(image.dataobj[tuple(selection)]).T
                if not np.isfinite(plane).all():
                    raise ValueError("Non-finite QC slice")
                low, high = np.percentile(plane, [1, 99])
                remaining = [i for i in range(3) if i != axis]
                ax = axes[row, axis]
                ax.imshow(
                    plane,
                    origin="lower",
                    cmap="gray",
                    vmin=low,
                    vmax=high,
                    aspect=spacing[remaining[1]] / spacing[remaining[0]],
                )
                ax.set_title(f"{title}: {name}")
                ax.set_xlabel(("Anterior →", "Right →", "Right →")[axis])
                ax.set_ylabel(("Superior →", "Superior →", "Anterior →")[axis])
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(
            "Preprocessing QC only • same source-center world coordinate\n"
            "Nearest RAS+ planes; oblique source planes may differ from resampled planes"
        )
        fig.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
