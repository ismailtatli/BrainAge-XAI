"""Orientation-aware central orthogonal slices without loading a complete volume."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform


def central_slices(path: Path) -> tuple[list[np.ndarray], tuple[float, ...]]:
    """Return sagittal/coronal/axial RAS+ planes; oblique images are not resampled."""
    image = nib.load(path)
    if image.ndim != 3 or any(s <= 0 for s in image.shape):
        raise ValueError("Slice inspection requires a non-empty 3D T1 image")
    affine = image.affine
    if (
        not np.isfinite(affine).all()
        or np.linalg.matrix_rank(affine[:3, :3]) != 3
        or not np.allclose(affine[3], [0, 0, 0, 1])
    ):
        raise ValueError("Image has an invalid affine")
    if not int(image.header["qform_code"]) and not int(image.header["sform_code"]):
        raise ValueError("No coded orientation; anatomical labels cannot be trusted")
    transform = ornt_transform(io_orientation(affine), axcodes2ornt(("R", "A", "S")))
    planes = []
    spacing = []
    for axis in range(3):
        original_axis = int(np.where(transform[:, 0] == axis)[0][0])
        length = image.shape[original_axis]
        center = length // 2
        if transform[original_axis, 1] == -1:
            center = length - 1 - center
        selection = [slice(None)] * 3
        selection[original_axis] = slice(center, center + 1)
        block = np.asarray(image.dataobj[tuple(selection)])
        canonical = apply_orientation(block, transform)
        plane = np.squeeze(canonical, axis=axis).T
        if not np.isfinite(plane).all():
            raise ValueError("Selected central slice contains non-finite voxels")
        planes.append(plane)
        spacing.append(float(nib.affines.voxel_sizes(affine)[original_axis]))
    return planes, tuple(spacing)


def plot_central_slices(path: Path, output: Path | None = None, *, show: bool = False) -> None:
    """Save/display neurological convention slices; labels describe nearest anatomical axes."""
    import matplotlib.pyplot as plt

    planes, spacing = central_slices(path)
    values = np.concatenate([plane.ravel() for plane in planes])
    low, high = np.percentile(values, [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [
        ("Sagittal", "Anterior →", "Superior →", spacing[2] / spacing[1]),
        ("Coronal", "Right →", "Superior →", spacing[2] / spacing[0]),
        ("Axial", "Right →", "Anterior →", spacing[1] / spacing[0]),
    ]
    for ax, plane, (title, xlabel, ylabel, aspect) in zip(axes, planes, labels):
        ax.imshow(plane, cmap="gray", origin="lower", vmin=low, vmax=high, aspect=aspect)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Research inspection only • nearest RAS+ axes (oblique scans not resampled)")
    fig.tight_layout()
    try:
        if output is not None:
            output = Path(output)
            if output.resolve() == Path(path).resolve():
                raise ValueError("Output must not overwrite source MRI")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect central T1 slices for research")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.output is None and not args.show:
        parser.error("Provide --output and/or --show")
    try:
        plot_central_slices(args.image, args.output, show=args.show)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Slice inspection failed: {exc}\n")


if __name__ == "__main__":
    main()
