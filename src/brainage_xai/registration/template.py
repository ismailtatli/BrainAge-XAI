"""Pinned local TemplateFlow resource validation; no runtime network access."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

TEMPLATE_ID = "MNI152NLin2009cAsym"
TEMPLATE_RESOLUTION = "01"
TEMPLATE_SHAPE = (193, 229, 193)
TEMPLATE_AFFINE = np.array(
    [[1, 0, 0, -96], [0, 1, 0, -132], [0, 0, 1, -78], [0, 0, 0, 1]], dtype=float
)
T1_NAME = f"tpl-{TEMPLATE_ID}_res-01_T1w.nii.gz"
MASK_NAME = f"tpl-{TEMPLATE_ID}_res-01_desc-brain_mask.nii.gz"
T1_SHA256 = "1f27aabea9f7183dc0c69dafa71e1787b921ca778d655d9c1bf302b273d5627a"
MASK_SHA256 = "e40bb1816736504d4c25abc243c1c8503df1d308f64c048318aa4b829e09a7ae"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_grid(
    image: nib.spatialimages.SpatialImage,
    *,
    shape: tuple[int, ...] = TEMPLATE_SHAPE,
    affine: np.ndarray = TEMPLATE_AFFINE,
) -> None:
    """Validate exact requested reference geometry; alternate templates are not substituted."""
    if image.shape != shape or not np.allclose(image.affine, affine, rtol=0, atol=1e-5):
        raise ValueError("Template has incompatible shape or physical grid")
    if not np.allclose(nib.affines.voxel_sizes(image.affine), [1, 1, 1], rtol=0, atol=1e-6):
        raise ValueError("Template requires 1 mm isotropic spacing")
    if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        raise ValueError("Template requires RAS+ orientation")
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Template spatial units must be millimeters")
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(data).all() or not np.any(data):
        raise ValueError("Template has empty or non-finite intensities")


@dataclass
class TemplateBundle:
    image: nib.spatialimages.SpatialImage
    mask: nib.spatialimages.SpatialImage | None
    provenance: dict[str, Any]


def load_template(
    t1_path: Path, mask_path: Path | None, *, use_mask: bool = True
) -> TemplateBundle:
    """Require byte-identical pinned TemplateFlow T1/mask, metadata and binary mask geometry."""

    def read(path: Path, name: str, expected: str) -> tuple[nib.spatialimages.SpatialImage, dict]:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Missing local template resource: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Template checksum mismatch: {path.name}")
        image = nib.load(path)
        validate_grid(image)
        return image, {
            "file_name": path.name,
            "canonical_resource_name": name,
            "sha256": actual,
            "shape": list(image.shape),
            "spacing_mm": nib.affines.voxel_sizes(image.affine).tolist(),
            "orientation": list(nib.aff2axcodes(image.affine)),
            "affine_ras": image.affine.tolist(),
        }

    image, t1_info = read(t1_path, T1_NAME, T1_SHA256)
    mask = None
    mask_info = None
    if use_mask:
        if mask_path is None:
            raise ValueError("Fixed-image masking requires the local TemplateFlow brain mask")
        mask, mask_info = read(mask_path, MASK_NAME, MASK_SHA256)
        values = np.asanyarray(mask.dataobj)
        if not np.isin(values, [0, 1]).all():
            raise ValueError("Template brain mask must be binary")
    image = nib.Nifti1Image(
        image.get_fdata(dtype=np.float32, caching="unchanged"), image.affine, image.header.copy()
    )
    if mask is not None:
        mask = nib.Nifti1Image(
            np.asanyarray(mask.dataobj).astype(np.uint8), mask.affine, mask.header.copy()
        )
    return TemplateBundle(
        image,
        mask,
        {
            "id": TEMPLATE_ID,
            "resolution": TEMPLATE_RESOLUTION,
            "source": "https://github.com/templateflow/tpl-MNI152NLin2009cAsym",
            "template": t1_info,
            "brain_mask": mask_info,
        },
    )
