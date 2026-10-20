"""Fixed public-template slice selection and deterministic MRI inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from brainage_xai.preprocessing.final import load_spec, validate_finished
from brainage_xai.registration.template import sha256

VIEWS = {"axial": 2, "coronal": 1, "sagittal": 0}
COUNTS = {"train": 393, "validation": 84, "test": 84}


def validate_splits(frame, expected=COUNTS):
    if frame.subject_id.isna().any() or frame.subject_id.duplicated().any():
        raise ValueError("Each subject must occur exactly once")
    if frame.split.value_counts().to_dict() != expected:
        raise ValueError("Frozen subject split counts differ")
    if not np.isfinite(frame.age).all():
        raise ValueError("Nonfinite chronological age")
    return frame.sort_values("subject_id").reset_index(drop=True)


def slice_selection(mask, affine):
    coordinates = np.array(np.nonzero(mask))
    if coordinates.shape[1] == 0:
        raise ValueError("Empty public template mask")
    lo, hi = coordinates.min(1), coordinates.max(1)
    fractions = np.linspace(0.2, 0.8, 5)
    indices = {
        view: np.rint(lo[axis] + fractions * (hi[axis] - lo[axis])).astype(int).tolist()
        for view, axis in VIEWS.items()
    }
    if any(len(set(v)) != 5 for v in indices.values()):
        raise ValueError("Template extent cannot provide five distinct slices")
    return {
        "rule": "round-to-even(min + [0.2,0.35,0.5,0.65,0.8]*(max-min))",
        "extent_min_xyz": lo.tolist(),
        "extent_max_xyz": hi.tolist(),
        "indices": indices,
        "coordinates_mm": {
            v: [float(affine[a, a] * i + affine[a, 3]) for i in indices[v]]
            for v, a in VIEWS.items()
        },
        "baseline_axial_index": int((mask.shape[2] - 1) // 2),
    }


def create_inputs(volume, selection):
    """Full planes, no participant selection/crop; top of image is superior/anterior."""
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Require finite 3D MRI")
    planes = [volume[:, :, selection["baseline_axial_index"]].T[::-1]]
    for view, axis in VIEWS.items():
        for index in selection["indices"][view]:
            planes.append(np.take(volume, index, axis=axis).T[::-1])
    result = []
    for plane in planes:
        x = torch.from_numpy(np.ascontiguousarray(plane, dtype=np.float32))[None, None]
        x = (x.clamp(-3, 3) + 3) / 6
        x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False, antialias=True)
        result.append(x[0])
    return torch.stack(result)


def imagenet_channels(gray):
    rgb = gray.expand(*gray.shape[:-3], 3, *gray.shape[-2:])
    shape = [1] * (rgb.ndim - 3) + [3, 1, 1]
    mean = rgb.new_tensor([0.485, 0.456, 0.406]).reshape(shape)
    std = rgb.new_tensor([0.229, 0.224, 0.225]).reshape(shape)
    return (rgb - mean) / std


def verify_frozen(root, output, progress=print):
    """Read-only full saved-volume/provenance validation before any training."""
    _, _, cohort, _ = load_spec(root / "configs/preprocessing.frozen.yaml")
    cohort = validate_splits(cohort)
    saved = pd.read_csv(output / "cohort.csv")
    fields = ["subject_id", "split", "age", "acquisition_site", "sex", "t1_path"]
    pd.testing.assert_frame_equal(
        cohort[fields],
        saved.sort_values("subject_id").reset_index(drop=True)[fields],
        check_dtype=False,
    )
    manifest = pd.read_csv(output / "manifest.csv")
    if manifest.subject_id.duplicated().any() or set(manifest.subject_id) != set(cohort.subject_id):
        raise ValueError("Final manifest subject set differs")
    if not manifest.status.eq("success").all():
        raise ValueError("Phase 2 contains failed outputs")
    run = json.loads((output / "run_provenance.json").read_text())
    records = manifest.set_index("subject_id")
    for i, row in enumerate(cohort.to_dict("records"), 1):
        source = Path(row["t1_path"])
        source_hash = sha256(source)
        if run["source_sha256"][row["subject_id"]] != source_hash:
            raise ValueError("Raw MRI source checksum differs")
        record = validate_finished(row, output, source_hash, run["fingerprint"])
        m = records.loc[row["subject_id"]]
        if (
            m.split != row["split"]
            or m.site != row["acquisition_site"]
            or m.output_sha256 != record["output_sha256"]
        ):
            raise ValueError("Final manifest/provenance disagree")
        if i % 25 == 0 or i == len(cohort):
            progress(f"Verified frozen MRI/provenance {i}/{len(cohort)}", flush=True)
    return cohort


class FeatureDataset(Dataset):
    """Subject-level loader; cached frozen prefix maps, never slice-level membership."""

    def __init__(self, frame, cache, kind):
        self.rows = frame.reset_index(drop=True)
        self.cache, self.kind = Path(cache), kind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        maps = np.load(self.cache / f"{row.subject_id}.npy", allow_pickle=False)
        maps = maps[:1] if self.kind == "A" else maps[1:]
        return torch.from_numpy(maps.copy()), torch.tensor(row.age, dtype=torch.float32)
