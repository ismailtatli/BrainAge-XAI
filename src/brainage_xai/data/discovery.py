"""Deterministic discovery without assuming that filenames imply valid images."""

import re
from pathlib import Path

_PATTERN = re.compile(r"^IXI(\d+)(?:-[A-Za-z0-9]+)*-T1\.nii(?:\.gz)?$", re.IGNORECASE)


def parse_subject_id(filename: str | Path) -> str | None:
    """Return a canonical ID only for an IXI T1 filename."""
    match = _PATTERN.fullmatch(Path(filename).name)
    if match is None or int(match[1]) <= 0:
        return None
    return f"IXI{int(match[1]):03d}"


def parse_acquisition_site(filename: str | Path) -> str | None:
    """Extract only a known site in the IXI ID-site-acquisition-T1 filename structure."""
    name = Path(filename).name
    if parse_subject_id(name) is None:
        return None
    match = re.fullmatch(r"IXI\d+-(Guys|HH|IOP)-\d+-T1\.nii(?:\.gz)?", name, re.IGNORECASE)
    if match is None:
        return None
    return {"guys": "Guys", "hh": "HH", "iop": "IOP"}[match[1].lower()]


def discover_t1_files(directory: Path) -> list[Path]:
    """Include all NIfTI candidates in the configured T1 tree, even malformed names."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    return sorted(
        (
            p
            for p in directory.rglob("*")
            if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))
        ),
        key=lambda p: str(p),
    )
