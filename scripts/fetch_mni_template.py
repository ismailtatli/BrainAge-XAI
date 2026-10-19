"""Explicit, checksum-pinned TemplateFlow fetch; never imported by runtime preprocessing/tests."""

import argparse
import tempfile
from pathlib import Path
from urllib.request import urlopen

from brainage_xai.config.preprocessing import validate_output_directory
from brainage_xai.registration.template import (
    MASK_NAME,
    MASK_SHA256,
    T1_NAME,
    T1_SHA256,
    TEMPLATE_ID,
    load_template,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly fetch the pinned TemplateFlow 1 mm T1/mask"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/resources") / f"tpl-{TEMPLATE_ID}"
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    validate_output_directory(output, [])
    output.mkdir(parents=True, exist_ok=True)
    for name, digest in ((T1_NAME, T1_SHA256), (MASK_NAME, MASK_SHA256)):
        destination = output / name
        if destination.exists():
            if sha256(destination) != digest:
                raise ValueError(f"Existing resource checksum mismatch: {name}")
            continue
        temporary = None
        try:
            with urlopen(
                f"https://templateflow.s3.amazonaws.com/tpl-{TEMPLATE_ID}/{name}", timeout=60
            ) as source:
                with tempfile.NamedTemporaryFile(
                    dir=output, suffix=".nii.gz", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        stream.write(block)
            if sha256(temporary) != digest:
                raise ValueError(f"Download checksum mismatch: {name}")
            temporary.replace(destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    bundle = load_template(output / T1_NAME, output / MASK_NAME)
    print(f"Verified {bundle.provenance['id']} resolution 01: {output}")


if __name__ == "__main__":
    main()
