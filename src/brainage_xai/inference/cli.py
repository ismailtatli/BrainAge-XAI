"""Single-subject CLI; the same service is used by the local Streamlit application."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_runtime_config, safe_destination
from .pipeline import export_bundle, infer_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="BrainAge-XAI research-only local inference")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_runtime_config(args.config)
        if args.output:
            output = safe_destination(args.output, [args.input])
            if output.exists():
                raise FileExistsError("Output already exists; choose a new analysis directory")
        bundle = infer_file(
            args.input,
            args.age,
            args.config,
            progress=lambda text: print(text, file=sys.stderr, flush=True),
        )
        output = args.output or config.output_dir / bundle.result.analysis_id
        destination = export_bundle(bundle, output, [args.input])
        print(json.dumps({"analysis_id": bundle.result.analysis_id, "output": str(destination)}))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"BrainAge-XAI analysis failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
