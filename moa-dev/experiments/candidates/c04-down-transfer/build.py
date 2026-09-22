#!/usr/bin/env python3
"""Build frozen C04 sources using the audited exact-image/package pipeline."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c04_build_common", HERE.parent / "c02-c03-e55/build.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
common.HERE = HERE
common.ARTIFACTS = common.ROOT / "moa-dev/experiments/artifacts/c04-down-transfer"
common.BASE = "9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    from prepare import CONFIGS
    parser.add_argument("variant", choices=list(CONFIGS))
    parser.add_argument("--extra", action="store_true")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    common.build(args.variant, extra=args.extra, label=args.label)
