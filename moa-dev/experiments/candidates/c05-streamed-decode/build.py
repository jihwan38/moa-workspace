#!/usr/bin/env python3
"""Reuse the audited compiler/package pipeline for the C05 architecture variants."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c05_build_common", HERE.parent / "c02-c03-e55/build.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
common.HERE = HERE
common.ARTIFACTS = common.ROOT / "moa-dev/experiments/artifacts/c05-streamed-decode"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["direct", "partials", "partials-sub", "direct-upgate", "direct-down", "partials-upgate", "partials-down"])
    parser.add_argument("--extra", action="store_true")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    common.build(args.variant, extra=args.extra, label=args.label)
