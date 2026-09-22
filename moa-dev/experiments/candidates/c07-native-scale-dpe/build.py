#!/usr/bin/env python3
"""Build isolated C07 architectures against E79, preserving exact provenance."""
import argparse
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c07_build_common", HERE.parent / "c02-c03-e55/build.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
common.HERE = HERE
common.ARTIFACTS = common.ROOT / "moa-dev/experiments/artifacts/c07-native-scale-dpe"
common.BASE = "2735b5899f41195dedeab1e076c82317e24efcb2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--extra", action="store_true")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    assert (HERE / args.variant / "mlp.rs").is_file()
    common.build(args.variant, extra=args.extra, label=args.label)
