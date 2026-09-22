#!/usr/bin/env python3
"""C08 isolated QKV builds using the existing provenance builder."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c08_common", HERE.parent / "c02-c03-e55/build.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
common.HERE = HERE
common.BASE = "7f891f9c48e670fe014b2ac5bd2c92bae3bac433"
common.ARTIFACTS = common.ROOT / "moa-dev/experiments/artifacts/c08-fused-qkv"
original = json.loads((HERE.parent / "c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json").read_text())
common.IMAGES = {k: v["sha256"] for k, v in original["images"].items()}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant", help="c03- prefix selects the shared builder's QKV file set; this is a C08 experiment")
    p.add_argument("--label", required=True)
    args = p.parse_args()
    assert args.variant.startswith("c03-")
    common.build(args.variant, label=args.label)
