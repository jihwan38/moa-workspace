#!/usr/bin/env python3
"""Build one structural candidate at the compile gate; preserve failed evidence too."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = "7f891f9c48e670fe014b2ac5bd2c92bae3bac433"


def build(candidate, variant, label, package=False):
    here = ROOT / "moa-dev/experiments/candidates" / candidate
    config = json.loads((here / "candidate.json").read_text())
    spec = importlib.util.spec_from_file_location("structural_common", here.parent / "c02-c03-e55/build.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.HERE, common.BASE = here, BASE
    common.ARTIFACTS = ROOT / "moa-dev/experiments/artifacts" / candidate
    e104 = json.loads((here.parent / "c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json").read_text())
    common.IMAGES = {k: v["sha256"] for k, v in e104["images"].items()}
    record_path = here / f"build-{label}-repro.json"
    if record_path.exists():
        raise SystemExit(f"Refusing to replace existing evidence: {record_path}")
    try:
        common.build(variant, label=label, target_kernel=config["target_kernel"],
                     changed_sources=config["changed_sources"], compile_only=not package)
    finally:
        if record_path.exists():
            record = json.loads(record_path.read_text())
            record["branch"] = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
            record["builder"] = {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                                 **common.digest(Path(__file__).read_bytes())}
            record["common_builder"] = common.digest((here.parent / "c02-c03-e55/build.py").read_bytes())
            record["frozen_files"] = {}
            evidence = here / "evidence" / label
            evidence.mkdir(parents=True, exist_ok=False)
            out = common.ARTIFACTS / label
            for target in record["changed_sources"]:
                source = out / "crate" / target
                dest = evidence / target
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                assert common.digest(dest.read_bytes()) == record["source_inputs"][target]
                record["frozen_files"][target] = str(dest.relative_to(ROOT))
            for name in record["logs"]:
                dest = evidence / name
                shutil.copyfile(out / name, dest)
                record["frozen_files"][name] = str(dest.relative_to(ROOT))
            record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("variant")
    parser.add_argument("--label", required=True)
    parser.add_argument("--package", action="store_true", help="Also build the host after the feasibility gate")
    args = parser.parse_args()
    build(args.candidate, args.variant, args.label, args.package)
