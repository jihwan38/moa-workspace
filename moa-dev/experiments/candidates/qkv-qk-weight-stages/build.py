#!/usr/bin/env python3
"""Build the explicitly selected source recipe; performance controls do not select code."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[4]

def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def source_manifest(commit):
    """Read the selected committed crate, plus the pinned compilation lock."""
    prefix = "furiosa-opt-gemma4-12B/"
    data = subprocess.check_output(["git", "archive", commit, prefix.rstrip("/")], cwd=ROOT)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name.removeprefix(prefix)
            if name.startswith("src/") or name in {"Cargo.toml", "rust-toolchain.toml"}:
                files[name] = digest(archive.extractfile(member).read())
    files["Cargo.lock"] = digest((ROOT / "moa-dev/locks/gemma4-Cargo.lock").read_bytes())
    return files


def resolve_plan(candidate, variant, extra=False):
    here = ROOT / "moa-dev/experiments/candidates" / candidate
    # A variant recipe is complete: it does not silently inherit another
    # variant's source origin, component set or expected image hashes.
    config_path = here / variant / "recipe.json"
    if not config_path.is_file():
        config_path = here / "candidate.json"
    config = json.loads(config_path.read_text())
    commit = config["base_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("base_commit must be an explicit full Git commit")
    subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=ROOT, check=True)
    comparison_path = ROOT / config["comparison_recipe"]
    comparison = json.loads(comparison_path.read_text())
    changed = config["changed_sources"]
    if not changed or len(set(changed.values())) != len(changed):
        raise ValueError("Each target file must have one composed source; duplicate targets would discard a method")
    replacements = {}
    for name, target in changed.items():
        if ".." in Path(target).parts or not (target == "src/ops.rs" or target.startswith("src/device/")):
            raise ValueError(f"Outside competition source contract: {target}")
        source = (here / variant / name).resolve()
        source.relative_to(ROOT)
        replacements[target] = {"path": str(source.relative_to(ROOT)), **digest(source.read_bytes())}
    origin_inputs = source_manifest(commit)
    expected_images = {}
    reference = config.get("source_reference_build")
    reference_digest = None
    if reference:
        reference_path = ROOT / reference
        old_build = json.loads(reference_path.read_text())
        if old_build["source_inputs"] != origin_inputs:
            raise ValueError("source_reference_build does not match the selected source origin; do not use performance-control images here")
        expected_images = {k: v["sha256"] for k, v in old_build["images"].items()}
        if "unchanged_kernels" in config:
            expected_images = {k: expected_images[k] for k in config["unchanged_kernels"]}
        reference_digest = digest(reference_path.read_bytes())
    plan = {
        "recipe": {"path": str(config_path.relative_to(ROOT)), **digest(config_path.read_bytes())},
        "source_base_commit": commit,
        "source_origin_inputs": origin_inputs,
        "source_overrides": replacements,
        "performance_comparison": {"recipe": str(comparison_path.relative_to(ROOT)),
                                   "id": comparison["id"], "commit": comparison["git"]["commit"],
                                   **digest(comparison_path.read_bytes())},
        "source_reference_build": reference,
        "source_reference_build_digest": reference_digest,
        "expected_unchanged_images": expected_images,
        "target_kernel": config["target_kernel"],
        "scope": "Source selection and provenance only; not proof that selected methods compose correctly or run faster.",
    }
    if extra:
        sys.path.insert(0, str(ROOT / "moa-dev"))
        from extra_seeds import extended_harness
        host_path = "src/bin/test_kernels.rs"
        host = subprocess.check_output(
            ["git", "show", f"{commit}:furiosa-opt-gemma4-12B/{host_path}"], cwd=ROOT
        )
        plan["development_harness"] = {
            "path": host_path,
            "original": digest(host),
            "transformed": digest(extended_harness(host)),
            "run_ids": list(range(3, 9)),
            "scope": "Host-only extra-input validation; unchanged PRNG, tolerance and device sources. Timing is diagnostic only.",
        }
    return here, config, plan


def build(candidate, variant, label, package=False, extra=False, dump_graph=False):
    here, config, plan = resolve_plan(candidate, variant, extra)
    spec = importlib.util.spec_from_file_location("structural_common", here / "common_build.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.HERE, common.BASE = here, plan["source_base_commit"]
    common.ARTIFACTS = ROOT / "moa-dev/experiments/artifacts" / candidate
    record_path = here / f"build-{label}-repro.json"
    if record_path.exists():
        raise SystemExit(f"Refusing to replace existing evidence: {record_path}")
    expected_inputs = dict(plan["source_origin_inputs"])
    expected_inputs.update({target: {k: value[k] for k in ("bytes", "sha256")}
                           for target, value in plan["source_overrides"].items()})
    if extra:
        validation = plan["development_harness"]
        expected_inputs[validation["path"]] = validation["transformed"]
    try:
        common.build(variant, extra=extra, label=label, target_kernel=config["target_kernel"],
                     changed_sources=config["changed_sources"], compile_only=not package,
                     expected_images=plan["expected_unchanged_images"],
                     expected_source_inputs=expected_inputs, dump_graph=dump_graph)
    finally:
        if record_path.exists():
            record = json.loads(record_path.read_text())
            record["source_plan"] = plan
            assert record["base_commit"] == plan["source_base_commit"]
            record["branch"] = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
            record["builder"] = {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                                 **common.digest(Path(__file__).read_bytes())}
            record["common_builder"] = common.digest((here / "common_build.py").read_bytes())
            record["frozen_files"] = {}
            evidence = here / "evidence" / label
            evidence.mkdir(parents=True, exist_ok=False)
            out = common.ARTIFACTS / label
            frozen_sources = list(record["changed_sources"])
            if extra:
                frozen_sources.append(plan["development_harness"]["path"])
            for target in frozen_sources:
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
            if dump_graph and "graph" in record:
                for name in ("graph.json", "schedule.json"):
                    dest = evidence / name
                    shutil.copyfile(out / name, dest)
                    record["frozen_files"][name] = str(dest.relative_to(ROOT))
            record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("variant")
    parser.add_argument("--label")
    parser.add_argument("--plan", action="store_true", help="Resolve and show source/control selection without compiling")
    parser.add_argument("--package", action="store_true", help="Also build the host after the feasibility gate")
    parser.add_argument("--extra", action="store_true", help="Validate run3..8 with a host-only harness change; timing is diagnostic")
    parser.add_argument("--dump-graph", action="store_true", help="Preserve exact DMA descriptors and schedule for the feasibility gate")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(resolve_plan(args.candidate, args.variant, args.extra)[2], indent=2))
    else:
        if not args.label:
            parser.error("--label is required for a build")
        build(args.candidate, args.variant, args.label, args.package, args.extra, args.dump_graph)
