#!/usr/bin/env python3
"""Verify M02 ownership coverage, source boundaries and immutable build/run evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FAMILY = "m02-cluster-k"
LABEL = "m02-khalf-screen"


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def coordinate_check(grouped):
    # Identity tags, not a Rust/DPE emulator: verify coverage and that padding
    # cannot be counted as a second partial. Real arithmetic is checked on NPU.
    segments = []
    padded = [[[None] * 30 for _ in range(256)] for _ in range(2)]
    for c in range(2):
        for s in range(256):
            for t in range(15):
                h = 480*(s//32) + 32*t + s%32 if grouped else 256*t+s
                assert 0 <= h < 3840
                start = 4096*h + 2048*c
                assert start % 256 == 0
                segments.append((start, start+2048))
                padded[c][s][2*t] = 10000*h + c
                assert (2*t*4) % 8 == 0
    assert len(segments) == 7680
    ordered = sorted(segments)
    assert ordered[0][0] == 0 and ordered[-1][1] == 3840*4096
    assert all(a[1] == b[0] for a,b in zip(ordered, ordered[1:]))
    for h in range(3840):
        if grouped:
            g, local = divmod(h, 480)
            t, r = divmod(local, 32)
            s = 32*g+r
        else:
            t,s = divmod(h,256)
        pair = [padded[c][s][2*t] for c in range(2)]
        assert sum(pair) == 20000*h+1
        assert all(padded[c][s][2*t+1] is None for c in range(2))
        e, local = divmod(h,480)
        assert 480*e+local == h and (480*4) % 8 == 0
    for s in range(256):
        assert (s//32)*32 <= s < (s//32+1)*32
    return {"row_mapping": "480*(s//32)+32*t+s%32" if grouped else "256*t+s",
            "unique_contiguous_weight_segments": len(segments), "weight_bytes": ordered[-1][1],
            "two_partial_rows_and_padding_verified": 3840,
            "activation_leader_bytes": 2*8*2048*2,
            "activation_replica_bytes": 2*256*2048*2,
            "padded_partial_bytes": 2*256*15*8, "useful_partial_bytes": 2*3840*4}


def check():
    origin_commit = json.loads((HERE / "khalf-v1/recipe.json").read_text())["base_commit"]
    origin = subprocess.check_output(["git", "show", origin_commit+":furiosa-opt-gemma4-12B/src/device/sliding/projection.rs"], cwd=ROOT, text=True)
    def outside(code):
        a = code.index("pub(crate) fn project_output(")
        b = code.index("\npub(crate) type OutputEpilogueSlices", a)
        return (code[:a]+code[b:]).replace("Dummy2, Dummy256,", "Dummy2,")
    variants = {}
    for path in sorted(HERE.glob("khalf-v*/projection.rs")):
        code = path.read_text()
        assert outside(code) == outside(origin)
        body = code[code.index("pub(crate) fn project_output("):code.index("\npub(crate) type OutputEpilogueSlices")]
        assert body.count("fetch_table_lookup::<bf16>()") == 1
        assert body.count("weight.to_dm(&mut device.tdma)") == 1
        assert body.count(".cast::<bf16") == 1
        assert body.index("FpBinaryOp::AddF") < body.index("FpBinaryOp::MulF") < body.index(".cast::<bf16")
        variants[path.parent.name] = {"source": digest(code.encode()), "outside_output_unchanged_except_import": True}
    builds = {}
    for path in sorted(HERE.glob("build-*-repro.json")):
        r = json.loads(path.read_text())
        for target, frozen in r["frozen_files"].items():
            expected = r["source_inputs"].get(target, r["logs"].get(target))
            assert digest((ROOT / frozen).read_bytes()) == expected, frozen
        recipe = r["source_plan"]["recipe"]
        assert digest((ROOT / recipe["path"]).read_bytes()) == {k:recipe[k] for k in ("bytes", "sha256")}
        for target, override in r["source_plan"]["source_overrides"].items():
            assert digest((ROOT / override["path"]).read_bytes()) == r["source_inputs"][target]
        if r.get("host_build_exit_code") == 0:
            artifacts = ROOT / "moa-dev/experiments/artifacts" / FAMILY / r["label"]
            package = artifacts / "package"
            assert {p.name:digest(p.read_bytes()) for p in package.iterdir()} == r["package_files"]
            host = (package / "test_runtime").read_bytes()
            for kernel, image in r["images"].items():
                data = (artifacts / "kernels" / f"furiosa_opt_gemma4::ops::{kernel}.bin").read_bytes()
                assert digest(data) == {k:image[k] for k in ("bytes", "sha256")}
                assert host.count(data) == image["host_occurrences"] == 1
                if kernel == r["target_kernel"]:
                    assert digest(data) == r["exact_image"]
                else:
                    assert image["sha256"] == r["source_plan"]["expected_unchanged_images"][kernel]
            first = json.loads((HERE / f"build-{r['label'].removesuffix('-package')}-repro.json").read_text())
            assert first["exact_image"] == r["exact_image"] and first["schedule"] == r["schedule"]
        builds[path.name] = {"frozen_files_checked":len(r["frozen_files"]),
                             "exact_compile_exit_code":r.get("exact_compile_exit_code"),
                             "host_build_exit_code":r.get("host_build_exit_code"),
                             "failed_exit_code":r.get("failed_exit_code")}
    arena = None
    if (HERE / f"{LABEL}-results.json").exists():
        sys.path.insert(0, str(ROOT / "moa-dev"))
        from measure import parse_log
        result = json.loads((HERE / f"{LABEL}-results.json").read_text())
        raw = (HERE / "evidence" / f"arena-{result['job']['id']}.log").read_bytes()
        assert digest(raw)["sha256"] == result["log_sha256"]
        manifest = json.loads((HERE / f"{LABEL}-build.json").read_text())
        assert {k:v for k,v in result["package_files"].items() if k != "job.yaml"} == manifest["package_files"]
        sections = re.findall(r"^STRUCTURAL_BEGIN (\w+)\n(.*?)^STRUCTURAL_END \1$", raw.decode(), re.M | re.S)
        assert [n for n,_ in sections] == [s["name"] for s in manifest["stages"]]
        for stage, log in sections:
            assert parse_log(log) == result["stages"][stage]["kernels"]
        arena = {"job":result["job"]["id"], "valid":result["valid"],
                 "stage_validity":{n:s["valid"] for n,s in result["stages"].items()},
                 "total_comparisons":sum(s["comparisons_passed"] for s in result["stages"].values()),
                 "log_and_downloaded_package_manifest_checked":True}
    return {"scope":"Coordinate model, source scope, frozen build/package/log provenance; not a CPU emulator or extended numerical validation.",
            "coordinate_models":{"global_interleave":coordinate_check(False), "grouped_interleave":coordinate_check(True)},
            "variants":variants, "builds":builds, "arena":arena}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = check()
    path = HERE / "verification.json"
    if args.check:
        assert json.loads(path.read_text()) == data
    else:
        path.write_text(json.dumps(data, indent=2)+"\n")
    print(f"Verified {len(data['variants'])} sources, two complete ownership maps, {len(data['builds'])} builds; Arena={data['arena']}")
