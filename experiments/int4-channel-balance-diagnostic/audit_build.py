#!/usr/bin/env python3
"""Read-only fresh ELF/source binding against actual323213. Never runs ELF."""
import hashlib
import json
import subprocess
from pathlib import Path

import registry_audit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROUND2 = ROOT.parent
BUILD = "a003-int4-channel-balance-device-build"
OLD_SHA = "3444ef561e2d39d6a15ec6340548d07e65b873d2884b62ac04f9ea2ccd66cdd4"
SOURCE = "e6e45d96f29b49adbde28a04b892449ead4f715e"
PARSER_SHA = "3a101f3df7dc0b27dec888865ee866d54312fd16bebfac289630a6fcb2e0e743"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def git_sources(commit, hashes):
    names = [p for p in hashes if p != "Cargo.lock"]
    queries = "".join(f"{commit}:{p}\n" for p in names).encode()
    data = subprocess.check_output(["git", "cat-file", "--batch"], input=queries, cwd=ROOT)
    at = 0
    for name in names:
        end = data.index(b"\n", at)
        header = data[at:end].split()
        assert header[1] == b"blob", name
        size = int(header[2]); blob = data[end + 1:end + 1 + size]
        assert hashlib.sha256(blob).hexdigest() == hashes[name], name
        at = end + size + 2
    assert at == len(data)
    assert sha(ROOT / "Cargo.lock") == hashes["Cargo.lock"]


def index(artifact):
    assert artifact["record_count"] == 19
    assert all(r["key"] == [] for r in artifact["records"])
    result = {r["path"]: r for r in artifact["records"]}
    assert len(result) == 19
    return result


def main():
    build_dir = ROUND2 / "runs" / BUILD
    run = load(build_dir / "run.json")
    hashes = load(build_dir / "source-hashes.json")
    assert run["commit"] == SOURCE and run["status"] == ""
    assert run["exit_code"] == 0 and run["result"] == "command_succeeded"
    assert run["build_inputs_unchanged"] and run["fresh_package"]["contents_changed"] is False
    assert (build_dir / "source.diff").read_bytes() == b""
    assert len(hashes) == 423
    git_sources(SOURCE, hashes)
    for name, digest in hashes.items(): assert sha(ROOT / name) == digest, name
    deps = {k: hashes[k] for k in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")}
    assert deps == {
        "Cargo.toml": "a4154d908d25a512d031dea859b1b013794f821a47deb739334332bd0e4fdf85",
        "Cargo.lock": "a99652661c588fe5c3a8b23278f2390d8cd2e99f2b33bcab6f52e97dbba04e2e",
        "rust-toolchain.toml": "1e6c9fed0427862faf47f5993fca93848e1b5d62127b38a058fd20ef6a03cc6d",
    }
    prior_dir = ROOT / "experiments/int4-grouped-complete-ffn/actual-323213"
    prior_run = load(prior_dir / "run.json")
    prior_status = load(prior_dir / "arena-status.json")
    assert prior_status["id"] == 323213 and prior_status["status"] == "SUCCEEDED" and prior_status["exit_code"] == 0
    prior_hashes = load(prior_dir / "source-hashes.json")
    assert len(prior_hashes) == 421
    git_sources(prior_run["commit"], prior_hashes)
    for name, digest in prior_hashes.items(): assert hashes[name] == digest, name
    assert set(hashes) - set(prior_hashes) == {"src/bin/test_int4_channel_balance.rs", "src/bin/support/int4_channel_balance.rs"}
    captured = run["built_artifacts"]["target/release/test_int4_channel_balance"]
    current_path = Path(captured["path"])
    assert current_path.stat().st_size == captured["size"] and sha(current_path) == captured["sha256"]
    prior = prior_run["submitted_artifact"]
    assert prior["sha256"] == OLD_SHA
    for name in ("path", "source_path"): assert sha(Path(prior[name])) == OLD_SHA
    assert sha(HERE / "registry_audit.py") == PARSER_SHA
    current_registry = registry_audit.read_artifact(current_path)
    prior_registry = registry_audit.read_artifact(Path(prior["path"]))
    current, previous = index(current_registry), index(prior_registry)
    assert current.keys() == previous.keys()
    out_option = next(x for x in run["command"] if x.startswith("FURIOSA_OPT_OUT_DIR="))
    out_dir = ROUND2 / out_option.split("=", 1)[1].removeprefix("/round2/") / "furiosa-opt-gemma4"
    comparisons = []
    for name in sorted(current):
        now, old = current[name], previous[name]
        assert (now["bytes"], now["sha256"]) == (old["bytes"], old["sha256"]), name
        image = out_dir / (name + ".bin")
        assert image.stat().st_size == now["bytes"] and sha(image) == now["sha256"], name
        comparisons.append({"path": name, "bytes": now["bytes"], "sha256": now["sha256"], "actual323213_equal": True, "fresh_output_equal": True})
    assert current_registry["section_sha256"] == prior_registry["section_sha256"]
    out = {
        "source_commit": SOURCE, "functional_source": "52d821e2", "build": run,
        "source_inputs": 423, "original_actual_inputs_unchanged": 421,
        "original_dependencies": deps, "captured_artifact": captured,
        "registry": current_registry, "all19_comparisons": comparisons,
        "prior_actual": {"job": 323213, "run": prior_run["name"], "status": prior_status, "artifact": prior,
                         "files": {name: sha(prior_dir / name) for name in ("run.json", "arena-status.json", "source-hashes.json", "arena.log")}},
        "record_files": {name: sha(build_dir / name) for name in ("run.json", "source-hashes.json", "source.diff", "output.log")},
        "parser_sha256": PARSER_SHA,
        "decision": "FRESH_BUILD_AND_ALL19_IMAGE_IDENTITY_PASS; only host data preparation/flow differs; gamma inverse, scalar products, intended output parity and drift still require root device execution.",
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
