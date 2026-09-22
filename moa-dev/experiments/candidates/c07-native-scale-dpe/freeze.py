#!/usr/bin/env python3
"""Preserve exact attempted sources and logs, including unsuccessful builds."""
import argparse
import json
from build import HERE, common


def freeze(check=False):
    manifest = {"base_commit": common.BASE, "builds": {}, "arena_logs": {}}

    def save(data, suffix):
        value = common.digest(data)
        path = HERE / "evidence" / (value["sha256"] + suffix)
        if check:
            assert path.read_bytes() == data, path
        else:
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                assert path.read_bytes() == data, path
            else:
                path.write_bytes(data)
        return dict(value, path=str(path.relative_to(HERE)))

    for path in sorted(HERE.glob("build-*.json")):
        record = json.loads(path.read_text())
        if "source_inputs" not in record:
            continue
        label = record.get("label", path.stem.removeprefix("build-"))
        out = common.ARTIFACTS / label
        entry = {"sources": {}, "logs": {}}
        for name, expected in record["source_inputs"].items():
            content = (out / "crate" / name).read_bytes()
            assert common.digest(content) == expected, (label, name)
            # Full builds derive every other source from the recorded E79 commit.
            # Probe wrappers/host and extra-input hosts are independent source inputs.
            if name.endswith("mlp.rs") or "candidate_build" in record or (record.get("extra_inputs") and name == "src/bin/test_kernels.rs"):
                entry["sources"][name] = save(content, ".rs" if name.endswith(".rs") else ".txt")
        for name, expected in record["logs"].items():
            content = (out / name).read_bytes()
            assert common.digest(content) == expected, (label, name)
            entry["logs"][name] = save(content, ".log")
        manifest["builds"][path.name] = entry
    for path in sorted(HERE.glob("*-results.json")):
        record = json.loads(path.read_text())
        if "log" in record:
            content = (common.ROOT / record["log"]).read_bytes()
            assert common.digest(content)["sha256"] == record["log_sha256"]
            manifest["arena_logs"][path.name] = save(content, ".log")
    target = HERE / "frozen-evidence.json"
    if check:
        assert json.loads(target.read_text()) == manifest
    else:
        target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(manifest['builds'])} attempted builds and {len(manifest['arena_logs'])} Arena logs")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    freeze(p.parse_args().check)
