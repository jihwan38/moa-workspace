#!/usr/bin/env python3
"""Read-only source/host-check provenance; no ELF, SDK compilation or device calls."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "d21224649eef82f5c3bb293aac44b2d1c28b58e8"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def old(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def function(text, name):
    match = re.search(r"async fn " + name + r"\(", text)
    assert match
    start = match.start()
    body = text.index("{", start)
    level = 1
    i = body + 1
    while level:
        if text[i] == "{": level += 1
        elif text[i] == "}": level -= 1
        i += 1
    return text[start:i]


def main():
    original = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE, "src"], cwd=ROOT, text=True).splitlines()
    assert len(original) == 418
    for path in original:
        assert (ROOT / path).read_bytes() == old(path), path
    donor = old("src/bin/test_int4_grouped_complete_ffn.rs").decode()
    current = (ROOT / "src/bin/test_int4_channel_balance.rs").read_text()
    exact_functions = {}
    for name in ("candidate_rest", "production_chain"):
        a, b = function(donor, name), function(current, name)
        assert a == b
        exact_functions[name] = sha(a.encode())
    a = function(donor, "intended_step")
    b = function(current, "intended_step")
    b = b.replace("p:&ProductionWeights,pre:&HbmTensor<bf16,Chip,m![H]>,w:", "p:&ProductionWeights,w:")
    b = b.replace("&r,pre,&mut n", "&r,&p.pre_ffn,&mut n")
    assert a == b
    exact_functions["intended_step_except_explicit_gamma_argument"] = sha(a.encode())
    records = []
    for directory in ("a001-hostcheck", "a002-hostcheck"):
        d = HERE / directory
        run = json.loads((d / "run.json").read_text())
        hashes = json.loads((d / "source-hashes.json").read_text())
        assert run["exit_code"] == 0 and run["result"] == "command_succeeded"
        # a002 was run with only a001 evidence snapshots still untracked.
        # Do not mistake those recorded documents for a source mutation.
        allowed_evidence = {
            "?? experiments/int4-channel-balance-diagnostic/a001-hostcheck/",
            "?? experiments/int4-channel-balance-diagnostic/source-hostcheck.json",
        }
        assert set(run["status"].splitlines()) <= allowed_evidence
        assert (d / "source.diff").read_bytes() == b""
        assert "check" in run["command"] and "furiosa-opt" not in run["command"]
        for path, digest in hashes.items():
            data = (ROOT / path).read_bytes() if path == "Cargo.lock" else subprocess.check_output(["git", "show", f"{run['commit']}:{path}"], cwd=ROOT)
            assert sha(data) == digest, (directory, path)
        if directory == "a002-hostcheck":
            for path, digest in hashes.items(): assert sha((ROOT / path).read_bytes()) == digest, path
        records.append({"run": run, "verified_inputs": len(hashes), "files": {p.name: sha(p.read_bytes()) for p in sorted(d.iterdir())}})
    new = ["src/bin/support/int4_channel_balance.rs", "src/bin/test_int4_channel_balance.rs"]
    print(json.dumps({
        "base": BASE, "source_commit": records[-1]["run"]["commit"],
        "unchanged_existing_src_count": len(original),
        "new_host_only_files": {p: sha((ROOT / p).read_bytes()) for p in new},
        "unchanged_reused_host_bodies": exact_functions,
        "plan_sha256": sha((HERE / "plan.json").read_bytes()), "host_checks": records,
        "source_review": {"agent": "/root/e2e_evidence", "source": "52d821e2", "result": "PASS", "scope": "Read-only arithmetic, signed inverse/range prerequisites, actual-prefix guard order, sampled products, source/control ownership, truthful feedback labels and post-replay scratch immutability. No numerical execution."},
        "predeclared_counts": {"transformed_weight_values": 117964800, "prefix_inverse_values": 7864320,
                               "unquantized_product_pairs": 15728640, "unquantized_dots": 4096,
                               "intended_bf16_values_both_arms": 393216000, "quantized_output_samples": 12288},
        "status": "SOURCE_AND_PLAIN_HOSTCHECK_PASS_ONLY; actual numerical prerequisites, drift and performance remain unexecuted; no ELF/Arena/model adoption.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
