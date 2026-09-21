#!/usr/bin/env python3
"""Build-machine-only CLI checks. Arena runs the Rust ELF, never this script."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def check(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    binary = Path(sys.argv[1]).resolve()
    artifact = Path(sys.argv[2]).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    contract_raw = (Path(__file__).parent.parent / "loader-header-contract.json").read_bytes()
    contract = json.loads(contract_raw)
    passed = []

    def run(root, args=()):
        result = subprocess.run([str(binary), *args], env={} if root is None else {"RNGD_MODEL_DIR": str(root)},
                                text=True, capture_output=True, timeout=15, check=False)
        check(not result.stderr, "unexpected_stderr")
        report = json.loads(result.stdout)
        if root is not None:
            check(str(root) not in result.stdout, "private_path_in_stdout")
        check("private_fixture_data" not in result.stdout, "contents_in_stdout")
        check("private_tensor_name" not in result.stdout, "tensor_name_in_stdout")
        return result.returncode, report

    code, report = run(None)
    check(code == 1 and report["error"]["code"] == "rngd_model_dir_unset", "unset_env_failed")
    passed.append("unset_env_sanitized_error")
    code, report = run(None, ["--invalid"])
    check(code == 2 and report["error"]["code"] == "invalid_arguments", "invalid_args_failed")
    passed.append("invalid_args_sanitized_error")

    with tempfile.TemporaryDirectory(prefix="private-rust-fixture-", dir=artifact) as temp:
        root = Path(temp)
        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json"]:
            (root / name).write_bytes(b"private_fixture_data")
        header, offset = {}, 0
        for tensor in contract["required_tensors"]:
            size = tensor["storage_bytes"]
            header[tensor["name"]] = {"dtype": tensor["dtype"], "shape": tensor["shape"] or [],
                                      "data_offsets": [offset, offset + size]}
            offset += size
        raw = json.dumps(header, separators=(",", ":")).encode()
        weight = root / "model.safetensors"
        with weight.open("wb") as f:
            f.write(len(raw).to_bytes(8, "little"))
            f.write(raw)
            # Sparse logical extent permits a real CLI+fstat check with only a small
            # header physically allocated. --header-only reads no tensor payload.
            f.truncate(8 + len(raw) + offset)
        logical_size = weight.stat().st_size
        allocated_size = weight.stat().st_blocks * 512
        check(allocated_size < 1024 * 1024, "fixture_not_sparse")
        code, report = run(root, ["--header-only"])
        check(code == 0 and report["status"] == "header_only", "valid_header_cli_failed")
        w = report["files"][0]
        check(w["header_contract"]["loader_contract_pass"] is True, "loader_contract_not_pass")
        check(w["header_contract"]["semantic_dtype_pass"] is True, "semantic_contract_not_pass")
        check(w["header_contract"]["required_tensors_found"] == 1149, "count_not_1149")
        check(w["header"]["sha256"] == hashlib.sha256(raw).hexdigest(), "header_sha_mismatch")
        check(w["full_file"]["sha256"] is None and w["full_file"]["bytes_hashed"] == 0, "header_only_hashed_payload")
        check(w["file_unchanged_by_stat"] is True, "sparse_stat_unstable")
        check(len(report["files"]) == 5, "wrong_basename_count")
        for record in report["files"][1:]:
            check(record["full_file"]["sha256"] == hashlib.sha256(b"private_fixture_data").hexdigest(), "metadata_sha_mismatch")
        passed.append("actual_elf_header_only_1149_sparse_contract_and_four_metadata_hashes")

        (root / "chat_template.json").unlink()
        code, report = run(root, ["--header-only"])
        check(code == 0 and report["status"] == "partial", "optional_missing_not_partial")
        passed.append("optional_metadata_missing_partial_not_error")

        tiny = json.dumps({"private_tensor_name":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}).encode()
        full = len(tiny).to_bytes(8, "little") + tiny + b"abc"
        weight.write_bytes(full)
        code, report = run(root)
        check(code == 1 and report["files"][0]["error"]["code"] == "header_contract_mismatch", "missing_required_not_error")
        check(report["files"][0]["full_file"]["sha256"] == hashlib.sha256(full).hexdigest(), "full_file_digest_wrong")
        passed.append("small_weight_full_sha_correct_and_contract_failure_retained")

    print(json.dumps({"status":"pass", "test_count":len(passed), "tests":passed,
        "binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest(),
        "contract_sha256":hashlib.sha256(contract_raw).hexdigest(),
        "sparse_fixture_logical_bytes":logical_size, "sparse_fixture_allocated_bytes":allocated_size,
        "fixture_removed":True, "npu_execution":False}, indent=2))


if __name__ == "__main__":
    main()
