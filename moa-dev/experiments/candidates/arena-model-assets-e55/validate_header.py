#!/usr/bin/env python3
"""Local, stdlib-only header audit; Arena has no Python interpreter.

Read only the 8-byte prefix and <=16-MiB JSON header, using unbuffered I/O.
loader_contract_pass covers *header-observable* loader checks, not successful
loading. semantic_dtype_pass adds intended BF16/FP8 dtype checks where SDK
0.8.1 actually reinterprets raw bytes. Never inspect tensor values or use NPU.
The generated contract is also consumed by the standalone Rust inspector.
"""

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
from fingerprint import (AssetError, MAX_HEADER_BYTES, model_directory,
                         public_stat, reject_json_constant, safe_error,
                         stat_identity, unique_object)

CONTRACT_PATH = Path(__file__).with_name("loader-header-contract.json")
USIZE_MAX = (1 << 64) - 1  # The installed x86_64 safetensors 0.4.5 host.
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}


def generated_contract():
    """Explicit call expansion of the unchanged original/E55 load.rs."""
    tensors = []

    def add(name, dtype, shape, strict=False, symbol="load_bf16"):
        tensors.append({
            "name": name, "dtype": dtype, "shape": shape,
            "storage_bytes": 4 if shape is None else math.prod(shape) * DTYPE_BYTES[dtype],
            "loader_dtype_check": "strict" if strict else "semantic_only",
            "scalar_shape_unchecked": shape is None,
            "loader_symbol": symbol,
        })

    add("model.language_model.embed_tokens.weight", "BF16", [262144, 3840])
    add("model.language_model.norm.weight", "BF16", [3840])
    add("model.vision_embedder.pos_embedding", "BF16", [1120, 2, 3840],
        symbol="load_vision::HostTensor::from_safetensors")
    for suffix, shape in (
        ("patch_ln1.weight", [6912]), ("patch_ln1.bias", [6912]),
        ("patch_dense.weight", [3840, 6912]), ("patch_dense.bias", [3840]),
        ("patch_ln2.weight", [3840]), ("patch_ln2.bias", [3840]),
        ("pos_norm.weight", [3840]), ("pos_norm.bias", [3840]),
    ):
        add("model.vision_embedder." + suffix, "BF16", shape)
    add("model.embed_vision.embedding_projection.weight", "BF16", [3840, 3840])
    add("model.embed_audio.embedding_projection.weight", "BF16", [3840, 640])
    for layer in range(48):
        prefix = f"model.language_model.layers.{layer}"
        for norm in ("pre_feedforward_layernorm", "post_feedforward_layernorm", "input_layernorm"):
            add(f"{prefix}.{norm}.weight", "BF16", [3840])
        for projection, rows, cols in (
            ("up_proj", 15360, 3840), ("gate_proj", 15360, 3840),
            ("down_proj", 3840, 15360),
        ):
            name = f"{prefix}.mlp.{projection}"
            add(name + ".weight_packed", "U8", [rows, cols // 2], True, "load_f4")
            add(name + ".weight_scale", "F8_E4M3", [rows, cols // 16], symbol="load_f8")
            add(name + ".weight_global_scale", "F32", None, True, "read_f32")
        full = layer % 6 == 5
        q, kv, head = (8192, 512, 512) if full else (4096, 2048, 256)
        projections = [("q_proj", q), ("k_proj", kv)]
        if not full:
            projections.append(("v_proj", kv))
        for projection, rows in projections:
            add(f"{prefix}.self_attn.{projection}.weight", "F8_E4M3", [rows, 3840], symbol="load_f8")
        for projection, rows in projections:
            add(f"{prefix}.self_attn.{projection}.weight_scale", "BF16", [rows, 1], True, "load_bf16_column")
        for norm in ("q_norm", "k_norm"):
            add(f"{prefix}.self_attn.{norm}.weight", "BF16", [head])
        add(f"{prefix}.post_attention_layernorm.weight", "BF16", [3840])
        add(f"{prefix}.self_attn.o_proj.weight", "F8_E4M3", [3840, q], symbol="load_f8")
        add(f"{prefix}.self_attn.o_proj.weight_scale", "BF16", [3840, 1], True, "load_bf16_column")
        add(f"{prefix}.layer_scalar", "BF16", [1], symbol="load_scalar_bf16")
    assert len(tensors) == len({item["name"] for item in tensors}) == 1149
    return {
        "schema": "MOA_MODEL_LOADER_HEADER_CONTRACT_V1",
        "source_sha256": {
            "load_rs": "fb05a5d79a63c92909f64e1d4cdd699556a42b4f369e88e2c0c11d46160ad892",
            "axes_rs": "a9507011038928391bcae1068ab443a6005140f339ba803e41e0ffb07f7f843a",
            "lib_rs": "60b0a34c636ef6fe7857d74f560f13334a7cfc5e18a0c9e3a77549516f16141e",
            "std_tensor_memory_rs": "fec623ddf8fadd5cf31b32beb747243c2c463c48e4f73bb29bb2b6ef5d1aaff1",
        },
        "source_provenance": {
            "original_upstream_commit": "8559916",
            "original_crate": "moa-dev/experiments/artifacts/original-output-boundary-probe-control/crate",
            "e55_crate": "furiosa-opt-gemma4-12B",
            "original_e55_load_axes_lib_byte_identical": True,
            "sdk": "furiosa-opt-std 0.8.1",
            "safetensors": "0.4.5",
            "verified_date": "2026-09-22 Asia/Seoul",
        },
        "tensor_count": len(tensors),
        "required_storage_bytes": sum(item["storage_bytes"] for item in tensors),
        "intended_dtype_counts": dict(sorted(Counter(item["dtype"] for item in tensors).items())),
        "loader_dtype_check_counts": dict(sorted(Counter(item["loader_dtype_check"] for item in tensors).items())),
        "scope": {
            "required_views_including_cpu_only_position_embedding": True,
            "payload_values_checked": False,
            "global_scale_finite_nonzero_checked": False,
            "reciprocal_global_scale_finite_checked_by_loader": False,
            "global_scale_shape_policy": "No exact shape requirement; F32 and exactly 4 data bytes. Safetensors format still requires shape product one.",
            "semantic_only_dtype_policy": "SDK from_safetensors checks exact shape/raw byte length, then reinterprets; intended dtype is an additional audit check.",
            "extra_tensors_policy": "Allowed, but each must satisfy safetensors format/offset/byte-length validation.",
            "not_validated": ["tensor numerical values", "tokenizer assets", "device initialization", "HBM upload", "model execution"],
        },
        "required_tensors": tensors,
    }


def load_contract():
    raw = CONTRACT_PATH.read_bytes()
    actual = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_json_constant)
    if actual != generated_contract():
        raise AssetError("contract_does_not_match_checked_loader_expansion")
    return actual, hashlib.sha256(raw).hexdigest()


def checked_product(shape, width):
    if not isinstance(shape, list):
        raise AssetError("invalid_tensor_shape_type")
    count = 1
    for value in shape:
        if type(value) is not int or not 0 <= value <= USIZE_MAX:
            raise AssetError("invalid_tensor_dimension")
        count *= value
        if count > USIZE_MAX:
            raise AssetError("tensor_shape_overflow")
    count *= width
    if count > USIZE_MAX:
        raise AssetError("tensor_byte_length_overflow")
    return count


def finite_json_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AssetError("nonfinite_json_number")
    return parsed


def check_json_depth(root):
    pending = [(root, 1)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, (dict, list)):
            # serde_json's default budget starts at 128 and rejects at zero.
            if depth >= 128:
                raise AssetError("json_container_depth_limit")
            children = item.values() if isinstance(item, dict) else item
            pending.extend((value, depth + 1) for value in children)


def parse_header(stream, file_size):
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise AssetError("short_safetensors_length_prefix")
    size = int.from_bytes(prefix, "little")
    if not 0 < size <= MAX_HEADER_BYTES:
        raise AssetError("safetensors_header_length_out_of_bounds")
    if size > file_size - 8:
        raise AssetError("safetensors_header_exceeds_file")
    raw = stream.read(size)
    if len(raw) != size:
        raise AssetError("short_safetensors_header")
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                            parse_constant=reject_json_constant, parse_float=finite_json_float)
        check_json_depth(header)
        # Python JSON accepts escaped lone surrogates; Rust serde_json does not.
        json.dumps(header, ensure_ascii=False).encode("utf-8")
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise AssetError("invalid_safetensors_json_header") from exc
    if not isinstance(header, dict):
        raise AssetError("safetensors_header_not_object")
    metadata = header.get("__metadata__")
    if metadata is not None and (not isinstance(metadata, dict) or
                                 any(not isinstance(v, str) for v in metadata.values())):
        raise AssetError("invalid_safetensors_metadata")
    tensors = {name: desc for name, desc in header.items() if name != "__metadata__"}
    spans = []
    payload_bytes = file_size - 8 - size
    for desc in tensors.values():
        if not isinstance(desc, dict):
            raise AssetError("tensor_descriptor_not_object")
        dtype = desc.get("dtype")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise AssetError("unsupported_safetensors_dtype")
        nbytes = checked_product(desc.get("shape"), DTYPE_BYTES[dtype])
        offsets = desc.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
            type(v) is not int or not 0 <= v <= USIZE_MAX for v in offsets
        ):
            raise AssetError("invalid_tensor_offsets_type")
        start, end = offsets
        if end < start or end > payload_bytes:
            raise AssetError("tensor_offsets_out_of_bounds")
        if end - start != nbytes:
            raise AssetError("tensor_dtype_shape_byte_length_mismatch")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise AssetError("tensor_offsets_gap_or_overlap")
        cursor = end
    if cursor != payload_bytes:
        raise AssetError("unaccounted_payload_bytes")
    return tensors, {
        "length_bytes": size, "sha256": hashlib.sha256(raw).hexdigest(),
        "sha256_scope": "raw_json_header_only_not_model_payload_identity",
        "bytes_read": 8 + size, "tensor_payload_bytes_read": 0,
        "tensor_count": len(tensors), "declared_payload_bytes": payload_bytes,
        "format_valid": True,
        "additional_parser_policy": "Header limit 16 MiB and duplicate JSON keys rejected.",
    }


def check_contract(tensors, contract):
    loader_errors, semantic_errors = [], []
    required = contract["required_tensors"]
    for index, rule in enumerate(required):
        name = rule["name"]
        actual = tensors.get(name)
        if actual is None:
            loader_errors.append({"code": "missing_required_tensor", "contract_index": index})
            continue
        if rule["shape"] is not None and actual["shape"] != rule["shape"]:
            loader_errors.append({"code": "loader_shape_mismatch", "contract_index": index})
        begin, end = actual["data_offsets"]
        if end - begin != rule["storage_bytes"]:
            loader_errors.append({"code": "loader_byte_length_mismatch", "contract_index": index})
        if actual["dtype"] != rule["dtype"]:
            errors = loader_errors if rule["loader_dtype_check"] == "strict" else semantic_errors
            errors.append({"code": "loader_dtype_mismatch" if errors is loader_errors else "semantic_dtype_mismatch", "contract_index": index})
    return {
        "loader_contract_pass": not loader_errors,
        "semantic_dtype_pass": not loader_errors and not semantic_errors,
        "loader_contract_scope": "Header-observable checks only; global scale values remain unchecked.",
        "required_tensor_count": len(required),
        "extra_tensor_count": len(set(tensors) - {rule["name"] for rule in required}),
        "loader_error_count": len(loader_errors), "semantic_error_count": len(semantic_errors),
        "loader_errors": loader_errors[:32], "semantic_errors": semantic_errors[:32],
        "errors_truncated": len(loader_errors) > 32 or len(semantic_errors) > 32,
    }


def inspect_file(path, contract):
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise AssetError("not_regular_file")
    # Unbuffered is intentional: BufferedReader may prefetch tensor payload.
    with path.open("rb", buffering=0) as stream:
        opened = os.fstat(stream.fileno())
        if stat_identity(before) != stat_identity(opened):
            raise AssetError("file_changed_before_open")
        tensors, header = parse_header(stream, opened.st_size)
        checks = check_contract(tensors, contract)
        after_fd = os.fstat(stream.fileno())
    after_path = path.stat()
    if any(stat_identity(value) != stat_identity(before) for value in (after_fd, after_path)):
        raise AssetError("file_changed_during_header_read")
    return {"file_stat": public_stat(before), "file_unchanged_by_stat": True,
            "header": header, **checks}


def self_test(contract):
    """Synthetic header bytes only: the 9.3-GB payload is never allocated."""
    import copy

    if not __debug__:
        raise AssetError("self_test_requires_assertions_no_python_optimized_mode")

    def synthetic(changes=None):
        header = {r["name"]: {"dtype": r["dtype"], "shape": r["shape"] or []}
                  for r in contract["required_tensors"]}
        if changes:
            changes(header)
        cursor = 0
        for desc in header.values():
            nbytes = math.prod(desc["shape"]) * DTYPE_BYTES[desc["dtype"]]
            desc["data_offsets"] = [cursor, cursor + nbytes]
            cursor += nbytes
        return header, cursor

    class HeaderOnly(io.BytesIO):
        def read(self, size=-1):
            if size < 0 or self.tell() + size > len(self.getbuffer()):
                raise AssertionError("attempted_payload_read")
            return super().read(size)

    def parse(header, payload):
        raw = json.dumps(header).encode()
        return parse_header(HeaderOnly(len(raw).to_bytes(8, "little") + raw), 8 + len(raw) + payload)

    passed = []
    header, payload = synthetic()
    tensors, parsed = parse(header, payload)
    assert payload == 9304785568 and parsed["tensor_payload_bytes_read"] == 0
    assert check_contract(tensors, contract)["semantic_dtype_pass"]
    passed.append("all_1149_required_header_only_positive")
    first = contract["required_tensors"][0]["name"]
    globals_ = next(r["name"] for r in contract["required_tensors"] if r["shape"] is None)
    fp8 = next(r["name"] for r in contract["required_tensors"] if r["dtype"] == "F8_E4M3")
    strict = next(r["name"] for r in contract["required_tensors"] if r["dtype"] == "BF16" and r["loader_dtype_check"] == "strict")
    for label, change, expected in (
        ("missing", lambda h: h.pop(first), (False, False)),
        ("semantic_only_dtype", lambda h: h[first].update(dtype="F16"), (True, False)),
        ("semantic_only_fp8_dtype", lambda h: h[fp8].update(dtype="U8"), (True, False)),
        ("strict_dtype", lambda h: h[strict].update(dtype="F16"), (False, False)),
        ("shape_same_bytes", lambda h: h[first].update(shape=[3840, 262144]), (False, False)),
        ("wrong_width_dtype", lambda h: h[first].update(dtype="U8"), (False, False)),
        ("global_scalar_shape_1_1", lambda h: h[globals_].update(shape=[1, 1]), (True, True)),
        ("global_wrong_size", lambda h: h[globals_].update(shape=[2]), (False, False)),
    ):
        h, n = synthetic(change)
        actual = check_contract(parse(h, n)[0], contract)
        assert (actual["loader_contract_pass"], actual["semantic_dtype_pass"]) == expected, label
        passed.append(label)
    bad = copy.deepcopy(header)
    bad[first]["data_offsets"][1] += 1
    for label, h, n, code in (
        ("offset_byte_mismatch", bad, payload, "tensor_dtype_shape_byte_length_mismatch"),
        ("trailing_payload", header, payload + 1, "unaccounted_payload_bytes"),
        ("truncated_payload", header, payload - 1, "tensor_offsets_out_of_bounds"),
        ("metadata_wrong_type", {**header, "__metadata__": {"test": 1}}, payload, "invalid_safetensors_metadata"),
    ):
        try:
            parse(h, n)
        except AssetError as exc:
            assert str(exc) == code, (label, str(exc))
        else:
            raise AssertionError(label)
        passed.append(label)
    for label, raw, file_size, code in (
        ("duplicate_json_key", b'{"x":{},"x":{}}', None, "duplicate_json_key"),
        ("nonstandard_nan", b'{"x":NaN}', None, "nonstandard_json_constant"),
        ("json_float_overflow", b'{"x":1e400}', None, "nonfinite_json_number"),
        ("invalid_json", b'{', None, "invalid_safetensors_json_header"),
    ):
        encoded = len(raw).to_bytes(8, "little") + raw
        try:
            parse_header(HeaderOnly(encoded), len(encoded) if file_size is None else file_size)
        except AssetError as exc:
            assert str(exc) == code, (label, str(exc))
        else:
            raise AssertionError(label)
        passed.append(label)
    # Zero-sized extras and nullable metadata match safetensors 0.4.5.
    h = {**header, "__metadata__": None,
         "empty_extra": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]}}
    assert check_contract(parse(h, payload)[0], contract)["semantic_dtype_pass"]
    passed.append("zero_size_extra_and_nullable_metadata")
    deeply_nested = []
    for _ in range(127):
        deeply_nested = [deeply_nested]
    try:
        check_json_depth(deeply_nested)
    except AssetError as exc:
        assert str(exc) == "json_container_depth_limit"
    else:
        raise AssertionError("json_container_depth_limit")
    passed.append("json_container_depth_limit")
    for label, h, n, code in (
        ("offset_gap", {"a": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}},
         2, "tensor_offsets_gap_or_overlap"),
        ("offset_overlap", {
            "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
            "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
        }, 3, "tensor_offsets_gap_or_overlap"),
        ("boolean_dimension", {"a": {"dtype": "U8", "shape": [True], "data_offsets": [0, 1]}},
         1, "invalid_tensor_dimension"),
        ("negative_dimension", {"a": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 1]}},
         1, "invalid_tensor_dimension"),
        ("dimension_overflow", {"a": {"dtype": "U8", "shape": [USIZE_MAX + 1], "data_offsets": [0, 1]}},
         1, "invalid_tensor_dimension"),
        ("product_overflow", {"a": {"dtype": "U8", "shape": [USIZE_MAX, 2], "data_offsets": [0, 1]}},
         1, "tensor_shape_overflow"),
        ("byte_width_overflow", {"a": {"dtype": "F32", "shape": [USIZE_MAX], "data_offsets": [0, 1]}},
         1, "tensor_byte_length_overflow"),
        ("negative_offset", {"a": {"dtype": "U8", "shape": [1], "data_offsets": [-1, 0]}},
         1, "invalid_tensor_offsets_type"),
        ("fp8_wrong_serialized_spelling", {"a": {"dtype": "F8_E4M3FN", "shape": [1], "data_offsets": [0, 1]}},
         1, "unsupported_safetensors_dtype"),
    ):
        try:
            parse(h, n)
        except AssetError as exc:
            assert str(exc) == code, (label, str(exc))
        else:
            raise AssertionError(label)
        passed.append(label)
    for label, encoded, logical_size, code in (
        ("short_prefix", b"\x00", 1, "short_safetensors_length_prefix"),
        ("header_limit", (MAX_HEADER_BYTES + 1).to_bytes(8, "little"),
         MAX_HEADER_BYTES + 9, "safetensors_header_length_out_of_bounds"),
        ("header_exceeds_file", (8).to_bytes(8, "little") + b"{}", 10,
         "safetensors_header_exceeds_file"),
        ("header_short_read", (8).to_bytes(8, "little") + b"{}", 16,
         "short_safetensors_header"),
    ):
        try:
            parse_header(io.BytesIO(encoded), logical_size)
        except AssetError as exc:
            assert str(exc) == code, (label, str(exc))
        else:
            raise AssertionError(label)
        passed.append(label)
    return {"status": "passed", "test_count": len(passed), "tests": passed,
            "payload_bytes_allocated_or_read": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    report = {"schema": "MOA_MODEL_HEADER_AUDIT_V1", "tensor_values_checked": False,
              "global_scale_finite_nonzero_checked": False, "model_execution_validated": False}
    try:
        contract, digest = load_contract()
        report.update(contract_sha256=digest, source_sha256=contract["source_sha256"])
        if args.self_test:
            report["self_test"] = self_test(contract)
            report["status"] = "self_test_passed"
            code = 0
        else:
            selector, directory = model_directory()
            report["selector"] = selector
            report.update(inspect_file(directory / "model.safetensors", contract))
            code = 0 if report["semantic_dtype_pass"] else 1
            report["status"] = "header_checks_passed" if code == 0 else "header_checks_failed"
    except Exception as exc:
        report.update(status="error", error=safe_error(exc))
        code = 1
    print(json.dumps(report, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
