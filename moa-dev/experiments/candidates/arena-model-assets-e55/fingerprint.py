#!/usr/bin/env python3
"""Read-only model asset fingerprints; never print paths or model contents.

Selection matches src/host/load.rs::model_dir. The weight SHA256 covers the
entire file, including its length prefix and JSON header, only when complete.
Its monotonic 40-second budget is checked around each 8-MiB read; a blocked OS
read cannot be interrupted by this cooperative deadline. No partial digest is
emitted. Metadata files are the four names checked by the existing probe.sh.
"""

import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import time


CHUNK_BYTES = 8 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
WEIGHT_HASH_SECONDS = 40.0
METADATA_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
)


class AssetError(Exception):
    """A fixed error code safe to include in output."""


def stat_identity(value):
    # Internal only: detect replacement as well as size/timestamp changes.
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def public_stat(value):
    return {"size_bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def model_directory():
    value = os.environ.get("RNGD_MODEL_DIR")
    if value is None:
        raise AssetError("rngd_model_dir_unset")
    root = Path(value)  # Empty string means current directory, as in PathBuf.
    for selector, candidate in (
        ("direct", root),
        ("Gemma4-12B-it", root / "Gemma4-12B-it"),
        ("gemma-4-12b-it-NVFP4", root / "gemma-4-12b-it-NVFP4"),
    ):
        try:
            if stat.S_ISREG((candidate / "model.safetensors").stat().st_mode):
                return selector, candidate
        except OSError:
            # Rust Path::is_file also returns false on a metadata error.
            continue
    raise AssetError("no_model_safetensors_in_three_candidates")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssetError("duplicate_json_key")
        result[key] = value
    return result


def reject_json_constant(_value):
    raise AssetError("nonstandard_json_constant")


def safetensors_header(stream, file_size):
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise AssetError("short_safetensors_length_prefix")
    size = int.from_bytes(prefix, "little", signed=False)
    if not 0 < size <= MAX_HEADER_BYTES:
        raise AssetError("safetensors_header_length_out_of_bounds")
    if size > file_size - 8:
        raise AssetError("safetensors_header_exceeds_file")
    raw = stream.read(size)
    if len(raw) != size:
        raise AssetError("short_safetensors_header")
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                            parse_constant=reject_json_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AssetError("invalid_safetensors_json_header") from exc
    if not isinstance(header, dict):
        raise AssetError("safetensors_header_not_object")
    if "__metadata__" in header and not isinstance(header["__metadata__"], dict):
        raise AssetError("safetensors_metadata_not_object")
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if any(not isinstance(value, dict) for value in tensors):
        raise AssetError("safetensors_tensor_descriptor_not_object")
    return {
        "status": "complete",
        "length_bytes": size,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sha256_scope": "raw_json_header_bytes_excluding_8_byte_length_prefix",
        "tensor_count": len(tensors),
        "validation": "JSON object and descriptor-object structure only",
    }


def full_file_hash(stream, expected_size, budget_seconds=None):
    started = time.monotonic()
    deadline = None if budget_seconds is None else started + budget_seconds
    stream.seek(0)
    digest = hashlib.sha256()
    count = 0
    complete = True
    while count < expected_size:
        if deadline is not None and time.monotonic() >= deadline:
            complete = False
            break
        chunk = stream.read(min(CHUNK_BYTES, expected_size - count))
        if not chunk:
            raise AssetError("file_short_read")
        digest.update(chunk)
        count += len(chunk)
        if deadline is not None and time.monotonic() >= deadline:
            complete = False
            break
    result = {
        "status": "complete" if complete else "incomplete",
        "sha256": digest.hexdigest() if complete else None,
        "sha256_scope": "entire_file" if complete else None,
        "bytes_hashed": count,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    if budget_seconds is not None:
        result["budget_seconds"] = budget_seconds
        result["budget_enforcement"] = "monotonic_checks_around_each_chunk"
    if not complete:
        result["reason"] = "time_budget_exceeded_no_digest_published"
    return result


def safe_error(exc):
    if isinstance(exc, AssetError):
        return {"code": str(exc)}  # All AssetError messages are fixed literals.
    if isinstance(exc, OSError):
        return {"code": "os_error", "errno": exc.errno}
    return {"code": "unexpected_error", "exception_type": type(exc).__name__}


def fingerprint_file(directory, basename, weight=False):
    path = directory / basename
    result = {"basename": basename}
    try:
        before = path.stat()
    except OSError as exc:
        result["status"] = "missing" if exc.errno in (errno.ENOENT, errno.ENOTDIR) else "error"
        result["error"] = safe_error(exc)
        return result
    result["stat_before"] = public_stat(before)
    opened = after_fd = after_path = None
    failure = None
    try:
        if not stat.S_ISREG(before.st_mode):
            raise AssetError("not_regular_file")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            try:
                if stat_identity(opened) != stat_identity(before):
                    raise AssetError("file_changed_before_open")
                if weight:
                    result["header"] = safetensors_header(stream, opened.st_size)
                result["full_file"] = full_file_hash(
                    stream, opened.st_size, WEIGHT_HASH_SECONDS if weight else None)
            finally:
                after_fd = os.fstat(stream.fileno())
    except Exception as exc:
        failure = safe_error(exc)
    try:
        after_path = path.stat()
        result["stat_after"] = public_stat(after_path)
    except OSError as exc:
        if failure is None:
            failure = safe_error(exc)
    snapshots = (before, opened, after_fd, after_path)
    stable = all(value is not None for value in snapshots)
    if stable:
        stable = all(stat_identity(value) == stat_identity(before) for value in snapshots)
    result["file_unchanged_by_stat"] = stable
    if failure is not None or not stable:
        result["status"] = "error"
        result["error"] = failure or {"code": "file_changed_during_read"}
        for key in ("header", "full_file"):
            if key in result:
                result[key]["status"] = "invalidated"
                result[key]["sha256"] = None
                result[key]["sha256_scope"] = None
        return result
    result["status"] = result["full_file"]["status"]
    return result


def main():
    report = {
        "schema": "MOA_MODEL_ASSET_FINGERPRINT_V1",
        "selector": None,
        "chunk_bytes": CHUNK_BYTES,
        "max_header_bytes": MAX_HEADER_BYTES,
        "weight_hash_budget_seconds": WEIGHT_HASH_SECONDS,
        "files": [],
        "scope": "Read-only asset identity; no model-content output, NPU execution, or model validation.",
    }
    exit_code = 0
    try:
        selector, directory = model_directory()
        report["selector"] = selector
        # Collect small metadata before spending the weight hash time budget.
        metadata = [fingerprint_file(directory, name) for name in METADATA_FILES]
        weight = fingerprint_file(directory, "model.safetensors", weight=True)
        report["files"] = [weight, *metadata]
        statuses = [entry["status"] for entry in report["files"]]
        if "error" in statuses or weight["status"] == "missing":
            report["status"] = "error"
            exit_code = 1
        elif any(status != "complete" for status in statuses):
            # Missing optional template files and incomplete weight SHA are explicit.
            report["status"] = "partial"
        else:
            report["status"] = "complete"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = safe_error(exc)
        exit_code = 1
    print(json.dumps(report, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
