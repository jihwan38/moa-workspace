#!/usr/bin/env python3
"""Read-only SDK 0.9 embedded-kernel comparison; writes JSON only to stdout.

Wire contract: furiosa-opt-std-0.9.0/src/backend/npu/registry.rs:10.
Bincode's deserialize() uses fixed-width little-endian sequence lengths here.
No subprocesses, objcopy, compiler invocation, or writes to ELF/cache files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys


class FormatError(ValueError):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def bounded(data, offset, size, label):
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise FormatError(f"{label}: range {offset}+{size} exceeds {len(data)} bytes")
    return memoryview(data)[offset:offset + size]


def elf_section(data, wanted=b"furiosa_kernels"):
    """Parse ELF section headers directly, never extract by rewriting an ELF."""
    bounded(data, 0, 64, "ELF header")
    if data[:7] != b"\x7fELF\x02\x01\x01":
        raise FormatError("requires an ELF64, little-endian, version-1 file")
    offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", data, 58)
    if entry_size < 64 or not count or names_index == 0xffff or names_index >= count:
        raise FormatError("missing/unsupported section table (extended numbering unsupported)")
    bounded(data, offset, entry_size * count, "section table")
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * entry_size)
               for i in range(count)]
    names_header = headers[names_index]
    if names_header[1] != 3:  # SHT_STRTAB
        raise FormatError("section-name table is not SHT_STRTAB")
    names = bytes(bounded(data, names_header[4], names_header[5], "section names"))
    matches = []
    for header in headers:
        start = header[0]
        if start >= len(names):
            raise FormatError("section-name offset outside string table")
        end = names.find(b"\0", start)
        if end < 0:
            raise FormatError("unterminated section name")
        if names[start:end] == wanted:
            if header[1] != 1:  # SHT_PROGBITS, not an unbacked NOBITS section
                raise FormatError("kernel section is not SHT_PROGBITS")
            matches.append(bounded(data, header[4], header[5], "kernel section"))
    if len(matches) != 1:
        raise FormatError(f"expected one {wanted.decode()} section; found {len(matches)}")
    return matches[0]


def parse_registry(data):
    position = 0

    def u64():
        nonlocal position
        value = struct.unpack("<Q", bounded(data, position, 8, "u64"))[0]
        position += 8
        return value

    def byte_vector():
        nonlocal position
        length = u64()
        result = bounded(data, position, length, "byte vector")
        position += length
        return result

    count = u64()
    # Every entry needs three u64 lengths, even with empty strings/keys/images.
    if count > (len(data) - position) // 24:
        raise FormatError("entry count exceeds possible registry size")
    records = []
    seen = set()
    for _ in range(count):
        try:
            path = bytes(byte_vector()).decode("utf-8")
        except UnicodeDecodeError as error:
            raise FormatError("function path is not UTF-8") from error
        key_length = u64()
        if key_length > (len(data) - position) // 8:
            raise FormatError("generic-key length exceeds registry size")
        key = tuple(u64() for _ in range(key_length))
        image = byte_vector()
        identity = (path, key)
        if identity in seen:
            raise FormatError(f"duplicate registry identity: {identity}")
        seen.add(identity)
        records.append({"path": path, "key": list(key), "bytes": len(image),
                        "sha256": sha256(image)})
    if position != len(data):
        raise FormatError(f"trailing registry bytes: {len(data) - position}")
    return sorted(records, key=lambda row: (row["path"], row["key"]))


def read_artifact(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    data = path.read_bytes()  # opens in rb, never r+b/wb
    digest = sha256(data)
    section = elf_section(data)
    records = parse_registry(section)
    # Detect accidental concurrent modification; this tool itself never writes.
    after = path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or
            sha256(path.read_bytes()) != digest):
        raise FormatError(f"artifact changed while reading: {path}")
    return {"elf": str(path), "elf_sha256": digest,
            "section_bytes": len(section), "section_sha256": sha256(section),
            "record_count": len(records), "records": records}


def compare(left, right):
    def index(artifact):
        return {(row["path"], tuple(row["key"])): row for row in artifact["records"]}

    a, b = index(left), index(right)
    common = a.keys() & b.keys()
    different = [key for key in sorted(common)
                 if (a[key]["bytes"], a[key]["sha256"]) !=
                    (b[key]["bytes"], b[key]["sha256"])]
    identity = lambda key: {"path": key[0], "key": list(key[1])}
    return {"left": left, "right": right, "common_count": len(common),
            "equal_common_count": len(common) - len(different),
            "different_common_count": len(different),
            "differences": [{**identity(key), "left": a[key], "right": b[key]}
                            for key in different],
            "left_only": [identity(key) for key in sorted(a.keys() - b.keys())],
            "right_only": [identity(key) for key in sorted(b.keys() - a.keys())]}


def known_suite(root):
    root = Path(root)
    pairs = [("q021", root / "runs/a003-b16-f4-mse-lm-fp8-model-build/built/target/release/compare_b16_f4_mse_lm_fp8",
              root / "runs/a006-q021-mse-lm-fp8-server-build/built/target/release/server", 305, 0)]
    for name, common, different in [("test_physical_pool", 21, 19),
                                    ("compare_physical_pool_history", 301, 50),
                                    ("compare_physical_pool_group", 301, 50)]:
        pairs.append((name,
                      root / f"physical-pool32/experiments/physical-pool32/artifacts/{name}-07718f07",
                      root / f"physical-pool64-control/experiments/physical-pool64-control/artifacts/{name}-89c0097c",
                      common, different))
    results = {}
    for name, left, right, common, different in pairs:
        result = compare(read_artifact(left), read_artifact(right))
        result["expected"] = {"common_count": common, "different_common_count": different}
        result["expectation_pass"] = (result["common_count"] == common and
                                      result["different_common_count"] == different)
        results[name] = result
    return {"schema_version": 1, "scope": "archived artifacts only; no universal cache guarantee",
            "all_expectations_pass": all(r["expectation_pass"] for r in results.values()),
            "comparisons": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="list embedded entry hashes")
    inspect.add_argument("elf", type=Path)
    pair = commands.add_parser("compare", help="compare by path and numeric generic key")
    pair.add_argument("left", type=Path)
    pair.add_argument("right", type=Path)
    suite = commands.add_parser("known-suite", help="reproduce the four frozen audit pairs")
    suite.add_argument("--round2-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            result = read_artifact(args.elf)
        elif args.command == "compare":
            result = compare(read_artifact(args.left), read_artifact(args.right))
        else:
            result = known_suite(args.round2_root)
        print(json.dumps(result, indent=2))
        return 0 if result.get("all_expectations_pass", True) else 1
    except (OSError, FormatError, struct.error) as error:
        print(f"registry audit failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
