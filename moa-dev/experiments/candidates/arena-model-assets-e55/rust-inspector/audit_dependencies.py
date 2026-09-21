#!/usr/bin/env python3
"""Local build provenance only; not deployed or required on Arena."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(value, code):
    if not value:
        raise RuntimeError(code)


def main():
    crate = Path(sys.argv[1])
    metadata = json.loads((crate / "dependency-metadata.json").read_text())
    # Cargo-generated lockfile fields used here have simple quoted values.
    # Parse only these three exact fields; do not pretend to be a TOML parser.
    lock_hash = {}
    for block in (crate / "Cargo.lock").read_text().split("\n[[package]]\n")[1:]:
        fields = {}
        for field in ["name", "version", "checksum"]:
            found = re.findall(r'^' + field + r' = ("[^"\\]*")$', block, re.MULTILINE)
            check(len(found) <= 1, "ambiguous_cargo_lock_field")
            if found:
                fields[field] = json.loads(found[0])
        check("name" in fields and "version" in fields, "invalid_cargo_lock_package")
        lock_hash[(fields["name"], fields["version"])] = fields.get("checksum")
    cargo_cache = Path(os.environ.get("CARGO_HOME", "/home/moa/.cargo")) / "registry/cache"
    rows = []
    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        if package["source"] is None:
            continue
        source = Path(package["manifest_path"]).parent
        expected_archive = lock_hash[(package["name"], package["version"])]
        archives = list(cargo_cache.glob("*/" + package["name"] + "-" + package["version"] + ".crate"))
        check(len(archives) == 1, "ambiguous_or_missing_cached_archive")
        archive_sha = sha(archives[0])
        check(archive_sha == expected_archive, "dependency_archive_checksum_mismatch")
        file_hashes = []
        prefix = package["name"] + "-" + package["version"] + "/"
        with tarfile.open(archives[0], "r:gz") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                check(member.isfile() and member.name.startswith(prefix), "unexpected_archive_member")
                name = member.name[len(prefix):]
                check(".." not in Path(name).parts and not Path(name).is_absolute(), "unsafe_archive_member")
                expected = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                actual = sha(source / name)
                check(actual == expected, "dependency_source_archive_mismatch")
                file_hashes.append(name + "\t" + actual + "\n")
        rows.append({"name":package["name"], "version":package["version"],
            "cached_archive_sha256":archive_sha,
            "verified_source_file_count":len(file_hashes),
            "source_file_hash_list_sha256":hashlib.sha256("".join(sorted(file_hashes)).encode()).hexdigest()})
    print(json.dumps({"schema":"MOA_INSPECTOR_DEPENDENCY_AUDIT_V1", "status":"pass",
        "scope":"All resolved Cargo.lock registry packages; may include packages not linked into the selected target.",
        "source_list_algorithm":"SHA256 of sorted relative-path TAB actual-file-SHA256 LF records. Every unpacked file checked against cached .crate archive members; complete archive SHA checked against Cargo.lock. No extraction/writes.",
        "package_count":len(rows), "packages":rows}, indent=2))


if __name__ == "__main__":
    main()
