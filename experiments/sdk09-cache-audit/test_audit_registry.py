"""Small format/safety tests; synthetic files only, no SDK or NPU dependency."""
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

import audit_registry as audit


def vector(value):
    return struct.pack("<Q", len(value)) + value


def registry(rows):
    result = struct.pack("<Q", len(rows))
    for path, key, image in rows:
        result += vector(path.encode()) + struct.pack("<Q", len(key))
        result += b"".join(struct.pack("<Q", item) for item in key) + vector(image)
    return result


def elf(payload):
    names = b"\0.shstrtab\0furiosa_kernels\0"
    start = 64 + len(names)
    table = (start + len(payload) + 7) // 8 * 8
    result = bytearray(table + 3 * 64)
    result[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<Q", result, 40, table)
    struct.pack_into("<HHH", result, 58, 64, 3, 1)
    result[64:64 + len(names)] = names
    result[start:start + len(payload)] = payload
    struct.pack_into("<IIQQQQIIQQ", result, table + 64,
                     1, 3, 0, 0, 64, len(names), 0, 0, 1, 0)
    struct.pack_into("<IIQQQQIIQQ", result, table + 128,
                     11, 1, 0, 0, start, len(payload), 0, 0, 1, 0)
    return bytes(result)


class RegistryTests(unittest.TestCase):
    def test_concrete_and_generic_entries(self):
        rows = audit.parse_registry(registry([("crate::a", [], b"abc"),
                                               ("crate::b", [32, 1], b"xyz")]))
        self.assertEqual(rows[0]["key"], [])
        self.assertEqual(rows[1]["key"], [32, 1])
        self.assertEqual(rows[0]["sha256"], hashlib.sha256(b"abc").hexdigest())

    def test_section_extraction(self):
        payload = registry([("crate::a", [], b"image")])
        self.assertEqual(bytes(audit.elf_section(elf(payload))), payload)

    def test_reject_wrong_elf_class(self):
        raw = bytearray(elf(registry([])))
        raw[4] = 1
        with self.assertRaises(audit.FormatError):
            audit.elf_section(raw)

    def test_reject_out_of_bounds_section(self):
        raw = bytearray(elf(registry([])))
        table = struct.unpack_from("<Q", raw, 40)[0]
        struct.pack_into("<Q", raw, table + 128 + 32, 1 << 63)
        with self.assertRaises(audit.FormatError):
            audit.elf_section(raw)

    def test_reject_duplicate_identity(self):
        with self.assertRaises(audit.FormatError):
            audit.parse_registry(registry([("a", [1], b"x"), ("a", [1], b"y")]))

    def test_generic_key_distinguishes_same_path(self):
        rows = audit.parse_registry(registry([("a", [16], b"x"), ("a", [32], b"y")]))
        self.assertEqual(len(rows), 2)

    def test_reject_truncated_and_trailing_bytes(self):
        payload = registry([("a", [], b"image")])
        for malformed in [payload[:-1], payload + b"\0", payload[:6]]:
            with self.subTest(size=len(malformed)), self.assertRaises(audit.FormatError):
                audit.parse_registry(malformed)

    def test_reject_impossible_counts(self):
        with self.assertRaises(audit.FormatError):
            audit.parse_registry(struct.pack("<Q", 1 << 63))
        with self.assertRaises(audit.FormatError):
            audit.parse_registry(struct.pack("<Q", 1) + vector(b"a") +
                                 struct.pack("<Q", 1 << 63) + b"\0" * 16)

    def test_reject_non_utf8_path(self):
        with self.assertRaises(audit.FormatError):
            audit.parse_registry(struct.pack("<Q", 1) + vector(b"\xff") +
                                 struct.pack("<Q", 0) + vector(b"image"))

    def test_comparison_matches_identity_not_position(self):
        left = {"records": audit.parse_registry(registry([("a", [], b"x"),
                 ("b", [16], b"same"), ("left", [], b"only")]))}
        right = {"records": audit.parse_registry(registry([("b", [16], b"same"),
                  ("right", [], b"only"), ("a", [], b"changed")]))}
        result = audit.compare(left, right)
        self.assertEqual(result["common_count"], 2)
        self.assertEqual(result["equal_common_count"], 1)
        self.assertEqual(result["differences"][0]["path"], "a")
        self.assertEqual(result["left_only"], [{"path": "left", "key": []}])
        self.assertEqual(result["right_only"], [{"path": "right", "key": []}])

    def test_read_artifact_preserves_read_only_input(self):
        original = elf(registry([("a", [], b"data")]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.elf"
            path.write_bytes(original)
            path.chmod(0o444)
            before = path.stat()
            result = audit.read_artifact(path)
            after = path.stat()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
            self.assertEqual(before.st_mode, after.st_mode)
            self.assertEqual(result["elf_sha256"], hashlib.sha256(original).hexdigest())


if __name__ == "__main__":
    unittest.main()
