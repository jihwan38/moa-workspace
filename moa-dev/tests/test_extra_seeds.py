"""Regression tests for development validation guards, without Docker or NPU jobs."""

import importlib.util
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / "extra_seeds.py"
spec = importlib.util.spec_from_file_location("extra_seeds", MODULE)
extra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extra)


def synthetic_six_run_log():
    # Parser fixture only: copy the authentic output format, never pretend these
    # invented six-run measurements were obtained from hardware.
    baseline = (Path(__file__).parent / "fixtures/baseline-69783.txt").read_text()
    lines = ["NPU kernel tests -- 3 cases x 6 independently-seeded runs against a precomputed reference"]
    for kernel, outputs in extra.OUTPUTS.items():
        lines.append(f"\n==> {kernel}")
        for run in range(3, 9):
            for output in outputs:
                start = kernel + (" " + output if output else "") + " run0"
                original = next(line for line in baseline.splitlines() if line.startswith("[" + start))
                lines.append(original.replace("run0", f"run{run}"))
        lines.append("    median cycles=13 (of 6 runs: [10, 11, 12, 13, 14, 15])")
    lines.append("\nall 3 tests passed\n")
    return "\n".join(lines)


class ExtraSeedParserTests(unittest.TestCase):
    def test_complete_six_run_output_is_development_only(self):
        result = extra.parse_log(synthetic_six_run_log())
        self.assertEqual(result["output_comparisons_passed"], 30)
        self.assertEqual(result["run_ids"], [3, 4, 5, 6, 7, 8])
        self.assertFalse(result["official_score"])
        self.assertNotIn("score", result)
        self.assertEqual(result["cycles_diagnostic_only"]["decoder_feedforward"]["upper_median_cycles_diagnostic"], 13)

    def test_real_official_three_run_log_is_rejected(self):
        log = (Path(__file__).parent / "fixtures/baseline-69783.txt").read_text()
        with self.assertRaises(ValueError):
            extra.parse_log(log)

    def test_failed_element_cannot_be_hidden_by_final_summary(self):
        for change in (lambda s: s.replace("-> PASS", "-> FAIL", 1),
                       lambda s: s.replace("tol=100.00%", "tol=99.99%", 1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                extra.parse_log(change(synthetic_six_run_log()))

    def test_missing_duplicate_or_wrong_run_comparisons_fail(self):
        log = synthetic_six_run_log()
        line = next(line for line in log.splitlines() if line.startswith("[sliding_project_qkv q run3"))
        variants = [log.replace(line + "\n", "", 1),
                    log.replace(line, line + "\n" + line, 1),
                    log.replace("q run3", "q run0", 1)]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                extra.parse_log(variant)

    def test_cycles_require_six_positive_samples_and_upper_median(self):
        log = synthetic_six_run_log()
        variants = [log.replace("median cycles=13", "median cycles=12", 1),
                    log.replace("[10, 11, 12, 13, 14, 15]", "[0, 11, 12, 13, 14, 15]", 1),
                    log.replace("of 6 runs", "of 3 runs", 1),
                    log.replace("[10, 11, 12, 13, 14, 15]", "[10, 11, 12]", 1)]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                extra.parse_log(variant)

    def test_missing_or_duplicate_summary_fails(self):
        log = synthetic_six_run_log()
        cycle = "    median cycles=13 (of 6 runs: [10, 11, 12, 13, 14, 15])"
        for variant in (log.replace(cycle, "", 1), log.replace(cycle, cycle + "\n" + cycle, 1),
                        log.replace("all 3 tests passed", ""), log + "\nall 3 tests passed\n"):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                extra.parse_log(variant)


class SnapshotTests(unittest.TestCase):
    def test_harness_and_generator_patch_only_count_and_offset(self):
        harness = extra.read(extra.CRATE / "src/bin/test_kernels.rs")
        generator = extra.read(extra.CRATE / "scripts/generate_references.py")
        patched_harness = extra.extended_harness(harness)
        restored_harness = patched_harness.replace(b"const RUNS: usize = 6;\nconst RUN_OFFSET: usize = 3;",
                                                     b"const RUNS: usize = 3;").replace(
                                                         b"for run in RUN_OFFSET..RUN_OFFSET + RUNS {", b"for run in 0..RUNS {")
        patched_generator = extra.extended_generator(generator)
        restored_generator = patched_generator.replace(b"\nRUNS = 6\nRUN_OFFSET = 3\n", b"\nRUNS = 3\n").replace(
            b"for run in range(RUN_OFFSET, RUN_OFFSET + RUNS):", b"for run in range(RUNS):")
        self.assertEqual(restored_harness, harness)
        self.assertEqual(restored_generator, generator)
        compile(patched_generator, "scratch/generate_references.py", "exec")

    def test_changed_upstream_layout_fails_closed(self):
        with self.assertRaises(ValueError):
            extra.extended_harness(b"const RUNS: usize = 4;")
        generator = extra.read(extra.CRATE / "scripts/generate_references.py")
        with self.assertRaises(ValueError):
            extra.extended_generator(generator + b"\nRUNS = 3\n")

    def test_snapshot_file_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "file").write_bytes(b"before")
            manifest = {"kind": "extra_seed_references", "run_ids": extra.RUN_IDS,
                        "files": {"file": extra.fingerprint(b"before")}}
            (directory / "references.json").write_text(json.dumps(manifest))
            self.assertEqual(extra.load(directory, "references.json", manifest["kind"]), manifest)
            (directory / "file").write_bytes(b"after")
            with self.assertRaises(ValueError):
                extra.load(directory, "references.json", manifest["kind"])

    def test_duplicate_labels_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(extra, "ARTIFACTS", Path(temporary)):
            extra.new_directory("example")
            with self.assertRaises(FileExistsError):
                extra.new_directory("example")
            with self.assertRaises(ValueError):
                extra.new_directory("../outside")

    def test_fixture_requires_exact_seed_set_and_original_tensor_schema(self):
        def fixture(header):
            payload = json.dumps(header).encode()
            return struct.pack("<Q", len(payload)) + payload

        shape = {"dtype": "F32", "shape": [3840], "data_offsets": [0, 15360]}
        baseline = fixture({"run0.decoder_feedforward.expected": shape})
        header = {f"run{run}.decoder_feedforward.expected": shape for run in extra.RUN_IDS}
        self.assertEqual(extra.verify_fixture(baseline, fixture(header))["tensors"], 6)
        for altered in ({**header, "run0.decoder_feedforward.expected": shape},
                        {key: value for key, value in header.items() if "run8" not in key},
                        {**header, "run8.decoder_feedforward.expected": {**shape, "shape": [32]}}):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                extra.verify_fixture(baseline, fixture(altered))


if __name__ == "__main__":
    unittest.main()
