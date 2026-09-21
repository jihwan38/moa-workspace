"""Exercise collector trust boundaries using the authentic Arena job 69783 log.

Run without Docker, an Arena connection, or third-party packages:
    python3 -m unittest discover -s moa-dev/tests -v
"""

from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import call, patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "measure.py"
SPEC = importlib.util.spec_from_file_location("moa_measure_under_test", MODULE_PATH)
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)
FIXTURE = (Path(__file__).parent / "fixtures/baseline-69783.txt").read_text()
QKV_CYCLES = "    median cycles=244051 (of 3 runs: [251764, 244051, 243259])"
PACKAGE_PAYLOAD = {
    "test_runtime": b"synthetic executable bytes, not a live NPU binary",
    "fixtures.safetensors": b"synthetic fixture bytes",
    "remote_entrypoint.sh": b"#!/bin/sh\nexec ./test_runtime\n",
    "job.yaml": b"entrypoint: remote_entrypoint.sh\n",
}
PACKAGE_FILES = {
    name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    for name, content in PACKAGE_PAYLOAD.items()
}


def make_run(job_id, kernels=None):
    return {
        "job": {"id": job_id},
        "kernels": kernels if kernels is not None else measure.parse_log(FIXTURE),
        "valid": True,
        "package_files": copy.deepcopy(PACKAGE_FILES),
    }


class ParseLogTests(unittest.TestCase):
    def test_authentic_baseline_reports_all_outputs_and_samples(self):
        result = measure.parse_log(FIXTURE)
        self.assertEqual(set(result), {
            "sliding_project_qkv", "sliding_attention_output", "decoder_feedforward"
        })
        expected = {
            "sliding_project_qkv": (244051, [251764, 244051, 243259], 9),
            "sliding_attention_output": (404373, [404373, 402874, 404457], 3),
            "decoder_feedforward": (4264297, [4263585, 4267547, 4264297], 3),
        }
        for kernel, (median, cycles, comparisons) in expected.items():
            with self.subTest(kernel=kernel):
                self.assertEqual(result[kernel]["correctness"], "passed")
                self.assertEqual(result[kernel]["median_cycles"], median)
                self.assertEqual(result[kernel]["cycles_by_run"], cycles)
                self.assertEqual(result[kernel]["output_comparisons_passed"], comparisons)

    def test_direct_cli_log_needs_no_submission_wrapper(self):
        # `furiosa-arena logs` omits rngd_test.sh's local submission/wait messages.
        raw_log = FIXTURE[FIXTURE.index("NPU kernel tests --"):]
        self.assertEqual(measure.parse_log(raw_log), measure.parse_log(FIXTURE))

    def test_one_failed_output_invalidates_a_success_summary(self):
        bad = FIXTURE.replace("-> PASS", "-> FAIL", 1)
        self.assertIn("all 3 tests passed", bad)
        with self.assertRaises(ValueError):
            measure.parse_log(bad)

    def test_partial_tolerance_cannot_masquerade_as_pass(self):
        bad = FIXTURE.replace("within tol=100.00%", "within tol=99.99%", 1)
        with self.assertRaises(ValueError):
            measure.parse_log(bad)

    def test_missing_input_configuration_is_rejected(self):
        bad = re.sub(r"^\[sliding_project_qkv [qkv] run2[^\n]*\n", "", FIXTURE, flags=re.M)
        self.assertNotIn("sliding_project_qkv q run2", bad)
        with self.assertRaises(ValueError):
            measure.parse_log(bad)

    def test_duplicate_output_comparison_is_rejected(self):
        line = next(line for line in FIXTURE.splitlines() if line.startswith("[sliding_project_qkv q run0"))
        bad = FIXTURE.replace(line, line + "\n" + line, 1)
        with self.assertRaises(ValueError):
            measure.parse_log(bad)

    def test_missing_cycles_are_rejected(self):
        with self.assertRaises(ValueError):
            measure.parse_log(FIXTURE.replace(QKV_CYCLES, ""))

    def test_wrong_reported_median_is_rejected(self):
        bad = FIXTURE.replace("median cycles=244051", "median cycles=243259", 1)
        with self.assertRaises(ValueError):
            measure.parse_log(bad)

    def test_incomplete_cycle_samples_are_rejected(self):
        bad = FIXTURE.replace(QKV_CYCLES, "    median cycles=244051 (of 2 runs: [244051, 244051])")
        with self.assertRaises(ValueError):
            measure.parse_log(bad)

    def test_zero_cycles_are_rejected_even_when_median_matches(self):
        bad = FIXTURE.replace(QKV_CYCLES, "    median cycles=244051 (of 3 runs: [0, 244051, 251764])")
        with self.assertRaises(ValueError):
            measure.parse_log(bad)

    def test_conflicting_cycle_summaries_are_rejected(self):
        extra = "    median cycles=1 (of 3 runs: [1, 1, 1])"
        bad = FIXTURE.replace(QKV_CYCLES, QKV_CYCLES + "\n" + extra)
        with self.assertRaises(ValueError):
            measure.parse_log(bad)

    def test_missing_kernel_section_is_rejected(self):
        bad = re.sub(r"^==> sliding_attention_output\n.*?(?=^==> decoder_feedforward)",
                     "", FIXTURE, flags=re.M | re.S)
        with self.assertRaises(ValueError):
            measure.parse_log(bad)

    def test_duplicate_kernel_sections_are_rejected(self):
        with self.assertRaises(ValueError):
            measure.parse_log(FIXTURE + "\n" + FIXTURE)

    def test_missing_completion_summary_is_rejected(self):
        with self.assertRaises(ValueError):
            measure.parse_log(FIXTURE.replace("all 3 tests passed", ""))


class CollectTests(unittest.TestCase):
    def test_server_failure_cannot_be_overridden_by_passed_kernel_log(self):
        status = {"id": 69783, "status": "FAILED", "exit_code": 1}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(measure, "ROOT", Path(directory)), \
             patch.object(measure, "cli", side_effect=[json.dumps(status), FIXTURE]), \
             patch.object(measure, "package_fingerprint", return_value=copy.deepcopy(PACKAGE_FILES)), \
             redirect_stdout(io.StringIO()):
            result = measure.collect(69783, "failed-experiment", 30)
            self.assertFalse(result["valid"])
            self.assertNotIn("kernels", result)
            self.assertIn("did not succeed", result["error"])
            self.assertEqual((Path(directory) / result["log"]).read_text(), FIXTURE)

    def test_success_keeps_status_and_hash_of_collected_log(self):
        status = {"id": 69783, "status": "SUCCEEDED", "exit_code": 0}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(measure, "ROOT", Path(directory)), \
             patch.object(measure, "cli", side_effect=[json.dumps(status), FIXTURE]) as cli, \
             patch.object(measure, "package_fingerprint", return_value=copy.deepcopy(PACKAGE_FILES)) as package, \
             redirect_stdout(io.StringIO()):
            result = measure.collect(69783, "genuine-baseline", 30)
            self.assertTrue(result["valid"])
            self.assertEqual(result["job"], status)
            self.assertEqual(result["package_files"], PACKAGE_FILES)
            package.assert_called_once_with(69783)
            self.assertEqual(result["log_sha256"], hashlib.sha256(FIXTURE.encode()).hexdigest())
            self.assertEqual((Path(directory) / result["log"]).read_text(), FIXTURE)
            self.assertEqual(cli.call_args_list, [call("status", 69783), call("logs", 69783)])


class PackageFingerprintTests(unittest.TestCase):
    def fingerprint_archive(self, entries, mtime=0):
        """Stand in only for download; exercise the real tar reader and hashing."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def download(*args):
                self.assertEqual(args, ("download", 69783, "--out",
                                       "/workspace/target/arena-packages/job-69783.tar", "--create-dirs"))
                archive_path = root / "furiosa-opt-gemma4-12B/target/arena-packages/job-69783.tar"
                with tarfile.open(archive_path, "w") as archive:
                    for name, content in entries:
                        info = tarfile.TarInfo(name)
                        info.size = len(content)
                        info.mtime = mtime
                        archive.addfile(info, io.BytesIO(content))
                return "downloaded\n"

            with patch.object(measure, "ROOT", root), patch.object(measure, "cli", side_effect=download):
                return measure.package_fingerprint(69783)

    def test_hashes_actual_payload_and_ignores_tar_timestamps(self):
        first = self.fingerprint_archive(list(PACKAGE_PAYLOAD.items()), mtime=1)
        later = self.fingerprint_archive(list(reversed(PACKAGE_PAYLOAD.items())), mtime=1_000_000)
        self.assertEqual(first, PACKAGE_FILES)
        self.assertEqual(first, later)

    def test_duplicate_tar_member_is_rejected(self):
        entries = list(PACKAGE_PAYLOAD.items()) + [("test_runtime", b"second conflicting binary")]
        with self.assertRaisesRegex(ValueError, "Duplicate archive member"):
            self.fingerprint_archive(entries)

    def test_missing_required_payload_is_rejected(self):
        for missing in PACKAGE_PAYLOAD:
            with self.subTest(missing=missing):
                entries = [(name, content) for name, content in PACKAGE_PAYLOAD.items() if name != missing]
                with self.assertRaisesRegex(ValueError, "Unexpected package contents"):
                    self.fingerprint_archive(entries)

    def test_unexpected_payload_is_rejected(self):
        entries = list(PACKAGE_PAYLOAD.items()) + [("untracked_dependency.so", b"dependency")]
        with self.assertRaisesRegex(ValueError, "Unexpected package contents"):
            self.fingerprint_archive(entries)


class ArtifactVerificationTests(unittest.TestCase):
    def test_identical_packages_and_matching_binary_fixture_provenance_pass(self):
        provenance = {"artifacts": {
            "target/release/test_kernels": copy.deepcopy(PACKAGE_FILES["test_runtime"]),
            "ref/fixtures.safetensors": copy.deepcopy(PACKAGE_FILES["fixtures.safetensors"]),
        }}
        runs = [make_run(job_id) for job_id in (69783, 69913, 69931)]
        self.assertIsNone(measure.verify_artifacts(runs, provenance))

    def test_each_payload_mismatch_rejects_cross_job_aggregation(self):
        for changed in PACKAGE_FILES:
            with self.subTest(changed=changed):
                runs = [make_run(69783), make_run(69913)]
                runs[1]["package_files"][changed]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "same immutable benchmark package"):
                    measure.verify_artifacts(runs)

    def test_wrong_recorded_binary_or_fixture_hash_is_rejected(self):
        aliases = {"target/release/test_kernels": "test_runtime",
                   "ref/fixtures.safetensors": "fixtures.safetensors"}
        for source, uploaded in aliases.items():
            with self.subTest(source=source):
                provenance = {"artifacts": {source: {"sha256": "0" * 64}}}
                with self.assertRaisesRegex(ValueError, f"Provenance does not match uploaded {uploaded}"):
                    measure.verify_artifacts([make_run(69783)], provenance)


class AggregationTests(unittest.TestCase):
    def test_duplicate_existing_job_ids_are_not_independent_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            # Use a fresh interpreter to verify the command's exit code. Block CLI
            # access inside the child, so a future regression cannot contact Arena.
            bootstrap = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('measure_child', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def no_arena(*args):
    raise AssertionError('Argument validation must finish before any Arena call')
module.cli = no_arena
sys.argv = ['measure.py'] + sys.argv[2:]
raise SystemExit(module.main())
"""
            result = subprocess.run(
                [sys.executable, "-c", bootstrap, str(MODULE_PATH),
                 "--jobs", "69783", "69783", "--label", "duplicate", "--output", str(output)],
                capture_output=True, text=True, timeout=10, check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse(output.exists())

    def test_reruns_aggregate_job_medians_without_pooling_seed_samples(self):
        # Median(job medians) is 4; pooling all 9 samples would yield 5.
        samples = ([1, 2, 100], [3, 4, 5], [1000, 2000, 3000])
        job_ids = (69783, 69913, 69931)
        runs = []
        for job_id, cycles in zip(job_ids, samples):
            middle = sorted(cycles)[1]
            line = f"    median cycles={middle} (of 3 runs: {list(cycles)})"
            kernels = measure.parse_log(FIXTURE.replace(QKV_CYCLES, line))
            runs.append(make_run(job_id, kernels))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = ["measure.py", "--jobs", "69783", "--reruns", "2",
                    "--label", "serial-reruns", "--output", str(output)]
            responses = [f"submitted job {job_id} (beef_initial_baseline) from job 69783\n"
                         for job_id in job_ids[1:]]
            with patch.object(measure.sys, "argv", argv), \
                 patch.object(measure, "collect", side_effect=copy.deepcopy(runs)) as collect, \
                 patch.object(measure, "cli", side_effect=responses) as cli, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(measure.main(), 0)
            result = json.loads(output.read_text())
            self.assertTrue(result["valid"])
            self.assertNotIn("pending_job", result)
            self.assertEqual(result["rerun_parent"], {"69913": 69783, "69931": 69783})
            self.assertEqual([c.args[0] for c in collect.call_args_list], list(job_ids))
            self.assertEqual(cli.call_args_list, [call("rerun", 69783), call("rerun", 69783)])
            aggregate = result["aggregate"]["sliding_project_qkv"]
            self.assertEqual(aggregate["job_medians"], [2, 4, 2000])
            self.assertEqual(aggregate["median_cycles"], 4)
            self.assertNotIn("indicative_score", result)

    def test_mixed_packages_produce_no_aggregate_or_score(self):
        runs = [make_run(69783), make_run(69913)]
        runs[1]["package_files"]["test_runtime"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = ["measure.py", "--jobs", "69783", "69913", "--label", "mixed-artifacts",
                    "--output", str(output)]
            with patch.object(measure.sys, "argv", argv), \
                 patch.object(measure, "collect", side_effect=runs), \
                 patch.object(measure, "cli") as cli, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(measure.main(), 1)
                cli.assert_not_called()
            result = json.loads(output.read_text())
            self.assertFalse(result["valid"])
            self.assertIn("same immutable benchmark package", result["error"])
            self.assertEqual(len(result["runs"]), 2)
            self.assertNotIn("aggregate", result)
            self.assertNotIn("indicative_score", result)


if __name__ == "__main__":
    unittest.main()
