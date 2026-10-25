"""Offline safety tests: no docker, network, scheduler, or real subprocesses."""
from contextlib import redirect_stdout
from datetime import timedelta
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("submit_q020_once", Path(__file__).with_name("submit_q020_once.py"))
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
ACCEPTED = b"Submitted as: BEEF\nSubmission:   00549d43\n"
AUTH = b"GitHub: example\nTeam: BEEF\nStage 2: advanced\n"


class Clock:
    def __init__(self):
        self.value = mod.EARLIEST
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += timedelta(seconds=seconds)


class FakeRunner:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.auth = AUTH
        self.quota = [b"Submissions in the last 24 hours: 3 of 4\n"]
        self.submit_response = ACCEPTED
        self.statuses = [b"Submission:   00549d43\nTeam: BEEF\nStatus: completed\n"]
        self.mount = root
        self.on_final_auth = None

    def pop(self, values):
        return values.pop(0) if len(values) > 1 else values[0]

    def __call__(self, args, timeout):
        assert isinstance(args, list)
        assert "--bin" not in args
        self.calls.append((args[:], timeout))
        if args[1] == "inspect":
            value = json.dumps([{"Source": str(self.mount), "Destination": "/round2"}]).encode()
        else:
            assert args[:6] == ["docker", "exec", "--workdir", mod.CONTAINER_CANDIDATE, mod.CONTAINER, "moa-submitter"]
            command = args[6:]
            if command == ["whoami"]:
                if self.on_final_auth and sum(c[0][6:] == ["whoami"] for c in self.calls) == 2:
                    self.on_final_auth()
                value = self.auth
            elif command == ["status"]:
                value = self.pop(self.quota)
            elif command[0] == "submit":
                assert command == ["submit", "--source", mod.CONTAINER_CANDIDATE, "--concurrency", "16"]
                marker = self.root / "runs" / mod.RUN_NAME / "submit-attempt.json"
                assert marker.is_file(), "durable marker MUST exist before submit"
                assert json.loads(marker.read_text())["argv"] == args
                value = self.submit_response
            elif command == ["status", "00549d43"]:
                value = self.pop(self.statuses)
            elif command == ["log", "00549d43"]:
                value = b"terminal log\n"
            else:
                raise AssertionError(f"unexpected command: {args}")
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, tuple):
            return subprocess.CompletedProcess(args, value[0], value[1], value[2])
        return subprocess.CompletedProcess(args, 0, value, b"")

    def submits(self):
        return [c for c in self.calls if c[0][6:7] == ["submit"]]


class SubmissionSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.candidate = self.root / "candidates" / mod.CANDIDATE
        (self.candidate / "src").mkdir(parents=True)
        hashes = {}
        for name in [f"src/file{i}.rs" for i in range(394)] + ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]:
            data = (name + "\n").encode()
            (self.candidate / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        elf = self.candidate / "target/release/server"
        elf.parent.mkdir(parents=True)
        elf.write_bytes(b"fake frozen ELF")
        self.elf_hash = hashlib.sha256(elf.read_bytes()).hexdigest()
        self.manifest = {"source_commit": mod.SOURCE_COMMIT, "binary_sha256": self.elf_hash,
                         "bytes": elf.stat().st_size, "source_hashes": hashes}
        self.manifest_path = self.candidate / "candidate.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.manifest_hash = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.addCleanup(patch.stopall)
        patch.object(mod, "MANIFEST_SHA", self.manifest_hash).start()
        patch.object(mod, "ELF_SHA", self.elf_hash).start()
        self.clock = Clock()
        self.runner = FakeRunner(self.root)
        self.directory = self.root / "runs" / mod.RUN_NAME

    def execute(self, submit=True):
        with redirect_stdout(io.StringIO()):
            return mod.execute(self.root, submit, self.runner, self.clock.now, self.clock.sleep)

    def rejected(self):
        with self.assertRaises((mod.Refused, OSError)):
            self.execute()
        self.assertEqual([], self.runner.submits())
        self.assertFalse((self.directory / "submit-attempt.json").exists())

    def repin_fixture_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        patch.object(mod, "MANIFEST_SHA", hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()).start()

    def assert_restart_blocked(self):
        calls = len(self.runner.calls)
        marker = (self.directory / "submit-attempt.json").read_bytes()
        with self.assertRaisesRegex(mod.Refused, "already attempted"):
            self.execute()
        self.assertEqual(calls, len(self.runner.calls))
        self.assertEqual(marker, (self.directory / "submit-attempt.json").read_bytes())

    def test_happy_source_only_c16_and_provenance(self):
        self.assertEqual(0, self.execute())
        self.assertEqual(1, len(self.runner.submits()))
        state = json.loads((self.directory / "status.json").read_text())
        self.assertEqual("terminal", state["phase"])
        self.assertEqual("00549d43", state["submission_id"])
        self.assertTrue(state["log_collected"])
        self.assertEqual(ACCEPTED, next(self.directory.glob("*-submit.stdout")).read_bytes())
        self.assertEqual(b"terminal log\n", next(self.directory.glob("*-terminal-log.stdout")).read_bytes())
        self.assert_restart_blocked()

    def test_check_is_readonly_even_early_with_full_quota(self):
        self.clock.value -= timedelta(days=1)
        self.runner.quota = [b"Submissions in the last 24 hours: 4 of 4\n"]
        self.assertEqual(0, self.execute(False))
        self.assertFalse((self.root / "runs").exists())
        self.assertEqual([], self.runner.submits())
        self.assertEqual([], self.clock.sleeps)

    def test_manifest_hash_mismatch(self):
        self.manifest_path.write_text("{}")
        self.rejected()
        self.assertEqual([], self.runner.calls)

    def test_source_hash_mismatch(self):
        (self.candidate / "src/file0.rs").write_text("changed")
        self.rejected()

    def test_config_hash_mismatch(self):
        (self.candidate / "Cargo.toml").write_text("changed")
        self.rejected()

    def test_elf_hash_mismatch(self):
        (self.candidate / "target/release/server").write_bytes(b"other ELF")
        self.rejected()

    def test_unexpected_source_file(self):
        (self.candidate / "src/unexpected.rs").write_text("unexpected")
        self.rejected()

    def test_missing_source_file(self):
        (self.candidate / "src/file0.rs").unlink()
        self.rejected()

    def test_source_symlink_refused(self):
        path = self.candidate / "src/file0.rs"
        path.unlink()
        path.symlink_to(self.candidate / "src/file1.rs")
        self.rejected()

    def test_exact_count_required(self):
        del self.manifest["source_hashes"]["src/file0.rs"]
        self.repin_fixture_manifest()
        self.rejected()

    def test_wrong_pinned_commit(self):
        self.manifest["source_commit"] = "0" * 40
        self.repin_fixture_manifest()
        self.rejected()

    def test_early_refused_without_commands(self):
        self.clock.value -= timedelta(microseconds=1)
        self.rejected()
        self.assertEqual([], self.runner.calls)

    def test_late_refused_without_commands(self):
        self.clock.value = mod.LATEST + timedelta(microseconds=1)
        self.rejected()
        self.assertEqual([], self.runner.calls)

    def test_window_rechecked_after_preflight(self):
        self.runner.on_final_auth = lambda: setattr(self.clock, "value", mod.LATEST + timedelta(seconds=1))
        self.rejected()

    def test_hash_rechecked_after_quota(self):
        original = self.runner.pop
        def pop(values):
            if values is self.runner.quota:
                (self.candidate / "src/file0.rs").write_text("late mutation")
            return original(values)
        self.runner.pop = pop
        self.rejected()

    def test_wrong_team(self):
        self.runner.auth = b"Team: OTHER\nStage 2: advanced\n"
        self.rejected()

    def test_not_advanced(self):
        self.runner.auth = b"Team: BEEF\nStage 2: not advanced\n"
        self.rejected()

    def test_ambiguous_identity(self):
        self.runner.auth = AUTH + b"Team: OTHER\n"
        self.rejected()

    def test_auth_failure_never_retries(self):
        self.runner.auth = OSError("offline")
        self.rejected()
        self.assertEqual([], self.clock.sleeps)

    def test_wrong_container_mount(self):
        self.runner.mount = self.root / "other"
        self.rejected()

    def test_bad_quota_fails_closed(self):
        self.runner.quota = [b"Submissions in the last 24 hours: 0 of 5\n"]
        self.rejected()
        self.assertEqual([], self.clock.sleeps)

    def test_duplicate_quota_fails_closed(self):
        self.runner.quota = [b"Submissions in the last 24 hours: 3 of 4\n" * 2]
        self.rejected()

    def test_full_quota_poll_is_bounded(self):
        self.runner.quota = [b"Submissions in the last 24 hours: 4 of 4\n"]
        self.rejected()
        self.assertEqual(600, sum(self.clock.sleeps))
        self.assertTrue(all(s == 20 for s in self.clock.sleeps))

    def test_quota_opens_then_one_attempt(self):
        self.runner.quota = [b"Submissions in the last 24 hours: 4 of 4\n",
                             b"Submissions in the last 24 hours: 3 of 4\n"]
        self.assertEqual(0, self.execute())
        self.assertEqual([20], self.clock.sleeps)
        self.assertEqual(1, len(self.runner.submits()))

    def test_transient_quota_failure_only_readonly_retried(self):
        self.runner.quota = [subprocess.TimeoutExpired(["status"], 45),
                             b"Submissions in the last 24 hours: 3 of 4\n"]
        self.assertEqual(0, self.execute())
        self.assertEqual([20], self.clock.sleeps)
        self.assertEqual(1, len(self.runner.submits()))

    def test_permanent_quota_transport_failure_is_bounded(self):
        self.runner.quota = [(1, b"", b"network unavailable")]
        self.rejected()
        self.assertEqual(600, sum(self.clock.sleeps))

    def test_late_catchup_does_not_wait_for_full_quota(self):
        self.clock.value = mod.EARLIEST + timedelta(hours=1)
        self.runner.quota = [b"Submissions in the last 24 hours: 4 of 4\n"]
        self.rejected()
        self.assertEqual([], self.clock.sleeps)

    def test_concurrent_lock_refuses(self):
        self.directory.mkdir(parents=True)
        with (self.directory / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.rejected()
        self.assertEqual([], self.runner.calls)

    def test_timeout_after_possible_acceptance_never_resubmits(self):
        self.runner.submit_response = subprocess.TimeoutExpired(["submit"], 180, output=b"uploading...\n", stderr=b"timeout")
        with self.assertRaises(mod.Refused):
            self.execute()
        self.assertEqual(1, len(self.runner.submits()))
        self.assertEqual(b"timeout", next(self.directory.glob("*-submit.stderr")).read_bytes())
        self.assert_restart_blocked()

    def test_timeout_with_id_only_monitors(self):
        self.runner.submit_response = subprocess.TimeoutExpired(["submit"], 180, output=ACCEPTED)
        self.assertEqual(0, self.execute())
        self.assertEqual(1, len(self.runner.submits()))
        state = json.loads((self.directory / "status.json").read_text())
        self.assertFalse(state["response_complete"])
        self.assert_restart_blocked()

    def test_network_exception_after_attempt_never_retries(self):
        self.runner.submit_response = OSError("connection reset")
        with self.assertRaises(mod.Refused):
            self.execute()
        self.assertEqual(1, len(self.runner.submits()))
        self.assert_restart_blocked()

    def test_ambiguous_success_never_retries(self):
        self.runner.submit_response = ACCEPTED + b"Submission:   12345678\n"
        with self.assertRaises(mod.Refused):
            self.execute()
        self.assertEqual(1, len(self.runner.submits()))
        self.assert_restart_blocked()

    def test_accepted_then_status_timeout_never_retries_submit(self):
        self.runner.statuses = [subprocess.TimeoutExpired(["status"], 45),
                               b"Submission: 00549d43\nTeam: BEEF\nStatus: completed\n"]
        self.assertEqual(0, self.execute())
        self.assertEqual([30], self.clock.sleeps)
        self.assertEqual(1, len(self.runner.submits()))
        self.assert_restart_blocked()

    def test_collection_stops_at_two_hours(self):
        self.runner.statuses = [b"Submission: 00549d43\nTeam: BEEF\nStatus: running\n"]
        self.assertEqual(2, self.execute())
        self.assertEqual(7200, sum(self.clock.sleeps))
        self.assertTrue(all(s == 30 for s in self.clock.sleeps))
        self.assertEqual(1, len(self.runner.submits()))
        self.assert_restart_blocked()

    def test_terminal_log_timeout_uses_remaining_collection_budget(self):
        original = self.runner.pop
        def pop(values):
            if values is self.runner.statuses:
                self.clock.value += timedelta(hours=2, seconds=-5)
            return original(values)
        self.runner.pop = pop
        self.assertEqual(0, self.execute())
        log = [c for c in self.runner.calls if c[0][6:7] == ["log"]]
        self.assertEqual(5, log[0][1])
        self.assertEqual(1, len(self.runner.submits()))

    def test_wrong_status_identity_does_not_collect_wrong_log(self):
        self.runner.statuses = [b"Submission: 12345678\nTeam: BEEF\nStatus: completed\n",
                               b"Submission: 00549d43\nTeam: BEEF\nStatus: completed\n"]
        self.assertEqual(0, self.execute())
        self.assertEqual([30], self.clock.sleeps)
        self.assertEqual(1, len([c for c in self.runner.calls if c[0][6:7] == ["log"]]))


if __name__ == "__main__":
    unittest.main()
