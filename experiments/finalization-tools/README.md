`submit_q020_once.py` submits the frozen q020 source directory at concurrency16 at most once from this host run directory. It does not schedule itself, modify the CLI/network configuration, build code, run Arena, upload a binary, or promote a repository.

Root-reviewed invocation (root owns scheduling and execution):

```sh
python3 /home/jihwan38/moa-worktrees/round2/fixed-e4-kv-storage/experiments/finalization-tools/submit_q020_once.py --root /home/jihwan38/moa-worktrees/round2 --check
python3 /home/jihwan38/moa-worktrees/round2/fixed-e4-kv-storage/experiments/finalization-tools/submit_q020_once.py --root /home/jihwan38/moa-worktrees/round2 --submit
```

`--check` validates local frozen inputs, Docker's `/round2` bind, account and quota using read-only CLI commands. It writes no run files and may run before the submission window or while quota is full. `--submit` refuses before2026-10-25T21:02:00Z (October26 06:02 KST) or after2026-10-26T11:30:00Z. It polls full quota or quota transport failures every20seconds until21:12UTC; successful malformed responses and authentication failures stop immediately. A later timer catch-up receives one quota read, without a renewed polling window.

Pinned candidate manifest/source/ELF identities are literal constants. Verification requires397 matching source/config hashes, an exact `src` inventory, original three config names and the frozen ELF bytes/size. The final upload is only `moa-submitter submit --source /round2/candidates/q020-b16-f4-mse-parallel --concurrency 16`. Source remains externally frozen between final hashing and the CLI's upload.

Before that single subprocess, the tool holds a nonblocking `flock` and fsyncs an exclusive `submit-attempt.json` under `runs/official-q020-20261026-c16`. **Do not remove this marker to retry.** A timeout, connection error, rejected/ambiguous response, or local crash after marker creation permanently consumes the local attempt. The remote outcome may be unknown; inspect the official service read-only. This guarantee assumes one shared host root/marker directory and does not coordinate independent manual submissions or other machines.

An unambiguous BEEF/8-hex submission ID enables read-only status collection every30seconds for up to2hours, even if the original subprocess timed out after printing the ID. Terminal log collection stays within that budget. All command argv, timestamps, raw stdout/stderr and SHA hashes are stored alongside `status.json`; no secrets are read by this tool. Exit0 with `phase=terminal` means a terminal status and log were collected, **not** that evaluation passed or the score should be adopted. `remote_status` and the official log determine that outcome. Expired or interrupted collection never permits another submit; follow up manually with read-only status/log commands.

Offline verification:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/finalization-tools -p 'test_submit_q020_once.py' -v
```

The tests replace the runner and clock, create temporary397-file candidates, and exercise hash/inventory/auth/time/quota guards, bounded read-only retries, concurrent locking, marker-before-submit ordering, process restart, ambiguous responses, accepted-response timeouts and bounded collection. No test launches Docker or a real submission. Local real-candidate hash verification also passed; no live `--check`, submission or timer installation was performed by this task.
