# BEEF Stage 2 experiments

The active goal is a verified final top-three result before the final submission
deadline, 2026-10-26 20:59 KST. No official valid score has been recorded in this
checkpoint. Kernel parity, public-case accuracy, model latency and official rank
are separate gates.

Work is isolated under `/home/jihwan38/moa-worktrees/round2`. `server` integrates
committed changes; `e154-port` preserves the measured scalar control; `prefill`,
`batch-kernels`, `attention` and `io` contain independently developed variants.
The original E154 repository and older experiments remain intact.

`runs/<id>` records each command, source hashes, diff, result and Arena logs.
Important probes also include the exact executable, binary hash and source
archive. `candidates/<id>` freezes a server executable together with its source,
lockfile and public test harness. Run HTTP tests from those immutable candidate
directories so a concurrent development build cannot replace the tested binary.
An ambiguous early upload is explicitly flagged in run r026; it is not evidence
for a particular candidate. Its replacement is r032 on frozen c001.

The scripts in `tools` are checkpoint copies. Set `MOA_ROUND2_ROOT` to the round2
directory when invoking these copies. `record_run.py` creates a new run directory
and never overwrites an existing one. `freeze_candidate.py` verifies that current
source still matches a successful build's hashes and commit before snapshotting.
`collect_job.py` saves actual Arena status and logs; inspect the raw log when a
probe uses plain text instead of `_RESULT` records.

The Arena account permits two active jobs. Only the root agent dispatches them
and makes quota-limited official submissions. Official source submissions allow
four attempts per rolling 24 hours. Public probe timings are not leaderboard
scores. Keep `src/api` compatible with the organizer's replacement, retain actual
model weights and all request-specific state isolation, and do not branch on
evaluation requests. All source and checkpoints belong in the private workspace
remote; final materials belong in `micro2026-moa/stage2-final-beef`.

The goal JSON and candidate manifests here are dated checkpoints. Live results
remain in the round2 `state`, `runs` and `candidates` directories.
