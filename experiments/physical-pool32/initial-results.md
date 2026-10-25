# Physical32 initial gates

Frozen functional/diagnostic source07718f07 built successfully in a002 (6m43s).
All393 recorded source/config hashes match. The ignored Cargo.lock changed only
by adding the already-resolved ring dependency to the root package; reversing
that one entry reproduces its recorded pre-build SHA256. The built lock and four
immutable ELF hashes are preserved in build-result.json. Ring is used only by
read-only diagnostic SHA256 reporting.

All131 package host tests passed with0 filtered. These include physical slot31,
rejection of32, an active512-token checkpoint through receiver cancellation and
session-vector compaction, queued33rd allocation, stale reservation rejection,
and the unchanged8192-context/256-page reservation rules.

Root-dispatched initial gates all passed:

- r1209/321226: every compact pool zero/read/write cell, all32 ring addresses,
  live-prefix commits1..16, immutable inputs and unchanged full page255.
- r1210/321227: group case0, two rounds, reset/order reversal, candidate-only B16
  interleaves and sentinel.20,185,088 paired logit values and3,774,873,600 cache
  bytes (one side of each exact comparison).
- r1211/321228:512-token histories, width17,48 active prompt layers with six R32
  interleaves.12,845,056 paired logit values;27,000,832 values have individual
  SHA256 reports, including both sides/unpaired decoder.3,170,893,824 cache bytes.

Available memory after warmed group/cache work is14,276,537,856 bytes; history17
reports14,276,297,216. This is consistent with the predeclared10GiB smaller KV
payload, but a matched physical64 program has not executed yet. All memory is
reported after retained joined weights and lazy group/R32 scratch are present.

No old64-versus32 model equality, HTTP throughput, task quality, or queued>32
latency result is inferred. The separately frozen physical64-control source
89c0097c uses the identical bounded32-owner probes with only the final Group
human label corrected. It is building after the compact ELFs were copied out of
the shared target. compare_program_digests.py requires a complete final PASS and
matches the complete ordered SHA256 event stream, including row labels/positions,
cache physical IDs and byte counts. Its result is explicitly a cross-program
SHA256 comparison, alongside each program's independent paired-byte gates.

Unit clarification: Model::compare_canonical_kv divides the underlying helper's
byte count by2, so Group/Sentinel bf16_values is correctly named. Aggregation here
uses KV_ALL_BYTES_EXACT compared_bytes directly. An earlier prose caveat missed
that Model wrapper; the numerical totals are unchanged.
Independent e2e source and probe ownership reviews found no concrete blocker.
