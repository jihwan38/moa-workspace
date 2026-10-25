# Optional parallel MSE encoder: local CPU gate

Functional source `2d1cba71`; standalone Arena entry `40864041`; base q019
`13174d8b`. No startup caller, model selection, D2H, upload, kernel, or dependency
changed. `source-equivalence.json` proves that removing the added optional API
restores the entire original serial module byte-for-byte, also identical to
`8e7ce99e` (SHA256 `8aad3c106adeb77c69b2fa6343e775b01ef67b2c59d4b77b547529e6a5d5e014`).

Six pure-std tests pass, including the original tie/sign/zero checks. Full
15,360x3,840 and 3,840x15,360 deterministic provided-shaped fixtures and true group
permutations compare **117,964,800 packed bytes and 245,760 scale values bitwise**.
Fixtures cover all254 finite E4 codes, negative zero, midpoint values, varied
group maxima and independent row groups. Uneven row-block counts, serial
fallback, shape/size/length errors and multiple failing shards preserve the
same earliest global NonFinite index. These are deterministic fixture tests,
not a claim that actual provided weight files were downloaded here.

Each worker calls the unchanged serial encoder on complete contiguous240-row
blocks. Scoped immutable input borrows end before return. Ordered joins select
errors and concatenate outputs in input order; no numeric reduction changes.
Independent e2e review passed. Worker cap is min(available_parallelism,4,blocks).

## Local completed CPU timing

Input is exactly **58,982,400 source bytes** per projection;
output29,491,200 packed bytes plus61,440 F32 scales. Allocation, thread creation,
lookup, histogram fitting, packing, ordered joins and final assembly are inside
each timer. Input generation, checksums and parity comparisons are outside.

| CPU affinity | Serial range, ms | Parallel range, ms | Mean serial / parallel |
|---|---:|---:|---:|
| 24 available CPUs, four workers | 77.635–82.389 | 23.533–36.969 | 2.719x |
| 0,1: two SMT threads of one core | 77.565–82.627 | 69.594–78.986 | 1.095x |
| 0,2: two different cores | 77.511–82.907 | 42.464–51.831 | 1.758x |

All ABBA values are retained in `local-results.json` and individual logs. The
first two-CPU test exposed SMT sibling placement, so the separate two-core
comparison was added; neither result is discarded. Docker has24 logical CPUs
and cpu.max=`max 100000`, not a two-CPU quota. No Arena CPU topology/quota is
inferred from these tests. A two-CPU limit can substantially shrink the benefit;
it must be measured in the actual evaluation environment.

Separate encoding-only processes report peak VmHWM89,512KiB serial versus
118,648KiB parallel, **+29,136KiB (28.45MiB)**. All shard outputs temporarily
coexist with the final output allocation. This is not zero-copy, and peak RSS
includes the immutable source, lookup tables, allocator and process runtime.
The standalone `gate` process additionally holds full correctness fixtures;
its204,540KiB local peak is therefore a different scope.

## Frozen actual-environment gate

`arena-manifest.json` specifies:

- ELF `/round2/f4-mse-parallel/experiments/f4-mse-parallel/artifacts/probe-arena`
- SHA256 `89ec00db8f0814b3bd61c6f0b4551a6a586c97f63638c79a59a1a3efe2b13027`
- 4,452,184 bytes; arguments `["gate"]`; timeout70 seconds; no model/NPU needed.
- Full parity first, then8 completed serial/parallel calls. Prints actual
  available_parallelism, selected worker cap, cgroup/cpuset and process affinity.
- x86_64 ELF, interpreter `/lib64/ld-linux-x86-64.so.2`, GLIBC requirements up to
  2.34, dependencies only libc, libgcc_s and the dynamic loader. Readelf records
  are retained. Runtime compatibility still requires the actual launch.

Only standalone `rustc -O` builds were used, avoiding the SDK/library debug
rebuild. Root owns the actual-environment launch and any later caller change.
Current result: **local CPU mechanism passes; caller adoption awaits Arena CPU
evidence.** There is no startup-duration, task-quality or serving-score claim.
