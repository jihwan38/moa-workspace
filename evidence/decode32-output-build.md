# Exact32 output build provenance

The source plan and measured6.8ms stage budget are in
`decode32-stage-profile-actual.md` (Arena319388). The independent source review
by batch_kernels found no row/cluster/store mapping or original BF16 tree issue.
Hardware correctness and inclusive speed are pending.

- a009 /43ef914: host compile failed because SDK f8e4m3 has no public `to_bits`.
  The synthetic weight lookup now serializes a64-element HostTensor through its
  supported raw byte API; no device arithmetic changed.
- a010 /4bdbc62: both new partial-dot kernels compiled. Final kernels rejected
  a no-op16-row tile whose zero index came from a one-iteration loop variable.
- a011 /b45a6f2: replaced that variable with literal0, retaining one identical
  BF16 addition pass. All device kernels and the complete probe compile;32.78s.

The original a007 model and a008 stage executables remain separately named and
unchanged. The new test executable is copied into this tree and all source hashes
match the a011 snapshot. Its artifact manifest records the exact hash and size.
Use `FURIOSA_OPT_PROFILE=trace` for its explicitly enabled completed-call traces;
correctness and ABBA periods run with the collector disabled.

Scoped host source662e309 also passed a012 plain server/model-probe check (11.29s)
and a014 unfiltered library suite:88 passed,0 failed,0 ignored,0 filtered (0.23s
execution). This is the older isolated a052254-derived tree's complete suite;
root's newer integrated suite has additional unrelated tests. The model ELF was
built before the host-test run and its source hash manifest remains unchanged.
