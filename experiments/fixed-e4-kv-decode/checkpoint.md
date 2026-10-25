# Decode storage-port checkpoint — provisional stop

Functional device port is e2d021f2, with whitespace/check evidence 58e6b754.
All original BF16 device modules and production host dispatch remain unchanged in
this worktree. E2e imported the device modules into a separate integration tree.
No Arena run, model ELF, default activation, or serving-performance claim exists
for this Rc32 port.

Completed fresh-cache resource compiles (recorded commands in completed-runs.json):

| Entry | Run | Estimated cycles | Unique DramStack allocations |
|---|---|---:|---|
| B16 complete sliding_16 | a002 | 204,984 | one 32-byte index bounce |
| R32 policy reader | a003 | 504,824 | two 32-byte index bounces |
| R64 HeadRows reader | a004 | 520,240 | one 32-byte index bounce |

All three use 67,108,864-byte E4 K/V pools, matching physical Rc32. The R32
conditional schedule includes both alternatives; its estimate is not a measured
wall time or a direct speed comparison. Resource summaries do not substitute for
actual byte/layout proof. Every resource invocation used a distinct documented
FURIOSA_OPT_OUT_DIR; the source kernels did not change between these runs.

Two new probes are preserved as unfinished work:

* `test_fixed_e4_decode_b16`: sixteen cases, live widths 1..16, sparse Rc32 slots,
  supplied O layers 0/6/46, two feedback/permutation steps, complete attention/O,
  every pool byte, rounded-append idempotence, finite exhaustive codec gate.
  The timed BF16 control uses the same CPU-rounded histories/current K/V as the
  E4 candidate. Raw BF16 history drift is a separate untimed diagnostic. There is
  no per-width performance acceptance threshold in this storage-parity probe.
* `test_fixed_e4_decode_readers`: nine cases covering capacity32, prefix512,
  mixed512, prefix768, HeadRows, and the old full-window reference; live17/31/32,
  masked physical R64 tails, and one 64-query read-only alias test over32 slots.
  The last case does not claim64 independent owners. Each candidate compares to
  its unchanged BF16 reader and the full-window BF16 reference, using independent
  CPU E4 decoding. Inputs and all pool bytes are checked for mutation.

CPU typecheck a005 passed for both probes in 11.27 seconds. After that check,
B16 timing was corrected from the raw-BF16 diagnostic arm to the CPU-rounded BF16
intended arm, and an unintended rustfmt-only change to the existing profile
support file was reverted. These final source bytes are **not rechecked** under
the provisional stop; wip-probe-sha256.json binds them. Neither probe has been
device-built or executed. Prefix512/mixed512/prefix768/old-full readers and the
remaining B16 wrapper widths still require fresh device compile/runtime gates.
No build/process remained active when this checkpoint was written.
