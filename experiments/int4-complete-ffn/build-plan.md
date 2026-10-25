# Proposed fresh device build — not executed

First confirm exclusive idle `/round2/ffn-k256/target`, unchanged original
Cargo/lock, clean source, and a nonexistent/empty new public kernel output path.
From `/home/jihwan38/moa-worktrees/round2`, after root authorization only:

```bash
python3 tools/record_run.py \
  --name a003-int4-complete-ffn-device-build \
  --worktree int4-complete-ffn-probe \
  --purpose 'Complete layer0 A4 FFN; current production whole-graph control; fresh kernels and captured standalone ELF; no serving change' \
  --fresh-package \
  --capture-artifact target/release/test_int4_complete_ffn \
  -- docker exec --user moa --workdir /round2/int4-complete-ffn-probe \
  --env CARGO_TARGET_DIR=/round2/ffn-k256/target \
  --env CARGO_INCREMENTAL=0 --env CARGO_BUILD_JOBS=4 \
  --env FURIOSA_OPT_OUT_DIR=/round2/int4-complete-ffn-probe/experiments/int4-complete-ffn/kernel-out-a003 \
  moa2026-round2 cargo furiosa-opt build --release --locked \
  --bin test_int4_complete_ffn
```

Do not use a shared stale device-output directory. Preserve build failure logs;
do not repair by changing arithmetic or gates. Capture ELF before any other
shared-target use, verify all recorded input hashes and frozen dependencies.
Read the ELF registry without rewriting it and require:

1. Both typed I4 projections and both integer-coded E4 reference images are
   byte-identical to bound artifact
   `1cb160656f2cc0c1006184f6d7fafde0d61a3f7a49e2659cd3cee72eb5f69e5b`.
2. `quantize_pack_input_aligned` and `quantize_pack_hidden` images match the
   actual-passed fused-pack artifact
   `be18ad3ee9fcc13c04e19df4776e99ac95599ff2a12167538aba7444fe430c68`.
3. The current production whole-graph control and original P128 norm/tail
   images match existing captured current-control images where available.
   New BF16 bridge/reference wrapper images are explicitly new and need the
   runtime boundary comparisons; resource success does not prove correctness.

After capture and parity, report exact ELF/source hashes and no-argument CLI.
Root alone decides/submits any <=70s Arena run. The diagnostic prints both the
strict correctness result and a separately reported10% complete-chain gate.
Neither device build nor a speed pass alone authorizes model adoption or releases
the independent DOWN resource HOLD.
