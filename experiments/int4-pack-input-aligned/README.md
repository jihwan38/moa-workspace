The single H3840 mapping change passed the fixed resource gate: 19,994 cycles
versus 41,668 allowed. The original H52,085>50,000 and four-row staging
H54,722>41,668 HOLD results remain unchanged. This is a resource result, with no
standalone executable or device execution at this checkpoint.

Source15a335b6 calls the unchanged pack helper, then performs real DM movement
to `[H/960,P512%256/16,1#4]` with local `[P512%16,H%960/8]`. The64 live leaders
cover every one of the245,760 packed words once; all256 physical slices remain
in the mapping. Both root and e2e_evidence independently reviewed this mapping.

Compared with the preserved four-row H schedule, the added DM movement remains
3,276 cycles and the final store changes35,968→1,240. The compiler allocates
3,932,160 bytes for the physically padded staging tensor, versus983,040 live
bytes; its983,040-byte packed source remains live during the copy. Peak live
SRAM address-union occupancy is8,847,360 bytes and there is no DramStack.
The public `dma_tails` contract explains the changed contiguous transfer tail;
the internal TDMA cost heuristic is unavailable. These results support the
specific mapping mechanism without proving a hardware alignment threshold.

Replay without compiling or executing a device:

```sh
python3 experiments/int4-pack-input-aligned/analyze_resource.py
```

The script verifies recorded build inputs, original dependencies and unchanged
pack/hidden/adapter/probe files, preserves all three control schedules, and emits
the full source/resource-image hashes. The current resource task-cycle metric
is the maximum LIR lifetime endpoint. The historical original-pack report used
the schedule-view endpoint: original H is51,982 LIR /52,085 timeline; four-row
H is54,722 /54,825 and this candidate is19,994 /20,097. The candidate passes
the unchanged41,668 cap under either metric. All prior HOLDs remain intact.

The subsequent root-authorized standalone diagnostic is frozen at source
`58dadd8eb1ff4e30d7075bba73b0c961ef2e9be0`. Host check passed in10.44 seconds;
the fresh six-image build passed in34.81 seconds. Its captured executable is
`/round2/runs/a004-int4-pack-input-aligned-build/built/target/release/test_int4_pack_input_aligned`,
SHA256 `9900b5af49813e0e3a8ad167ccea3f8d9f3ba540ce78153999a378ff4ef23715`.
All405 source/config inputs match. All six embedded kernels match the new empty
output directory, and the original five kernels match root's frozen hidden-only
A/B executable exactly. The sixth aligned-H image also matches its resource
compile image. No captured ELF was modified during inspection.

`actual-plan.json` fixes a20-percent per-case H gain: worst candidate sample
must be at most80percent of the best original sample, separately for cases0/4.
The probe completes all49,152,000-code /24,576,000-packed-byte checks and six
invalid bindings before asserting that speed criterion. L keeps the existing
staged/original comparison; the failed four-row H candidate is never executed.
Both correctness and performance still await root's single bounded device run.

Replay the artifact/source checks without compilation or device execution:

```sh
python3 experiments/int4-pack-input-aligned/audit_build.py
```
