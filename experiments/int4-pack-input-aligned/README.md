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
