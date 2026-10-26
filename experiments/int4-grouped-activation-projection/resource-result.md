# Grouped activation projection resource checkpoint

Frozen functional source: `71cdf450`; independent e2e source review PASS.
Host check a001 passed in10.88s without source repair. Original Cargo files
and all captured source hashes remain unchanged. No actual device run occurred.

| Projection | Old endpoint | Grouped endpoint | Reduction |
| --- | ---: | ---: | ---: |
| down | 245,684 | 230,595 | 6.142% |
| upgate | 356,610 | 327,665 | 8.117% |

These are complete compiler timeline endpoints, including the retained initial
offsets, not loop-local LIR or actual microseconds. Both schedules have no
DramStack. The two resource compilations use separate initially empty OUT_DIRs.

Every dot program in the schedule preserves weight Mul0, activation Mul1,
inter-slice F32 Add, then direct BF16 cast. Dot TU duration is unchanged:
UP/GATE4×57925 and DOWN2×58033 cycles. The4/2 post-reduction scaling
programs (each7961cycles) disappear. DOWN retains2×7961cycles for its
separate BF16/global/BF16 pass. Sum of removed TU spans47766 and sum of
separate endpoint reductions44034 are different quantities; neither is a
measured whole-FFN saving.

The old four UP and two DOWN F32 reduced-column SRAM allocations
(62914560B padded extent per instance at different lifetimes) disappear.
The BF16 results, transpose/store and original packed inputs/weights/outputs
remain. Scale input owners grow2048→8192B (H) and2048→32768B (L).
Live scale delivery grows262144→1048576B UP and32768→524288B DOWN;
padded per-chunk scale VRF allocation remains262144B.

The historical complete-FFN speed failure and old DOWN resource HOLD are
preserved. The new DOWN endpoint is below240000, but this changes arithmetic
and does not retroactively pass the old branch or any full-path gate. The
grouped quantizer, independent dense-E4 device reference, actual all-output
correctness, full-chain >=10% speed and model/task quality are still untested.

Replay: `python3 experiments/int4-grouped-activation-projection/analyze_resources.py --verify`.
The JSON binds all three run records/logs/input hashes, old/new schedules,
operation counts, retained/removed storage, and resource-generated kernel images.
Reference source is host-typechecked; no reference device compile was included
in this resource checkpoint.
