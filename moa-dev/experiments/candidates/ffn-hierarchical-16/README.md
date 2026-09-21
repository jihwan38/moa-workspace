# FFN hierarchical input replication

The patch changes only the FFN entry-point body, on top of QKV/O hierarchical broadcasting and the unrolled four-row FFN. It copies the normalized BF16 vector to 16 groups, then broadcasts within each 16-slice group. The physical 256 copies are identical before reshaping to Replicated.

Exact isolated compile succeeded with makespan 382949 versus 392312 for the previous unrolled FFN. Combined hardware job70074 passed all output checks, but FFN took828972 cycles versus820226 in preceding job70070 (1.07% slower). This single comparison does not support adopting the FFN broadcast change for speed; the simultaneous QKV change is independently measured in a different kernel. No changed arithmetic, scales, reduction, dtype or kernel signatures. Raw schedule is in the ignored artifacts/ffn-hierarchical-16 directory.
