# Independent E142 emitted-contract review

`independent-audit.py e142-full30-v1 --check` passes for source6ebb74ca… and the exact compiled whole image. The checker, JSON and this note are frozen. All trial checks ran against a temporary output before publishing the successful candidate gate.

Actual graph O58/O118 are the two independent full30 direct FP4 lookup/native producers. Each retains F32 block16 partials, applies input scale, and commits7200 F32 values/slice as1800×16B packets into `[Row30,H/16=240]`:28,800B/slice. Ten separate Main consumers read matching six-row/1440-value windows, apply block scale, reduce H and write three8B F32 row pairs per tile. The five windows per projection start at0/5760/11520/17280/23040 bytes and end exactly28,800. Partial materialization is28.125MiB of logical writes and the same logical read volume across both projections.

The checked path is native→F32 input-scale multiplication→F32 store/reload→F32 block-scale multiplication→H reduction→canonical same-cluster F32 gather→BF16. Source outside `project_rows_with_scale_contraction` remains byte-identical to frozen E141. The separate block multiply uses Mul0 where E141's fused stream used Mul1; equal semantic ordering does not substitute for the finite numerical oracle on the new image.

All six public FP4/scale matrices have exact HBM coverage; compact up/gate cluster/slice ownership is unchanged. The actual graph has two up/gate lookup tables and one inherited down table. Thirty-seven whole-kernel VRF allocations fit capacity without overlapping live physical intervals, and all recorded TRF/VRF consumers fit their lifetimes. Hidden hardware arbitration and runtime correctness remain outside this static gate.

The repaired up-term writes are `[17664,18144)`, `[18144,18624)`, `[18624,19104)` per slice, each480B. The actual ring32 bitmap is `[8704,8736)`, with its embedded route mask and load/consumer dependency checked; none of the three writes reaches it. Both source and emitted partial/copy contracts are checked instead of inheriting the old E103/E135 failures or treating compilation as accuracy evidence.

Whole NPU and unchanged production up/gate oracle results are required separately. No speedup follows from fewer lookup tasks or the static150,905-cycle estimate.
