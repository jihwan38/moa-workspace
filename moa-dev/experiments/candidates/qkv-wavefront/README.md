# E130 — QKV Main F32 producer / Sub complete consumer

**Measured regression; not adopted.** Arena109339 passed all45 comparisons, but QKV119,470 cycles versus E128104,736 before/104,976 after is **+13.9372%** against the bracket104,856. The frozen source, compile failure/repair, actual uploaded archive and original logs are retained. E128 remains adopted; no further parameter tuning or official submission.

Source: E128 `baseline-e128`, work branch `opt/qkv-wavefront` from `ff7b9df`. The public kernel, original host and fixture are unchanged. Candidate sources are isolated in `sub-consumer-v1` and `sub-consumer-v2`; the working runtime is unchanged.

The complete path is pre-RMS/BF16 replication → Q and shared K/V TRFs → independent FP8 weight ingress → inline lookup and full-H BF16 DPE on Main → compact F32 DM → Sub channel scale → first BF16 → Sub row Switch → original head gathers → Sub Q/K/V RMS and Q/K RoPE → original public stores. Main/Sub share VE; moving scale and BF16 conversion away from the Main contraction is necessary for resource separation. Original source already calls all projections before head normalization, so this is not a call-site reorder.

The three valid raw F32 projections add 32KiB write and 32KiB live-scalar read. V2 commits two F32 rows in each 8B packet. Sub reads an 8B packet per row: packet sizes sum to **64KiB**, including ignored padding. The final packet reads four padding bytes beyond each logical per-slice tensor; the frozen graph/schedule gate checks the live scalar mapping and that this tail does not intersect another live SRAM payload. These logical/descriptor counts are not measured bus traffic. Three scale-consumer tasks and Sub contention are added. Weight bytes, two activation TRFs, head gathers and head arithmetic remain.

V1 failed because its four-row F32 transpose exceeded native `max_in_rows=2`; this was a source packet configuration error. V2 uses supported two-row 8B commits with the same arithmetic and 32KiB raw output. Both source snapshots and compiler logs are preserved. V2 exact compilation succeeds: image `6cc87899c58ad2ed73f9c1c180ae30530725b83aa75ae3adb26b0976710fd675`, static 57,034 versus E128 QKV 51,293 (**+11.1926%**, an estimate).

[`verify_graph.py`](verify_graph.py) and its [frozen result](graph-verification-qkv-wavefront-sub-v2.json) bind source/build/image/graph. They check three pure F32 Main contractions without VE or BF16 cast, exact-once full weight HBM source ranges, matched Q/K/V channel scales on Sub before first BF16, original O/public ops and unchanged head arithmetic. Static Main DPE/Sub VE overlap exists but is only **268 cycles**. Legacy table preparation, TRF/VRF loads, switches and all head work contend on Sub; the complete static schedule is slower. This justifies one bounded original-host NPU screen to measure the actual complete path, rather than interpreting static cycles as NPU evidence.

Build label: `qkv-wavefront-sub-v2`; screen label: `e130-qkv-wavefront-screen`. `package.py` verifies actual E128 source manifest, all three images, host embedding and unchanged fixture/official runner. `screen.py` executes E128 before → candidate → E128 after in one allocation with 45 expected comparisons. Only QKV changes; O/FFN timing fluctuations are controls, not gains. `results.py` retains the actual uploaded archive and stage mapping.

The [final decision](decision.json) separates adoption, tuning and major research. The screen was slower than both controls, so this frozen prototype is not adopted and additional parameter tuning stops. Extra inputs and a boundary oracle were not run. A positive result, even below 5%, would instead call for bounded adoption validation. Main/Sub resource separation is proven in the compiled graph, but the whole-path NPU result is slower; no trace was collected, so a particular hardware stall or Sub contention fraction is not established. Other QKV ownership/dataflow designs remain open when their complete costs differ. Existing V-only reorder evidence bounds an unchanged-readiness opportunity to 223–2107 cycles; this candidate does not claim all head-processing duration is removable.

Verification:

```sh
python3 moa-dev/experiments/candidates/qkv-wavefront/verify_graph.py qkv-wavefront-sub-v2 --check
python3 moa-dev/experiments/candidates/qkv-wavefront/package.py qkv-wavefront-sub-v2 --check
python3 moa-dev/experiments/candidates/qkv-wavefront/results.py e130-qkv-wavefront-screen 109339 --check
python3 moa-dev/experiment_index.py --check --check-live
```
