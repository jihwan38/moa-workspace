Actual323407: sampled numerical gate PASS only

All18 expected records passed replay. The two row orders have identical owner-associated metric records; the binary additionally asserts pair-bit equality. Bothsteps use only original q020 residual feedback, and DOWN receives its production GeGLU. These are supplied layer0 weights and synthetic512-row incoming states, not a model/task-quality evaluation.

| Step | Projection | Original A4W4 L2 | Rotated A4W4 L2 | Ratio | Rotation-only L2 |
|---:|---|---:|---:|---:|---:|
| 0 | up_proj | 0.324810 | 0.130683 | 0.402335 | 0.001564 |
| 0 | gate_proj | 0.531357 | 0.254753 | 0.479439 | 0.002653 |
| 0 | down_proj | 0.238084 | 0.129676 | 0.544666 | 0.001410 |
| 1 | up_proj | 0.261713 | 0.115583 | 0.441642 | 0.001327 |
| 1 | gate_proj | 0.433333 | 0.232804 | 0.537240 | 0.002763 |
| 1 | down_proj | 0.231466 | 0.126478 | 0.546423 | 0.001471 |

All12 projection/order/step records satisfy the preregistered ratio<=0.75 and rotation-only L2<=0.01. Activation-only and weight-only dot errors also improve in these samples; their errors are not additive. F64 ideal-rotation checks stay inside the per-pair absolute bounds. Finite normal-or-zero checks pass on the evaluated rows. The9 arm records and zero fractions are retained in actual-result.json.

Counts:6144 pairs per arm,9 arms,47,185,920 inner terms per arm. There are1536 distinct sampled weight rows reused across orders/steps. Restored replay compares290,996,224 bytes; pre-replay immutable scratch/input reads compare306,724,864 bytes. Full retained production tensors are fingerprinted before/after; that is separately labeled from exact byte comparisons.

The frozen raw field permutation_bytes=27,795,456 has mixed units: existing compare_rows returns27,525,120 BF16 values, while the pair records contribute270,336 bytes. All comparisons ran. The source-derived total compared extent is55,320,576 bytes. Raw source/logs are preserved unchanged; the analyzer records both the raw value and split units.

Provenance:429 submitted/build/current inputs agree, all8 images equal the prior validated images and fresh compiler output, and the q020 control image is exact. Captured/submitted ELF c591b27a69a73e3b6d9a58e596d6ea2fc6612200fb5e8290cbc889f4965cf5fd,4,859,280bytes. Job323407 SUCCEEDED7.689726249s. CPU sampled-weight preparation981.088ms includes decoding, rotation and quantization; neither number is a kernel/model speed or all48 startup estimate.

All reported dots are pre-global FP64. F64 reconstruction of integer codes and stored scales does not reproduce the device localF32 scaling/reduction or subsequent BF16/global/GeGLU cuts. Rotation has not been implemented on the device. Whole-FFN quality, task quality, runtime cost and promotion remain unproven; official q020/final unchanged.

Replay: python3 experiments/int4-hadamard64-diagnostic/analyze_actual.py --verify-local-artifacts

Independent batch_kernels actual audit PASS:18 records,429 inputs,8 images, both order records, six unique numerical gates and byte-count correction independently checked.
