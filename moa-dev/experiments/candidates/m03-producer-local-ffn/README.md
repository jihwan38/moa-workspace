**2026-09-27 후속: 동일 M03 source를 E122 / `baseline-e122`로 채택했다.**
[6allocation·추가입력·상쇄 검증](validation/README.md), [채택 source 연결](source-build-audit.json), [현재 구성표](../../../../baseline/e122.json). 아래 최초 screen의 종료 판단은 당시 기록이다.

# M03 — producer-local one-stage FFN

후속 [E122 제한적 채택 검증](validation/README.md)은6개 allocation 성능, 추가6입력과7개 projection case를 확인했다. 아래 E117 단회 screen은 최초 실험의 역사 기록이다.

Branch `opt/m03-producer-local-ffn`, integrated parent `6ea3664`.
Source origin E74 `9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7`, with explicit
C14-B v4 and E104 method composition in `native-local-v1/recipe.json`.
Performance comparison is independently pinned to E104. This is a **C14 family
recombination / whole-FFN ownership redesign**, not another independent family.

## Execution path and changed boundaries

The unchanged public entry is `ops::decoder_feedforward`: pre-RMS calls
`mlp::feedforward_sharded`, then the existing post-RMS/residual/layer-scale code.
Only `src/device/shared/mlp.rs` is replaced. Other source inputs and the QKV/O
images were checked equal to E104, despite selecting E74 as the source origin.

```text
runtime BF16 activation -> actual E5M2 three-term preparation and replication
 -> cluster0 produces L[0:7680], cluster1 produces L[7680:15360] native up/gate
 -> compact four adjacent 30-row F32 producers within each cluster
 -> BF16 120-row packets -> local scaled GeGLU
 -> each cluster compacts only its own L7680 and distributes four K1920 columns
 -> local BF16 down dots, block16 F32 partials, block scales, four-column sums
 -> gather two F32 H3840 cluster partials -> sum -> first down BF16 cast
 -> global scale -> unchanged post-RMS/residual/layer scale -> HBM
```

`project_up_and_gate` uses the validated FP4-to-E5M2 **DM materialization**
boundary. The unresolved direct-FP4/native diagnostic is not part of M03.
`prepare_ffn_fp8_split` and `project_rows_with_scale_contraction` retain E104's
actual arithmetic. Prepared inputs are reused, but no input/weight conversion is
performed for free on the host. The candidate includes all runtime preparation.

`geglu` uses C14's local-cluster path and preserves the first BF16 projection
rounding, F32 global scale followed by BF16, the erf expression, and final BF16.
The proposal originally said 60-row local packets. This implementation compacts
**120 rows directly** to match the existing eight-lane F32 packet width without a
60-row tail or a separate 60-to-120 Switch step. This changes packetization, not
L-half ownership. It is not counted as a distinct architecture or sweep.

`project_down_tile` now has one L stage and no offset argument. Its name denotes
each cluster's L half; it does not execute two sequential stages. Unlike E104,
both clusters compute all H rows using disjoint L halves. `finish_down` sums only
the cluster axis. It has no C14-B second-stage VRF read/addition.

| Tensor / operation | Owner and storage | Lifetime / consumer |
|---|---|---|
| Input preparation | E104 three E5M2 terms in DM, then TRF; scalar F32 restoration in VRF | Native up/gate; actual preparation is included |
| Up/gate packed and expanded weights | `l=7680*c+30*s+t`; each slice FP4 57,600B, E5M2 115,200B | Main lookup then native DPE; complete FP8 boundary retained |
| Up/gate block partial / scales | F32 DM30×240; scale VRF6×240×4=5760B | Five scale/reduce streams |
| Local GeGLU packet | `l=7680*c+120*r+t`, physical slice `4*r`; F32 compact then BF16 | Main/Sub VE with global scales; no cross-cluster L gather |
| Down input | Cluster-local BF16 L7680 -> ring32 leaders -> four K1920 columns -> TRF3840B/slice | Down BF16 DPE |
| Down packed / expanded / partial | `s=4*r+k`, `h=60*r+t`, `l=7680*c+1920*k+u`; local60×1920 | FP4 DM57,600B -> E4M3 DM115,200B -> F32 partial28,800B |
| Down block scale | VRF12×120×4=5760B | Five12-row block scale/reduce passes, then four-column F32 sum |
| Final cluster partials | 2×H3840 F32, 30,720 logical bytes; DMA to `Dummy2,H%480` epilogue | Cluster sum before first BF16 rounding, then existing global scale |

The logical layouts describe values, not measured memory traffic. Allocated
lifetimes and emitted transfers are inspected from the compiler schedule; none
of its intervals is a hardware stall measurement.

## Cost hypothesis and first gate

Remove E104's full-L up/gate gather and cross-cluster activation redistribution.
Down ring leaders fall from128 to64 overall: logical leader payload491,520 to
245,760B. Fully replicated down activation remains1,966,080B. Add local compaction
and the F32 cluster-partial gather/sum. Weight bytes and total FP8 materialization
remain unchanged. The whole graph still waits for up/gate before down, so this
prototype does not claim early-down pipelining.

Prior C14-B v4 was17.58% slower but used BF16 up/gate and two L stages; its result
is not the prediction for this coherent recombination. E104 native up/gate gains
are already in the control and cannot be counted again. M04 retained E104
ownership and lost13.67%; its repeated row-tile setup is removed from this design.

Compile the complete graph, then run the unchanged NPU benchmark with adjacent
E104 controls. Close regression or target-kernel improvement below5% without a
tile sweep. A promising result needs independent repeats and expanded numerical
validation; changed down sum order cannot inherit C14/E104 correctness.

## Completed screen — E117, modest positive result, not adopted

`m03-native-local-v1`: first source compile and host build pass. Exact compilation
and host recompilation produce identical FFN images. Static makespan154440
(E104157045); this is **not** an NPU performance result.

Arena104289, actual uploaded package verified:

| Process | FFN median cycles | Basic comparisons |
|---|---:|---:|
| E104 before | 335669 | 15/15 |
| M03 | 330837 | 15/15 |
| E104 after | 335107 | 15/15 |

M03 is **1.36% faster than the bracketing median335388** (1.44% versus before,
1.27% versus after). Its three FFN inputs /11520 output elements pass the
unchanged benchmark criteria. This is one allocation, not independent repeats
or expanded numerical validation. QKV/O timing variation is not an optimization.

The complete new ownership path works, but this gain is below the5% experiment
budget cutoff. Close without a tile sweep; retain the source as a **modest positive
composition option**, not an adopted replacement. E104 remains live and no
official submission was made. The result does not prove the memory mapping is
globally optimal or identify how the leading teams implement their kernels.

Emitted instructions293→287, DMA35→37; the compiler's early pre-RMS-to-amax
gap remains23250 and peak FP8 allocation remains56.25MiB. All three full matrices
still expand to FP8 (168.75MiB logical stores total). Reduced local communication
does not remove those costs. Static intervals are not added as NPU savings.

`analyze.py` verifies row/segment coverage and summarizes the emitted graph.
`verify.py` checks component sources, unchanged quantization/native arithmetic,
GeGLU arithmetic under explicit layout renaming, frozen compile logs, image
embedding, unchanged QKV/O, actual uploaded archive and parsed raw Arena log.
See `decision.json`, `verification.json` and `docs/experiments/e117-m03-producer-local-ffn.json`.
