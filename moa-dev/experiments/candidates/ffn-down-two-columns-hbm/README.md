# Down 2-column 입력을 HBM을 거쳐 분배

상태: 독립 FFN 컴파일 성공, 정적 makespan **172233**. 2-column DM 직접 분배 190591보다 9.63% 낮지만, safe-decode 8-column 기준 171429보다 **804 cycles(0.47%) 높아 정적 우위는 없다**. NPU 정확성·성능은 미검증이다. 이 작은 차이만으로 hardware 실행 순위를 확정하지 않는다.

변경은 `project_down` 입력에 아래 한 번의 BF16 복사를 추가하는 것이다.

```rust
let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut device.tdma);
let x: DmTensor<bf16, Chip, DownCluster, DownRowsByColumns, m![L % 7680]> =
    x.to_dm(&mut device.tdma);
```

기존 GeGLU의 분산 `Cluster/UpGateRowsPaired/L%120`을 canonical L15360 HBM vector에 모은다. 그 HBM tensor에서 두 cluster × 120 valid row groups × 2 column segments로 복제한다. 임시 HBM payload는 30720 bytes다. QKV에서 사용했던 공개 `to_hbm` 경로와 같다.

추가 산술이나 BF16 cast가 없고 up/gate, GeGLU, safe FP4→FP8 DM→BF16 LUT DM, down blockdot/scale/reduction 코드는 이전 2-column 후보와 동일하다. `validate_candidate.py`는 이 두 줄을 제거하면 이전 source와 byte-identical함을 확인한다. 다만 [2-column 원후보](../ffn-down-two-columns/README.md)의 8×120→2×480 F32 합산 그룹 변경은 그대로 남아 있으므로 별도 NPU 정확성 검증이 필요하다.

| 항목 | 8-column 기준 | 2-column DM 직접 분배 | 2-column HBM relay |
| --- | ---: | ---: | ---: |
| makespan | **171429** | 190591 | 172233 |
| instructions | 323 | 313 | 312 |
| DMA union | 135461 | 157920 | **131187** |
| Main union | 132375 | 134094 | 134094 |
| Vector union | 41849 | 41768 | 41768 |
| Sub union | 22059 | 23296 | 23296 |
| activation 분배 DMA 합 | 9392 | 39136 | 12403 |

새 activation 경로는 HBM write 2219 + HBM read/replicate 10184 = **12403** cycles다. 직접 DM 분배의 39136보다 26733 줄었고, 다른 Main/Sub/Vector 작업량은 동일하다. 그러나 8-column 기준의 입력 분배보다 여전히 비싸며, 2-column의 decode 작업 증가까지 포함하면 전체 makespan은 약간 높다. HBM 임시 tensor와 memory lifetime도 바뀌므로 DMA 합만으로 NPU 개선을 주장하지 않는다.

`mlp.dm-relay.rs`는 이전 2-column snapshot, `mlp.rs`와 `change.patch`가 새 후보다. `static-comparison.json`은 두 기준과 새 schedule, source/artifact SHA256을 기록한다.

Scratch `/tmp/moa-ffn-down-two-columns-hbm`, 별도 target `/tmp/moa-ffn-down-two-columns-hbm/target`에서 FFN entrypoint만 컴파일했다. [Schedule](../../artifacts/ffn-down-two-columns-hbm/decoder_feedforward.json), [컴파일 로그](../../artifacts/ffn-down-two-columns-hbm/compile.log). Live source, 공유 target, NPU, Git에는 변경하지 않았다. 이 bounded 비교에서 추가 분할이나 계산법은 실험하지 않았다.
