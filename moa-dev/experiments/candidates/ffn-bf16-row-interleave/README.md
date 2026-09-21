# BF16 up/gate의 8-way row interleave

2026-09-21, SDK 0.8.1. 검증된 E15 BF16 FFN core에서 up/gate weight와 local scale의 row 분배만 바꾼 후보다. **Exact compile 통과, NPU 미실행.** FP4→FP8→BF16 decode, 16원소 F32 block dot, local scale, F32 block sum, BF16 출력 경계를 유지한다. Native FP8 activation quantization은 포함하지 않는다.

정적 makespan은 **185673 cycles**, 같은 E15 core의 171429보다 8.31% 높다. E16 tail은 두 쪽 모두 사용하지 않았다. Main 132375→133413, Vector 41849→41849, DMA 135461→153547이다. 실제 HBM bank 효율과 NPU 성능은 이 정적 값으로 판정할 수 없다.

## Weight와 scale이 이동하는 위치

한 cluster의 원래 row를 `r=256*g+8*u+s`로 쓴다. `0≤g<30`, `0≤u<32`, `0≤s<8`이다. 새 weight/scale DM은 slice `8*g+s`, local row `u`를 사용한다.

- Slice: `[L/256%30#32, L%8]` — cluster당 240 active slices, 16 padding slices.
- Weight Element: `[L/8%32,H]`.
- Scale Element: `[L/8%32,H/16]`.
- Row tile: 기존과 같이 8개 local rows. H 방향의 연산 순서와 16원소 block 경계는 그대로다.

기존 분배는 32개 연속 row를 같은 slice에 넣는다. 새 분배는 연속 8개 HBM row를 서로 다른 8개 slice에 넣는다. FP4 packed weight 한 row는 1920B, local scale 한 row는 240B이므로, FP8 QKV weight 3840B row에서 얻은 실측 결과가 그대로 이전된다고 가정하지 않는다.

## BF16 출력은 Switch 이후 canonical 순서로 복구

처음에는 새 분배에서 기존 `to_dm` gather만 호출했지만, 2B 간격의 BF16 scatter가 TDMA의 8B alignment 규칙에 걸렸다. 실패 소스와 로그는 `mlp.direct-dma-failed.rs`, `compile-direct-dma-failed.log`에 남겼다.

현재 `restore_interleaved_rows`는 `InterTranspose { slice1:8, slice0:1, time0:1 }`로 low slice 좌표 `s`와 local row의 하위 3bits `u%8`을 교환한다. 출력은 다음과 같다.

- Slice: `[L/256%30#32,L/8%8]`.
- Element: `[L/64%4,L%8]`.

이제 각 slice의 연속 8개 BF16 값이 실제 연속 global rows에 대응한다. 16B chunk를 만들 수 있어 그 뒤의 기존 typed `to_dm`이 padding을 제거하고 `[L%7680]`으로 모은다. Cluster 축을 원래 높은 L 축으로 복구한 뒤 기존 `UpGateRows` 분배와 GeGLU로 넘긴다. 새 연산은 데이터 이동이며 FP 연산을 추가하지 않는다.

`validate_layout.py`는 15360개 row 전체에 대해 weight/scale 위치가 일치하고, 이동이 일대일이며, Switch 이후 canonical row가 원래 값으로 복구됨을 확인했다. Padding slice에는 유효 row가 배정되지 않고, Switch 출력의 모든 8행 chunk가 16B 정렬이라는 것도 확인했다. `feedforward` 이후 GeGLU/down 소스와 기존 canonical output copy는 원본과 byte-identical이다. 이 검사는 NPU 정확도 검사를 대신하지 않는다.

## 정적 비용이 늘어난 지점

| HBM load, projection 하나당 | 기준 cycles | 새 mapping cycles |
|---|---:|---:|
| Packed weight | 26475 | 31659 |
| Local scale | 3789 | 6966 |

Up와 gate를 합친 네 load가 16722 cycles 늘었고, 두 출력의 `to_dm` 비용도 각각 2392→3074로 늘어 1364 cycles를 추가한다. 이 여섯 항의 증가가 DMA 전체 증가 18086 cycles와 정확히 일치한다. 출력 Switch 두 개를 추가한 Main은 1038 cycles 늘었다. Vector 계산량과 spill I/O 0은 유지됐다. 따라서 이 후보를 실측한다면 확인할 가설은 새 row 분산이 compiler의 증가한 전송 비용을 넘어 실제 HBM 동작을 개선하는지다. 현재는 그 개선을 주장할 근거가 없다.

후보 SHA, 기준 capture/ops SHA, 정확한 schedule 집계는 `source-hashes.json`, `evidence.json`, `static-comparison.json`에 기록했다. 원시 schedule은 ignored `../../artifacts/ffn-blockscale-schedules/bf16-row-interleave*`에 있다. 새 layout은 검증된 E15 소스에서 독립 구현했으며 외부 공개 fork 코드를 복사하지 않았다.
