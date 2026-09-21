# Native FP8 O projection의 4-way weight row interleave

2026-09-21, SDK 0.8.1. **Exact compile 통과, NPU 미실행.** 고정한 E24 source를 새 scratch에서 그대로 컴파일해 기준 **32252 cycles**를 재현했다. 새 mapping은 **34435 cycles(+6.77%)**다. Weight DMA 13512, 전체 DMA 22995, Vector 7914, Sub 2578은 그대로이고, 출력 Switch 때문에 Main이 8418→10601로 2183 늘었다. Compiler는 weight DMA 이득을 예측하지 않았다.

기준은 `../../artifacts/e24-native-output-safe-qkv-ffn/src`다. 이 실험은 root의 별도 parallel epilogue 변경을 포함하지 않는다. QKV, 입력 quantizer의 산술, channel scale, normalization/residual 경로를 보존하고 O의 row 분배와 BF16 출력 복구만 바꾼다.

## 실제 실행 경로

`ops::sliding_attention_output` → `project_output` → 512-column별 기존 hi/lo quantizer → native FP8 two-lane dot → 기존 F32 partial DM → 기존 8-column F32 sum → **기존 위치에서 BF16 변환** → 새 row 복구 Switch → 기존 canonical gather/channel scale → 기존 fused normalization/residual.

Weight는 여전히 2 clusters × 256 slices에 slice당 `[60 rows,512 columns]`를 저장한다. 다만 cluster 안의 row `r=240g+4u+s`를 slice `(4g+s)*8+c`, local row `u`로 보낸다. `c`는 같은 Qs/512 column shard다. 연속 4개 HBM row가 서로 다른 4개 rowgroup에 배치된다.

| 구분 | Slice | Element |
|---|---|---|
| E24 weight DM | `[H/60%32,Qs/512]` | `[H%60,Qs%512]` |
| 새 weight DM | `[H/240%8,H%4,Qs/512]` | `[H/4%60,Qs%512]` |
| 새 BF16 column-sum 결과 | `[H/240%8,H%4,1#8]` | `[H/4%60]` |
| Switch 복구 결과 | `[H/60%32,1#8]` | `[H%60]` |

8개 column shards는 각 row에서 같은 물리적으로 연속한 8개 slices에 남는다. Qs 방향 dot 순서, hi/lo 합산, F32 column reduction과 BF16 반올림 경계는 바꾸지 않았다. 입력 quantizer는 각 column shard의 실제 입력으로 기존과 동일하게 동작한다.

## 60개 local rows를 정확히 복구하는 방법

`u=15q+t`로 나누면 원래 row는 `r=240g+60q+4t+s`다. BF16 결과에서 `InterTranspose { slice1:4, slice0:8, time0:15 }`를 실행하면 slice의 `s`와 local row의 `q`가 교환된다. 출력 rowgroup은 `4g+q`, local row는 `4t+s`가 되어 E24의 60행 분배와 일치한다. `time0=15`를 포함한 이 설정은 실제 SDK exact compile에서 통과했다.

Switch를 BF16 column sum 뒤에 두므로 추가 FP 연산이나 반올림을 넣지 않는다. 이후 canonical gather와 channel scale은 원래 코드 그대로 실행한다. 새 mapping에서 weight scale을 조기에 적용하거나 같은 값을 두 번 적용하지 않는다.

`validate_layout.py`는 3840 rows × 8 column shards의 **30720개 좌표**를 전수 확인했다. 각 physical slice에 60개 row가 중복 없이 들어가고, column shard가 유지되며, Switch 이후 모든 global row가 원래 위치로 복구된다. QKV 앞부분과 quantizer/epilogue 뒷부분이 type alias 한 줄을 제외하고 byte-identical이라는 source 확인도 포함한다. 이 검사는 compiler/runtime의 실제 정확도를 보증하지 않는다.

후보와 baseline의 source/hash는 `source-hashes.json`, 정적 비교는 `static-comparison.json`, 전체 검사 결과는 `logical-index-proof.json`과 `evidence.json`에 있다. 원시 schedule은 ignored `../../artifacts/output-row-interleave-schedules/o-row-*`에 보존했다. 모든 source는 E24의 자체 구현에서 파생했고 외부 공개 fork 코드를 복사하지 않았다.
