# Native FP8 O projection의 2-way row interleave

2026-09-21, SDK 0.8.1. **Exact compile 통과, NPU 미실행.** 같은 E24 기준의 4-way 후보보다 출력 복구 비용을 줄이는 별도 mapping이다. 정적 makespan은 **33475 cycles**, E24 32252 대비 +3.79%이며 4-way 34435보다 960 낮다. Weight DMA 13512, 전체 DMA 22995, Vector 7914, Sub 2578은 그대로다. Main에는 Switch 1223 cycles가 추가되어 8418→9641이다. 실제 HBM bank 효율이 개선된다는 증거는 아직 없다.

기준은 `../../artifacts/e24-native-output-safe-qkv-ffn/src`이고 같은 source의 fresh exact baseline compile은 `../output-fp8-row-interleave-four/control-compile.log`에 있다. Root의 별도 parallel epilogue는 포함하지 않는다. 두 후보 모두 NPU를 제출하지 않았다.

Cluster 내 row를 `r=120g+2u+s`로 쓴다. Slice는 `[H/120%16,H%2,Qs/512]`, local row는 `[H/2%60]`이다. 기존 512-column shard, slice당 60행, native FP8 hi/lo 산술, F32 partial DM, 8-column reduction은 그대로 유지한다. 연속 두 HBM row만 서로 다른 rowgroup에 보낸다.

BF16 column sum 이후 `u=30q+t`를 이용한다. `InterTranspose { slice1:2, slice0:8, time0:30 }`가 slice의 `s`와 local row의 `q`를 교환하면 rowgroup은 `2g+q`, local row는 `2t+s`가 된다. 이는 기존 `[H/60%32,1#8]` / `[H%60]` mapping이다. 이후 canonical gather, channel scale, normalization/residual은 E24의 원래 코드다. Switch는 BF16 비트 이동만 수행하며 FP 연산이나 반올림 경계를 추가하지 않는다.

`validate_layout.py`로 3840×8=30720개 row/column 좌표를 전수 확인했다. 각 physical slice가 60개 row를 갖고 모든 좌표가 중복 없이 원래 row로 복구된다. Column shard 순서는 바뀌지 않는다. QKV와 quantizer/epilogue 소스가 type alias 변경 외에는 byte-identical임을 확인했다. 정확한 compiler/kernel 동작은 별도 NPU 검증이 필요하다.

`projection.rs`와 `projection.original.rs`가 후보/기준이다. `source-hashes.json`, `logical-index-proof.json`, `static-comparison.json`, `evidence.json`에 source identity, 검증, 정적 비용을 기록했다. 원시 schedule은 ignored `../../artifacts/output-row-interleave-schedules/o-row-two*`에 있다. 외부 공개 fork 코드를 복사하지 않았다.
