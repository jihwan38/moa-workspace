# 단일 pass BF16 생성 + 2클러스터 blockscale-after-dot

**NPU 검증 실패로 채택하지 않는다.** Root의 job70162에서 FFN run0 max delta 0.63281, mean delta 0.041770, tolerance 내 비율 30.96%로 FAIL했고 다음 launch가 timeout으로 exit101 종료했다. QKV/O는 통과했다. 아래 compile/수학 증거는 실제 fused decode 경로의 정확성을 증명하지 않는다. 원인은 아직 확정하지 않았으며 raw log와 hash는 `evidence.json`에 있다. 두 단계 decode를 쓰는 E12는 job70157에서 official 15 comparisons PASS였다.

`ffn-blockscale-two-cluster-bf16-reuse`에서 up/gate full unscaled BF16 생성만 바꾼 후보다. FP4 packed → `fetch_table_lookup::<f8e4m3>()` → `fetch_cast::<f32>()` → `collect` → `cast::<bf16>()` → BF16 DM으로 한 번에 생성한다. 중간 FP8 DM 및 FP8→BF16 table lookup이 사라진다. SDK 0.8.1은 LUT 연속 호출과 F4→BF16 직접 LUT를 제공하지 않지만, 이 LUT 뒤 F32 cast 경로는 compile 가능한 공개 API다. `/root/kernel_audit`가 검증한 방식에 근거한다.

FP4의 모든 16개 값은 F8/F32/BF16에 정확히 표현되므로 이 전환은 weight 근사를 추가하지 않는다. 이후 8행 tile의 16원소 dot→local scale→FP32 block 합산은 부모 후보와 동일하다. `feedforward` 이하(GeGLU 및 2클러스터 down12)와 blockdot helper는 byte-identical임을 확인했다. 부모와 동일한 `ops.rs` 입력 복제 제거 한 줄만 필요하다(`change-ops.patch`).

Exact `ops::decoder_feedforward` compile 통과. 정적 FFN makespan **176264 cycles**로 부모의 181155 대비 2.7%, 일반 dequant up8/down12의 204460 대비 13.8% 낮다. Main 120386, Vector 106775, DMA 132947, Sub 17662다. Compiler가 보고한 spill I/O는 0이며 전체 table lookup DRAM 상수는 3개다. Source/결과 hash는 `source-hashes.json`, `evidence.json`, 세부 비교는 `static-comparison.json`; raw schedule은 ignored `../../artifacts/ffn-blockscale-schedules/fused-bf16*`다.

Root가 release full build 후 NPU job70162를 실행했으며 위에 기록한 FFN FAIL/timeout으로 후보를 기각했다. 기존 1클러스터 blockdot PASS와 수학적 exactness는 이 fused decode 경로의 정확성을 보장하지 않았다. 공식 benchmark 및 추가 seed 개발 검사는 별도로 기록한다.
