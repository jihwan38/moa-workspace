# 2클러스터 BF16 재사용 + blockscale-after-dot 후보

`mlp.rs`의 up/gate는 FP4 packed HBM → FP8 DM → **전체 unscaled BF16 DM 1회 생성** → 8행씩 16원소 block dot → FP32 local scale 곱 → block 합산 → BF16 출력 순서다. FP8→BF16 lookup을 행 tile마다 반복하지 않고 up/gate 각각 한 번 생성한다. 2클러스터×240개 유효 slice×32행 배치는 `ffn-blockscale-two-cluster-eight-row`와 같다. Down은 `ffn-larger-tiles/mlp.rs`의 `DownCluster`부터 EOF를 복사한 2클러스터 12행 일반 dequant 버전이고, GeGLU 뒤 불필요한 중간 `x.to_dm` 복제를 제거했다. GeGLU 자체는 동일하다.

Exact `ops::decoder_feedforward` compile은 통과했다. 정적 makespan **181155 cycles**로 동일 down12를 사용하는 일반 dequant up8/down12의 204460보다 11.4% 낮다. Main 143952, Vector 71689, DMA 134623, Sub 20766이며 compiler가 보고한 spill I/O는 0이다. 이는 compiler schedule 추정이고 NPU 실측/정확도 결과가 아니다. Raw schedule과 graph는 ignored `../../artifacts/ffn-blockscale-schedules/bf16-reuse*`, 집계는 `static-comparison.json`, source/compile 기록은 `source-hashes.json`과 `evidence.json`에 있다.

각 up/gate weight의 slice당 full storage는 FP4 61440B, FP8 122880B, BF16 245760B이며 세 항의 합은 430080B다. 실제 liveness는 compiler가 정하며 이 단순 합산만으로 placement를 증명하지 않는다. Scale VRF 7680B, activation/table/다른 tensor도 존재한다. 이번 compiler는 전체 프로그램을 spill 없이 배치했다. DRAM `table_lookup` 상수는 전체 FFN에서 5개다(3개 FP4→FP8와 2개 FP8→BF16). 이전 8행 blockdot은 11개였다.

`ops.rs`는 기존 single-cluster 입력 복제 한 줄을 없앤 상태다. 적용할 때 전체 ops 파일을 덮어쓰지 말고 `change-ops.patch`의 한 줄만 반영해야 한다. Root의 다른 커널 변경과 무관하다. 이 후보의 원본 up/gate는 `mlp.original.rs`, 복사한 down suffix는 `down.original.rs`에 고정했다.

FP4×FP8 local scale이 BF16에 정확히 표현된다는 CPU 증명은 `../ffn-blockscale-after-dot/numerical-proof.json`을 따른다. 그러나 scale을 16원소 dot 뒤로 이동하면 FP32 누산 순서가 바뀐다. 첫 1클러스터 구현이 official 3 seeds를 통과한 사실이 이 2클러스터/8행 commit의 정확성을 대신하지 않는다. NPU 실행과 extra seeds가 필요하며 공식 score와 구분해서 기록해야 한다.
