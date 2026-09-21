# 2클러스터 8행 blockscale-after-dot 후보

`ffn-upgate-two-cluster/mlp.rs`의 up/gate에서만 FP4 block scale을 16원소 dot 뒤로 옮긴다. 각 클러스터 240개 유효 slice, slice당 32행을 8행씩 4회 처리하므로 FP8→BF16 LUT setup은 up/gate 합계 8회다. Scale VRF는 slice당 7680B다. 출력은 8행을 4행 BF16 packet 2개로 나눠 commit한다. `logical-index-proof.json`은 15360개 행을 한 번씩 다루는 정수 인덱스 검사이며 NPU 수치 검증을 대신하지 않는다.

Exact `ops::decoder_feedforward` compile은 통과했다. 정적 makespan 291658 cycles(Main 186731, Vector 144394, DMA 176298, Sub 35999)이며 NPU 실행과 release full build는 아직 하지 않았다. 원본 single-cluster blockdot의 정적 403875 대비 setup이 줄었지만, 이 후보의 down은 기존 1클러스터라 전체 2클러스터 down 후보와 직접 비교하면 안 된다. 후속 `ffn-blockscale-two-cluster-bf16-reuse`는 2클러스터 down을 사용한다.

`ops.rs`의 변경은 `decoder_feedforward`에서 이미 하는 `x` 전체 slice 복제 한 줄 제거다. 실제 반영 시 전체 `ops.rs`를 교체하지 말고 `change-ops.patch`의 한 줄만 적용해야 한다. MLP의 `feedforward`가 입력을 두 클러스터로 broadcast한다. `feedforward` 이하 GeGLU/down은 원래 2클러스터 up/gate 후보와 byte-identical이다.

FP4×FP8의 BF16 exactness는 `../ffn-blockscale-after-dot/numerical-proof.json`에서 모든 유한 코드 조합으로 검사했지만 FP32 합산 순서 변화는 남는다. 첫 1클러스터 구현만 official 3 seeds에서 통과했으며, 이 후보의 다른 mapping/commit 경로는 별도 실행해야 한다. 검증 기록은 `evidence.json`, 원본 schedule은 ignored `../../artifacts/ffn-blockscale-schedules/two-cluster-eight*`, 전체 정적 비교는 `static-comparison.json`이다. 실패했던 output padding/8행 transpose 로그도 보존했다.
