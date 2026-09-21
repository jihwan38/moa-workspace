# Up/gate/down 전체 2클러스터 fused BF16 + blockscale-after-dot

**제출 보류/대체 후보 필요.** 이 후보의 up/gate와 동일한 fused decode 경로가 parent job70162에서 FFN FAIL 및 다음 launch timeout을 일으켰다. 이 후보 자체는 NPU 제출하지 않았으며 원본을 진단용으로 보존한다. 수학적 scalar 변환 exactness가 컴파일된 packet 경로의 정확성을 증명하지 않는다. 두 단계 decode 대체는 `../ffn-blockscale-all-two-cluster-bf16-reuse`다.

`ffn-blockscale-two-cluster-fused-bf16/mlp.rs`에서 down만 바꾼다. `DownCluster` 선언 전까지 up/gate, `feedforward`, GeGLU는 byte-identical이다. Down은 2클러스터 ×32개 row group ×60행, 각 row group에 8개의 column slice를 둔다. Slice당 1920개 열을 120개의 16원소 block으로 나누고, 12행 tile을 5회 처리한다.

입력 경로는 packed FP4 DM → F8 LUT → F32 cast → unscaled BF16 DM 1회 생성 → 16원소 BF16 dot의 FP32 결과 → block별 FP8 local scale을 FP32로 곱 → 120개 block의 FP32 합 → 8개 column slice의 FP32 합 → BF16 출력이다. 원래 BF16 cast는 slice 합산 **뒤에** 있었으며 그대로 유지했다. 12행 출력은 4행 packet 3개로 commit한다. Scale VRF는 slice당 5760B이며 full packed/BF16 weight 합계는 288000B다. 실제 liveness/배치는 compiler가 결정한다.

Exact `ops::decoder_feedforward` compile 통과, compiler spill I/O 0. 정적 makespan은 **160349 cycles**, 부모 176264보다 9.0%, 일반 dequant up8/down12 204460보다 21.6% 낮다. Main 97746, Vector 93398, DMA 132947, Sub 17403이다. 정적 비율은 실측 향상 비율이 아니다. `evidence.json`, `source-hashes.json`, `static-comparison.json`, `compile.log`에 기록했으며 raw dump는 ignored `../../artifacts/ffn-blockscale-schedules/all-fused-bf16*`에 있다. 이 에이전트는 scratch compile만 수행했고 NPU를 제출하지 않았다.

`validate_candidate.py --output NEW.json`은 stdlib로 다음을 검사한다: 전체 3840행/15360열/960개 scale block의 누락·중복 없는 배치, local FP32→slice FP32→BF16 순서, up/gate/GeGLU 보존, 전체 유한 FP4×FP8 코드 4064개의 BF16 exactness, FP4 fused 변환 16개 코드의 exactness. 설치된 SDK decode table 자체 검증은 이전 `../ffn-blockscale-after-dot/numerical-proof.json`에 있다. 결과는 `numerical-layout-proof.json`이다.

대수적으로는 `sum_b,j (s_b*q_bj)*x_bj = sum_b s_b*sum_j q_bj*x_bj`다. 하지만 FP32 누산 순서가 달라 bitwise 동치는 아니며, 모든 유한 BF16 입력에서 overflow 동치라는 주장도 성립하지 않는다. 구체적으로 16개 activation이 `2^124`, q=1, local scale=1/512이면 원래 scaled dot은 `2^119`로 유한하지만 unscaled dot은 `2^128`이어서 FP32 최대 유한값을 넘는다. 이 반례는 수학적 범위 분석이며 실제 NPU 관측 결과는 아니다.

조건부 안전 범위는 fixture 값에 의존하지 않고 전체 FP4/FP8 유한 범위에서 유도했다. Down 입력이 유한하며 `|x| <= 2^101`일 때, 최악 coefficient 합 `15360*6*448=41287680`, 보수적인 FP32 rounding 계수 2와 subnormal error 여유를 포함해 중간 dot/합과 마지막 BF16이 모두 최대 유한값 안에 든다. 이 조건은 현재 kernel에서 검사하지 않으며 global scale 이후의 overflow, NaN/Inf, hardware flush-to-zero 동치를 증명하지 않는다. 하드웨어 official/추가 seed 검증이 별도로 필요하다. 테스트 입력 범위를 kernel에 하드코딩하거나 특수 분기하지 않았다.

적용 시 `ops.rs` 전체 교체 대신 `change-ops.patch`의 입력 복제 제거 한 줄만 반영한다. 부모 후보 및 공식 source/harness/live binary는 이 작업에서 변경하지 않았다.
