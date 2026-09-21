# Stage 1 커널 코드 분석과 실험 우선순위

분석일: 2026-09-21. 분석 기준은 공식 소스 `8559916051162e5fd682c29ec0fe3e9cc6e420cf`와 Arena job `69783`이다. 아래 줄 번호는 이 소스 기준이다. 소스 변경 이후에는 함수 이름으로 다시 찾는다. 이 문서는 코드와 기존 정적 스케줄을 분석한 결과이며, 제안한 변경의 성능 개선을 실측한 보고서는 아니다.

## 1. 실제로 최적화하는 실행 경로

현재 Stage 1은 전체 Gemma 모델의 토큰 생성 속도가 아니라 세 개의 독립 커널을 검사한다. `src/bin/test_kernels.rs::main`은 `run_test`를 통해 각 커널을 세 가지 독립 입력으로 실행하고 출력 정확성과 중앙값 사이클을 수집한다. 따라서 전체 모델을 빨리 만드는 기법이라도 이 세 커널 밖에만 적용하면 현재 평가에는 반영되지 않는다. 예를 들어 `sliding_attention` 자체의 softmax/attention 최적화는 이번 세 대상과 다르다.

근거: [test_kernels.rs](../../furiosa-opt-gemma4-12B/src/bin/test_kernels.rs)의 파일 설명, `TESTS`(448행), `run_test`(466행), `main`(777행). 전체 모델에서는 [runtime.rs](../../furiosa-opt-gemma4-12B/src/host/runtime.rs)의 `sliding`(401행)이 `sliding_project_qkv → sliding_attention → sliding_attention_output → mlp`를 실행한다. 같은 파일의 `mlp`(542행)가 `decoder_feedforward`를 호출한다.

| 항목 | 코드에서 확인한 값 |
| --- | --- |
| hidden H | 3840 |
| FFN intermediate L | 15360 |
| sliding KV heads Ns / query groups Gs / head dim Ds | 8 / 2 / 256 |
| Qs / Ps | 4096 / 2048 |
| KV cache positions Ts | 1024 |
| RoPE table positions E | 32768 |
| RMS epsilon | 1e-6 |

근거: [axes.rs](../../furiosa-opt-gemma4-12B/src/axes.rs), [lib.rs](../../furiosa-opt-gemma4-12B/src/lib.rs):25. 이 shape은 프롬프트 전체의 dense GEMM보다 한 토큰의 GEMV에 가깝다. 큰 batch를 전제로 한 GPU GEMM 최적화 수치를 그대로 목표로 삼으면 안 된다.

### QKV

`test_kernels::sliding_project_qkv` → `ops::sliding_project_qkv`(38행):

```text
x[3840] bf16 HBM
 → DM load → shared::rmsnorm::normalize(input_rms_weight)
 → layout::broadcast_hidden (256-way slice broadcast)
 → sliding::projection::{project_query, project_key_value}
   FP8 weights → BF16 conversion → contraction → per-output-channel scale
 → sliding::rmsnorm::{normalize_query, normalize_key, normalize_value}
 → sliding::rope::apply_rope
 → q_out HBM write; k/v runtime offset 위치에 cache scatter
```

[ops.rs](../../furiosa-opt-gemma4-12B/src/ops.rs):58–76이 순서를 정의한다. Q는 `[8,2,256]`, K/V는 `[8,256]`이다. K만 RoPE를 적용하고 V에는 적용하지 않는다. V에는 weight 없는 RMS normalization이 존재하므로 일반적인 다른 모델의 QKV 구현을 복사하면서 생략해서는 안 된다. `kv_offset`과 `rope_offset`은 입력이고 0이라고 가정할 수 없다. 출력만 비슷하게 만드는 것뿐 아니라 올바른 cache 위치를 쓰는 부수 효과도 지켜야 한다.

[sliding/projection.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs):13의 `QueryRows=m![Qs/16]`과 84행의 `KvRows=m![Ps/8]`은 각각 256개 row group을 사용한다. Q와 KV의 activation TRF load는 별도로 존재하고, K/V는 한 `x_trf`를 공유한다. 각 projection은 FP8 → BF16 변환을 DM에 materialize한 뒤 다시 fetch하여 contraction한다. contraction 결과도 BF16으로 한 번 내보낸 다음 channel scale을 F32에서 적용하고 다시 BF16으로 변환한다.

head RMSNorm은 [sliding/rmsnorm.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/rmsnorm.rs)의 `normalize_query`, `root_mean_square`, `normalize_key`, `normalize_value`에 있다. RoPE는 [sliding/rope.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/rope.rs)의 `apply_rope`가 입력 위치의 cos/sin을 gather한 뒤 head를 slice에 분산하고 rotate-half, 곱, 합을 수행한다. cos와 sin 곱의 임시 F32 DM 및 다시 원래 배치로 모으는 단계가 있다.

### Attention output

`test_kernels::sliding_attention_output` → `ops::sliding_attention_output`(132행):

```text
attention result[8,2,256] bf16 HBM
 → reshape[4096] → layout::broadcast_sliding_heads
 → sliding::projection::project_output
 → shared::rmsnorm::normalize(post_attn_rms_weight)
 → shared::residual::add(original residual)
 → residual_hbm 덮어쓰기
```

여기에는 attention score 계산이나 softmax가 없다. [sliding/projection.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs):178의 `project_output`은 reduction 차원 4096을 1024씩 네 번 나누고, 각 `output_partial`이 FP8 weight를 BF16으로 변환하여 contraction한다. 네 BF16 결과를 `(p0+p1)+(p2+p3)`의 트리로 합한다. `add_partials`도 각각 결과를 BF16으로 변환하므로 누적 과정에 여러 반올림 경계가 있다.

`HiddenRows=m![H/120,1#8]`은 32개의 유효 output row group과 padding을 둔다. 그룹당 120 output rows와 1024 reduction elements를 다룬다. QKV의 256 row group과 차이가 크므로 output row를 더 넓게 분산하거나 reduction을 slice에 분할하는 것이 구체적인 실험 후보다. shape의 축을 바꾸는 일과 단순 `unsafe reshape`는 다르다. 실데이터 이동이 필요한 배치 변경을 reshape만으로 처리하면 안 된다.

### Feedforward

`test_kernels::decoder_feedforward` → `ops::decoder_feedforward`(207행):

```text
residual[3840] bf16 HBM
 → pre-FF RMSNorm
 → activation DM 복제
 → shared::mlp::feedforward
   up/gate FP4 unpack + per-16-element FP8 local scale
   → BF16 GEMV → reciprocal global scale → BF16
   → GeGLU
   → down FP4 unpack + local scale → BF16 GEMV
   → reciprocal down global scale → BF16
 → post-FF RMSNorm
 → residual add → BF16
 → per-layer scalar multiply → BF16
 → residual_hbm 덮어쓰기
```

[shared/mlp.rs](../../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs):12의 `project_up_and_gate`는 256 row groups, 각 60 rows를 4 rows씩 15 passes로 처리한다. up과 gate 각각 다음을 수행한다.

1. FP4 packed weight HBM → DM.
2. `fetch_table_lookup::<f8e4m3>()` 후 FP8 DM 전체를 생성.
3. 각 4-row tile마다 scale를 SubContext에서 F32 VRF에 적재.
4. MainContext에서 FP8 → F32 cast, local scale multiply, BF16 DM 생성.
5. BF16 DM을 다시 fetch하여 TRF의 activation과 contraction, BF16 output.

`project_down`(340행)은 output row 32그룹과 reduction column 8그룹을 조합한다. 120 rows를 4 rows씩 30 passes로 나누고, 각 pass를 다시 2-row halves로 처리한다. half마다 packed weight DMA, FP4→FP8 DM, local-scale 적용 BF16 DM, `DownRowsByColumns`로 TDMA redistribution, 합칠 4-row tensor에 TDMA copy가 있다. 이후 contraction과 8-way inter-slice sum으로 BF16 down output을 만든다. 한 줄의 matmul로 보아서는 설명되지 않는 변환과 이동이 상당하다.

검증된 소스상 tensor 크기로 계산하면 FFN 세 행렬은 합계 FP4 payload 88,473,600 bytes, local scales 11,059,200 bytes이다. 합계 약 94.92 MiB는 논리적 HBM weight+scale payload일 뿐이며, 실제 버스 traffic·padding·중간 DM traffic 또는 DMA 병목 여부를 의미하지 않는다. QKV FP8 행렬 payload는 30 MiB, attention-output FP8 행렬은 15 MiB다.

## 2. 기존 실측과 정적 병목

[baseline-result-69783.json](../../moa-dev/baseline-result-69783.json)의 실측:

| 커널 | run0 | run1 | run2 | 중앙값 |
| --- | ---: | ---: | ---: | ---: |
| QKV | 251764 | 244051 | 243259 | 244051 |
| Attention output | 404373 | 402874 | 404457 | 404373 |
| Feedforward | 4263585 | 4267547 | 4264297 | 4264297 |

모두 정확성 통과. FFN의 절대 시간이 가장 크다는 사실만으로 최우선이라고 결론내리지 않는다. 세 커널 speedup의 기하평균을 목표로 하면 동일 비율의 개선은 어느 커널에서 얻어도 같은 기여를 한다. 예를 들어 한 커널만 2배 빨라지면 전체 기하평균은 약 1.26배이며, 두 커널을 각각 1.5배 개선하면 약 1.31배다. 따라서 현재 근거가 가장 분명한 attention-output와 QKV broadcast부터 검증할 가치가 있다.

기존 파일 `furiosa-opt-gemma4-12B/target/schedules/baseline_sliding_attention_output.json`의 163 instructions를 읽어 각 context의 겹치는 lifetime을 합집합으로 계산했다. 이 파일은 Git에서 제외되는 빌드 artifact다.

| 정적 항목 | cycles | makespan 비율 |
| --- | ---: | ---: |
| makespan | 195823 | 100% |
| MainContext union | 184277 | 94.10% |
| VectorEngine union | 40114 | 20.48% |
| DmaEngine union | 35525 | 18.14% |
| SubContext union | 12134 | 6.20% |

Context들은 겹쳐 동작하므로 이 비율을 더하면 안 된다. 특히 VectorEngine이 MainContext와 동시에 표시되는 instruction이 있다. 표는 하드웨어 실측 이용률이 아니라 컴파일러 정적 계획의 interval 점유율이다.

| 정적 병목 후보 | 위치 | cycles | 전체 비율 |
| --- | --- | ---: | ---: |
| 256-way activation broadcast | `layout.rs::broadcast_sliding_heads`, 31행 | 66311 | 33.86% |
| FP8→BF16 weight conversion 4회 | `sliding/projection.rs::output_partial`, 218행 | 19463 × 4 = 77852 | 39.76% |
| contraction 4회 | 같은 함수, 226행 | 7945 × 4 = 31780 | 16.23% |
| weight HBM DMA 4회 | 같은 함수, 214행 | 6966 × 4 = 27864 | 14.23% |

큰 weight DMA는 긴 compute와 겹쳐 있다. 정적 스케줄에서 broadcast와 conversion만 약 73.6%를 차지하므로 첫 실험은 이 MainContext 작업을 줄이거나 다른 엔진으로 이동하는 것이다. 작은 RMSNorm이나 residual add를 먼저 미세 조정하는 것은 현재 근거와 맞지 않는다. DMA weight node의 `total_util`은 약 0.415이지만 이를 곧바로 전체 HBM bandwidth가 41.5%라고 해석하면 안 된다.

정적 195823과 실측 404373은 크게 다르다. 원인을 확정하지 않았으며 상수 보정계수를 만들어 새 커널 성능을 예측하지 않는다. 정적 변화는 후보 선별에 사용하고 성능 주장은 반드시 RNGD 측정으로 한다.

추가로 root가 같은 baseline으로 생성한 `target/schedules/baseline/sliding_project_qkv.json`과 `decoder_feedforward.json`을 분석했다.

| 정적 항목 | QKV cycles / 비율 | FFN cycles / 비율 |
| --- | ---: | ---: |
| makespan | 118815 / 100% | 1918472 / 100% |
| MainContext union | 114019 / 95.96% | 872451 / 45.48% |
| DmaEngine union | 66284 / 55.79% | 1066334 / 55.58% |
| VectorEngine union | 22529 / 18.96% | 675900 / 35.23% |
| SubContext union | 10795 / 9.09% | 129898 / 6.77% |

QKV에서는 `layout.rs::broadcast_hidden` 62215 cycles가 전체의 **52.36%**다. FP8→BF16 변환은 Q 9863, K/V 각각 5063 cycles이고, 세 weight DMA는 각각 26475/13512/13512 cycles지만 대부분 broadcast와 겹친다. broadcast 엔진을 바꾸면 이후 DMA가 노출될 수 있으므로 62215를 그대로 실측 절감량으로 예상하지 않는다.

FFN의 원인은 훨씬 구체적이다. 다음 표는 해당 소스 줄에서 생성된 instruction들의 **지정 context** lifetime 합이다. PeCore의 작은 control instruction은 제외했다.

| FFN down 단계 | Main 또는 DMA cycles | 전체 비율 |
| --- | ---: | ---: |
| 2-row weight를 4-row tensor에 TDMA copy (`mlp.rs`:412) | DMA 8115 × 60 = 486900 | 25.38% |
| BF16 dequant tile의 column-split redistribution (`mlp.rs`:410) | DMA 4275 × 60 = 256500 | 13.37% |
| local scale 적용 → BF16 DM (`mlp.rs`:394) | Main/Vector 7961 × 60 = 477660 | 24.90% |
| FP4→FP8 LUT/DM (`mlp.rs`:373) | Main 2663 × 60 = 159780 | 8.33% |
| packed weight HBM load (`mlp.rs`:369) | DMA 1350 × 60 = 81000 | 4.22% |
| down contraction+interslice reduction (`mlp.rs`:420) | Main/Vector 752 × 30 = 22560 | 1.18% |

현재 FFN의 주된 문제는 down GEMV 산술 자체보다 **dequant와 두 단계 DM 이동**이다. 특히 두 이동은 합계 743400 cycles, 정적 makespan의 38.75%다. 전체 FFN에서 MainContext와 DmaEngine이 겹친 시간은 56239 cycles(2.93%)뿐이어서 상당 부분이 직렬 dependency로 연결된다. DMA 점유가 높다고 모두 HBM bandwidth 문제라고 부르면 이 개선 지점을 놓친다.

첫 FFN 변경은 위의 두 이동을 한 번에 재설계하기보다, 이미 column-split으로 변환된 2-row `down_weight_tile`에 즉시 contraction하고 2-row output view에 쓰는 방식이 좋다. 현재는 두 half를 4-row 임시 tensor에 모은 뒤 contraction하므로 486900-cycle copy가 추가된다. 2-row contraction을 60회 하면 기존 4-row contraction 30회보다 명령 오버헤드가 늘 수 있지만, 제거 대상에 비해 현재 contraction 시간은 작다. 각 row의 reduction·scale·BF16 경계를 유지하기 쉬워 직접 packed layout 변경보다 좁은 가설이다.

## 3. 우선순위를 둔 실험

| 순서 | 변경 가설 | 근거와 기대 | 주요 검증/실패 조건 |
| --- | --- | --- | --- |
| 1 | QKV/attention activation broadcast를 TDMA `to_dm` 변환으로 대체 | attention의 66311-cycle MainContext broadcast 제거 가능. FFN `ops.rs`:226에 이미 같은 `Slice → Replicated` 타입의 `to_dm` 경로가 있다 | 정확한 broadcast 의미와 compile 가능성. TDMA traffic/대기 증가 여부. QKV·attention을 각각 분리 검증 |
| 2 | attention-output row mapping을 256 groups × 15 rows로 변경하거나 row/column 분할 조정 | 현재 유효 row groups 32개. 변환·contraction의 직렬 길이를 줄일 후보. 단일 4096 contraction이면 네 partial과 세 add를 줄일 수 있다 | TRF 수용량, 15-row padding, DMA layout, BF16 partial rounding 변경. compile 실패시 2048/1024 split은 유지하고 row mapping만 실험 |
| 3 | FP8→BF16 conversion과 contraction 사이 DM materialization 제거 또는 변환 엔진 이동 | attention weight conversion 정적 77852 cycles. QKV도 동일한 구조 | 실제 SDK의 chain/engine 지원 확인. Main/Sub는 Tensor Unit 자원을 공유하므로 단순 `.main → .sub` 치환의 효과를 가정하지 않는다 |
| 4a | FFN down의 2-row tile를 즉시 contract, 4-row assembly copy 제거 | `mlp.rs`:412의 DMA 486900 cycles가 명확한 제거 대상. 다른 scale/layout은 유지 가능 | contraction shape의 `H%120=4`를 2로 일관되게 변경. output index는 `4*i+2*j`. BF16과 8-way reduction 보존. 60회 contraction 오버헤드 측정 |
| 4b | FFN down을 packed/scale 단계부터 `DownRowsByColumns`에 직접 적재 | 남은 BF16 dequant 후 redistribution은 DMA 256500 cycles | packed FP4 nibble 순서, scale의 16-element grouping, strided DMA 효율, 정적 변환/이동이 다른 노드로 재등장하는지 검증 |
| 5 | FFN unpack→scale→contract fusion 또는 타일 크기 sweep | FP4→FP8 DM→BF16 DM 왕복과 up/gate 15 passes, down 30×2 halves 감소 후보 | LUT가 지원하는 타입/packet 경계, VRF/TRF/DM live memory, scale broadcasting과 reduction order. row-pass 4→8 등은 축 전체와 tail 처리를 같이 변경해야 함 |
| 6 | QKV projection 이후 head 분산 layout을 RMSNorm/RoPE까지 유지 | 현재 projection 결과를 `Slice`로 모았다가 RoPE에서 head에 다시 분산 | query head order `[Ns,Gs,Ds]`, RMS reduction axis, sin low-half 부호, cache scatter 의미 |
| 7 | RMSNorm의 rsqrt, distributed output, residual/layer-scale fusion | sqrt와 elementwise divide, DM 중간 결과, 최종 gather가 줄 후보 | epsilon, accumulation precision, BF16 rounding 경계. 상위 병목 제거 뒤 재측정해서 우선순위 재평가 |

위 순서는 점수·구현 위험·측정 근거를 함께 고려한 출발점이다. FFN 스케줄까지 확인한 이후에는 1번과 4a를 우선 독립 실험할 근거가 생겼다. 1번이 실제 개선을 보이지 않으면 후보를 폐기하고 2번으로 이동한다. 성능이 좋지 않은 결과도 artifact와 이유를 남겨 같은 실험을 반복하지 않는다. 3번, 4b, 5–7번은 지원 API와 스케줄을 더 확인해야 하며 아직 구현 가능한 최종 설계로 확정하지 않았다.

## 4. 수학적으로 지켜야 할 것

FP8 projection의 참조식은 `bf16((x.float @ codes.T) * row_scale.float)`다. [generate_references.py](../../furiosa-opt-gemma4-12B/scripts/generate_references.py):265의 `scaled_projection`이 직접 정의한다. 현재 NPU skeleton의 중간 BF16 partial 경계와 이 이상적인 참조식은 완전히 같지 않다. 따라서 더 높은 정밀도로 contraction/scale을 합친 새 경로가 참조에 가까워질 수 있지만, 연산 순서 변경이 모든 출력의 허용오차 통과를 보장하지는 않는다.

NVFP4 local scale는 입력 채널 16개마다 다르다. `sum_k x[k] * code[r,k] * local[r,k/16]`의 `local`을 행 전체 contraction 밖으로 빼면 일반적으로 틀린다. global scale는 행렬 전체 scalar이고 입력 tensor에 이미 reciprocal로 들어온다. 또 `GloballyScaled.forward`(같은 파일 371행)는 BF16 linear output을 F32로 올려 scale을 곱한 다음 다시 BF16으로 내린다. fusion 시 이 경계를 무심코 삭제하지 않는다.

`RMSNorm(a*x)`에서 양수 scalar `a`는 epsilon이 있을 때 정확히 사라지지 않는다. 식은 `x / sqrt(mean(x²)+eps/a²)`가 된다. 여기에 BF16 round가 끼면 더 달라진다. post-FF RMSNorm이 있다는 이유로 down global scale을 생략하거나, head norm이 있다는 이유로 임의의 per-channel scale을 생략하는 것은 타당하지 않다.

`shared/mlp.rs::geglu`는 erf 기반 GELU를 계산하지만 참조 [gemma4.py](../../furiosa-opt-gemma4-12B/scripts/reference/gemma4.py):197은 `F.gelu(..., approximate="tanh")`다. 기존 skeleton이 현재 허용오차에 통과한다는 사실은 확인됐으나 GELU 근사법 교체의 무조건적 안전성을 뜻하지 않는다. residual 합 뒤 BF16 반올림 후 layer scalar를 곱하는 순서도 참조가 명시한다.

정확성 조건은 `abs(actual-expected) <= atol + 0.01*abs(expected)`를 **모든 출력 원소**에서 만족하는 것이다. absolute tolerance는 QKV 0.04, attention-output 0.05, FFN 0.01이고 non-finite 값은 즉시 실패다. 근거: `test_kernels.rs::TESTS`, `compare`(626행). 일부 비율만 맞거나 평균 오차만 작다고 통과하지 않는다.

세 seeded input만 통과하도록 값·seed·offset을 고정하거나 참조 출력·global scale을 하드코딩하지 않는다. 일반 입력에 유효한 계산을 유지하고 fixture/harness는 그대로 둔다.

## 5. 각 변경의 검증 순서

1. 공식 baseline의 source commit, compiler/toolchain, fixture SHA256, binary SHA256, Arena job id와 3-run cycles를 보존한다.
2. 한 가설만 변경하고 대상 커널을 `cargo furiosa-opt compile ops::<name> --exact --dump-schedule <new-file>`로 확인한다. baseline 파일을 덮어쓰지 않는다.
3. makespan, context interval union, 긴 nodes, DMA `util`, 임시 tensor lifetime·메모리를 비교한다. peak memory나 serialization이 새 병목이면 원인과 함께 기록한다.
4. 공식 `test_kernels` binary를 빌드하고 동일 fixture로 RNGD 평가한다. 기존 서버 timeout 설정은 `RNGD_SUBMIT_TIMEOUT=70`이다. 변경 대상 외 두 커널도 모두 통과해야 한다.
5. 작은 개선이 process 간 잡음과 비슷하면 후보/기준을 번갈아 반복하여 중앙값과 범위를 비교한다. test harness의 run 수·settle·profiling 설정을 임의 변경한 수치를 공식 수치와 섞지 않는다.
6. 채택 조건은 전체 correctness 통과와 재현되는 기하평균 점수 개선이다. 새로운 schedule 변화가 없고 실측만 작게 좋아진 결과는 확정 성과로 기록하지 않는다.

이 문서의 코드 분석만으로 최종 순위를 보장할 수 없다. 상위권 진입 판단은 동일 규칙으로 얻은 재현 가능한 실측과 최신 공개 leaderboard의 격차로 업데이트해야 한다.
