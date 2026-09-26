# 국내외 최적화 대회에서 MOA로 가져올 수 있는 기법

조사일: 2026-09-21. 공식 대회 안내, 주최자 자료, 수상자의 공개 보고서/코드, 원 논문을 읽고 현재 저장소의 호출 경로와 대조했다. 아래 외부 성능 수치는 해당 저자·주최자가 보고한 결과이며 우리 RNGD에서 재현한 결과가 아니다. 링크된 영상 전체나 외부 저장소 전체를 검토한 것은 아니다.

**역사적 조사 기록이다.** 아래 baseline·후보 순서는 작성 당시 기준이며,
현재 E104의 비용 가설·최신 점수·추가 NVFP4 GEMV 자료는
[2026-09-26 보완 조사](e104-bottleneck-20260926.md)를 따른다.
실험 단위와 결합 설계 규칙은 루트 AGENTS.md가 최신 기준이다.

현재 우선순위는 **측정 가능한 기준선 → broadcast/전송 경로 → projection의 타일·변환 융합 → FP4 MLP의 복호화·재배치 비용 제거**다. FlashAttention, 모델 재양자화, CPU launcher 최적화는 Stage 1의 직접 해법이 아니다. 상위 3위는 경쟁 팀의 최종 점수에 달려 있으므로 특정 기법만으로 보장할 수 없다. 실제 순위와 재현 가능한 점수 차이로 목표를 갱신한다.

## 1. 먼저 구분해야 하는 MOA의 문제

코드에서 확인한 Stage 1 진입점은 다음과 같다.

| 진입점 | 실제 경로 | 직접 수정 후보 |
|---|---|---|
| `ops::sliding_project_qkv` | 입력 RMSNorm → hidden broadcast → Q/K/V projection → Q/K/V normalization → RoPE → Q 출력과 KV cache scatter | `src/device/layout.rs`, `sliding/projection.rs`, `sliding/rmsnorm.rs`, `sliding/rope.rs` |
| `ops::sliding_attention_output` | head broadcast → O projection → RMSNorm → residual add | `layout.rs`, `sliding/projection.rs::project_output`, `shared/rmsnorm.rs`, `shared/residual.rs` |
| `ops::decoder_feedforward` | pre-RMSNorm → up/gate projection → GeGLU → down projection → post-RMSNorm → residual/layer gate | `shared/mlp.rs`, `shared/rmsnorm.rs`, `shared/residual.rs` |

`sliding_attention_output`에는 QKᵀ, softmax, AV가 없다. 실제 attention인 `sliding::attention::attend`는 별도 `ops::sliding_attention`에서 호출된다. 따라서 FlashAttention 구현을 먼저 바꿔도 Stage 1의 세 측정값은 직접 개선되지 않는다. 이 구분은 `src/ops.rs`의 함수 본문으로 확인했다.

기존 RNGD 기록 [baseline-result-69783.json](../../moa-dev/baseline-result-69783.json)은 세 입력 구성 모두 통과했고 커널 중앙값은 QKV 244,051, O projection 경로 404,373, FFN 4,264,297 cycles다. 이는 과거 기준선 한 job의 기록이며 현재 제출 서버의 공식 순위가 아니다.

현재 [대회 README](../../furiosa-opt-gemma4-12B/README.md)의 제출 절은 세 커널 speedup의 기하평균을 명시한다.

```text
score = (speedup_qkv × speedup_output × speedup_ffn)^(1/3)
log(score) = (log(speedup_qkv) + log(speedup_output) + log(speedup_ffn)) / 3
```

따라서 FFN이 이 세 cycle 합계의 약 87%여도 Stage 1 가중치가 87%라는 뜻은 아니다. FFN만 2배 빨라지면 점수는 약 1.260배이고, 세 커널이 각각 1.5배 빨라지면 점수는 1.5배다. FFN의 큰 절대 시간과 세 커널의 동일한 로그 기여를 함께 고려해 **예상 log-speedup / 구현·컴파일·측정 비용**이 큰 실험부터 한다. 이는 공식 식에 대한 계산이며 외부 대회 성능을 이식한 추정이 아니다.

## 2. 엄선한 8개 자료군과 이전 가능한 내용

### S1. GPU MODE / AMD Developer Challenge 공식 FP8 GEMM 문제

- 자료: [공식 문제집](https://github.com/gpu-mode/reference-kernels), [FP8 matmul task.yml](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/fp8-mm/task.yml), [reference.py](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/fp8-mm/reference.py).
- 시점: AMD Developer Challenge 2025; 확인 2026-09-21.
- 확인한 사실: MI300에서 DeepSeek 계열 고정 shape와 block scale을 다루며, 문제 규격에 기하평균 순위와 메모리 레이아웃이 명시되어 있다. 참조 코드는 scale을 큰 tensor로 펼쳐 복호화한 뒤 matmul한다. 테스트와 benchmark의 shape/seed 목록이 구별되어 있다.
- 가져올 것: dtype만 읽지 말고 scale의 변화 축, stride, reduction 길이, 출력 레이아웃을 계약으로 정리한다. 공개 shape에 맞춘 타일 특화와 **입력값을 미리 안다는 가정**은 별개다. scale을 reduction 밖으로 옮길 수 있는지는 해당 축에서 상수인지 증명해야 한다.
- MOA 차이: MLP local scale은 `H/16` 또는 `L/16`마다 달라진다. 모든 scale을 GEMV 뒤에 한 번 곱하는 변경은 일반적으로 동치가 아니다. AMD의 FP8 instruction·block 크기·cache 최적값도 RNGD에 그대로 적용되지 않는다.

### S2. RadeonFlow: AMD Inference Sprint 수상자 공개 구현

- 자료: [작성자 저장소](https://github.com/RadeonFlow/RadeonFlow_Kernels), [TechnicalReport.md](https://github.com/RadeonFlow/RadeonFlow_Kernels/blob/main/TechnicalReport.md).
- 시점: FP8 GEMM/MoE/MLA 코드 공개 2025-06; 확인 2026-09-21.
- 저자 보고: MI300X와 ROCm 6.3.1에 맞춘 구현이며, PyTorch reference 대비 적어도 8배 가속이라고 설명한다. 비교 대상은 강하게 튜닝된 RNGD baseline이 아니다.
- 가져올 것: 메모리 전치·연속 전송, bank conflict 완화, 작은 출력과 긴 reduction에서 split-K, load/compute overlap을 검토한 순서가 유용하다. MOA에서는 이에 대응하는 DM/TRF 배치와 `DmaEngine`, `MainContext`, `SubContext` 점유율을 측정한다.
- 제한: GPU CU occupancy를 RNGD slice 배치와 동일시하지 않는다. 저자의 unsafe BF16 cast는 사용하지 않는다. MLA matrix absorption은 잠재 KV 구조를 전제로 하므로 Gemma의 dense projection을 임의로 줄일 근거가 아니다. CPU timing에서 효과가 난 graph/launch 최적화는 Stage 1 NPU cycle 점수와 분리한다.

### S3. MLSys 2026 FlashInfer 공식 대회와 Team Wombat

- 자료: [공식 수상 목록·평가 환경](https://mlsys26.flashinfer.ai/), [MoE 수상 보고서 PDF](https://github.com/flashinfer-ai/mlsys26-contest/blob/main/writeups/TeamWombat_MoE_AgentAssisted.pdf), [공개 구현](https://github.com/cheshire/flashinfer-challenge).
- 시점: 보고서 2026-04-25; 공식 시상 일정 2026-05-22; 확인 2026-09-21. PDF 본문을 내려받아 읽었다.
- 저자 보고: B200의 19 workload에서 FlashInfer 대비 기하평균 1.48배. 작은 token 수에는 fused kernel, 큰 경우에는 grouped GEMM 경로를 사용한다. 이 수치는 보고서 작성 시점의 저자 측정값이다.
- 가져올 것: workload별 병목을 나누고, 각 변경의 전체 결과표를 commit/실험에 연결하며, 실패한 큰 구조 변경도 기록한다. 공통 함수 변경으로 다른 workload가 퇴보했는지 별도로 본다. 한 번의 최저 latency를 선택하지 않고 독립 trial의 중앙값을 확인한다.
- 중요한 차이: 보고서 MoE의 완화된 정확도 gate는 `atol=1`, `rtol=0.3`, 일치율 90% 이상이다. MOA는 모든 원소와 모든 입력 구성이 통과해야 한다. 해당 정밀도 절충을 가져오면 안 된다. 공식 대회도 cloud 참고 측정과 bare-metal 최종 측정을 구분한다.

### S4. MLSys 2026 DSA 우승자 Dogacel

- 자료: [수상자 저장소](https://github.com/Dogacel/DeepSeek-Sparse-Attention-Kernels), [공식 보관 보고서 PDF](https://github.com/flashinfer-ai/mlsys26-contest/blob/main/writeups/Dogacel_DSA_AgentAssisted.pdf).
- 시점: MLSys 2026 대회; 확인 2026-09-21. PDF 본문을 내려받아 읽었다.
- 저자 보고: B200에서 평균 37.07배, DSA human-agent 부문 1위. 큰 기여 중 하나는 유효 token 수가 top-K보다 작을 때 모든 유효 token을 반환해도 정확하다는 **입력 크기에서 증명되는** fast path다. split 수와 warp 수를 전체 workload에 걸쳐 탐색했다.
- 가져올 것: 병목은 소스 연산 수뿐 아니라 실제 tensor 계약에서 찾는다. 긴 reduction의 분할과 재결합 비용을 함께 탐색한다. 컴파일 실패 한 번만으로 전략 자체가 불가능하다고 결론내리지 않는다.
- MOA 차이: 공개 fixture의 양수/특정 seed/RoPE 위치를 보고 연산을 생략하는 것은 구조적 동치와 다르다. benchmark 입력·정답을 하드코딩하는 실험은 제외한다. 저자의 attention 및 GPU atomic barrier 구현은 Stage 1에 직접 대응하지 않는다.

### S5. 국내 KCC 2026 온디바이스 AI 최적화 경진대회

- 자료: 공동 주최자 Nota의 [기술 회고](https://blog.nota.ai/kr/insights/llama-raspberry-pi-optimization).
- 시점: 대회 2026-06, 회고 2026-07-15; 확인 2026-09-21.
- 주최자 보고: 본선은 Raspberry Pi 5, Llama-3.2-1B, ExecuTorch/XNNPACK을 대상으로 했다. 재학습·모델 구조 변경은 금지하고 양자화·runner·커널 최적화는 허용했다. 우승 결과는 token 생성 8배 이상, TTFT 3.8배 개선이라고 설명한다. 원 측정 로그나 전체 수상 코드를 재현한 것은 아니다.
- 가져올 것: 검증된 실행 pipeline을 먼저 만든 뒤 설정별 실험을 표로 관리한다. 품질 손실을 각 단계에서 분리해 찾아내고 해당 단계만 복구한다. 실제 장치의 TTFT와 TPOT를 별도로 측정하는 관점은 Stage 2에 유용하다.
- MOA 차이: AWQ/GPTQ 재양자화부터 모델 export까지 바꿀 수 있던 국내 대회와, 고정된 입력·기존 양자화 tensor·device signature를 유지하는 Stage 1은 허용 범위가 다르다. 8배라는 숫자를 MOA 기대치로 쓰지 않는다.

### S6. 국내 LG Aimers 8기 EXAONE 모델 경량화

- 자료: [공식 과제 설명](https://dacon.io/competitions/official/236673/overview/description), [공식 평가](https://dacon.io/competitions/official/236673/overview/evaluation), [주최 측 GPTQ baseline 게시물](https://dacon.io/competitions/official/236673/codeshare/13800).
- 시점: 온라인 대회 2026-02, baseline 게시물 2026-01-28; 확인 2026-09-21.
- 확인한 사실: EXAONE-4.0-1.2B를 L4 GPU의 고정 vLLM 환경에서 평가하며 온라인 단계는 HF weights/config만 제출한다. 오프라인 단계는 vLLM 수정까지 확장한다. 평가용 token당 추론 시간과 제출 전체 실행 시간을 구분한다. GPTQ baseline 게시물은 확인했으나 다운로드 코드의 전체 내용은 열람하지 않았다.
- 가져올 것: 단계마다 수정 가능 범위를 새로 확인하고 점수에 들어가는 시간 구간을 먼저 정의한다. 실행이 쉬워진 것, 모델 파일이 작아진 것, kernel cycles가 줄어든 것을 별개 지표로 보관한다.
- MOA 차이: 여기서 유리했던 가중치 경량화/추가 학습을 MOA Stage 1의 고정 수학 연산 대신 적용할 수 없다. vLLM과 CUDA를 RNGD Rust device pipeline에 추가하는 것이 해법도 아니다.

### S7. FlashAttention 원 논문

- 자료: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
- 시점: 최초 2022-05-27, v2 2022-06-23; 확인 2026-09-21.
- 확인한 핵심: 수학적 attention을 유지하면서 tile과 on-chip state로 HBM 왕복을 줄이는 접근이다. FLOP 수만 줄이는 것과 실제 시간을 줄이는 것은 같지 않다.
- 가져올 것: **중간 tensor materialization을 어느 메모리 계층에서 제거하는가**라는 관점을 projection과 MLP에도 적용한다. HBM→DM 비용을 줄였는지, DM에 쓰고 다시 읽는 구간을 줄였는지를 분리한다.
- 적용 단계: Stage 1은 이 IO 관점만 사용한다. online softmax와 KV page 처리 자체는 Stage 2 attention 후보이며, 실제 입력 길이와 KV layout을 확인한 다음 평가한다. 논문의 학습 benchmark 가속률로 decode의 개선 폭을 예상하지 않는다.

### S8. MLPerf Inference 공식 제출·검증 방법

- 자료: [MLCommons Inference Submission Guide](https://docs.mlcommons.org/inference/submission/).
- 시점: 계속 갱신되는 문서; 확인 2026-09-21.
- 확인한 핵심: 같은 모델·reference를 비교하는 Closed와 모델 변경까지 허용하는 Open을 구분한다. 시스템 정보, accuracy 검사, scenario별 측정, 결과 구조와 submission checker를 함께 관리한다.
- 가져올 것: 성공 로그 한 줄이 아니라 소스/도구 버전, 입력, 정확도, 성능, 실행 환경을 묶어 결과 하나로 보관한다. latency와 throughput, 초기화와 steady state를 혼합하지 않는다.
- MOA 차이: MLPerf의 runner나 정확도 임계값을 가져오는 것이 아니다. 공식 MOA harness와 채점 정의를 그대로 두고 실험 증거를 관리하는 원칙만 사용한다.

이번 검색에서는 위 자료보다 MOA에 직접 맞는 **기존 국내 RNGD Rust kernel 대회의 재현 가능한 수상 코드**를 확인하지 못했다. 회로 설계 대회나 WARBOY의 ONNX 지원 사례는 RNGD decoder kernel 수상 전략으로 인용하지 않았다.

## 3. 실험 후보 순서

아래는 소스에서 발견한 비용 구조에 근거한 **가설**이다. 이 문서 작성 과정에서는 커널을 수정하거나 새 하드웨어 job을 제출하지 않았다. 실제 채택 여부는 Arena의 정확도·cycles로 결정한다.

| 순서 | 변경 가설 | 현재 구현의 근거 | 성공 판단 및 중단 조건 |
|---|---|---|---|
| E0 | 세 커널의 정적 schedule과 RNGD 결과를 같은 source/toolchain으로 고정 | 기존 측정은 job 69783 한 회; `OPTIMIZATION.md`는 static plan과 실측 구분을 요구 | 각 입력별 cycles·정확도, binary/fixture hash, schedule bottleneck을 하나의 record로 재현 |
| E1 | QKV/O broadcast를 consumer의 실제 mapping에 맞춰 전송하거나 switch 범위를 줄임 | `layout::broadcast_hidden`과 `broadcast_sliding_heads`는 `CustomBroadcast { ring_size: 256 }`; `decoder_feedforward`는 이미 `to_dm`를 사용 | 동일한 논리 tensor와 모든 element 유지. broadcast MainContext 시간 및 총 cycles 감소. 단순 치환이 DMA를 악화시키면 mapping을 다시 설계 |
| E2 | O projection에서 FP8 table lookup과 BF16 contraction 사이의 DM materialization을 줄임 | `output_partial`은 FP8→BF16 변환을 `.commit()`한 뒤 다시 `.begin(weight.view())`로 읽음 | API가 해당 stream 조합을 허용하는지 먼저 작은 compile로 확인. 실제 DM traffic·makespan·RNGD가 모두 유리한 후보만 유지 |
| E3 | O projection의 4×1024 partial을 2×2048 또는 다른 적합한 tile로 재구성 | `project_output`은 네 partial과 세 `add_partials`를 사용; 현재 각각 BF16로 round됨 | compile-time mapping/용량을 만족하는 후보만. reduction tree와 BF16 rounding 변화가 `atol=0.05, rtol=0.01` 전 원소 gate를 통과해야 함 |
| E4 | MLP FP4 unpack→scale→BF16 tile 생성→수축에서 불필요한 저장·재배치 제거 | `project_up_and_gate`는 전체 FP4→FP8 변환 뒤 4행씩 재변환; `project_down`은 30 pass×2 half에서 2행 복호화와 `to_dm` 재배치 | 먼저 down/up/gate의 static critical path를 분리. local scale 축과 기존 round 지점 유지 가능성을 확인. 총 MLP cycles와 DM peak lifetime 함께 감소 |
| E5 | MLP row tile과 down reduction partition을 소수 후보로 탐색 | up/gate는 slice당 60행 중 4행씩; down은 120행 중 4행씩, 열 1920개 partition | 4행만을 정답으로 가정하지 않되 60/120 분할의 나머지·패딩과 capacity를 확인. compile 실패는 shape/stream/용량 원인을 기록. 작은/큰 tile의 총 비용을 비교 |
| E6 | normalization·residual·layer gate의 adjacent stream을 융합하고 round-trip 제거 | `ops.rs`에서 별도 helper로 연결되며 shared helper는 다른 model 경로도 사용 | source 함수 수 감소만으로 인정하지 않음. round/epsilon 순서 보존, 다른 세 kernel 회귀 없음, 필요시 full/vision/audio 영향도 확인 |
| E7 | Stage 2에서 KV 관리, attention tiling, host-device boundary와 model-serving 직렬 구간 최적화 | Stage 1 밖인 `sliding_attention`, full attention page, `src/host/runtime.rs`가 별도 존재 | Stage 2 workload·정확도·E2E metric 확정 후 프로파일. TTFT/TPOT/초기화와 NPU 시간을 나눠 판단 |

E1~E3은 작은 수정으로 두 개 이상의 측정 경로를 이해하거나 개선할 수 있어 먼저 확인한다. E4~E5는 구현 부담이 크지만 FFN에서 큰 여지를 검증할 수 있다. 스케줄이 다른 병목을 보이면 표의 순서를 바꾼다. double buffering, fusion, 큰 tile이 항상 빠르다는 가정은 두지 않는다. 추가 buffer가 TRF/DM capacity와 스케줄 중첩을 악화시킬 수 있다.

## 4. 재현 가능한 비교 방식

1. 공식 baseline source와 커널별 원본 schedule을 보존한다. compiler, Rust, crate lockfile, fixture hash, binary hash, Git commit/patch hash를 결과에 연결한다.
2. 후보 하나당 한 가설만 바꾼다. 먼저 해당 커널의 schedule을 `--exact`로 다시 생성해 makespan·context 점유·주요 DMA와 tensor lifetime을 비교한다. static 감소는 후보 선별 자료이고 성능 확정이 아니다.
3. 공식 runner의 세 입력 구성과 모든 출력 비교를 모두 통과시킨다. `src/bin/test_kernels.rs::compare`의 조건은 `abs(actual-expected) <= atol + rtol*abs(expected)`이며 모든 원소가 통과해야 한다. 허용 오차는 QKV 0.04, output 0.05, FFN 0.01이고 공통 rtol은 0.01이다.
4. 성능 기록에는 세 구성의 개별 cycles와 중앙값을 함께 남긴다. **세 입력 구성의 중앙값은 동일 입력을 반복한 noise 추정이 아니다.** 개선 폭이 작거나 분산이 의심될 때만 같은 artifact의 baseline→candidate→baseline을 추가 실행해 환경 변동과 입력 차이를 분리한다.
5. compile 실패, timeout, correctness 실패도 실험표에 보관하되 speedup은 부여하지 않는다. 한 구성만 통과하거나 좋은 job 한 개만 선택한 결과를 채택하지 않는다.
6. 새 후보가 통과하면 `(qkv, output, ffn)` speedup vector와 기하평균을 함께 보고한다. 개선되지 않은 커널이나 회귀를 숨기지 않는다. 공식 제출 결과를 얻으면 Arena 참고 결과와 별도 열로 관리한다.
7. 일반성 검사가 필요하면 별도의 개발 harness/fixture로 보강한다. 공식 grader, tolerance, PRNG, 제출 범위 자체를 바꾸어 성능을 만든 것으로 간주하지 않는다. 개발용 검사는 제출 시 고정되는 공식 harness를 대체하지 않는다.

실험표 최소 필드:

```text
experiment_id, hypothesis, source_commit_or_patch_hash, compiler_version,
fixture_hash, binary_hash, schedule_hash, arena_job_id,
correctness_per_output_per_config, cycles_per_kernel_per_config,
median_cycles, speedup_vector, geometric_mean, result, explanation
```

## 5. 채택하지 않을 지름길

- fixture seed·값·예상 정답에 맞춰 계산을 생략하거나 외부 정답을 내장하는 변경.
- 특정 offset에만 맞는 KV scatter, 다른 cache 영역을 손상시키는 write, non-finite 결과를 숨기는 검사 수정.
- `src/axes.rs`의 shape 축소, device function signature 변경, host 전처리로 Stage 1의 입력 계약 변경.
- 원소 일부를 허용 오차 밖으로 보내는 저정밀화, 다른 대회의 일치율 90% 기준을 MOA에 적용하는 것.
- 수학적으로 실수 범위에서 같다는 이유만으로 BF16 rounding·reduction 순서·scale 위치를 바꾸고 검증을 생략하는 것.
- NPU cycles만 채점하는 단계에서 tokenizer, Docker startup, 네트워크 upload, Python launcher 시간을 줄인 것을 점수 향상으로 보고하는 것.

## 6. 상위권을 향한 의사결정 기준

우선 유효한 기준 제출과 재현 가능한 세 커널 측정값을 확보한다. 그다음 현재 3위 점수 `S3`와 우리 공식 점수 `S` 사이의 격차 `log(S3)-log(S)`를 실제 실험의 로그 개선량과 비교한다. 예측치에는 측정 오차와 앞으로 경쟁자가 개선할 여지를 별도로 둔다. 3위가 확인되지 않거나 제출 서버 결과가 없으면 순위 달성을 주장하지 않는다.

일정 후반에는 검증된 개선을 보존하고, 큰 구조 변경은 독립 실험으로 유지한다. 빠른 코드 한 개보다 **왜 빨라졌는지 설명할 수 있는 schedule, 모든 구성의 정확도, 동일 artifact의 RNGD 결과, 제출 가능한 소스**가 함께 있는 후보를 최종 선택한다.
