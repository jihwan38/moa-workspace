# O projection ingress: primary literature audit (2026-09-27)

현재 shortlist는 **행쌍을 보존하는 4-way row interleave로 producer/consumer를 함께 변경하고 기존 Switch 비용을 제거하는 설계**와 **C09의 첫 NPU evidence를 얻는 제한 진단**이다. 더 넓은 대안으로 ingress·출력 소유권·local 누산을 함께 바꾸는 전체 경로를 남긴다. 논문에서 반복되는 double buffering과 dequant fusion은 이미 일부 구현되어 있으므로 그 이름만으로 신규 후보가 되지 않는다. 이번 작업은 연구만 수행했으며 compile/Arena/source 수정은 없다.

## 고정 조건과 관측의 의미

Gemma O는 Hout=3840, K=4096, batch1이다. 원본 weight는 row-major FP8 E4M3 15,728,640 B이고 x는 BF16이다. 현재 2 clusters × 16 output groups × 16 K columns에서 slice당120×256 weight를 처리한다. FP8 DM → lookup BF16 → BF16 TRF dot → F32 column reduce → channel scale → BF16가 산술 경계다. 현재 E122에서도 비교 대상 O image는 E104와 같다.

weight marker 약26.7k, dot/reduce 약12k, tail 약9.4k와 weight 종료+2에 시작하는 dot은 **weight 완료 dependency를 확인**한다. marker는 wait를 포함한다. 26.7k를 실제15.7 MB 전송 service time으로 간주하거나 각 구간을 합산해 가능한 speedup을 계산하지 않는다. 모든 원본 weight를 한 번 읽는 비용은 유지된다. 목표는 유용한 일부가 도착했을 때 계산을 시작하고, 읽기 순서/목적지와 추가 작업을 함께 개선하는 것이다.

현재 byte 수만의 계산으로 약2 FLOPs/weight-byte(곱셈과 덧셈을 각각1 FLOP로 계산)이지만, 이것으로 attained HBM bandwidth나 원인을 판정하지 않았다. 공식 광고 bandwidth와 CPU/GPU/다른 NPU 결과를 RNGD 예측에 사용하지 않았다.

## 출처별 메커니즘과 적용 경계

### R1. [Furiosa Opt — Memory performance](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html)

출판: 페이지에 날짜 없음. 확인 버전: rolling official documentation; no publication date shown. 직접 확인일: 2026-09-27. 절: Memory organization; HBM performance / address mapping; Bank starvation.

**확인한 메커니즘:** 256 B HBM packet 정렬, address XOR에 따른 channel 분산, bit 8에 따른 stack 선택, 연속 접근과 DM 목적지 분산이 전송 효율을 좌우한다. DM 접근 우선순위와 bank starvation 조건도 설명한다.

**O 대응:** O는 4096 B row pitch와 256 B K chunk를 갖는다. 단순 정렬은 이미 좋은 조건이다. emitted descriptor의 실제 source 주소·루프 순서와 slice/DMN 목적지를 함께 열거하여 두 stack 및 channel 분포를 확인할 근거다.

**추가 비용/재개 조건:** 추가 weight 복사 없는 주소/소유권 재배치가 우선. row-major를 offline 변경할 수 없으므로 runtime transpose를 쓰면 원본 읽기+변환 DM write/read가 추가된다. 문서가 말하는 제약은 원인 후보이며 현재 O bank starvation 증거가 아니다.

**한계:** 공개 rolling 문서를 설치 SDK 0.8.1 동작 보증으로 취급하지 않는다. 문서 요약의 two-DMN 표현과 geometry의 eight DMNs/cluster는 일관되지 않으므로 정확 topology는 실제 descriptor/설치 자료로 확인한다. 광고 bandwidth를 attained bandwidth나 시간 하한으로 사용하지 않았다.

### R2. [Furiosa Opt — Schedule](https://developer.furiosa.ai/furiosa-opt/book/scheduling/schedule.html)

출판: 페이지에 날짜 없음. 확인 버전: rolling official documentation; no publication date shown. 직접 확인일: 2026-09-27. 절: Operation contexts; Resource and bank contention; Operation order and dependencies; Example: Scheduling of an MNIST model.

**확인한 메커니즘:** Main/Sub/DMA는 독립 자원·메모리 dependency 조건에서 겹친다. 같은 context 순서, RAW/WAR/WAW 및 공유 VectorEngine이 중첩을 제한한다. 특정 TU bank-access 제약은 compiler가 DMA와 겹치지 않게 일정을 잡는 원인이 될 수 있다.

**O 대응:** weight를 여러 buffer에 나누는 설계는 source 문장 순서가 아니라 첫 buffer ready→dot, 다음 buffer DMA와의 dependency 분리 및 shared resource를 확인해야 한다.

**추가 비용/재개 조건:** C09는 이미 실제 static overlap 1825 cycles를 만들었다. 단순 double buffering은 신규 근거가 아니다. task 수, lookup 준비, DM bank 사용까지 바뀌어야 재개 가치가 있다. C12 Main/Sub 분리도 다음 weight가 늦어 의도한 overlap이 0이었다.

**한계:** 문서 일반 규칙만으로 현재 O의 26.7k marker를 DMA service 또는 bank stall로 분해할 수 없다. engine marker는 wait를 포함한다.

### R3. [Mapping Gemma3 onto an Edge Dataflow Architecture](https://arxiv.org/html/2602.06063v2)

출판: 2026-01-27. 확인 버전: arXiv v2, 2026-02-24; v1 also inspected. 직접 확인일: 2026-09-27. 절: 3.1.1 Dequantization Engine (Q4NX format); 3.2.1 FusedDQP; 3.2.3 Efficient Mapping to the NPU.

**확인한 메커니즘:** FusedDQP는 32×256 weight block을 16×8 subblock으로 나누어 dequant와 GEMV를 같은 compute tile에서 처리한다. activation을 broadcast하고 각 tile이 담당 출력의 K 축을 순차 누산한다.

**O 대응:** 출력 소유권과 K 누산을 함께 설계하고 변환된 BF16 weight의 별도 DM materialization을 피하는 모델이다. 현재 O도 이미 lookup→BF16 dot을 같은 Main 경로에서 수행하므로 fusion 자체는 구현된 성질이다.

**추가 비용/재개 조건:** 논문의 Q4NX INT4+group metadata layout은 FP8 E4M3 row-major와 다르다. 우리에게 새로 가능한 부분은 owner-local accumulation과 실제 ingress 순서의 조합이다. full-K activation 복제, local accumulation 형태 및 최종 epilogue 재배치 비용을 포함해야 한다.

**한계:** AMD AIE tile/DMA/stream API는 RNGD에 이식되지 않는다. Gemma3 end-to-end 수치를 Gemma O kernel gain으로 환산하지 않는다. Q4NX로 재양자화하는 것은 원본 numerical/input 계약을 유지하는 변환이 아니다.

### R4. [JAX Pallas TPU — Matrix Multiplication](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)

출판: 페이지에 날짜 없음. 확인 버전: rolling official JAX documentation; fetched 2026-09-27. 직접 확인일: 2026-09-27. 절: Your first matrix multiplication kernel; bfloat16 matrix multiplication; Performance of pipelined kernels.

**확인한 메커니즘:** K 축을 순차 처리하며 FP32 scratch accumulator를 유지하고 마지막 단계에만 출력 dtype으로 내린다. block별 데이터 이동/계산 비율, on-chip capacity, pipeline 첫/끝 bubble이 block 크기를 제한한다.

**O 대응:** O의 K-stage 설계는 BF16 partial spill 없이 F32 누산을 유지해야 한다. whole matrix의 산술량 대신 각 stage가 다음 transfer를 덮을 만큼의 유효 작업을 갖는지 판단해야 한다.

**추가 비용/재개 조건:** 작은 stage는 더 많은 DMA/task/setup/partial write-read를 만들 수 있다. C09의 두 stage도 추가비용을 냈으므로 tile 수를 늘리는 것만으로 정당화되지 않는다. K 누산을 늘리면 accumulator lifetime과 활성 x tile 교체도 비용이다.

**한계:** 예제는 TPU GEMM이며 batch1 RNGD DPE의 shape support 증거가 아니다. TPU 수치와 block-sweep 방법을 RNGD 성능 추정이나 무작위 sweep 근거로 사용하지 않는다. TPU Pipelining 페이지는 초기 fetch 후 재시도 실패가 있어 인용의 주 근거는 성공적으로 읽은 matmul 문서다.

### R5. [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models — author implementation](https://github.com/IST-DASLab/marlin)

출판: 2024-08-21. 확인 버전: paper v1; mutable author README inspected 2026-09-27. 직접 확인일: 2026-09-27. 절: README Techniques.

**확인한 메커니즘:** 비동기 weight loads, activation 재사용, shared-memory double buffering, dequant/compute instruction 교차 실행을 조합한다. weight와 scale을 offline reshape해 Tensor Core 소비 순서에 맞추며, striped work partition은 load balance와 global reduction을 함께 최적화한다.

**O 대응:** 단일-token weight는 재사용이 거의 없고 x는 작다는 비대칭이 중요하다. FP8을 compact하게 가져오고 소비 직전에 변환하되 여러 단계를 통해 full BF16 weight를 왕복시키지 않는 방향이 맞는다.

**추가 비용/재개 조건:** 우리 입력은 사전 재배열할 수 없다. Marlin의 prepack을 흉내 내면 15.7 MB 원본 read와 runtime pack 비용이 반드시 붙는다. cache eviction 정책/CUDA async copy/warp striped scheduler는 SDK에서 같은 control이 확인되지 않았다.

**한계:** FP16×INT4와 FP8 E4M3×BF16은 다른 경로다. README의 global partial FP16 downcast는 우리 F32 누산→scale→BF16 경계를 그대로 만족하지 않는다. GPU의 4×는 적용할 수 없다.

### R6. [Fast Matrix Multiplications for Lookup Table-Quantized LLMs (FLUTE)](https://arxiv.org/html/2407.10960v4)

출판: 2024-07-15. 확인 버전: arXiv v4, 2025-01-17; v2 also inspected. 직접 확인일: 2026-09-27. 절: 3.1 Offline Matrix Restructuring; 3.2 Vectorized Lookup in Shared Memory; 3.3 Stream-K Workload Partitioning.

**확인한 메커니즘:** offline weight reorder, pair-index vector LUT, bank별 LUT duplication, Stream-K load balancing을 결합한다. lookup table을 kernel loop 밖에서 준비하고 stage마다 weight를 공급한다.

**O 대응:** table lifetime과 staged ingress를 공동 설계해야 한다는 근거다. 그러나 SDK fetch_table_lookup는 공개 table/handle 인자가 없고 compiler가 변환 table을 생성한다. 외부에서 FLUTE식 prepared table을 주입할 수 있다는 증거는 없다.

**추가 비용/재개 조건:** 4-bit pair table과 달리 FP8 pair table은 2^16×4 B=256 KiB/복제본이라는 계산이 된다. 현 SDK의 실제 table 형태와 별개다. 따라서 FP4 pair-LUT 성공을 O FP8에 옮길 근거가 아니다. staged 호출의 table 준비를 늘리면 C09 비용이 재발할 수 있다.

**한계:** 원래 Stream-K는 GPU SM wave imbalance를 줄인다. 현재 O는 512 slices에 균일 logical payload를 주므로 동일 imbalance 증거가 없다. FLUTE의 global FP16 partial reduction도 우리 numerical 경계와 다르다. free prepack은 허용되지 않는다.

### R7. [T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup](https://arxiv.org/html/2511.11248v1)

출판: 2025-11-14. 확인 버전: arXiv v1. 직접 확인일: 2026-09-27. 절: 4.2 Pipelining to Hide Dequantization Overhead; 4.3 Mapping of LUT-based Decoding on NPU; 5 Implementation: Graph optimization / Efficient table look-up by VLUT / Asynchronous DMA.

**확인한 메커니즘:** Qualcomm HVX VLUT16 기반 activation-dependent lookup, register 수를 고려한 두 수준 tiling, TCM spill buffer, asynchronous DMA를 결합한다. 공통 activation을 쓰는 projection은 table precompute를 공유한다.

**O 대응:** runtime LUT는 고정 FP8→BF16 cast table과 다른 계산이다. O에는 Q/K/V 같은 다중 projection 공유도 없다. 더 큰 tile의 register spill과 local memory 비용을 함께 계산한다는 교훈만 바로 적용된다.

**추가 비용/재개 조건:** SDK 0.8.1에는 runtime LUT 인자/handle이 확인되지 않았다. INT2/INT4 bit-serial 표를 E4M3의 signed exponent/mantissa 의미에 맞게 바꾸는 작업과 BF16 입력 정확도 검증이 필요하다. 실행가능 primitive 증거가 없으므로 우선 후보로 삼지 않는다.

**한계:** HVX DMA bandwidth, VLUT throughput, phone NPU speedup을 RNGD로 환산하지 않는다. fixed conversion lookup API가 있다는 사실은 activation-dependent LUT 지원을 뜻하지 않는다.

### R8. [TileFuse: A Fused Mixed-Precision Kernel Library for Efficient Quantized LLM Inference on AMD NPUs](https://arxiv.org/html/2606.11357v2)

출판: 2026-06-09. 확인 버전: arXiv v2, 2026-07-01. 직접 확인일: 2026-09-27. 절: 4.2 Weight Pre-tiling; 4.3 Mixed-precision Microkernels; 4.4 Dataflow for Weight Distribution.

**확인한 메커니즘:** weight·metadata를 offline pre-tile하고 compute core에서 dequant+GEMV를 수행한다. memory core가 연속 weight bundle을 여러 compute rows에 분배한다. 각 core가 자신의 출력을 끝까지 누산하며 memory core는 부분합 reduction을 하지 않는다.

**O 대응:** O에서 ingress layout, 출력 소유권, K reduction 범위를 한 설계로 묶어 비교할 직접적인 NPU 근거다. 현재 16-column partial reduction을 유지할 필연성은 없지만 새 owner 배치의 producer/consumer 비용까지 계산해야 한다.

**추가 비용/재개 조건:** 논문은 offline pre-tiling에 runtime 비용이 없다고 명시한다. MOA에서 same benefit은 공짜가 아니다. HBM→compact DM→새 배치 copy는 all-weight barrier와 DM read/write를 추가할 수 있으며 E124 joint QKV의 실제 회귀가 그 위험을 보여준다.

**한계:** TileFuse의 기본 GEMV는 32개 중 8개 core만 사용하던 경로를 32개로 바꾼다. 우리 O는 이미 512 slice에 작업을 분배하므로 4× utilization 회복 논리가 그대로 적용되지 않는다. W8도 INT8 asymmetric이므로 FP8 E4M3 cast와 동일하지 않다.

### R9. [Accelerating LLM Inference with GemLite, TorchAO and SGLang](https://pytorch.org/blog/accelerating-llm-inference/)

출판: 2025-01-21. 확인 버전: page also displays May 5th, 2025 update; primary teams PyTorch/Mobius Labs/SGLang. 직접 확인일: 2026-09-27. 절: 2. GemLite: Kernel Development — Kernel Selection; Single Sample Inference; Batched Inference; Maximizing High Performance: Key Implementation Insights.

**확인한 메커니즘:** batch1에서 GEMV, moderate batch에서 GEMM Split-K를 구분한다. packed quantization에서는 두 연속 block의 scale/zero 재사용을 위해 GEMV_REVSPLITK를 사용한다. non-packed data는 GEMV_SPLITK로 K를 순회하며 tl.dot 없이 계산한다.

**O 대응:** 우리 FP8은 non-packed이며 scale은 output-channel별이다. packed INT4의 metadata reuse 이득을 기대할 수 없다. K 병렬도는 최대치가 목표가 아니라 유효 parallelism·부분합 통신·activation 복제를 맞추는 선택이다.

**추가 비용/재개 조건:** 현재 16 K partitions를 줄이는 owner-local 설계는 F32 cross-slice reduction을 줄이지만 slice별 dot 길이와 x 복제/재공급을 늘린다. 이 비용 변화를 함께 설계해야 하며 단순 split count sweep로 제안하지 않는다.

**한계:** GPU CUDA cores/tl.sum와 RNGD DPE는 다른 execution unit이다. tl.dot을 쓰지 않는다는 GPU 결론은 RNGD contraction을 vector multiply로 바꾸면 빠르다는 증거가 아니다. 보고된 GPU end-to-end percentages는 예측에 사용하지 않았다.

### R10. [FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/html/2311.01282v4)

출판: 2023-11-02. 확인 버전: arXiv v4, 2024-01-05. 직접 확인일: 2026-09-27. 절: 4 Flat GEMM Optimization with Double Buffering; 5 Heuristic Dataflow with Hardware Resource Adaption.

**확인한 메커니즘:** small-M flat GEMM의 padding을 줄이고, N tiling의 parallelism 대 memory ratio tradeoff를 분석한다. 두 shared buffers로 다음 K tile 공급을 compute와 겹치며 shape별로 GEMV/flat GEMM/일반 GEMM 구현을 선택한다.

**O 대응:** 현재 O delayed dot은 전체 weight 완료 경계를 갖는다. stage별 공급을 조기에 소비하는 경로가 관련되지만, 실제 병목 shape에서 효과가 있다는 조건을 요구한다. O와 FFN을 동일 타일 정책으로 처리할 이유가 없다.

**추가 비용/재개 조건:** C09가 이미 두 buffer로 DMA/dot overlap을 얻었다. 새 경로는 반복 decode/setup 또는 소유권·부분합 비용 중 하나를 구체적으로 바꿔야 한다. 더 작은 stage는 launch/setup와 fill/drain 비용을 늘린다.

**한계:** 논문의 M64→M8 Tensor Core padding 제거는 현재 RNGD batch1 DPE 경로에 존재하는 문제라고 확인되지 않았다. 논문에서조차 batch1과 batch4 선택이 달라 GPU tensor-core 결과를 직접 이식할 수 없다.

## 실패 prior와 혼동하면 안 되는 점

- **C09** — Two local 60-row tiles already overlap first dot and second weight DMA by 1825 static cycles. Static makespan 27239→27432 (+0.71%), instructions32→49, DMA loads6→8. No NPU screen. This is a local staging prototype, not proof all pipelines regress. [로컬 근거](/home/jihwan38/moa-workspace/moa-dev/experiments/candidates/c09-output-pipeline/README.md)
- **C12** — C09 stages + Main dot/Sub reduction + Main packing. Intended Sub reduction/next Main dot overlap0; second weight arrives4500 static cycles after first Sub reduction. NPU79329:50555→51862→50680, +2.46% bracket regression. Moving only reduction does not fix ingress wait. [로컬 근거](/home/jihwan38/moa-workspace/moa-dev/experiments/candidates/c12-output-main-sub/README.md)
- **C10** — QKV lookup→weight TRF StoTrf failed target lowering. O was unchanged. Relevant only if a new O design requires the same unsupported route; not an O ownership/performance rejection. [로컬 근거](/home/jihwan38/moa-workspace/moa-dev/experiments/candidates/c10-direct-weight-trf/README.md)
- **E124** — Corrected joint QKV uses compact read→explicit packet copy; 45 comparisons PASS, QKV +24.41% bracket regression. Extra DM traffic/all-weight dependency are structural costs, but no candidate trace attributes exact cycles to either. Original C08 DMA bounds bug means old accuracy failure was not general architecture evidence. [로컬 근거](/home/jihwan38/moa-workspace/docs/experiments/e124-c08-corrected-joint.json)
- **LUT API** — fetch_table_lookup has no public runtime table/handle parameter. Frozen QKV graph had4096 B constant and StoTab output8388608 B aggregate/16384 B per physical slice. Those exact QKV counts are not claimed as a newly measured O table extent. [로컬 근거](/home/jihwan38/moa-workspace/docs/research/fp8-lookup-table-sharing-audit.md)
- **O row4** — E24 native FP8 hi/lo, K8, BF16 output Switch. Static32252→34435(+6.77%), weight DMA13512 unchanged, Main+2183 from Switch. No NPU run. Pair-preserving F32 gather redesign changes this exact added cost; it does not prove row4 will improve actual HBM service. [로컬 근거](/home/jihwan38/moa-workspace/moa-dev/experiments/candidates/output-fp8-row-interleave-four/README.md)

## 다음 설계 판단에 쓸 순서

### 1. 행쌍을 보존한 4-way O row interleave: 기존 family를 구체적 비용 변경으로 재개

범위: first bounded diagnostic candidate; producer/consumer layout co-design hypothesis, not a predicted winning kernel. 근거: R1, R2, R3, R8.

h=1920c+480g+8t+2r+j, s=64g+16r+q, k=256q+u; c<2,g<4,t<60,r<4,j<2,q<16,u<256. K16 reduction을 유지하고 F32 인접 두행8B packet을 canonical gather로 보내는 경로를 함께 설계한다. lookup/dot의 산술이나 BF16 경계를 바꾸지 않으며 weight row-major 원본을 runtime prepack하지 않는 표현이 목표다.

검증 조건: 예상 gather packet 증가 때문에 추가 stage가 없어도 성공을 예측하지 않는다. 먼저 producer 개선과 consumer penalty를 함께 분리하는 bounded diagnostic으로 판단한다. 좌표상 인접 r owner의 첫 행 간격은8192 B가 되지만 t 증가는32768 B다. 어떤 축이 실제 HBM bus에서 먼저 issue되는지는 emitted descriptor/ISA semantics가 필요하다. q 및256 B chunk는 그대로이므로 aligned source 조건과 stack bit8 선택은 이 수식 자체로 개선되지 않는다. 큰 source stride 감소는 channel/bank conflict 감소의 증명이 아니다. F32 두행8B canonical gather가 별도 BF16 Switch 없이 정확히 표현되는지 확인해야 한다.

prior와 차이: 예전 output-fp8-row-interleave-four는 E24 native hi/lo, K8, BF16 output Switch였다. 이번 K16·행쌍 보존은 그 Switch 추가비용을 제거하는 다른 producer/consumer 계약이다. family 전체가 새롭다고 주장하지 않는다.

추가 비용: 복구 gather/typed DMA descriptor의 source pitch와 destination ownership; consumer의 새로운 local row 순서와 기존 scale/RMS의 canonical row 대응; 원래 연속 row traversal 대신 pair 단위 interleave가 만든 local 주소 간격; root exact-graph audit: baseline F32 gather는32개의480 B transfer packet이고 새 direct F32 gather는1920개의8 B packet 예상. 총15360 B는 같아도 packet 처리비용이 커질 수 있다. 이 수는 HBM bus transaction 실측이 아니다.

### 2. C09의 첫 NPU evidence 진단을 별도로 분류

범위: bounded diagnostic option; not a new architecture; not executed. 근거: R2, R4, R10.

C09 two-stage는 static +0.71% 때문에 NPU 미실행이다. 이미1825 static cycles의 DMA/dot overlap을 확인했으므로 정확한 frozen source/descriptor contract 아래 첫 NPU 측정이 static 모델과 실제 weight-wait 변화를 판별할 수 있다.

검증 조건: 기존 logical coverage만으로 correctness를 주장하지 않는다. 먼저 실제 DMA bounds/stride를 확인한다. 결과가 없는데 regression이라고 부르거나 같은 helper를 새 candidate로 재명명하지 않는다. frozen prototype의 가치와 table/setup 비용을 바꾼 신규 설계의 가치는 분리한다.

prior와 차이: 새 기법이 아니라 static-only closure의 NPU 증거 공백을 메우는 제한된 진단이다.

### 3. Inspect the O ingress address stream and destination ownership before another tile proposal

범위: read-only diagnostic design. 근거: R1, R2.

Use emitted descriptor source extents/pitch/offset, transfer ordering, destination slice address, and decoder-supported loop semantics. Separate transaction alignment from channel/stack/DMN distribution. Keep original FP8 bytes and one logical read.

검증 조건: Do not claim hardware bus ordering from descriptor nesting unless SDK/ISA semantics establish it. An aligned256 B logical chunk does not prove balanced aggregate issue pattern. No new bandwidth number or starvation diagnosis.

prior와 차이: Targets actual ingress mapping rather than re-running C09 staging; current marker establishes dependency location, not service cause.

### 4. Output-owned streaming architecture, activation-resident TRF, local F32 accumulation

범위: whole-projection architecture hypothesis, not implemented. 근거: R3, R8, R9.

Assign fewer K partitions/more output owners together with contiguous HBM weight bundles and per-owner local F32 accumulation. Keep compact FP8 weight until direct lookup/dot consumption; preserve x as TRF operand to avoid C10 weight-to-TRF route. Finish scale before BF16; remap final output into existing epilogue contract.

검증 조건: Check full graph/API/layout feasibility and byte-exact descriptors. Require a changed cost mechanism relative C09/C12; keep F32 partials; no imported GPU speedup. Whole output-owned arithmetic/support remains unproved.

prior와 차이: Changes row/column ownership, partial reduction, activation supply and ingress layout together. Fewer K splits by itself is only parameter tuning.

추가 비용: More activation replication or streamed x updates; Longer local K accumulation / possible DPE lane inefficiency; Different consumer traversal and final gather; If full-K owned rows are uneven across512 slices, output balancing or idle slices; Any runtime weight packing must add its full read/write costs.

### 5. Staged ingress only with explicit decode/setup amortization or changed ownership

범위: conditional architecture component. 근거: R2, R4, R5, R6, R10.

Allow a complete first consumer region to start without waiting for whole15.7 MB, while later independent buffers fill. Coupled with owner-local execution or proven table lifetime reuse, not repeated clone of current helper.

검증 조건: Public prepared-table handle is absent; shared table cannot be assumed. Retain only if emitted graph shows a concrete lower-cost lifetime or another decode path pays its complete conversion cost. C09 static small regression is not NPU universal rejection, but is sufficient reason not to rename it as new.

prior와 차이: Same helper staged again is not a new family; requires a demonstrated cost change.

추가 비용: Extra descriptors/instructions and buffer lifetimes; Potential repeated StoTab/lookup preparation; Partial accumulator traffic and fill/drain bubbles; Main/Sub/DMA bank contention.

## 설계의 수량화 예시와 금지한 추론

다음은 실행 추천이나 성능 예측이 아닌 ownership 비용의 산술 예시다. 16 K columns ×32 global output groups를 8 K columns ×64 output groups로 바꾸면 slice당120×256이60×512가 되어 logical weight payload는30,720 B로 동일하다. F32 partial element 수는3840×16에서3840×8로 반감하지만, 각 output group에 K activation을 공급한다면 logical activation replication은 두 배가 된다. 실제 padded bytes, TRF layout, DPE lane utilization, HBM issue ordering은 emitted graph 없이는 확정할 수 없다. 따라서 이 비율 변경만으로 후보를 내거나 gain을 약속하지 않는다.

더 과감한 full-K output ownership은 cross-slice sum을 없앨 수 있지만3840/512=7.5의 uneven rows, full activation 공급, 긴 local accumulator lifetime, 최종 gather를 함께 해결해야 한다. 이는 GPU Split-K 숫자를 그대로 옮기는 문제가 아니다. 현재 경로의 weight TRF 전환을 필수로 두지 말고 activation-resident TRF라는 대안도 비교해야 C10 실패 경로를 그대로 반복하지 않는다.

HBM source/destination 계약은 별도 필수 조건이다. compact tensor를 padded tile에 바로 DMA하는 경우 source pitch와 destination extent가 실제 descriptor에서 의도와 달라질 수 있었다. 원본 bounds·중복 read·destination ownership·출력 gather를 모두 확인해야 하며 logical coverage만으로 correctness를 주장하지 않는다.

GemLite/FlashDecoding++는 이번 조사에서 새로 맞춰 본 batch1 자료다. 두 자료가 추가한 판단은 K 분할 수나 buffer 수를 최대화하는 것이 아니라 non-packed weight의 실제 소유권·reduction·setup 비용을 shape별로 묶어 선택하라는 것이다. Marlin/FLUTE/TileFuse의 offline packing, T-MAN runtime LUT, TPU MXU padding은 원래 전제가 충족되어야 한다. 현 제약에서 그대로 실행 가능한 최적화라고 표시하지 않았다.

기존 profile만으로 다음 whole architecture의 승패는 결정되지 않는다. 현재 source의 전체 weight wait를 유지하면서 tail만 줄이는 방향은 ingress 문제를 직접 해결하지 않는다. C09는 아직 NPU를 실행하지 않았으므로 첫 hardware 진단의 가치는 남는다. 그것을 신규 pipeline 발명이나 이미 입증된 hardware 회귀로 표현하지 않는다. 연구 결과는 후속 설계의 조건이지 새 compile/Arena 승인 또는 성능 보장이 아니다.
