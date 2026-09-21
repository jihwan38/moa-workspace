# O residual을 8개 slices에서 병렬 처리

확인일 2026-09-21, compiler/std 0.8.1. `output-two-clusters`의 O projection 이후 tail을 분석하고, 가장 큰 동치 최적화 후보로 residual의 8개 tile 직렬 실행을 제거했다. 새 helper `sliding::projection::add_output_residual`와 O callsite만 바꾼다. 기존 projection, channel scale, RMSNorm, 다른 kernel의 공유 residual helper는 수정하지 않는다.

현재 root의 QKV ring sweep 및 FFN 변경을 유지하려면 **`candidate.patch`의 O callsite/새 helper 변경만 적용**한다. 완성 `ops.rs`/`projection.rs`는 이 실험 출발점의 전체 파일 사본이며 그 후 root의 다른 변경을 포함하지 않는다. `residual.parent-snapshot.rs`는 공정한 비교를 위해 읽은 shared residual macro 버전이며 적용 대상이 아니다.

## 정적 비교

| O2-cluster 이후 residual 구현 | static makespan | instruction 수 |
| --- | ---: | ---: |
| 원래 rolled 8 tiles | 42,293 | 129 |
| parent의 static-unrolled 8 tiles | 38,588 | 120 |
| 이 후보: 8 slices 병렬 add | **35,557** | 43 |

rolled 대비 15.9%, parent unrolled 대비 7.85% cycles 감소다. 세 schedule을 같은 O2-cluster projection/동일 compiler에서 비교했다. 후보와 parent-unrolled 기준 모두 exact O 컴파일 exit 0이다. **정확성 및 RNGD 실제 성능은 이 실험에서 미검증**이다.

## 남은 경로 분석

42,293-cycle schedule의 producer/consumer tensor ID와 instruction lifetime을 따라 결과까지 추적했다. 다음은 실제 예약된 completion chain이다. schedule JSON에는 전체 resource-edge DAG가 없으므로 모든 idle gap을 특정 하드웨어 원인으로 단정하지 않는다. 관련 instruction IDs, input/output tensor IDs 및 sync 노드는 `evidence.json`에 보존했다.

| 단계 | 시작→종료 cycles | 관찰 |
| --- | ---: | --- |
| activation TRF | 7,879→9,166 | 다음 GEMV 입력 하나가 준비됨 |
| weight DMA | 3,523→18,036 | GEMV의 마지막 미완료 입력 |
| GEMV | 18,036→23,419 | weight/activation/LUT 입력을 소비 |
| gather + padding 제거 + HiddenRows DMA | 23,419→25,726 | BF16 projection 값을 scale layout으로 이동 |
| explicit sync와 예약 gap | 25,726→27,326 | 600-cycle sync + 이어지는 1,000-cycle gap 관찰 |
| channel scale + canonical DMA | 27,326→28,106 | BF16 cast 위치 유지 |
| RMSNorm 재배치와 계산 | 28,106→33,229 | mean-square, reduce+EPS, sqrt, div×weight, gather+BF16 |
| rolled residual 8 tiles | 33,229→39,659 | 8×Sub VRF load와 Main add, PE 제어 포함 |
| 종료 barrier/HBM write/sync | 39,659→42,293 | 최종 output 저장과 완료 |

activation TRF는 weight DMA보다 8,870 cycles 일찍 준비된다. 따라서 이 schedule에서 O ring16의 계산 시간을 줄이는 것만으로는 GEMV 시작을 앞당길 수 없다. weight DMA와 입력 전송의 선행 순서를 바꾸면 다른 결과가 가능하지만 이 실험에서는 그 변형을 섞지 않았다.

parent의 unrolled residual은 norm이 33,226에 끝나고 add를 33,426…37,554 동안 8번 실행한다. 기존 unroll이 이미 줄인 비용을 후보 성과에 중복 계산하지 않기 위해 38,588 기준을 별도로 재현했다.

## 새 데이터 흐름과 수치 동치 근거

입력은 **기존 RMSNorm이 이미 BF16으로 cast한 H 벡터**와 원래 residual HBM의 BF16 벡터다. RMSNorm 내부 EPS/sqrt/div/weight 곱이나 projection channel scale의 순서는 바꾸지 않는다.

1. normalized BF16 값을 typed TDMA로 `ResidualSlices=m![1#32,H/480]`, Element=`H%480`에 배치한다. slice s∈[0,8), element i∈[0,480)는 기존 H index `480*s+i`다.
2. residual HBM을 같은 8-slice layout으로 직접 읽고 BF16→FP32 변환 후 VRF에 적재한다. 기존 canonical DM 중간 사본이 필요 없다.
3. 모든 8개 slice에서 normalized BF16→FP32, 동일 `vector_clip(ClipBinaryOpF32::Add, &residual_vrf)`, BF16 cast를 수행한다. 원래 tile 루프와 같은 원소 쌍을 한 번씩 더하며 새로운 reduction이나 cast 생략은 없다. slice당 480×4=1,920 bytes의 FP32 VRF payload로 8,192-byte 한도 이내다.
4. 이미 BF16인 결과를 `Broadcast1 { slice1:8,slice0:1 }`으로 canonical H 순서에 gather한다. gather 이후 추가 산술/cast 없이 원래 HBM residual output에 저장한다.

새 helper에는 unsafe reshape가 없고 모든 input 분배는 typed DMA다. padding은 valid element와 섞는 reduction이 없으며 output에 포함되지 않는다. shared `residual::add`와 다른 ops는 그대로다. 기존 projection 파일 전체가 새 helper 앞에 byte-identical하게 남아 있음을 확인했다.

후보 schedule의 norm 이후 경로는 33,229→33,678 input DMA(449), →34,019 add(341), →34,523 gather(504), →34,957 HBM write(434), →35,557 sync(600)이다. residual HBM DMA와 VRF 적재는 앞부분과 겹쳐 이 tail에는 나타나지 않는다.

근거: 현재 원본 `src/device/shared/residual.rs::add`와 `src/device/shared/rmsnorm.rs::normalize`, [std 0.8.1 memory.rs의 typed DMA](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs), [Vector Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/vector-engine/index.html), [Register Files](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html). 컴파일/매핑 증명은 실제 oracle 검증을 대체하지 않는다.

## 재현과 보존

Scratch `/tmp/moa-output-residual-parallel`에서 parent residual macro snapshot을 사용해 기존 O2-cluster를 먼저 컴파일하여 `schedules/sliding_attention_output-parent-unrolled.json`을 저장했다. 이후 candidate ops/projection을 복사하고 다음을 실행했다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-output-residual-parallel; export CARGO_TARGET_DIR=/tmp/moa-output-residual-parallel/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-output-residual-parallel/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule schedules/sliding_attention_output.json 2>&1 | tee logs/parallel-residual.log'
```

미사용 `broadcast_sliding_heads` 경고만 발생했다. `evidence.json`에 schedule/source hashes, longest nodes, 이전 tail의 dependency milestones를 기록했다. 기존 O2-cluster 후보와 그 scratch는 그대로 남겨 fallback할 수 있다. 라이브 소스·Git·NPU 작업은 수행하지 않았다.
