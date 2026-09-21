# QKV projection을 두 cluster / 512 slices에 분산

확인일 2026-09-21, compiler/std 0.8.1. `hierarchical-broadcast-16`의 QKV FP8 fusion + O256 조합을 출발점으로 QKV만 수정했다. `#[device]` 함수 시그니처는 변경하지 않는다. 라이브 소스·Git·NPU 작업은 수행하지 않았고 컴파일/정적 schedule만 확인했다.

## 결과와 적용 파일

| QKV 변형 | static makespan | instruction 수 |
| --- | ---: | ---: |
| 최초 원본 | 118,815 | 135 |
| hier16 + FP8 fusion, 한 cluster | 79,148 | 133 |
| 두 cluster, Q/KV helper별 activation 복제 | 72,577 | 160 |
| 두 cluster, 입력 hier16 복제 공유 | **61,994** | 153 |

최종 후보는 이전 조합보다 정적 cycles가 21.7% 작다. 원본 대비 static speedup은 1.917x다. 실측 cycles와 정확성 통과는 아직 이 실험에서 확인하지 않았다.

`ops.rs`와 `projection.rs`가 최종 완성 소스다. `candidate.patch`는 hier16+QKVfusion+O256 조합에 적용할 증분 패치, `combined-from-baseline.patch`는 원본 commit `90f3c3fe16f4fe445f81e82d30b17b1b951e0082`에 적용할 전체 조합 패치다. `projection-per-helper-copy.rs`는 첫 번째 성공안이며, 원래 hier16 `ops.original.rs`와 함께 컴파일한 것이다. `evidence.json`에 source/schedule hash와 긴 instruction을 기록했다. raw schedule은 Docker scratch에만 남긴다.

최종 O projection suffix와 `full_project_qkv` 이후 모든 ops는 출발점 조합과 byte 단위로 동일함을 확인했다. 파일에 포함된 O256 변경의 reduction rounding 차이는 기존 조합에서 온 것이며 이번 QKV 변경이 추가한 것이 아니다.

## 데이터 흐름과 불변식

1. 입력 RMSNorm까지 기존 단일 cluster 코드를 유지한다. 정규화한 BF16 `H=3840` 벡터가 source slice 한 개에 있다.
2. 기존 hier16 destination Cluster를 `m![2]`로 바꾼다. TDMA가 각 cluster의 slice 0,16,…,240에 동일 벡터를 복제한다. 총 32개 source copy다. 각 cluster에서 ring16을 수행하면 2×256 slices 모두 같은 H 벡터를 가진다. Cluster가 익명 broadcast 축이므로 서로 다른 입력 값을 암묵적으로 합치지 않는다.
3. Q helper는 이 복제된 activation view를 Cluster=`m![Qs/2048]`, Slice=`m![Qs/8%256]`로 재해석한다. 각 cluster 2048개 Q rows, 각 slice 8개 rows를 처리한다. 원래 row-major FP8 weight는 typed TDMA로 해당 row 위치에 실제 배치한다.
4. K/V helper는 Cluster=`m![Ps/1024]`, Slice=`m![Ps/4%256]`이며 각 slice 4개 rows를 처리한다. 두 weight matrix는 동일한 activation TRF를 재사용한다.
5. 각 output row의 H=3840 reduction, FP8 lookup→BF16, FP32 contraction, BF16 cast, FP32 scale 곱, BF16 cast 순서는 기존 fusion과 같다. row 계산을 둘로 나누거나 partial을 새롭게 합하지 않는다.
6. cluster 내부 Switch gather가 Q의 하위 2048개 / KV의 하위 1024개 elements를 slice0에 모은다. typed `output.to_dm`이 Cluster의 상위 row 축을 Element로 옮겨 원래 한 cluster/slice의 `Qs` 또는 `Ps` 순서를 복원한다. 그 뒤 원래 RMSNorm/RoPE/cache path를 사용한다.

activation의 `unsafe reshape`는 모든 512개 copy가 같은 H 벡터라는 사실에 근거한다. row-major weight 및 scale은 reshape하지 않고 TDMA로 변환한다. gather 이후 cluster merge 역시 순서를 임의로 해석하지 않고 named axis를 보존한 typed copy로 수행한다.

## K/V의 네 값과 padding

K/V 결과 4×BF16은 8 bytes로 최소 commit 크기를 충족한다. contraction은 `transpose::<m![1],m![Ps%4#16]>()` 후 유효 네 값만 commit한다. scale 및 contraction 값을 FP32 vector로 가져올 때 Packet=`m![Ps%4#8]`의 앞 네 값이 유효하고 뒤 네 값은 padding이다.

0.8.1 `vector_tensor.rs`의 `vector_narrow_trim::<m![Ps%4]>()`는 뒤 dummy 네 lanes를 타입 수준에서 제거한다. `Mul0`은 유효 네 값에만 scale 곱을 수행한다. 이어 `vector_widen_pad::<m![Ps%4#8]>()`로 padding을 복원하고 BF16 cast 후 앞 네 값만 commit한다. 이 API들은 std 소스에서 hardware no-op으로 문서화되어 있다. 임의 padding 값이 reduction 또는 유효 output 값으로 들어가지 않는다. VRF 역시 같은 유효 네 scale과 padding으로 저장된다.

근거: [std 0.8.1 vector_tensor.rs, narrow_trim/widen_pad](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/vector/tensor/vector_tensor.rs), [std 0.8.1 memory.rs, DmTensor::to_dm](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs), [spatial/temporal mapping](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/spatial-temporal-dimensions.html), [tensor semantics](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/tensor-semantics.html). API 코드를 실제 읽고 컴파일했다. 수치 동일성은 위 구조에서의 예상이며 RNGD oracle로 검증해야 한다.

## 병목 변화와 범위

Q weight DMA는 26,475→13,512 cycles, K/V 각각은 13,512→7,030 cycles다. Q contraction은 5,063 cycles, K/V 각각 2,663 cycles다. 두 cluster로 복제하는 초기 TDMA는 8,352 cycles, ring16은 4,103 cycles다. 모든 DMA가 같은 엔진에 있으므로 개별 절감의 합이 최종 makespan 절감과 같지는 않다.

첫 시도는 이미 256개 slice에 복제한 activation을 Q helper와 KV helper 각각에서 두 cluster로 다시 복제했다. 8,550-cycle DMA가 두 번 발생하여 72,577 cycles에 그쳤다. 최종안은 처음 hier16부터 두 cluster에 복제하여 이 중복 작업을 제거했다. 양쪽 시도 모두 exact QKV 컴파일 성공이다.

## 재현

Scratch `/tmp/moa-qkv-two-clusters`의 source copy, 전용 target 및 전용 kernel 출력 directory를 사용했다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-qkv-two-clusters; export CARGO_TARGET_DIR=/tmp/moa-qkv-two-clusters/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-two-clusters/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedules/sliding_project_qkv.json 2>&1 | tee logs/attempt-2.log'
```

exit 0, exact kernel 1개 성공. 기존 미사용 helper `broadcast_sliding_heads` 경고만 발생했다. 다음 단계는 root의 전체 oracle/NPU 측정이며, 특히 cluster1의 row 순서, KV 네 valid lanes, gather 이후 K/V cache 결과를 확인한다.
