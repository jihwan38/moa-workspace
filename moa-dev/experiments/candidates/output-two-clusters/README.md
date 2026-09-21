# O projection: 두 cluster, 480 valid slices, slice당 8 output rows

확인일 2026-09-21, compiler/std 0.8.1. `qkv-two-clusters` 최종 후보의 QKV 구현을 유지하고 O만 수정했다. 완성 소스는 `ops.rs`, `projection.rs`다. `candidate.patch`는 그 후보 기준 증분 패치, `combined-from-baseline.patch`는 원본 commit `90f3c3fe16f4fe445f81e82d30b17b1b951e0082` 기준 전체 조합이다. 라이브 소스·Git·NPU 제출은 수행하지 않았다.

## 확인 결과

| O 변형 | static makespan | instruction 수 |
| --- | ---: | ---: |
| 최초 원본 | 195,823 | 163 |
| hier16 + O256, 한 cluster | 60,708 | 121 |
| 이 후보: hier16 + 두 cluster × 240 slices | **42,293** | 129 |

이전 조합 대비 정적 cycles 30.3% 감소, static speedup 1.435x다. weight DMA는 26,475→14,513 cycles, contraction은 9,863→5,383 cycles, cluster 내부 gather는 1,032 cycles다. 시작 input DMA 957 cycles와 ring16 4,359 cycles는 양 cluster에 복제해도 해당 schedule에서는 동일하다.

exact O 컴파일은 성공했다. **NPU 정확성과 실제 cycles는 아직 이 실험에서 검증하지 않았다.** QKV 함수 본문/ops가 `qkv-two-clusters`와 같고 O channel-scale epilogue가 기존과 같음을 byte 비교했다. 최종 source hash와 raw schedule hash는 `evidence.json`에 있다.

## 입력에서 output까지

1. O의 HBM `Ns×Gs×Ds=4096` BF16 입력을 처음부터 Cluster=`m![2]`, Slice=`m![Dummy256/16,1#16]`에 복제한다. 두 cluster에서 기존 ring16을 실행하면 총 512개 slice가 동일한 Qs 벡터를 가진다. HBM 입력 및 `#[device]` 시그니처는 그대로다.
2. activation은 Cluster=`m![Dummy2]`, Slice=`m![H/8%240#256]`로 view를 재해석한다. 각 cluster에서 slice 0…239가 유효하고 240…255는 padding이다. 원래 512 copies가 모두 같은 벡터를 가지고 있으므로 이 재해석은 유효 위치의 값을 바꾸지 않는다. padding slice의 copy를 버리는 것이며, 다른 row의 값을 읽는 permutation은 없다.
3. 원본 weight의 H=3840 row-major 순서는 물리적으로 연속된 1920-row 블록 두 개다. HBM view를 `m![Dummy2,H%1920,Qs]`로 재해석한 후 typed DMA로 Cluster=`Dummy2`, Slice=`H/8%240#256`, Element=`H%8,Qs`에 배치한다. 유효 위치 `(cluster c,slice s,local row j)`는 원본 row `1920*c+8*s+j`와 정확히 대응한다. c∈[0,2), s∈[0,240), j∈[0,8)로 모든 3840 rows를 한 번씩 계산한다.
4. 각 row의 FP8 lookup→BF16 및 Qs=4096 FP32 contraction은 기존 O256과 동일하다. BF16 cast 결과를 4개씩 묶어 8-byte 단위로 commit한다. 두 cluster가 각각 1920개 독립 rows를 계산하며 cluster 간 합을 새로 수행하지 않는다.
5. cluster 내부 gather는 Slice row 축을 Time으로 이동하여 `m![H/8%240#256,H%8]`를 만든다. `to_dm`으로 각 cluster의 canonical `H%1920` 배열을 만들어 뒤쪽 padding 128 rows를 제거한다.
6. canonical 결과의 Cluster `Dummy2`를 `H/1920`으로 재해석한다. 실제 물리 순서는 두 연속 1920-row 블록이므로 안전한 축 이름 복원이다. 이후 typed DMA로 기존 한 cluster의 `HiddenRows=m![H/120,1#8]`와 Element=`H%120` 배치를 만들고, 원래 BF16 channel scale 및 후속 RMSNorm/residual을 그대로 사용한다.

padding slice는 contraction 결과에서도 padding으로 유지되고 gather/TDMA의 유효 결과에 포함되지 않는다. padding 값의 임의성이나 테스트 seed를 이용하지 않는다. weight·input·scale의 모든 유효 원소를 사용한다.

## 첫 실패와 해결

처음에는 Cluster=`m![H/1920]`로 바로 weight를 DMA했다. NPU lowering에서 source H axis와 destination의 H_1920 cluster / H_512 PE / H_256 DMN / H_8 slice factor를 동시에 분해하지 못하여 `cannot factorize common axis H`로 실패했다. 소스와 로그는 `failed-attempts/240-rows-projection.rs`, `240-rows.log`에 남겼다.

성공안은 root가 제안한 **상위 row 축 이름 분리**다. 연속 row-major 데이터를 바꾸지 않고 HBM view에서 높은 축을 `Dummy2`로 이름 붙여 cluster와 local H factor를 분리했다. 이렇게 하면 목적지의 240→256 slice padding을 compiler가 lowering한다. output을 canonical local layout으로 정리한 뒤 축 이름을 복원하므로 외부 tensor 의미는 원래 H로 돌아온다. 더 적은 128 slices/cluster fallback은 검토만 했고 실행하지 않았다.

이것은 shape 크기만 맞추어 데이터를 임의로 해석하는 우회가 아니다. 모든 unsafe view에 대해 위의 physical row 대응과 Element 순서를 보존한다. 근거: [std 0.8.1 HbmTensorView::reshape / DmTensor::reshape / to_dm](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs), [Tensor semantics](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/tensor-semantics.html), [Mapping expressions](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/mapping-expressions.html).

## 정밀도 경계

이 변경은 **직전 O256 후보 대비** 새 reduction이나 cast 위치를 추가하지 않는다. 하지만 O256 자체는 최초 원본의 4×1024 BF16 partial 및 BF16 합을 4096-element FP32 reduction 뒤 BF16 cast로 바꿨다. 따라서 최초 원본과 비트 동일함을 주장하지 않는다. 기존 O256의 정확성 검증을 이 두-cluster 매핑의 검증으로 대신할 수도 없다.

## 재현

Docker scratch `/tmp/moa-output-two-clusters`를 사용했고 전용 target/kernel output directory를 지정했다. `/workspace/target`은 원본 schedule 읽기 외에 사용하지 않았다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-output-two-clusters; export CARGO_TARGET_DIR=/tmp/moa-output-two-clusters/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-output-two-clusters/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule schedules/sliding_attention_output.json 2>&1 | tee logs/attempt-2.log'
```

exit 0, exact kernel 1개 성공. 미사용 `broadcast_sliding_heads` 경고만 나온다. 다음 단계는 root의 전체 oracle/NPU 측정이다. 특히 1920-row cluster 경계와 240-row-group padding 경계의 출력 및 최종 residual을 확인한다.
