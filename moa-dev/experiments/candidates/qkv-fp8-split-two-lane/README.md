# QKV hi/lo를 두 TRF lanes에서 동시에 contraction

**NPU 검증 실패: E20 job 70290.** Q/K는 각각 3개 설정 모두 PASS였으나 V는 3개 모두 FAIL이다. V max absolute error는 [2.23047, 2.93213, 3.0625], tolerance 내 비율은 [97.95, 98.39, 91.41]%다. 측정 QKV median 107,699 cycles는 정확성 실패 때문에 채택 가능한 성능 수치가 아니다. 원본 소스는 변경하지 않고 실패 증거를 보존한다. 원인은 아직 확인되지 않았으며, E19 두 별도 dot의 PASS와 V에 집중된 오류를 근거로 register lifetime·pair/packing 경로를 분리 검증한다.
로그: [`e20-qkv-fp8-two-lane-output-rms-job-70290.log`](../../../logs/e20-qkv-fp8-two-lane-output-rms-job-70290.log).

2026-09-21, compiler/std 0.8.1. `qkv-fp8-split`의 독립 FP8 분해를 유지하면서 hi/lo 두 내적을 하나의 weight stream으로 계산했다. **Exact QKV compile 성공, 후속 NPU 검증에서 V 실패**이다. 최초 두-dot 후보의 E19 oracle PASS를 이 후보의 정확성 증거로 대신하지 않는다.

| 구현 | static cycles | Q contraction+combine | K/V 각각 |
| --- | ---: | ---: | ---: |
| BF16 activation baseline | 52,971 | 5,063 | 2,663 |
| FP8 두 별도 dot | 54,866 | 1,225 × 2 | 745 × 2 |
| **FP8 두 lanes + pair combine** | **53,305** | **1,226** | **746** |

두-dot 대비 전체 static 1,561 cycles/2.85% 감소다. 아직 BF16 static보다 334 cycles 높다. 개별 dot 절감 합계와 전체 makespan 절감이 같지 않은 것은 DMA 및 다른 context와의 중첩 때문이다. 실제 NPU 결과는 별도 측정해야 한다.

`ops.rs`/`projection.rs`가 완성 소스이고, `candidate.patch`는 `qkv-fp8-split` 대비 증분이다. 원본 head RMSNorm/RoPE는 수정하지 않았고 O projection/residual suffix 및 다른 ops가 기준과 byte-identical임을 검사했다. row interleave 최적화는 아직 합치지 않았다. source/schedule hashes와 주요 stage lifetimes는 `evidence.json`에 있다.

## 데이터와 두 항의 순서

공통 quantizer는 이전과 같은 runtime amax, power-of-two scale, hi/lo FP8 값을 만든다. [기존 수학·CPU 오차 분석](../qkv-fp8-split/README.md)이 그대로 적용된다. global scale에서 작은 값 소실, cancellation에 따른 상대오차 확대, 실제 NPU cast/native FP8 semantics 검증 필요성도 같다.

각 cluster의 8개 quantization slices는 H480씩 가진다. 이 작은 분산 배열에서 hi와 lo를 Element=`Dummy2,H%480`의 각각 component 0/1에 typed DMA로 쓴다. 두 destination은 `.tile::<m![Dummy2],1,...>(0/1)`로 선택한 **in-slice Element** 영역이다. API에 있어도 backend가 지원하지 않는 non-in-slice `slice_tile`은 사용하지 않았다.

기존 두-dot 후보는 hi/lo 각각 anchor DMA와 ring32 broadcast를 수행했다. 여기서는 두 component가 들어 있는 배열을 한 번에 anchor DMA하고 ring32 broadcast한다. 수신 Element=`Dummy2,H`의 첫 3840 bytes는 hi, 다음 3840 bytes는 lo다. broadcast의 모든 valid slices가 두 component의 같은 H 순서를 갖는다. logical pair `(lane,h)` 7680개에 대한 shard→canonical 대응을 전수 검사했다.

Sub TU가 이 7680 bytes를 TRF Lane=`Dummy2`, Element=`H`에 넣는다. 각 weight matrix는 원래 FP8 값 전부를 한 번 streaming하고, 두 lanes가 같은 weight와 hi/lo를 병렬로 곱한다. Packet/Time reducer가 H3840 전체를 합친 뒤 두 lane 값은 아직 별도 FP32 dot 결과다. weight HBM payload와 channel scale을 줄이거나 입력 값을 생략하지 않는다.

## 합친 다음 scale을 적용하는 pair-mode epilogue

Q의 정확한 mapping은 다음과 같다. K/V는 `Qs%8` 대신 `Ps%4`다.

```text
TRF Lane=Dummy2, Element=H
  → contract_outer / packet / time: lane별 FP32 H3840 dot
  → contract_lane Sequential:
      Time=[Qs%8,Dummy2], Packet=1#8
  → vector_intra_slice_unzip<Dummy2>:
      group0=hi-dot, group1=lo-dot, Time=Qs%8
  → narrow_split: Time=[Qs%8,1#2], Packet=1#4
  → Mul0: group0 그대로, group1 ×1/16
  → vector_fp_zip(AddF): hi-dot + lo-dot/16
  → Mul1: 합에 runtime scale s 적용
  → widen_concat → 기존 BF16 projection cast/commit
  → 기존 BF16 channel scale → head-local RMSNorm/RoPE
```

pair에는 `narrow_trim`이 없어 padded `narrow_split`/`widen_concat`을 사용한다. padding에는 유효 데이터를 새로 만들지 않는다. Mul0, AddF, Mul1을 서로 다른 허용 ALU로 사용한다. 별도의 coefficient VRF나 추가 FP32 DM partial, `vector_stash`가 필요 없다.

공식 SDK의 일반 intra-slice reduce 뒤에는 Fp multiply 단계로 돌아갈 수 없다. 따라서 lane 두 값을 weighted reduce하고 다시 s를 곱하는 방식은 추가 pass가 필요하다. pair-mode는 Fp stage의 `vector_fp_zip`에서 합치므로 **합 이후 scale**을 같은 pass로 유지한다. `s*hi`를 먼저 계산할 때 발생할 수 있는 중간 overflow를 피하는 순서도 유지한다.

`TagMode::AxisToggle`은 host API에 있지만 NPU lowering에서는 지원되지 않는다. 이 후보는 공식적으로 지원되는 `vector_intra_slice_unzip`을 사용한다. [공식 pair-mode 설명](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/vector-engine/intra-slice-chain.html), [std pair API](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/vector/tensor/vector_tensor_pair.rs), [Lane Folder](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/lane-folder.html), [DM Element tile](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs)를 확인했다.

## 재현과 한계

scratch는 `/tmp/moa-qkv-fp8-split-two-lane`, target/output 모두 해당 디렉터리 내부다. 기존 두-dot scratch의 source/manifests를 복사한 뒤 candidate 두 파일만 교체했다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-qkv-fp8-split-two-lane; export CARGO_TARGET_DIR=/tmp/moa-qkv-fp8-split-two-lane/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-fp8-split-two-lane/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedules/all-qkv.json 2>&1 | tee logs/attempt-2.log'
```

exit 0, exact kernel 1개 성공. 처음 Element tile의 Index generic에 `Dummy2` AxisName을 넣어 Rust trait error가 났으며, mapping을 요구하는 API에 맞게 `m![Dummy2]`로 수정했다. quantizer/pair mapping 전체가 그 뒤 NPU lowering에 성공했다.

컴파일과 logical layout 검사는 NPU oracle 검증을 대신하지 않는다. 두 lanes, Sequential lane folder, pair unzip 및 실제 arithmetic의 조합은 root의 별도 NPU 검증이 필요하다. 이 작업에서 live source/Git/NPU 제출은 하지 않았다.

## Schedule 라벨 정정

실제 E20 실행 순서는 V→Q→K다. 첫 Ps dot O164(24,281→25,027)는 T86→O203/T100→O204/T101→O216/T106→O227/T110→normalize_value O285에 연결되어 V임을 확인했다. 늦은 O300(43,917→44,663)은 K다. 초기 분석에서 source의 K/V 호출 순서를 근거로 이 둘을 반대로 불렀으며 해당 추정을 철회한다. 따라서 “K가 먼저 소비한 register가 긴 시간 뒤 V에서 손상된다”는 이전 가설의 전제도 성립하지 않는다. register 배치 자체의 공개 schedule 검토만으로 원인은 확정하지 못했다.
