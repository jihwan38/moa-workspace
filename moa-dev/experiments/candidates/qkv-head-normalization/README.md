# QKV를 head layout으로 직접 모아 RMSNorm/RoPE 연결

확인일 2026-09-21, compiler/std 0.8.1. 현재 두 cluster QKV/input RMS + hier32 broadcast를 scratch에 복사하여 baseline **56,719 cycles**를 재현했다. 후보는 **52,971 cycles**, 3,748 cycles / **6.61% 감소**로 exact QKV 컴파일에 성공했다. NPU 정확성과 실제 성능은 이 실험에서 검증하지 않았다.

적용 패치는 `candidate.patch`다. `ops.rs`, `projection.rs`, `rmsnorm.rs`, `rope.rs` 네 파일의 QKV 연결을 함께 적용해야 한다. 이후 root의 다른 변경을 유지하려면 증분 패치를 사용한다. O projection/parallel residual suffix 및 다른 ops는 변경하지 않았다. 최종 네 파일의 hash가 실제 성공한 compile scratch와 일치함을 `evidence.json`으로 확인했다.

## 실제 경로의 변화

기존 경로는 Q/K/V projection 결과를 각 cluster의 한 slice로 Switch gather한 뒤 한 cluster로 합친다. `sliding::rmsnorm`은 Q16/K8/V8 heads를 한 slice에서 순서대로 normalize한다. 그 뒤 `sliding::rope::apply_rope`가 Q/K를 Ns8개의 slices로 InterTranspose하고 padding을 제거하는 두 pass씩을 추가로 수행한다.

새 경로는 다음과 같다.

```text
두 cluster의 row-wise projection + 기존 BF16 scale 결과
  → row 축을 Ns/Gs/Ds로 같은 physical order에서 이름 변경
  → typed DMA로 HeadSlices=m![1#32,Ns]에 직접 gather
  → head별 Ds256 RMSNorm, BF16 output
  → Q/K는 이미 분산된 상태로 기존 RoPE 산술
  → 기존 최종 Q/K gather 및 HBM/cache write
  → V는 BF16 norm 후 canonical Ns×Ds로 typed DMA하여 기존 scatter
```

Q는 Ns slice 하나에 Gs=2 heads를 두며, K/V는 Ns slice 하나에 head 한 개를 둔다. Q 16개 heads를 16 slices로 더 쪼개지 않아 기존 RoPE가 기대하는 Ns8-slice/Gs2-local layout과 맞춘다. RoPE의 첫 두 InterTranspose와 두 padding 제거 pass만 삭제했다. `let first_half_q` 이후 실제 half rotation, FP32 cos/sin 곱/합, BF16 cast, 최종 Q/K gather는 원본과 byte-identical하다.

## row→head reinterpret의 정확한 대응

Query의 scaled tensor는 Cluster=`Qs/2048`, Slice=`Qs/8%256`, Element=`Qs%8`이다. 같은 물리 위치를 Cluster=`Ns/4`, Slice=`Ns%4,Gs,Ds/8`, Element=`Ds%8`로 재해석한다.

- 이전 row: `q=2048*c+8*s+j`.
- 새 좌표: `ns=4*c+s//64`, `gs=(s//32)%2`, `ds=8*(s%32)+j`.
- 항상 `q=(ns*2+gs)*256+ds`다.

K/V는 Cluster=`Ps/1024`, Slice=`Ps/4%256`, Element=`Ps%4`를 Cluster=`Ns/4`, Slice=`Ns%4,Ds/4`, Element=`Ds%4`로 재해석한다.

- 이전 row: `p=1024*c+4*s+j`.
- 새 좌표: `ns=4*c+s//64`, `ds=4*(s%64)+j`.
- 항상 `p=256*ns+ds`다.

`validate_layout.py`가 Q4096/KV2048 모든 row를 전수 검사하여 대응과 bijection을 확인한다. 이 reshape는 scale을 마친 BF16 값을 같은 물리 순서로 이름만 바꾸며, 이후 실제 이동은 typed DMA가 한다. 외부 weight/scale layout 또는 연산을 변경하지 않는다. 각 source slice의 유효 Q16 bytes/KV8 bytes는 DMA alignment를 충족한다. Ns=3/4 cluster 경계도 위 검사에 포함된다.

## RMSNorm 산술과 반올림

각 head는 기존과 동일하게 BF16→FP32, Ds%4 packet으로 제곱, Ds256 intra-slice sum, Div256, EPS 더하기, Sqrt를 수행한다. Q/K는 이후 원래 DivF 다음 weight Mul0, V는 원래 Div를 수행하고 BF16으로 cast한다. Ns만 Time에서 Slice로 옮겼으며 Ds256 내부의 packet/time 순서와 reduction axis는 그대로다. 서로 다른 head를 합하는 연산은 없다.

Q의 mean-square/rms 임시값은 `m![Gs,1#8]`, K/V는 `m![1#8]`로 보관한다. 유효 scalar마다 padding을 둬 최소 commit/VRF packet 크기를 맞춘다. F32 임시값의 저장이므로 새 BF16 rounding을 추가하지 않는다. sqrt 역시 유효 scalar마다 같은 연산이다. `rms_vrf`의 Gs 축은 각 query head에 맞는 scalar를 선택하며 Ds축으로 broadcast된다. norm weight는 동일 Ds256 값을 Ns8 slices에 복제한다. padding을 합에 포함하거나 테스트 값으로 output을 대체하지 않는다.

이 구조적 검토와 compile 성공은 장치의 oracle을 대신하지 않는다. 특히 서로 다른 Q의 Gs0/Gs1 RMS scalar, Ns3/4 경계, Ds127/128 RoPE half 경계 및 K/V cache를 하드웨어로 확인해야 한다.

근거: 원본 `sliding/rmsnorm.rs`, `sliding/rope.rs`와 [std 0.8.1 reshape/DMA 계약](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs), [Tensor semantics](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/tensor-semantics.html), [Vector Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/vector-engine/index.html).

## 정적 효과와 한계

| 소스 영역 | 기존 Main cycles 합 | 후보 Main cycles 합 |
| --- | ---: | ---: |
| QKV projection | 14,332 | 11,236 |
| Q/K/V head RMSNorm | 6,630 | 3,045 |
| RoPE | 8,356 | 4,182 |

projection의 세 Switch gather가 typed head DMA로 바뀌고 RMSNorm의 heads가 8배 공간 병렬화되며 RoPE의 재배치 네 pass가 빠진다. 대신 projection DMA 합은 31,351→32,188, RMS weight DMA 합은 900→1,006 cycles로 늘어난다. 다른 context와 overlap하기 때문에 위 Main 절감 합 10,855를 전체 개선으로 주장하지 않는다. 실제 makespan 개선은 3,748 cycles이며 instruction 수는 149→143이다.

## 재현

Scratch `/tmp/moa-qkv-head-normalization`에서 baseline/후보를 순차 컴파일했다. raw schedule은 scratch, source/schedule hashes와 module별 context 합계는 `evidence.json`에 있다.

```sh
python3 moa-dev/experiments/candidates/qkv-head-normalization/validate_layout.py
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-qkv-head-normalization; export CARGO_TARGET_DIR=/tmp/moa-qkv-head-normalization/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-head-normalization/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedules/sliding_project_qkv.json 2>&1 | tee logs/head-normalization.log'
```

layout validation PASS, compile exit 0. 기존 미사용 `broadcast_sliding_heads` 경고만 있다. 라이브 소스·Git·NPU 작업은 수행하지 않았다.
