# 16개 TDMA 복제와 16-slice ring의 계층형 broadcast

기준: `90f3c3fe16f4fe445f81e82d30b17b1b951e0082`, compiler/std 0.8.1, 확인일 2026-09-21. 실험자는 라이브 소스·Git·NPU 작업을 변경하지 않았다. 아래 값은 schedule makespan이며 실측 성능이나 정확성 통과를 뜻하지 않는다.

## 변경과 사용 파일

`candidate.patch`는 원본 `src/ops.rs`에만 적용한다. QKV는 원래 RMSNorm의 BF16 결과를 TDMA로 16개 slice에 복제한다. O는 HBM 입력을 처음부터 그 16개 slice에 읽는다. 두 경로 모두 각 16-slice 그룹에서 `CustomBroadcast { ring_size: 16 }`을 수행하여 최종 256개 slice의 기존 `Replicated` 입력을 만든다. 기존 projection, scale, RMSNorm, 반올림은 그대로다.

`combined.patch`는 여기에 QKV FP8 lookup→contraction 융합과 O 256-slice/15-row projection을 더한 조합이다. 완성 소스는 `ops.rs` 및 `projection-combined.rs`다. **O 조합은 4개 BF16 partial 합을 하나의 4096-element FP32 reduction 뒤 BF16 cast로 바꾸므로 원본과 반올림 순서가 다르다.** 자세한 O 매핑은 `../output-256rows/README.md`를 참고한다. `source-hashes.json`과 `evidence.json`으로 정확히 컴파일한 파일을 식별한다.

## 매핑과 안전성 근거

입력 Slice `m![Dummy256 / 16, 1 # 16]`는 물리 slice 0,16,…,240에 같은 BF16 벡터를 가진다. `/16`의 상위 그룹 축을 보존하고 출력 Slice를 `m![Dummy256 / 16, 16]`로 지정하면 각 그룹의 익명 하위 축 16개가 broadcast된다. 그 결과 256개 slice의 유효 Element 값이 모두 동일하다. 마지막 `unsafe { x.reshape() }`는 익명 하위 축을 `Replicated=m![Dummy256]`로 이름 붙이며 물리 순서·Element 배치를 바꾸지 않는다. 모든 slice가 동일한 논리 벡터를 가져야 한다는 불변식에 근거한 해석이다. 패딩 위치를 데이터로 사용하여 정확성을 우회하지 않는다.

`to_dm`은 std 0.8.1 `tensor/memory.rs`의 `assert_dm_dma_layout`과 `self.inner.transpose(true)` 경로를 사용한다. 데이터 타입 BF16을 유지하는 mapped copy이므로 산술이나 새로운 반올림이 없다. Switch output과 source의 그룹 축을 유지해야 compiler가 ring16 범위를 증명할 수 있다. 이 타입/매핑 검토와 성공한 컴파일은 하드웨어 oracle 검증을 대체하지 않는다.

근거: [Mapping expressions](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/mapping-expressions.html), [Tensor semantics](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/tensor-semantics.html), [Switch Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/switch-engine.html), [0.8.1 memory.rs](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs). 상세 API 조사와 적용 한계는 `docs/research/rngd-furiosa-opt.md`에 기록했다.

## 정적 결과

| 변형 | QKV cycles | O cycles |
| --- | ---: | ---: |
| 원본 | 118,815 | 195,823 |
| 이 broadcast 변경만 | 84,747 | 136,106 |
| broadcast + QKV FP8 fusion + O256 | 79,148 | 60,708 |
| 비교: direct HBM replication + O256 | — | 78,618 |

원래 MainContext ring256은 QKV 62,215 / O 66,311 cycles였다. 새 O 입력의 HBM→16개 복제 DMA는 957 cycles, ring16은 4,359 cycles다. 단독 O 136,106은 direct-HBM 단독 134,996보다 약간 느리다. 원래 별도 weight decode와 입력 DMA가 겹치는 효과가 있기 때문에 이동 노드만 비교해서 전체 이득을 주장하면 안 된다. O256까지 결합하면 이 후보 60,708이 direct-HBM 결합 78,618보다 22.8% 작다.

## 실패한 매핑도 보존

- `failed-attempts/named-axis-ops.rs`: 그룹 input에서 즉시 `Replicated` output으로 switch하면 `Switch snoop bitmap disagrees with (OutSlice x OutTime) enumeration at output slice 1`로 실패한다.
- `failed-attempts/anonymous-source-ops.rs`: source 그룹을 익명으로 바꾸고 `Replicated` output을 쓰면 `Switch ring size mismatch: slot sweep computed 256, but 16 asserted`로 실패한다.
- 성공안은 source와 output에 동일한 상위 `Dummy256/16` 축을 유지하고, 복제 후에만 reinterpret한다. 실패 로그는 같은 폴더에 있다.

## 재현과 다음 검증

Docker scratch `/tmp/moa-hierarchical-broadcast-16`은 broadcast 단독, `/tmp/moa-hierarchical-combined`는 조합이다. 각 scratch의 `src`와 manifest 사본으로 컴파일했고 live `/workspace/target`은 사용하지 않았다. 원본 schedule은 읽기 전용 비교에만 썼다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-hierarchical-combined; export CARGO_TARGET_DIR=/tmp/moa-hierarchical-combined/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-hierarchical-combined/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedules/sliding_project_qkv.json 2>&1 | tee logs/qkv-combined.log'
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-hierarchical-combined; export CARGO_TARGET_DIR=/tmp/moa-hierarchical-combined/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-hierarchical-combined/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule schedules/sliding_attention_output.json 2>&1 | tee logs/o-combined.log'
```

두 명령 모두 exit 0, 각각 1 exact kernel 컴파일 성공. 사용하지 않게 된 `broadcast_sliding_heads` dead-code warning만 나온다. 원본·후보 모든 값의 oracle 비교와 RNGD 실측은 root가 수행한다. raw schedule JSON은 scratch에 남겨두고 저장소에는 요약만 둔다.
