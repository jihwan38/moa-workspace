# FFN residual + layer scalar: 8-slice 병렬 epilogue

확인일 2026-09-21, compiler/std 0.8.1. `ffn-all-two-cluster`의 MLP를 그대로 두고 FFN 후단의 residual add 및 layer scalar 곱을 같은 8-slice layout에서 처리했다. **정적 makespan 213,950→210,215 cycles, 3,735 cycles / 1.75% 감소**다. exact FFN 컴파일에 성공했으며 NPU 정확성/실측은 하지 않았다.

적용 대상은 `candidate.patch`다. FFN ops의 두 call을 새 `shared::residual::add_and_scale_parallel` 한 번으로 바꾸고, shared residual 파일 뒤에 helper를 추가한다. 기존 `add`, `scale_by_layer_gate`, vision helper 및 다른 ops는 유지한다. `ops.rs`, `residual.rs`는 출발점의 완성 파일 사본이며 `mlp.baseline-snapshot.rs`는 비교용이다. 현재 root의 독립 변경을 유지하기 위해 증분 패치를 사용한다.

## 기존 tail과 변경

기준 소스의 `shared::mlp`가 보존된 `ffn-all-two-cluster/mlp.rs`와 같음을 확인하고 scratch에서 baseline **213,950**을 재현했다. 기존 post-RMSNorm은 207,268에 끝나고 static-unrolled residual의 여덟 add가 211,675에 끝난다. 한 slice의 layer scalar 곱이 212,916, HBM write/sync가 213,950에 끝난다.

새 helper는 `ResidualSlices=m![1#32,H/480]`로 normalized output과 기존 residual DM을 typed DMA한다. 각 slice는 H의 연속 480개 값을 갖는다. residual을 FP32 VRF에 넣고 모든 여덟 slice에서 동시에 add한다. layer scalar도 같은 slices에 복제하여 다음 pass에서 곱한다. 마지막 BF16 결과만 한 번 gather하여 canonical H로 복원한다.

후보의 post-RMSNorm은 207,486에 끝나며, 그 뒤 x DMA 449, add 341, scalar pass 401, gather 504, HBM write 434, final sync 600 cycles다. tail은 총 2,729 cycles다. 앞부분의 norm 완료가 기준보다 218 cycles 늦어졌으므로 tail 절감만 합하여 이득을 부풀리지 않는다. 전체 schedule의 max(end)로 3,735-cycle 개선을 계산했다.

## BF16 경계 보존

새 helper의 연산 순서는 다음과 같다.

```text
기존 post-RMSNorm의 BF16 output + 기존 BF16 residual
  → 각 피연산자 FP32 변환
  → 같은 ClipBinaryOpF32::Add
  → BF16 cast 및 DM commit
  → 다시 FP32 fetch_cast
  → 기존 BF16 layer scalar를 FP32로 변환한 값과 Mul0
  → BF16 cast 및 DM commit
  → gather → 원래 HBM output
```

add와 scalar 사이의 BF16 반올림을 생략하거나 하나의 FP32 식으로 합치지 않는다. post-RMSNorm의 EPS/sqrt/div/weight, MLP global/local scales 및 residual 입력 값도 바꾸지 않는다. helper에는 unsafe reshape와 inter-slice arithmetic reduction이 없다. 새 residual VRF는 slice당 480×4=1,920 bytes이고 scalar VRF는 32 bytes다. 유효 값의 인덱스는 `h=480*slice+element`이며 3,840개 원소를 한 번씩 처리한다.

근거는 실제 `src/device/shared/residual.rs::{add,scale_by_layer_gate}` 및 `shared::rmsnorm::normalize`와 [std 0.8.1 typed DMA](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs), [Vector Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/vector-engine/index.html)다. 위 구조적 동치 검토와 컴파일 성공은 NPU oracle을 대신하지 않는다.

## 재현

Docker scratch `/tmp/moa-ffn-residual-parallel`에서 baseline과 후보를 순차 컴파일했다. source/schedule hashes, instruction 수(525→449), 후단 node lifetime은 `evidence.json`에 보존했다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-ffn-residual-parallel; export CARGO_TARGET_DIR=/tmp/moa-ffn-residual-parallel/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-residual-parallel/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule schedules/decoder_feedforward.json 2>&1 | tee logs/parallel-residual.log'
```

exit 0. 기존 미사용 `broadcast_sliding_heads`와 새 callsite로 사용하지 않게 된 `scale_by_layer_gate` 경고만 있다. 라이브 소스·Git·NPU 작업은 수행하지 않았다.
