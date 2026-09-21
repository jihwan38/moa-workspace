# O residual 16/32-slice 제한 sweep

확인일 2026-09-21, compiler/std 0.8.1. 기존 O2-cluster + residual8 후보에서 residual helper만 두 가지로 바꿨다. projection/scale/RMSNorm 및 다른 함수는 동일하다. NPU 정확성/실측은 하지 않았다.

| slices | local elements | gather | static cycles |
| --- | ---: | --- | ---: |
| 8 | 480 | TU Broadcast1 | 35,557 |
| 16 | 240 | TU Broadcast1 | 35,527 |
| 32 | 120 | typed DMA | 35,457 |

최대 절감은 100 cycles / 0.28%다. **hardware PASS된 8-slice 구현을 유지한다.** static 차이가 작으며 새로운 매핑의 검증 비용을 들일 우선순위가 낮다. 이 판단은 16/32가 하드웨어에서 느리다는 측정 결과를 뜻하지 않는다.

16-slice는 `m![1#16,H/240]`, 32-slice는 `m![1#8,H/120]`이며 `h=local_size*slice+element` 순서로 같은 FP32 add→BF16 cast를 수행한다. BF16 sum을 각 slice에서 만든 뒤 gather하므로 산술 순서를 바꾸지 않는다.

32-slice의 첫 TU gather는 local 120 BF16 값이 32-byte flit의 정수 배가 아니어서 `Collect packet mismatch. Expected: H % 120 # 128 % 16, got: H % 16`로 실패했다. 최종 32안은 BF16 결과의 typed `result.to_dm`으로 canonical H를 만든다. local 240 bytes는 DMA의 8-byte alignment를 충족한다. 따라서 32안은 slice 수와 gather 경로가 함께 바뀐 변형이다. 최초 소스/로그를 `projection.32-tu-gather-failed.rs`, `residual-32-tu-gather-failed.log`로 보존했다.

두 최종안 모두 exact O 컴파일 exit 0. 완성 소스는 `projection.16.rs`, `projection.32.rs`, 8안 기준 패치는 `candidate.16.patch`, `candidate.32.patch`다. source hashes는 `source-hashes.json`, static schedule 및 마지막 node는 `evidence.json`에 있다. 기존 소스 파일 앞부분이 같음을 byte 비교했다.

Scratch `/tmp/moa-output-residual-sweep`, `CARGO_TARGET_DIR=/tmp/moa-output-residual-sweep/target`, `FURIOSA_OPT_OUT_DIR=/tmp/moa-output-residual-sweep/target/furiosa-opt/kernel`을 사용했다. 각 소스를 scratch에 복사한 뒤 `cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule schedules/sliding_attention_output-16.json` 또는 `-32.json`으로 컴파일했다. 라이브 소스·Git·NPU 작업은 수행하지 않았다.
