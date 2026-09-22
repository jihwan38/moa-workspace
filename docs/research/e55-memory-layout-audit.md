# E55 모델·RNGD 메모리 배치 감사와 E57–E64 실측

2026-09-22, `baseline-e55` (`873f64e`), compiler/std 0.8.1, Rust nightly 2026-05-01.
작업 브랜치는 `opt/ffn-dataflow`다. **이번 8개 장치 실행 후보에서는 재현 가능한 개선을 확보하지 못해 E55를 유지한다.** QKV·O·FFN의 배치를 모두 조사하고 각각 Arena에서 변경 후보를 실행했다. 메모리 배치의 모든 가능성을 소진했거나 세 커널의 최적화를 완료했다는 결론은 아니다.

## 모델 파일에서 연산까지 확인한 경로

[axes.rs](../../furiosa-opt-gemma4-12B/src/axes.rs)의 실제 대상은 H=3840, L=15360, Qs=4096, Ps=2048이다. [host/load.rs](../../furiosa-opt-gemma4-12B/src/host/load.rs)의 `load_f4`는 checkpoint의 U8 `[rows, columns/2]`를 검사하고 원래 packed bytes를 `HostTensor<f4e2m1>`에 넣는다. `load_f8`도 실제 FP8 tensor를 읽는다. 이후 `to_hbm`이 device allocation에 전송한다. Safetensors 파일 offset을 HBM 물리 주소로 해석하지 않았고, 이번 실험에서 host-side weight 재포장은 하지 않았다.

[host/runtime.rs](../../furiosa-opt-gemma4-12B/src/host/runtime.rs)의 `Decode::sliding`은 QKV → attention → O → `self.mlp`를 실행한다. `Decode::full`도 같은 `self.mlp`를 호출하므로 FFN 공유 코드의 영향은 sliding layer에만 한정되지 않는다. 실제 진입점과 출력은 [ops.rs](../../furiosa-opt-gemma4-12B/src/ops.rs)에 있다.

| 진입점 | HBM → 계산용 배치 → 출력 |
|---|---|
| `sliding_project_qkv` | BF16 x → pre-RMS → ring32 broadcast → BF16 TRF; FP8 Q/K/V → DM → lookup+BF16 dot → F32 channel scale → BF16 → row Switch → head RMS/RoPE → Q 및 지정 KV cache 슬롯 |
| `sliding_attention_output` | BF16 attention 결과 → 16개 column shard의 TRF; FP8 O → DM → lookup+BF16 dot → F32 column 합산 → H480×8 gather → channel scale → BF16/RMS/residual → HBM |
| `decoder_feedforward` | residual/pre-RMS → up/gate FP4 → FP8 DM → BF16 DM → 16원소별 F32 dot·local scale·합산 → GeGLU → down의 같은 decode/dot → global scale/post-RMS/residual/layer scale → HBM |

QKV/O는 [sliding/projection.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs), FFN은 [shared/mlp.rs](../../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)의 실제 구현을 추적했다. QKV/O의 lookup은 dot pipeline 안에 있고, FFN의 lookup은 전체 중간 weight를 DM에 저장한다. 이 차이가 배치 최적화의 주요 제약이다.

## 정렬 규칙과 현재 배치

공식 [Memory Performance](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html)는 DM을 2 clusters × 8 DMNs × 32 slices, slice당 16 banks·512 KiB로 설명한다. Slice 내부 byte address의 bit3–6이 bank이며 연속 128B는 모든 bank에 분산된다. HBM의 권장 접근 단위는 256B다. [DMA Engine](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html)의 HBM↔DM 필수 조건은 주소와 packet의 8B 정렬이다. **512B를 모든 tensor에 붙이는 규칙은 없다.** Padding, 연속 전송, slice/DMN 분산, 후속 fetch/commit의 제약을 함께 확인해야 한다.

아래 pitch는 한 slice에서 다음 논리 행이 시작되는 byte 간격이다. HBM allocation의 절대 물리 주소를 측정한 표가 아니다.

| E55 데이터 | cluster당 slice 배치 | slice당 유효 weight | 행 pitch |
|---|---|---|---:|
| Q FP8 | 32개 그룹 × 8-way row interleave | 8×3840 | 3840 B |
| K/V FP8, 각각 | 32개 그룹 × 8-way row interleave | 4×3840 | 3840 B |
| O FP8 | 16 row groups × 16 column shards | 120×256 | 256 B |
| up/gate FP4, 각각 | 유효 240 / 물리 256 slices | 32×3840 | 1920 B |
| up/gate FP8 → BF16 | 같은 row 배치 | 32×3840 | 3840 → 7680 B |
| up/gate FP8 block scale | 같은 row 배치 | 32×240 | 240 B |
| down FP4 → FP8 → BF16 | 32 row groups × 8 column shards | 60×1920 | 960 → 1920 → 3840 B |
| down FP8 block scale | 같은 row/column 배치 | 60×120 | 120 B |

두 cluster를 사용한다. Up/gate의 16개 padding slices는 유효 행으로 취급하지 않는다. Scale은 16원소마다 다르므로 전체 GEMV 뒤 하나의 scalar로 옮길 수 없다. Up/gate의 8행 scale tile은 F32 VRF 7680B, down의 12행 scale tile은 5760B를 사용한다.

Q/K/V의 weight row pitch는 이미 256B의 배수다. O의 column stripe도 256B다. 반면 up/gate의 packed row는 1920B지만 **현재 DMA는 32행을 합친 61,440B 연속 구간을 256B packet으로 전송**한다. 행 하나의 길이만 보고 이 전송을 비정렬이라고 판단하면 잘못된 후보를 고르게 된다.

Down의 원래 weight HBM 주소는 `7680*h + 960*shard + local_byte`, scale 주소는 `960*h + 120*shard + local_byte`다. 목적지 DM pitch를 1024/128B로 바꿔도 이 source 주소는 그대로다. 따라서 DM padding만으로 HBM 읽기 경계가 정렬되지는 않는다. 실제 descriptor의 source는 960/120B를 유지했고 destination row stride만 바뀌었다.

Compiler schedule의 Sram extent를 `physical_pages`에 대응시켜 lifetime별 합집합을 구하면 E55의 peak는 QKV 40.26 MiB, O 25.11 MiB, FFN 238.47 MiB다. Alias는 중복 계산하지 않고 기록된 padding은 포함한다. 이는 runtime memory counter나 최악 slice의 점유량이 아니다. Schedule의 aggregate virtual address를 그대로 slice bank 주소나 HBM channel 번호로 해석하지 않았다. 근거와 재계산기는 [schedule-summary.json](../../moa-dev/experiments/candidates/memory-layout-e55/schedule-summary.json), [analyze.py](../../moa-dev/experiments/candidates/memory-layout-e55/analyze.py)에 있다.

## Arena 결과

원래 E55 package를 이번에도 3회 실행했다. 기준 job은 72092/72126/72127이며 QKV **103,858**, O **51,445**, FFN **379,073** cycles다. 아래 변화율은 해당 커널의 이 기준에 대한 값이다. `1회`도 job 안의 서로 다른 공식 입력 3개에 대한 중앙값이며, `3회`는 독립 job 중앙값들의 중앙값이다.

| 후보 | 변경 | 대상 cycles | 기준 대비 | 독립 job 수 | 판정 |
|---|---|---:|---:|---:|---|
| E57 | up/gate packed DM pitch 1920→2048 B | FFN 420,958 | +11.05% | 1 | 회귀 |
| E58 | down packed DM pitch 960→1024 B | FFN 400,543 | +5.66% | 1 | 회귀 |
| E59 | down scale DM pitch 120→128 B | FFN 378,149 | −0.24% | 3 | 개선 입증 못 함 |
| E60 | E58+E59 | FFN 394,900 | +4.18% | 1 | 회귀 |
| E61 | up/gate BF16 DM 행 간격 +32 B | FFN 387,214 | +2.15% | 1 | 회귀 |
| E62 | down BF16 DM 행 간격 +32 B | FFN 382,908 | +1.01% | 1 | 개선 없음 |
| E63 | BF16 QKV의 8→16-way row interleave | QKV 104,525 | +0.64% | 3 | 개선 없음 |
| E64 | O weight를 column-major slice 순서로 적재 후 계산용 순서로 DMA | O 128,411 | +149.61% | 1 | 회귀 |

**8개 후보와 기준을 합친 15개 고유 job에서 225개 비교가 모두 PASS**했다. 원래 공식 harness의 입력·허용오차·공개 signature를 바꾸지 않았다. 후보별 나머지 두 kernel image가 E55와 같고, 측정용 host에 정확한 후보 image가 들어 있는지 확인했다. Collector가 다시 내려받은 실제 업로드 package도 build hash와 일치한다. [전체 결과 연결과 해시](../../moa-dev/experiments/candidates/memory-layout-e55/hardware-summary.json).

E59의 FFN job 중앙값은 **378149 / 381396 / 376421**, E55는 **379073 / 377936 / 381489**다. 분포가 겹치고 각 범위는 약 1.32%/0.94%다. 0.24% 차이를 반복 가능한 개선으로 선언할 근거가 없다. [E59 반복](../experiments/e59-down-scale-repeats.json), [E55 대조 반복](../experiments/e55-memory-layout-control-repeats.json). QKV E63도 **104525 / 104761 / 104236**으로 개선을 보이지 않았다. [QKV 반복](../experiments/e63-qkv-sixteen-row-repeats.json).

큰 회귀 후보는 추가 반복으로 자원을 쓰지 않았다. 새 모델 전체 실행이나 공식 leaderboard 제출은 이번 메모리 실험의 결과가 아니다. 현재 runtime은 기존 E55이므로 이전 모델 실행 근거와 이번 후보의 정확도 검사를 섞지 않는다.

## 원인으로 확인한 것과 남은 범위

E57에서는 원래 `240×256B`로 연속 전송하던 up/gate 구간에 row padding이 생겼다. Compiler가 `32 rows × 2×960B` 읽기/쓰기 대응으로 바꾸었고, 각 weight DMA 예상 시간은 **26,475→32,891** cycles로 늘었다. 전체 Main 작업량은 그대로다. 행 정렬을 강화하면서 연속 전송을 깨뜨린 비용이 확인됐다. 정확한 hardware transaction 수나 stall 비율을 측정한 것은 아니다.

E58/E59/E60은 source HBM 전송 길이가 그대로다. E61/E62는 HBM 배치를 유지하고 BF16 중간 행의 시작 bank를 상대적으로 네 bank씩 이동시키는 실험이다. BF16 행 간격이 32B 늘면 `(row_address >> 3) & 15`가 네 칸씩 바뀐다. E61의 정적 전체 시간은 **162209→161666**으로 줄어도 실제 FFN은 느려졌다. 따라서 정적 schedule만으로 bank 충돌 해결이나 장치 가속을 단정할 수 없다.

E63은 weight DMA 3개의 예상 시간은 같고 작은 scale/gather DMA 합집합을 554 cycles 줄였다. 대신 Main 624와 Sub 576 cycles가 늘어 전체 정적 시간은 **51293→51446**이었다. 기존 E52의 FP8 activation 기반 QKV16 실험과 다르게, 이번 후보는 E55의 BF16 입력과 dot 내부 F32 channel scale을 유지한다. 기존 결과를 새 결과처럼 재사용하지 않았다.

O의 직접 column-major 계산은 `VRU reduce axes must be innermost in Partitioning`으로 컴파일이 거부됐다. 계산 전에 typed DMA로 원래 slice 배치를 복원한 E64는 정확하지만, **HBM weight load 13,383→13,435**에 **DM 재배치 17,112** cycles가 더해졌다. 전체 정적 시간은 **27,239→44,406**, 실제 O는 약 2.50배로 늘었다. 이 비용까지 포함해 제외한다.

FFN에서 더 큰 비용은 남아 있다. 세 matrix의 논리 HBM packed weight+scale만 **94.92 MiB**이고, FP4→FP8→BF16 중간 materialization의 논리 write payload만 **506.25 MiB**다. 후자는 세 행렬의 FP8+BF16 저장량을 더한 값으로 padding·다른 임시 buffer·read traffic은 제외한다. 현재 static Main에서 decode 관련 작업은 86,178 cycles, DMA 구간 합집합은 135,607 cycles다. 둘은 겹치므로 합산해 실행 시간으로 쓰지 않는다.

따라서 다음 FFN 후보는 단순 padding보다 **중간 weight 저장·재읽기와 DMA/계산의 실제 중첩을 함께 줄이는 설계**여야 한다. 이전 E56의 16행 staging은 정확도 실패 기록이 있으므로 그대로 채택하지 않는다. QKV/O도 현재 row/column 배치가 유일한 최적해라고 결론 내리지 않는다. 이번에 시험한 정렬·순서 변경에서 개선을 확인하지 못했다는 범위로 한정한다.

## 재현과 구현 제약

[후보 폴더](../../moa-dev/experiments/candidates/memory-layout-e55/README.md)에 14개 source 변형을 보존했다. 8개는 image/host를 만들고 NPU에서 실행했고, 6개는 아래 컴파일 제약으로 제외했다.

- 분할된 L 축에 직접 padding한 down packed/scale/both 3개는 `cannot factorize common axis L`로 거부됐다. HBM의 기존 `8×1920`, `8×120` 인자를 별도 이름으로 표현해 DMA하고, 같은 physical order와 padding을 유지한 채 L 이름을 복원한 3개가 E58/E59/E60이다.
- BF16 행에 8B만 추가한 up/gate·down 2개는 32B commit packet이 destination의 최내부 연속 구간과 일치하지 않아 거부됐다. E61/E62는 packet 경계를 유지하는 32B 간격을 사용했다.
- O 직접 column-major 1개는 앞서 설명한 reduction 축 제약으로 거부됐다. E64는 실제 재배치 비용을 지불한다.

[verify.py](../../moa-dev/experiments/candidates/memory-layout-e55/verify.py)는 변경 파일 범위, unchanged FFN/O 연산 stream, QKV 행 이름 변경 전후 dot body, 모든 Q/KV 행·scale·head 대응, down의 30,720개 row-shard 구간과 padding 보존을 검사한다. 정수 좌표 증명과 NPU 정확도 검사는 별개다.

격리 build의 46개 실제 입력을 해시로 검사했고 exact image와 host 내부 image를 비교했다. 공유 Cargo target이 다른 scratch crate를 fresh로 취급한 중간 host는 검증에서 거절해 **제출하지 않았다**. 이후 해당 package의 release unit만 clean해서 재빌드했다. 실패 source/log도 보존했다. Raw schedule·graph·대용량 binary는 ignored artifacts에, source/patch·검증/결과 JSON은 Git에 남긴다.
