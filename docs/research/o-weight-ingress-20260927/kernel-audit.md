# O ingress / memory mapping 감사 — 2026-09-27

읽기 전용 조사다. 새 compiler/Arena 실행이나 kernel source 수정은 하지 않았다. 계산은 `docs/research/o-weight-ingress-20260927/kernel-audit.py`로 재현되며 `docs/research/o-weight-ingress-20260927/kernel-audit.json`에 입력 파일 SHA256, 실제 descriptor, byte coverage, trace 통계가 있다.

핵심 결론은 **현재 O의 weight 공급 완료가 첫 dot을 막는 것은 실측 사실이지만, HBM bandwidth 포화 또는 잘못된 DM bank 배치가 원인이라는 증거는 아직 없다**는 것이다. 실제 descriptor는 15MiB를 중복·누락·범위초과 없이 256B 단위로 읽고, 한 DMA pair의 두 DMN을 packet마다 번갈아 방문한다. 행쌍 보존 4-way interleave는 이 순회의 HBM 주소 간격을 바꿀 수 있는 구체적 가설이다. 다만 기존 480B F32 gather가 8B strided gather로 잘게 나뉘는 비용까지 포함해야 한다.

## 1. 이번에 확보한 exact image 근거

처음에는 E104 profile 디렉터리에 graph가 없어 E53의 동일 weight-layout graph만 찾았다. ignored artifacts를 추가 검색한 결과 `moa-dev/experiments/artifacts/output-bf16-activation-e47/`에서 **현재 O와 byte-identical인 image 및 동반 graph**를 찾았다. 이후의 actual descriptor 분석은 E53 추정이 아니라 이 E47 artifact를 사용한다.

- E47 `candidate.bin`: 71,168B, SHA256 `359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b`. E104 profile 및 현재 E122 O image와 동일하다.
- E47 `candidate-graph.json`: SHA256 `24d45fc706964d6dbfb7eb3a52a29381dd59347cd714d681ed2ab603f506cee0`.
- E47 `candidate.json`과 E104 profile `sliding_attention_output.json`은 source-description 문자열을 제외한 모든 instruction field와 tensors가 일치한다. 32개 instruction, static makespan 27,239가 동일하다.
- E104 profile의 `.visa.txt`는 high-level `Program`이고 `.ir.txt`는 serialized binary이다. 그 파일명만 보고 low-level DMA descriptor를 얻었다고 주장하지 않는다.

실측 근거는 `moa-dev/experiments/candidates/e104-qkv-output-profile/profile-analysis.json`의 job 104914/104933, 12개 run-cluster observation이다. compiler graph의 cycle 예측과 NPU marker interval은 구분한다.

## 2. 실제 호출과 데이터 흐름

`furiosa-opt-gemma4-12B/src/host/load.rs:131`의 `load_f8`가 safetensors FP8 bytes를 HostTensor에서 HBM으로 보낸다. `:293`의 `o_weight`는 `[H=3840, Qs=4096]` row-major FP8이고, `:294`의 channel scale은 BF16 `[H]`다. 호스트에서 O weight를 미리 BF16으로 바꾸거나 transpose하지 않는다.

`src/host/runtime.rs:441`은 sliding attention 결과 BF16 `[8,2,256]`, O weight/scale, post-attention RMS weight, residual을 `ops::sliding_attention_output`으로 전달한다. `src/ops.rs:141–155`의 실제 경로는 다음과 같다.

```
BF16 attention output [4096] HBM
  → replicated local BF16 x[256] DM → Sub StoTrf
FP8 O weight [3840,4096] HBM
  → row-owned / 16 K-column groups / local [120,256] DM
  → Main FP8→BF16 table lookup → BF16 dot with x_TRF
  → F32 sum over contiguous 16 slices → F32 [120] row output
  → DM-to-DM gather into 8 H480 epilogue slices
  → BF16 channel scale converted to F32, F32 multiply → BF16
  → output RMS normalization → residual add → BF16 residual HBM overwrite
```

`src/device/sliding/projection.rs:160–198`이 ingress부터 channel-scale 호출까지 담당한다. Input conversion/amax/FP8 activation quantization이 없고, 모든 stored BF16 activation을 사용한다. FP8 E4 weight는 BF16에서 정확히 표현되므로 lookup은 이 단계에서 근사 오차를 추가하지 않는다. F32 16-column sum과 channel-scale 후 BF16 반올림 경계가 수치 계약이다. 후속 `normalize_output_fused`와 residual 경로도 전체 비교에 포함된다.

좌표를 `c<2, r<16, q<16, t<120, u<256`라 두면 현재 배치는 다음과 같다.

| 항목 | 현재 주소/소유권 |
|---|---|
| 출력 행 / 입력 열 | `h=1920*c+120*r+t`, `k=256*q+u` |
| physical slice | `16*r+q` |
| DMN / DMN 안 slice | `floor(r/2)`, `16*(r%2)+q` |
| HBM weight 상대 byte | `4096*h+k` |
| slice 안 weight byte | `256*t+u` — 30,720B/slice |
| activation | 512B/slice; logical input 8,192B, descriptor payload 262,144B |
| F32 결과 | 120개/surviving row slice; 32개 row owner, logical 15,360B |

activation descriptor에는 source stride 0 반복이 있다. 이것은 broadcast 의미와 descriptor payload를 확인해 주지만, 실제 HBM bus에서 같은 bytes가 몇 번 읽히는지나 hardware multicast가 어떻게 구현되는지 측정한 것은 아니다.

## 3. 실제 weight DMA descriptor

exact graph `DmaCommand#O4`, input T8=`IO(2)+0`, FP8 15,728,640B → output T9 DM local allocation address 32,768이다. 8개의 PE descriptor가 있고 PE p의 HBM base는 `p*1,966,080B`다.

첫 descriptor의 paired sequencer:

```
src: [H_1=120:4096, H_120=2:491520,
      Qs_256=16:256, H_240=2:983040, Qs_1=1:256]
dst: [H_1=120:256, H_120=2:67108864,
      Qs_256=16:4194304, H_240=2:134217728, Qs_1=1:256]
```

마지막 항은 256B packet이다. 따라서 `8×120×2×16×2×256 = 15,728,640B`이며 61,440개 packet이 HBM `[0,15,728,640)`를 정확히 한 번 덮는다. 시작 상대주소가 모두 256B 배수다. HBM runtime allocation의 절대 base는 이 graph에 없으므로 절대 channel 선택까지 확인했다고 말하지 않는다.

공식 문서의 last-axis-fast row-major stream 해석을 적용하면 첫 PE의 상대 source 순서는 `0,983040,256,983296,512,983552,...`다. H240가 두 DMN을 번갈아 선택하며, 각 DMN의 32 slices를 모두 방문한다. 상대 bit8은 `0,0,1,1,...`이고 전체 packet의 절반씩 두 stack bit값으로 나뉜다. 따라서 “한 DMN만 연속 사용한다” 또는 “256B 단위가 아니어서 비효율적이다”는 현재 descriptor 설명으로 맞지 않는다. [DMA engine 공식 문서](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html)

descriptor의 `4,194,304B` slice stride는 주소 인코딩이다. 이것을 slice의 물리 용량으로 읽으면 안 된다. 공식 memory 문서는 slice 512KiB, 16 banks, bank row 8B와 bit3–6 bank 선택을 설명한다. 현재 weight base 32,768와 local row pitch 256B는 매 행 같은 bank에서 시작하지만, 각 256B packet은 모든 16 banks를 각각 16B씩 덮는다. Main은 연속 FP8 원소를 fetch하며 lookup 후 BF16 stream을 사용한다. 주소 coverage의 균등함은 순간 queue conflict 부재를 증명하지 않지만, row-start가 같다는 이유만으로 bank starvation을 단정할 수도 없다. [Memory performance 공식 문서](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html)

공식 문서의 DM `256B/cycle/cluster`와 `2TB/s/chip` 등 서로 범위가 다른 수치를 섞어 실효율을 계산하지 않았다. HBM channel XOR의 전체 구현/절대 base/실제 arbitration counters가 없으므로 source jump 크기만으로 bandwidth 향상을 예측하지 않는다.

## 4. consumer dependency와 SDK가 실제로 노출하는 제어

exact graph의 Main `RenegadeCommand#O7`은 T9 weight, T7 activation TRF, T12 lookup table을 입력으로 가진다. `GReduceAdd ratio=16`, F32 transpose 및 `commit_in_size=8B`가 확인된다. Main task는 weight DMA 완료 뒤에 시작한다. HBM DMA와 dot이 동일 tensor의 부분완료를 이용해 자연히 streaming되는 graph가 아니다.

| 구간 | compiler | NPU median | 해석 |
|---|---:|---:|---|
| O weight DMA interval | 13,383 | 26,736 | weight 생산 완료 경계까지 시간; 순수 bus service / 포화율 아님 |
| dot + 16-slice reduce | 5,063 | 12,014 | lookup·dot·reduce·commit이 묶인 Main interval |
| F32 gather | 470 | 1,218 | 실제 `32×480B` descriptor |
| input TRF ready → dot | — | 20,955 | activation을 더 빨리 준비해도 이 window를 직접 없애지 못함 |
| weight end → dot | — | 2 (모든 12관측) | 현재 직접 prerequisite |
| dot end → task end | — | 9,389.5 | gather, scale, RMS, residual, store 포함 tail |

독립 marker interval의 median들을 더하여 whole-kernel cycle을 재구성하지 않는다. table도 현재 O에서는 weight보다 먼저 준비돼 있다. 따라서 단순히 table을 조금 일찍 만들거나 activation만 재배치하는 것은 관측된 첫 dot 지연을 직접 해결하지 못한다.

설치된 SDK 0.8.1의 읽기 전용 copy는 `/tmp/moa-o-sdk-std`에 있으며 원본 경로와 hash는 JSON에 기록했다.

- `src/context.rs:27–70`: `Dma::{Tensor,Pcie}` / `DmaContext`; public engine ID, DMA stream order, bank 번호 설정 인자가 없다.
- `src/tensor/memory.rs:612–623`: `HbmTensorView::to_dm`의 generic mapping이 layout을 결정한다. `:957–966`의 DM-to-DM도 같은 형태다. 의도한 layout은 표현할 수 있지만 실제 engine/loop 순서는 emitted descriptor를 확인해야 한다.
- `src/tensor/memory/dma_layout.rs:8–90`: destination DMA tail 및 stream stride를 `DM_WRITE_ALIGN_BYTES=8`에 맞춰 검사한다. padding은 실제 transfer 범위를 바꿀 수 있으므로 “tile shape만 맞다”는 검증으로 충분하지 않다.
- `src/tensor/memory.rs:755–772`: DM address는 내부 필드이고 `from_parts`도 crate 내부 API다. shape 변경만으로 원하는 absolute bank 주소를 지정할 수 없다. compiler allocator/lifetime 변경 없이 고정 bank 배치를 약속하면 안 된다.
- `src/engine/fetch_adapter.rs:177–218`: `fetch_table_lookup`은 **Main 전용**이며 별도 준비된 table을 인자로 받지 않는다. public API에서 Sub lookup이나 사용자가 관리하는 resident table 재사용 방법은 확인되지 않았다.
- public `to_dm`에서 slice별 producer 완료 이벤트를 따로 기다리거나, 같은 full tensor의 앞부분만 먼저 소비하도록 요청하는 인자를 찾지 못했다. 실제 early-consumer 경로는 별도 producer tensor/stage를 만드는 C09 계열처럼 graph 경계를 바꿔야 한다. 내부 hardware 기능 부재를 뜻하는 것은 아니다.

## 5. shortlist A — 행쌍 보존 4-way interleave와 직접 F32 gather

이것은 새로운 최적화 family가 아니라 E24/E34 row-interleave family의 재검토다. 과거 8-column row2/row4는 복원 Switch를 추가했고 row2 실제 측정은 회귀했다. 새 가설의 구체적인 차이는 **16-column reduction을 그대로 두고, DMA pair의 H480 소유권 및 연속 F32 두 행을 보존해 canonical gather를 바로 수행한다**는 것이다. 과거 실패 비용이 없어지는지 확인하는 첫 gate가 필요하다.

```
h = 1920*c + 480*g + 8*t + 2*r + j
slice = 64*g + 16*r + q
K = 256*q + u
c<2, g<4, t<60, r<4, j<2, q<16, u<256
```

표현 가능한 mapping 초안:

```
cluster: [Dummy2]
weight slice: [H/480%4, H/2%4, Qs/256]
weight local: [H/8%60, H%2, Qs%256]
reduced output slice: [H/480%4, H/2%4, 1#16]
reduced output local: [H/8%60, H%2]
```

global HBM tensor는 원래 row-major 그대로다. 새 weight-copy stage 없이 source-to-DM mapping이 바뀌며, 각 slice는 여전히 120×256 FP8 bytes를 가진다. Python 좌표 열거상 15MiB coverage는 정확하고 중복이 없다. current descriptor와 같은 loop factor order를 선택한다는 **가정하에서** src strides는 `t:32768, j:4096, b:8192, q:256, d:16384`이며 `r=2*d+b`다. DMN pair source jump가 983,040B에서 16,384B로 줄고 첫 offsets는 `0,16384,256,16640,...`이다. 실제 compiler가 이 order를 내는지는 아직 확인하지 않았다. 상대 bit8 분포는 그대로이므로 이 변화가 HBM channel을 더 고르게 쓴다고 주장할 수 없다.

F32 gather 정렬은 행쌍 j가 보존한다. 같은 slice local source byte는 `8*t+4*j`, canonical destination local byte는 `32*t+8*r+4*j`다. j 두 개를 8B packet으로 옮기면 source stride8B, destination stride32B, row-group offset8B가 모두 정렬된다. 수학상 단일 DM-to-DM mapping으로 H480 gather를 표현할 수 있고 전체 15,360B를 한 번씩 쓴다. 잘못된 초기 4-way식 `h=...+4*t+r`는 같은 slice의 연속 F32가 global H에서4행 떨어져 4B tail 문제가 생길 수 있었으며, 현재 행쌍식은 그것을 피한다.

하지만 current actual gather는 8 descriptors×4 packets×480B=`32×480B`다. 새 mapping의 최소 연속 격자는 `1920×8B`로 packet 수가 60배다. logical bytes와 graph stage 수가 같아도 DM gather service/setup/synchronization은 늘 수 있다. 이것은 “공짜 stride 변경”이 아니다.

첫 gate는 exact emitted descriptor로 weight OOB/duplication/stride/DMN order, F32 gather packet size/coverage를 확인하고, extra copy/restore Switch/padding 확대가 생겼으면 비용 원인을 명시하는 것이다. NPU 진단도 weight-only roundtrip에서 끝내지 않고 **weight + lookup/BF16 dot + F32 reduce + 전체 gather + 기존 epilogue**를 포함해야 한다. weight interval의 절감이 gather 증가를 이기지 못하거나 current weight-ready dependency가 그대로인 채 전체 gain이 없으면 종료한다. 지금은 성능이득을 약속할 근거가 없다.

## 6. shortlist B — 동결 C09의 최초 하드웨어 진단

C09 `two-stage-v1`은 이미 같은 16-column/H-owned 경로를 local60+60 rows로 나눠 다음 weight와 앞 dot을 겹치려 했다. compiler overlap 1,825 cycles가 있었지만 makespan 27,239→27,432(+0.71%)였고 NPU 미실행이다. C12의 Sub reduction 병행은 별도 added partial/padding/pack을 가진 실제 회귀 사례이므로 C09 자체의 NPU 결과로 섞지 않는다.

새 baseline profile에서는 weight와 dot의 compiler→NPU 배율이 서로 다르다. 따라서 +0.71% static만으로 C09의 하드웨어 회귀를 확정할 수 없다. source를 고정한 C09 최초 NPU 측정은 **static 판단이 pipeline 선택을 잘못 닫았는지 확인하는 bounded diagnostic**으로 가치가 있다. 새 architecture나 기대 gain으로 재명명하지 않는다. 이것은 repeated table 제거 방법을 발견했다는 뜻도 아니다.

단순 이상적 equal-half 모델에서 `W=26736, D=12014, W>D`이면 serial `W+D=38750`, 2-stage `W+D/2=32743`이며 잠재 절감은 `D/2≈6007 cycles`다. 이는 각각 반씩 나눈 DMA/dot이 독립적으로 겹치고 table/setup/small-task/gather 추가비용이 없다는 가정이다. 측정된 성능도 아니고 26,736-cycle 전체를 숨길 수 있다는 주장도 아니다. 실제 판단은 matched current control과 whole O 시간, per-stage weight/table/compute overlap, correctness로 한다. 효과가 없으면 tile sweep으로 확대하지 않는다.

O weight-resident TRF의 무복사 대안은 우선순위를 내린다. 8lanes×15rows×256 BF16=61,440B가 frontend 64KiB보다 작다는 계산만으로 API 지원은 증명되지 않는다. Sub table lookup은 public API에 없고, Main lookup→TRF는 같은 61,440B였던 C10 Q 경로에서 StoTrf lowering이 실패했다. C10은 O 실험이 아니므로 O가 영구적으로 불가능하다고 단정하지 않되, 지금 바로 실행할 shortlist로 제시할 근거도 부족하다.

## 7. 수정 시 지켜야 할 구분

- DMA interval은 marker의 dependency window이지 HBM bus의 포화 측정값이 아니다.
- 고정 input layout의 반복 source strides와 multicast의 실제 bus 비용은 다르다.
- descriptor의 정렬·coverage는 확인됐지만 channel conflict와 runtime queue stall은 별도 계측 대상이다.
- current pair interleave가 이미 존재하므로 “DMN interleave를 처음 도입한다”는 설명은 틀리다. 바꾸는 것은 HBM 행 순회와 consumer gather이다.
- 한 mapping/API 호출은 동일 packet 수·동일 memory service 비용을 보장하지 않는다.
- Python mathematical feasibility, 공개 SDK 표현 가능성, compiler emitted descriptor, NPU correctness, whole-kernel speedup은 서로 다른 gate다.

재현 명령: `python3 docs/research/o-weight-ingress-20260927/kernel-audit.py > docs/research/o-weight-ingress-20260927/kernel-audit.json`.

재현에는 이 workspace의 ignored `moa-dev/experiments/artifacts/` 원본과 기록된 SDK copy가 필요하다. JSON은 실제 descriptor와 원본 SHA를 포함한다. SDK copy가 없는 환경에서는 `sdk_manifest.original_container_path`의 0.8.1 파일을 같은 해시로 복원해야 한다.
