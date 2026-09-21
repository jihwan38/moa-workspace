# SDK 0.8.1에서 가능한 scheduling 제어

**확인한 공개 API에는 weight prefetch 순서를 강제하는 barrier·event·priority·lifetime primitive가 없다.** 실제 지원되는 레버는 tensor mapping/tiling, 실제 destination buffer에 쓰는 DMA/commit, Main/Sub 경로 선택, 정적 loop 노출이다. 따라서 E49의 load 호출을 함수 앞으로 옮기거나 helper 인자를 DM으로 바꾸는 것만으로 겹침이 보장되지는 않는다.

2026-09-21, 설치 SDK `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1/`를 읽었다. 아래 `src/...:행`은 그 root 기준이다. 로컬 mirror의 runtime/context/control/lib/memory/commit 6개 파일 SHA256을 설치본과 대조해 모두 같음을 확인했다. Compile·NPU·live source/Git 변경은 없다.

## 실제 API와 보장 범위

| 확인 대상 | 실제 source/API | E49에 대한 의미 |
|---|---|---|
| Device/context | `src/runtime.rs:35–44`: `main`, `sub`, `tdma`, `pdma` 네 필드 | `device.control`이나 별도 공개 scheduling context가 없다. |
| TDMA 제어 | `src/context.rs:49–70`: `DmaContext`; `on`, `device`는 `pub(crate)` | 공개 wait/event/barrier/priority setter가 없다. TDMA를 임의로 여러 queue로 만드는 API도 확인되지 않았다. |
| Control | `src/control.rs:5–9`, `src/lib.rs:30–35,88` | `control`은 숨김 `__loop_hint_unroll` marker뿐이다. 사용자용은 `#[unroll]`; fence나 순서 지정 기능이 아니다. |
| HBM→DM | `src/tensor/memory.rs:373–384` `HbmTensor::to_dm`; `:612–623` `HbmTensorView::to_dm` | typed tensor를 반환한다. 완료 event, 우선순위 또는 prefetch depth 인자는 없다. |
| 기존 DM에 실제 쓰기 | `src/tensor/memory.rs:484–499` `HbmTensorView::to_dm_view`; `src/engine/commit.rs:33–39` `commit_view` | 지정한 실제 destination view를 채운다. 진짜 read/write dependency를 표현할 수 있지만 다른 독립 연산보다 먼저 실행하라는 계약은 아니다. |
| DM 할당 | `src/tensor/memory.rs:753–755,772–780,784–800` `DmTensor::new` | 주소가 없는 handle이며 backend가 배치한다. `address`와 `from_parts`는 비공개다. 공개 주소/bank 지정, `free`, `release`, lifetime-end intrinsic을 찾지 못했다. |
| tile | `src/tensor/memory.rs:523–540` HBM view tile, `:1149–1162` DM view tile, `:1321–1334` mutable DM tile | 실제 필요한 weight 부분만 staging하는 설계에 쓸 수 있다. 특정 mapping의 backend lowering 성공은 별도 검증 대상이다. |
| host async 기능 | `src/tensor/memory.rs:1761–1784,1790–1810`; `src/backend/npu/function.rs:31–79` | `.await`, launch `.wait()`, `Function::alloc`는 host transfer/launch/HBM buffer 경로다. 한 `#[device]` kernel 안의 DM scheduling barrier로 사용할 근거가 없다. |

`DmTensor`의 공개 구현에서 별도 `Drop` hook이나 scheduling-lifetime primitive를 찾지 못했다. 따라서 Rust block 종료 또는 `std::mem::drop(tensor)`를 “여기서 SRAM을 해제하고 다음 load를 시작”하는 명령으로 가정하지 않는다. 이 조사에서는 drop의 특정 MIR lowering을 추가 컴파일해 시험하지 않았다.

공식 [Schedule 문서](https://developer.furiosa.ai/furiosa-opt/book/scheduling/schedule.html)는 context/resource와 실제 memory dependency를 기준으로 독립 연산을 재배치할 수 있음을 설명한다. RAW/WAR/WAW hazard는 보존하지만 written source order로 성능을 추정하지 말라고 명시한다. 같은 문서의 bank/resource 제약 때문에 다른 context라도 직렬화될 수 있다. E49에서 보인 `ExplicitSync(...)`는 emitted schedule의 compiler 노드이며 호출 가능한 public barrier API의 증거가 아니다.

## 후보 하나: 8행 weight tile의 실제 staging buffer 두 개

직접적인 prefetch 강제 API는 없으므로 “priority 설정” 후보는 만들 수 없다. 대신 위의 **실제 tile/`to_dm_view`/`commit_view` API**로 가능한 다음 구조를 검토할 수 있다. 아직 구현·컴파일하지 않은 설계이며 overlap 보장은 없다.

- 전체 32행 BF16 matrix를 만들지 않고, 기존 8행 dot 단위에 맞춰 FP4 input을 8행씩 읽는다. 두 FP4 DM slot에 현재/다음 tile을 보관하고, 현재 tile을 같은 fused chain으로 BF16 변환한 뒤 기존 16원소 block dot/local-scale/F32 reduction/BF16 row-output 경계를 유지한다.
- 다음 FP4 tile의 TDMA와 현재 tile의 Main decode/dot가 같은 scheduling unit에 보이게 한다. Buffer를 다시 쓸 때의 의존성은 실제 tile 데이터의 read/write에서 생긴다. 불필요한 출력이나 가짜 계산 dependency를 추가하지 않는다.
- Scale은 FP8 DM에 staging하고 VRF에는 현재 8행만 적재한다. 8행 scale은 F32 VRF 7,680B이므로 두 개를 동시에 VRF에 보관하면 8KiB 용량을 넘는다. VRF double buffering은 이 설계에 포함하지 않는다.
- Tile 선택 전 HBM의 `Dummy2/L%7680`을 같은 wire order의 row-group/local-row 축으로 나눈다. 이미 tiled view를 뒤늦게 reshape하는 경로는 기존 backend 실패가 있었으므로 사용하지 않는다. 정확한 source/destination mapping은 후속 compile에서 검증해야 한다. Public `#[device]` signature는 그대로이고 private helper만 tile을 받게 바꿀 수 있다.

Slice당 주요 새 working-set 계산은 FP4 slot 두 개 `2×8×3840/2=30,720B`, BF16 tile `8×3840×2=61,440B`, FP8 scale tile 두 개 `2×8×240=3,840B`, 현재 fused lookup table 약 8,192B, activation DM 7,680B로 **111,872B**다. 이는 이 부분의 subtotal이다. Compiler가 미리 적재할 수 있는 down weights, 다른 tensors, alignment를 포함한 kernel 전체 peak나 fit 보장은 아니다.

공식 [Tuning의 double buffering 설명](https://developer.furiosa.ai/furiosa-opt/book/scheduling/tuning.html#double-buffering)은 두 stage 또는 full unroll로 다음 staging과 현재 contraction을 함께 보이게 하는 방법을 다룬다. 예시는 Sub→TRF이며 위 TDMA→DM 후보의 성능을 직접 검증한 것은 아니다. E49의 기존 `static_passes!`는 이미 4개 row pass를 펼치므로 단순 `#[unroll]` 추가는 새 레버가 아니다. 후보의 실제 변경점은 **transfer 단위와 materialized buffer 크기/재사용**이다.

이 구조는 arithmetic을 유지할 수 있는 API 기반의 가설이다. 유효성을 보려면 실제 slice/bank layout, alias hazard, schedule의 gate weight overlap을 확인해야 한다. 현재 E49 compile-only 제외 상태와 E48 hardware 결과는 그대로 유지한다.
