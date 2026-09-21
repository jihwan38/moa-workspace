# FP8→BF16 lookup table 공유 API 감사 — E55 / SDK 0.8.1

2026-09-21 확인. **설치된 SDK 0.8.1에는 준비된 lookup table을 Q/K/V에 전달하거나 그 주소·수명을 지정하는 공개 API가 없다.** 같은 private helper를 재사용해도 현재 graph는 세 개의 table DMA/`StoTab`을 생성한다. 따라서 E55의 BF16 DPE, scale 순서, row layout을 유지하면서 테이블 준비만 한 번으로 만드는 즉시 구현 가능한 후보는 이번 조사에서 찾지 못했다. 새 compile, NPU 실행, live source 변경은 하지 않았다.

## 실제 경로와 공개 API

대상은 [E55 frozen projection](../../moa-dev/experiments/candidates/qkv-bf16-fused-channel-scale-e54/projection.rs)이다. `project_query:7`의 lookup은 41행, `project_one_kv_matrix:77`의 lookup은 99행이며, `project_key_value:133`가 이 helper를 K/V에 각각 한 번 호출한다. [Frozen ops](../../moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/crate/src/ops.rs)의 `sliding_project_qkv` 72–73행이 두 helper를 호출한다. O는 별도 public device op `sliding_attention_output:151` → `project_output:160` → 181행 lookup이다. 네 projection 모두 FP8 weight를 BF16 stream으로 바꿔 BF16 activation TRF와 contraction한다. QKV는 완성된 FP32 dot에 channel scale을 적용한 다음 첫 BF16 cast를 한다.

아래 경로는 컨테이너 `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1/src/` 기준이다. Host mirror `/tmp/moa-opt-std-release/furiosa-opt-std-0.8.1/src/`와 조사한 7개 파일의 SHA256가 일치함을 확인했다.

| 근거 | 확인된 의미 |
|---|---|
| `engine/fetch_adapter.rs:177–218` | `fetch_table_lookup<OutD>(self)`는 입력 stream만 소비한다. 런타임 table/handle 인자가 없고 `D: TableLookupCast<OutD>`로 변환을 고른다. Main 전용이다. |
| `cast.rs:44–85` | `TableLookupCast`는 scalar `lookup(self) -> D`의 functional model이다. 고정 impl은 F4→F8, F8→BF16이며 table 객체·메모리 관리 API가 아니다. |
| `engine/fetch_adapter.rs:96–123` | `FetchTableLookupTensor`는 변환 직후 stream의 typestate다. 준비된 table 객체가 아니다. 생성자는 `pub(crate)`이며 device borrow와 stream tensor를 보관한다. |
| `context.rs:79–112` | `TuContext`의 공개 진입은 `begin`과 `begin_interleaved`다. table prepare/load/reuse 메서드는 없다. |
| `runtime.rs:35–44,63–89` | Public `Device`는 Main/Sub/TDMA/PDMA context와 launch를 제공한다. table register/주소를 노출하지 않는다. |
| `backend/npu/backend.rs:24–58` | Host runtime의 함수 cache는 device image/function load를 재사용한다. 커널 안의 lookup-table DMA/`StoTab`을 제어하거나 생략하는 handle이 아니다. |
| `control.rs:1–9` 및 `lib.rs` exports | 공개 control의 unroll marker나 exports에 table pinning, table 수명, scheduling priority API가 없다. |

SDK 전체의 lookup 관련 symbol과 위 caller/return type을 확인했다. `furiosa-opt-lower-0.8.1/src/verify/engine/fetch_adapter.rs`는 fetch 폭/정렬을 검증하고, `lowering/mod.rs`와 `sys.rs`는 일부 private lowering의 FFI를 감싼다. 이 공개 source에서 table 공유/CSE 정책을 구현한 코드는 확인하지 못했다. **공개 handle이 없다는 결론과 compiler 내부에서 공유가 원천적으로 불가능하다는 주장은 다르다.** 후자는 이 감사로 증명하지 않는다.

[공식 Fetch Adapter 문서](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/fetch-adapter.html#table-lookup)도 table을 dtype으로 선택하고 런타임 인자를 받지 않는다고 설명한다. 이 문서의 F4 decode에 대한 설명을 FP8 table 준비가 무료라는 근거로 확대하지 않았다. 아래 수치는 실제 FP8→BF16 compiled graph/schedule의 관측이다.

## E55와 O의 실제 table 준비·수명

[E55 schedule](../../moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/candidate.json) / [graph](../../moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/candidate-graph.json)의 전체 makespan은 **51,293**이다. Q/K/V 이름은 source 호출 순서가 아니라 graph의 HBM IO1/2/3 → weight → dot 연결로 확인했다. Graph ID와 schedule ID는 서로 다른 공간이다.

| 경로 | Graph table / dot | Constant DMA | Sub `StoTab` | Dot | Prepared-table lifetime | Slice당 주소 |
|---|---|---:|---:|---:|---:|---:|
| Q | tensor38 / O19 | 24,690–25,528 | 25,528–26,821 | 26,900–31,963 | 25,528–31,963 | 65,536 |
| K | tensor53 / O33 | 3,827–4,665 | 4,870–6,163 | 32,494–35,157 | 4,870–35,157 | 3,584 |
| V | tensor66 / O45 | 34,378–35,216 | 36,463–37,756 | 43,041–45,704 | 36,463–45,704 | 4,608 |
| O (별도 kernel) | tensor12 / O7 | 1,003–1,841 | 1,841–3,134 | 16,157–21,220 | 1,841–21,220 | 256 |

O는 E55에서 유지한 [BF16 O schedule](../../moa-dev/experiments/artifacts/output-bf16-activation-e47/candidate.json) / [graph](../../moa-dev/experiments/artifacts/output-bf16-activation-e47/candidate-graph.json)이며 전체 makespan은 **27,239**다. Table은 dot보다 13,023 cycles 먼저 준비된다. QKV/O는 별도 device 함수/launch이므로 한 함수 내부의 DM 객체를 다음 함수의 resident table로 간주할 수 없다.

세 QKV chain은 각각 graph `DmaCommand O17→StoTab O18→dot O19`, `O31→O32→O33`, `O43→O44→O45`다. 각 constant tensor는 graph상 **4,096B**이고 source 주소는 각각 `Weight+24576`, `Weight+4096`, `Weight+0`이다. DMA 직후 중간 DM도 4,096B다. `StoTab` 출력은 각각 **8,388,608B aggregate = 16,384B × 512 physical slices**이며, graph shape는 slice 내부 `128×8×8` BF16 cell이다. 단순히 FP8 코드가 256개라고 해서 prepared table을 512B라고 계산하지 않았다. 이 expanded/padded physical shape를 임의의 사용자 tensor로 만들어 `fetch_table_lookup`에 전달하는 API는 없다.

각 table DMA는 **838**, Sub 준비는 **1,293** cycles다. QKV의 resource 시간 합은 DMA **2,514** + Sub **3,879** = **6,393**이며, 이 합이 elapsed 시간이나 가능한 speedup은 아니다. K/V 준비는 weight DMA/다른 작업과 겹친다. Q weight는 24,690에 준비되고 Q dot은 table 준비 뒤 79-cycle gap을 더해 26,900에 시작한다. 반면 K table은 6,163에 이미 준비되어 있다. Q와 K의 prepared-table 수명이 겹치는 동안 32KiB/slice가 동시에 예약되고, V table은 K 종료 뒤 준비된다. 실제 table 사용과 allocator 선택을 보존하지 않은 채 6,393을 전체 시간에서 빼면 안 된다.

**Late-K table은 E55 자체가 아닌 [shared-TRF 후보](../../moa-dev/experiments/candidates/qkv-bf16-shared-trf-e55/README.md)의 관측이다.** 그 후보에서는 Q/K table이 slice당 65,536의 같은 주소를 겹치지 않는 수명으로 재사용한다. K weight 준비 31,553 뒤 table DMA 31,553–32,391 → Sub 32,391–33,684 → dot 33,763–36,426이 된다. E55 K dot 시작 32,494보다 1,269 늦고 whole makespan도 52,562로 1,269 증가한다. 주소 재사용과 늦어진 준비는 확인했지만, allocator가 그 주소를 택한 내부 인과관계나 public table handle 하나가 이 회귀를 반드시 고친다는 결론은 확인하지 못했다.

## `fetch_cast`가 대체할 수 있는 범위

`cast.rs:19–42`는 F8→F32와 F32→BF16 `FetchCast`를 지원하지만 **F8→BF16 직접 FetchCast는 지원하지 않는다.** `engine/mod.rs:267–272`에는 `PositionFetchCast` 다음에 두 번째 FetchCast를 적용하는 edge도 없다. F8→F32 stream을 곧바로 BF16 TRF와 contraction할 수도 없다. `cast.rs:352–377`의 floating `ContractionWeight`는 BF16/BF16, E4M3/E4M3, E5M2/E5M2이고 F32는 operand `ContractionCast`가 아니다. 이 제약은 별도 mixed-DPE 감사와 대조했다.

공개 stage만 사용한 별도 **materialization 설계**는 가능하다. F8 DM에서 `.fetch_cast::<f32>()` → `collect` → `.cast::<bf16, ...>()` → `commit`으로 BF16 weight DM을 만든 뒤 기존 BF16 contraction을 수행한다. `cast.rs:151–152`가 F32→BF16 Cast Engine 변환을, `engine/mod.rs:293–341`가 이 stage 전이를 지원한다. Cast 이후 `contract_outer`로 되돌아가는 typestate는 없으므로 중간 DM을 없앤 동일 pipeline 대체는 아니다. 이 설계를 새로 compile하거나 NPU 검증하지 않았다.

유한 E4M3 값은 BF16에 정확히 표현 가능하므로 functional scalar model에서는 같은 값이 된다(`cast.rs:80–85`). 이것만으로 새 hardware conversion 경로의 signed zero/subnormal/NaN 처리나 전체 kernel 정확성을 증명하지 않는다. 성능 면에서도 Q에 60KiB/slice, K/V 각각 30KiB/slice의 BF16 weight를 쓰고 다시 읽는다. 세 projection 합 BF16 materialization 크기만 60MiB이고 readback이 추가된다. Current lookup 준비 2,131 cycles/호출을 없애는 대신 전체 weight stream의 추가 Main/DM 작업과 lifetime이 생기므로, table 공유의 작은 변경으로 분류하거나 이득을 예고하지 않는다.

`begin_interleaved`는 두 개의 같은 dtype/layout DM view를 한 stream으로 시작하는 API다. 런타임 table 공유 인자는 아니다. 현재 Q4096과 KV2048의 row 수/축이 다르며, K/V를 한 stream으로 합치는 것조차 scale selector·commit·head gather를 새로 설계하는 별도 구조 변경이다. 이번 감사에서는 그것을 table 재사용의 검증된 대안으로 제시하지 않는다.

현재 결론은 **직접 table 공유 후보 없음**이다. 진행 근거가 되려면 공식 prepared-table handle 또는 compiler의 같은 decode table 공유를 보장하는 구체적 API/변경 증거가 추가로 필요하다. DM/table 주소를 강제로 맞추거나 fake output dependency로 예약을 유도하는 우회는 검토 결과에 포함하지 않았다.

## 증거 식별자

| 고정 산출물 | SHA256 |
|---|---|
| E55 projection source | `380e57d6502e26086bd6d933ce762f4ad5de18587d52a8ef4aac7fa87e3b9c9f` |
| E55 QKV image | `5bdab7eafec5ac0aca352101deee334180b550c87db23e33c70d008892edba9e` |
| E55 schedule | `79105c61f6e174e6564f837e4680e8ee652e4257197e82443a9b7e23e26a6d12` |
| E55 graph | `fbba7ca3011926ffa3d5d45d6718314f36c741530736b098a73e6b05de12ad70` |
| BF16 O image | `359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b` |
| O schedule | `c33d3a353603781fb9a4f79dde83281a5015abdab72c3f051e4074f571e5c173` |
| O graph | `24d45fc706964d6dbfb7eb3a52a29381dd59347cd714d681ed2ab603f506cee0` |
| 설치 SDK `cast.rs` | `a5bc81ff9e06b89f4d5e25567592782af7e99ef90a2789a00c45a49012430d28` |
| 설치 SDK `engine/fetch_adapter.rs` | `7e06090bccd65935e6e6793761831a9e3b1d08e6ffcfe3eff89f48344319b27e` |

이번 변경은 이 문서 한 개뿐이다. 기존 frozen candidate/source/artifact, live source/target, Git, NPU job은 변경하지 않았다.
