# SDK 0.8.1 mixed-dtype DPE 감사

2026-09-21. **설치된 공개 SDK 0.8.1의 `contract_outer`는 FP8 stream × BF16 TRF를 지원하지 않는다.** Operand를 반대로 배치해도 같다. E55의 BF16 activation을 보존하면서 weight lookup만 제거하는 직접 mixed-dtype 경로는 찾지 못했다. 이는 확인한 API와 공개 문서의 결론이며, 비공개 instruction이나 다른 하드웨어·SDK 버전까지 불가능하다는 주장이 아니다. 이 감사에서는 구현·compile·NPU 실행을 하지 않았다.

## 실제 호출과 타입 제약

E55 [ops.rs](../../furiosa-opt-gemma4-12B/src/ops.rs)의 `sliding_project_qkv`는 입력 RMS 뒤 BF16을 gather/broadcast하여 [projection.rs](../../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs)의 `project_query`와 `project_key_value`에 전달한다. `project_query` 및 `project_one_kv_matrix`의 실제 경로는 다음과 같다.

```text
activation BF16 DM → Sub → BF16 TRF
weight FP8 HBM → FP8 DM → Main fetch_table_lookup<bf16>
→ BF16 stream × BF16 TRF → F32 dot → F32 channel scale → BF16
```

SDK 문서는 stream을 activation, TRF를 weight라고 부르지만 E55는 모델 weight를 stream으로, activation을 TRF로 사용한다. dtype 제약은 모델에서의 이름이 아니라 이 두 물리적 입력에 적용된다. 현재 lookup은 DPE와 같은 Main pipeline에 있어 BF16 weight 전체를 DM에 저장하지 않는다. E55 source SHA256은 `380e57d6502e26086bd6d933ce762f4ad5de18587d52a8ef4aac7fa87e3b9c9f`다.

설치된 `furiosa-opt-std-0.8.1/src/engine/contraction/outer/mod.rs:104–134`는 stream의 `D: ContractionCast`와 TRF의 `TrfD: ContractionWeight<D>`를 동시에 요구한다. `src/cast.rs:352–377`의 float impl은 아래 세 개뿐이다.

| Stream | TRF | Accumulator |
|---|---|---|
| `bf16` | `bf16` | `f32` |
| `f8e4m3` | `f8e4m3` | `f32` |
| `f8e5m2` | `f8e5m2` | `f32` |

`ContractionWeight<f8e4m3> for bf16`와 역방향 impl이 없고, e4m3/e5m2 혼합도 없다. 서로 다른 타입을 허용하는 impl은 정수의 같은 precision family인 i4/i5와 i8/i9에 한정된다. 공식 [Outer의 operand 표](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/outer.html#operand-types)도 같은 제약을 명시한다.

Outer 구현의 두 입력을 F32로 넓히는 코드(`outer/mod.rs:151–162`)는 이 bound를 통과한 입력의 기능 모델이다. `contract_lane`의 `contraction_prewidened` 호출(`engine/contraction/lane.rs:63–78`)이나 F32 accumulator가 존재한다는 사실은 F32·mixed-float DPE 입력 허용 근거가 아니다. `cast.rs:307–328,379–395`에서 `f32`는 `ContractionCast` 구현 대상이 아니다.

## `fetch_cast`로의 우회도 직접 연결되지 않는다

`src/cast.rs:19–42`와 `engine/fetch_adapter.rs:220–244`를 함께 확인했다.

- FP8→F32 `fetch_cast`는 허용된다. 그러나 결과 stream은 F32이며 `contract_outer`의 입력 bound를 만족하지 않는다.
- FP8→BF16 `fetch_cast`는 없다. BF16→F32와 F32→BF16 변환이 있다고 해서 FP8→BF16 변환도 제공되는 것은 아니다.
- `.fetch_cast::<f32>().fetch_cast::<bf16>()`는 같은 pipeline에서 쓸 수 없다. `engine/mod.rs:267–272`에 `PositionFetchCast → CanApplyFetchCast` 연결이 없다.
- Collect 뒤 Cast Engine으로 BF16을 만든 후 같은 pipeline에서 DPE로 되돌아가는 연결도 없다. `CanApplyContractOuter`는 `PositionCollect`만 받으며, `PositionCast`는 transpose/commit 쪽으로 진행한다(`engine/mod.rs:303–334`).

현재의 FP8→BF16 `fetch_table_lookup`은 `cast.rs:44–85`에 있는 별도의 타입 기반 decode다. `engine/fetch_adapter.rs:177–215`는 이 연산을 Main에만 제공하며, 입력 타입으로 고르는 고정 table을 사용한다. 공식 [Fetch Adapter 문서](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/fetch-adapter.html#table-lookup)도 이 경로와 Main/Sub 구분을 설명한다. 유한 FP8 값은 BF16에 정확히 표현되므로 이 변환 자체는 BF16 activation을 FP8로 양자화하는 방식과 다르다.

별도 pass에서 FP8→F32→BF16을 DM에 materialize한 뒤 다음 pass에서 BF16 DPE를 쓰는 구조는 직접 mixed-dtype dot이 아니다. 그 단순 경로는 Q/K/V weight의 `(4096+2048+2048)×3840`개 값을 기준으로 BF16 DM write 62,914,560 B와 read 62,914,560 B를 추가한다. **HBM 추가 전송량이 아니라 DM staging 양**이며, 실행 시간 추정이나 검증된 새 후보가 아니다. 따라서 lookup 호출을 지우면 그 비용만큼 바로 빨라진다고 판단할 수 없다. Lookup table 준비·공유 비용은 별도 감사 범위다.

## Lowering·ISA 확인 범위

`outer/mod.rs:140–141`은 stream/TRF verifier를 호출한다. 공개 stream verifier는 `D::BITS`를 `StreamAdapterInput.element_bits`로 넘기고, lower의 `config_stream_adapter`는 packet이 storage 기준 32/64 B인지와 lane1/2/4/8 및 mapping을 검사한다. 이 width 검사는 위의 dtype-pair bound를 대체하지 않는다. 한편 설치된 `outer/trf_sequencer.rs:12–19`의 TRF mapping 검사는 TODO stub이다. 이 빈 검사를 mixed dtype 지원으로 해석해서는 안 된다.

`furiosa-opt-lower-0.8.1/src/lib.rs:3–9`는 공개 verification과 prebuilt 비공개 lowering 구현의 경계를 명시한다. `src/sys.rs`는 그 FFI 선언이며, 공개 패키지에서 임의의 DPE dtype pair를 선택하는 ISA constructor나 dtype-pair lowering을 확인하지 못했다. Compiler 내부 `tu-ops`/`npu-visa-translate`의 모든 dtype emission을 소스 수준에서 검증한 것은 아니다.

기존 [E55 lowered graph](../../moa-dev/experiments/artifacts/qkv-bf16-fused-channel-scale-e54/candidate-graph.json)의 Q/K/V `RenegadeCommand#O19/#O33/#O45`는 FP8 DM input과 BF16 TRF를 받아 lookup 후 **BF16 fetch stream**을 operation unit으로 전달한다. 이는 현재 경로가 lookup을 통해 동일 dtype dot을 만든다는 증거다. Mixed pair를 담은 instruction이 존재하지 않는다는 하드웨어 전체 증거는 아니다.

## 설치본 식별

실제 확인 위치는 Docker `moa2026-dev`의 `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`다. std crate의 `.cargo_vcs_info.json`은 commit `9b9cf0fdc78df00cdc430eae725a5ad9084a735e`, `path_in_vcs=furiosa-opt-std`를 가리킨다. 아래는 읽은 파일의 SHA256이며, 위 줄 번호도 이 설치본 기준이다.

| 설치본 상대 경로 | SHA256 |
|---|---|
| `furiosa-opt-std-0.8.1/src/cast.rs` | `a5bc81ff9e06b89f4d5e25567592782af7e99ef90a2789a00c45a49012430d28` |
| `furiosa-opt-std-0.8.1/src/engine/contraction/outer/mod.rs` | `1e8b114ba2b4d04d83cc77ce4cad0e2f08b30358f888b5543eb66aab73a964ac` |
| `furiosa-opt-std-0.8.1/src/engine/contraction/outer/stream_adapter.rs` | `ae72a39d5dcb90b71ca37dc422433140b32d20063709384811ca9201b5a1207c` |
| `furiosa-opt-std-0.8.1/src/engine/contraction/outer/trf_sequencer.rs` | `e739173043a07692aec43185af6a4237570cef36d5fac5c08f691090b5f1f083` |
| `furiosa-opt-std-0.8.1/src/engine/contraction/lane.rs` | `9842e8b17b5854d2bdbc0bbc94ec6046077228e4bb51389beec58cf45d8032df` |
| `furiosa-opt-std-0.8.1/src/engine/fetch_adapter.rs` | `7e06090bccd65935e6e6793761831a9e3b1d08e6ffcfe3eff89f48344319b27e` |
| `furiosa-opt-std-0.8.1/src/engine/mod.rs` | `79627b91df5d73f26843b8a1975851cca4c27260b4581a29e66434f4314791bd` |
| `furiosa-opt-lower-0.8.1/src/lib.rs` | `975096638f6e95a82ea26ec090c61b1b3141ce3c8733d791327bf59f8650b2f8` |
| `furiosa-opt-lower-0.8.1/src/verify/engine/contraction/outer/stream_adapter.rs` | `1f0005455be315b25fa8e5509753db18ea036da96d14968518fa21fe9bed9ba8` |
| `furiosa-opt-lower-0.8.1/src/sys.rs` | `11d1132d0286d4cb292e2b7ba4309cf8e81112105fef391463940e38247ce59f` |
