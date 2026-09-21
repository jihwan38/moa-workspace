Ring32 broadcast를 Sub에서 공통 BF16 TRF로 직접 적재하려던 한정 후보다. **첫 exact compile이 MIR bitmap 검증에서 실패해 종료했다.** 유효 kernel image, graph, schedule이 생성되지 않았고 추가 compile·NPU 실행은 하지 않았다. 이번 source/mapping 조합의 실패이며 모든 Sub→TRF fusion이 불가능하다는 결론은 아니다.

기계적 기준은 [shared-TRF 후보](../qkv-bf16-shared-trf-e55/README.md)(projection SHA`26b7ada776900a82ccc716deaa26e1f6d7ed55c4e86fce9f1098132e1d766fb5`)다. 이 기준의52,562는 정적 회귀로 미채택한 값이다. 성능 비교 대상인 E55 fused 원형은51,293이지만, **이번 후보에는 비교할 유효 makespan이 없다.** 제거하려던 기존 Main broadcast7,943와 Sub TRF1,223을 합해 예상 성능으로 제시하지 않는다.

## 시도한 연결과 사전 증명

기존 HBM 입력, H480×8 pre-RMS, BF16 cast, typed DMA의 H3840 ring-leader gather는 그대로다. Ops에서 Main broadcast→full-copy DM commit과 이후 DM reshape만 제거했다. `prepare_shared_input`은 `[Dummy256/32,1#32]` leader DM을 받아 다음 경로를 시도한다.

```text
실제로 채운16개 leader(2 clusters×8 rings)의 BF16 H3840
→Sub.fetch(Time1, PacketH)
→CustomBroadcast32(OutSlice=Dummy256)
→collect(TimeH/16, PacketH%16)
→공통 TRF(Lane1, H3840)
```

Cluster만 기존의 실제 두 copy를 나타내는 named `Dummy2`로 해석했다. Source Slice의 `[Dummy256/32,1#32]` padding은 그대로다. Source를 full256 유효 slice로 바꾸는 reshape나 TRF transmute는 없다. 출력의 named `Dummy256`은 Switch가 실제 복제한 뒤 만들어야 하는 layout이다.

수학적으로 원하는 대응은 output `(c,s,h)`가 input `(c,32*floor(s/32),h)`를 읽는 것이다. Preserved typed gather는 leader0,32,…,224 각각에 H0..3839 전체를 쓴다. [verify_candidate.py](verify_candidate.py)는 compile 전에2×256×3840=**1,966,080개 출력 좌표**의 원천이 초기화된 leader인지 검사했다([사전 결과](verification-before-compile.json)). 이는 **의도한 routing의 좌표 증명**이다. 아래 compiler 거부로 인해 실제 lowered routing이 이 좌표를 구현한다고 증명한 것은 아니다.

`project_query`부터 projection 파일 끝까지는 shared 기준과 byte-identical이다. Q/K/V weight/scale mapping, DPE/reduction, F32 channel scale 뒤 첫 BF16 cast, head RMS/RoPE/cache, O와 FFN, public signature·harness는 변경하지 않았다. Private input 준비만 바꿨다. 나머지41개 source와 Cargo/toolchain도 동일하고 [46개 실제 compile 입력](compiled-source-manifest.json)을 host snapshot과 대조했다.

## 정확한 실패와 중단 이유

[compile-attempt1.log](../../artifacts/qkv-bf16-broadcast-trf-e55/compile-attempt1.log)의 진단은 다음과 같다.

```text
mir: Switch snoop bitmap disagrees with the (OutSlice x OutTime)
enumeration at output slice 1
--> src/device/sliding/projection.rs:22:10
.to_trf()
```

SDK0.8.1 `engine/switch.rs:57–72`와 `engine/collect.rs:74–86`은 const context generic으로 `Sub.switch→collect→to_trf`라는 public API 순서를 노출한다. 그러나 그 타입/API 존재가 arbitrary named-output bitmap의 lowering 성공을 보장하지 않는다. 이 source에서는 bitmap 검증 단계에서 막혔다.

기존에 검증된 output `[Dummy256/32,32]`는 마지막32개가 anonymous broadcast copy다. 하지만 공통 TRF/DPE의 named `Dummy256`과 타입이 다르다. SDK `tensor/memory.rs:1391–1437`에는 공개 TRF reshape가 없고 `engine/contraction/outer/mod.rs:124–127`은 stream/TRF의 Cluster/Slice 타입이 같아야 한다. 서로 다른 Q/K/V weight row를 anonymous broadcast 축이라고 재해석하면 slice마다 같은 값이라는 잘못된 의미를 부여한다. 이 우회를 쓰지 않았다.

두 번까지 허용된 compile 중 한 번만 사용했다. 안전한 한정 수정의 API/주소 근거를 확보하지 못했고 root가 실패 동결을 지시해 종료했다. 향후 재검토에는 예를 들어 다음 중 하나에 대한 **새로운 실제 지원 근거**가 필요하다: named-output bitmap이 원하는256개 leader 주소를 나타내는 compiler 경로, 또는 실제 broadcast된 TRF의 물리 주소를 보존하는 공개 relabel API. 지금 그런 API가 존재하거나 compiler 수정으로 반드시 해결된다고 주장하지 않는다.

[output-artifact-check.json](output-artifact-check.json)은 isolated scratch에서 image, schedule, graph 세 경로 모두 존재하지 않았음을 기록한다. 그러므로 새 후보의 TRF/DM 예약량, bitmap control lifetime, 전체 비용 또는 하드웨어 정확성은 보고할 수 없다. 실패 전 source의 논리 TRF 값7,680B/slice를 실제 allocator 예약으로 혼동하지 않는다.

## 재현과 동결

Docker `/tmp/moa-qkv-bf16-broadcast-trf-e55`와 그 내부 target/kernel output만 사용했다.

```bash
cd /tmp/moa-qkv-bf16-broadcast-trf-e55
CARGO_HOME=/home/moa/.cargo \
CARGO_TARGET_DIR=/tmp/moa-qkv-bf16-broadcast-trf-e55/target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-bf16-broadcast-trf-e55/target/furiosa-opt/kernel \
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
  --dump-schedule candidate.json --dump-graph candidate-graph.json
```

Projection SHA는 `41d67eee289f5e995f0f4a79f64f03f63e530cd42925742ae27f8d83def3530b`, ops SHA는 `a79a53f749d5161118ef77f52a758babfce4844f72c7fc8ce99a821e66b453c7`다. [prepare_source.py](prepare_source.py)의 재생성 결과와 실제 attempted source가 byte-identical함을 확인했다. [manifest](manifest.json), [patch](candidate.patch), source crate와 실패 로그를 보존했다. E54 checkpoint`2fceb2d`와 tracked [Cargo.lock](../../../locks/gemma4-Cargo.lock)을 기반으로 복원할 때에는 최종 source 두 파일을 적용하고46개 입력 해시를 대조해야 한다.

Live source/target, 이전 candidate, Git, NPU는 변경하지 않았다. 이 실패를 회피하는 추가 variant는 만들지 않았다.
