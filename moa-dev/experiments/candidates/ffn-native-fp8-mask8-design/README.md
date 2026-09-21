# FP8 masked lane8 검토 메모 — 미구현·중단

2026-09-21. Native FP8 retained packet4 timeout의 우회로를 검토했으나 **E35 job70485에서 기존 packet4+Sequential pair 자체를 유지한 full256 경로가 PASS**해 root 요청으로 중단했다. 구현/compile/NPU 검증을 하지 않았으며 채택 가능한 후보가 아니다.

아이디어는 각64원소 input packet에서 네16원소 구간 중 하나만 실제 값으로 남기고 나머지48개를 실제0으로 만들어 TRF mask lane4를 구성하는 것이다. 기존 hi/lo2term까지 쓰면8lanes이며, 각 lane의64원소 dot을 packet scalar까지 줄여도 해당16원소 block dot에 해당한다. Mask는 H와 별도 named axis여야 하며, H/16%4를 Lane과 ElementH 양쪽의 동일 축으로 중복시키면 안 된다.

확인한 SDK0.8.1 구현:

- std `engine/collect.rs:144–165`는 TRF총용량65,536B를 frontend에 전달한다. lower `verify/engine/collect.rs:183–244`는 lane1/2/4/8, 전체 bytes, Time의 outer Lane 및 remaining Element 대응을 검사한다. FP8 8×3840=30,720B는 이 용량 조건 안에 있다. 실제 allocator 배치/동시 lifetime은 확인하지 않았다. payload만으로 allocation 보장을 하지 않는다.
- lower `verify/engine/contraction/lane.rs:129–222`는 lane8을 허용한다. packet1, time유지, Sequential이면 accumulator조건은 lane8×inner1×paddedPacket32=256 cells로1024 한도 안이다. 이는 frontend 타입 조건이며 lowering/하드웨어 통과 증거가 아니다.
- std `tensor/memory.rs:1369–1386`의 전체-view `memset(const { f8e4m3::from_f32(0.0) },...)`는 실제 live cells를0으로 쓰는 API다. subview memset은 지원하지 않는다. masked DM 전체를 live로 만들고 먼저0 fill, 각 mask에서 해당16원소 stripe만 typed copy하는 경로가 후보였다. padding이0이라고 가정하거나 추가 FP8 cast로 mask를 만들 계획은 없었다. periodic tile/typed-copy lowering은 시험하지 않았다.
- Lane을 `[Mask4,FfnTerm2]`로 두면 Sequential의 Time은 `[row,H/64,Mask4,FfnTerm]`가 되어 hi/lo pair를 원래 순서로 합칠 수 있다. 합친 F32 DM에서 `[H/64,Mask4]`를 logical H/16으로 복원할 때는 물리 padding을 그대로 유지하는 wire-order 증명이 필요하다. 실제 코드를 만들지 않아 이 경로도 미확정이다.

E35 성공은 retained packet4를 없애야만 실행된다는 주장을 반박한다. 이 메모의 lane8 설계가 필요하다거나 기존 timeout 원인을 해결한다는 결론은 없다. 후속 원인 구분은 `docs/experiments/e35-full256-native-cause-audit.md`를 따른다.
