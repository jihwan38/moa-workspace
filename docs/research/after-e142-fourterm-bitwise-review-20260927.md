# E142 이후 네 E4 항의 직접 비트 구성 — 미선정 연구 검토

2026-09-27. Read-only source/SDK audit; CPU scalar examples만 실행. Compile/NPU/production source 변경 없음. 원 검토 메모를 이 문서와 [입력 해시](after-e142-fourterm-bitwise-review-20260927-digests.json), [CPU 예시](after-e142-fourterm-bitwise-review-20260927-cpu-examples.json)로 보존한다. 검토 당시 채택 baseline은 [current.json](../../baseline/current.json)이 가리킨 [E142](../../baseline/e142.json); QKV는 E138 후보가 아닌 기존 BF16 경로이며 O는 E137 native-v3다.

**판정: 변수 shift가 없어서 불가능한 설계는 아니다. 그러나 현재 네 항과 동치인 1–2 pass 비트 구성식을 확보하지 못했으며, 지금 구현 대상으로 선정할 근거는 부족하다.** 먼저 스칼라 byte-equivalence 식과 실제 SDK stage/ALU 배치가 필요하다. 기존 residual loop를 한 pass에 그대로 접는 것은 public stage 계약상 불가능하지만, 별도로 유도한 병렬 bit-field 식까지 불가능하다는 판정은 아니다.

## 반드시 보존할 계산

[E137 O prepare_output_terms (168–324행)](../../moa-dev/experiments/candidates/o-native-activation-resident/native-v3/projection.rs#L168)와 [E138 QKV prepare_qkv_terms (13–190행)](../../moa-dev/experiments/candidates/qkv-native-consumer-local/native-v2/projection.rs#L13)의 핵심은 동일하다. Max 및 floor로 `s = 2^floor(log2(max(amax,2^-119)))/128`을 만든 뒤 `u = F32(x/s)`에 대하여:

```
f = [1, 16, 2^22, 2^27]
a = 0
q[t] = E4_RNE(F32((u-a)*f[t]))
a = F32(a + F32(q[t]/f[t]))   # 다음 항을 만드는 누적합
```

Native는 네 q의 dot을 계산하고 `[1, 1/16, 2^-22, 2^-27]`로 복원/합산한다. 공통 s와 채널 scale은 그 뒤 F32에서 곱하고 첫 BF16 경계를 지난다. 입력 비트 분해가 normalized value를 재현하는 것만으로 충분하지 않다. **항 자체가 달라지면 native의 네 dot·term reduction 반올림도 달라진다.** 첫 제안은 term별 E4 byte 동일성을 요구하는 편이 가장 작은 검증 범위다. 다른 decomposition은 기존 full boundary 정확성을 별도로 다시 증명해야 한다.

고정 BF16 가수 chunk로 간단히 자르는 것은 동치가 아니다. 기존 CPU 모델과 SDK literal RNE 변환을 사용한 [CPU 예시](after-e142-fourterm-bitwise-review-20260927-cpu-examples.json):

| normalized u | E4 네 항의 값 | 원인 |
|---|---|---|
|1.0625|[1,1,0,0]|첫 tie는 아래 짝수로 반올림; lo residual 양수|
|1.1875|[1.25,-1,0,0]|첫 tie는 위 짝수로 반올림; 양수 입력인데 lo 음수|
|129×2^-22|[0,0,128,32]|hi/lo underflow 뒤 third rounding과 fourth 보정 모두 필요|

Sign, exponent carry, RNE tie parity, E4 subnormal threshold, 잔차 부호를 함께 다뤄야 한다. 기존 normalized BF16-significand finite-domain 검사는 참고 모델이며 arbitrary BF16/F32 denormal 동작 보장이 아니다. SDK F32 semantic model도 subnormal을 zero로 정규화한다. PRNG grid에만 맞춘 단순화는 제외한다.

## 실제 SDK0.8.1의 허용 범위와 막히는 조합

아래 SDK 경로는 설치된 `furiosa-opt-std-0.8.1/src/` 기준이다. 실제 읽은 crate 내부 경로와 각 파일 해시는 [digest JSON](after-e142-fourterm-bitwise-review-20260927-digests.json)에 기록했다. SDK 파일은 저장소 파일과 구분하며, 해당 JSON의 `environment_observed`는 읽은 설치 위치를 찾기 위한 당시 환경 정보다.

1. **가변 shift 표면 API는 존재한다.** `engine/vector/op/mod.rs:29–64,108–158`에 Logic left/logical-right/arithmetic-right, Fxp shift가 있다. `engine/vector/tensor/vector_tensor.rs:1025–1053,1130–1151`의 `vector_logic/vector_fxp`는 immediate뿐 아니라 `IntoBranchedOperand<i32>`를 받는다. `engine/vector/operand.rs:813–832`에 VRF 입력의 같은 Chip/Cluster/Slice, Time/Packet 주소 조건이 명시되어 있다. 따라서 per-element shift amount를 VRF로 미리 만들 수 있다. 다만 그 벡터 계산·적재 비용은 공짜가 아니며, 해당 가변 shift의 새 pipeline target-lowering/NPU 증거는 이번에 만들지 않았다.
2. **Stage가 단방향이다.** `engine/vector/stage/markers.rs:10–13,238–345`: Tag→Logic→Fxp→FxpToFp→Narrow→Fp→Reduce→… 순서다. Fp divide/잔차 산술 뒤 reinterpret해도 Logic/Fxp로 돌아갈 수 없다. `vector_reinterpret`는 scalar와 bits만 바꾸며 stage를 유지한다(`engine/vector/tensor/vector_tensor.rs:1840–1875`). 따라서 먼저 x/s를 Fp로 구한 뒤 같은 pass에서 exponent/mantissa를 추출하는 구성은 안 된다. 정규화부터 integer 영역에서 유도하거나 경계를 한 번 저장해야 한다.
3. **각 ALU는 pass당 한 번이다.** `engine/vector/alu.rs:7–48`, `engine/vector/stage/state.rs:28–35`가 재사용을 거절한다. AND/OR/XOR/좌shift/우shift 각각 하나, Fxp Add/Sub 공용 하나 등이다. `engine/vector/tensor/vector_tensor_pair.rs:213–240`은 두 그룹도 같은 ALU 자원을 공유한다고 명시한다. Pair/unzip이 독립적인 네 범용 ALU 프로그램이 되지는 않는다. Stash도 한 번 저장/한 번 소비이므로 원 비트를 여러 중간 단계에서 반복 사용하려면 VRF 또는 다른 pass가 필요하다.
4. **정수 round-shift는 E4 RNE 대체가 아니다.** `engine/vector/op/semantics.rs:11–21`의 `ArithRightShiftRound`는 tie를 zero 반대 방향으로 보낸다(예: +1/2→1, -1/2→-1). E4는 tie-to-even이다. 최하위 parity 보정을 별도로 구현해야 한다. `Clip` 이후 다시 logic mask로 돌아갈 수도 없다.
5. **Cast는 재귀식의 중간 rounding oracle로 재진입할 수 없다.** `engine/cast.rs:1–5,58–80`은 32B→32B 한 flit cast이며 `engine/mod.rs:307–322`에서 Cast 뒤 VectorInit/ToTrf가 없다. E4 cast→subtract→다음 cast를 한 stream으로 잇는 API는 없다. VE scalar 자체도 i32/f32만 허용(`engine/vector/scalar.rs:5–16`). Raw E4 code를 i32로 만든 뒤 `reinterpret::<f8e4m3>()`하는 직접 경로는 없다. `cast.rs:120–153`의 integer cast는 i32→i4/i8/i16이며, f8 cast는 f32에서만 가능하다. i8 저장 후 raw memory로 dtype을 재표현하는 우회는 별도 검증 사항이며 이번에 지원 증거를 찾지 못했다; 불가능으로 단정하지 않는다.
6. **단일 arbitrary mantissa LUT도 현재 public decode API가 아니다.** `cast.rs:41–92`와 `engine/fetch_adapter.rs:195–215`는 고정 FP4→FP8, FP8→BF16 table만 제공한다. User runtime table handle 또는 BF16→4-byte custom table argument가 없다.

따라서 현실적인 연구 단위는 '각 q를 원 x 비트·scale exponent에서 독립 계산하는 식'을 먼저 유도하고, Term4 중복 fetch/개별 mask·shift VRF를 포함한 1–2 pass DAG에 맞추는 것이다. 원 x의 참조를 Term 축에 복제하면 원본 payload가 같아도 fetched/processed work는 늘 수 있다. 최종 packed E4 DM와 Sub→TRF는 public `ToTrf`가 Collect에서만 지원되므로 자동으로 사라지지 않는다. 고정 native 계수32B/VRF도 arithmetic consumer 계약을 바꾸지 않는 한 남는다.

## 과거 기록과 중복 여부

- [packed term 저장 경로 기록](../../moa-dev/experiments/candidates/qkv-fp8-direct-packed-quant/README.md): packed destination에 직접 commit하고 hi VRF를 tile view에서 읽는 **저장 경로 수리**다. 이미 현재 four-term helper에 반영된 종류이며, 새 bit arithmetic이 아니다.
- [기존 abs/exponent bitmask](../../moa-dev/experiments/candidates/qkv-fp8-split/README.md#L50): abs/exponent bitmask는 이미 사용 중이다. 'power-of-two scale을 비트로 얻는다'는 재제안은 중복이다.
- [C13 INT8 전체 경로 기록](../../moa-dev/experiments/candidates/c13-int8-ffn/README.md): 실제 integer shift/zip→F32가 compile/NPU까지 갔으나, 같은 stream의 scale 결합 오답이 별도 stream에서 해결되었다. 최종 whole +18.58%; 해당 결과는 새 bit constructor의 실패 증거도, Fxp→Fp 임의 결합의 성공 보장도 아니다.
- [RMS stage 제약 기록 (24–30행)](../../moa-dev/experiments/candidates/output-rms-reciprocal/README.md#L24): Sqrt→Stash→Logic(0) 경로가 Way4/Fp→Way8/Logic 제약으로 실제 거절됐다. 위 stage 제약에는 과거 compile 근거도 있다.
- 후보 `.rs`와 docs의 제한된 검색에서 four E4 term code를 sign/exponent/mantissa로 직접 만드는 과거 구현은 찾지 못했다. 전체 git history 부재의 증명은 아니다.

## 우선순위와 첫 판별 기준

E138 v2는 [보존된 판정](../../moa-dev/experiments/candidates/qkv-native-consumer-local/decision-native-v2.json)에서 whole QKV108007 vs103749(+4.1041%, 한 allocation)이며 미채택이다. v1 실측에서 activation TRF 종료1cycle 뒤 첫 V native가 시작했지만, 뒤 Q/K는 각각 weight DMA 완료2cycles 뒤 시작했다([E138 v1 실측 readiness](../../moa-dev/experiments/candidates/qkv-native-consumer-local/profile/README.md)). v2는 입력 routing을 수리했으나 내부 trace가 없으므로 **남은4.1%를 quantizer 비용이라고 확정할 수 없다.** 현재 baseline QKV에는 four-term quantization 자체가 없다.

E137 v3 O는 [보존된 검증 판정](../../moa-dev/experiments/candidates/o-native-activation-resident/decision-v3.json)의 세 allocation paired -14.1496/-12.2978/-12.2654%지만 내부 계측은 이전 v2뿐이다. v3가 없앤512KiB term redistribution과 남은 local512KiB terms/TRF를 혼동하지 않는다. O는 K256, E138 QKV는 H960이라 residual VRF와 소비자 복제 비용이 다르다. FFN D-E141 weight/table 계측을 이 둘의 병목으로 전이하지 않는다.

**다음 판별 순서:** (a) CPU에서 sign/exponent/7-bit mantissa와 admissible scale만으로 기존 네 q byte와 동일한 독립식 제시, tie/carry/zero/tiny/fourth-active 경계 포함; (b) 실제 공개 API로 합법적인 ALU/stash/VRF/cast/packing DAG를 종이에 완성하고 새 scratch와 반복 read 수 계산; (c) 둘 다 성립할 때만 작은 exact diagnostic을 검토한다. 이번에는 (a)/(b)가 성립하지 않아 구현을 선정하지 않는다. 다른 큰 질문은 **QKV의 후반 Q/K weight ingress와 first-consumer 준비의 소유권/동시성**이다. 먼저 동결 v2의 readiness가 어느 쪽에 남았는지 확인해야 quantizer 재설계와 weight-supply 재설계 중 선택할 수 있다. 이 메모에서는 새 NPU 계측을 실행하지 않았다.

## 추가: 네 상수 coefficient를 직접 VRF에 넣는 API

소스 정정: E137 [E137 계수 생성 (295–318행)](../../moa-dev/experiments/candidates/o-native-activation-resident/native-v3/projection.rs#L295)은 **네 Main stream**의 `scale*0+constant`이며 그 뒤 Sub packet transpose이다. 현재 scale을 읽는 의존성은 값 생성에 수학적으로 필요하지 않다. 하지만 다음 공개 API를 확인한 범위에서는 임의의 `[1,1/16,2^-22,2^-27]` array를 device VRF에 즉시 넣는 constructor를 찾지 못했다.

- `tensor/memory.rs:1456–1470`: `VrfTensor::new(Tensor)`는 `pub(crate)`다. Public `zero()`만 있고 arbitrary constant array constructor는 없다. Unit tests의 `VrfTensor::new(Tensor::splat(...))`는 crate 내부 코드여서 사용자 kernel 예제로 쓸 수 없다. `Tensor::from_vec/splat`과 `HostTensor::from_vec`의 존재는 device VRF materialization 증거가 아니다.
- `DmTensor::new()`는 uninitialized handle 계약이다. CPU backing이 zeroed라는 구현을 device 상수0의 증거로 쓰면 안 된다. `DmTensor::from_parts` 역시 private다.
- **실제로 노출된 대안**은 `DmTensorViewMut::memset(value,&mut device.sub)`(`tensor/memory.rs:1368–1386`)이다. Compile-time scalar의 whole allocation fill을 `ParallelMemSet` 한 command로 낮춘다고 명시한다. 그러나 ranged/tile memset은 현재 translation에서 거절되므로 `coefficients.tile(term).memset(c)` 네 번으로 바로 교체할 수 없다.
- Supported source-level 조합은 각각 전체32B constant DM 네 개를 memset하고, bounded32B Sub copies 네 개로 기존128B padded term array를 조립한 다음 현재 pairwise pack/replicate/VRF load를 유지하는 것이다. Scale 의존성과 Main 계산 네 개는 없어지지만 **memset4 + copy4**가 생겨 command 수가 늘고 나머지 pack/load도 남는다. 새 allocation/lifetime와 실제 lowering을 검증해야 한다. 지금 성능 개선 후보로 선택할 근거는 없다.
- `Branched`의 immediate slots도 임의의 lane array가 아니다. `engine/vector/branch.rs:25–45`의 `TagMode`는 Zero, AxisToggle(mod2), data Comparison이다. 네 Term lane ID에 네 값을 무조건 채워주는 primitive를 여기서 확인하지 못했다. 이것을 coefficient VRF 삭제의 증거로 쓰지 않는다.

결론: **상수 계산의 불필요한 scale 의존성은 실제 기회지만, cost-free literal VRF path는 확인되지 않았다.** 공개 memset 조합은 가능한 비용 교환일 뿐이다. 현재 frozen v3에서 coefficient 준비가 quantizer/TRF보다 critical인지도 계측되지 않아, 네 항 bit-constructor 연구와 별개의 작은 기회로만 남긴다.

## CPU 예시 재현 범위

[원 scalar 모델](../../moa-dev/experiments/candidates/qkv-native-weight-resident/numerical/moa-e4-fourterm-audit.py)의 `terms(u, [1,16,2**22,2**27])`와 `sdk_code(term["cast_input"])`를 예시 JSON의 다섯 `u`에 적용해 그대로 재계산할 수 있다. 파일의 `__main__` 전체 실행은 이 검토에 필요하지 않다. 이 다섯 점의 동일성은 NPU 변환·F32 denormal·일반 출력 오차 보장이 아니며, 새 bitwise 구현은 아직 존재하지 않는다. 세 보존 파일에는 개발자별 workspace 절대 경로를 근거 링크로 사용하지 않는다.
