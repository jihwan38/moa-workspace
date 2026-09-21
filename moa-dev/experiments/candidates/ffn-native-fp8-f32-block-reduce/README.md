# Native FP8 FFN: F32 block partials와 VE scale/reduce

E41 row30+row2의 `project_rows_with_scale_contraction` 본문만 바꿨다. 같은 E41 전체 소스·manifest로 exact `ops::decoder_feedforward`를 비교한 결과 **170,176 → 172,313 cycles(+1.26%)**, 196→286 instructions다. 첫 후보 compile에 성공했고 baseline을 포함해 총2회 compile했다. 중간 BF16 반올림을 제거했지만 속도 개선은 확인되지 않았다. root의 E46 job70652는 공식15 comparisons PASS, FFN434,284로 E41의426,442보다1.84% 느려 미채택했다. 이 subagent는 NPU를 실행하지 않았다.

[mlp.rs](mlp.rs) SHA256: `b9b7d6e07127c4747f3a39f83f1480a8e7e546e3303723e260f4903c60be7551`. [후보 kernel image](../../artifacts/ffn-native-fp8-f32-block-reduce-static/decoder_feedforward.bin)는215,040 bytes, SHA256 `476baf365346b947bd28367dc5e66d0e803596f07ec4a4aad2e947ceaf81744c`다. 코드 차이는 [candidate.patch](candidate.patch), source/compile/schedule 근거는 [evidence.json](evidence.json)에 있다.

## 기준과 변경 범위

[실제 E41 snapshot](../../artifacts/e41-native-row30/provenance.json)의 source43개와 manifest3개를 별도 scratch `/tmp/moa-ffn-native-fp8-f32-block-reduce`로 복사했다. baseline MLP는 `dff506dea523b17c7e32c4c200cda8c5138a800ea61e40ed805da1b1e3df3129`, ops는 `48cfec9c9fd390d54e1ef1e560f984d8968d7c75d3faf41a397663e6a3cd2a1a`다. 기존 row30 scratch는 ops `afc2d119…`였으므로 그대로 재사용하지 않았다. 이번 integrated baseline도170,176으로 같았고 schedule hash도 기존 static `c2d39676…`와 같았다. 동일 수치만 믿고 source를 혼용한 비교가 아니다.

`project_up_and_gate`는 FP4 HBM weight를 row2 배치로 읽고 FP8 DM으로 정확히 풀어 놓는다. `feedforward`의 runtime BF16→FP8 hi/lo quantizer가 제공하는 lane2 TRF와 scale을 넘겨 up/gate에 helper를 각각 호출한다. helper 출력은 기존 BF16 scalar1#4 row30이다. 이후 `compact_full256_rows`의 adjacent2-slice Switch → canonical gather → GeGLU/global scale → 기존 BF16 down → epilogue가 그대로 이어진다. 이 경로의 HBM views, gather, quantizer, down, epilogue와 test harness는 byte-identical이다. 추가 unsafe reshape나 tail row는 없다.

기존 helper는 첫 native FP8 DPE로 각16-element block dot을 계산한 뒤 hi+lo/16, runtime x_scale을 적용하고 **BF16로 저장**했다. FP8 local scales도 BF16 DM으로 풀어 TRF에 올린 뒤 두 번째 BF16 DPE가240 blocks를 합산했다. 후보는 첫 dot/hi-lo/x_scale 명령을 그대로 두고 결과를 **F32 DM[row30,240]**에 저장한다. 다음 다섯6-row tile에서 FP8 scales를 F32 VRF로 정확히 변환하여 partial과 곱하고 H block축을 VE에서 합산한 뒤 BF16 scalar1#4로 저장한다.

## 실제 layout·메모리 확인

첫 pair stream은 `vector_widen_concat`/`vector_final` 다음4개 유효 F32 값을16B씩 commit한다. 그 뒤 별도 `vector_init` stream을 시작하므로 pair zip과 intra-slice reduce를 한 pipeline에 넣는 제약을 피한다. 최종6행 결과는 transpose하지 않고 각 행8B짜리 BF16 scalar1#4로 기존 mutable output tile에 commit한다. 5×6행이 정확히30행을 덮는다.

- F32 partial: slice당30×240×4=28,800B. graph의 실제 producer2개 모두 Float32이고 전체512 physical slices 기준14,745,600B다.
- scale VRF: tile당6×240×4=5,760B. 실제 schedule의10개 scale tensor도 이 크기다. 명시적 `vector_stash`를 추가하지 않았다.
- 첫 pair input TRF payload:2×3840 FP8=7,680B, runtime scale VRF32B로 기존과 같다.
- output: slice당30×8=240B. graph의10개 reduction command는 `commit_in_size: 8B`, `commit_projection: 0+[6:4,4:1]`이며6행 tile을 원래 row30 stride로 쓴다. padding3개 BF16 slot을 유효 데이터로 승격하지 않는다. 기존 gather는 각 행의 scalar1만 읽는다.

[verify_invariants.py](verify_invariants.py)는 source 본문 경계, 첫 DPE/pair/x_scale의 동일성, source46개 비교, 30행의 disjoint tile coverage, scalar padding의 비승격, graph의 F32 dtype·BF16 commit stride를 검사한다. [verification.json](verification.json)은 PASS다. 이는 실제 NPU가 올바른 값을 실행한다는 증명은 아니다.

## 정밀도와 정적 비용

실수식으로 원래 첫 block 결과를 `p_b = s_x*(dot(w_b,hi)+dot(w_b,lo)/16)`라 하면, 기존은 `BF16(sum_b BF16(p_b)*BF16(local_scale_b))`, 후보는 `BF16(VE_sum_b F32(p_b*F32(local_scale_b)))`다. FP8 scale→BF16/F32는 모두 정확하지만 기존 block당 BF16 반올림은 없어졌다. 첫 input FP8 근사 자체는 그대로이며 두 번째 합산 방식이 BF16 DPE→VE F32로 바뀌어 결과의 bitwise 동치나 모든 입력에서 오차 감소를 보장하지 않는다.

CPU RNE 예제로, BF16 입력 `[1, 1/256, 0…, 1/256]`, 해당 위치의 FP4 weight1, local scale1을 쓰면 두 block partial은1.00390625와0.00390625다. 기존 중간 BF16 경계와 최종 cast 결과는1.0, F32 partial을 유지한 결과는1.0078125다. FP8 hi 입력128과0.5도 정확해 이 예제는 제거하는 반올림 경계를 구체적으로 보여준다. NPU 전체 FFN 정확도 검사는 별도다. 입력 및 외부 global scale의 극단값으로 인한 overflow 가능성까지 제거한 변경은 아니다.

| 정적 항목 | E41 baseline | 후보 |
|---|---:|---:|
| 전체 makespan |170,176|172,313|
| 전체 Main command duration 합 |97,895|100,095|
| 전체 Sub command duration 합 |20,337|18,055|
| 전체 DmaLoad duration 합 |122,161|120,485|
| up/gate 각각 첫 native dot |7,484|7,484|
| up/gate 각각 scale 적용 Main |lookup1,388+second dot717|5×641=3,205|

projection당 Main work가1,100 cycles씩 늘고 scale lookup 관련 작업은 제거된다. 이 합을 그대로 latency로 더하면 안 되며 실제 전체 schedule 회귀는2,137 cycles다. shape 선택은6-row VRF 예산과 기존8B scalar transport를 지키기 위한 것이고 새로운 variant 탐색은 하지 않았다.

E41의 [job70562](../../../logs/e41-native-row30-job-70562.log)는 공식15 comparisons PASS/FFN426,442였다. 과거 E22 F32-block 계열은240개 유효 slice, 다른 배치와 입력 scale 순서를 사용했고 timeout했다. 이번 full256/row30 후보와 동일 대조군이 아니므로 E22가 이 경로의 성공·실패를 예측하거나 packet4 자체의 불가능성을 증명하지 않는다. E41이 통과했어도 새 VE reduction 경로의 정확도는 승계하지 않는다. 후속 E46의 공식 입력 통과만으로 모든 입력에서의 정확도 개선을 주장하지 않는다.

후속 [E46 기록](../../../../docs/experiments/e46-native-f32-block-reduce.json) 및 [job70652 로그](../../../logs/e46-native-f32-block-reduce-job-70652.log)는 Q97,797/O45,852/FFN434,284, 공식15 comparisons PASS다. FFN은 E41의426,442보다1.84% 느려 한 번의 screen 뒤 미채택했다. 중간 BF16 cast 제거 경로가 세 공식 configuration에서 실행됨은 확인됐지만, extra configurations나 반복 latency는 검사하지 않았다. `verification.json`의 hardware 미검증 표기는 초기 검사 시점이고, 후속 결과는 `evidence.json.hardware`에 보존한다.
