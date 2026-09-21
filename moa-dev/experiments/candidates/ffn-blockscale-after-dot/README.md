# FP4 local scale을 16항 dot 뒤에 적용하는 후보

2026-09-21: **FP4×FP8의 BF16 정확 표현은 전체 유한 code에 대해 증명했다. 32B block-dot prototype는 root의 Arena job 70107에서 공식 3-seed 정확도를 통과했지만 FFN은 e6보다 12.37% 느렸다.** FFN median은 820226→921693 cycles다. 기준 source는 frozen `e6-all-unrolled` snapshot이다. 이 작업은 후보와 scratch만 작성했으며 실제 제출은 root가 수행했다. 실제 제출 snapshot의 mlp.rs가 이 후보와 byte-identical인지 확인했다.

## 수치 증명의 범위

`prove_products.py`는 설치된 furiosa-opt-std 0.8.1의 `scalar.rs` FP4→FP8 table과 `float/f8e4.rs` FP8→FP32 table을 읽는다. 모든 table entry를 독립적인 지수/가수 공식과 `fractions.Fraction`으로 대조한 다음 **16×254=4,064개의 유한 곱 전체**를 검사한다. 대회 generator의 특정 exponent band는 사용하지 않는다. NaN code `0x7f`, `0xff`만 유한 입력 집합에서 제외한다.

- 모든 곱이 FP32에 정확하고 하위16bit가0이므로 BF16에도 정확하다.
- 최대 필요한 유효숫자는6bit로 BF16의8bit보다 작다.
- 비영 곱의 절댓값은 `1/1024` 이상, `2688` 이하다. BF16 overflow/subnormal 문제가 없다.
- ±0을 포함하는536개 곱의 부호도 확인했다.

따라서 `BF16(F32(q) * F32(scale))`의 BF16 cast는 유한 FP4/FP8 code에 대해 값을 바꾸지 않는다. 실수 수학으로는 한 block의

```text
sum_i x_i * (q_i * s) = s * sum_i x_i * q_i,  i = 0..15
```

가 성립한다. **FP32 reduction의 반올림까지 동치라는 뜻은 아니다.** 16항 FP32 balanced tree CPU 모델에서 scale-before와 scale-after가 `2.9802322387695312e-8`만큼 다른 반례를 찾았다. 2만 개 독립 block 탐색에서는 마지막 BF16 cast 결과가 다른 예를 찾지 못했지만 이는 동치 증명이 아니다. 실제 NPU는 packet/time/vector의 서로 다른 누적 경로를 사용하므로 CPU tree 탐색을 하드웨어 정확도 검사로 대신할 수 없다. NaN scale, BF16 activation의 극단적인 overflow/underflow, cancellation도 별도 수치 위험이다.

재현:

```bash
python3 moa-dev/experiments/candidates/ffn-blockscale-after-dot/prove_products.py \
  --output /tmp/fp4-fp8-proof-new.json
```

전체 결과와 반례의 code/activation은 `numerical-proof.json`에 있다.

## 구현 경로

새 private helper `project_four_rows_scaled_after_dot`를 up/gate가 공유한다. 기존의 full FP4→FP8 DM materialization은 유지한다. local scale은 원래와 같은 `[row4, H/16]` FP32 VRF에 적재한다.

```text
FP8 DM weight tile [row4,H]
 -> fetch_table_lookup<bf16>
 -> collect: Time=[row4,H/16], Packet=[H%16]
 -> BF16 x BF16 contract_outer, 32B packet
 -> contract_packet<1>: block마다 16항 FP32 dot
 -> contract_time<[row4,H/16]>: identity, block 축 보존
 -> contract_lane: scalar result in padded 8-element FP32 packet
 -> vector narrow-trim to 4-way
 -> FP32 multiply by scale[row,block]
 -> vector_intra_slice_reduce<H>: 240개 block 합
 -> BF16 cast -> transpose -> destination의 기존 4-row tile에 commit
```

`contract_time`의 identity mapping은 0.8.1 source의 `valid_identity` test가 허용한다. [Time reducer](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/time-reducer.html)는 남길 Time 축을 명시하는 API이고, [Packet reducer](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/packet-reducer.html)는 packet 내부의 축만 줄인다. 이 prototype은 32B BF16 packet으로 block 경계를 유지한다. [Outer](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/outer.html)는32B 또는64B operand packet을 허용한다.

down projection, GeGLU, global scales, output dtype, 원래의 4-row static unrolling은 보존했다. `candidate.patch`와 `source-hashes.json`으로 변경 범위를 확인할 수 있다.

## 비용과 남은 검증

up/gate 두 행렬의 논리적인 local-scale 곱은 `2×15360×3840 = 117,964,800`회에서 `2×15360×240 = 7,372,800`회로16분의1이 된다. BF16 expanded weight tile의 DM commit/refetch도 사라진다. 그러나 현재 packet은 scalar block 결과를 padded VE packet으로 전달하고 contraction 폭도64B에서32B로 줄였으므로, **실제 cycle 개선을16배로 해석하면 안 된다**. 작은 단위의 block 결과를 많이 내보내는 비용과 vector reduction의 직렬화가 이득을 상쇄할 수 있다.

검토한 후속 변형은64B contraction에서 `contract_packet::<m![H/16%2]>()`로 두 block을 남긴 뒤 lane fold에서 block 축을 Time으로 옮기는 것이다. 이 변형은 아직 구현·compile 검증하지 않았다.

실행한 검증:

- 전체 유한 code 수치 증명: 통과.
- 독립 `/tmp/moa-ffn-blockscale-api` release build: 통과,20.32초. 기존 official harness로3개 kernel compile 완료.
- 사용 image/compiler/Cargo.lock/candidate/binary hash와 command: `evidence.json`.
- compile 출력: `build.log`.
- NPU 실행: root의 job 70107에서 모든 출력 PASS. FFN cycles는 `[923075, 921510, 921693]`, median 921693이다. 추가 run3..8은 이 변경에 대해 아직 검증하지 않았다.
- exact schedule: e6 392312→403875로 2.95% 증가했다. Main/Vector 작업은 감소했지만 각 tile의 FP8→BF16 lookup table setup이 추가된다. `static-comparison.json`과 `schedule-analysis.md`에서 engine/critical-path 영향을 확인한다.

scratch binary는 컨테이너 `/tmp/moa-ffn-blockscale-api/target/release/test_kernels`에 있으며 SHA-256은 `36cbc65314abb4232f4760941f65c61a8b7bf33d975cfb3eae113ffbfe6a62bc`다. `mlp.rs` 자체 SHA-256은 `e79fc490f16f94959990c32356b7e96999aa69dc8541a8d579d417f2120ad780`이다.
