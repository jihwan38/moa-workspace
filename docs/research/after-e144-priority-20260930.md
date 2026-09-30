# E144 이후: 현재 이미지의 첫 입력 준비와 후반 weight 완료를 한 번만 판별

**다음 실험은 새 커널이 아니라 E142/E144의 실제 이미지에 이름만 바꾸는 paired QKV profile 한 allocation을 권한다.** E144 fullh-v2는 3allocations에서 QKV7.4094/8.0340/7.9907% 개선됐고 standard9, raw9 73728exact, full2 head/cache가 통과했다. 따라서 E138을 느린 native 계열로 폐기할 근거는 없어졌다. 그러나 E144 내부 NPU trace는 아직 없다. D-E143의 Q/K `weight-end+2`, 첫V `TRF-end+1`은 **E138-v2**의 관측이며 E144 병목이라고 재사용할 수 없다. 현재 `baseline/current.json`은 E144이며 source tag `baseline-e144` / `3586c41`로 채택했다. 공식 제출 결과는 별도 기록한다.

**확인된 현재 데이터흐름.** `qkv-fullh-weight-switch/fullh-v2/ops.rs:58`의 preRMS→`project_qkv_native`는 원 BF16 입력을 ring32 seed로 옮기고 BF16 Main broadcast→128 consumer-group의 H3840/K4 scale·네 E4항→shared activation TRF로 만든다(`projection.rs:13–216`). Seed는120KiB, broadcast된 BF16은960KiB, local Term4 DM은1.875MiB이며 Sub TRF load도 남는다. Q/K/V는 original E4 총30MiB를 fullH row-owned DM으로 읽고 Main fetch→InterTranspose→native로 보낸다(`:219–312`). 실제 source slice는row의low2 bits를 소유하고, 전32768 row-quarter와256B DMA 범위가 검증됐다. 새 full-weight transpose DM이나 weight TRF는 없다. 대신 Main Switch/collect·달라진 DM reader 비용이 생긴다. Term/K4 F32합→q0 canonical gather→common s→channel scale→첫BF16→원head/RoPE/cache다. 이 구조 사실만으로 HBM 효율, Switch 지연 또는 quantizer critical 비중을 계산하지 않는다.

**진단의 질문·고정조건·관측량.** D-E143 도구를 그대로 적응하여 actual E142 control와 E144 `687f1c8d…` image의 실행명만 바꾼다. instruction/constant/address/I/O byte동일성, 전체 marker순서와 cluster0-only RoPE/cache visibility를 먼저 확인한다. 한 allocation에서 `E142 whole-before → E142 info/trace → E144 info/trace → E144 whole → E142 whole-after`를 실행한다. 같은 run/cluster 시계로 preRMS 끝, BF16 seed/broadcast 끝, K4max/scale DMA·마지막 term·coeff 준비, sharedTRF 끝, 각 V/Q/K weight와 native 시작/끝, canonical gather/scale/head/RoPE/cache의 마지막 완료를 연결한다. 이름계측 오버헤드는 info/trace 및 whole로 분리한다. 특히 (1)첫V가 어느 producer를 마지막으로 기다리는지, (2)Q/K가 여전히 weight끝 직후 시작하는지, (3)마지막native 이후 head/store가 얼마나 남는지를 본다. 중첩 구간 합계나 old/new DMA duration차를 회수 가능한 cycles로 부르지 않는다.

**관측이 바꿀 다음 선택.**

- Q/K가 여전히 weight 완료를 기다리면 full-DMA→native 완료 경계를 바꾸는 row-stage ingress/consumer 조합을 먼저 설계한다. 공개 `.to_dm`→`.begin(DmView)`에는 부분-DMA readiness token이 확인되지 않아, 명시적인 stage allocation·DMA·native·F32 output조립 비용을 포함해야 한다. C09/O의 두-stage는 가능한 전례지만 QKV 효과를 승계하지 않는다. source `let` 재정렬(E132 image동일)이나 무료 streaming HBM fetch는 후보가 아니다.
- 첫V가 quant/coeff/TRF를 기다리면 그 **실제 마지막 producer**만 대상으로 지원 경로를 검토한다. 네 constant 생성의 scale 의존성 제거는 public memset4+bounded copy4→pack/load라는 비용 교환이 가능하나 free VRF literal은 없다. 네 E4항 bit-constructor는 아직 byte동치식·합법적인 저pass SDK DAG가 없고, direct Sub Switch→TRF는 E55 named-bitmap 실패가 해결되지 않았다. 이들을 이미 구현 가능한 큰 이득으로 선정하지 않는다. `e143-quantization-dependency-review-20260930.md`의 현재 H960 VRF 배치 검토에서도 기존 head와 두 residual을 동시 보존하면9,824B/slice로8,192B를 넘었다. 이는 현재 배치의 제약이며 in-place/rescheduling을 포함한 모든 설계의 불가능 증명은 아니다.
- native 이후가 지배하면 canonical gather/head/cache 소유권을 재검토한다. C03 cluster-head와 E130 Main/Sub head의 whole 회귀를 우회하는 구체적인 producer/consumer 비용 변화가 있어야 하며, 단순 fusion 재명명은 피한다. 먼저 이 진단이 어느 경로를 선택하는지 확인한다.

**수치·다른 커널·순위 범위.** 네 항 `[1,16,2^22,2^27]`, 누적복원잔차, coefficient→Term→K4 F32합과 common/channel scale-before-BF16은 고정한다. 세 항은 일반 BF16 작은값 반례가 있고, 현재 finite oracle는 임의입력/E2E 증명이 아니다. O는 E137-v3이고 내부 trace는 예전v2뿐이다. FFN은 E142 full30 direct+F32partial로 바뀌어 E139/E140전 profile를 현재 병목으로 삼을 수 없다. QKV/O/FFN은 기하평균에 같은 가중치이므로 QKV 확인 후 다른 두 커널의 최신 완료 경로에도 연구를 배분한다. 2026-09-30T13:25Z snapshot에서 E142는58/79위, score6.526159였고 30위8.068830까지+23.6383% score, 세 커널 동일비율이면 각각19.1189% 감소가 필요했다. E144 QKV7.99%를 그 비율에 기계적으로 대입해도 총점효과는약2.81%일 뿐이며 **공식 관측값이 아니다**. E144 공식 결과와 새leaderboard 후 남은 격차를 다시 계산한다. 30위/10위 보장은 하지 않는다.

근거(main에 통합된 파일):

- `moa-dev/experiments/candidates/qkv-fullh-weight-switch/{audit.md,audit-e144-fullh-v2.json,adoption/independent-final-review.json,adoption/performance-summary.json}` 및 `evidence/e144-fullh-v2/{graph,schedule}.json`.
- `moa-dev/experiments/candidates/e142-e138v2-qkv-readiness/profile-analysis.json` / job136455: 이전 native-v2의 후반weight 실측.
- `docs/research/after-e142-fourterm-bitwise-review-20260927.md:53–69`: finiteprecision·O trace시점·constant literal 비용/지원 한계.
- `docs/research/after-e142-priority-20260927.md`, `docs/research/m06-copy-retrospective-20260927.md`: 점수산술·이전FFN시점과 재개한계.

이번 메모는 다음 한정 진단의 제안이다. 새 build/Arena/소스 수정은 하지 않았다.
