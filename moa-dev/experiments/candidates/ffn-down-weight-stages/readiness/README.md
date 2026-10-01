# E150 고정 FFN readiness — job 141326

E150의 마지막 down native 이후 첫 block-scale reduction까지 같은 run/cluster에서 **10,244 cycles 중앙값**이 남았다. Block-scale HBM DMA는 마지막 native 시작 7 cycles 뒤에 시작해 native와 겹치지만, native 종료보다 8,965 cycles 늦게 끝났다. 첫 scale VRF 종료 다음 1 cycle에 reduction이 시작하며, 다섯 scale/reduce 쌍 모두 이 관계가 6관측에서 재현된다. 이는 다음 scale 공급 경로를 검토할 실제 준비 시점 근거다. 10,244를 회수 가능한 시간이나 순수 bus stall로 해석하지 않는다.

고정 source `staged-v1/mlp.rs`와 실제 300032B FFN image `52dc83a983fe11f810545552123a2587746e47eafce46b4c87058943497c4ed2`의 profile 이름 영역만 바꿨다. Task/code/constants/addresses/I/O/marker 수·순서, 나머지 QKV/O images, 원본 입력과 허용오차는 동일하다. `package-profile-v1.json`, `independent-review.json`, `verification.json`이 source46·실제 업로드 archive·142 operations 대응을 검증한다.

한 allocation에 E147 original whole before → E150 diagnostic info → trace → E150 original whole → E147 original whole after를 실행했다. **5 stages / 51 comparisons PASS**. Whole FFN은 304600 → 297229 → 304989; contemporaneous bracket 대비 −2.482164%. Diagnostic info/trace는 각각 295491/298166이며 원본 whole과 구분한다. 이 job은 별도 채택 성능 반복이 아니다. E150의 세 독립 allocation 채택 검증은 상위 `adoption/`에 보존돼 있다.

## 실제 준비 시점

아래 begin/end는 task 기준 3 runs × 2 clusters의 **endpoint 중앙값**이다. 두 중앙값의 차이는 같은 run의 차이에 대한 중앙값과 다를 수 있다.

| Source role (schedule index) | Begin | End |
|---|---:|---:|
| 첫 down weight HBM (430) | 150105 | 183447 |
| Up canonical gather (498) | 180219 | 184125 |
| GeGLU output (513) | 184997 | 186551 |
| Down term gather (631) | 206603 | 208009 |
| Down term leader DMA (637) | 208011 | 215554 |
| Down term ring (646) | 215560 | 227455 |
| Down activation TRF (653) | 227456 | 229231 |
| 첫 down native (657) | 229232 | 239535 |
| 두번째 down weight HBM (647) | 215916 | 250938.5 |
| 두번째 down native (665) | 250940.5 | 262052.5 |
| Block-scale HBM (666) | 250947.5 | 270975.5 |
| 첫 block-scale VRF (681) | 271279.5 | 272255.5 |
| 첫 partial scale/reduce (690) | 272256.5 | 274486.5 |
| 마지막 partial scale/reduce (739) | 285076.5 | 287308.5 |
| Output store (812) | 296871 | 297712 |

Up gather는 첫 weight의 marker 종료 전에 시작해 tail과 겹치며, 완료는 첫 weight 종료 **678 cycles 뒤**다(6/6). GeGLU 완료는 두번째 weight 시작 이전이다(6/6). 따라서 strict predicate `up_gather_geglu_between_down_weight_transfers`는 0/6이다. Gather의 **전체 span**이 두 전송 사이에 있다는 뜻이 아니며, gather 완료와 GeGLU는 첫 weight 이후·두번째 weight 이전에 위치한다.

같은 run/cluster endpoint로 계산한 간격 중앙값:

- GeGLU 종료 → down activation TRF 종료: 43049 cycles; TRF 종료 → 첫 native 시작: +1 (6/6).
- 첫 native 종료 → 두번째 native 시작: 8910 cycles; 두번째 weight 종료 → 두번째 native 시작: +2 (6/6).
- 각 table 준비 종료 → 해당 native 시작: 36300 / 51003 cycles. Table은 이 시작 경계보다 먼저 준비돼 있다. 추가 lookup의 전체 비용이 0이라는 뜻은 아니며, 동시 E147 trace가 없어 추가 setup 원가를 차감 계산하지 않는다.
- 마지막 native 종료 → 첫 reduce 시작: 10244 cycles; scale HBM 종료 → 첫 VRF 시작: 304; scale HBM 종료 → 첫 reduce 시작: 1281.
- 각 scale VRF 종료 → 대응 reduce 시작: 다섯 쌍 모두 +1 (6/6).
- 마지막 native 종료 → 최종 output store 종료: 38364 cycles. 이 suffix interval은 overlapping spans의 합이 아니다.

## 판단과 범위

- **채택:** 고정 E150의 채택 판단은 기존 세 독립 allocation·extra6·raw9·fullFFN 검증에 근거한다. 이번 계측이 그 증거를 대체하거나 임의의 새 kernel을 채택하지 않는다.
- **튜닝:** 이 source에 대한 추가 profiling·row-size/lookup sweeps는 중단한다. 이번 승인 범위의 allocation은 한 번으로 완료했다.
- **다음 연구:** 실제 늦은 block-scale 공급과 바로 뒤따르는 VRF/reduce 경계를 근거로 source-supported 공급 변경을 검토한다. 다른 TDMA/VRF 수명과 전체 completion에 미치는 영향까지 새 complete path에서 측정해야 한다. 단순 let 순서나 10k cycles 회수를 약속하지 않는다.

여섯 관측은 한 allocation의 cluster/run들이며 여섯 독립 반복이 아니다. 동시 E147 내부 trace는 없다. 과거 E148에 보존된 E147 FFN trace는 참고 기록이며 이번 paired control이 아니다. DMA/lookup/native spans는 hardware waits와 overlap을 포함할 수 있으므로 합산하거나 독점 service time으로 해석하지 않는다.

재현(새 NPU 실행 없음):

```sh
python3 moa-dev/experiments/candidates/ffn-down-weight-stages/readiness/independent-review.py --check
python3 moa-dev/experiments/candidates/ffn-down-weight-stages/readiness/analyze.py e150-readiness-v1 141326 --check
python3 moa-dev/experiments/candidates/ffn-down-weight-stages/readiness/verify.py e150-readiness-v1 141326 --check
```
