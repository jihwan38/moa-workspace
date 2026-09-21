# E20 V 진단 결과: 첫8 slices의 첫 row

2026-09-21. Root가 실행한 job70363의 [raw log](../../../logs/qkv-fp8-two-lane-diagnostics-job-70363.log)를 분석했다. `docs/experiments/qkv-fp8-two-lane-diagnostics.json`의 다운로드된 package binary SHA256은 `3c2e6ee3a2a672e2bae6070c0e3b619c3d316ad2ad8b5f29bd82e4e45534527c`, fixture SHA256은 `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`로 준비한 파일과 일치한다. 공식 판정은 예상대로 Q/K PASS, V FAIL이며 E20/E23과 run별 오류 통계가 같다.

**직접적인 큰 오차는 각 cluster의 slice0–7, local row0에 한정되는 패턴이다. 같은 head의 나머지 오류는 단일 RMS 배율 변화로 설명된다. 원인이 된 write나 하드웨어 state는 아직 확정하지 않았다.**

## 실제 값에서 확인한 패턴

KV canonical mapping은 `cluster=p/1024`, `slice=p/4%256`, `local=p%4`다. 의심 위치16개는 `{0,4,…,28,1024,1028,…,1052}`다. 이는 각각 head0/4의 처음32개 중 4의 배수다.

공식 tolerance 실패 수는 다음과 같다. 다른6 heads의 실패는 세 run 모두0이며 non-finite도0이다.

| run | head0 | head4 | 전체 |
| --- | ---: | ---: | ---: |
| 0 | 34 | 8 | 42 |
| 1 | 26 | 7 | 33 |
| 2 | 146 | 30 | 176 |

큰 오차의 예: run0 index1052는 expected−0.26953125, actual1.9609375다. run1 index4는 expected0.02587890625, actual−2.90625다. run2 index16은 expected0.875, actual3.9375다. 이는 일반적인 BF16 마지막 자리 차이가 아니다.

각 손상된 head에서 위 의심 위치를 제외한248개 값으로 `beta=sum(actual*expected)/sum(expected²)`를 구하고 `actual/beta`를 원래 expected와 동일 tolerance로 비교했다.

| run | head0 beta | head4 beta | 보정 후 의심 위치 밖 실패 |
| --- | ---: | ---: | ---: |
| 0 | 0.966277902 | 0.996521980 | 0 |
| 1 | 0.968517714 | 0.997541214 | 0 |
| 2 | 0.884336662 | 0.968126609 | 0 |

보정 후 남은 실패는 해당16개 중 일부 또는 전부뿐이다. 의심 위치 밖 최대 보정 오차는 head/run별0.0227–0.0316이며 모두 기존 tolerance 이내다. 이 보정은 **분석에만 사용**하며 공식 oracle/출력/판정을 바꾸지 않았다. RMS 자체가 잘못된 scalar를 읽은 것만으로는16개 위치의 부호 변화와 큰 개별 오차를 설명하기 어렵다. 16개 raw 값의 손상이 후속 head RMS를 바꾸는 가설과는 일치한다.

머리별 actual RMS는 모두 약0.9995–1.0005다. 따라서 V가 정규화되지 않았다는 설명은 데이터와 코드 모두에 맞지 않는다. 전체 통계는 `../../artifacts/qkv-fp8-two-lane-diagnostics/analysis-job-70363.json`, 배율 분석은 같은 폴더 `spatial-pattern-job-70363.json`에 보존했다.

## Schedule 소비자를 추적한 순서 정정

`/tmp/moa-qkv-fp8-split-two-lane/schedules/all-qkv.json`에서 이른 Ps contraction을 함수 호출 순서만으로 K라고 부르면 잘못이다. 실제 소비자 연결은 다음과 같다.

```text
O164 Ps dot (24281–25027) → T86
→ O203 channel scale → T100
→ O204 reshape T101
→ O216 typed copy T106 → O227 T110
→ O266/O271 root_mean_square
→ O285 normalize_value → T127
```

따라서 **이른 O164가 V**다. 늦은 O300 Ps dot(43917–44663)은 K다. 기존 문서의 K/V register lifetime 라벨은 실제 출력 소비자를 확인하여 해석해야 한다. 이 발견은 기존 데이터값을 바꾸지 않지만, “K가 먼저 TRF를 쓰고 나중 V가 재사용해서 실패”라는 설명을 뒷받침하지 않는다.

V는 quantizer lo Main(O117,11482–11883), packed broadcast Main(O135,15115–23058) 뒤의 **첫 two-lane/pair dot**이다. Q dot 및 K dot은 나중이다. quantizer의 유효 slices 역시0–7이며 실패의 물리 위치와 일치한다. 이것은 연관된 관측이며 어떤 vector unit 또는 compiler backend state가 잘못됐다는 증명은 아니다.

초기 확인에서 V contraction T86(address4194304,size4096,24281–33694)의 표시된 SRAM lifetime과 이전 같은 주소 scale T59(8077–17847)는 겹치지 않는다. V pair 임시 VRF T87(address1556480,size65536,24281–25027)도 schedule이 보여 주는 다른 같은 주소 VRF lifetime과 겹치지 않는다. 따라서 눈에 보이는 interval overlap을 찾았다고 주장하지 않는다. device write의 실제 폭이나 내부 state는 이 JSON만으로 검증되지 않는다.

## 다음 분리 실험의 우선순위

1. 첫 pair pass의 operand skip/config를 분리한다. 예를 들어 group0 Mul0의 `()`를 명시적1.0으로 바꾸는 별도 후보는 finite값의 수식은 유지하면서 양쪽 group 설정을 강제할 수 있다. 결과 개선 여부와 schedule을 확인해야 하며 현재 수정안을 검증한 것은 아니다.
2. two-lane contraction을 F32 DM으로 한 번 materialize하고 별도 Main에서 unzip/pair 합을 수행하여 contraction→pair 연결과 math를 분리한다. BF16 cast는 원래 위치에 한 번만 남긴다. 성능 후보보다 원인 구분용이다.
3. 추가 개발용 entrypoint에서 projection의 scale 전/후와 norm 전 값을 읽으면 최초 손상 단계를 확인할 수 있다. 현재 host-only dump는 최종 V만 보여 주므로 이 단계까지 특정할 수 없다.

단순 source call 재배열은 compiler가 다시 schedule할 수 있으므로 실행 순서 분리 증거가 아니다. 특정 fixture나16개 출력 위치만 특수 처리하는 수정은 이 진단의 목적에 맞지 않는다. 현재 활용 가능한 우회는 root가 별도 측정하는 E19 방식 V 두-dot이며, 그것이 통과해도 two-lane 실패 원인이 자동으로 확정되지는 않는다.
