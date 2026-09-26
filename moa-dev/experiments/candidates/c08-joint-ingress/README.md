# D-C08-JOINT-INGRESS — E105 descriptor 검증

시작 branch `diag/c08-joint-ingress`, Git parent `cf30ad7`. 소스는 E104 commit
`7f891f9c48e670fe014b2ac5bd2c92bae3bac433`에 frozen C08 v2의 네 파일을 그대로 덮은
E105다. 비교 recipe `baseline/e104.json`은 source 선택과 분리한다.
새 후보나 최적화가 아니라 [기존 의심 패턴](../../../../docs/research/c08-joint-ingress-audit-20260927.md)의 bounded diagnostic이다.

기존 E105는 joint DM 네 matrix slot에 Q2/K1/V1 HBM weight와 channel scale을
padded destination tile로 직접 읽었다. Job74530에서 QKV9비교가 실패했지만 원인은
확인하지 못했다. 이번에는 동일 source·input 계약으로 exact compile 한 번만 실행하고
schedule·IR·vISA·graph를 보존한다. 기존 exact image와 byte identity가 확인되어야
새 descriptor를 E105의 증거로 취급한다.

첫 stop 조건은 Q/K/V와 scale HBM binding → source offset/extent/stride → 마지막 byte의
범위 판별이다. Production source 변경, host build, Arena 실행은 하지 않는다.
잘못된 범위가 입증되어도 compact ingress → 명시적 배치 copy는 설계·추정 byte 비용만
기록하고 구현하지 않는다. Static duration이나 descriptor byte 수를 NPU 시간·bus traffic으로
환산하지 않는다.

## Exact E105 재현 결과

`exact-v2` compile 1회로 기존 image와 byte identity를 확인했다.
152,320B, SHA256 `0d3cf9807cd6ddca31c0115e569ff51edb1c577370859b5cbd54b11dc004397e`.
Compiler makespan도 100,091로 같다. [빌드 기록](build-exact-v2.json),
[전체 graph](evidence/exact-v2/graph.json), [descriptor 분석](descriptor-analysis-exact-v2.json).

| HBM 입력 | 논리 bytes | descriptor volume | 마지막 byte 다음 주소 | source bounds |
|---|---:|---:|---:|---|
| Q weight |15,728,640|15,728,640|15,728,640|PASS|
| K / V weight 각각 |7,864,320|31,457,280|7,910,400|FAIL: 46,080B 초과|
| Q scale |8,192|8,192|8,192|PASS|
| K / V scale 각각 |4,096|16,384|4,120|FAIL: 24B 초과|

K/V weight descriptor의 local extent는 `240×256=61,440B`인데 다음 row-group
source pitch는 15,360B다. 입력의 유효 네 행 대신 padded 열여섯 행 범위를 읽는다.
Q는 별도 `QkvMatrix_1=2` loop를 보존하여 정상이다. Scale도 K/V의32B packet이
8B source pitch와 결합한다. Packet별 전수 열거에서 K/V weight 각각92,160B,
scale 각각48B의 범위 밖 descriptor volume이 나온다. 이는 중복을 포함한 volume이며
고유 초과 범위나 실제 측정된 bus traffic과 다르다.

따라서 **E105에는 K/V HBM transfer contract 결함이 있다.** 당시 실패를 일반적인
joint QKV architecture 실패로 취급하면 안 된다. 이 결함이 모든 accuracy 오류의
유일한 원인이거나 수정 뒤 성능이 좋다는 결론까지 증명한 것은 아니다.

Output의 세 typed gather도 전수 검사했다. Graph의 IndexAccess 상수0/2/3과
BF16 element stride4를 반영하면 slice별 Q `[0,16)`, K `[16,24)`, V `[24,32)`를
읽는다. 전체 source/destination byte 대응, 유효 output coverage와 중복 없음이
검증되었다. 이 gather에는 동일한 범위 확대가 없다.

수정 설계는 Q ingress를 유지하고 K/V weight·scale만 compact DM으로 읽은 다음
정확한 길이의 packet stream으로 joint slot에 복사하는 것이다. 단순 padded DMA
재복사는 M07의 실패와 같을 수 있어 그대로 쓰지 않는다. Weight descriptor volume은
75MiB→필요30MiB, scale은40KiB→필요16KiB가 된다. 대신 K/V weight의 DM read+write
30MiB와 scale read+write16KiB, 임시 buffer/작업 경계가 추가된다. 이 수치는 설계상
byte 수이며 allocator padding·실행 시간·speedup의 예측이 아니다. 통합 dot이 모든
weight를 기다리는 제약과 Q/K/V head 처리의 늦은 시작도 남는다.

```bash
python3 moa-dev/experiments/candidates/c08-joint-ingress/build.py NEW-LABEL
python3 moa-dev/experiments/candidates/c08-joint-ingress/analyze.py --check
```

분석 검증은 기존 결과를 덮어쓰지 않는다. 원본 v2 재현과 descriptor 판별의 첫 gate는
완료했으며, 수정 source/host/NPU 실행은 이 진단 commit에 포함하지 않는다.
