# E142 공식 결과 이후 우선순위

E142는 채택 및 공식 제출까지 완료했다. 2026-09-27 22:07 KST 조회에서
BEEF는 **6.526159×,46/70위**,30위는7.896678×다. 총점+21.00%가 더 필요하다.
이는 세 커널을 같은 비율로 줄이면 각각17.36%, O를 고정하고 QKV/FFN만 같은
비율로 줄이면 각각24.87%의 감소에 해당한다. FFN만으로 메우려면302799→170921cycles,
43.55% 감소가 필요하다. 다른 팀 점수를 고정한 산술이며 개선 예측은 아니다.
[원본 리더보드](../experiments/leaderboard-e142-20260927.json) · [재계산](e142-rank-gap-20260927.json).

| 커널 | E142 공식 | 현재30위 | 그 팀의 해당 값까지 감소율 |
|---|---:|---:|---:|
|QKV|104474|78524|24.84%|
|O|42694|38138|10.67%|
|FFN|302799|254571|15.93%|

특정 팀과 같은 커널별 조합이 필수라는 뜻은 아니다. 세 speedup의 기하평균이 점수다.
QKV/O image는 E140과 같으므로 공식 실행에서 관측된 QKV+2.02%/O-0.84%를
FFN 변경의 효과로 세지 않는다. FFN의 채택 근거는 독립3allocation의 paired
감소율 중앙3.6193%이며 공식 FFN도3.6004% 감소했다.

## 다음 판별 대상: QKV 후반 공급과 첫 입력 준비

현재 [QKV](../../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs)는 BF16 입력을
TRF에 유지하고 original E4 weight를 lookup하여 dot한다. 채택되지 않은
[E138 v2](../../moa-dev/experiments/candidates/qkv-native-consumer-local/native-v2/projection.rs)의
`project_qkv_native` 경로는 `prepare_qkv_terms`의 소비자 H960 네 항을 TRF에 두고
`project_native_query`/`project_native_kv`에서 original weight를 native로 계산한 뒤
Term/K4 F32 합산과 channel scale, 기존 head/RoPE/cache를 거친다.

E138 v1의 [실제 readiness](../../moa-dev/experiments/candidates/qkv-native-consumer-local/profile/README.md)는
첫 V native가 activation TRF 완료1cycle 뒤, 후반 Q/K는 각각 자기 weight DMA 완료
2cycles 뒤 시작함을 확인했다. 입력 routing을 바꾼 v2는 회귀가12.90%→4.10%로 줄었지만
여전히 현재 BF16 경로보다 느렸고 v2 내부 trace는 없다. 따라서 남은 회귀를 전부
양자화 탓으로 보거나, native 연산 자체가 무효라고 닫을 근거가 없다.

다음 실행 단계는 **current QKV와 frozen E138 v2의 한정된 실제-image readiness 비교**다.
기존 trace/marker 도구를 재사용하고 executable/constants/주소/IO 동일성을 검증한다.
preRMS 종료, 입력 seed/ring, max/마지막 term/coefficients, shared TRF, 각 Q/K/V의
weight DMA와 native 시작·종료, Term/K4 합산, head/RoPE/cache 완료를 연결한다.
표준 whole 전후 대조를 함께 두고 profile 자체의 cycle을 채택 성능으로 쓰지 않는다.
단순 source 순서나 static schedule로 NPU 완료 의존성을 대신하지 않는다.

- 후반 Q/K weight 도착이 계속 native 시작을 결정하면, weight ingress와 projection
  소유권/소비 순서를 바꾸는 완전한 경로를 우선 설계한다. 바뀐 staging·복제·partial
  비용과 공개 API의 실제 target 지원을 확인해야 하며 단순 DMA 쪼개기는 재선정하지 않는다.
- 입력/term/TRF가 완료를 결정하면, 같은 수치 계약의 입력 준비를 우선 설계한다.
  [네 항 bit 구성 검토](after-e142-fourterm-bitwise-review-20260927.md)는 아직 동치식과
  합법적인1–2pass 배치를 확보하지 못했다. 상수계수 생성도 공짜 VRF literal 경로가
  확인되지 않았다. 미완성 식을 구현 후보로 포장하지 않는다.
- 준비가 충분히 앞서 끝나면, 실제 native/reduction/head 종료 구간을 기준으로
  소유권과 계산 표현을 다시 고른다. 겹치는 각 구간을 합산해 예상 이득을 만들지 않는다.

새 진단은 아직 실행하지 않았으며 다음 후보 커널도 정하지 않았다. 위 비교는 구현을
선택할 질문 하나를 해결하기 위한 단계다. FFN row/term sweep과 QKV 입력 routing
변형을 근거 없이 늘리는 방향은 종료 상태로 유지한다.

## 이번 FFN 결과에서 유지할 판단 방식

E141은 큰 FP8 저장을 없앴지만 전체 FFN이3.15% 느렸다. 이를 가족 전체의 실패로
닫지 않고 동일-image 계측으로 첫 native는15434cycles 빨라졌으나 마지막 up native는
10888cycles 늦어졌음을 확인했다. E142는 lookup/native10→2로 줄이는 대신 F32
부분합28.125MiB write/read를 각각 되살렸고 전체가 빨라졌다. 저장 제거 자체보다
그 대신 들어온 작업까지 포함한 완료 시간이 판단 기준이다.
[실측과 검증](../../moa-dev/experiments/candidates/ffn-direct-full30-partials/README.md).

[M06 재검토](m06-copy-retrospective-20260927.md)에서는 v4에 문제의 세 packing DMA가
없음을 확인했다. v5는 위험한 복사 형태지만 실제 descriptor가 없어 조건부 hazard다.
이번480B 복사 수리만으로 두 timeout을 설명하거나 곧바로 재개하지 않는다.
E142 이후 FFN의 남은 지연을 논하려면 바뀐 이미지의 근거가 필요하다.
