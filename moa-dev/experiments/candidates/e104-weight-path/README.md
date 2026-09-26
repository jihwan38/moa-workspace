# E104 weight 경로 NPU 진단

브랜치 `diag/e104-weight-path`. M02·보완 조사 기록을 main의 `913f9ac`까지 통합한 뒤 시작했다.
소스와 비교군은 모두 `baseline-e104` (`7f891f9c48e670fe014b2ac5bd2c92bae3bac433`)다.
새 최적화 가족이나 제출 후보가 아닌 비용 진단이다. live runtime은 수정하지 않는다.

**완료: 새 경계 커널을 만들지 않고 E104 whole-FFN에서 내부 NPU 구간을 측정했다.**
Arena102797과 동일 패키지의 독립 재실행102842가 모두 성공했다. 전체78개 출력 비교가
통과했다. 이 가운데 진단 host의 FFN 비교는18개, 원본 control의 비교는60개다.
정확도 입력·fixture·허용 오차는 원본과 같다. 새 최적화의 정확도나 속도 향상은 주장하지 않는다.

## 실측 결과

[원본 profile·소스 대응](profile-analysis.json), [120개 작업 전체 대응표](marker-source-map.json),
[첫 job](e104-profile-screen-v1-job-102797-results.json),
[독립 재실행](e104-profile-screen-v1-job-102842-results.json).
아래 각 값은 해당 구간 duration을 cluster별로 구한 뒤 입력마다 큰 값을 선택하고,
세 입력의 중앙값을 구한 것이다. 서로 다른 cluster의 timestamp를 빼지 않는다.

| 구간 | job102797 NPU cycles | job102842 NPU cycles |
|---|---:|---:|
| Gate FP4 weight 적재 | 47826 | 47578 |
| Up FP4 weight 적재 | 48168 | 47982 |
| Down FP4 weight 적재 | 65650 | 66046 |
| Gate FP4→FP8 전체 DM 저장 | 20127 | 20127 |
| Up FP4→FP8 전체 DM 저장 | 20128 | 20128 |
| Down FP4→FP8 전체 DM 저장 | 21075 | 21079 |
| Gate native dot·입력 scale·부분합 저장 | 12023 | 12021 |
| Up native dot·입력 scale·부분합 저장 | 11991 | 11995 |
| Down BF16 lookup·dot·부분합 저장 | 40783 | 40783 |
| 입력 세 항 ring32 방송 | 23415 | 23415 |
| 입력을 양 cluster로 복제하는 DMA 구간 | 41708 | 41528 |

**초반 입력 복제의 긴 duration을 순수 복사 비용으로 해석하면 안 된다.**
예를 들어 첫 job/run1/cluster0에서 Task 시작을0으로 놓으면 다음과 같다.

```text
gate weight DMA:          8159 -------------------------------- 55985
pre-RMS 마지막 Main:               14371--15672
입력 cluster 복제 DMA:                    15675 ---------------- 57431
activation amax Main:                                            58311--60606
```

`mlp.rs:181`의 복제 구간은 약41.8k cycles지만 weight DMA가 끝나고 약1.4k cycles 뒤
완료된다. 전체12개 run/cluster 조합 모두 `pre-RMS 완료 ≤ 복제 시작 < weight DMA 완료
≤ 복제 완료 ≤ activation amax 시작` 순서를 만족한다.
pre-RMS 마지막 Main 종료부터 activation amax 시작까지 다른 Main 연산이 없으며,
이 구간은 중앙값**42190 cycles**, 범위**40495–43845**다.
전체 NPU가 이 시간 동안 놀았다는 뜻은 아니다. Weight DMA는 진행 중이다.

같은 소스의 compiler schedule에서도 gate weight DMA가3589–28201이고,
입력 복제가28201–29161로 바로 뒤에 배치되어 있다. 따라서 **큰 weight 적재와 작은
입력 복제가 같은 전송 경로를 사용하면서 activation 준비가 늦어지는 관계**가 강한 가설이다.
이번 계측은 전송 엔진의 큐·메모리 포트·bank별 stall counter를 제공하지 않으므로
어느 자원의 대기가 정확히 몇 cycles인지는 확정하지 않는다. 42k를 그대로 제거 가능한
시간이나 hardware idle 비율로 환산하지 않는다.

## 설계에 주는 결론

1. **Weight 공급과 입력 준비 순서를 함께 바꾸는 설계가 우선이다.**
   큰 gate weight를 먼저 적재한 뒤 작은 activation/상수 전송이 기다리는 현재 순서를
   줄이는 것이 첫 검증 대상이다. 정규화→입력 분해·배포의 producer/consumer를 함께
   배치하거나 큰 weight 전송을 나누는 설계에서는 새 복제·padding·명령 비용도 포함한다.
   Rust 호출 순서를 바꾸는 것만으로 compiler의 전송 순서가 바뀐다고 가정하지 않는다.
2. **FP8 전체 중간 저장도 별도 큰 비용이다.** Up/gate 각각 약20.1k로 native dot 구간
   약12k보다 길다. 타일 단위 준비·소비 설계는 검토 가치가 있지만 C07의 직접 decode→dot
   정확도 실패 원인을 다루는 구체적 변경이 있어야 한다. 기존 실패 후보를 그대로 반복하지 않는다.
3. **Activation 방송도 약23.4k다.** 입력 표현/복제 구조를 weight 경로와 별개로 붙이면
   Main 자원 경쟁이나 추가 DMA를 늘릴 수 있다. M03 또는 다른 전체 구조를 고를 때 이 비용과
   위42k 대기를 실제로 줄일 경로를 먼저 명시한다. 진단 자체는 E104를 구조 제약으로 만들지 않는다.

구간끼리 겹치므로 표를 합산해 FFN의 비용 비중이나 예상 speedup을 계산하지 않는다.
필수 weight payload를 읽는 시간 전체가 제거 가능한 것도 아니다. 이 결과만으로
HBM 정렬이 잘못되었다거나 FP4 복호화 하나가 단독 최대 병목이라고 결론내리지 않는다.
첫 구간을 개선하더라도 현재 3위와의 FFN41.5% 격차를 모두 닫는다는 보장은 없다.

## 계측의 검증과 한계

- exact FFN compile1회, diagnostic host build1회. Host build는 세 원본 kernel을
  재compile하며 모든 image SHA256이 기존 E104와 일치함을 검사했다.
- host의 FFN profile 이름만 원래 길이의 marker ID로 바꿨다. NPU task code, constant
  bytes, marker 숫자, 입출력 슬롯은 동일하다. 변경은 진단 복사본뿐이며 live source는 E104다.
- SDK runtime은 source 위치 대신 generic operation 이름을 보고한다. 대응은 duration의
  크기로 추측하지 않았다. 정적120개 engine 연산의 **종류·전체 순서**가120개 native marker의
  종류·순서와 일치하고, 모든12개 run/cluster의 실측 시작 순서와도 일치함을 검증했다.
  이것은 순서 기반 대응이며 runtime이 source annotation을 직접 제공한 것은 아니다.
- info/debug는 Task span만, trace는 입력·cluster마다128개 span을 보고했다. 모든120개
  engine 연산, Task1개, 추가 Cluster wrapper7개다. Core/나머지 wrapper에 비용을 임의 배분하지 않았다.
- 원본 before/after FFN 중앙값은 첫 job335683/335387, 재실행336291/337326이다.
  trace FFN은331152/331802로 약1.3–1.5% 낮았고 진단 info/debug도 원본보다 조금 낮았다.
  실행 순서·host 선택·계측 level 차이가 있으므로 이름 변경이나 trace가 최적화라고 주장하지 않는다.
  내부 구간은 **trace가 활성화된 실행의 관측**이다. Clock 단위 변환도 가정하지 않는다.
- profile의 DMA/연산 구간에는 의존성·큐 대기가 포함될 수 있다. 독립 service time이나
  critical-path 절감률이 아니다. 이번 목적에 필요한 전체 경로 관측을 얻었으므로
  추가 isolated probe와 parameter sweep은 실행하지 않고 진단을 종료했다.

검증: `python3 moa-dev/experiments/candidates/e104-weight-path/verify.py`.
재분석: `python3 moa-dev/experiments/candidates/e104-weight-path/analyze.py`.
소스·계측 도구·스케줄·원본 Arena log·다운로드 payload 확인·판정을 함께 보존한다.
공식 제출·runtime 채택·순위 변경은 없다. 다음 단계는 이 비용 근거를 반영한 전체 구조 설계다.

## 실행 전 계획

질문: 실제 FFN에서 weight 적재·FP8 중간 저장과 activation 준비 중 무엇이 소비 시작과
완료를 지연시키는가? C07에서 직접 decode→dot 경로가 일부 projection 정확도 검사에
실패했고 M01/M02는 O projection 실험이었다. 같은 실패 구조를 재구현하지 않는다.

설치된 SDK0.8.1의 `furiosa-opt-abi::image::Profile`과 E104 image를 읽어보니 Task 외에
DMA, Core, Cluster, Renegade 작업의 marker/name이 들어 있다. 따라서 첫 gate는
기존 whole FFN의 세부 profile을 읽는 것이다. 별도 stage 출력이나 사전 변환을 추가하지 않는다.

1. `structural_build.py --plan`으로 E104 source와 비교군을 고정한다. 별도 clone에서
   unchanged FFN을 exact compile하며 schedule/LIR/vISA를 보존하고 기존 image hash와 대조한다.
2. 진단 host만 span name/cluster/begin/end를 기록하도록 수정한다. 공식 입력 생성·fixture·
   accuracy 기준은 유지한다. 순위 제출용 host가 아니다.
3. image의 같은 길이 span name 문자열만 marker ID로 식별되게 바꾼다. Task code,
   marker ID, weights, 입출력 슬롯은 byte-identical이어야 한다. 이 변경은 진단용 복사본에만
   적용하고 원본 image/host 및 대응표를 보존한다. 이름 변경 자체는 NPU 작업을 추가하지 않는다.
4. 원본 E104 info control → 진단 info/debug/trace → 원본 E104 info control 순서로 실행한다.
   높은 profile level의 추가 marker 실행 비용·buffer 제한을 점검한다. 실패 level의 시간을
   성능 수치로 사용하지 않는다. 세 입력 각각의 출력 정확도를 확인한다.
5. 코드와 대응되는 내부 실측을 얻으면 소수의 반복으로 차이를 확인하고 설계 결론을 낸다.
   충분한 attribution이 불가능하면 실제 H3840/L15360 및 E104 mapping을 유지한
   bounded stage probe로 진행하되 출력 경계·DCE·사전 변환 비용을 별도 기록한다.

HBM 접근이 지배적이면 weight 소유권과 접근 순서를, 변환·저장이 지배적이면 타일별
변환·소비 및 기존 정확도 실패를, activation 준비가 지배적이면 표현·복제·TRF 소유권을
다음 설계의 중심으로 삼는다. 원인 불명 구간을 특정 stall로 이름 붙이지 않는다.
중첩 구간을 더하거나 계측 overhead를 원본 시간의 특정 기여율로 해석하지 않는다.
