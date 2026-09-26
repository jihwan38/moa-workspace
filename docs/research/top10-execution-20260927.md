# 9월 27일 감사 후 실제 실행 결과

[앞선 감사](top10-audit-20260927.md)의 권고를 실행했다. 작은 이득은 재검증해
채택하고, QKV/O의 실제 준비 경계를 계측했으며, 과거 실패의 전송 계약을 재검사했다.
현재 채택 runtime은 [E122](../../baseline/e122.json)다. 공식 제출 결과는 아직 E104다.

| 실행 | 결과 | 판단 |
|---|---|---|
| M03 재검증 | 6allocation 모두 FFN 감소; paired 비율 중앙값−1.4846%, 새 clean 비교−1.4802% | 동일 소스를 E122로 채택 |
| E104 QKV/O 내부 계측 | 두 allocation,108개 비교 PASS; 모든 QKV76/O22 engine 대응 확인 | 입력 방송만 줄이는 최적화의 근거가 약함. Weight 준비·table 준비·Main 실행 순서를 함께 검토할 우선순위가 높음 |
| Direct FFN 전송 수정 | 독립 정답122880행과 전체 FFN 기본 정확도 통과 | 큰 성능 개선과는 구별해야 함 |
| E123 전체 direct FFN | 단회329395 cycles; E104 앞뒤335836 대비−1.9179%, 동시 M03 331848 대비−0.7392% | 5% 탐색 기준 미달. 반복·확장 정확도·채택 없이 소폭 이득 후보로 보존 |
| C08 원 E105 재현 | 기존 image와 byte-identical; K/V weight·scale DMA source 범위 초과 확인 | 공동 QKV 구조 자체의 실패로 분류했던 판단을 정정 |
| E124 corrected C08 | 45개 비교 PASS; QKV129258 vs 앞뒤 E104103899 cycles, +24.41% | 정확도는 회복됐지만 명백한 회귀. 반복·추가 조정·채택 없이 종료 |
| M06 timeout 원인 감사 | generic code4 및 공통 global-Max/scalar 경계 확인; bank starvation 장치 증거 없음 | 특정 수리 근거 없이 재실험하지 않음 |

M03의 [검증 기록](../../moa-dev/experiments/candidates/m03-producer-local-ffn/validation/README.md)은
동일 stage 실행 파일·fixture·runner를 쓴 과거5개 동반 측정과 새 단독 비교1개를 합친다.
전체 batch 순서가6회 같았다는 뜻은 아니다. FFN 집계 중앙값336142→331354의 비율은
−1.4244%이며, allocation별 비율 중앙값−1.4846%와 계산 방식이 다르다.
기본90개 비교, 추가6개 입력의30개 비교, 독립7-case projection241920개 출력이 통과했다.
두 cluster partial을 합치기 전에 BF16으로 바꾸면 틀리는 상쇄 입력도 포함했다.

이 회수 이득만으로10위권 격차가 해소되지는 않는다. QKV/O가 같다고 가정하면
FFN−1.4846%의 점수 기여는 계산상 약+0.500%다. 이는 공식 채점 결과가 아니다.
감사 당시 필요한 점수 증가 약49.48%와 규모를 구별해야 한다.

## 과거 실패를 다시 보게 만든 전송 오류

공동 tensor의 일부에 `to_dm_view`할 때 destination tile의 논리 extent만 보고
실제 전송 길이를 판단했던 것이 문제였다. 과거 v7c에서 물려받은 FP4 direct FFN 공동 입력 코드를
새 진단에서 컴파일하니 한 projection의30행보다 두 배 넓은 DMA descriptor가 생성됐다. Compact HBM 입력을
받아 유효 payload만 명시적으로 공동 DM에 복사한 뒤에는, 경계 행 fallback 없이
독립 정답을 통과했다. 새 복사는 실행 순서도 바꾸므로 이것만으로 모든 하드웨어 내부
원인이 밝혀졌다고 주장하지는 않는다.
과거 v7c image 자체를 byte-identical하게 재현하거나 당시16행 오류의 유일 원인을
입증한 결과는 아니라는 점에서, 아래 C08 exact 재현과 구별한다.

[C08 exact 감사](../../moa-dev/experiments/candidates/c08-joint-ingress/README.md)는
보다 직접적인 반례다. 각 K/V weight의 원본은7864320B인데 emitted descriptor의
byte volume은31457280B다. 각 source packet은61440B를 읽으면서 다음 slice의
source stride는15360B를 유지한다. 마지막 source end는7910400B로46080B 초과한다.
K/V scale도 같은4배 volume 문제와 마지막24B 초과가 있다. Q ingress와 Q/K/V output
gather의 실제 byte 대응·coverage·bounds는 통과했다.

따라서 원 E105의 정확도 실패와100091 compiler cycles를 근거로 공동 QKV 전체를
배제할 수 없다. 해당 수치는 잘못된 전송을 포함한 실행의 결과다.
수정한 E124의 표준9개 Q/K/V 비교는 모두 통과했지만, 같은 job의 앞뒤 대조보다
24.41% 느렸다. 추가 DM 복사와 모든 weight 준비를 기다리는 구조가 남아 있다.
후보 내부 trace는 수집하지 않았으므로 회귀를 이 비용별로 몇 cycles씩 분해하지 않는다.
이제 이 prototype은 전송 오류로 실패한 상태와 구별되는 **정확하지만 느린 실행**으로 닫는다.

## 다음 최적화에서 바꿀 판단 기준

1. **Padding이 있는 destination view를 쓰면 emitted DMA를 먼저 확인한다.**
   논리 원소 수, packet 크기, source/destination stride, index access, 마지막 byte와
   중복 coverage를 실제 HBM binding까지 연결한다. Source snippet과 schedule duration만으로
   복사 범위를 입증하지 않는다. Compact DMA 뒤 명시적 복사를 추가했다면 그 읽기·쓰기도 센다.
2. **작업을 지운 횟수보다 마지막 consumer가 언제 시작하는지 본다.**
   [QKV/O 실측](../../moa-dev/experiments/candidates/e104-qkv-output-profile/README.md)에서
   Q 입력 준비→dot 중앙값은20118, O는20955 cycles다. O dot은 weight 종료2cycles 뒤,
   Q dot은 table 준비 종료1cycle 뒤 시작한다. 입력 방송 구간을 통째로 절감량에 넣지 않는다.
3. **BF16/FP32 경계와 scale 좌표를 보존한 채 완성 경로를 비교한다.**
   올바른 primitive만으로 전체 FFN/QKV를 승격하지 않는다. 실제 quantizer, block scale,
   gather, head/cache, epilogue와 모든 복사를 포함한 표준 host에서 대조한다.
4. **정적 소폭 차이와 NPU 성능을 구별한다.**
   E123은 정적157045→157208이었지만 실제 단회는 약1.92% 감소했다.
   반대로 이 작은 실제 이득이10위권 전략의 큰 성공은 아니다. 조건을 만족한 작은 이득은
   회수하고, 미세 조정으로 예산을 소모하지 않는다.
5. **실패 종류를 정확히 기록한다.**
   소스의 축·범위·packet 오류, 지원되지 않는 lowering, 수치 실패, runtime timeout,
   정확하지만 느린 실행은 다른 결론이다. 알려진 소스 오류를 수정할 수 있는데 임의의
   compile 횟수 때문에 닫지 않는다. 원인이 없는 parameter sweep도 재개하지 않는다.

O의 dot 이후 tail은 실측 중앙값 약9390cycles다. 현재 약51k 전체에서 이 구간만
줄이는 접근으로 감사 당시10위의30228cycles까지 가기는 어렵다. Weight 공급과 dot의
대기를 함께 줄이는 완성 설계가 필요하다. 기존 C09 row staging과 C12 Main/Sub 분리는
이미 시험한 가족이므로, lookup/setup/task/partial 비용이 어떻게 달라지는지 없이
새로운 후보라고 다시 제안하지 않는다.

현재 결과는 제한된 입력의 커널 검증과 Arena 측정이다. Native FP8 입력 표현의 근사,
프로파일 구간의 중첩·대기, 단회 E123 측정의 한계는 그대로 남는다. 공식 순위나 모델
전체 처리량의 개선으로 바꾸어 보고하지 않는다.

다음 설계를 고르기 위한 구체 질문은 **O의 약26.7k weight 준비 구간이 같은 HBM
byte 수에서도 DMA 순서·DM 배치에 얼마나 달라지는가**다. 현재 trace는 weight가
마지막 prerequisite라는 사실은 보여도 HBM 포화·bank 충돌·scatter 비용을 구분하지
못한다. 이를 먼저 구분하는 작은 전송 진단은 타당하지만, 그 단독 시간을 전체 O의
절감으로 더해서는 안 된다. 유망한 배치가 나오면 실제 BF16 lookup/dot과 F32 reduction,
추가 복사·table/setup을 포함한 완성 O에서 검증해야 한다. 공개 weight 형식을 미리
변환하거나 변환 비용을 측정 밖으로 빼는 후보는 이 비교에 포함하지 않는다.
