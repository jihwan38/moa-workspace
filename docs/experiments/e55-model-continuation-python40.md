# Python 생성의 EOS까지 비교: 실제 동점 관측

2026-09-22 KST. 최대40 checkpoint를 허용한 Arena **job71362**는14번째 선택에서
원본 EOS106을 만나25.073초에 exit0으로 정상 종료됐다. 원본/E55의 **14개 선택이
모두 일치**했고, **3,670,016개 logits**가 finite였다. 40단계를 채운 실험으로 세지
않는다. [수집 결과](e55-model-continuation-python40.json).

이번에는 선택 여유가 사라지는 실제 입력을 관측했다. 0부터 세는 **step9, position36**에서
원본의 허용 top1 margin은0.125였지만 E55에서는 두 token이 동점이 됐다.

| 값 | 원본 | E55 |
|---|---:|---:|
| 실제 선택 ID | 236929 | 236929 |
| 선택 logit | 24.5 | 24.5 |
| Runner-up ID | 236800 | 236800 |
| Runner-up logit | 24.375 | 24.5 |
| 허용-token margin | 0.125 | 0.0 |
| 허용 최댓값 동점 수 | 1 | 2 |

직전 step8도 margin이 원본0.25에서 E55 0.125로 줄었다. 두 단계 모두 실제 공개
`sample`의 `top_k=1` 선택은 같았다. 이 sampler의 동점 순서는 최저 ID를 선택한다는
계약이 없으며, 이번에 같은 token을 골랐다는 사실로 다른 입력·구현 버전에서도 선택을
보존한다고 결론내리지 않는다. 원본 역시 독립적인 정확도 oracle은 아니다.

Prompt는 `Complete the Python expression: len([1, 2, 3])`이며 실제 chat token 수는
이전과 같은28이었다. 원본 선택13개를 차례로 feed한 뒤14번째 선택 EOS를 feed하지 않고
종료했다. 모델당 **41forward**, 최대position40, completion13개다. E55가 다른 token을
골라도 원본 이력을 이어가는 host지만 이번에는 EOS까지 불일치가 없었다. 문자열
decode/callback이나 두 독립적인 자유 생성 실행을 검사한 것은 아니다.

| 관측 범위 | Checkpoint 수 | 비교 원소 수 | 최대 절대 차이 | Checkpoint별 RMSE 범위 |
|---|---:|---:|---:|---:|
| 이번 실행 전체 | 14 | 3,670,016 | 0.65625 | 0.064635–0.207814 |
| 기존8개 이후 step8..13 | 6 | 1,572,864 | 0.53125 | 0.072833–0.172595 |

[이전71329](e55-model-continuation.md)의 Python case3와 이번 case0의 **첫8개 보고값이
정확히 재현**됐다. 같은 asset SHA/길이에서 입력 이력, position, 실제 선택,
legal-choice와 전체 vocabulary의 오차 통계를 대조했다. Case index·filename·timing은
제외했다. 이는 host가 보고한 값의 일치이며, 내려받지 않은 전체 logits 배열의
byte identity를 증명하지 않는다. 기존8개를 새로운 독립 입력으로 다시 세지 않는다.

[새 host와 빌드 증거](../../moa-dev/experiments/candidates/model-continuation-python40-e55/evidence.json)는
원본/E55 각각47개 실제 컴파일 입력, 기존46개 모델 입력 보존, 각각10개 kernel image
일치를 확인한다. 변경은 진단의 schema·prompt·생성 한도와 해당 guard/test에 한정한다.
Async main의 실행 흐름은 이전 host와 byte-identical이며 E55 모델/커널은 변경하지 않았다.
실제 helper·공개 sampler를 사용하는 CPU 검사11개도 통과했다.

같은 job의 weight 전체 SHA는 기존과 같은 `7c2ee232…8ce704b`다. 실행 전후
reference14개+manifest의15개 hash, 종료 뒤 weight stat/header 및 metadata hash를
확인했다. 종료 뒤 weight 전체를 재해시한 것은 아니다. Manifest SHA는
`29da29a20b747a011a3b2731dbab9c00fe1b376310667a7d57ad49f2aad0c747`다.
실제 업로드 archive/실행 ELF는 [패키지 manifest](../../moa-dev/experiments/candidates/model-continuation-python40-pair-e55/package-manifest.json)와
일치한다. Reference 배열은 원격에 남아 있으므로 수집기는 audited host의 보고와 출처를
검증하며 로컬 원소별 재계산을 주장하지 않는다.

[공통 수집기](../../moa-dev/experiments/collect_model_continuation.py)는 불변 config로
schema·prompt·한도·marker·package를 받도록 확장했다. 기존4×8 기본 동작의63개
회귀 검사는 유지됐고 [Python40 adapter](../../moa-dev/experiments/collect_model_continuation_python40.py)는
40개 검사를 통과했다. 실제 후반6개 누락·EOS 뒤 진행·잘못된 동점 수/bit 보고를
거절하고, 합법적인 다른 동점 선택 자체를 구조 오류로 취급하지 않는 검사도 포함한다.
[검사 기록](../../moa-dev/experiments/candidates/model-continuation-python40-pair-e55/collector-tests.json).
이전71329 보고서와 당시 collector hash는 그 시점의 증거로 보존했다.

원본/E55 host 전체 시간은 각각11.946초/6.719초지만 reference 쓰기와 파싱·비교 비용이
다른 순차 단일 실행이다. 공식 처리량이나 개선 배율로 사용하지 않는다. 정확도 허용기준과
합격 판정은 계속 null이다. 수치 차이가 실제 동점을 만든다는 근거를 확보했으며, 이를
임의로 tie-breaking 코드를 바꾸거나 원본 kernel로 되돌릴 이유로 삼지 않는다.
정답 평가·다른 prompt·긴 context·멀티모달 및 공식 평가는 여전히 남아 있다.
