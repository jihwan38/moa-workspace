# E60: down packed padding과 scale padding 조합

**결론: 미채택, E55 유지.** E58의 packed weight padding과 E59의 scale padding을
동시에 적용한 E60은 기준보다 FFN cycles가 4.18% 높았다. 아래는 기존 Arena 기록의
정리이며 이 문서를 만들면서 새 NPU 실행을 하지 않았다.

## 기준과 구성

기준은 [E55 recipe](../../../baseline/e55.json)의 `baseline-e55`,
commit `873f64ed94229c88cd18d2731bdcca7eea0d6365`다. 소스 부모도 이 commit이다.
[F1/F2 배치 설명](../../../baseline/methods.md)에 해당하는 down DM pitch를 바꾼다.
가설은 행 간격 정렬이 DMA/DM 접근을 개선할 수 있는지이며,
**HBM source 주소와 pitch는 그대로**라는 제약을 둔다.

| 구분 | 적용 구성 |
|---|---|
| 유지 | Q1–Q6, O1–O3, R1, F1–F5의 기존 연산·호출 구조 |
| 추가·제거 | 없음 |
| 변경 A = E58 | F1의 down packed DM row pitch: 960→1,024B |
| 변경 B = E59 | F1/F3이 사용하는 down scale DM row pitch: 120→128B |
| 조합 A+B = E60 | 두 목적지 pitch 변경을 같은 `mlp.rs`에 적용 |
| 산술 | F2 decode, F3 dot·scale·합산, F4 tile, F5 epilogue는 보존 |

## 단독·조합 실측

각 수치는 공식 3-input job 중앙값이며, 3회인 행은 job 중앙값들의 중앙값이다.
변화율은 동일한 **동시기 E55 대조** 379,073 cycles에 대한 값이다.

| 조합 | FFN cycles | 대조 대비 | 독립 job 수 | job 중앙값 범위 / 중앙값 | 기본 정확도 | 측정 |
|---|---:|---:|---:|---|---|---|
| C: E55 | 379,073 | — | 3 | 0.94% | 모두 PASS | [대조 반복](../e55-memory-layout-control-repeats.json) |
| C+A: E58 | 400,543 | +5.66% | 1 | 미측정 | PASS | [E58](../e58-down-packed-screen.json) |
| C+B: E59 | 378,149 | −0.24% | 3 | 1.32% | 모두 PASS | [E59 반복](../e59-down-scale-repeats.json) |
| C+A+B: E60 | 394,900 | +4.18% | 1 | 미측정 | PASS | [E60](../e60-down-both-screen.json) |

E59의 작은 감소는 대조군의 실행 변동과 겹쳐 개선 미확인이다. E60은 단회에서 E58보다
낮지만 E55보다 높다. A/B/조합 모두를 충분히 반복한 상호작용 추정은 아니므로
“두 방법이 서로의 회귀를 확실히 상쇄한다”고 결론 내리지 않는다. E58/E60의 큰 회귀는
추가 반복 없이 중단했다. 별도의 추가 입력·경계 검사나 모델 전체 측정은 하지 않았다.

## 재현 근거와 최종 소스

- [후보 디렉터리와 실행법](../../../moa-dev/experiments/candidates/memory-layout-e55/README.md):
  A=`down-packed-split`, B=`down-scale-split`, A+B=`down-both-split`.
- [A+B 전체 소스·patch·입력 해시](../../../moa-dev/experiments/candidates/memory-layout-e55/down-both-split):
  기준 대비 변경을 하나의 최종 source로 보존한다.
- [검증 도구](../../../moa-dev/experiments/candidates/memory-layout-e55/verify.py):
  down row-shard 좌표와 padding 보존, FFN 산술 stream 불변을 확인한다.
- [빌드·실제 업로드 연결](../../../moa-dev/experiments/candidates/memory-layout-e55/hardware-summary.json):
  후보 image·host·실제 업로드 hash를 연결하고, 다른 두 커널이 E55와 같음을 확인한다.
- [모델·DMA·메모리 분석](../../research/e55-memory-layout-audit.md):
  46개 compile 입력과 compiler/std 0.8.1, Rust nightly 2026-05-01 기준을 기록한다.

다음 조합에서도 단독·조합 모두의 source와 대조군을 남긴다. E60을 채택본으로 승격하거나
E59를 “검증된 가속 방법”으로 등록하지 않는다.
