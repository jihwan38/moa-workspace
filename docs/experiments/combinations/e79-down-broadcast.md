# E79: F32 partial 계산과 down 입력 방송의 결합

실제 출발 source는 [E74](../../../baseline/e74.json)의 `9e2058e`다.
채택 source는 `baseline-e79` / `2735b58`이며 [E79 구성표](../../../baseline/e79.json),
[C04 전체 결과](../../../moa-dev/experiments/candidates/c04-down-transfer/README.md)에 연결된다.
E55/E74는 비교군이고 후보의 구조를 제한하지 않는다.

| 구분 | 구성 | E74 → E79 |
|---|---|---|
| 유지 | Q1–Q6 / O1–O3 / R1 | 동일 QKV/O image, 동일 RMS 후단 |
| 유지 | F2 / F3 / F4 / F5 | FP4→FP8 DM→BF16 stream/DPE→F32 partial, scale/reduce, GeGLU·epilogue |
| 변경 | F1의 down 입력 전달 | 전체 row group으로 직접 TDMA 복제 → ring의8개 leader로 TDMA → Main ring32로4개 row group에 방송 → Sub TRF |
| 유지 | F1의 최종 계산 배치 | 2 clusters×32 row groups×8 columns, local60×1920 |

선택한 E79에서 유지한 사항과 **탐색 때 허용한 변경 범위**는 다르다. C04에서는
2열16행, 4열32행, padding 없는4열30행, F32 output gather, Main/Sub 방송과
선행 full-L gather까지9개 조합을 실행했다. 그중 E79가 가장 좋았다.

| 전체 조합 | FFN cycles | 해당 batch의 E74 대비 | 반복 / 판정 |
|---|---:|---:|---|
| E85: E74 고정 비교군 | 369997 | — | 독립3 jobs |
| E78: 2열 + Main ring16 | 375135 | +1.39% | 독립3 jobs, 미채택 |
| E79: E74 계산 + 8열 Main ring32 | 353136 | −4.56% | 독립3 jobs·추가 입력·projection 진단 통과, 채택 |
| E80: 4열 + Sub ring16 | 399701 | +8.03% | 독립3 jobs, 미채택 |
| E81: 4열30행 + Main ring16 | 364496 | −1.65% | 별도 batch 단회, 추가 진단 전 후보로 보존 |

E78/E79/E80와 E85는 jobs74142/74154/74161의 서로 다른 stage다. Stage를 독립 반복으로
세지 않는다. E81은 job74152의 별도 E74 대조370623을 사용하며 E85를 소급 적용하지 않는다.
나머지 단회 후보는 [전체 분류](../optimization-results.md)에 각각 남긴다.

과거 E43의 방송 경로는 E55 계열에서 정적 분석만 하고 NPU에는 실행하지 않았다.
그 결과만으로 E74와 결합한 E79의 효과를 배제하지 않았다. E79도 정적 전체 cycles는
157822→160203으로 늘었지만 NPU에서는 개선됐다. 방송 자체와 F32 partial의 상호작용을
완전히 분리한2×2 실험은 아니므로 각 방법의 기여율이나 상승효과를 따로 주장하지 않는다.

E79는 기본45개 비교와 추가30개 비교를 통과했고, 네 case의 projection 출력138240개가
CPU oracle과 정확히 일치했다. Full-model 성능이나 임의 입력 전체에 대한 보장은 아니다.
[반복](../e79-c04-eight-ring32.json), [대조](../e85-c04-e74-control.json),
[추가 입력](../e79-c04-extra-correctness.json), [projection 경계·상쇄](../e79-c04-probe-correctness.json),
[공식 제출 추적](../e79-official-submit-20260922.json).
