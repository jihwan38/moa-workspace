# E74: decode·dot·partial 저장·scale pass 결합

비교군은 [E55 recipe](../../../baseline/e55.json)의 `873f64e`다.
후보 public shell도 편의상 이 source에서 시작했지만 E55 구조 유지가 제약은 아니다.
채택 source는 `baseline-e74` / `9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7`이며
[현재 recipe](../../../baseline/e74.json), [재현 소스와 빌드](../../../moa-dev/experiments/candidates/c05-streamed-decode/README.md)에 연결된다.

| 구분 | 구성 | E55 → E74 |
|---|---|---|
| 유지 | Q1–Q6 / O1–O3 / R1 | QKV/O source와 image, RMS 후단 유지 |
| 유지 | F1 / F5 | up/gate 2×240×32행, down 2×32×60행·8 columns, GeGLU/global scale/epilogue 유지 |
| 대체 | F2 | FP8→BF16 전체 weight 저장 → BF16 stream/DPE 후 compact F32 block partial 저장 |
| 대체 | F3 | dot과 같은 stream의 scale/reduce → 별도 VE pass의 F32 scale/reduce |
| 변경 | F4 | dot+scale8/12행 tile → dot32/60행 전체 stream, scale만8/12행 tile |

F32 partial은16원소 dot 하나당 한 값이다. E74는 transpose2로 compact 저장하고
F32로 scale·합산한 다음 기존 projection BF16 경계로 나간다. Scalar scale을 전체 GEMV 뒤로
옮기거나 block partial을 BF16으로 반올림하지 않는다. F32 합산 순서 변경은 별도로 검사했다.

| 전체 설계 | FFN cycles | E75 대조 대비 | 독립 Arena jobs | 검증 / 판정 |
|---|---:|---:|---:|---|
| E75: E55 | 377804 | — | 3 | control_before |
| E73: tile마다 lookup→dot→scale/reduce | 416500 | +10.24% | 3 | 기본45 PASS, 회귀 |
| E74: 전체 dot→F32 partial→scale/reduce | 369246 | −2.27% | 3 | 기본45·추가30·진단138240 PASS, 채택 |

세 행은 같은 세 독립 batch jobs의 각 stage다. 한 job의 stage들을 서로 독립 반복으로
세지 않는다. Up/gate-only와 down-only 분해 후보는 이번에 실행하지 않았다. 개별 변경의
기여나 개선율 합은 미측정이며, 표는 **조합 전체**의 측정이다.

E73은 BF16 weight와 F32 partial 저장을 모두 없앴지만 lookup을13번 반복했다.
E74는 lookup3번과 작은 F32 partial 저장을 택했다. SRAM peak 모델은 감소했으나
DMA와 compute overlap도 변했다. 메모리 감소만을2.27%의 단독 원인으로 확정하지 않는다.
Sub reduction 조합은 SDK 내부 compiler 오류로 실행되지 않았다.

[E73 반복](../e73-c05-direct-repeats.json), [E74 반복](../e74-c05-partials-repeats.json),
[E75 대조](../e75-c05-control-repeats.json), [공식5.7805× 결과](../e74-official-submit-20260922.json).
다음 C04는 column 분할·activation 전달·partial 배치를 함께 바꿀 수 있다.
새 후보를 E55나 E74 주변으로 제한하지 않는다.
