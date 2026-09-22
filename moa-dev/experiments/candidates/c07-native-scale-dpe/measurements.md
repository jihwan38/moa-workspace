# C07 기본 입력 측정

동일 package를 서로 다른 Arena allocation에서 실행한 경우만 반복으로 센다. 실패 stage의 cycles는 제외한다. [최종 판정](decision.json)에서 별도 정확도 진단과 채택 여부를 함께 확인한다.

| ID | 구조 | E79 대조 → FFN cycles | 변화 | 독립 jobs |
|---|---|---:|---:|---|
| [E86](../../../../docs/experiments/e86-c07-pair-row2.json) | pair-row2 | 352,565 → 392,413 | +11.30% | 74263 |
| [E87](../../../../docs/experiments/e87-c07-time-row2.json) | time-row2 | 352,902 → 377,290 | +6.91% | 74289 |
| [E88](../../../../docs/experiments/e88-c07-time-contiguous.json) | time-contiguous | 353,238 → 325,543 | -7.84% | 74310 |
| [E89](../../../../docs/experiments/e89-c07-full-pair-row2.json) | full-pair-row2 | 354,440 → 387,879 | +9.43% | 74315 |
| [E90](../../../../docs/experiments/e90-c07-full-time-contiguous.json) | full-time-contiguous | 정확도/실행 실패 | — | 74315 |
| [E91](../../../../docs/experiments/e91-c07-pair-contiguous.json) | pair-contiguous | 354,440 → 351,437 | -0.85% | 74315 |
| [E92](../../../../docs/experiments/e92-c07-late-up.json) | late-up | 352,705 → 335,983 | -4.74% | 74336 |
| [E93](../../../../docs/experiments/e93-c07-full-time4-late.json) | full-time4-late | 352,705 → 316,977 | -10.13% | 74336 |
| [E94](../../../../docs/experiments/e94-c07-up-f32.json) | up-f32 | 352,176 → 332,066 | -5.71% | 74335 |
| [E95](../../../../docs/experiments/e95-c07-full-f32.json) | full-f32 | 352,176 → 327,093 | -7.12% | 74335 |
| [E96](../../../../docs/experiments/e96-c07-e5-dpe.json) | e5-dpe | 352,176 → 333,980 | -5.17% | 74335 |
| [E97](../../../../docs/experiments/e97-c07-e5-f32.json) | e5-f32 | 정확도/실행 실패 | — | 74335 |
| [E98](../../../../docs/experiments/e98-c07-two-column-time.json) | two-column-time | 정확도/실행 실패 | — | 74354 |
| [E99](../../../../docs/experiments/e99-c07-two-column-pair.json) | two-column-pair | 정확도/실행 실패 | — | 74354 |
| [E100](../../../../docs/experiments/e100-c07-compensated-dpe.json) | compensated-dpe | 352,993 → 507,518 | +43.78% | 74371 |
| [E101](../../../../docs/experiments/e101-c07-e5-f32-late.json) | e5-f32-late | 352,610 → 330,531 | -6.26% | 74389 |
| [E102](../../../../docs/experiments/e102-c07-e5-materialized.json) | e5-materialized | 352,610 → 335,696 | -4.80% | 74389 |
| [E103](../../../../docs/experiments/e103-c07-e5-reload.json) | e5-reload | 353,154 → 327,830 | -7.17% | 74420, 74476, 74481 |
| [E104](../../../../docs/experiments/e104-c07-e5-reload-materialized.json) | e5-reload-materialized | 353,154 → 335,761 | -4.93% | 74420, 74476, 74481 |
