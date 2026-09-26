# E127 동결 C09 성능 채택 검증

Three distinct Arena allocations of identical standard benchmark-stage host/fixture/runner: one historical E126 seven-stage outer batch and two prospective clean E122 -> C09 -> E122 batches. All uploaded archives and raw logs verified. Original three inputs reused; diagnostic timings excluded.

| Job | 전체 O candidate | E122 before / after | paired 변화율 |
|---|---:|---|---:|
| 105820 | 49,923 | 51,237 / 51,165 | -2.496045% |
| 105902 | 48,835 | 50,903 / 50,652 | -3.825513% |
| 105909 | 49,180 | 51,653 / 50,731 | -3.930302% |

Paired 변화율 중앙값 **-3.825513%**; job 중앙값 aggregate 비율 **-3.930302%**.

각 stage의 실제 host/fixture/runner bytes가 E126과 동일함을 업로드 tar에서 검증했다. 한 job의3개 입력 또는 before/after를 독립 allocation으로 세지 않았다. QKV/FFN timing 흔들림은 최적화로 세지 않는다.

5%는 탐색 예산 기준이며 채택 검증 금지 기준이 아니다. 이번 검증은 기존 구현의 재현성 확인으로 한정했다. 원본3입력 성능 반복만으로 채택을 선언하지 않으며 확장 정확도 결과와 종합 검토가 별도로 필요하다.

검증 상태: `performance_reproduced_pending_expanded_correctness`. 이 기록 작성 시점 adopted=false, 공식 제출 없음.
