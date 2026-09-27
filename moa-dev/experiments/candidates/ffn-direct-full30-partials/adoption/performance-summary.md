# E142 동결 full30 F32 partial 성능 검증

| Job | Candidate FFN | E140 before / after | Paired 변화율 |
|---|---:|---|---:|
| 111706 | 302,889 | 313,548 / 313,520 | -3.395166% |
| 111719 | 301,615 | 315,456 / 314,319 | -4.214997% |
| 111721 | 302,796 | 314,498 / 313,835 | -3.619259% |

Paired 변화율 중앙값 **-3.619259%**; aggregate 중앙값 비율 **-3.619259%**.

세 allocation의 실제 업로드 host·fixture·runner와 바깥 실행 패키지가 byte-identical하다. 원본 세 입력을 반복했으며45개 candidate 비교(FFN9개), 전체135개 비교가 PASS다. 입력 수나 앞뒤 대조를 독립 allocation으로 세지 않았다.

첫 screen 및 기존 보고서는 보존했다. 추가6입력과 별도 projection oracle는 이 성능 집계에 포함하지 않으며 최종 채택 검토에서 따로 확인해야 한다. QKV/O timing 변화는 최적화로 해석하지 않는다.

5%는 추가 parameter tuning 예산 기준이며 최소 채택 이득이 아니다. 이 기록은 성능 재현 여부만 보고하고 root의 최종 채택 결정을 대신하지 않는다.

상태: `performance_reproduced_pending_separate_correctness_review`. 작성 시점 adopted=false, 공식 제출 없음.
