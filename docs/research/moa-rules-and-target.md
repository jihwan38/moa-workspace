# MOA 평가 계약과 상위권 목표

**2026-09-23 갱신:** [공식 홈페이지](https://micro2026-moa.github.io/index.html)는
1라운드 마감을 **9월 30일 23:59 AoE**, 결과 발표·2라운드 시작을 10월 2일로 안내한다.
최종 평가는 **마감 전 마지막 제출**을 사용한다. SDK 0.6.0/0.8.1 지원(0.8.1 권장),
마감 전 competition code 공개 금지 안내도 확인했다.
[최신 후보 조사와 근거](structural-candidates-20260923.md).
아래 일정·순위·접속 상태는 9월 21일 조사 당시 기록이며 현재 상태로 사용하지 않는다.

확인일: 2026-09-21. 공식 안내, 공개 leaderboard API, baseline 구현과 제출 CLI를 대조했다.
원본 소스는 `8559916051162e5fd682c29ec0fe3e9cc6e420cf`이며 조사 시 upstream HEAD와 일치했다.

## 평가 범위

[공식 안내](https://micro2026-moa.github.io/index.html)는 1라운드를 9월 1–25일,
2라운드를 10월 1–25일로 제시한다. 저장소 README에는 정확한 제출 마감이 아직 TBD로
남아 있으므로, 종료 시각과 최종 평가 규칙을 추정해서 확정하지 않는다.

1라운드는 `ops::sliding_project_qkv`, `ops::sliding_attention_output`,
`ops::decoder_feedforward` 세 커널이다. 2라운드는 모델 전체 실행 경로이며
입력 수·길이, E2E 지표, 정확도 기준 등이 아직 확정되지 않았다.
[대회 README](../../furiosa-opt-gemma4-12B/README.md)를 기준으로 한다.

- 제출에 반영되는 것은 `src/ops.rs`의 함수 본문과 `src/device/` 구현이다.
- 모든 `#[device]` 함수의 이름·인자·타입·반환형을 보존한다.
- kernel module 경로를 유지한다. `src/axes.rs`, host/API/test 코드 변경은 1라운드 채점에 반영되지 않는다.
- 정확성은 모든 출력 원소와 세 개의 독립 입력 구성이 모두 통과해야 한다.
- 각 커널의 성능은 세 입력 구성 cycle의 중앙값이다.

실제 비교식은 `abs(actual-expected) <= atol + rtol*abs(expected)`이며 non-finite는 실패다.
QKV/O/FFN의 `atol`은 각각 `0.04/0.05/0.01`, 공통 `rtol`은 `0.01`이다.
구현 근거는 [test_kernels.rs](../../furiosa-opt-gemma4-12B/src/bin/test_kernels.rs)의
`TESTS`, `compare`, `main`이다. 허용오차나 입력 생성기를 바꾸어 점수를 만들지 않는다.

## 점수와 목표선

[공개 leaderboard](https://micro2026-moa.github.io/leaderboard.html)는 다음 식을 명시한다.

```text
score = ((baseline_qkv / candidate_qkv)
       * (baseline_output / candidate_output)
       * (baseline_ffn / candidate_ffn)) ** (1/3)
```

공식 API가 반환한 기준값은 QKV `250514`, O `404633`, FFN `3703473` cycles였다.
[원본 응답과 수집 시각](../experiments/leaderboard-20260921-initial.json)을 보존했다.
확인 시 58개 항목이 있었고 상위권은 다음과 같았다.

| 공개 순위 | 팀 | QKV | O | FFN | 점수 |
|---|---|---:|---:|---:|---:|
| 1 | Participant #398 | 78,870 | 30,228 | 226,475 | 8.8590 |
| 2 | IA -ARIA ON THE PLANETES- | 76,394 | 36,024 | 212,627 | 8.6247 |
| 3 | Fluidstack | 80,573 | 34,010 | 231,258 | 8.3985 |

단일 커널만 두 배 개선하면 총 점수는 약 1.26배다. 절대 시간이 큰 FFN만 추적하지 않고
세 커널의 로그 speedup 기여를 같이 평가한다. 예를 들어 `(75000, 30000, 210000)`은
이 API 기준으로 약 9.26배인 설계 목표이며, 달성된 성능이나 미래 3위 보장값이 아니다.

리더보드는 참고용이며 공식 순위는 동일 조건에서 제출물을 다시 실행해 결정한다고 명시돼 있다.
과거 공개 점수와 현재 Arena의 측정 조건이 같다고 가정하지 않는다.
`BEEF` 이름의 과거 제출도 존재하지만, 현재 사용자 팀과 같은 항목인지와 기존 코드의
출처는 아직 확인 중이다. 이를 현재 작업의 공식 성과로 표시하지 않는다.

## Arena 측정과 공식 제출

Arena는 실행 파일·fixture·entrypoint를 실행하는 개발용 NPU 서버다.
공식 점수는 별도 `moa-submitter`가 소스를 올려 서버에서 다시 빌드·평가해서 얻는다.
따라서 Arena job 성공이나 예상 기하평균만으로 공식 순위 달성을 주장하지 않는다.

[제출 CLI](https://github.com/micro2026-moa/moa-submitter-cli)의 `find_repository`는
`src/ops.rs`를 기준으로 source root를 찾고, `collect_sources`는 `src/ops.rs`와
`src/device/`만 수집한다. 따라서 부모 Git 저장소와 Docker `/workspace` 구조에서도
대회 코드 디렉터리에서 실행하면 경로 계약을 만족한다.

```bash
./moa-dev/moa moa-submitter login
./moa-dev/moa moa-submitter submit
./moa-dev/moa moa-submitter status
```

위 login은 Arena와 다른 인증이다. `submit` 요청이 네트워크 오류를 반환하면 `status`를
먼저 확인한다. CLI 구현도 중복 제출을 방지하기 위해 업로드를 자동 재시도하지 않는다.

사이트에는 평가 기준 갱신에 따른 제출 서버 점검 공지가 남아 있었지만 조회 시
`/api/health`와 `/api/leaderboard`는 정상 JSON을 반환했다. 이 사실만으로 인증된 제출이
가능하다고 단정하지 않는다. Arena의 실제 작업 시간 제한은 70초였고 기존 스크립트의
120초 기본값은 서버가 거절했다. 개발 실행은 `RNGD_SUBMIT_TIMEOUT=70`을 사용한다.

## 해석상 주의할 불일치

- README 앞부분의 일부 점수 항목은 TBD지만 제출 절과 live leaderboard는 기하평균을 명시한다.
- 정적 schedule makespan과 실제 NPU cycles는 다른 값이다. 상수 변환비를 가정하지 않는다.
- 세 입력 구성의 중앙값은 독립 process 반복 측정에 따른 잡음 추정이 아니다.
- 원본 실제 FFN 중앙값 약 4.264M과 공개 기준 3.703M이 다르다. 공식 동일 환경 결과를 따로 확보한다.
- 튜토리얼 영상 전체 시청이나 아직 열리지 않은 2라운드 조건 검증을 완료한 것으로 기록하지 않는다.

등록·팀·학생 자격 문의의 공식 근거는 [FAQ](https://micro2026-moa.github.io/faq.html)다.
이 문서는 평가 계약을 고정하기 위한 조사이며, 규칙 변경 여부는 실제 공식 제출 전에 다시 확인한다.
