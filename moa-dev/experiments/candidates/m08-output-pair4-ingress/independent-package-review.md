# M08 package / screen 독립 검토 — 2026-09-27

검토 결과 현재 패키징·실행 정의에서 차단할 오류는 발견하지 않았다. 초기 발견한 실패 stage 보존과 실제 다운로드 tar digest 누락은 담당 agent가 수정했고 반영본을 다시 읽었다. 검토자는 build/package/submit/collect/device를 실행하지 않았으며 소스와 보존 파일의 해시만 확인했다. NPU 정확도·성능을 검증했다는 뜻이 아니다.

검토 범위는 `moa-dev/experiments/candidates/m08-output-pair4-ingress/{package,screen}.py`, E122 recipe→source-build-audit→measured build/package, 공통 `e104-weight-path/build.py::host_source`, `measure.py::parse_log`, 실제 `test_kernels.rs` 입력 생성/fixture checksum 경로다. Profile attribution analyzer는 다른 검토자가 맡아 중복하지 않았다. 확인 시점 source hashes와 이미지 수치는 [기계 기록](/tmp/moa-m08-package-review-evidence.json)에 보존했다.

## E122 control

- `baseline/current.json`은 E122, recipe commit은 `c3553ab87046fb4d59c10149a82517f8440bba27`이다.
- 실제 control package는 `artifacts/m03-producer-local-ffn/m03-native-local-v1/package`. source manifest와 measured-build 입력 목록이 동일하고 fixture/runner/host 파일 SHA가 build record와 모두 일치했다.
- QKV `5bdab7ea…`, O `35947381…`, FFN `16914f77…` exact image를 실제 파일로 읽어 build/audit와 비교하고, control host에 각각 한 번 들어 있음을 독립 확인했다.
- `package.py::checked_control`은 여기에 git commit source manifest까지 대조한다. `package()`는 candidate 선언 comparison ID·commit·recipe digest가 이 control과 같아야 진행한다. 오래된 E104 FFN을 전체 benchmark control로 사용하는 경로가 아니다.

## standard / diagnostic 이미지와 입력

실제 ready package에 대해 local 파일을 재검사했다. standard package 파일들과 diagnostic probe/metadata/marker-map SHA가 package JSON과 일치했다. standard와 unlabelled diagnostic의 세 device image가 byte-identical이고 두 host에 각각 한 번 들어 있다. Candidate O는 exact `dac23720…`이며 QKV/FFN은 E122 image다.

Candidate labelled probe와 unlabelled host의 차이는 metadata가 허용한 profile-name 영역의315 bytes뿐이었다. task instruction·weight·입출력 구조 변경을 의미하지 않는다. Diagnostic host source SHA는 기존 검증된 `host_source` 변환 결과와 같다.

`host_source`는 trace span 이름 수집/출력과 `DIAG_KERNEL` 필터만 넣는다. `Synth::new(test.name, run, fixture)` 및 input checksum 생성/검증, numerical tolerance,3개 run 입력은 바꾸지 않는다. Same fixture를 모든 stage에 제공한다. 따라서 official whole과 O-only diagnostic의 논리 입력 값은 동일하다. 다만 이전 kernel 실행 여부, HBM allocation 주소, cache/queue 이력은 달라질 수 있어 timings가 같은 실행환경이라는 주장은 하지 않는다.

Control diagnostic은 이전 E104 generic host의 O-only 실행이다. O image는 E122와 정확히 같고 host transform도 같다. 그 binary 안의 E104 FFN은 실행하지 않는다는 scope가 metadata에 명시되어 있다. Whole controls는 별도의 실제 E122 host를 실행한다.

## 결과 분리와 업로드 경로

7개 serial processes는 control whole before → control info → control trace → candidate info → candidate trace → candidate whole → control whole after다. 각3개의 원래 입력을 사용한다. 3 whole stages의15 comparisons씩과4 diagnostic stages의3 comparisons씩으로57개를 요구한다. 한 allocation이며 독립 반복으로 표현하지 않는다.

`whole_o_comparison`은 official whole hosts의 O 중앙값 세 개만 사용한다. Info/trace timings를 채택 성능 비교에 섞지 않는다. Diagnostic parser는 O-only3개 PASS, run별 두 cluster task, trace operation marker completeness 및 host cycle/task marker 관계를 검증한다. Trace 시작시각의 해석/critical-window 분석은 별도 analyzer 검토 범위다.

Prepare는 파일 SHA와 descriptor gate를 다시 확인한다. Submit은 복사 후 container 파일 SHA를 확인한다. Collect는 Arena에서 실제 업로드 archive를 다운로드해 모든 regular member의 bytes/SHA를 saved package와 exact 비교한다(`job.yaml` 제외). Member 이름 중복을 거부한다. 수정 후 result의 `archive`에도 다운로드 tar의 path/bytes/SHA를 기록한다. 최종 수집 후 reviewer가 보존된 job105767 실제 업로드 tar를 직접 열어 아래와 같이 검증했다. 다운로드 명령이나 Arena 실행을 추가로 수행하지는 않았다.

## 실패 처리 검토

초기 버전은 모든7개 stage와 SCREEN_COMPLETE를 먼저 assert하여 timeout시 완료된 stage도 structured results에서 빠졌다. 수정본은 완료 sections를 먼저 수집하고 각 expected stage별로 parse한다. Missing/incomplete/duplicate stage는 valid=false와 started 여부·completed count·partial spans·malformed count를 남긴다. 최종 valid는 성공 job/exit0, SCREEN_COMPLETE, 정확한 stage sequence, 모든 stage PASS가 모두 필요하다. 실패가 정상 결과로 승격되는 경로는 확인되지 않았다.

Nonzero stage는 shell이 exit code를 남기고 다음 process를 계속 실행하며 최종 screen exit는 nonzero다. Package build 실패는 error/failed command/exit code 및 build logs를 package record에 남긴다. 원본 Arena log도 stage parser 전에 파일/hash로 보존한다.

남은 비차단 구석은 complete-stage 예외 처리 내부의 `json.loads` list comprehension이다. 완결된 stage에 깨진 DIAG_SPAN JSON이 있으면 exception handler에서 다시 예외가 발생하여 그 뒤 stage의 구조화 파싱이 멈출 수 있다. 원본 log와 앞서 파싱된 stage는 남고 valid=false이므로 false PASS는 아니다. 담당 agent에 안전 파서 재사용을 선택적 보완으로 전달했다.

## 최종 job105767 실제 파일 감사

- Submission ID와 결과 job105767이 일치하고 SUCCEEDED/exit0다.
- 실제 업로드 tar15,020,032 B, SHA-256 `8ee37c95747a68608c532ecf9f614194454bed415a34f386828374340e7a1272`를 직접 확인했다. 모든 regular member의 bytes/SHA가 준비 package와 exact 일치한다(`job.yaml` 제외 비교; 중복 member 없음).
- 원본 log50,196 B, SHA-256 `08e0b2bec114c5a1ca95c0bb41097113752bd52b1008cd48cc5249dac2e72387`가 결과 metadata와 같다.7개 stage 순서·exit0·SCREEN_COMPLETE, 중복 없는 정확한 output label과57 PASS를 독립 검사했다.
- Whole O 값만 원본 log에서 다시 추출했다: E122 before51,418 → candidate58,095 → E122 after50,906 cycles. Bracket51,162 대비+13.551073% 회귀다. Info/trace timing은 이 계산에 포함하지 않는다.
- 이 결과는 해당 후보의 종료 근거다. Weight ingress/gather 중 어느 항목이 회귀를 만들었는지는 본 검토에서 분해하지 않았다. 별도 profile analyzer가 필요하며 descriptor packet60배를 latency60배로 읽지 않는다.
