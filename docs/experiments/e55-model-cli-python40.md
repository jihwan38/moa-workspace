# 무수정 E55 CLI의 실제 문자열 생성 확인

2026-09-22 KST. Arena **job71395**가12.284초에 exit0으로 완료됐다.
이미 빌드된 E55 `gemma4`를 그대로 실행했고, 다음 입력에 실제 생성문을 출력했다.
[수집 결과](e55-model-cli-python40.json).

입력: `Complete the Python expression: len([1, 2, 3])`

```text
The result of the expression is:

**3**
```

입력문 echo를 제외한 관측 출력은 위39 UTF-8 bytes다. 표시한 값3은 이 식의 결과와
맞는다. 이 한 사례로 정답률이나 전체 모델 정확도 합격을 선언하지 않는다.
CLI는28 prompt tokens,11 completion tokens와 `finish: Stop`을 보고했다.
현재 요청은 stop string이 없고 callback이 항상 true이므로 Stop은 EOS 계열 종료다.
CLI는 실제 token ID를 출력하지 않으므로 어떤 EOS였는지는 기록하지 않는다.

[gemma4.rs](../../furiosa-opt-gemma4-12B/src/bin/gemma4.rs)의 실제 경로는
Device/모델 적재 → tokenizer/chat template → Workspace → `generate::generate` →
`decode_stream.step` → `on_delta` → mpsc printer → 종료 summary다.
이전의 logits 진단이 우회했던 문자열 decode와 callback/출력 경로까지 실행했다.
입력 echo, 생성 구간, 최종 summary를 분리하고 명시적인 Rust panic 흔적이 없음을
확인했다. 원본 CLI가 printer의 join 오류를 무시하므로 exit0만으로 모든 출력 byte의
완전성을 증명했다고 주장하지 않는다.

실행 파일은 [이전71173 timeout](e55-arena-model-smoke.json) 때와 정확히 같은
12,378,864B, SHA `7e801ed65016b5d0faa71da3fd3f3f9d012db393adc47863ad8e4665532678eb`다.
새 컴파일이나 모델 소스 변경 없이 실제46개 보존 입력과 기존 compiled manifest,
14개 embedded kernel image와 compiler 출력, 그중10개 text image와 E55 진단 host의
일치를 확인했다. 나머지4개는 vision/audio entry이며 이번 text 입력으로 실행을
검증한 것은 아니다. [패키지 준비와 재현](../../moa-dev/experiments/candidates/model-cli-python40-e55/README.md).

```sh
GEMMA4_MAX_NEW_TOKENS=40 FURIOSA_OPT_PROFILE=warn \
  ./gemma4 'Complete the Python expression: len([1, 2, 3])'
```

CLI의 기본 sampling은 temperature1/top-k64/top-p0.95이며 seed 제어가 없다.
[71362](e55-model-continuation-python40.md)의 greedy13개 non-EOS 선택과 이번11개
completion을 같은 token sequence로 비교하지 않는다. 이번 실행은 full logits 배열도
출력하지 않아 새로운 전체 logits 유한값 검사로 세지 않는다.

같은 job의 실행 ELF2개 및 실제 업로드3파일+`job.yaml`을 확인했고, 실행 전 전체
model/metadata SHA가 기존 자산과 같았다. 종료 뒤 weight stat/header 및 metadata hash도
같았다. 종료 뒤 weight 전체를 다시 해시한 것은 아니다. 실행 전 전체 weight 읽기는
파일 cache에 영향을 줄 수 있으며71173에는 없었던 단계다. 따라서 이번 성공은 현재
CLI 경로의 작동 근거이며, 이전 timeout의 원인이나 해결책을 확인한 결과가 아니다.

CLI가 표시한 모델 적재3.48초, TTFT746.57ms, decode11tokens/336.72ms는 단일 실행의
반올림된 자체 로그다. 모델 적재 타이머는 Device 생성 후 시작하며, TTFT는 Workspace
준비 후 첫 delta까지다. Decode 처리량의 분자에는 첫 completion도 포함되므로 이 값을
표준 TPOT나 공식 성능으로 그대로 사용하지 않는다. 전체12.284초에는 asset 검사 등도
포함된다. 통제된 반복 벤치마크는 아니다.

[수집기](../../moa-dev/experiments/collect_model_cli_python40.py)는 실제 package/실행 hash,
전후 asset, model-loaded→echo→생성문→summary 순서, Stop0..39/Length40 count와
최종 성공 marker를 확인한다. 실제 작업을 포함한45개 검사를 통과했다. [수집기 검사](../../moa-dev/experiments/candidates/model-cli-python40-e55/collector-tests.json),
[wrapper의 실패 전파 검사4건](../../moa-dev/experiments/candidates/model-cli-python40-e55/runner-tests.json)을
별도로 보존했다. 수집기의 정확도 threshold/verdict는 null이다.

E55를 유지한다. 다음은 다른 입력과 정답이 있는 평가, 긴 context/cache 경계,
멀티모달, 공식 workload에 맞춘 정확도·성능 검증이다. 공식 제출·점수·순위는 아직 없다.
