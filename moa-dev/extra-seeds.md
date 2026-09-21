# 추가 seed 정확도 검증

`extra_seeds.py`는 공식 run0..2를 통과한 후보에 run3..8을 추가로 적용하는 개발용 검사다. 공식 점수나 순위를 계산하지 않는다. 호스트에서는 Python 표준 라이브러리만 사용하고, CPU reference 생성·컴파일·Arena CLI는 이미 실행 중인 `moa2026-dev` 컨테이너에서 실행한다.

공식 `src/bin/test_kernels.rs`와 `scripts/generate_references.py`를 복사한 뒤 `RUNS`를 6으로, 반복 범위를 3..8로 바꾼다. 변경 위치가 정확히 한 곳씩 있는지 확인하고, PRNG·입력 분포·shape·tolerance·입력 checksum·비교 함수를 그대로 보존한다. 참조 구현 `scripts/reference/gemma4.py`와 `scripts/fixture_prng.py`도 원본 그대로 복사한다. 실험은 호스트의 ignored `moa-dev/experiments/artifacts/LABEL/`와 컨테이너의 `/tmp/moa-extra-seeds-LABEL/`에 저장한다. 기존 label을 덮어쓰지 않으며 실패한 label도 보존한다.

2026-09-21 생성 완료한 reference:

- 경로: `moa-dev/experiments/artifacts/extra-seeds-run3-8/references.json`
- Fixture: `crate/ref/fixtures.safetensors`, 405,448 bytes, 228개 tensor(출력 30개 + 입력 checksum 198개)
- SHA-256: `f85f6e03d8eb359cc8589d375f89ba7b501924b095059800e322378e017cb4df`
- 공식 fixture·harness·generator는 변경하지 않았다. 이 단계에서 NPU 컴파일이나 Arena 제출은 실행하지 않았다.

새 reference가 필요한 경우에만 새 label로 생성한다:

```bash
python3 moa-dev/extra_seeds.py prepare --label extra-seeds-run3-8-new
```

공식 3-run 검사에서 통과한 후보를 `capture.py`로 보존한 다음, 그 snapshot을 명시적으로 선택한다. 아래 `SELECTED_CAPTURE_LABEL`과 `EXTRA_LABEL`을 실제 이름으로 바꾼다. `stage`는 capture의 source hash와 원래 harness가 reference 생성에 사용한 것과 같은지 검사하며, 기존 3-run binary를 실행하지 않는다.

```bash
python3 moa-dev/extra_seeds.py stage \
  --label EXTRA_LABEL \
  --references moa-dev/experiments/artifacts/extra-seeds-run3-8 \
  --snapshot moa-dev/experiments/artifacts/SELECTED_CAPTURE_LABEL

python3 moa-dev/extra_seeds.py build \
  --directory moa-dev/experiments/artifacts/EXTRA_LABEL
```

`build`는 독립 `CARGO_HOME`, `CARGO_TARGET_DIR`, `XDG_CACHE_HOME`과 고정 `Cargo.lock`을 사용한다. 공유 target/cache를 사용하지 않으므로 첫 빌드에는 dependency 다운로드 및 전체 컴파일 비용이 든다. source·fixture의 호스트/컨테이너 hash를 빌드 전후에 확인하고, 성공한 binary와 fixture·entrypoint의 hash를 `build.json`에 고정한다. 2026-09-21 `e5-ffn-four-row` snapshot으로 stage/build/submit/report 경로를 실제 검증했다.

```bash
python3 moa-dev/extra_seeds.py submit \
  --directory moa-dev/experiments/artifacts/EXTRA_LABEL

# submission.json의 job_id 상태가 terminal이 된 뒤:
python3 moa-dev/extra_seeds.py report \
  --directory moa-dev/experiments/artifacts/EXTRA_LABEL \
  --output docs/experiments/EXTRA_LABEL-correctness.json
```

제출 제한은 70초다. 하나의 label에서 제출은 한 번만 시도한다. 네트워크 오류로 제출 결과가 불명확하면 `submit-attempt.json`과 CLI 출력을 먼저 확인한다. `report`는 아직 queued/running이면 제출을 반복하지 않고 종료하므로 나중에 다시 실행한다. Arena status의 성공·exit 0, 실제 업로드 package와 보존한 payload의 hash 일치, 30개 출력의 중복 없는 PASS, 각 출력의 100% tolerance 통과를 모두 요구한다. 실패 job은 `failed_or_incomplete`와 원본 log를 남긴다.

짝수 6개의 cycle에서 공식 harness의 기존 `sorted[len/2]`는 **upper median**이다. report는 이를 `upper_median_cycles_diagnostic`으로 명시하고 공식 3-run `measure.py` 결과와 합산하지 않는다. 추가 seed 통과는 같은 입력 분포 안에서 검증 범위가 넓어졌다는 뜻이며 모든 입력의 정확성이나 실제 checkpoint 전체 모델 검증을 의미하지 않는다.

`e5-four-row-extra`의 실제 검증 결과는 [보고서](../docs/experiments/e5-four-row-extra-correctness.json)에 있다. Arena job **70057**, 12.688초, run3..8의 30개 출력이 모두 100% tolerance PASS였다. 업로드한 binary/fixture/entrypoint hash도 빌드 결과와 일치했다. 첫 제출은 active job quota 초과로 job 생성 전에 거절됐으며, 거절 기록을 보존하고 slot이 열린 뒤 승인된 재시도 한 번으로 70057을 생성했다. 성공 log는 `tests/fixtures/extra-seeds-70057.txt`에 원문 그대로 보존했다.

회귀검증:

```bash
python3 -m unittest discover -s moa-dev/tests -v
```
