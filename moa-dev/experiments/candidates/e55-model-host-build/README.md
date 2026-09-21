E55의 **실제 모델용 `gemma4`와 `server`를 격리 빌드하고 ELF의 kernel registry를 검사한 기록**이다. 두 빌드가 성공했고 각각14개 device image를 포함한다. Stage1 세 image는 NPU에서 측정한 E55와 정확히 같다. 모델 가중치·tokenizer를 적재하거나 NPU를 실행하지 않았으므로 전체 모델 정확도·성능은 미검증이다. [결과](../../../../docs/experiments/e55-model-host-build.json).

기존 `test_kernels` 빌드는 세 device 함수만 도달한다. 반면 `gemma4::main`과 서버 worker는 `generate → Decode::run/run_with_embedding → run_layers`를 통해 embedding, attention, logits와 멀티모달 경로도 참조한다. 실제 caller를 추적해 얻은 집합과 산출물의 registry를 비교했다. [실행 흐름·필수 자산](../../../../docs/research/e55-model-runtime-readiness.md).

| 검사 | 관측 결과 |
|---|---|
| Source/Cargo/toolchain | 실제 scratch 입력46개가 frozen E55와 동일, 빌드 전후 동일 |
| `gemma4` | 새 target에서 release build 성공, Cargo 보고58.02초 |
| `server` | 앞선 target을 재사용한 release build 성공, Cargo 보고1.10초 |
| 두 ELF의 `furiosa_kernels` | 각각14개 entry, 모든 generic key `[]`, 전체 registry1451229B byte-identical |
| Registry image | 각 entry가 격리 compiler의 해당 `.bin`과 byte-identical |
| Stage1 연계 | 실측 `test_kernels` ELF의 세 image와 양 모델 ELF의 해당 image가 byte-identical |

Registry 집합은 `ops`의10개, vision의3개, audio의1개다. 선언만 있고 host caller가 없는 `ops_vision::layernorm`은 포함하지 않는다. CLI는 text/image를 입력받아 실제 도달 가능한 집합은13개지만, 공통 `generate`의 audio branch도 registry에 남아 있다. 서버의 chat `input_audio` 경로는14개 모두 도달할 수 있다. `full_attention_page`는 position512부터 필요하므로 짧은 text 실행만으로 모든 text 커널이 실행되는 것도 아니다.

`verify_registry.py`는 ELF64 little-endian section header에서 정확한 `furiosa_kernels` section을 찾고, 설치 SDK0.8.1의 `backend/npu/registry.rs`가 정의한 bincode1.3.3 `Vec<(path, Vec<u64>, image)>`를 끝까지 읽는다. Entry 중복·정렬·empty key·expected path set를 검사하고 실제 image bytes를 비교한다. 문자열 검색이나 binary 안에 source 함수 이름이 존재한다는 사실로 registry를 추정하지 않는다. SDK는 첫 device launch 때 path/key로 이 registry에서 image를 선택하지만, 이 검사에서 실제 NPU loader를 실행한 것은 아니다.

별도 decoder로 실제 산출물을 독립 재검증했다. 이 검토에서 Python `-O`가 `assert`를 제거할 수 있는 문제를 찾아 최종 검사기는 시작 시 optimization mode를 거부하도록 수정했다. 수정 후 실제 ELF 검사를 다시 통과했고, `python -O`에서는 PASS 출력 전에 nonzero로 종료함을 확인했다.

원시 [빌드 로그·ELF·image·검사 출력](../../artifacts/e55-model-host-build)은 Git 제외 경로에 있고, [실제 입력 manifest](compiled-source-manifest.json)와 [검사 코드](verify_registry.py)는 추적한다. 빌드 기준은 commit `10603864d28f2a8fe292a5198cd09ad8d21c4302`의 crate source 및 별도 추적한 [Cargo.lock](../../../locks/gemma4-Cargo.lock)이다. 현재 live source/target은 변경하지 않았다.

재현할 때 source43개와 Cargo/toolchain3개를 새 scratch로 복사하고 manifest를 먼저 대조한다. 아래는 실제 사용한 경로·환경이다. 이미 있는 artifact를 덮어쓰지 않으려면 다른 새 scratch를 사용한다.

```bash
cd /tmp/moa-e55-model-host-build
export CARGO_HOME=/home/moa/.cargo
export CARGO_TARGET_DIR=/tmp/moa-e55-model-host-build/target
export FURIOSA_OPT_OUT_DIR=/tmp/moa-e55-model-host-build/target/furiosa-opt/kernel
cargo furiosa-opt build --release --bin gemma4
cargo furiosa-opt build --release --bin server
```

두 실행 파일을 artifact 디렉터리의 `gemma4`, `server`로, compiler의 `kernel/furiosa-opt-gemma4/`를 `kernels/`로 복사한 뒤 workspace root에서 검사한다.

```bash
python3 moa-dev/experiments/candidates/e55-model-host-build/verify_registry.py \
  --artifact moa-dev/experiments/artifacts/e55-model-host-build \
  --stage1-host furiosa-opt-gemma4-12B/target/release/test_kernels
```

이 검사 코드는 E55의 세 image hash와 현재14-entry 호출 계약에 고정되어 있다. 커널 또는 SDK/호출 경로를 바꾸면 새 근거로 기대값을 검토해야 하며, 실패를 통과시키기 위해 이전 실측 hash만 바꾸면 안 된다. 현재 검사에서는 모델 자산, device loading, logits, 생성 token, 처리량 및 공식 채점 결과를 검증하지 않았다.
