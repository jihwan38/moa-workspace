# E20 V 실패 진단 실행 파일

2026-09-21. `qkv-fp8-split-two-lane`의 V 실패 원인을 구분하기 위한 **개발용 host 진단**이다. 공식 제출용 최적화 소스가 아니다. E20의 저장된 커널·fixture·입력 생성·허용오차·판정 함수를 유지하며, 원래 비교 후 V 실패 위치와 실제 BF16 값을 출력한다. 이 agent는 NPU를 실행하지 않았다.

원형은 `../../artifacts/e20-qkv-fp8-two-lane-output-rms/`의 동결된 source49개 파일이다. 먼저 원형을 독립 scratch에서 `--locked` fullbuild하니 기존 E20 실행 파일과 byte-identical인 SHA256 `9bfc571e4319439c085925d94b63d2add3bfdd7d3b99296dbe36615e7bc12f2e`를 얻었다. 이후 **`src/bin/test_kernels.rs`만** 변경했다. `compare` 본문은 byte-identical이며 public kernels와 세 device `.bin`도 원형 재빌드 결과와 byte-identical이다.

진단 실행 파일: `../../artifacts/qkv-fp8-two-lane-diagnostics/test_kernels`.

- SHA256: `3c2e6ee3a2a672e2bae6070c0e3b619c3d316ad2ad8b5f29bd82e4e45534527c`
- 크기: 3,655,760 bytes
- fixture SHA256: `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`
- build: compiler/std0.8.1, 원형 Cargo.lock 고정, `cargo furiosa-opt build --release --bin test_kernels --locked -j 8`
- scratch: Docker `/tmp/moa-qkv-fp8-two-lane-diagnostics`, 독립 target와 kernel output 사용

`DIAG_V_FAIL`은 모든 실패 위치의 index/head/dim, canonical KV cluster/slice/local row, expected/actual 및 BF16 상위16bit를 출력한다. `DIAG_V_SUMMARY`는 head별 및 4개 local row별 실패 수다. `DIAG_V_BF16`은 통과 값을 포함한 V2048개를 64개씩 lossless hex로 출력한다. 실제값 전체를 보존하므로 재실행 없이 head RMS, 공통 배율, 오류의 공간 패턴을 분석할 수 있다. 추가 출력 때문에 이 실행 파일의 성능을 공식 성능 증거로 쓰지 않는다.

raw 로그가 확보되면 workspace root에서 다음과 같이 분석한다.

```sh
python3 moa-dev/experiments/candidates/qkv-fp8-two-lane-diagnostics/parse_diagnostics.py \
  PATH_TO_RAW_LOG \
  moa-dev/experiments/artifacts/e20-qkv-fp8-two-lane-output-rms/ref/fixtures.safetensors
```

parser는 Python 표준 라이브러리만 사용한다. 별도 synthetic 검사로 원형 fixture 세 입력 roundtrip에서 실패0개, index257에 한 개 오류를 주었을 때 해당 위치만 검출함을 확인했다. 실제 NPU 결과는 아직 분석하지 않았다.

## 코드에서 확인한 범위

`ops.rs:65–83`의 경로는 input RMSNorm→FP8 hi/lo 분해→Q/K/V projection→각 head RMSNorm→Q/K RoPE→cache/output이다. **V도 `sliding::rmsnorm::normalize_value`로 정규화된다.** Q/K와 달리 gamma 및 RoPE가 없으며, 마지막 나눗셈이 `vector_fp_div`라는 점이 다르다. “V만 정규화되지 않아 드러난 오류”라는 설명은 잘못이다.

`project_one_kv_matrix`는 K/V에 같은 수식과 layout을 적용한다. 두 lane의 순서는 hi/lo이며, `dot_hi + dot_lo/16` 후 공통 입력 scale, BF16 cast, HBM에서 읽은 row별 channel scale, BF16 cast 순서다. SDK `engine/contraction/outer/mod.rs`는 f8 입력 누적형을 f32로 정의한다. H3840은64로 나누어떨어져 H tail은 없다. lane2 입력은 slice당7680bytes다. 컴파일 성공과 SDK 검증만으로 장치의 lane packing·allocator·backend 정확성을 증명하지는 않는다.

공식 입력 weight exponent band7–14는 최대 절댓값240이다. quantizer의 hi는 최대256, lo는 최대128이므로 한 row의 단순 절댓값합 상한은 hi235,929,600, lo117,964,800이다. 이는 F32 overflow 범위와는 멀다. 다만 F32 타입이라는 사실이나 이 상한은 FP32 누적 반올림 순서를 증명하지 않으며, cancellation 뒤의 상대오차 보장도 아니다.

E19 두 별도 dot은 동일 분해를 사용하여 V 세 입력 모두 PASS였다. E20 two-lane와 E23 K/V register 재적재는 V의 run별 max/mean/허용오차비율이 동일하게 FAIL했다. 따라서 **일반 FP8 분해의 근사 오차 또는 단순 공유 TRF 재사용만으로 원인이 확인됐다고 할 수 없다.** O의 two-lane 공식3개 PASS도 QKV packed/Ps layout의 정확성 증거는 아니다.

다음 판단은 실제 V dump에 달려 있다. 특정 modulo4/8 혹은 head에 집중되면 commit/typed movement/padding을 우선 조사하고, head 내부 공통 배율이면 RMS scalar 또는 scale을 조사한다. 산발적인 잘못된 값이나 순열이면 packing/readback 위치를 조사한다. 출력만으로 projection·norm·cache 중 발생 단계를 구분하지 못하면 별도 개발 entrypoint에서 intermediate를 읽는 두 번째 진단이 필요하다. 어느 경우도 현재 코드나 공식 fixture를 특수 처리하는 최적화 근거로 사용하지 않는다.

원본 read-only source 복사를 수정하려면 scratch 파일 쓰기 권한이 필요했다. 또한 `docker cp`가 기존 mtime을 보존해 첫 incremental build는 stale이었으므로 명시적으로 scratch test 파일을 touch하고 재빌드했다. 최종 binary에 진단 문자열이 포함됐고 SHA가 변경됐으며, 모든 source49개 및 kernel hashes를 다시 확인했다. 로그와 검증 원본은 `../../artifacts/qkv-fp8-two-lane-diagnostics/`, 요약은 `evidence.json`에 있다.
