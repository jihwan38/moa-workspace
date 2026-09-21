# E47 O-projection 경계 정확도 검사

**Arena job70785에서 8-case를 모두 실행했고 case0–6은 PASS, case7은 3,840개 모두 FAIL이었다.** 공식 성능·순위 측정이 아닌 8-case 정확도 진단이다. E47의 `ops::sliding_attention_output` 및 모든 device/ops 소스를 그대로 사용한다. 공식 `test_kernels.rs`도 수정하지 않았다.

제출 디렉터리는 `../../artifacts/output-boundary-probe-e47-package/`다. `test_runtime`, `remote_entrypoint.sh`, `boundary.safetensors` 세 파일을 함께 사용한다. Root만 Arena에 제출한다.

| 파일 | 크기 | SHA256 |
| --- | ---: | --- |
| test_runtime | 2,327,432 B | `f4b05761a4efc7c1f34ab76f44b4009e3f510ba20be2c04ebf40a1ed6dc5acca` |
| boundary.safetensors | 379,768 B | `35df91a34ca00b077b35327c513bb36783355363855e214674ddaa04a47ec907` |
| remote_entrypoint.sh | 318 B | `be29bae1a4647066c647ce899e6eeba3e75aec1cacaf7a0bcb3f0742f4728da8` |

ELF registry는 O 커널 한 개뿐이다. 87,040 B device image SHA256은 E47 공식 실행 이미지와 같은 `00d8a969edf26d5a34fbf9bc5430cdd889f60fa58f122ebcbd7aa2208f584353`이며 전체 host에 정확히 한 번 들어 있다. 추출한 image는 별도 artifact의 `official-o.bin`에 복사했다. `registry-summary.json`, `source-audit.json`, `manifest.json`이 해당 검증을 기록한다.

실제 결과와 엄격한 업로드/log 검증은 [실측 JSON](../../../../docs/experiments/e47-output-boundary-probe.json), [간결 결과](../../../../docs/experiments/e47-output-boundary-probe.md), `evidence.json`에 있다. exit1은 수치 비교 실패이며 모든 case가 완료됐다. nonfinite=0, 입력 checksum 40/40과 업로드 세 파일 SHA 검증이 통과했다. 원본 커널 대조 job70799는 같은8-case 모두PASS했고 case7도3840개exact였다([비교](../../../../docs/experiments/output-boundary-comparison.md)).

## Oracle과 입력 전달

`generate_suite.py`는 frozen E47 extra snapshot의 `scripts/generate_references.py`에서 **원래 `scaled_projection`와 `rms_norm` 함수를 직접 import**한다. `reference/gemma4.py::RMSNorm.forward`를 그대로 실행한다. 현재 공식 reference와 세 함수 `scaled_projection`, `rms_norm`, `gen_sliding_attention_output`의 AST가 같음을 추가 확인했다.

계산은 BF16 x → F32 matmul(정확히 decode한 FP8 weight) → F32 channel scale → BF16 → F32 mean-square/`rsqrt(mean+1e-6)`/gamma → BF16 → residual 합 → BF16이다. 결과를 F32로 저장한다. raw dot, scale 곱, BF16 projected, mean-square, inverse RMS, norm 전후, 최종 output이 전부 finite임을 8-case 모두 검사했다. quantizer의 hi/lo 결과로 oracle을 만들거나 보정하지 않는다. E47이 내부 raw dot에 추가 BF16 경계를 갖더라도 oracle을 그 구현에 맞추지 않는다.

작은 BF16 입력 4개와 expected output은 payload에 실제 바이트로 들어 있다. FP8 weight는 Python/Rust 양쪽의 같은 좌표식으로 생성한다. 각 weight는 15,728,640 B이지만 패키지에 넣지 않는다. Rust `--host-check-only --dump-weights`가 생성한 **8개 weight 전체 바이트의 SHA256**이 Python의 것과 모두 일치했다. 실행 시에는 공식 fixture와 같은 위치 가중 32-bit checksum으로 5개 입력 × 8-case를 검증한다. 이 checksum 자체를 cryptographic hash라고 주장하지 않는다.

CPU 생성은 PyTorch 2.14.0+cpu, 4 threads, deterministic algorithms, grad disabled 조건이다. `cpu-references.json`에 입력별 SHA256/checksum, 유한성, 중간값 범위, 결과의 distinct BF16 수, oracle 원본 파일 hash가 있다. `host-check-summary.json`에는 독립 Rust 생성의 SHA256 검증이 있다. Host-only 점검은 Device 생성 없이 0.217초/최대 RSS 19,508 KiB였다. 이 수치는 실제 NPU 전송·실행 시간과 메모리를 포함하지 않는다.

## 여덟 경우

기본 weight는 공식 generator가 사용하는 유한 E4M3 부호/지수 7–14/가수 0–7 범위(절댓값 1–240)에 있다. 기본 channel scale은 `2^-(11+h%4)`, gamma는 `(1+h%7)/8`, residual은 `(h%5)/8`을 BF16으로 만든다. RNG seed를 사용하지 않는 진단 fixture이며 실제 공식 PRNG 조합을 찾았다는 뜻이 아니다.

| case | 입력과 관측 목적 | 한계 |
| --- | --- | --- |
| 0 | x의 모든 원소가 교대 +0/-0. 기대 output은 원래 residual. zero-amax와 floor 경로. | 최종 비교는 공식 수치 tolerance이므로 signed-zero bit 보존 검사가 아니다. |
| 1 | 영벡터 256열 group과 nonzero group을 섞고 q=0/255/256/511/512/2047/2048/4095에 별도 값을 넣는다. h=119/120/1919/1920/3839의 scale도 다르게 한다. | 일부 원래 zero group의 경계 anchor는 의도적으로 nonzero로 덮는다. 모든 가능한 순열을 식별하지는 않는다. |
| 2 | group별 2^-7..1의 바로 아래 BF16 값이 amax다. 양쪽 부호 anchor와 작은 다른 성분을 포함한다. | 비교적 작은 지수 범위의 경계다. |
| 3 | 같은 group의 amax가 정확히 2의 거듭제곱이다. | case 2/4와 weight 식의 case salt도 달라 독립 검사가 되며 단일변수 성능 비교는 아니다. |
| 4 | amax가 각 거듭제곱 바로 위 BF16 값이다. | 1 위의 1.0078125 및 음수 대칭은 공식 [-1,1] 활성값 범위 밖이다. 자료형으로는 유한하다. |
| 5 | amax=2^e를 두고 s=2^(e-7)의 `s*2^-6`, `s*2^-14` 바로 아래/같음/위 BF16을 양쪽 부호로 넣는다. 작은 성분을 반복한다. | 이 scalar 경계가 최종 tolerance 실패를 반드시 일으키지는 않는다. |
| 6 | 부호가 섞인 BF16 bit 0x0000..0x03ff, minsubnormal 0x0001/minnormal 0x0080 포함. amax<2^-119이고 residual=0/gamma=1. | 기대 output 절댓값 ≤8.19e-34. atol .05 때문에 전부 zero여도 PASS한다. floor 분기·finite 실행 control이며 tiny 값 보존 증명이 아니다. |
| 7 | 각 256열 group에서 `[+1,-1,254×2^-22]`, 모든 row weight는 `[1,1,254×240]`, channelScale=2^-10, gamma=residual=.5. cancellation 뒤 작은 성분이 RMS를 거쳐 관측된다. | 개별 값은 공식 분포의 표현 가능한 support 안이지만 의도적으로 상관시킨 adversarial 배열이다. |

Case 7은 독립 수학/CPU 감사로 확인한 반례다. 원본 F32 raw dot은 모든 행 `0.2325439453125`, channel-scale 뒤 BF16은 `0.0002269744873046875`, 최종 BF16 expected는 **0.609375(0x3f1c)**다. group amax=1이면 quantizer s=2^-7이고 tiny/s=2^-15라 hi와 lo가 모두 0으로 반올림된다. 이 모델의 output은 0.5이며 오차 0.109375는 허용값 0.05609375보다 크다. 이 실패 예측은 job70785의 실제 NPU에서도 재현됐다. oracle의 raw dot은 모든 항이 정확한 이진수이고 현재 합산 범위에서 F32 순서 민감성이 없는 구성이다. 이 case의 power-of-two channel scale은 추가 BF16 경계 차이로 인한 혼입도 피한다.

## 실행과 판정

`probe_output.rs`가 payload/생성 weight checksum을 검사하고 공식 host와 같은 `HbmTensor` 타입 및 인자 순서로 O만 실행한다. x `[Ns,Gs,Ds]`, weight `[H,Qs]`, gamma/channel-scale/residual `[H]`다. 각 case는 residual을 새로 올리므로 이전 실행 상태를 입력으로 쓰지 않는다.

모든 3,840개 output을 원래 `abs(got-want) <= .05 + .01*abs(want)`로 비교하며 nonfinite는 무조건 실패다. case별 실패 수/nonfinite/exact count/max·mean error, 32개 120-row group별 실패 수, 대표 row boundary와 첫 32개 mismatch의 BF16 bits를 출력한다. **수치 FAIL을 만나도 8-case를 모두 수행**한 뒤 총실패가 있으면 exit1이다. NPU launch/전송 오류는 안전하게 이어갈 수 있다고 가정하지 않고 프로세스 실패로 남긴다. 이 경우 완료하지 못한 case는 PASS가 아니다. profiler/cycle/rank 보고는 없다.

이 검사는 전체 유한 BF16/FP8 영역, NaN/Inf, 극대 exponent, 모든 cancellation 구조를 열거하지 않는다. 기존 E47 공식 9개 입력 PASS와 독립적인 관측이며, 이 suite PASS라도 전체 입력 정확도 증명이 아니다.

## 재현

동결 원본은 `../../artifacts/e47-output-sixteen-columns/`의 src/Cargo/toolchain이며 CPU reference scripts만 별도 동결된 `e47-output-sixteen-columns-extra/crate/scripts/`에서 가져왔다. 기존 src 43개 파일은 byte-identical이고 추가 파일은 `src/bin/probe_output.rs`다. Scratch와 로그/전체 재현 copy는 `../../artifacts/output-boundary-probe-e47/`에 보존했다. payload와 binary는 ignored artifacts다.

```bash
cd /tmp/moa-output-boundary-probe-e47
python generate_suite.py --official-scripts scripts --out suite
export CARGO_TARGET_DIR=$PWD/target
export FURIOSA_OPT_OUT_DIR=$PWD/target/furiosa-opt/kernel
cargo furiosa-opt build --release --locked --bin probe_output
target/release/probe_output --fixture suite/boundary.safetensors --host-check-only --dump-weights host-generated-weights
```

Host build 1회, compile 실패 0회다. 첫 generator 실행의 grad/numpy 오류는 grad disabled로 수정했고 oracle 계산식은 바꾸지 않았다. 코드/API는 SDK/compiler 0.8.1 실제 사용 경로를 확인했다. 공식 host의 입력/launch 방식은 `src/bin/test_kernels.rs::sliding_attention_output`, 수치 기준은 그 파일 `compare`, reference는 `scripts/generate_references.py::gen_sliding_attention_output`에서 가져왔다. 기존 [E47 정확도 감사](../output-fp8-sixteen-columns/correctness-audit.md)의 미관측 경계를 보완한다. 확인일 2026-09-21.

독립 수학 검토는 [boundary design audit](../../../../docs/research/output-boundary-design-audit.md)에 있다. 이 기록은 공식 seeded score의 PASS를 취소하거나 전체 입력 규칙을 새로 정의하는 문서가 아니다. 실제 correlated 입력에서 오차 허용치를 벗어남을 명시한다.

## Variant별 read-only 결과 수집

`collect_job.py`는 최초 E47 job70785를 수집한 고정 collector다. 이후 원본/E53 대조는 공통 `collect_boundary.py --config PATH`를 사용한다. 각 candidate의 `collector-config.json`이 이미 제출된 job ID, 이름, frozen manifest, 예상 O image SHA, 공통 CPU reference 및 결과 경로를 지정한다. 출력의 variant label도 config에서 가져오므로 원본/E53을 E47이라고 잘못 표시하지 않는다.

Collector가 호출하는 Arena 동작은 해당 ID의 `status`, `logs`, `download` 세 가지뿐이다. terminal status가 FAILED여도 로그/다운로드를 보존하고 수치 비교 실패와 미완료 실행을 구분한다. 업로드 세 파일 SHA, job manifest의 runner/no args/no env, image 정확히1회 포함, host_only=false, 입력 checksum40개, 8-case의 결과와 histogram/대표 bits/mismatch를 교차 검사한다. 공식 score나 cycle 결과는 만들지 않는다. Frozen binary/fixture/probe source를 바꾸거나 재제출하지 않는다.
