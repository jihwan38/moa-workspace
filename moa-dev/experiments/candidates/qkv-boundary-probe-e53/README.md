# QKV 상쇄 반례와 보존 대조군

**실제 NPU 결과: E53 job70943은 case0에서8,072개 출력이 실패했고, 원본 job70944와 E54 job70958은 두 case의 Q/K/V를 모두 통과했다.** 세 job 모두2/2 case를 완료했고 nonfinite=0이었다. E53의 exit1은 수치 비교 실패이며 timeout이 아니다.

| 구현 / 실제 결과 | case0 (`2^-23`) | case1 (`2^-12`) |
|---|---|---|
| [E53 job70943](../../../../docs/experiments/e53-qkv-boundary-probe.json) | Q4016/K2008/V2048개 FAIL | Q/K/V8192개 모두 exact |
| [원본 job70944](../../../../docs/experiments/original-qkv-boundary-probe.json) | Q/K 최대 오차0.0078125, V2048개 exact; 모두 PASS | Q/K/V8192개 모두 exact |
| [E54 job70958](../../../../docs/experiments/e54-qkv-boundary-probe.json) | Q/K 최대 오차0.0078125, V2048개 exact; 모두 PASS | Q/K/V8192개 모두 exact |

원본과 E54의 case0 exact count는 각각 Q2928/K1464/V2048이다. 집계값이 같다는 사실만으로 전체 출력 배열이 서로 동일하다고 추론하지 않는다. 세 실행 모두 입력 checksum38/38, cache before/after8/8을 통과했고 **각 job의 비대상 KV cache8,380,416개 BF16 bits가 보존**됐다. payload·runner는 동일하고 각 구현의 서로 다른 QKV image가 해당 host에 한 번씩 포함됨을 업로드 package에서 확인했다. 두 입력 모두 현재 공식 QKV generator의 x 지원 범위 밖이라는 한계는 유지한다. [전체 세 구현 대조](../../../../docs/experiments/qkv-boundary-comparison.md).

E54는 같은 probe와 payload로 추가 빌드한 `artifacts/e54-package/`다. [E54 manifest](e54-manifest.json)에 host SHA256 `515a45f43281df98d1f8fa07b857fab3e1dbdc43ee2ccd051fc29e7cb4352720`, QKV image SHA256 `57a52d1aa37f16116d14638b30f34947d2c7e7ed653986436cdfa51ee54089dc`와171,776B/1회 포함을 기록했다. **각 manifest의 상태는 빌드 준비 당시 동결 기록이며, 실제 NPU 판정은 위 job별 결과 JSON을 기준으로 읽는다.**

## E55 실제 후속 결과

**[E55 job71032](../../../../docs/experiments/e55-qkv-boundary-probe.json)는 같은2개 case의 Q/K/V6개 비교를 모두 PASS했다.** exit0, nonfinite=0, 출력·cache 실패0이었다. case0의 Q/K 최대 절대 오차는0.0078125이고 exact count는 Q2928/K1464/V2048, case1은 Q/K/V8,192개 모두 exact다. 원본/E54와 같은 집계가 전체 출력 배열의 동일성을 입증하지는 않는다.

기존 probe·payload·runner를 그대로 사용한 `artifacts/e55-package/`의 결과다. 입력 checksum38/38, cache 검사8/8이 PASS하며 비대상 cache8,380,416개 BF16 bits가 보존됐다. CPU 준비에서도 실제 Rust 입력 전체38개 SHA256이 Python과 일치했다. [E55 manifest](e55-manifest.json)에 host2,657,592B SHA256 `c1e2e5fa47e1f4d1b5cfafca7135eb6f14aa9497c1cbc1f89f85ab7dc32e8316`과 QKV image168,960B SHA256 `5bdab7eafec5ac0aca352101deee334180b550c87db23e33c70d008892edba9e`를 기록했다. [Source 감사](e55-source-audit.json)는 동결 source43개와 Cargo/toolchain3개가 일치하고 동일 probe만 추가됐음을 확인하며, [registry 감사](e55-registry-summary.json)는 QKV image가 정확히 한 번 포함됨을 확인한다. 실제 업로드 package hash도 collector가 대조했다.

manifest의 `NOT_NPU_TESTED`는 준비 당시 보존한 상태이며 현재 판정은 job71032 결과 JSON이다. 이 후속은 공식 generator 지원 범위 밖의 두 입력에 대한 정확도 관측이다. [성능 반복3회](../../../../docs/experiments/e55-fused-qkv-repeats.json)와 [추가 seed30개 비교](../../../../docs/experiments/e55-fused-qkv-extra-correctness.json)도 별도 완료됐다. 이 probe만으로 성능 개선이나 모든 입력 동치를 주장하지 않는다. [기존 대조와 E55 후속](../../../../docs/experiments/qkv-boundary-comparison.md#e55-후속--job71032).

## 준비 당시의 구현·검증 기록

2026-09-21. **E53와 upstream 원본의 별도 host 패키지를 빌드하고 CPU 입력 검증을 완료했다. 이 준비 작업에서는 NPU를 실행하지 않았다.** 공식 `test_kernels.rs`나 device 코드를 변경하지 않고 `probe_qkv.rs`만 추가했다. 공식 성능·점수·순위 측정이 아닌 일반 입력의 수치 진단이다. NPU 제출과 결과 수집은 root가 담당한다.

두 패키지는 이 폴더 아래 ignored 디렉터리 `artifacts/e53-package/`, `artifacts/original-package/`다. 각각 `test_runtime`, `boundary.safetensors`, `remote_entrypoint.sh`를 함께 제출한다. manifest는 `e53-manifest.json`, `original-manifest.json`이다. `artifacts/`는 이 폴더의 `.gitignore`로 제외했다.

| 패키지 | host SHA256 | 포함 QKV image SHA256 |
|---|---|---|
| E53 | `e4df9d00afcad6650ffdd9469af781ccf4a1823af0fc72a7649123e209441461` | `91b232f0179b37b126f2ba8755b6125e532364418516948fd98dadcc9d7e77ea` |
| upstream 8559916 | `ca7352cf245db33153dad37b84b3ac607d7d06c60c2e914e17f43c53b9303c29` | `75eacabd18926cbefdebdded06d8bb3313b810d25d83faf8006e807d7384fe5b` |

두 host의 ELF registry에는 `furiosa_opt_gemma4::ops::sliding_project_qkv` 한 개만 있고, 해당 image가 각각 정확히 한 번 들어 있다. E53 image는 187,648 B이며 실제 E53 공식 실행 host의 이미지와 byte-identical이다. 원본 image는 183,040 B이며 초기 원본 baseline host와 byte-identical이다. `inspect_registry.py`와 variant별 `*-registry-summary.json`이 이를 검증한다.

Payload는 두 패키지에서 동일한 138,912 B, SHA256 `fad33f6da059c34bbdb69b9817a3bbe1ef6d2dadfad59f6b134ef2d4cca848ed`다. runner는 기존 O probe와 동일한 318 B, SHA256 `be29bae1a4647066c647ce899e6eeba3e75aec1cacaf7a0bcb3f0742f4728da8`다. host source SHA256은 `893e8d17e21387962f5ff97ac8fb25420272c6c4469fd277ac30bcbad4b25292`다.

## 두 입력과 원래 oracle

모든 H480 chunk를 다음처럼 만들고 H3840 전체에 반복한다. 두 경우 모두 pregamma=1, 모든 Q/K/V weight row는 `[1,1,478×240]`을 8회 반복한다. FP8 1/240 code는 `0x38`/`0x77`이고 channel scale은 BF16 `2^-10`, Q/K head gamma는1이다. V는 원래 unweighted head RMS를 사용한다.

| case | x의 H480 chunk | 목적 | CPU 최종 V |
|---|---|---|---:|
| 0 | `[+1,-1,478×2^-23]` | pre-RMS 뒤 작은 성분이 FP8 hi/lo에서 사라지고 큰 항이 상쇄되는 반례 | 0.85546875 |
| 1 | `[+1,-1,478×2^-12]` | 같은 배치에서 작은 성분을 양자화가 보존하는 크기로 올린 대조군 | 1.0 |

**두 입력 모두 현재 공식 QKV generator의 x 지원 범위 밖이다.** 공식 generator는 `BF16(signs/input_rms_weight)`를 사용하므로 pregamma=1인 경우 x가 ±1이다. 이 두 배열은 공식 PRNG 입력 실패를 주장하는 근거가 아니며 전체 유한 입력의 정확성을 요구하는 새 규칙도 아니다. 앞선 [CPU 수학 감사](../../../../docs/research/qkv-boundary-design-audit.md)의 진단을 실제 장치에서 비교할 준비다.

`generate_suite.py`는 동결된 원래 `scripts/generate_references.py`에서 `rms_norm`, `scaled_projection`, `apply_rotary_pos_emb`, `rope_tables`, `_run_pos`를 직접 import한다. 실제 fullshape Q4096×H3840, K/V2048×H3840 행렬을 만들어 원래 F32 matmul과 BF16 경계를 실행한다. pre-RMS → projection/channel scale → head RMS → Q/K RoPE가 oracle이며, hi/lo 근사 결과를 정답에 반영하지 않는다. 원래 run0 position1763, cache slot739를 사용한다. CPU oracle은 원래 sin을 사용하고 device용 sin의 low half만 원래 `negate_low_half` 함수로 반전한다.

fullshape CPU matmul의 case0 raw는 **1.69500732421875**다. 이전 단일 행/해석적 합1.6956024169921875와 F32 합산 순서가 다르지만, channel scale 뒤 BF16과 최종 Q/K/V BF16은 앞선 감사와 같다. 이 차이를 보정하거나 oracle에 해석적 값을 대입하지 않았다. case1 raw는3472.96875다. `cpu-references.json`에는 이 값과 실제 모든 중간 범위가 기록돼 있다. x, pre-RMS mean/inverse/output, raw projection, F32 scale, BF16 projection, head RMS mean/inverse/output, 최종 output의 유한성을 두 case 모두 확인했다.

`oracle-function-audit.json`은 위 함수들과 `gen_sliding_project_qkv` 등7개 함수의 AST가 현재 원래 reference와 같음을 확인한다. 별도 CPU 양자화 모델 `activation-decomposition.json`에서는 case0의3824개 작은 activation이 소실되고 case1은3840개 모두 정확히 복원됐다. 이 모델은 oracle을 생성하거나 수정하는 데 사용하지 않았으며 SDK/NPU의 bit-exact 에뮬레이션을 주장하지 않는다.

## 입력 전달과 별도 cache 검사

`probe_qkv.rs`는 원래 `src/bin/test_kernels.rs::sliding_project_qkv`의 HBM 타입·launch 인자 순서·offset 계산·cache read 위치를 그대로 사용한다. Q/K/V weight는 Python/Rust의 같은 좌표식으로 생성하며 대형 weight를 패키지에 넣지 않는다. RoPE는 payload의 BF16 한 행을 E32768×D256 전체 zero buffer의 position1763에 넣는다. K/V cache 전체는 각각 BF16 `0x3e80`(+.25), `0xbf00`(-.5), Q output은 `0x3fc0`(1.5)로 초기화한다. 각 case마다 모든 입력과 output buffer를 새로 올린다.

각 case의 19개 입력에 공식 fixture와 같은 위치 가중 u32 checksum을 적용한다. 원래 입력들에 더해 cos/sin row와 전체 buffer, cache/Q 초기화 전체 buffer도 검사한다. checksum을 cryptographic hash라고 주장하지 않는다. `--host-check-only --dump-generated DIR`은 Device를 만들지 않고 실제 Rust 입력 전체 바이트를 저장한다. `verify_host.py`는 이 38개 파일 전체의 SHA256을 Python CPU 원장과 대조한다. E53와 원본 host 각각 **38/38 checksum, 38/38 전체 SHA256, 2/2 case 완료**가 통과했다. `host-check-summary.json`, `original-host-check-summary.json`이 결과다.

NPU 모드에서는 upload 직후 K/V cache 전체를 read-back해 sentinel bit를 검사한다. QKV 실행 후 다시 전체 cache를 읽고 대상2048개 값 `[1513472,1515520)`만 제외한 영역을 bit 단위 비교한다. case별 before/after×K/V의4개 검사, 전체8개 `CACHE_CHECK`/`cache_check` JSON이 나온다. before 각2,097,152개, after 각2,095,104개를 검사하며 after 총8,380,416개 BF16 bits다. 일정 sentinel이므로 비대상 슬롯끼리의 교환 등 모든 손상을 식별하지는 않는다. Q output 초기 sentinel이 남으면 일반 Q 비교가 실패한다.

Q4096/K2048/V2048의 모든 값을 각각 `abs(got-want) <= .04 + .01*abs(want)`로 비교한다. **actual NaN/Inf는 무조건 실패**이며 expected는 host에서 유한성을 먼저 assert한다. 출력별 fail/nonfinite/exact/max error와 첫 mismatch bits를 JSON 한 줄에 출력한다. 추가 text에는 첫16개 mismatch가 있다. cache sentinel 결과는 별도 JSON이다. `QKV_SUITE_RESULT`는 두 case를 모두 수행한 후 output/cache 실패 합계를 보고하고 실패가 있으면 exit1이다. NPU launch/전송 자체가 실패하면 미완료 case를 PASS로 처리하지 않고 프로세스가 실패한다. 이 host에는 profiler/cycle/score 수집이 없다.

## source 동결과 재현

E53는 `../../artifacts/e53-bf16-output-clean-source`의 기존 src43개를 byte-identical하게 복사했다. 원본은 앞선 원본 O control에 보존된 upstream8559916의43개 source를 재사용하고 `source-origin.json`의 파일별 SHA와 모두 대조했다. 두 scratch의 실제 src44개(기존43+probe1)를 빌드 후 다시 읽어 동결 copy와 일치하는지 확인했다. `source-audit.json`, `original-source-audit.json`, `*-compiled-input-hashes.json`을 참고한다. 공식 host `test_kernels.rs`는 두 source 모두 변경하지 않았다.

Docker scratch는 `/tmp/moa-qkv-boundary-probe-e53`, `/tmp/moa-qkv-boundary-probe-original`이다. 각각 기존 O probe의 dependency target을 `cp -a --reflink=auto`로 독립 복사했으므로 shared target을 수정하지 않았으며 fresh-target build라고 주장하지 않는다. 같은 compiler/std0.8.1과 고정 Cargo.lock/toolchain을 사용했다. 두 host 모두 build1회/실패0회다. E53 compile은10.19초, 원본은8.84초이며 실제 NPU 성능과 무관하다.

```bash
cd /tmp/moa-qkv-boundary-probe-e53
python generate_suite.py --official-scripts scripts --out suite
export CARGO_TARGET_DIR=$PWD/target
export FURIOSA_OPT_OUT_DIR=$PWD/target/furiosa-opt/kernel
cargo furiosa-opt build --release --locked --bin probe_qkv
target/release/probe_qkv --fixture suite/boundary.safetensors \
  --host-check-only --dump-generated host-generated > host-check.log
python verify_host.py --cpu suite/cpu-references.json \
  --dump host-generated --log host-check.log
```

통과 범위는 이 두 입력·단일 RoPE/cache 위치·sentinel 패턴뿐이다. 준비 후 root가 수행한 실제 NPU 결과는 문서 상단의 collector 기록에 연결했다. 빌드 준비 기록만으로 hardware PASS나 실패 원인을 단정하지 않는다.
