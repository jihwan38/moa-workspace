# QKV 상쇄 진단: E53·원본·E54 대조와 E55 후속

2026-09-21. **E53 job70943은 작은 성분이 `2^-23`인 case0에서 Q/K/V 총8,072개가 tolerance를 벗어났고, 원본 job70944와 BF16 QKV 대안 E54 job70958은 같은 두 case의 Q/K/V를 모두 통과했다.** 세 실행 모두 계획된2개 case를 완료했고 nonfinite=0이었다. E53의 exit1은 출력 수치 실패이며 launch timeout이나 실행 누락이 아니다. 근거는 [E53 결과](e53-qkv-boundary-probe.json), [원본 결과](original-qkv-boundary-probe.json), [E54 결과](e54-qkv-boundary-probe.json)다.

이 입력들은 **현재 공식 QKV generator의 x 지원 범위 밖**이다. 공식 seeded 평가 실패나 공식 점수·순위 결과로 해석하지 않는다. 별도 [probe 설계와 재현](../../moa-dev/experiments/candidates/qkv-boundary-probe-e53/README.md)의 일반 입력 수치 진단이며 공식 `test_kernels.rs`를 변경하지 않았다.

## 같은 입력에서의 결과

두 case 모두 H480 chunk마다 `[+1,-1,478×작은 성분]`을 반복한다. pregamma=1, 모든 Q/K/V weight row는 `[1,1,478×240]`의8회 반복, channel scale=`2^-10`, Q/K head gamma=1이다. 실제 원래 pre-RMS·head RMS·run0 RoPE(position1763)를 포함한다. case0의 작은 성분은 `2^-23`, case1은 `2^-12`다. 비교는 `abs(got-want) <= .04 + .01*abs(want)`이며 actual NaN/Inf는 무조건 실패다.

각 칸은 `실패수/전체 개수; 최대 절대 오차`다.

| case / 출력 | E53 job70943 | 원본 job70944 | E54 job70958 |
|---|---:|---:|---:|
| 0 / Q |4016/4096; 1.2109375|0/4096; 0.0078125|0/4096; 0.0078125|
| 0 / K |2008/2048; 1.2109375|0/2048; 0.0078125|0/2048; 0.0078125|
| 0 / V |2048/2048; 0.85546875|0/2048; 0|0/2048; 0|
| 1 / Q |0/4096; 0|0/4096; 0|0/4096; 0|
| 1 / K |0/2048; 0|0/2048; 0|0/2048; 0|
| 1 / V |0/2048; 0|0/2048; 0|0/2048; 0|

E53 case0의 각 출력 첫 불일치는 모두 got=0이다. Q/K index0의 expected는−0.265625, V index0은0.85546875다. 실패 개수는 앞선 FP8 hi/lo CPU 모델의 예측과 일치한다. 원본과 E54의 case0은 각각 Q2928개/K1464개/V2048개가 expected와 정확히 일치했고, 나머지 Q/K도 허용오차 안이다. 두 구현의 Q/K를 bit-exact PASS라고 주장하지 않는다. **원본과 E54의 집계값이 같다는 사실만으로 전체 출력 배열이 서로 동일하다고 추론하지 않는다.** case1은 세 구현 모두 Q/K/V8,192개가 expected와 정확히 일치했다.

두 case 모두 공식 generator의 `x=BF16(signs/input_rms_weight)`와 다르다. pregamma=1이면 공식 x는±1이므로 `2^-23`뿐 아니라 control의 `2^-12`도 그 지원 범위에 포함되지 않는다. 이 대조는 E53의 FP8 입력 근사에 일반 입력 정확도 한계가 있음을 보여 주며, 모든 가능한 입력의 실패·성공을 증명하지 않는다.

## 입력·실행 파일·cache 증거

같은 `probe_qkv.rs`(SHA256 `893e8d17e21387962f5ff97ac8fb25420272c6c4469fd277ac30bcbad4b25292`), payload, runner를 세 frozen source에 사용했다. Payload는138,912B, SHA256 `fad33f6da059c34bbdb69b9817a3bbe1ef6d2dadfad59f6b134ef2d4cca848ed`다. 원래 Python `rms_norm`, `scaled_projection`, `apply_rotary_pos_emb`가 정답을 계산하며 장치 결과에 맞춰 oracle을 바꾸지 않았다. E54 package의 생성 시점 증거는 [E54 build manifest](../../moa-dev/experiments/candidates/qkv-boundary-probe-e53/e54-manifest.json), 실제 실행 판정은 위 E54 결과 JSON에 있다.

CPU 준비 단계에서 E53·원본 host가 실제 생성한38개 입력 전체 바이트의 SHA256이 Python 원장과 일치했다. E54도 같은 probe/payload로 host-only checksum38개를 통과했다. NPU 실행에서는 **세 job 각각 입력 checksum38/38이 통과**했다. Collector가 Arena에서 다운로드한 binary/payload/runner의 SHA를 동결 package와 대조했고, 세 구현의 서로 다른 QKV image가 각각 해당 host에 정확히 한 번 포함됨을 확인했다.

| 항목 | E53 | 원본 upstream8559916 | E54 |
|---|---|---|---|
| host SHA256 | `e4df9d00afcad6650ffdd9469af781ccf4a1823af0fc72a7649123e209441461` | `ca7352cf245db33153dad37b84b3ac607d7d06c60c2e914e17f43c53b9303c29` | `515a45f43281df98d1f8fa07b857fab3e1dbdc43ee2ccd051fc29e7cb4352720` |
| QKV image SHA256 | `91b232f0179b37b126f2ba8755b6125e532364418516948fd98dadcc9d7e77ea` | `75eacabd18926cbefdebdded06d8bb3313b810d25d83faf8006e807d7384fe5b` | `57a52d1aa37f16116d14638b30f34947d2c7e7ed653986436cdfa51ee54089dc` |
| image 크기 / host 내 개수 |187,648B /1|183,040B /1|171,776B /1|
| 실행 완료 / exit code |2/2 case /1|2/2 case /0|2/2 case /0|

세 job 모두 K/V before/after cache 검사8/8이 통과했다. 대상 slot739의2048개 값 `[1513472,1515520)`을 제외한 **비대상 cache 총8,380,416개 BF16 bits가 각 job에서 그대로 보존**됐다. cache mismatch는0이다. 일정 sentinel의 보존 검사이므로 모든 가능한 순열·손상 탐지를 보장하지 않는다. 이 관측은 출력 수치 실패와 비대상 cache 손상을 구분한다.

세 구현은 각각 전체 frozen QKV 경로를 실행한다. 이 비교만으로 다른 변경이 전혀 없는 단일 stage 인과 실험이나 compiler 결함을 주장하지 않는다. FP8 소실 모델과 결과의 일치는 근거가 되지만 내부 중간값을 NPU에서 직접 측정한 것은 아니다.

## CPU raw dot 차이와 후속 상태

선행 [CPU 감사](../research/qkv-boundary-design-audit.md)의 단일 행/해석적 raw dot은1.6956024169921875였다. 이번 실제 전체 크기 Q4096×H3840, K/V2048×H3840 행렬을 원래 함수에 넣은 oracle의 F32 raw dot은 **1.69500732421875**다. 합산 순서에 따른 차이를 그대로 보존했고, 두 계산의 channel scale 뒤 BF16 및 최종 Q/K/V BF16은 같았다. 원본 NPU의 Q/K 최대 오차0.0078125와 이 CPU raw 차이를 같은 현상으로 단정하지 않는다.

**E54 job70958의 두 case 정확도 PASS는 실제 실행으로 확인했다.** 이 probe는 cycles를 측정하지 않는다. 당시 별도 진행하던 [E54 공식 harness 반복](e54-bf16-qkv-repeats.json)은 이후 완료됐으며, 그 성능 근거와 이 정확도 진단을 구분한다.

## E55 후속 — job71032

**[E55 job71032](e55-qkv-boundary-probe.json)는 같은2개 case의 Q/K/V6개 비교를 모두 PASS했고 exit0으로 완료됐다.** nonfinite=0, 출력 실패0이었다. case0은 Q/K 최대 절대 오차0.0078125, exact count는 Q2928/K1464/V2048이며 V는 전부 정확히 일치했다. case1은 Q/K/V8,192개 모두 expected와 정확히 일치했다. 원본·E54와 요약값이 같지만 전체 출력 배열의 동일성을 추론하지 않는다.

E55도 기존과 같은 probe source·138,912B payload·runner를 사용했다. 입력 checksum38/38, K/V before/after cache 검사8/8이 통과했으며 **비대상 KV cache8,380,416개 BF16 bits가 보존**됐다. 준비 시 실제 Rust 입력38개 전체 SHA256이 Python 원장과 같았고, collector가 실제 업로드한 binary/payload/runner hash 및 아래 E55 QKV image의1회 포함을 확인했다.

| E55 artifact | 크기 | SHA256 |
|---|---:|---|
| `test_runtime` |2,657,592B|`c1e2e5fa47e1f4d1b5cfafca7135eb6f14aa9497c1cbc1f89f85ab7dc32e8316`|
| QKV image |168,960B|`5bdab7eafec5ac0aca352101deee334180b550c87db23e33c70d008892edba9e`|

Package는 `moa-dev/experiments/candidates/qkv-boundary-probe-e53/artifacts/e55-package/`다. [E55 manifest](../../moa-dev/experiments/candidates/qkv-boundary-probe-e53/e55-manifest.json), [source 감사](../../moa-dev/experiments/candidates/qkv-boundary-probe-e53/e55-source-audit.json), [registry 감사](../../moa-dev/experiments/candidates/qkv-boundary-probe-e53/e55-registry-summary.json)는 빌드 준비 시점의 동결 증거이고 실제 NPU 상태는 위 결과 JSON을 따른다.

이 결과도 공식 generator 지원 범위 밖인 두 배열·단일 RoPE/cache 위치·일정 sentinel 패턴에 한정된다. E55의 [성능 반복3회](e55-fused-qkv-repeats.json)와 [추가 seed30개 비교](e55-fused-qkv-extra-correctness.json)도 별도 완료됐다. 이 정확도 probe 자체에서 성능 개선·최종 순위·모든 입력의 동치를 도출하지 않는다.
