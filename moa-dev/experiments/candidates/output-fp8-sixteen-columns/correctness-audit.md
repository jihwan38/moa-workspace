# O16 추가 정확도 검사의 범위

2026-09-21 E47의 첫 [job70664](../../../logs/e47-output-sixteen-columns-job-70664.log)는 공식15 comparisons PASS다. O run0..2의 max|Δ|는 모두0.03125, mean|Δ|는0.000484/0.000503/0.000457이다. O tolerance는 **모든3840원소에 대해 `|got-want| ≤ 0.05+0.01*|want|`**, non-finite device output은 즉시 실패다. 이 세 run은 max error가 atol0.05보다도 작다. 이는 최종 residual 출력의 검사이며 intermediate projection/scale/partial을 각각 검사한 결과는 아니다.

동일 package 총 3회 성능 측정은 모두 PASS로 완료됐고, 추가 run3..8 job70694도 30 comparisons PASS였다. 공식 run0..2와 합쳐 서로 다른 9개 입력 구성을 검증했으며 root가 E47 O16을 채택했다. 같은 package 재실행은 동일 run0..2를 다시 쓰므로 새로운 입력 coverage는 늘리지 않는다. 반복 O job median 44,309/44,663/44,588의 중앙값 44,588은 E37 반복 중앙값 45,748보다 2.535% 낮고 job range는 0.794%다. 첫 screen의 −3.15%는 이 반복값으로 대체한다. 추가 입력의 cycle은 진단값이며 성능 aggregate에 합치지 않는다. [추가 기록](../../../../docs/experiments/e47-output-sixteen-columns-extra-correctness.json)과 [채택 source/build 감사](../../../../docs/experiments/e47-source-build-audit.json)를 참조한다.

## 실제 입력 생성과 reference 경로

공식 [test_kernels.rs](../../artifacts/e37-kernel-specific-rms/src/bin/test_kernels.rs)의 `Synth::seed`는 `run{n}.sliding_attention_output.{name}`으로 모든 텐서를 독립 생성한다. `sliding_attention_output`은 다음 값을 만든다.

| 입력 | 실제 생성 범위 |
|---|---|
| BF16 x,4096개 |F32 uniform[-1,1)을 RNE BF16로 반올림; ±1이 나올 수 있음|
| FP8 o_weight,3840×4096 |signed E4M3, exponent field7..14, mantissa0..7 → 절댓값1..240|
| BF16 o_weight_scale |F32 uniform[0,0.001)을 BF16로 반올림|
| BF16 post-attention RMS weight |uniform[0,1)을 BF16로 반올림|
| BF16 residual |uniform[0,1)을 BF16로 반올림|

`fixture_prng.py`와 Rust PRNG는 텐서 이름과 원소 index의 함수이며, 각 텐서 checksum을 fixture와 비교한 뒤 NPU에 업로드한다. `scripts/generate_references.py::gen_sliding_attention_output` → `scaled_projection`은 **양자화하지 않은 BF16 x를 F32로 올려 FP8 weight의 정확한 F32 값과 matmul**, channel scale을 곱해 BF16로 변환한다. 이어 `reference/gemma4.py::RMSNorm.forward`는 F32 mean-square+epsilon1e-6, rsqrt, RMS weight를 적용하고 BF16로 변환한다. 마지막 residual 더하기도 BF16 출력이다. CPU reference를 새 FP8 activation 근사에 맞춰 바꾸면 안 된다.

[extra_seeds.py](../../../extra_seeds.py)의 `extended_harness`/`extended_generator`는 RUNS를3→6, 시작 run을0→3으로만 바꾼다. PRNG·분포·shape·tolerance는 바꾸지 않고, [references.json](../../artifacts/extra-seeds-run3-8/references.json)은 run3..8의228개 tensor schema를 기존 run0과 대조한다. 따라서 extra의30 output comparisons는 Q/K/V18개+O6개+FFN6개다. **O 자체의 추가 검사는6개의4096-input 구성과6×3840 최종 출력**이다. 공식3개와 합하면9개의 O 입력 구성이지,30개의 O 입력 구성이 아니다.

## 실제 run0..8의 quantizer 경계 coverage

[coverage-audit.json](coverage-audit.json)은 기존 `fixture_prng.py`로 O x만 CPU에서 다시 생성한 읽기 전용 분석이다. run0..8 모두 각 원본 fixture의 x checksum과 일치함을 확인했다. CPU PyTorch E4M3 cast로 `s*(RN(z)+RN(16*(z-RN(z)))/16)`를 계산했으며 NPU·kernel compile·model reference 재생성은 하지 않았다. 이 CPU cast의 결과가 SDK hardware rounding/FTZ까지 증명하지는 않는다.

| run |16 columns 중 scale1/256 /1/128|8→16분할로 scale이 바뀐 column|작은 성분 영역 수|CPU 재구성 max abs error|
|---:|---:|---:|---:|---:|
|0|10 /6|4|0|0|
|1|7 /9|1|0|0|
|2|13 /3|3|0|0|
|3|12 /4|2|1|1.1920929e-7|
|4|7 /9|3|1|3.5762787e-7|
|5|7 /9|5|0|0|
|6|10 /6|2|1|4.7683716e-7|
|7|7 /9|7|1|4.7683716e-7|
|8|12 /4|4|1|4.7683716e-7|

작은 성분 영역은0이 아닌 `|z|<2^-6`, z=x/s다. 이 표에서 작은 성분마다 CPU 재구성이 정확하지 않은 원소도 하나씩이다. 공식3개는4096원소 모두 exact reconstruction이라 **activation 양자화 손실이 없는 구성에서의 dot/scale/reduction/layout 동작**을 주로 검증했다. extra는 작은 손실을 다섯 번 더 검사하고, 모든 run에서 양쪽 scale과1~7개의 실제 scale 변경이 있어 새로운256-group 분할 자체도 검사한다.

그럼에도 모든 group의 scale이1/256 또는1/128 두 값에만 머문다. all-zero column은 없고, 매우 넓은 dynamic range·극소/극대 input scale·고의로 설계한 cancellation은 없다. 추가 seed를 계속 늘리는 것만으로 이 빈틈이 효율적으로 채워지지는 않는다.

## 보존된 경계와 바뀐 경계

`prepare_output_fp8_split`의 floor2^-119와 exponent mask는 그대로다. finite BF16 입력에서 s∈[2^-126,2^120], |z|<256, RN E4M3 가정의 |16*(z-hi)|≤128이어서 두 FP8 cast의 saturation을 피하는 설계다. BF16의8-bit significand와 power-of-two scaling 때문에 |z|≥2^-6인 범위는 두 FP8 항으로 정확히 표현할 수 있다. 그보다 작은 영역의 scalar absolute reconstruction error 상계는 `s*2^-14`다. 매우 작은 값은0으로 사라질 수 있다. column내 amax가 커질수록 같은 작은 값이 이 영역으로 들어갈 수 있다.

512→256 group은 부분집합이라 새 scale이 기존보다 커지지는 않지만, **F32 dot의 길이512→256, F32 column 합산8→16**으로 grouping이 바뀐다. cancellation이 있으면 작은 scalar 오차도 output relative error를 제한하지 못한다. 현재 최종 BF16 cast와 channel-scale→RMS→residual 순서는 그대로지만, 기존 reference와의 중간 BF16 rounding 차이까지 없어진 것은 아니다. post-RMS는 mean-square가epsilon1e-6 부근인 경우 작은 projection 오차를 증폭할 수 있다.

finite 입력 전체에서 output finite를 보장하지는 않는다. scale 이후 dot/channel scaling이나 reference의 F32 accumulation·RMS square가 overflow하는 입력을 무작정 넣으면 candidate만의 오류인지 판단하기 어렵다. stress fixture를 추가할 때 CPU reference가 finite인지 먼저 확인해야 한다.

## 다음 검사 제안: 추가 seed 이후 작은 별도 suite

1. **256-column 경계·서로 다른 scale·row 배치.** 각 column에 다른 power-of-two 크기의 anchor를 두고 q255/256,511/512,4095에 값이 걸리도록 한다. weight의 부호/크기도 column마다 구별하고 h119/120,1919/1920,3839까지 비교한다. 이는16-way 합산에서 잘못된 column scale을 읽거나 rowgroup 경계를 섞는 오류를 드러낸다. 기존 coordinate 전수검사는 source layout 증거이며 device 값 검사를 대체하지 않는다.
2. **zero와 작은 성분의 rounding 경계.** all-zero x, 일부 column만zero, ±0, amax가2^e의 바로 아래/같음/바로 위인 구성을 검사한다. 같은 column에 anchor와 `|x|≈s*2^-6`, `|x|≈s*2^-14`의 BF16 이웃 값을 함께 둬 정확 분해→작은 손실→0으로 반올림하는 경계를 검사한다. amax<2^-119인 입력도 넣어 floor 경로를 검증한다. 알려진 일반 scale부터 시작하고 극대 scale은 CPU reference와 intermediates의 finite 범위를 보장할 수 있을 때 별도로 다룬다.
3. **column 간 cancellation과 RMS 민감 영역.** 다른256-column들이 큰 양/음 contribution을 만들지만 합은 작도록 하고, channel scale 뒤 mean-square가epsilon1e-6 부근인 구성을 포함한다. 진단 시 post-RMS weight1/residual0은 입력 범위를 바꾸는 별도 stress case로 명시해 masking을 줄인다. 공식 fixture나 tolerance를 바꾸지는 않는다. BF16 최종 출력뿐 아니라 가능한 별도 diagnostic harness에서는 F32 partial과 pre-RMS 결과도 reference와 비교해 quantizer·sum·RMS 원인을 구분한다.

우선순위는 계획된 extra run3..8 완료 → 위1·2를 묶은 소수 deterministic cases → 위3이다. 반복 성능 측정에 stress fixture를 섞거나 공식 score로 보고하지 않는다. 이 문서는 제안만 작성했으며 kernel/harness 변경, 추가 compile, NPU 실행은 수행하지 않았다.
