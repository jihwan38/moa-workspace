# E53 O projection 수치 경계 검토

2026-09-21 독립 source/graph 검토. [projection.rs](projection.rs) SHA256 `eb77ad0076e601606bf614ac990be3cba211378f66fda4e0fad83c2834724e3f`가 [E53 snapshot](../../artifacts/e53-bf16-output/src/device/sliding/projection.rs)과 byte-identical임을 확인했다. 기존 `verify_candidate.py`를 재실행해 저장된 결과와 일치했고 compile/NPU는 실행하지 않았다.

**E53는 확인된 FP8 activation 소실 원인을 제거하고, projection의 channel scale과 첫 BF16 반올림 순서를 원래 CPU reference에 맞춘다.** 채택 판단에는 이 source의 공식 입력 검증과 동일 boundary payload의 장치 결과가 필요하다. source 검토를 모든 finite 입력의 bitwise 동치 증명으로 확대하지 않는다.

## 실제 실행 경로

[동결 ops::sliding_attention_output](../../artifacts/e53-bf16-output/src/ops.rs)의 호출 순서는 `project_output` → `normalize_output_fused` → `add_output_residual` → HBM write다. ops와 shared RMS source는 E47 snapshot과 byte-identical이다. QKV prefix도 그대로이므로 이 변경은 QKV의 FP8 입력 근사를 수정하지 않는다.

| 단계 | E53 구현 | 원래 Python reference와의 관계 |
|---|---|---|
| activation / weight | x는 BF16 그대로 TRF. weight는 저장된 FP8을 `fetch_table_lookup::<bf16>()`로 확장 | finite E4M3의 254개 code는 BF16에 정확히 표현된다. activation 양자화나 runtime input scale이 없다. |
| dot / column sum | 각 256-column의 BF16 dot → F32 output → 16-column F32 sum | oracle의 F32 matmul과 같은 입력값·실수 연산식. reduction 순서는 다를 수 있다. |
| projection 보존 | F32를 two-row transpose/commit하고 H480×8로 typed DMA | BF16 cast 없이 F32 합계를 보존한다. graph는 `ratio=16`, `Float32`, `commit_in_size=8B (=2EA)`, `0+[60:2,2:1]`를 명시한다. |
| channel scale | `apply_output_channel_scale_f32`에서 F32 sum × widened BF16 channel scale → BF16 | `scaled_projection`: `((x.float() @ codes.T) * scale.float()).to(bf16)`와 같은 반올림 경계다. |
| post-RMS | BF16 projection → F32 square/mean + EPS → sqrt/div → gamma → BF16 | oracle도 BF16 projection을 F32로 확장하고 EPS=1e-6을 쓴다. mean의 합산 및 rsqrt 대 sqrt/div 차이는 남는다. |
| residual | normalized BF16과 residual BF16을 F32로 더하고 BF16 cast | oracle의 BF16 residual add와 같은 입력 및 최종 dtype 경계다. residual을 RMS 전이나 normalized BF16 cast 전에 섞지 않는다. |

E47는 raw projection을 BF16으로 반올림한 뒤 channel scale을 곱했다. E53는 그 중간 BF16 경계도 제거했다. 옛 `apply_output_channel_scale` 및 `prepare_output_fp8_split` 함수는 파일에 남지만 E53의 `project_output`에서 호출하지 않는다.

row 좌표는 `h=1920*c+120*g+r`, column은 `q=256*k+t`다. E47의 2 clusters×16 rowgroups×16 columns를 유지한다. 120개 F32 row는 60번의 2-value/8-byte commit으로 저장한다. high-H cluster 축만 rename한 뒤 typed DMA로 H480×8을 복원하며 padding을 유효 row로 해석하지 않는다.

## 남는 수치 가정

- **F32 reduction:** BF16 x와 정확히 확장한 FP8 weight의 곱은 유효 significand 기준 최대 12 bits이므로 exponent가 정상 범위인 개별 곱에는 F32 정밀도가 충분하다. 그러나 256-element local dot과 16-way sum의 F32 덧셈 순서는 PyTorch BLAS와 같다고 보장되지 않는다. cancellation이 심하면 합산 오차의 상대 크기가 커질 수 있다. graph의 F32 dtype 확인만으로 모든 내부 하드웨어 동작을 증명한 것은 아니다.
- **RMS:** source는 각 H480의 제곱 합을 3840으로 나누고 8개 partial mean을 합친다. oracle은 전체 H의 mean과 `rsqrt`를 사용한다. 같은 실수 공식이어도 F32 roundoff, sqrt/div/rsqrt 구현, BF16 tie 근처에서 결과가 달라질 수 있다. 이 경로는 E47에서 유지한 부분이다.
- **overflow/subnormal:** finite BF16 입력만으로 dot, F32 channel product, BF16 projected output, F32 mean-square가 모두 finite라고 보장할 수 없다. 다른 reduction 순서가 overflow 발생 여부에 영향을 줄 수도 있다. tiny 값의 FTZ/subnormal 처리와 LUT의 모든 code에 대한 실제 장치 변환은 이 source 검토만으로 확정하지 않는다.
- **범위:** 실제 boundary weights는 공식 exponent band인 ±1..240을 사용한다. finite E4M3 전체의 수학적 BF16 표현 가능성 검사는 ±0, subnormal, ±256..448까지 포함하지만, 그 모든 code의 장치 LUT 동작을 probe가 검사한 것은 아니다.

## boundary probe가 제공하는 근거

[E47 job70785](../../../../docs/experiments/e47-output-boundary-probe.json)는 8개 중 case7에서만 3840개 전부 got=0.5, want=0.609375였다. [원래 BF16 control job70799](../../../../docs/experiments/original-output-boundary-control.json)는 **같은 payload**, 40개 input checksum PASS, 8개 case 모두 tolerance PASS였고 case7은 3840개 exact/error0였다. 원본 control 로그 SHA도 검사했다. 따라서 이 반례가 원래 RNGD 경로에서도 실패하는 부적절한 oracle이라는 설명은 현재 관측과 맞지 않는다. 원본 control의 성공을 E53의 성공으로 대신 기록해서는 안 된다.

E53에서 각 case의 의미는 다음과 같다.

- case0은 zero/residual 경로, case1은 255/256 등의 input-column 경계와 119/120·1919/1920 등의 output-row 경계를 검사한다. 특히 F32 two-row commit과 typed gather를 바꾼 후보의 주소 검증에 유용하다.
- case2..5는 다양한 BF16 값, power-of-two 이웃과 작은 항을 검사한다. E53 O에는 amax/FP8 quantizer가 없으므로 이 후보에서 “quantizer의 분기 coverage”라고 부르지는 않는다.
- case6은 tiny/subnormal 입력의 실행과 유한 출력 확인이다. 출력이 절대 허용치 .05보다 훨씬 작아 zero 출력도 통과할 수 있다. 원본 control도 max difference 약 1.204e-35를 허용하므로 정확한 tiny 보존이나 FTZ 부재의 증거가 아니다.
- case7은 E47의 실제 소실 원인을 직접 겨냥한다. E53는 x의 tiny BF16 항을 버리지 않으므로 원래 raw dot 0.2325439453125 → scaled BF16 0.0002269744873046875 → final 0.609375 경로를 유지할 구조다. 실제 후보 장치 실행이 이 예상과 맞는지는 별도로 확인해야 한다.

검토 시점 [E53 공식 측정 기록](../../../../docs/experiments/e53-bf16-output.json)은 job70806 pending, 완료 run 0개였다. 공식/추가 입력과 동일 boundary package가 통과하면 확인된 FP8 정확도 결함을 제거한 O 대안으로 채택할 근거가 된다. 모든 입력의 동치나 공식 순위를 이 검토에서 주장하지 않는다.

후속 장치 검증에서는 공식·추가 입력9개 구성, 동일 O 경계8개 및 [같은 패키지3회 반복](../../../../docs/experiments/e53-bf16-output-repeats.json)이 모두 통과해 E53을 채택했다. 위 검토 시점과 후속 측정 시점을 구분한다.
