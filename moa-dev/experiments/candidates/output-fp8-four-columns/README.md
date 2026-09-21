# O projection: 4 columns × 1024

E37 O의8×512 column 분할을4×1024로 바꾼 독립 후보다. **Exact compile PASS, 전체 정적 시간24,997→25,873cycles(+3.50%)로 회귀**했다. Weight DMA 감소77보다 pair dot 증가952가 크다. NPU 검증은 하지 않았으며 추가 변형·compile 없이 동결했다.

[projection.rs](projection.rs)의 SHA256은 `3a996dd4171fde0ba0ac2d97d5576d7d1bd7a9ca03ea84cd811dc283c899ed77`다. [원본](projection.original.rs), [patch](candidate.patch), [evidence](evidence.json)에 비교를 보관한다. 기준 exact image `e9d3402c62fd866f417053e618e7a7af1d5cb60485c04cc292f3a702f5996cde`는 E37의 O와 byte-identical하다. 후보 image는 `a6421b94ac74a466dd6cb14ad476a9a4b14013367364a4867641a619be0f06fb`다.

## 변경과 호출 경로

E37 [ops.rs](../../artifacts/e37-kernel-specific-rms/src/ops.rs)의 `sliding_attention_output`은 canonical `Qs4096` HBM view → `project_output` → H480×8 RMS → residual → HBM write 순서다. 이번 변경은 `project_output`과 `prepare_output_fp8_split`, private `OutputColumns` alias에 한정한다. QKV projection 전체 prefix, QKV quantizer 전체 suffix, O channel-scale/residual helper, `ops.rs`는 byte-identical하다.

`OutputColumns`를 `[H/60%32,Qs/1024 #8]`로 둬 cluster마다128개만 유효하고 physical slice256개는 유지한다. Weight local shape는 `[H%60,Qs%1024]`, input packed shape는 `[Dummy2,Qs%1024]`다. 각 input column의1024개 값에서 amax와 기존 power-of-two scale을 계산하고 hi/lo FP8을 만든다. TRF의64-element dot 시간축은8→16, BF16 input quantizer의16/8/4-element 루프는 각각64/128/256으로 늘린다. OutputRows `[H/60%32,1#8]`,60행 결과의4값 BF16 transpose/commit, 마지막 H480×8 epilogue는 그대로다.

HBM weight `(h,q)`는 `c=h//1920`, `g=(h%1920)//60`, `r=h%60`, `j=q//1024`, `t=q%1024`를 통해 `(cluster=c,slice=8g+j,element=1024r+t)`로 옮겨진다. 유효 `j`는0–3이고4–7은 padding이다. [verify_layout.py](verify_layout.py)는 H3840과 Qs4096의 독립 좌표 인자를 전부 역변환하고, Cartesian product가15,728,640개 weight를 중복 없이 덮는지와 최종3840개 H 출력 좌표를 검사한다. 공식 fixture를 사용하지 않는다.

## 실제 DMA와 실행 비용

원본과 후보 [graph](../../artifacts/output-fp8-four-columns/candidate-graph.json)의 weight DMA operator18은 모두8개 src descriptor를 사용한다. **끝 packet은 두 경우 모두256B**다. 원본의 `Qs_256=2:256, Qs_512=8:512`가 후보에서 `Qs_256=4:256, Qs_1024=4:1024`로 바뀐다. HBM row stride4096B는 같고 SRAM row stride만512→1024B다. 두 경우 모두 논리256B segment는61,440개이며 실제 HBM transaction 수나 bank 충돌 수를 뜻하지 않는다. 논리1024-element chunk를 hardware1KB burst라고 부를 근거는 없다.

| 측정 | E37 8×512 | 후보4×1024 | 차이 |
| --- | ---: | ---: | ---: |
| Makespan | 24,997 | 25,873 | +876 |
| Instruction 수 | 52 | 51 | -1 |
| Weight HBM DMA | 13,512 | 13,435 | -77 |
| Input HBM DMA | 1,318 | 1,313 | -5 |
| Quantizer BF16 input copy | 947 | 947 | 0 |
| Quantizer scalar scale copy | 563 | 499 | -64 |
| 전체 DMA interval union | 19,414 | 19,268 | -146 |
| Amax / hi / lo Main 각1개 | 각각409 | 각각537 | 합+384 |
| Pair dot + scale Main | 1,238 | 2,190 | +952 |
| Column sum Main | 390 | 390 | 0 |
| 전체 Main/Vector interval union | 5,252 | 6,588 | +1,336 |
| Sub interval union | 2,668 | 2,860 | +192 |

원본 dot는 `[17346,18584)`, 후보는 `[17275,19465)`다. Weight는71cycles 일찍 도착하지만 유효 slice 수가 절반이고 각 dot의 K 길이가 두 배여서 후단이 늦어진다. 이 설명은 정적 schedule 비교이며 hardware 병목을 측정한 결과는 아니다.

## Padding과 수치 범위

Graph operator34의 column reduction은 입력 partition `[[Qs_1024=4]+4]=8`을 유지한다. API상 유효4개 column을 줄이지만, **물리 GReduce ratio는8로 남아** sum390cycles가 줄지 않았다. 소스는 typed `vector_inter_slice_reduce::<OutputRows>`만 사용한다. Padding을 valid로 reshape하거나 padding 메모리가0이라고 가정하는 코드는 추가하지 않았다.

SDK0.8.1 `engine/vector/tensor/vector_tensor.rs:614`는 이 API를 tensor reduction으로 정의하고, `engine/vector/op/semantics.rs:346`은 Add의 identity0을 지정한다. `storage/buf.rs:346`의 의미상 reduction은 live logical 좌표를 읽고 padding을 제외한다. 이것과 compiler graph의 padding 보존을 확인했으나, 실제 RNGD의 padding 처리 정확성을 검증한 것은 아니다. NPU 결과 없이4-column correctness를 주장하지 않는다.

이 변경은 수학적으로 같은 실수 dot을 목표로 하지만 BF16/FP8 결과의 동일성은 보장하지 않는다.1024개 단위 scale은 원본의 이웃512개 두 shard 중 더 큰 amax에 의존하므로 작은 값의 FP8 복원이 달라질 수 있다. Native FP8 hi/lo의 근사 한계는 그대로이며, K1024 dot와4개 column FP32 sum으로 reduction grouping도 바뀐다. Pair combine → input scale → FP32 DM partial → column sum → BF16 → channel scale → BF16 → RMS → BF16 → residual의 경계와 후단 formula는 유지했다. 기존 E37 PASS를 이 후보로 승계하지 않는다.

## 재현

Docker `moa2026-dev`의 독립 `/tmp/moa-output-fp8-four-columns`에 E37 source 전체를 복사했다. 최초 원본 O exact compile 뒤 `src/device/sliding/projection.rs`만 후보로 교체·touch했다. 기존 독립 scratch의 compiler cache를 재사용했고 main target은 사용하지 않았다. 원본1회와 후보1회, 총2회 모두 exit0이다. SDK/CLI0.8.1, Rust1.97.0-nightly다.

```bash
export CARGO_HOME=/home/moa/.cargo
export CARGO_TARGET_DIR=/tmp/moa-output-fp8-four-columns/target
export FURIOSA_OPT_OUT_DIR=/tmp/moa-output-fp8-four-columns/target/furiosa-opt/kernel
export CARGO_NET_OFFLINE=true
cd /tmp/moa-output-fp8-four-columns
# projection.rs = frozen E37
cargo furiosa-opt compile ops::sliding_attention_output --exact \
  --dump-schedule schedules/output-baseline.json \
  --dump-graph schedules/output-baseline-graph.json
# projection.rs = frozen candidate; touch after replacement
cargo furiosa-opt compile ops::sliding_attention_output --exact \
  --dump-schedule schedules/output-candidate.json \
  --dump-graph schedules/output-candidate-graph.json
```

원본/후보 [로그](../../artifacts/output-fp8-four-columns/candidate.log), schedule, graph, binary는 ignored artifacts에 두고 해시는 evidence에 기록했다. 새 live source/target 변경, Git mutation, fixture 생성, NPU 작업은 하지 않았다.
