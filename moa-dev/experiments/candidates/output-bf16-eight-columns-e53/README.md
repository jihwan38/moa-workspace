E53 BF16 O를 **16×256 columns에서 8×512 columns로 바꾼 단일 후보**다. 첫 exact compile은 성공했지만 전체 정적 시간이 **27,239→27,752 cycles(+513, +1.88%)**로 늘어 동결했다. NPU와 full host build는 실행하지 않았으며 채택 후보로 권하지 않는다. 과거 BF16 O8의 다른 RMS/후단 성능을 이 후보의 실측으로 사용하지 않았다.

기준은 clean E53 commit `989461e3bf886141c158edef0420537fd03f16ac`이다. [projection.rs](projection.rs)에서 O partition·이에 종속된 row/time 축만 바꿨다. QKV prefix, F32 channel-scale helper, RMS·residual, FFN, public ops와 harness는 그대로다. 나머지42개 source 및 Cargo/toolchain 입력을 비교했고, 46개 [compile 입력 해시](compiled-source-manifest.json)는 실제 격리 Docker scratch와도 일치한다. Git에 없는 Cargo.lock은 E53의 동결 clean snapshot에서 가져왔다.

## 좌표와 수치 경계

각 weight 좌표는 `h=1920*c+60*g+r`, `q=512*j+k`이고 `c<2, g<32, r<60, j<8, k<512`다. Physical slice는 `8*g+j`이므로 512개 slice가 모두 실제 데이터로 계산한다. Slice당 weight는 `60*512=30,720B`로 O16과 같고, 원래 BF16 입력512개를 1,024B TRF에 그대로 넣는다. FP8 weight는 지원되는 Main BF16 lookup으로 확장한다.

512-column BF16 dot의 F32 partial을 8개 column slice에서 합산한다. 결과의 유효 slice는 `[H/60%32,1#8]`이다. `H/2%30` time과 `H%2#8` packet에서 F32 두 값=8B만 commit하므로 60행 전체를 padding 없이 저장한다. 실제 row 순서를 유지한 채 `Dummy2→H/1920`으로 이름을 복원하고 typed DMA로 H480×8 epilogue에 모은다. Channel scale까지 F32이며 첫 BF16 cast는 scale 뒤다. E53 RMS와 residual 호출도 유지된다.

[verify_candidate.py](verify_candidate.py)의 [결과](verification.json)는 H3840·Qs4096 각각의 역변환을 모두 검사한다. 독립 row/column mapping의 Cartesian product가 전체15,728,640개 weight 좌표를 정확히 덮으며 표본 검사가 아니다. Graph에서도 `GReduceAddF ratio=8`, Float32 transpose와 8B commit을 확인했다. 단, F32 덧셈은 `16×256`에서 `8×512`로 묶음이 달라진다. 원래 BF16 값을 보존해도 누적 순서의 bit-identical 결과나 NPU 정확성을 주장하지 않는다.

## 실제 정적 비용과 DMA descriptor

| 항목 | E53 O16 | 후보 O8 | 차이 |
|---|---:|---:|---:|
| 전체 makespan | 27,239 | 27,752 | +513 |
| DMA 구간 합집합 | 18,229 | 18,742 | +513 |
| Main/Vector 각각 | 7,178 | 7,178 | 0 |
| Sub | 3,036 | 3,100 | +64 |
| BF16 입력 DMA | 933 | 1,318 | +385 |
| FP8 weight DMA | 13,383 | 13,512 | +129 |
| BF16 lookup+dot+column sum | 5,063 | 5,063 | 0 |
| F32 result gather | 470 | 469 | −1 |

Column reduction ratio는16→8, local output time은120→60으로 줄지만 local dot 길이는256→512가 된다. Slice당 dot 작업량과 weight stream 크기는 같다. 이들이 한 Main pipeline으로 합쳐져 있으므로 `5,063` 안에서 열 reduction만 별도 절감됐다고 주장하지 않는다. Sub의 입력 load327→391 증가는 critical path 앞부분에서 가려졌고, 전체 makespan 증가513은 이 schedule에서 DMA 증가와 같다.

Input DMA graph의 O1은 E53에서 다음과 같다.

```text
src0: [None=2:256, broadcast=2:0, Qs_256=16:512, broadcast=2:0, None=1:256]
```

후보는 `[None=4:256, broadcast=4:0, Qs_512=8:1024, broadcast=2:0, None=1:256]`이다. 8KB 고유 입력은 같지만 whole-vector row 복제 수가32→64, 논리적 DM 목적지 payload가256KiB→512KiB로 늘어난다. Descriptor는 이를 8개 PE에 각각 배치하며 source stride0으로 복제를 표현한다. 실제 HBM transaction 수를 측정한 결과는 아니다.

Weight DMA O4는 두 버전 모두 마지막 chunk가 **256B**다. O16에는 `H_1=120:4096, H_120=2:491520, Qs_256=16:256` 등이 있고, O8에는 `H_1=60:4096, Qs_256=2:256, H_60=4:245760, Qs_512=8:512` 등이 있다. 따라서 논리적512-column stripe를 512B hardware burst라고 해석하지 않는다. 전체 weight15MiB와 256B chunk 수는 같고 loop 분해가 바뀐다. 이 변화와 정적129 cycles 증가를 기록하며 bank/transaction 원인을 실측 없이 단정하지 않는다.

Result gather는120값×4B=480B씩4묶음에서60값×4B=240B씩8묶음으로 바뀐다. 목적지는 동일한 H480×8 F32 layout이며 이 copy의 정적 차이는−1뿐이다. 정확한 node lifetime과 `src0/dst0`은 [static-comparison.json](static-comparison.json)에 있다.

## 재현과 동결

기준 schedule은 [E53 BF16 exact artifact](../../artifacts/output-bf16-activation-e47/candidate.json)의27,239다. 해당 O image `35947381…`가 E53 clean host에 정확히 한 번 포함되고, cleanup 전후 full host도 byte-identical이다. O와 무관한45개 compile 입력도 같으므로 기준을 다시 compile하지 않았다.

`moa2026-dev:/tmp/moa-output-bf16-eight-columns-e53`에서 SDK0.8.1·고정 toolchain/Cargo.lock, `CARGO_HOME=/home/moa/.cargo`, offline mode로 아래를 한 번 실행했다. Target과 kernel output도 scratch 안이다.

```bash
cargo furiosa-opt compile ops::sliding_attention_output --exact \
  --dump-schedule schedules/candidate.json --dump-graph schedules/candidate-graph.json
```

[패치](candidate.patch), [evidence.json](evidence.json)과 [compile 로그](../../artifacts/output-bf16-eight-columns-e53/attempt-1.log)를 보존했다. Source SHA는 `17d4e1fa8af2c22fe1651a2f1a860cba0788afc7e63aec5b80d73c5cdc46cae0`, exact O image는71,680B / `8c649c87e70bf37e92aa778f93565151226d8755a20287e20763541311b85916`이다. Live source/target·Git·NPU는 변경하지 않았다. 정적 회귀를 확인한 뒤 추가 variant나 두 번째 compile은 하지 않았다.
