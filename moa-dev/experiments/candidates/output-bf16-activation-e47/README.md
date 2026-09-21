E47 O의 **activation FP8 hi/lo 근사를 제거하고 원래 BF16 입력을 그대로 계산하는 정확성 수정 후보**다. 공식 3+6 seeded 입력 통과와 별개로, [E47 경계 probe 70785](../../../../docs/experiments/e47-output-boundary-probe.json)의 case7에서 3,840개 출력이 모두 0.5로 나와 expected 0.609375와 달랐다. 이 후보는 작은 입력값 소실을 제거하고, channel scale 뒤에 처음 BF16으로 반올림하도록 공식 oracle 경계도 맞춘다.

**두 번째 exact compile PASS: 27,239 cycles**, E47 native FP8의 24,621보다 2,618(+10.63%) 높다. Root의 [E53 job70806](../../../../docs/experiments/e53-bf16-output.json)은 **공식 3입력의 15개 비교를 모두 통과**했다. O 중앙값은 **50,882 cycles**다(Q93,987 / FFN376,814). [추가 6입력 job70814](../../../../docs/experiments/e53-bf16-output-extra-correctness.json)의 30개 비교와 [경계 probe job70810](../../../../docs/experiments/e53-output-boundary-probe.json)의 8개 case도 모두 통과했다(입력 checksum40개 일치, 출력30,720개 실패0). [같은 패키지3회 반복](../../../../docs/experiments/e53-bf16-output-repeats.json)도 모두 통과했고 중앙값은 Q95,621/O50,882/FFN377,880이다. 모든 입력의 동등성을 주장하지 않는다.

## 실제 구현과 범위

[projection.rs](projection.rs)는 동결 E47에서 O `project_output`을 바꾸고 private `apply_output_channel_scale_f32`만 추가했다. QKV prefix와 기존 channel-scale helper부터 파일 끝까지는 byte-identical이다. 나머지 42개 source, public ops·harness·RMS·residual·RoPE·FFN과 Cargo/toolchain/lock도 그대로다. 사용하지 않게 된 기존 FP8 quantizer/helper는 소스에 남아 있으나 O가 호출하지 않는다.

실제 경로는 다음과 같다.

```text
BF16 x[4096] → 16개의 256-column shard → BF16 TRF
FP8 weight HBM → 같은 E47 row/column DM → Main FP8→BF16 lookup
→ 단일 BF16 dot의 FP32 부분합 → 16-column FP32 sum
→ F32 2행/8B commit → F32 H480×8 typed gather
→ F32 channel scale → 첫 BF16 cast
→ 기존 RMS/gamma → BF16 → residual add → BF16 HBM write
```

SDK 0.8.1 `src/engine/fetch_adapter.rs:185–215`의 `fetch_table_lookup::<bf16>()`를 Main weight stream에 사용한다. Input은 Sub의 기존 BF16 `to_trf` 경로다. 지원되지 않는 F8→BF16 direct fetch cast, Sub table lookup, runtime barrier를 쓰지 않는다. 과거 [BF16 column-split O](../output-two-cluster-column-split/projection.rs)와 같은 lookup+dot API이며, weight 전체를 BF16 DM으로 materialize하지 않는다.

기존 16-column mapping은 유지된다. `h=1920*cluster+120*row_group+local_row`, `q=256*column_group+local_column`이다. 2클러스터 × 16행 그룹 × 16열 그룹의 512개 slice가 모두 dot에 참여한다. Column sum 후 `[H/120%16,1#16]`에서 각 그룹의 유효 slice 하나에 120개 F32를 저장한다. H/1920 축 이름만 복원하고 typed DMA가 canonical H480×8로 모으므로 padding을 값으로 바꾸지 않는다.

공식 [`scaled_projection`](../../../../furiosa-opt-gemma4-12B/scripts/generate_references.py)는 `((x.float() @ codes.T) * scale.float()).to(bfloat16)`다. 과거 device의 raw-dot BF16 cast를 유지하면 이 oracle보다 이른 반올림이 남는다. 그래서 이 후보는 dot·column sum·DM·gather 전체를 F32로 유지하고 channel scale 후에만 BF16으로 변환한다. E47과 bit-identical한 산술을 목표로 하는 변경이 아니다.

## 컴파일 제약과 정적 비용

첫 [4행 F32 시도](projection.f32-four-row-failed.rs)는 transpose의 `max_in_rows=2` 제약으로 실패했다. 기존 [compact partial 조사](../output-compact-f32-partials/README.md)와 같은 제약이다. 두 번째 시도는 `Time=H/2%60`, `Packet=H%2#8`, `commit_trim(H%2)`로 유효 F32 두 값, 정확히 8B를 저장해 통과했다. 이후는 F32 typed gather이므로 과거 실패한 “2행 packet에서 BF16 4행으로 재transpose”를 하지 않는다. BF16 2값을 padding으로 저장했던 초기 FFN 후보와도 다른 경로다.

| 항목 | E47 native | BF16 후보 |
|---|---:|---:|
| 전체 makespan | 24,621 | 27,239 |
| instruction 수 | 51 | 32 |
| DMA 구간 합집합 | 18,621 | 18,229 |
| Main 구간 합집합 | 5,196 | 7,178 |
| Sub 구간 합집합 | 2,572 | 3,036 |
| weight DMA | 13,383 | 13,383 |
| dot + column sum | 1,242 + 522 | 5,063 |
| 결과 gather | 446 (BF16) | 470 (F32) |

Main은 1,982 cycles 증가하지만 quantizer input copy 691과 scale copy 563이 사라진다. 새 lookup 상수 DMA 838과 F32 gather 추가 24를 포함하면 전체 DMA는 392 줄어든다. BF16 변환·dot·16-column sum은 한 Main pipeline의 5,063 cycles에 포함된다. 이 수치들을 합산해 전체 시간이나 실제 NPU 비용을 예측하지 않는다. 과거 측정과 비용 해석은 [연구 노트](../../../../docs/research/output-bf16-fallback-e47.md)에 분리했다.

## 검증과 동결

[verify_candidate.py](verify_candidate.py)와 [결과](verification.json)는 QKV/후단 source 보존, H3840/Qs4096 좌표의 Cartesian coverage, F32 2행 commit 주소, channel-scale-before-BF16 경계를 검사한다. 유한 E4M3 코드 254개가 BF16에 정확히 표현됨도 scalar 계산으로 확인한다. 이는 SDK table/NPU 자체를 전수 실행한 결과가 아니다. Graph의 `GReduceAddF ratio=16`, Float32 transpose, `commit_in_size:8B`와 120값 저장도 확인했다.

BF16 DPE의 FP32 누적과 16개 partial 합산은 CPU matmul과 순서가 다르다. 기존 RMS는 sqrt/div를 사용하지만 CPU reference는 rsqrt를 사용한다. 극단적인 overflow/underflow, FTZ와 모든 cancellation 입력의 동등성을 주장하지 않는다. 공식3입력·extra6입력·8-case 경계 probe는 실제 통과했으나 이 범위를 넘어서는 보장은 아니다.

소스 SHA-256은 `eb77ad0076e601606bf614ac990be3cba211378f66fda4e0fad83c2834724e3f`, 정확 O image는 71,168B / `359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b`다. Root E53 host `d697c8aa09bdd4592b11cb5df142f7520500a1b3d5747f7dbb6dcdc6e29045db`에 이 image가 정확히 한 번 있고 E47 QKV/FFN image도 각각 한 번 있음을 직접 확인했다. [evidence.json](evidence.json)에 두 compile 로그, schedule/graph/image, 46개 [compile 입력 해시](compiled-source-manifest.json)를 연결했다.

격리 scratch는 `/tmp/moa-output-bf16-activation-e47`이며 target과 kernel output도 그 내부다. `CARGO_HOME=/home/moa/.cargo`, `CARGO_NET_OFFLINE=true`로 다음 명령을 두 번 실행했다.

```bash
cargo furiosa-opt compile ops::sliding_attention_output --exact \
  --dump-schedule schedules/candidate.json --dump-graph schedules/candidate-graph.json
```

독립 agent는 live source/target, Git, NPU를 변경하지 않았다. Root가 별도로 [E53](../../../../docs/experiments/e53-bf16-output.json) full build와 제출을 관리한다.

Root는 이후 사용하지 않는 기존 O helper 두 개와 낡은 주석만 정리했다. [clean source build 감사](../../../../docs/experiments/e53-source-build-audit.json)에서 전체 host가 측정한 `d697c8aa…`와 byte-identical임을 확인했다. 이 폴더의 동결 `eb77ad…` source·patch·실패 시도는 그대로 보존한다.
