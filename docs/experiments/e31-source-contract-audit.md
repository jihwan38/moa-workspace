# E31 소스 계약 독립 점검

2026-09-21 점검. 비교 기준은 원본 baseline commit `289017681fe71beaec39fdb08b7fa83278a9ab5b`, 대상은 동결된 `moa-dev/experiments/artifacts/e31-switch-gather-rms/`다. live prototype은 비교 대상에서 제외했다. 읽기 전용 `git show`로 원본을 추출하고 전체 source bytes, ops 함수 경계, 실제 호출 경로를 확인했다. 소스·Git index/commit·NPU는 변경하지 않았다.

**허용 수정 범위와 공개 entry point 계약을 위반한 변경은 발견하지 못했다.** `src/`43개 파일 중7개만 달라졌고 모두 `src/device/**` 또는 `src/ops.rs`다. source 추가/삭제는 없다. `ops.rs`의10개 `#[device(chip=1)]` 함수 선언과 본문 바깥 텍스트가 모두 byte-identical이다. 실제 본문 변경도 Stage1의 QKV/O/FFN 세 함수에 한정된다.

## 수정 범위와 함수 계약

대회 저장소 [README Stage1 규칙](../../furiosa-opt-gemma4-12B/README.md)의101–108행이 허용하는 범위와 비교했다. ops 각 함수의 여는/닫는 body brace를 찾고 body 내부만 제거한 파일 전체가 원본과 같은지 검사했다. 따라서 이름/인자 순서/타입/반환형뿐 아니라 imports, attributes, 함수 사이 텍스트도 그대로다.

| ops 함수 | E31 시작 행 | signature | body |
| --- | ---: | --- | --- |
| embed_token | 9 | 동일 | 동일 |
| sliding_project_qkv | 38 | 동일 | 수정 |
| full_project_qkv | 80 | 동일 | 동일 |
| sliding_attention | 120 | 동일 | 동일 |
| sliding_attention_output | 132 | 동일 | 수정 |
| full_attention_first_page | 150 | 동일 | 동일 |
| full_attention_page | 164 | 동일 | 동일 |
| full_attention_output | 178 | 동일 | 동일 |
| decoder_feedforward | 203 | 동일 | 수정 |
| final_norm_and_logits | 243 | 동일 | 동일 |

원본과 다른 source7개 및 E31 SHA256은 다음과 같다.

| 파일 | SHA256 |
| --- | --- |
| `src/device/shared/mlp.rs` | `8717b2b450e4eba1d6fa8ad22a184d1fd86f76852c892295ee1d6c75574b3450` |
| `src/device/shared/residual.rs` | `83d001788e3c1ae073ec8720435569fde062a47e52c181ab859a2d3f047accfc` |
| `src/device/shared/rmsnorm.rs` | `3c3c9525f21fe9916e0f6db00863cbb05e4f0c8cc35501ed08b1bbcdb4635cb0` |
| `src/device/sliding/projection.rs` | `b58d2ce261f1d12d2a219fd4edc2a715b163b4bf033d5e973ca3d64ba2df08e5` |
| `src/device/sliding/rmsnorm.rs` | `48bce532c0e9c0e39d6f17a6c38b78e34bfb5a00691291849e460b7a9044281f` |
| `src/device/sliding/rope.rs` | `cc23ad726baf191184b8b0ff000516cf38148e04c05b74925ec8a966e80b3c93` |
| `src/ops.rs` | `48cfec9c9fd390d54e1ef1e560f984d8968d7c75d3faf41a397663e6a3cd2a1a` |

`src/axes.rs`, `src/host/**`, `src/api/**`, `src/bin/**`, `src/lib.rs`, `src/ops_audio.rs`, `src/ops_vision.rs`는 원본과 같다. 캡처에 포함된 Cargo.toml/Cargo.lock/rust-toolchain.toml과 실행 스크립트도 원본과 같다. snapshot에 없는 README·일부 개발 스크립트 등은 비교 결과에 포함하지 않았다. signature가 다른 private device helper들은 허용된 내부 구현 변경이며 공개 `#[device]` 계약 변경으로 세지 않는다.

## QKV의 caller → 계산 → 캐시 기록

실제 모델 caller는 [host/runtime.rs](../../moa-dev/experiments/artifacts/e31-switch-gather-rms/src/host/runtime.rs):231의 `set_position_offsets`와:401의 `sliding`이다. 이 파일은 원본과 동일하다. `ps=pos%1024`, `offset_kv_s=ps*8*256*2`를 HBM i32 scalar에 저장하고, sliding QKV에 `ws.offset_kv_s`, `ws.offset_rope_s`, 해당 layer의 mutable K/V cache 및 `self.q_s`를 전달한다. 다음 attention 호출도 같은 K/V cache와 Q output을 읽는다.

[E31 ops.rs](../../moa-dev/experiments/artifacts/e31-switch-gather-rms/src/ops.rs):57–76의 흐름은 다음과 같다.

```text
HBM x → H480×8 input RMS → 전역 amax 기반 hi/lo FP8 입력
→ Q/K paired dot, V two-dot → 각 row scale/BF16
→ Q/K/V head별 RMS (V도 RMS 수행)
→ Q/K RoPE → canonical Q[Ns,Gs,Ds], K[Ns,Ds]
→ Q 전체 q_out에 기록
→ K를 kv_offset의 k_cache 슬롯에 scatter
→ V를 canonical [Ns,Ds]로 typed gather → 같은 kv_offset의 v_cache 슬롯에 scatter
```

원본과 E31의 최종 세 HBM 기록 호출은 같은 destination/인자를 사용한다. `q.view().to_hbm_view(...,q_out.view_mut())`, `k.dma_scatter::<m![1],_,_>(kv_offset,k_cache)`, `v.dma_scatter::<m![1],_,_>(kv_offset,v_cache)`다. E31이 scatter 전에 추가한 V `to_dm`은 SRAM 배치 변경이며 HBM cache 기록이 아니다. cache/offset 인자는 projection/RMS helper로 넘겨지지 않으므로 해당 helper가 별도 cache 위치를 선택하지 않는다.

SDK0.8.1 `tensor/memory.rs:899–932`의 `dma_scatter` 구현과 doc에서 index가 **바이트 offset**임을 확인했다. E31은 원본과 같은 canonical single-slice `[Ns,Ds]` 형상의 K/V를 넘긴다. 유효 slot과 `(n,d)`의 cache 주소는 `base + 4096*slot + 2*(256*n+d)`이고, K/V 각각2048개 BF16=4096 bytes를 기록한다. Q는4096개 BF16=8192 bytes다. offset은 입력 tensor 그대로 사용하며 fixture 위치를 상수로 대체하지 않는다. 유효 offset을 공급한다는 전제는 원본 host/API 계약과 같다.

`sliding::rope::apply_rope`는 [rope.rs](../../moa-dev/experiments/artifacts/e31-switch-gather-rms/src/device/sliding/rope.rs):20–21에서 원본과 같은 `rope_offset`으로 cos/sin을 읽는다. 원본과의 diff는 head 배치를 이미 갖춘 입력 signature와 중복 scatter 제거에 한정된다. 함수 끝:192–212의 typed Switch gather가 Q/K를 기존 canonical 순서로 복원한다.

새 projection의 Q4096개/KV2048개 논리 좌표를 모두 열거해 head reshape 전후 physical 위치가 같음을 별도로 확인했다. Q의 `q=512*n+256*g+d`는 `(cluster=q/2048,slice=(q/8)%256,local=q%8)`와 대응한다. KV의 `p=256*n+d`는 `(p/1024,8*((p/32)%32)+(p/8)%4,p%8)`와 대응한다. KV의4→8 Switch padding은 Slice에 padding으로 남고, 이후 typed head gather가 유효2048개만 옮긴다. scatter에 padding을 유효 cache 값으로 바꾸는 reshape는 없다.

## O 경로의 output 계약

모델 caller [host/runtime.rs](../../moa-dev/experiments/artifacts/e31-switch-gather-rms/src/host/runtime.rs):440은 원본과 같이 `attn_s`, O weight/scale, post-attention norm, mutable `self.x`를 전달한다. [ops.rs](../../moa-dev/experiments/artifacts/e31-switch-gather-rms/src/ops.rs):140–146은 attention 입력의 `[Ns,Gs,Ds]→[Qs]` wire order를 유지하고, projection → channel scale → RMS → residual add → 같은 residual HBM 전체 write를 수행한다. O에 cache/offset 인자는 없다.

[projection.rs](../../moa-dev/experiments/artifacts/e31-switch-gather-rms/src/device/sliding/projection.rs):283의 O는 Qs4096을8×512로 분할하고 각 부분의 입력 scale을 FP32 partial에 곱한 뒤8개 부분을 합산한다. `[Dummy2,H%1920,Qs]` weight view와 `H/1920` output cluster rename은 실제 high-row 좌표를 나타낸다. `h=1920*c+60*group+local`을 H480×8 배치로 typed DMA한 뒤 전체3840개 output을 처리한다. `add_output_residual`:372는 입력 residual을 HBM에서 읽어 같은 h에 더하고 BF16으로 반환한다. 마지막 typed HBM write는3840개 BF16=7680 bytes 전체를 원래 destination에 쓴다.

이 검토는 output shape/주소/scale 귀속 및 기록 범위가 유지됨을 확인한 것이다. FP8 입력 분해 및 RMS 합산 순서 때문에 원본과 모든 유한 입력에서 bit-identical하다는 뜻은 아니다. 일반 입력 범위와 cancellation 한계는 [O native FP8 독립 review](../../moa-dev/experiments/candidates/output-native-fp8-two-lane/review.md)에 이미 반례와 함께 적혀 있다.

## Harness·fixture와 실제 검증 범위

공식 [test_kernels.rs](../../moa-dev/experiments/artifacts/e31-switch-gather-rms/src/bin/test_kernels.rs)는 baseline과 byte-identical이며 SHA256은 `42db4e99ae788020726d530d1cd799e647779b7b9912e593401d6981cffc7516`이다. E31 frozen fixture SHA256은 `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`로 초기 baseline job69783의 내려받은 package와 같다. fixture는 Git baseline에 포함된 파일이 아니므로 이 비교는 baseline package evidence와 한 것이다.

수정된 device/ops 코드에는 fixture/expected/seed/run id 참조, fixture 파일 include, host 환경/파일 읽기, 입력별 `if`/`match` 예외 분기를 찾지 못했다. constants는 모델 차원, RMS divisor, GELU 상수, FP8 exponent/abs mask, 양자화 floor와16배 residual 분해 상수다. static pass 목록은 모든 고정 차원 tile을 처리하는 unroll이다. FP4/FP8 SDK table lookup은 입력 format 변환이며 expected output 표가 아니다. 코드 경로는 전달된 x/weight/scale/norm/residual/offset 데이터를 사용한다.

**기존 harness는 non-target cache 보존을 동적으로 검사하지 않는다.** :508–509에서 K/V 전체 cache를0으로 초기화한 후, :539–540에서 cache 전체를 CPU로 읽어도 `[slot*width..(slot+1)*width]`만 잘라 expected K/V와 비교한다. 다른1023개 슬롯의 값은 비교하지 않는다. O는:572에서 residual 전체3840개를 비교한다. 따라서 공식/extra PASS는 대상 슬롯의 K/V와 Q/O/FFN 출력 정확성 증거이며, 모든 cache 슬롯 보존의 동적 증거로 확장할 수 없다.

[E31 공식 job70441](e31-switch-gather-rms.json)은3개 입력×5개 출력=15 comparisons PASS, [추가 job70448](e31-rms-extra-correctness.json)은 run ids3–8의6개 입력×5개 출력=30 comparisons PASS다. 추가 harness는 별도 개발 artifact이며 frozen 공식 harness를 바꾸지 않았다. 이번 source 검토에서 비대상 cache write는 발견하지 못했지만, 보존을 실측으로 확인하려면 별도 개발 harness에서 cache를 서로 다른 nonzero sentinel로 채우고 대상 슬롯 이외 전체를 호출 전후 비교해야 한다. 이 source 감사 자체는 그 검사를 실행하지 않았으며, 이후 별도 harness 실측은 아래 후속 기록으로 구분한다.

확인한 작은 문서 불일치는 shared RMS helper 위의 `O-only experiment` 주석이다. E31에서는 Q 입력 RMS도 이 helper를 호출한다. 코드 계약 오류는 아니며, 이후 정리 시 주석 범위를 갱신하면 된다. Stage1 PASS를 shared MLP가 쓰이는 vision/audio/full 모델 전체 실행 검증으로 확장하지 않는다.

후속 별도검증: 원본harness의위한계를보강한 sentinel host harness가 Arena70507에서통과했다.
E31 devicekernel은byte-identical하고, 기존15출력비교에더해 비대상cache12,570,624개가
모두exact일치했다. [기록](e31-cache-preservation-correctness.json). 이검사는E31과해당3slot의검증이며
원래공식harness나후속QKV의검사범위를바꾸지않는다. 위O-only주석은E36에서공용경로로수정했다.
