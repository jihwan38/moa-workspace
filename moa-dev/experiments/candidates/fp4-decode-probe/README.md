# FP4 decode 분리 probe: 두 경로 모두 exact PASS

**[Arena job70620](../../../../docs/experiments/fp4-decode-probe.json)은 exit0으로 완료됐다.** 기존 `FP4→FP8 DM→BF16`와 fused `FP4 LUT→FP8→fetch_cast F32→cast BF16` 모두3개 입력 pattern에서 BF16 bits가 정확히 일치했다. signed zero까지 포함해 총 **25,165,824개 비교, mismatch/nonfinite/signed-zero-only mismatch 모두0**이다. 실제 업로드 binary/runner hash도 동결 패키지와 일치했다.

이는 **1개 active cluster,256개 live slice,32행×512열**의 decode만 검증한 결과다. E13의2cluster×240live/256physical slice,32×3840, 전체 FFN의 메모리·lifetime·후속 DPE 조건까지 통과했다는 뜻은 아니다. E13 full FFN은 기존대로 실패 기록이며 큰 FFN 코드를 변경하지 않았다. 이 probe는 공식 score/성능 측정이 아니다.

제출 패키지: `../../artifacts/fp4-decode-probe-package/`. `test_runtime` SHA256은 `2ac07eba5d2578b44ae6946b81fcc89bf3e9bd5cedbab7e891f0a92fbdfe6349`(2,191,152B)다. fixture 없이 host가 input과 oracle을 만든다. Raw log는 `moa-dev/logs/fp4-decode-probe-job-70620.log`, SHA `ece0e16a48efb60c33b0c6b9a32e853315fb0b2565ad3968ee8edce0ae27c7fb`다.

## SDK0.8.1에서 확인한 지원 범위

로컬 SDK와 실제 Docker 설치 SDK의 아래4개 source hash가 같음을 `sdk-sources.json`에 확인했다. 추측으로 API를 제외하지 않았다.

- `src/cast.rs:55–84`: `TableLookupCast`의 FP4 구현은 `f4e2m1→f8e4m3/f8e5m2` 둘뿐이다. BF16 출력은 F8 source용 구현에만 있다. **직접 `fetch_table_lookup::<bf16>()` on FP4는 이 SDK의 trait bound를 충족하지 않는다.**
- `src/cast.rs:27–40`: `FetchCast`에도 FP4→F32/BF16 구현이 없다. F8→F32는 있고 F32→BF16도 있다.
- `src/engine/fetch_adapter.rs:197–202`: FP4→FP8 paired lookup 뒤 `fetch_cast`로 F32를 넓히는 체인을 명시적으로 안내한다. `src/engine/mod.rs:272`의 `CanApplyFetchCast for PositionFetchTableLookup`도 이를 허용한다. **E13에서 실패했다고 이 체인이 금지된 API라고 볼 수 없다.**
- 같은 `mod.rs:261–265`에는 lookup 이후 다시 lookup을 연결하는 state가 없다. 한 pipeline에서 FP4→FP8→BF16의 두 lookup을 바로 연결하는 방법도 지원하지 않는다.
- `src/scalar.rs:918–977`은 e2m1의16개 값, low-nibble encoding, FP4→FP8 표와 signed zero 동작을 명시한다. CPU oracle은 이 SDK 변환 함수를 계산식으로 사용하지 않고 별도 상수 BF16 bit table로 만들었다.

따라서 현재 확인된1pass 방법은 F32를 경유하는 문서화된 fused 체인이며, 이 probe 범위에서는 실제 exact 동작을 확인했다. E37 FFN에 바로 적용할 근거는 아직 부족하다.

## 데이터와 검사 경로

`src/decode_probe.rs`에 두 새 device kernel만 넣었다. 두 kernel의 입력/출력 shape는 같고, DPE·scale·runtime activation quantizer는 없다. input `[32,512]` FP4를1active cluster/256live slice에 복제하여 모든 slice를 검사한다.

`legacy_decode`: packed FP4 DM→Main FP4 LUT→FP8 DM→Main FP8 LUT→BF16 DM→HBM.

`fused_decode`: packed FP4 DM→Main FP4 LUT→fetch_cast F32→Cast Engine BF16→BF16 DM→HBM. BF16 commit은8valid values/16B이고 나머지 cast packet padding은 trim한다.

Host `src/bin/probe_decode.rs`는 low nibble first로 직접 bytes를 packing한다. independent oracle은 `[+0,+0.5,+1,+1.5,+2,+3,+4,+6,-0,-0.5,-1,-1.5,-2,-3,-4,-6]`의 BF16 bits16개다. host SDK 변환과의 교차 검사도 하지만 oracle 생성은 상수표에 의존한다.

| case | pattern | 검사 목적 |
|---|---|---|
|0|한 행 전체를 같은 code로 채움,32행에16codes 두 번|같은 nibble 반복과 signed zero|
|1|각 행에 ordered low/high nibble쌍256개 전부, 행별순환|모든 packed byte 조합과 nibble 인접성|
|2|32element packet마다11, 행마다7 code offset|packet 위치와 경계에 따라 값 변화|

모든 case가 양쪽 nibble phase와 각 `index%64` 위치에서16개 code를 모두 포함한다. CPU `--oracle-only`에서 이를 확인했고 Device를 만들거나 NPU를 호출하지 않았다.

실제 실행에서는 input을 HBM에 upload한 뒤 다시 읽어 원래 nibble과 비교한다. 각 출력은 미리 NaN BF16 `0x7fc1`로 채워 누락된 쓰기도 발견하게 한다. 매 case에서 legacy를 먼저 실행·readback·검사하고 fused를 별도로 실행한다. 각각4,194,304개 BF16을 bitwise 검사하고, 오류가 나도 가능한 나머지 경로 검사를 계속해 마지막에 실패 exit를 만든다.

로그는 code/nibble/mod64별 mismatch 개수, 각 code/phase의 대표 실제 bits, 첫32개 mismatch coordinate, nonfinite와 signed-zero-only 개수를 따로 남긴다. 이번 결과는 아래와 같다.

| mode | case0 | case1 | case2 |
|---|---:|---:|---:|
|legacy mismatch|0|0|0|
|fused mismatch|0|0|0|

input HBM roundtrip3개도 모두 PASS다. sentinel/bitwise 검사 범위는 이 input 집합에 한정되며 임의 조건의 전체 FFN 정확성을 보장하지 않는다.

## 정적 IR과 기존 E13 조사

첫 host build와 두 exact compile이 모두 성공했다(총3compile 명령, 실패0;2variants). full host ELF에 두 image만 등록돼 있으며 exact compile image hash도 ELF의 image와 같다.

| 항목 | legacy | fused |
|---|---:|---:|
| 정적 makespan |41,390|39,003|
| instruction count |10|7|
| Main decode pipeline cycles |1,543 +2,823|2,567|
| FP8 intermediate DM |있음|없음|

이는 작은 probe의 compiler schedule이며 FFN runtime 개선율이 아니다. Raw schedules는 `../../artifacts/fp4-decode-probe/schedules/`, 요약은 `schedule-summary.json`에 있다. lookup source와 Main의 unnamed auxiliary tensor를 dataflow로 연결하면 legacy 각 lookup의 auxiliary SRAM은8,388,608B, fused는4,194,304B다. physical padded cluster까지 포함한 값이며 세부 register table layout은 이 schedule에 노출되지 않는다.

기존 [E13 IR 조사](../ffn-blockscale-two-cluster-fused-bf16/fused-decode-investigation.md)는 FP8 output table의 stride/복제와 fused F32 output table의 구성이 다름을 기록했다. 그 차이만으로 오류라고 단정할 수 없으며 이번 작은 probe PASS는 그 체인이 항상 실패한다는 가설을 반박한다. 아직 분리되지 않은 조건은 실제 큰 row stride/column 수,2cluster와240live padding, 전체 FFN의 주변 allocation과 후속 연산이다.

## 재현·동결 자료

E37 snapshot을 별도 Docker `/tmp/moa-fp4-decode-probe`로 복사하고 새 module/bin 및 `lib.rs` module 선언만 추가했다. 기존 모든 device/ops/공식 `test_kernels.rs`는 byte-identical하다. `compiled-input-hashes.json`에 실제51개 입력 hash를 보존했다. 원본 코드·Git·NPU 제출은 이 subagent가 변경하지 않았고 root가 동결 패키지로job70620을 제출했다.

```bash
# 아래 명령은 모두 isolated scratch에서 실행했다.
export CARGO_TARGET_DIR=/tmp/moa-fp4-decode-probe/target
export FURIOSA_OPT_OUT_DIR=/tmp/moa-fp4-decode-probe/target/furiosa-opt/kernel
cargo furiosa-opt build --release --locked --bin probe_decode
./target/release/probe_decode --oracle-only
cargo furiosa-opt compile decode_probe::legacy_decode --exact --dump-schedule schedules/legacy.json
cargo furiosa-opt compile decode_probe::fused_decode --exact --dump-schedule schedules/fused.json
```

SDK/compiler0.8.1, 기존 toolchain 사용. dependency target은 E37 guard target의 독립 복사이며 공유target을 수정하지 않았다. 소스 `decode_probe.rs`/`probe_decode.rs`, `lib.patch`, `evidence.json`, registry/hash 검증 script, read-only job70620 collector를 이 폴더에 동결했다. compiled images/log/binary와 실제 업로드 tar는 artifacts에 있다. 확인일2026-09-21.
