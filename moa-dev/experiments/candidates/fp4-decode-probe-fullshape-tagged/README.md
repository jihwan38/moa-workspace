# FP4 decode fullshape: cluster·slice·row별 다른 입력

**[Arena job70658](../../../../docs/experiments/fp4-decode-probe-fullshape-tagged.json)은 exit0으로 완료됐고 두 경로 모두 exact PASS다.** E37 up/gate와 같은 2개 cluster × 240개 유효 slice(물리 256개) × 32행 × 3840열에서 legacy 2pass와 fused 1pass decode를 비교하는 독립 진단이다. 큰 FFN이나 공식 score를 실행하는 프로그램은 아니다.

Root가 제출한 패키지는 `../../artifacts/fp4-decode-probe-fullshape-tagged-package/`다. fixture 없이 실행한다. `test_runtime`은 2,856,704B이고 SHA256은 `1c4e6c32ee59efa7e444e9feb0aeae4143e345bd3fb8cafbe124c2156f5d961c`다. runner/hash 목록은 `package-manifest.json`에 있다.


실제 업로드 binary/runner hash는 동결 패키지와 일치했다. input 58,982,400개 FP4 roundtrip과 실행 중 15,360개 distinct row/480개 cluster-slice coverage assertion이 통과했다. legacy와 fused 각 58,982,400개, 합계 **117,964,800개 BF16**에서 mismatch·nonfinite·signed-zero-only mismatch는 모두 **0**이었다. 양 cluster, 모든 code/nibble/mod64 오류 집계도 0이다.

Raw log는 `moa-dev/logs/fp4-decode-probe-fullshape-tagged-job-70658.log`, SHA256 `0fba213475fc15ca199f903c6b1c32674761cb11f288fa231d77f276a1cf85f7`다. Arena가 보고한 전체 job duration은 1.9613초였다. 이는 host 생성/검사/전송까지 포함한 진단 job 시간이며 kernel 성능 비교가 아니다.

## 이전 probe와의 차이

작은 probe job70620에서는 두 경로 모두 signed zero를 포함한 exact BF16 검사를 통과했다. 그 후 input을 모든 slice에 복제하는 fullshape 단계를 먼저 컴파일했다. 그 소스와 binary는 `../fp4-decode-probe-fullshape/` 및 별도 artifacts에 보존했다.

복제 입력은 slice나 cluster 간 순서 교환을 구별하지 못하므로 이번에는 HBM 입력 자체를 `[cluster,slice,row,k]`로 확장했다. FP4 입력은 **29,491,200B**이며, 각 cluster/slice/row가 서로 다른 code sequence를 갖는다. `decode_probe.rs`의 두 input 인자 mapping만 바뀌었음을 문자열 비교로 검증했다. 같은 DM shape에 typed DMA로 읽은 뒤의 LUT·collect·cast·commit·output 코드는 보존했다. 계산 contract를 바꾸는 실험이 아니다.

- `legacy_decode`: FP4 DM → FP4 LUT → FP8 DM → FP8 LUT → BF16 DM → HBM.
- `fused_decode`: FP4 DM → FP4 LUT → FP8 → fetch_cast F32 → cast BF16 → BF16 DM → HBM.

각 slice의 local DM shape는 `[32,3840]`다. padding slice 240–255는 논리 데이터가 없으며 typed output DMA가 이를 제외한다. padding을 실제 데이터로 승격하지 않는다. 직접 FP4→BF16 SDK 지원 여부와 문서화된 fused 체인의 근거는 [작은 probe의 SDK 감사](../fp4-decode-probe/README.md)에 있다.

## 입력과 exact oracle

Host는 `identity=(cluster*240+slice)*32+row`로 15,360개 row를 식별한다. 각 512원소 block의 low nibble offset에 identity의 한 nibble을 넣는다. k=0,512,1024,1536의 네 code에서 원래 identity를 정확히 복원할 수 있다. CPU 검사는 모든 row identity의 복원과 중복 부재를 확인한다. 따라서 480개 cluster/slice sequence도 서로 다르다.

각 row의 첫 512개 값에는 모든 256개 ordered low/high nibble 쌍이 들어간다. 전체 입력에서 16개 code, 양쪽 nibble phase 및 각 index%64 위치의 모든 code를 확인한다. 이 검사는 서로 다른 row stride와 packet 위치를 포함하지만 모든 가능한 tensor를 열거하는 검사는 아니다.

Oracle은 SDK 변환을 계산식으로 사용하지 않는 독립 16개 BF16 bit 상수표다. ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6을 bitwise 비교한다. SDK host 변환과의 교차 확인은 각 code에 한 번만 한다.

NPU 실행 순서는 packed input HBM roundtrip 검사 → NaN sentinel로 output 초기화 → legacy 실행/readback/exact 검사 → legacy HBM drop → fused 실행/readback/exact 검사다. 각 경로 **58,982,400개 BF16**, 합계 **117,964,800개**를 비교한다. cluster/code/nibble/mod64별 mismatch 수, 첫 32개 오류 좌표, 대표 bit값, nonfinite 및 signed-zero-only 오류를 기록한다.

## 시간·메모리와 검증 범위

출력 하나는 **117,964,800B(112.5MiB)**, 두 출력의 총량은 **235,929,600B**다. 두 HBM 출력을 동시에 유지하지 않는다. Host는 packed input 약 28.125MiB, code 약 56.25MiB, oracle 약 112.5MiB를 유지하며 readback에 추가 메모리가 든다.

`--host-memory-only`는 Device 생성 없이 같은 크기의 sentinel/HostTensor 변환과 두 전체 exact 검사를 수행한다. 로컬 측정은 **1.274879초, 최대 RSS 437,328KiB(약 427.1MiB)**였다. `--oracle-only`도 PASS다. 이 값은 실제 PDMA/NPU/driver 메모리를 포함하지 않으며 Arena의 70초 내 완료를 보장하지 않는다. CPU 측정 로그와 stats는 별도 artifacts에 있다.

이 probe에는 DPE, scale, activation quantizer, 두 weight buffer의 동시 배치, FFN 전체 lifetime이 없다. 이번 fullshape decode exact PASS로 E13의 실패가 같은 decode 체인과 local shape만으로 항상 재현되는 것은 아님을 확인했다. 전체 FFN의 allocation/lifetime 및 후속 연산은 여전히 검증하지 않았으며 E13 실패 원인을 단정하지 않는다.

## 재현 자료

Parent가 허용한 추가 host build **1회, 실패 0회**로 완료했다. 이전 replicated fullshape의 build 2회는 별도 기록이다. 추가 exact schedule compile은 실행하지 않았다. 실제 ELF에는 진단 kernel 2개만 등록되어 있고 각각 컴파일된 `.bin`과 hash가 같다(`registry-summary.json`).

E37 snapshot의 `src`, Cargo 파일과 toolchain을 별도 scratch에 복사하고 새 module/bin 및 `lib.patch`만 적용한다. `compiled-input-hashes.json`의 실제 48개 입력 중 기존 공식 device/ops/test_kernels는 작은 probe 및 E37과 동일하다. fixture와 원래 shell scripts는 이 binary의 입력이 아니므로 scratch에 복사하지 않았다.

```bash
cd /tmp/moa-fp4-decode-probe-fullshape-tagged
export CARGO_TARGET_DIR=$PWD/target
export FURIOSA_OPT_OUT_DIR=$PWD/target/furiosa-opt/kernel
cargo furiosa-opt build --release --locked --bin probe_decode
./target/release/probe_decode --oracle-only
./target/release/probe_decode --host-memory-only
```

SDK/compiler 0.8.1과 기존 toolchain을 사용했다. live source·Git·NPU 제출은 변경하지 않았다. 확인일: 2026-09-21.
