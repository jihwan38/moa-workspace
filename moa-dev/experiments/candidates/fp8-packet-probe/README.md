# FP8 packet reducer 최소 진단 harness

2026-09-21, compiler/std0.8.1. **두 release binary와 exact kernel compile 모두 성공했고 CPU oracle 검사도 통과했다. NPU는 실행하지 않았다.** E35 full256 FFN이 공식15개 검사를 통과하여 native FP8 retained packet4가 실제로 실행되는 사례를 얻었으므로, 이 최소 probe의 추가 NPU 실행은 보류한다. 공식 score 후보가 아니다.

별도 job용 패키지는 `moa-dev/experiments/artifacts/fp8-packet-probe/scalar/`와 `blocks/`다. 각각 `test_runtime`과 `remote_entrypoint.sh`가 있으며 fixture 파일은 필요 없다. 두 바이너리를 같은 job에 순차 실행하지 않아도 된다. 이 agent는 job을 제출하지 않았다.

## 무엇을 분리하는가

`packet_probe.rs`는 chip1, active cluster1, **256개 live slices**, K64, output rows4를 사용한다. Host에서 만든 동일한 FP8 activation64개와 weight4×64개를 TDMA로 모든 slice에 복제한다. Lane1 FP8 TRF와 FP8 weight stream의64-element Outer는 두 커널이 같다.

- `fp8_scalar`: `contract_packet<1>`로64항을 모두 합쳐 row당 F32 하나를 반환한다.
- `fp8_blocks`: `contract_packet<ProbeK/16%4>`로16항씩 네 dot을 남긴다. Interleaved lane1은 네 block 축을 Time으로 옮기므로 `contract_lane<Time=[ProbeR,ProbeK/16],Packet=1#8>`을 쓴다.

둘 다 F32 transpose2→8B commit으로 compact DM 출력을 만들고 HBM에 저장한다. Pair, VE 산술, runtime quantizer, FP4/LUT, local scale, BF16 cast는 없다. Result 수가 다르므로 마지막 output shape는 다르다.

CPU는1/8 단위의 finite FP8 값으로 서로 다른3개 case를 만든다. 모든 곱은1/64 단위이고64항 절댓값 합은256미만이므로 해당 데이터에서는 FP32 합도 정확히 표현된다. Oracle은 FP64로16항 dot 네 개와64항 dot을 각각 계산하고 정확한 FP32 변환 가능성과 `sum(blocks)==scalar`를 확인한다. NPU 실행 시 모든256slices의 scalar1,024개 또는 block4,096개를 case마다 **finite exact numerical equality**로 검사한다. Signed zero는 같은 값으로 취급한다. HBM output은 미리 NaN으로 채워 미작성 영역을 검사 실패로 검출한다.

이 probe가 둘 다 통과해도 FFN의 H3840/60packets,32rows,2clusters,quantizer/register lifetime을 검증한 것은 아니다. Scalar PASS/blocks FAIL이라면 작은 packet4 경로·output layout의 차이를 의심할 수 있지만 packet reducer만의 원인이라고 단정할 수 없다.

## 기존 API를 따른 실제 실행 경로

기존 `src/bin/test_kernels.rs`의 `HostTensor::to_hbm → Device::new(kernel.topology()) → launch(kernel,args) → HbmTensor::to_host`를 사용했다. 각각의 bin은 자기 kernel 하나만 launch한다. `--oracle-only`는 Device 생성 전에 반환하므로 로컬 검증에서 NPU를 사용하지 않는다.

SDK0.8.1 `backend/npu/registry.rs`는 `cargo furiosa-opt build`가 bin의 launch 함수별 kernel image를 ELF `furiosa_kernels` section에 넣고 runtime이 이를 읽는다고 설명한다. 따라서 integration test 대신 실제 `src/bin/*.rs` 두 개를 만들었다. `inspect_registry.py`로 최종 ELF/bincode를 직접 읽어 **각 binary에 자기 함수 한 개와 nonempty image만** 있는 것을 확인했다. Kernel 파일을 별도로 package에 넣을 필요가 없다.

| 최종 binary | SHA256 | 정적 makespan |
|---|---|---:|
| scalar |`9863c412947ae473c330f86033143b4016283fb8fd5b2323e8405f43257ec079`|17,538|
| blocks |`87ad48b9d01c6c41ae0eeca8c1c7d41b42f458cd12f2f8748b93868f39b3a38e`|12,865|

두 커널 모두 정적 instruction7개다. 핵심 Main contraction은 scalar284cycles, blocks284cycles이며 output store 비용이 다르므로 makespan 차이를 packet4 연산 성능으로 해석하지 않는다. 처음 scalar는 padded scalar DM을 바로 HBM으로 옮겨 store107,640cycles가 발생했다. 최종판은 arithmetic을 바꾸지 않고 F32 transpose2 compact output으로 바꿨다. 초기 source/registry hash는 `packet_probe.initial.rs`, `registry-initial.json`에 보존했다. Compile 실패는 없었다.

## 재현

E27의 immutable crate source를 `/tmp/moa-fp8-packet-probe`에 복사하고 `packet_probe.rs`를 `src/`에 추가했다. `lib.rs`에는 `pub mod packet_probe;`만 추가했다. Host 파일 두 개는 `src/bin/`, `probe_common/`은 `src/bin/probe_common/`에 둔다. Cargo manifest는 변경하지 않았다. 원래 커널을 이 binary들이 launch하지 않는 것을 registry로 확인했다.

```sh
./moa-dev/moa bash -c 'cd /tmp/moa-fp8-packet-probe; export CARGO_TARGET_DIR=/tmp/moa-fp8-packet-probe/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-fp8-packet-probe/target/furiosa-opt/kernel; cargo furiosa-opt build --release --bin probe_fp8_scalar --bin probe_fp8_blocks'
./moa-dev/moa bash -c 'cd /tmp/moa-fp8-packet-probe; target/release/probe_fp8_scalar --oracle-only; target/release/probe_fp8_blocks --oracle-only'
python3 moa-dev/experiments/candidates/fp8-packet-probe/inspect_registry.py
```

Exact compile은 같은 scratch/output env에서 `cargo furiosa-opt compile packet_probe::fp8_scalar --exact --dump-schedule schedules/scalar.json` 및 `fp8_blocks`로 실행했다. `evidence.json`, build/compile/oracle logs, source hashes, registry 및 schedule 요약을 보존했다. Live source, Git, NPU를 변경하지 않았다.
