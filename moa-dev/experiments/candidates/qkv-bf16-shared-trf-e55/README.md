공통 BF16 activation TRF 한 개를 Q/K/V가 빌리도록 바꾼 후보다. **첫 exact compile은 통과했지만 전체 makespan이51,293→52,562 cycles(+1,269, +2.474%)로 늘어 채택하지 않는다.** TRF 예약과 Sub 작업은 줄었지만 K lookup-table 준비가 critical path로 이동했다. 두 번째 compile, full host build, NPU 실행은 하지 않았다.

기준은 [frozen fused-channel-scale 후보](../qkv-bf16-fused-channel-scale-e54/README.md)의 projection SHA`380e57d6502e26086bd6d933ce762f4ad5de18587d52a8ef4aac7fa87e3b9c9f`와 E54 ops다. [projection.rs](projection.rs), [ops.rs](ops.rs) 두 source만 바꾸었다. O suffix, gather/ring32 broadcast, input RMS, BF16 activation, FP32 dot→channel scale→첫 BF16 cast, output row Switch/head RMS/RoPE/cache, FFN, public signature와 공식 harness는 그대로다. 나머지41개 source와 manifest/toolchain은 byte-identical이다.

## 실제 구현과 검증

`prepare_shared_input`이 기존 full-copy DM을 `Cluster=Dummy2, Slice=Dummy256, Lane=1, Element=H`의 TRF에 한 번 적재한다. 두 축은 기존의 named axes다. Q/K/V private helper는 같은 `&SharedInputTrf`를 받는다. 공개 TRF reshape는 사용하지 않는다.

Weight HBM→DM은 원래 typed Q/KV mapping을 유지한다. 그 DM view와 기존 inverse scale Switch가 만든 VRF를 동일한 물리 cluster/slice 순서의 neutral named axes로 바꾼 뒤 dot에 넣는다. Raw BF16 dot/scale 결과는 원래 Q/KV row 이름으로 복원하고 기존 output Switch에 전달한다. `Dummy2/Dummy256`의 축 이름만 바뀌며 H 순서, local row, scale의32-byte row stride, packet padding은 바꾸지 않는다.

Q 행 `q=2048*c+64*g+8*r+s`, K/V 행 `p=1024*c+32*g+8*r+s`에서 `(c,8*g+s,r,h)`를 그대로 neutral `(Dummy2=c,Dummy256=8*g+s,r,h)`로 대응시킨다. Q `r<8`, K/V `r<4`다. Weight 원소의 local byte 주소는 두 view 모두 `3840*r+h`, scale은 `32*r`, BF16 결과는 `2*r`이다. 각 row의 H0..3839 연속 구간에서 base와 unit stride가 같으므로 모든 weight 원소 주소를 보존한다. K/V의 row4..7 및 scalar packet padding을 읽는 데이터로 승격하지 않는다.

**Compile 전에** [verify_candidate.py](verify_candidate.py)를 실행해 Q4096/K2048/V2048 전 row와 총31,457,280 weight 원소의 연속주소 대응, scale·결과·padding, source scope를 확인했다([당시 결과](verification-before-compile.json)). DPE fetch/lookup/reduction/scale/cast pipeline은 `&x_trf`를 이미 borrowed 인자인 `x_trf`로 고친 부분 외 문자열상 동일하다. 성공 후 같은 검사를 반복했고 [46개 실제 Docker 입력 해시](compiled-source-manifest.json)도 host snapshot과 모두 일치했다.

Lowered graph의 Q/K/V dot `O21/O36/O51`은 모두 **TRF tensor32**를 입력으로 받고, 그 producer는 `RenegadeCommand#O13`의 단일 `StoTrf`다. Graph는 BF16 `H3840`, named `Dummy2×Dummy256`과 direct fetch를 명시한다. 세 weight/scale의 `IO(1/2/3)` 및 `IO(4/5/6)` 연결도 reshape chain을 따라 확인했다. 세 dot 모두 `operation_unit→FpMul0→BF16 cast→4행8B commit`을 유지한다. 새 relabel의 source/graph 검증은 NPU 정확성이나 모든 입력의 비트 동일성 증명을 대신하지 않는다.

## 비용과 회귀 구간

| 항목 | 기준 fused | Shared TRF | 차이 |
|---|---:|---:|---:|
| 전체 makespan | 51,293 | 52,562 | +1,269 |
| Instruction 수 | 140 | 147 | +7 |
| Sub duration 합 | 10,979 | 9,756 | −1,223 |
| Main duration 합 | 29,969 | 29,969 | 0 |
| DmaLoad duration 합 | 34,916 | 34,916 | 0 |
| Activation TRF 수 | 2 | 1 | −1 |
| 관측 TRF 예약/slice | 65,536B | 32,768B | −32,768B |

다른 DMA/Core duration 합도 동일하다. Instruction 수 증가는 zero-cost reshape 항목 등을 포함하므로7개 추가를 연산7개의 실행비용으로 해석하지 않는다. Resource duration 합은 overlap을 포함하며 전체 makespan과 다르다.

공통 TRF Sub는16,032–17,255(1,223 cycles)에 실행되고 schedule TRF lifetime은16,032–46,973이다. 논리 값은 slice당7,680B이지만 allocator 예약은32KiB다. 기준은 KV TRF15,866–45,704, Q TRF17,089–31,963의32KiB 두 개였다. 이 후보가 “사용하지 않는 TRF를 물리적으로 재배치할 수 있다”는 일반 API를 제공하는 것은 아니다.

| 중요한 의존 구간 | 기준 fused | Shared TRF |
|---|---|---|
| Q dot | 26,900–31,963 | 24,914–29,977 |
| K weight DMA 완료 | 32,494 | 31,553 |
| K lookup DMA | 3,827–4,665 | 31,553–32,391 |
| K lookup Sub 준비 | 4,870–6,163 | 32,391–33,684 |
| K dot | 32,494–35,157 | 33,763–36,426 |
| V weight DMA | 36,075–43,041 | 37,344–44,310 |
| V dot | 43,041–45,704 | 44,310–46,973 |
| 마지막 V cache 완료 | 50,693 | 51,962 |

Q는1,986 cycles 빨라졌다. 그러나 기준 K lookup은 미리 준비되어 K weight DMA 완료 직후 dot이 시작한다. Shared 후보는 K weight 완료31,553 뒤 lookup DMA838 + Sub1,293 +79 gap을 기다려33,763에 시작한다. Weight가941 cycles 빨라진 이득보다2,210-cycle 후속 준비가 커서 **K dot이1,269 늦어진다**. 이 차이가 V weight/dot과 최종 cache tail에 그대로 이어지며, 마지막600-cycle sync를 포함한 전체 makespan도1,269 증가한다.

관측된 allocation 차이도 명확하다. 기준 Q lookup DM의 schedule 주소는33,554,432, K는1,835,008이다. Shared 후보는 Q와 K 모두33,554,432을 서로 겹치지 않는 lifetime(Q4,870–29,977, K32,391–36,426)에 재사용한다. 이것은 실제 graph/schedule에서 확인한 주소·준비 시점 차이다. **TRF 공유가 왜 allocator/scheduler의 이 선택을 유발했는지 내부 원인은 확정하지 않는다.** 추가 prefetch나 lifetime 제어 variant는 만들지 않았다.

[static-comparison.json](static-comparison.json)에 graph 입력 identity, 단일 TRF, table DMA/Sub, dot lifetime 및 allocator 예약을 저장했고 [analyze_artifacts.py](analyze_artifacts.py)로 같은 artifact를 재검증할 수 있다. 가설 단계였던 [input staging audit](../../../../docs/research/qkv-bf16-input-staging-audit.md)의 “Sub1,223 감소가 전체 개선을 보장하지 않는다”는 제한을 이번 정적 결과로 구체화했다.

## 재현·동결

Docker `/tmp/moa-qkv-bf16-shared-trf-e55`와 그 내부 target만 사용했다. 실행한 exact compile은 한 번이다.

```bash
cd /tmp/moa-qkv-bf16-shared-trf-e55
CARGO_HOME=/home/moa/.cargo \
CARGO_TARGET_DIR=/tmp/moa-qkv-bf16-shared-trf-e55/target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-bf16-shared-trf-e55/target/furiosa-opt/kernel \
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
  --dump-schedule candidate.json --dump-graph candidate-graph.json
```

E54 checkpoint는 root가 제공한 `2fceb2d`다. 저장소에서 복원하려면 그 crate에 이 후보의 `projection.rs/ops.rs`를 적용하고 `Cargo.lock`은 tracked [moa-dev/locks/gemma4-Cargo.lock](../../../locks/gemma4-Cargo.lock)을 사용한다. Lock SHA`aaff8dd41f9c643dd8c9faa358b9bc780e88a42c09172c1a5d7a0726d07efe09`가 frozen/actual compiled lock과 같음을 확인했다. 최종46개 파일은 manifest와 대조해야 한다. 이 작업에서 Git checkout이나 Git 변경은 하지 않았다.

- Projection SHA: `26b7ada776900a82ccc716deaa26e1f6d7ed55c4e86fce9f1098132e1d766fb5`
- Ops SHA: `288d6d27561462ec480956ef40cd8cd8f6a3855ccfb3a3ca3df881b1bb66872e`
- Image SHA: `19896d832e5f20f753591868c4f535b42fc02837562bdfa8aed1cbe48bb9097b`(169,984B)

[prepare_source.py](prepare_source.py)의 최종 source 재생성이 byte-identical함을 확인했다. [manifest](manifest.json), [patch](candidate.patch), [성공 로그](../../artifacts/qkv-bf16-shared-trf-e55/compile-attempt1.log), source crate·image·graph·schedule을 동결했다. 전체 정적 회귀에 따라 종료하며 live source/target, 이전 candidate, Git, NPU는 변경하지 않았다.
