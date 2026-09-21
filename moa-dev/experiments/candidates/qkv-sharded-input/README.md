# QKV input RMS→quantizer의 H480 layout 유지

Frozen E27의 Q/K two-lane pair, V two-dot, direct packed quantizer를 유지하면서 input RMSNorm과 quantizer 사이의 canonical H gather/scatter를 제거한 후보이다. **Exact QKV compile PASS, 정적51,710→50,364 cycles(-2.60%)**다. NPU는 아직 실행하지 않았다.

`ops.rs`의 QKV 입력을 처음부터 `Cluster=2`, `Slice=[1#32,H/480]`, `Element=H%480`으로 typed DMA한다. 기존 `shared::rmsnorm::normalize_output_fused`가 같은 partition에서 BF16 결과를 반환한다. 이 함수 이름은 O용으로 만들어졌지만 Cluster generic이며 H3840 RMS 수식은 input norm과 같다. 함수 본문이나 shared rmsnorm 파일은 수정하지 않았다.

`prepare_fp8_split`은 sharded BF16을 직접 받는다. 각 slice의480개 absolute max를 구하고, typed `vector_inter_slice_reduce::<Slice, m![1]>(Max)`로 **전역 amax 하나**를 만든다. 출력 `Slice=[1#256]`에는 cluster당 유효 scalar 하나만 있다. 기존 exponent mask·floor·scale 생성과 scalar broadcast는 그대로 유지된다. per-shard scale을 사용하지 않는다. 이8→1 typed reduction이 컴파일됐으며 padding을 유효 데이터로 바꾸는 새 unsafe reshape는 없다.

layout 검증은 h0–3839 전부에 대해 `(slice=h/480,local=h%480)`가 일대일이며3840개 값을 모두 포함함을 확인했다. 유효 physical slice는0–7이고 Cluster0/1에는 같은 입력이 복제된다. 유한 BF16에서 다음 등식은 Max가 값을 선택하는 연산이므로 합산 반올림 없이 성립한다.

```text
max_s max(max_i abs(x[480*s+i]), 2^-119)
  = max(max_h abs(x[h]), 2^-119)
```

독립 CPU 검사로 zero vector를 포함한 유한 BF16 값32개 벡터 사례의 partition max가 같음을 확인했다. 수학/layout 검증은 `math-layout-verification.json`에 있다. 이것은 장치 reduction 구현을 대신 검증한 결과는 아니다.

RMS는 기존과 같은 H480 local square/sum, H3840 divide, 8-way sum, EPS, Sqrt, DivF, gamma multiply 순서다. 재사용한 fused helper는 generic normalize의 EPS `Clip Add`를 `Fp AddF`로 수행하며 Sqrt를 같은 pass에 둔다. F32 결과의 순수 gather를 없애고 같은 위치의 BF16 반올림을 local slice에서 수행한다. 새 BF16 경계는 추가하거나 삭제하지 않았다. 하드웨어에서 input RMS로도 같은 허용오차를 만족하는지는 별도 검증해야 한다.

`projection.rs`의 `prepare_fp8_split` 앞 전체 prefix는 E27과 byte-identical이므로 Q/K/V math, head transforms, O projection/epilogue/quantizer는 그대로다. `ops.rs`도 QKV의 입력 layout와 norm 호출 두 줄만 바뀐다. `rmsnorm.rs`는 E27 prerequisite의 정확한 사본이며 현재 live가 E27/E26 O parallel helper를 갖고 있으면 덮어쓸 필요가 없다.

정적 profile은 다음과 같다. node duration 합을 전체 makespan으로 해석하지 않는다.

| 항목 | 이전 | 후보 |
| --- | ---: | ---: |
| makespan | 51,710 | 50,364 |
| instructions | 174 | 171 |
| input canonical→8-slice copy | 538 | 제거 |
| norm F32 gather/BF16 | 744 | 제거 |
| norm 별도 Sqrt pass | 282 | 기존 fused helper에 통합 |
| amax Main | canonical1241 | local401 + global2562 |
| quantizer BF16 reshard | 존재 | 제거 |

amax 자체는 global reduction 때문에 느려진다. 전체 개선은 중간 이동 제거와 DMA/Main overlap을 포함한 최종 schedule에서 측정한 것이다. 성능을 노드 차이의 단순 합으로 추정하지 않았다. 정적 baseline은 `qkv-fp8-v-two-dot`의51,710-cycle schedule이며 E27의 QKV helper 및 generic RMS 본문이 같다(끝 공백 제외). O-only 코드 차이는 해당 exact QKV compile에 호출되지 않는다.

후보 파일은 `ops.rs`, `projection.rs`, prerequisite `rmsnorm.rs`, patch는 `candidate.patch`, source와 kernel SHA는 `evidence.json`이다. Raw schedule/log는 `../../artifacts/qkv-sharded-input/`, 독립 Docker scratch/target은 `/tmp/moa-qkv-sharded-input`이다. live source·Git·NPU 제출은 하지 않았다.
