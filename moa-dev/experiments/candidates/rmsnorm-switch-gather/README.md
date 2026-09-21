# RMS 8개 부분합을 Switch gather 후 local sum

Frozen E26 `output-parallel-epilogue`의 shared `normalize_output_fused`에서 scalar reduction만 바꾼 후보다. **Exact O compile PASS, 정적29,088→27,399 cycles(-5.81%)**다. 비교 기준은 E26 O이며 이후 direct-packed O와 결합한 성능이 아니다. NPU는 미검증이다.

기존 경로는 각 slice의 H480 square/sum을 H3840으로 나눈 FP32 부분합을 만든 뒤, inter-slice Add로8개를 더하고 EPS/Sqrt를 수행했다. 이 마지막 Main의 정적 비용2,579 cycles를 baseline schedule에서 직접 확인했다.

후보는 그8개 scalar를 다음과 같이 처리한다.

```text
8 valid slices의 FP32 partial means
→ typed Switch gather: H/480을 local Time8로 이동
→ local FP32 sum over H
→ 별도 Main: +EPS, Sqrt
→ typed scalar DMA broadcast to the same8 valid slices
→ 기존 DivF, gamma multiply, BF16 cast
```

Gather는 `fetch<1,1#8>`, `switch<[1#256],H/480>(Broadcast1 {slice1:8,slice0:1})`, `collect<H/480,1#8>`를 사용한다. 각 cluster의 `(slice=s,element0)` 값이 `(slice0,time=s,element0)`으로 이동한다. s0–7이 정확히 한 번씩 모이므로 local H reduction에 유효한8개 scalar만 들어간다. Padding을 logical data로 바꾸는 reshape는 없으며, 원래 Dummy8→H/480 unsafe reshape 한 개도 제거했다. EPS/Sqrt 결과는 canonical valid scalar에서 기존8-slice layout으로 typed broadcast한다.

Local reducer는 FP math 뒤의 stage라 EPS/Sqrt를 그 뒤 같은 pass에 붙이지 않았다. 추가 F32 DM과 scalar broadcast를 포함한 측정값은 다음과 같다.

| 단계 | cycles |
| --- | ---: |
| 기존 inter-slice sum + EPS + Sqrt | 2,579 |
| 후보 Switch gather + local sum | 289 |
| 후보 EPS + Sqrt | 282 |
| 후보 scalar DMA broadcast | 319 |

후보의 세 단계는 이 schedule에서 연속으로 실행되어890 cycles다. 차이1,689가 전체 O makespan 차이와 같다. instruction은50→51로 늘지만 수행 시간은 줄었다. 이 profile을 다른 결합 후보에 그대로 더해서 성능을 보장하지 않는다.

H480 부분합 계산과 Div3840, EPS→Sqrt 순서, 후속 DivF/gamma, BF16 cast는 유지된다. **8개 FP32 부분합의 덧셈 순서는 달라질 수 있다.** 부분합이 nonnegative여도 bitwise equality를 주장하지 않는다. CPU `check_reduction.py`는 두 가상 순서(sequential/balanced)를4,096개 positive-F32 부분합 벡터에서 비교했다. Sum892개, RMS277개에서 차이가 났고 max sum2ULP, max relative RMS차이1.572e-7이었다. 이 probe의 representative BF16 출력32,768개는 같았지만, 어느 순서도 실제 RNGD의 연산 순서를 재현하거나 모든 입력에서의 동치를 증명하지 않는다.

`rmsnorm.rs`의 generic `normalize`는 원형과 byte-identical이며 변경은 `normalize_output_fused` 안에 한정된다. Public entrypoint·ops·projection은 E26 그대로다. Root의 이후 Q-sharded 경로도 같은 helper를 호출하므로 이 변경을 결합하면 Q input RMS에도 영향을 준다. 여기서는 **E26 O exact compile만** 수행했으므로 결합 QKV/O의 전체 oracle 검증이 필요하다.

소스 SHA256: `3c3c9525f21fe9916e0f6db00863cbb05e4f0c8cc35501ed08b1bbcdb4635cb0`.
Patch는 `candidate.patch`, source/kernel/schedule 증거는 `evidence.json`, probe는 `cpu-order-probe.json`이다. Raw log/schedule은 `../../artifacts/rmsnorm-switch-gather/`, 독립 Docker scratch/target은 `/tmp/moa-rmsnorm-switch-gather`에 있다. live source·Git·NPU 제출은 하지 않았다.
