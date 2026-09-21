# O 8-slice epilogue 독립 검토

2026-09-21, kernel_audit. `evidence.json`의 세 source SHA256과 실제 파일이 일치함을 확인했다. 읽기 검토와 별도 CPU 좌표 열거만 수행했으며 소스 변경·compile·NPU 실행은 하지 않았다. **검토 범위에서 구체적인 행 배치 오류나 BF16 경계 누락은 발견하지 못했다.** 장치 정확성·성능은 별도 NPU 결과가 필요하다.

## 호출과 값의 흐름

`ops.rs:142–156`의 public `sliding_attention_output`은 기존 서명을 유지한다. `[Ns,Gs,Ds]` 입력을 Qs로 flatten하고 O projection, channel scale, RMSNorm, residual add를 거쳐 같은 HBM residual에 쓴다. 변경된 private helper의 입출력은 모두 `Cluster, [1#32,H/480], H%480`이며 이 호출부가 직접 연결한다. `full::projection::project_output` 등 별도 경로는 이 private helper를 사용하지 않는다.

`projection.rs:181–221`의 native FP8 분해·two-lane dot·column별 scale·FP32 partial·8-way sum·BF16 cast는 유지된다. 이후 기존의 cluster compact→HiddenRows→canonical H→8-slice 이동을 `224–225`의 high-row axis rename와 typed DMA 하나로 줄인다. channel scale도 이 8-slice layout에서 계산하며 RMSNorm과 residual이 같은 layout을 이어받는다. 마지막 HBM write는 typed `to_hbm_view`다.

## 3840개 H 좌표와 padding

원본 H row를 `h=1920*c+60*r+i`로 두면 `c∈[0,2)`, `r∈[0,32)`, `i∈[0,60)`다. column reduce 이후 layout은 `Cluster=Dummy2`, `Slice=[H/60%32,1#8]`, `Element=H%60`이므로 유효한 물리 좌표는 `(c,8*r,i)`다.

`projection.rs:224`는 Cluster 이름만 `Dummy2→H/1920`으로 바꾼다. 기존 Cluster0/1의 의미가 이미 H의 위1920개 구간이므로 실제 값의 순서·slice padding·local60개 순서는 그대로다. HBM weight도 `191–192`에서 `[Dummy2,H%1920,Qs]`로 동일 순서 분해되므로 이 Cluster 의미가 서로 일치한다.

다음 typed DMA의 목적지는 `Slice=[1#32,H/480]`, `Element=H%480`이다. 유효 좌표는 `(slice=h/480,local=h%480)`이고 나머지248 physical slices는 padding이다. source의 연속8개 row groups가 destination 한 slice의480개 값을 채운다. source cluster 하나는 destination4개 slices에 대응한다.

Python 표준 정수 연산으로 h0–3839 전부에 대해 다음을 독립 확인했다.

- source 및 destination 좌표가 각각3840개 모두 유일하다.
- rename 전후 source 물리 offset이 같다.
- 두 표현 모두 원래 h를 정확하게 복원한다.
- source 유효 slice는8의 배수이며, typed mapping의 논리 데이터에는 padding slice가 없다.
- destination은8 slices×480개를 누락 없이 채운다.

이는 논리 mapping 검증이며 compiler typed-DMA 구현의 장치 동작을 대신 증명하지 않는다. unsafe reshape는 이 자리의 **Cluster 의미가 H/1920일 때만** 정당하다. up/gate의 L-high Cluster나 다른 row 분할에 같은 이름 치환을 그대로 적용해서는 안 된다.

## 산술·반올림 경계

BF16 경계는 다음대로 유지된다.

```text
column FP32 partial sum → BF16
→ FP32 channel-scale multiply → BF16
→ FP32 RMS division and gamma multiply → BF16
→ FP32 residual addition → BF16 → HBM
```

channel scale(`projection.rs:231–259`)은 각 h에서 같은 BF16 value와 같은 HBM scale[h]를 F32로 읽어 한 번 곱한다. local120을480으로 늘렸지만 elementwise 연산이며 reduction이나 순서 변경이 없다. VRF/FETCH의 H%480 mapping도 입력과 동일하다.

RMS helper(`rmsnorm.rs:107–180`)는 기존과 같은 H480 partition, 동일한 순서의 local square/reduce, H3840 나누기, 8-way FP32 sum, EPS 더하기, Sqrt, per-element DivF 및 gamma multiply를 쓴다. 원래 normalize 결과를 F32 DM에 쓴 뒤 순수 gather를 거쳐 BF16으로 cast하던 것을 같은 local slice에서 바로 BF16으로 cast한다. 제거된 F32 DM 저장/읽기와 gather는 수치 연산이 아니므로 코드 수준에서 새 BF16 반올림이 추가되거나 제거되지 않는다. 최종 BF16은 `178–180`에서 materialize된 후 residual helper로 넘어간다.

residual helper(`projection.rs:263–290`)는 HBM residual을 같은 H480 layout으로 읽고 기존 `vector_clip(Add)`와 BF16 cast를 유지한다. normalized FP32와 residual을 직접 합치는 재결합은 없다. HBM write는 모든 h에 대해 이 final BF16 값을 쓴다. 기존 generic `normalize` 부분은 `output-rms-reciprocal/rmsnorm.fused-div.rs`와 byte-identical임을 확인했다.

상속된 native FP8 양자화의 일반 입력 한계는 이 복사 최적화와 별개로 남아 있다. `../output-native-fp8-two-lane/review.md`의 finite BF16 반례와 공식 입력 범위 제한은 그대로 적용된다. 이 리뷰는 native FP8을 임의 입력에서 정확하다고 보증하지 않는다.

정적 makespan32,252→29,088은 후보의 기존 compile evidence 값이다. 실제 성능 개선과 허용오차 통과는 root의 E26 실행 결과를 확인해야 한다. 검토 시점에는 그 결과를 아직 읽지 않았다.
