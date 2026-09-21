# O native FP8 two-lane 독립 검토

2026-09-21, kernel_audit. 읽기 검토 대상은 이 폴더 `projection.rs` SHA256 `afeb4ae8e9eab4f0d616de7ff53a8d28dbdcce6059c27bdff9de4f648a52b30d`다. `project_output`과 `prepare_output_fp8_split` 본문이 root의 E21 저장 스냅샷과 동일함을 확인했다. 소스 수정, 추가 compile, NPU 실행은 하지 않았다.

**공식 3개 입력에서 O는 통과했으며 scale/lane/FP32 경계에서 구체적인 구현 불일치는 발견하지 못했다. 다만 유한 BF16 전체에 대한 수치 동치는 아니며, 아래 반례에서는 큰 오차가 발생한다.**

## 실제 데이터 경로 확인

`ops.rs:140–146`은 입력 `[Ns,Gs,Ds]`를 Qs4096으로 같은 순서에서 flatten한 뒤 `project_output`, `normalize_output_fused`, `add_output_residual`을 호출한다. O의 public entrypoint 서명은 유지된다.

`projection.rs:176–178,391–392`의 layout은 2 clusters × 32 row groups × 8 column groups다. cluster c, row group r, local row i, column group j, local column k에 대해 `h=1920*c+60*r+i`, `q=512*j+k`이다. 입력 x는 H row groups와 두 cluster에 복제되며, 각 column group은 자신의 Qs512 값만 갖는다. 가중치 typed DMA는 동일한 h/q 주소를 사용한다.

`prepare_output_fp8_split`의 amax reduction(`402–414`)은 **local Qs512만** 줄이고 H row group이나 다른 column group을 합하지 않는다. 따라서 각 column의 scale은 그 512개 입력에만 대응하며 row replicas는 같은 입력에서 같은 scale을 계산한다. scale은 `s_j=2^(floor(log2(max(amax_j,2^-119)))-7)`이고, 유한 BF16 입력에서는 `2^-126 ≤ s_j ≤ 2^120`이다. normalized 크기는 256 미만이며 FP8 hi가 최대256으로 반올림될 수 있다. residual×16의 크기는 최대128이므로 유한 정상 입력에서 FP8 448 범위 포화를 이용하는 설계가 아니다.

`433–458`은 `hi=FP8(x/s_j)`, `lo=FP8(16*(x/s_j-hi))`를 계산한다. `459–463`은 packed Element의 lane0에 hi, lane1에 lo를 쓴다. `project_output:179–196`에서 이 축을 TRF Lane으로 넘기고 `LaneMode::Sequential`로 Time `[H%60,Dummy2]`에 다시 꺼낸다. SDK `contract_lane`은 Sequential에서 Lane→OutTime을 명시하며, `vector_intra_slice_unzip<Dummy2,H%60>`는 Time의 lane0/1을 두 group으로 나눈다. pair operand `(),0.0625`는 group0를 유지하고 group1만 1/16로 곱한다.

`197–205`의 각 row/column partial은 다음 순서다.

```text
FP32 dot(weight_j, hi_j)
  + (FP32 dot(weight_j, lo_j) /16)
  → × s_j
  → FP32 partial DM
```

**column마다 다른 s_j를 inter-slice sum 전에 적용한다**는 점이 맞다. 8개 column partial을 먼저 합친 뒤 scale 하나를 적용하는 오류는 없다. `207–217`은 이 FP32 partial을 8-way sum한 후 처음 BF16으로 cast한다. pair zip과 inter-slice sum을 나눈 DM도 FP32이므로 새 BF16 부분합 반올림은 추가되지 않았다. 원래 H별 channel scale은 이 BF16 결과에 기존대로 적용된다(`220–252`).

`OutputCluster`와 lane이 모두 `Dummy2`라는 이름을 쓰므로 처음에는 hidden-axis 혼동을 확인했다. typed API는 Cluster/Element/Lane/Time slot을 구분하며, 해당 tile은 Element, unzip은 Time을 검증한다. 실제 O 장치 결과도 통과했으므로 현재 오류라고 판단할 근거가 없다. 이후 유지보수에서는 두 의미를 명확히 구분해야 하지만, 검증 없이 이름을 바꿀 필요는 없다.

## 수치 한계와 명시적 반례

유한 BF16 z=x/s가 FP8 normal 범위에 있을 때 두 값으로 정확히 복원되는 경우가 많지만, 작은 z는 hi/lo 모두 반올림되어 사라질 수 있다. root의 독립 CPU 모델은 입력별 절대 오차 `|δx| ≤ s_j·2^-14`를 검사한다. 이것은 출력의 상대 오차나 RMSNorm 후 허용오차를 보장하지 않는다. dot의 민감도는 가중치 크기, 작은 결과를 만드는 cancellation, column 내부 outlier와 연결된다. FP32 native contraction의 packet/reduction 순서도 원래 BF16 contraction과 다르다.

다음은 모든 값이 유한 BF16/FP8이고, 원래 dot의 누적 순서에도 의존하지 않는 반례다.

1. 첫 512-column shard에서 `x[0]=2^20`, `x[1]=0.25`, 나머지 x를0으로 둔다.
2. 모든 H output row에서 `w[h,0]=0`, `w[h,1]=1`, 나머지 weight를0으로 둔다. channel scale와 post-norm gamma는1, residual은0이다.
3. 큰 x[0]은 weight0 때문에 원래 출력에 기여하지 않지만 quantizer의 scale을8192로 만든다.
4. 작은 x[1]은 z=`2^-15`, hi=0, residual×16=`2^-11`이다. E4M3 최소 양수 subnormal은 `2^-9`이므로 lo도0으로 반올림된다.
5. 원래 각 output은0.25다. 후속 RMSNorm은 `0.25/sqrt(0.0625+1e-6)=0.999992…`, BF16으로1이 된다. 후보는 projection0과 최종0을 만든다.

위 FP8 반올림은 독립 Fraction 기반 nearest-even E4M3 열거로 확인했다. 공식 fixture를 대체하거나 변경한 검사는 아니며 NPU에서 이 반례를 실행하지 않았다. **현재 공식 입력 통과를 임의 모델 activation에 대한 보장으로 확대할 수 없다.**

## 관측된 장치 검증 범위

[E21 job70300 로그](../../../logs/e21-row-fp8-qkv-native-output-job-70300.log)에서 O run0/1/2가 모두 PASS다. 각 max abs error는0.03125이며 mean abs error는0.000484 /0.000503 /0.000457이다. 이 오류 통계는 기존 BF16 O 경로를 쓴 E20의 세 공식 입력과 같다. O cycles는61,852 /58,442 /59,016, median **59,016**이다. E18 반복 O median65,662 대비 약10.1% 감소다.

**E21 전체 job은 V 실패 때문에 FAILED**이므로 전체 후보 allPASS로 기록하면 안 된다. 이 리뷰에서 검증된 하드웨어 범위는 O 공식3개 입력 구성이다. 채택을 검토한다면 추가6개 입력 구성과 반복 성능 측정이 필요하다. 추가 입력 통과 역시 위 반례를 없애거나 전체 유한 입력 동치를 증명하지는 않는다.

첫 compile의 pair AddF/global AddF 유닛 충돌은 `attempt-1.log`에 보존되어 있다. FP32 partial DM을 추가한 최종 형태는 `attempt-2.log`에서 compile 성공하며, root 보고 정적 makespan은32,252 cycles로 fused-div BF16 기준32,560보다308 cycles 작다. 실제 약10% 개선을 정적 차이0.95%만으로 설명하거나 일반화하지 않는다.

## 남은 정적 비용과 추가 정밀도 변경 없는 후보

Docker scratch `/tmp/moa-output-native-fp8/schedules/output.json`을 읽었다. makespan32,252, instruction56개다. 아래 값은 해당 node duration이며 합이 전체 makespan이라고 해석하지 않는다.

| 경로 | cycles | schedule 위치 |
| --- | ---: | --- |
| FP8 weight HBM→DM | 13,512 | 3,831–17,343 |
| hi/lo→packed DM copies | 각각947 | 17,343–19,237 |
| native FP8 two-lane dot + pair + scale | 1,238 | 19,628–20,866 |
| F32 partial 8-column sum→BF16 | 390 | 21,036–21,426 |
| cluster별 result compact + HiddenRows copy | 469+808 | 21,426–22,703 |
| output channel scale | 311 | 24,303–24,614 |
| canonical H gather + norm 8-slice scatter | 469+449 | 24,614–25,532 |
| RMS local mean-square | 401 | 25,532–25,933 |
| RMS inter-slice sum + EPS + Sqrt | 2,579 | 25,933–28,512 |
| RMS per-element normalization | 401 | 28,779–29,180 |
| RMS gather/BF16 + residual scatter | 744+449 | 29,180–30,373 |
| residual add + final gather | 341+504 | 30,373–31,218 |

우선 검토할 후보는 **O epilogue의 동일 8-slice layout을 유지하는 것**이다. 현재 post-channel-scale BF16이 canonical H로 모였다가 바로 `[1#32,H/480]`로 다시 흩어진다. normalize도 같은 layout에서 계산한 후 H로 gather하고 residual helper가 다시 같은8-slice로 scatter한다. private O helper들이 해당 8-slice BF16 layout을 직접 주고받으면 중간 왕복을 줄일 수 있다. RMS의 H480 합/8-way sum 순서를 그대로 유지하고, normalized FP32→BF16 경계를 residual add **이전**에 반드시 남겨야 한다. 복사와 독립적인 elementwise BF16 cast의 위치만 옮기는 후보이며, normalized FP32에서 residual을 바로 더하는 것은 이 후보에 포함하지 않는다. 실제 direct-copy 비용은 별도 compile로 확인해야 한다.

그 다음은 **현재 FP8 계산을 그대로 두고 hi/lo를 packed tile에 직접 commit하는 것**이다. 현재 두 FP8 임시 텐서를 만드는 Main은5,322까지 끝나지만 TDMA가 weight를 읽고 있어 packed copy 두 개는17,343 이후에 실행된다. packed의 lane0/1에 Main 결과를 바로 쓸 수 있다면 이1,894-cycle 직렬 copy와 임시 DM을 제거할 여지가 있다. hi를 lo 계산의 VRF operand로 읽는 typed tile view가 올바른지, Cluster/Lane Dummy2 구분과 유효512byte 구간을 compile로 검증해야 한다. 이 아이디어는 새로운 양자화나 산술 변경을 추가하지 않는다.

더 큰 변경으로는 channel scale을 기존 2-cluster row layout에서 적용한 뒤 normalization의8-slice layout으로 직접 이동하는 방법이 있다. 기존 dot BF16→scale FP32→BF16 순서를 유지하면서 중간 cluster compact/HiddenRows/canonical H 단계를 줄일 수 있는지 조사할 수 있다. 다만 source padding과 high H axis rename이 있는 typed DMA라 앞의 epilogue 왕복 제거보다 구현·검증 범위가 크다.

weight DMA 자체가 가장 긴 node지만 현재 코드 줄만 앞당기면 개선된다고 단정할 수 없다. scalar scale/input DMA 및 packed copies와 같은 TDMA 자원을 사용하며, compiler가 이미 독립 작업을 겹친다. 또한 HiddenRows 전달 뒤 ExplicitSync600 및 Main 시작까지의 추가 간격도 있으므로 source-line duration의 단순 합산 대신 새 schedule과 실제 NPU 결과로 판단해야 한다. 위 후보들은 **아직 구현하거나 컴파일하지 않았다**.
