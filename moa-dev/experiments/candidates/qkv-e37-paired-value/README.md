# E37 QKV의 V projection을 paired FP8로 통일

E37/HEAD `c79e433cce3b908d07436e72d7ff4bc4672d7fcf`의 V 호출 한 줄만 바꾼 후보다. exact QKV compile은 **50,364 → 50,728 cycles(+0.72%)**로 정적 회귀했다. V의 두 dot와 F32 중간 DM은 제거됐지만 전체 latency 개선은 확인되지 않았다. root의 E44 job70625는 공식15 comparisons 모두 PASS했다. QKV median95,892로 E37 반복 기준94,195보다1.80% 느려 한 번의 screen 뒤 미채택했다. 이 subagent는 NPU를 실행하지 않았다.

[projection.rs](projection.rs)의 SHA256은 `a1bad7a8e909a50130725860f88e32a52cb6a575a75fa817a0911dc96f20e863`이다. [candidate.patch](candidate.patch)는 `project_key_value`에서 V만 `project_value_two_dot(...)` 대신 기존 `project_one_kv_matrix(device, &x_trf, &scale_vrf, v_weight, v_weight_scale)`로 호출한다. 호출되지 않는 예전 helper도 삭제하지 않아 diff를 한 줄로 제한했다. Q, K, input quantizer, O, FFN, 모든 signature와 test harness는 그대로다.

[build-manifest.json](build-manifest.json)은 scratch의 43개 source와 3개 manifest를 [E37 snapshot](../../artifacts/e37-kernel-specific-rms/provenance.json)에 대조한 결과다. snapshot의 source와 manifest는 위 HEAD와 같고, ignored `Cargo.lock`은 해당 HEAD의 `moa-dev/locks/gemma4-Cargo.lock`과 같다. scratch에서 달라진 파일은 `src/device/sliding/projection.rs` 하나다. live source, Git 상태와 공유 target은 수정하지 않았다.

## 실제 데이터 경로와 보존한 범위

E37 `ops::sliding_project_qkv`는 BF16 입력을 H480×8 slice로 읽고 `normalize_sharded_inter_slice` → `prepare_fp8_split`을 거친다. 양쪽 모두 같은 FP8 packed hi/lo와 F32 runtime scale을 Q/K/V에 공급한다. 이후 head RMS, Q/K RoPE, Q 출력, kv_offset 위치의 K/V cache scatter는 변경하지 않았다.

기존 V는 hi/lo 각각 lane1 TRF로 읽고, `dot(lo, W)/16`을 F32 DM에 쓴 다음 VRF로 읽어 `dot(hi, W)`와 더했다. 후보는 K와 같은 lane2 TRF를 공유한다. `contract_packet`/`contract_time`으로 H3840을 줄인 뒤 `LaneMode::Sequential` → `vector_intra_slice_unzip<Dummy2>` → lo에 1/16 → `vector_fp_zip(AddF)` → runtime scale → BF16 cast 순서다. 채널별 BF16 weight scale을 F32로 곱한 뒤 두 번째 BF16 cast하는 원래 경계도 유지한다. 새 양자화는 없지만 DPE lane/ALU scheduling이 달라지므로 bitwise 동치를 주장하지 않는다.

V row `p = 1024*c + 32*g + 8*t + s`에서 c∈[0,2), g∈[0,32), t∈[0,4), s∈[0,8)다. weight load는 slice `8*g+s`, local row t에 전체 H3840을 둔다. paired dot의 4 BF16 결과는 8B commit한다. 기존 8-way Switch가 slice `8*g+t`, local s로 바꾼다. t의 padded 4..7 slot은 Slice padding으로 남고 typed head gather는 유효 2048행만 Ns8×Ds256으로 복원한다. K/V scale도 같은 row 배치를 쓴다. [verify_candidate.py](verify_candidate.py)가 2048개 좌표의 전단/후단 bijection과 padded slot 배제를 전수 검사한다. 기존 helper 전체는 E30과 E37에서 byte-identical이다.

pair TRF의 유효 payload는 slice당 2×3840=7680B, scale VRF는 32B다. weight DM은 slice당 4×3840=15360B다. 기존 low-dot DM은 schedule tensor92, 전체 65536B(512 physical slices×128B)이며 후보에서 사라진다. named low_vrf도 사라지지만 후보의 내부 pair 처리용 VRF tensor88이 65536B이므로 **VRF 용량이 64KiB 감소했다고 해석하면 안 된다.** 공통 K/V TRF lifetime은 baseline K 전용 29009–42098에서 후보 25782–41993으로 길어진다. 컴파일 성공만으로 hardware resource lifetime의 정확성을 증명하지 않는다.

입력 scale은 기존 sharded amax를 그대로 쓴다. `s=2^floor(log2(max(amax,2^-119)))/128`이므로 finite BF16 입력에서 s∈[2^-126,2^120], |x/s|<256이다. RN E4M3 가정에서 lo 입력 `16*(z-hi)`는 절댓값128 이하다. finite FP8 weight 최대448과 H3840에서 hi dot 절댓값 상계는 440401920, lo dot 상계는 220200960으로 F32 dot overflow 범위와 거리가 있다. 마지막 runtime scale/weight scale/RMS까지 모든 finite 입력이 finite 출력을 낸다는 보장은 아니다. 작은 성분의 FP8 손실 및 cancellation 관련 기존 정확도 제한도 그대로 남는다. 이 후보는 공식 입력을 하드코딩하거나 tolerance를 바꾸지 않는다.

## exact schedule에서 확인한 절감과 회귀

같은 scratch와 manifests에서 baseline 1회, 후보 1회만 compile했다. 명령은 `cargo furiosa-opt compile ops::sliding_project_qkv --exact`에 schedule/graph/summary dump를 붙였으며 [evidence.json](evidence.json)에 경로·환경·hash가 있다. opt0.8.1, 기존 Docker image를 사용했고 두 명령 모두 exit0이다.

| 항목 | E37 baseline | 후보 | 차이 |
|---|---:|---:|---:|
| 전체 makespan | 50,364 | 50,728 | +364 |
| instruction 수 | 171 | 162 | −9 |
| V dot Main command 합 | 745+745 | 746 | −744 |
| Main command duration 합 | 27,938 | 27,194 | −744 |
| Sub command duration 합 | 9,803 | 8,038 | −1,765 |
| DmaLoad duration 합 | 32,402 | 32,402 | 0 |
| DmaStos duration 합 | 8,744 | 8,744 | 0 |

V low/high TRF load 743+743 및 low-dot→VRF 279 cycles가 제거됐다. 후보는 K용 TRF 1223-cycle load를 V와 공유한다. 하지만 Main/Sub 작업량 감소를 makespan 절감으로 더하면 안 된다. weight/scale HBM IO는 그대로이고 후처리 스케줄도 바뀐다. baseline V pair 대체 구간은 26497–29009, 후보 pair는27005–27751이다. K dot은 오히려105 cycles 일찍 끝나지만, 최종 K normalize Main은46406–46751에서46770–47115로364 cycles 늦어졌다. 이어진 K RoPE/scatter가 같은364 cycles 뒤로 밀리고 terminal600-cycle sync도 그대로다. Q HBM store 완료도45953→47211로 늦어진다. 이 타임라인은 projection 절감이 전체 단축으로 이어지지 않았음을 보여주며, compiler가 그 순서를 고른 내부 이유를 규명한 것은 아니다. `Core` lifetime 합에는 sync/재사용 제약이 섞이므로 유효 연산량으로 계산하지 않았다.

## E30 선행 대조군과 hardware 검증 한계

[E30 control audit](../qkv-fp8-explicit-hi-identity/control-audit.md)의 [job70440](../../../logs/e30-pair-skip-control-job-70440.log)은 동일 paired helper로 Q/K/V 포함 공식15 comparisons PASS, QKV median98,599였다. 그러나 E30 입력 RMS/amax는 canonical 경로이며 E37의 sharded input과 다르다. 과거 E20/E21 V sparse error는 quantizer 배치 등도 달라 단일 원인 대조군이 아니다. E28 hi×1과 E30 hi bypass가 모두 통과했으므로 hi skip-state가 원인이라고 단정할 근거도 없다.

이 후보의 검증 대상은 **E37의 sharded 입력과 기존 pair helper의 조합**이었다. 정적0.72% 회귀와 별도로 실제 RNGD에서 한 번 비교한 후속 결과는 아래에 기록했다. E37의 측정 성능이나 correctness를 후보에 그대로 승계하지 않았다. [verification.json](verification.json)은 source/shape/static 검증 결과만 기록한다. 재검사는 `python3 moa-dev/experiments/candidates/qkv-e37-paired-value/verify_candidate.py`이며 build나 NPU 작업을 하지 않는다.

후속 [E44 측정 기록](../../../../docs/experiments/e44-paired-value.json) 및 [job70625 로그](../../../logs/e44-paired-value-job-70625.log): Q95,892/O46,090/FFN378,002, 공식15 comparisons PASS다. E37 sharded 입력과 paired V 조합의 세 공식 configuration 동작은 확인됐으나, extra configurations와 반복 latency는 검사하지 않았다. E37 QKV 반복 기준94,195 대비1.80% 느려 미채택했다. 과거 V 오류의 근본 원인은 이 결과로 확정하지 않는다. `verification.json`의 hardware 미검증 표기는 후보를 동결한 최초 검사 시점이며, 후속 실행은 `evidence.json.hardware`에 별도 기록한다.
