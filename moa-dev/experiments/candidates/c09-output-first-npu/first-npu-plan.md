# C09 동결 two-stage-v1 첫 NPU: 사전 감사와 제한된 실행 계획

검토 기준: `/home/jihwan38/moa-workspace`, 2026-09-27 KST. 이 문서는 기존 보존물 읽기와 CPU 해시/주소 계산만 수행한 감사 기록이다. 검토자가 새 compile, host build, Arena 제출을 실행하지 않았다. 부모 에이전트가 M08 종료를 main/origin `617aaef4f28790fac12de6efc6068b6d49199580`에 통합하고 별도 `diag/c09-first-npu-20260927`에서 새 exact compile을 완료했다. 그 결과는 아래에 구분한다.

## 보존물과 동일성

원본 후보는 `moa-dev/experiments/candidates/c09-output-pipeline/`의 `two-stage-v1/projection.rs`, `build-c09-two-stage-v1-repro.json`, `static-screen.json` 및 `evidence/c09-two-stage-v1/`이다. ignored artifact `moa-dev/experiments/artifacts/c09-output-pipeline/c09-two-stage-v1/`에 46개 crate 입력, exact-kernel.bin, schedule.json, compile.log가 실제로 남아 있다. 원 control도 `c09-e104-control` label로 같은 보존물을 갖고 있다. 두 crate의 46개 입력을 각 manifest와 대조했고 동결 projection과 실제 compile 입력의 동일성도 확인했다.

| 대상 | bytes | SHA256 |
|---|---:|---|
| C09 projection | 13,250 | `5692369dd5b8f6ddbeecdfaa9289a9683fae18798b920a71775362f306560fe6` |
| C09 exact O | 80,640 | `fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495` |
| C09 schedule | 30,955 | `198eda7db2beae902184646ec2b838651b10ded1e91ff94d596cfe50b49181b4` |
| E104/E122 exact O | 71,168 | `359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b` |
| C09 당시 control schedule | 20,636 | `0892c22c587d52873e82c9f7380e1f000154b471fb4fce5eb5e8c63382125084` |

원 C09는 schedule-only compile이므로 당시 graph/IR/vISA 보존물은 찾지 못했다. 따라서 과거 source/schedule 검토만으로 실제 DMA descriptor 전 범위를 검증했다고 쓰면 안 된다. 원본 NPU 결과와 host package도 없다. 과거 `compiled-not-measured` decision과 static +0.71%는 그대로 역사 기록으로 유지한다.

원 E104 source `7f891f9c48e670fe014b2ac5bd2c92bae3bac433`에서 E122 source `c3553ab87046fb4d59c10149a82517f8440bba27`까지 crate 변경은 `src/device/shared/mlp.rs` 하나다. 원 C09 manifest와 E122 manifest는 C09의 `projection.rs` 및 E122의 `mlp.rs` 두 파일만 다르다. O 호출은 `src/ops.rs:141`의 `sliding_attention_output` → `projection::project_output`(151) → `normalize_output_fused`(152) → `add_output_residual`(154) → HBM store(155)이며 MLP를 호출하지 않는다. C09 projection의 QKV 부분과 출력 epilogue는 E122와 동일하다. 이 때문에 O 동일 재현을 기대할 수 있었고, **부모가 생성한 새 `c09-output-first-npu/build-c09-first-npu-v1-repro.json` 및 실제 artifact를 읽어 original C09 exact image와 schedule의 byte 동일성을 확인했다.** 새 graph는 65,386B, SHA `ca9481ad50974e92f99a957070b616786322c799b5d1dfb20164abe52ad8fa99`이다. 이 동일성이 아직 실행 정확도나 성능 PASS를 뜻하지는 않는다.

## 원 source의 producer/consumer 계약

근거: `c09-output-pipeline/two-stage-v1/projection.rs:160–210`.

1. BF16 x를 256개 K 원소씩 기존 slice DM→Sub TRF에 넣는다(168–173). 두 stage가 같은 x TRF를 재사용한다.
2. HBM FP8 weight의 reshape는 `[Dummy2, H/120%16, H%120, Qs]`이고, `tile(...0)`와 `tile(...60)`가 각 120행 그룹의 앞/뒤 60행을 **별도 compact DM**에 전송한다(174–181). HBM tile의 `60 # 120`은 원래 120행 그룹 간 stride를 유지한다. 단순히 HBM 앞 절반/뒤 절반을 연속 복사하는 동작이 아니다.
3. stage Main은 FP8 table lookup→BF16 dot→기존 K16 F32 inter-slice reduce 후 각 60행 결과를 동일 F32 120행 result의 별도 창에 commit한다(197–210). K 차원을 두 번 나누거나 BF16 partial을 추가하지 않는다.
4. 두 stage가 완료된 result를 canonical H480 epilogue 배치로 한 번 gather하고, F32 channel scale 후 첫 BF16을 만든다(185–187,215 이후). 결과 H mapping과 F32/BF16 경계는 기존 O와 같다.

정확한 logical index는 `h=1920*c+120*g+60*stage+t`, `slice=16*g+q`, `k=256*q+u`이며 `c<2,g<16,stage<2,t<60,q<16,u<256`이다. 이 mapping을 CPU로 열거하면 3840행을 정확히 한 번씩 커버한다.

| source/address 조건 | 검토 결과 |
|---|---|
| stage 0 HBM source | 최소 0, exclusive 최대 15,482,880B; 총 7,864,320B |
| stage 1 HBM source | 최소 245,760, exclusive 최대 15,728,640B; 총 7,864,320B |
| 합계 | 15,728,640B = 15MiB, 두 stage의 valid source 집합은 disjoint이며 전체 weight를 커버 |
| stage DM logical tile | slice당 `60*256=15,360B`; 각 stage 전체 7.5MiB |
| stage packet 예상 | 각 stage 256B × 30,720개; 이는 source에서 계산한 기대값이며 descriptor gate에서 실제 값과 대조 필요 |
| F32 output per live leader | stage 0 `[0,240)`, stage 1 `[240,480)`; 각 30×8B commit, 경계 0/240 모두 8B 정렬 |
| 최종 gather | 기존 32×480B valid F32 = 15,360B; H480 consumer 유지 |

출력 `tile`의 타입은 `H%120=60 #{!}120`이다(210). 이는 HBM read의 `#120`과 바꿔 쓸 수 없다. 설치 SDK 추출본 `/tmp/moa-o-sdk-std/src/tensor/memory.rs:523`은 HBM tile에 Top padding을, `:1321`은 DM mutable tile에 Bottom padding을 검사한다. `/tmp/moa-o-sdk-std/src/tensor/view.rs:131–140`은 write 창 밖을 Bottom padding으로 두어 commit sequencer가 쓰지 않아야 한다고 명시한다. 두 60행은 각각 짝수이며 transpose/trim의 F32 두 원소 8B 단위와 맞는다. 현재 source에서 주소/타입/정렬 오류는 발견하지 않았다. emitted commit 및 DMA gate와 실제 correctness까지 완료되어야 실행 계약 검증이 완성된다.

새 graph gate는 두 stage의 HBM source/destination descriptor를 모두 열거하고 bounds, disjoint union/완전 커버, compact DM 주소, 별도 allocation, 각 Main의 실제 weight/x/table input과 F32 output offset를 검사해야 한다. result의 alias/version을 따라 둘 다 완료된 source가 canonical gather의 입력인지도 확인한다. 물리 allocation extent의 padding을 valid payload 또는 실측 traffic으로 세지 않는다.

## 이미 드러난 추가 비용과 관측할 질문

`static-screen.json`은 control 27,239 → C09 27,432 compiler cycles(+0.7085%), 실제 실행 instruction 32→49, HBM load 6→8을 기록한다. weight duration 합 13,383→13,932(각 stage 6,966), dot+sum 합 5,063→5,326(각 stage 2,663), Sub interval union 3,036→4,329다. LUT 준비도 한 번에서 두 번으로 늘어난다. 중첩 duration을 합쳐 전체 속도 향상으로 환산하지 않는다.

| C09 schedule node | producer→consumer 및 정적 interval |
|---|---|
| 27 | weight stage 0: tensor11→20, 2774–9740 |
| 32 | x TRF tensor21, 3134–3461; 두 dot 공통 입력 |
| 37 | dot 0: weight20/x21/table18/output-view9→22, 9740–12403 |
| 38→41 | stage 1 lookup constant DMA 9740–10578, Sub table 10578–11871 |
| 42 | weight stage 1: tensor15→26, 10578–17544 |
| 53 | dot 1: weight26/x21/table25/output-view28→29, 17544–20207 |
| 43→49→73→74 | first/second output alias와 view 버전을 이어 최종 F32 tensor40 생성 |
| 81 | canonical F32 gather tensor40→41, 21413–21883 |

정적 dot 0와 weight 1의 교집합은 1,825 cycles다. weight 1은 table constant DMA가 끝난 10578부터 시작한다. dot 1은 table 준비 종료11871보다 늦은 weight 완료17544를 기다린다. table 반복·작은 launch·두 commit 창을 포함해도 실제 전체 O가 개선되는지가 첫 질문이다. 마지막 dot 종료→gather 시작 및 gather 종료→scale 시작의 sync/queue 구간도 함께 기록한다. 정적 physical-lifetime union peak 26,329,088→33,738,752B는 padding과 alias를 고려한 compiler extent이지 측정 SRAM traffic/사용량이 아니다.

E125/M08의 contemporary control은 weight marker 약26.7k, dot 경로 약12k로 기존 정적 ratio와 다르다. 원 C09의 정적 +0.71%만으로 하드웨어 손실을 단정할 수 없어 최초 NPU 관측의 이유가 된다. M08의 gather fragmentation 실패와 다른 질문이며, M08 수치를 C09 이득의 근거로 전용하지 않는다. 이 실험은 새 architecture가 아니라 **기존 local dataflow replacement의 첫 하드웨어 검사**다.

## 한 allocation의 제한된 실행

새 폴더 `c09-output-first-npu`, variant `frozen-two-stage-v1`, build `c09-first-npu-v1`, screen `e126-c09-first-npu-screen`을 사용한다. source는 E122 c3553ab + 원 C09 projection이며 performance control은 `baseline/e122.json`이다. 원 C09 record는 수정하지 않는다.

- root의 exact compile/image 동일성 확인 다음, `verify_descriptors.py c09-first-npu-v1` 및 `--check`로 `descriptor-analysis-c09-first-npu-v1.json`의 `passed:true`와 `inputdigests`를 확정한다.
- `package.py c09-first-npu-v1`가 매번 gate `--check` 후 standard host와 generic diagnostic host를 만든다. E122 control recipe/audit/manifest/46 source inputs/실제 package를 확인하고, candidate의 QKV/FFN은 E122 image, O는 동결 C09 exact image를 각 host에서 정확히 한 번씩 찾는다. diagnostic은 test host만 바꾸며 세 device image가 standard와 같아야 한다.
- E104 generic diagnostic host는 **O-only launch**에 한하여 E122와 같은 O image임을 확인하고 재사용한다. E104 FFN을 실행하거나 full E122 diagnostic host로 표현하지 않는다. complete control은 실제 E122 3-kernel host다.
- metadata의 고정길이 name bytes만 patch하고 task/constant/ABI/marker ID를 모두 보존한다. C09 native engine marker는 **26개**, control은22개다. `package.py`는 image와 schedule 전체 순서/kind를 동적으로 매칭하므로 기존22개 하드코딩을 쓰지 않는다. 원본/새 image 모두 실제 mapping26 PASS를 확인했다.
- small host source/build logs/marker-map/image-profile JSON을 `evidence/c09-first-npu-v1/profile-package/`에 freeze한다. package와 screen record는 exact compile record와 분리한다.

root 실행 command(순서대로, shared target owner만 실행):

```sh
python3 moa-dev/experiments/candidates/c09-output-first-npu/verify_descriptors.py c09-first-npu-v1 --check
python3 moa-dev/experiments/candidates/c09-output-first-npu/package.py c09-first-npu-v1
python3 moa-dev/experiments/candidates/c09-output-first-npu/screen.py prepare e126-c09-first-npu-screen --build c09-first-npu-v1
python3 moa-dev/experiments/candidates/c09-output-first-npu/screen.py submit e126-c09-first-npu-screen
python3 moa-dev/experiments/candidates/c09-output-first-npu/screen.py collect e126-c09-first-npu-screen
python3 moa-dev/experiments/candidates/c09-output-first-npu/analyze_profile.py c09-first-npu-v1
python3 moa-dev/experiments/candidates/c09-output-first-npu/analyze_profile.py c09-first-npu-v1 --check
```

7 stage는 `control_before → control_info → control_trace → candidate_info → candidate_trace → candidate_whole → control_after`, timeout70초다. 각 stage 원본3입력, 전체57 correctness comparisons다. 이는 한 allocation의 serial process들이며 독립 반복7회가 아니다. standard-host complete O도 `PROFILE=info`이므로 unprofiled measurement라고 부르지 않는다. info/trace diagnostic은 mode와 측정 overhead를 분리한다. marker의 모든 run/cluster에서 Task 내부범위,26/22개 완전성, 실행순서, Task max-cluster cycles와 reported cycles 일치를 검사한다. 실패/timeout 때 raw log, tar 및 이미 끝난 stage를 보존한다.

전체 O trace로 첫 dot 시작, 두 weight duration/union, dot 0와 weight 1 overlap, 두 번째 LUT constant/prepare, dot 1 readiness, gather duration, scale/tail/전체 Task를 볼 수 있다. 이 질문은 whole O 안에서 관측 가능하므로 isolated microbench를 추가하지 않는다. trace duration은 합산 속도 이득이 아니다. 최종 판정은 E122 complete control_before/after와 candidate_whole의 O cycles에 기반한다. unchanged Q/FFN도 image 및 correctness control로 기록한다.

기본 stop: correctness/descriptor/marker attribution 실패 시 결과를 숨기지 않고 원인을 분류한다. source contract 오류면 이해된 수정만 별도 동결한다. 실측 회귀 또는 target O 이득5%미만이면 여기서 종료하고4/8-stage sweep·다른 row size·table API 재설계로 확장하지 않는다. 5%는 실험 예산 cutoff이며 통계 유의성이나 채택 기준이 아니다. 5%이상 유망한 결과여도 별도 contemporary independent repeat와 추가입력/취소경계 정확도 없이는 채택하지 않는다. 현재는 live E122 유지, 공식 제출 없음.

## 이번 준비 변경

명시적으로 소유한 새 후보의 `package.py`, `screen.py`만 M08 원형에서 C09 설명/동적 module명/기본 build label/전용 `/tmp/moa-c09-screen/` 경로로 바꾸었다. 실행 schema·stage·검증gate·timeout·이미지검증은 그대로다. 두 파일 syntax 검사를 했고 read-only `checked_control()` 및 두 exact image의 `label_image/map_markers` 검사를 했다. 새 host build/Arena는 실행하지 않았다.
