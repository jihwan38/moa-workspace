# M03 제한적 채택 검증 — E122

2026-09-27 KST, `validate/m03-adoption`. **기본 재측정·추가6입력·독립 projection 진단이 모두 통과했다.** 검증 담당 branch는 runtime이나 baseline을 변경하지 않았으며 채택은 root review 후 별도로 수행한다. 새 아키텍처나 tile tuning이 아니라 E117의 동결 M03를 검증한 후속 작업이다.

- [전체 검증](verification.json), [판정](decision.json)
- [6개 allocation 성능](../../../../../docs/experiments/e122-m03-adoption-validation.json), [E104 paired control](../../../../../docs/experiments/e122-m03-e104-paired-control.json)
- [추가6입력](../../../../../docs/experiments/e122-m03-extra-correctness.json), [7개 projection case](../../../../../docs/experiments/e122-m03-probe-correctness.json)

## 성능과 정확도

새 정상 allocation104931은 **E104336502 → M03 331399 → E104336254** cycles이며, 앞뒤 중앙값336378 대비 **1.4802% 감소**다. 각 process의 기본15비교가 모두 통과했다. M03 host/fixture/runner는 E117의 원본과 byte-identical하며 재compile하지 않았다.

이 결과와 기존5개 companion 관측을 합쳐 동일 M03 stage payload를6개 allocation에서 확인했다. FFN job median들의 중앙값은331354, 같은 allocation의 앞뒤 E104 중앙값들을 다시 요약하면336142라 **집계값 비율은−1.4244%**다. 각 allocation의 paired 비율을 먼저 구한 중앙값은 **−1.4846%**다. 계산 순서가 다르므로 두 값을 구분한다. 모든 M03 job median은모든 paired-control median보다 낮다.

`validation_results.py`는 서로 다른 전체 batch를 같은 package라고 표시하지 않는다. `package_files`는 실제 업로드 archive에서 뽑은 해당 benchmark stage의 host/fixture/runner를 canonical 이름으로 정규화한 것이다. 각 run의 `uploaded_archive_files`, 원본 stage 이름, 원본 JSON/로그 해시를 모두 보존했다. Paired control의 `median_cycles`는 두 process median의 중앙값이라는 derived reference이며, 새 benchmark process를 만들어낸 값이 아니다. Full3kernel schema를 유지하지만 byte-identical QKV/O의 timing 차이를 개선으로 주장하지 않는다.

기존5개 중 M06의 두 outer batch는 FAILED지만 M03와 앞뒤 E104는 PASS였다. 새 clean job은 모든 process가 정상 종료했다. 같은 기본3입력을6번 실행한 것이며 기본 M0390비교/FFN18비교는 새로운18입력을 뜻하지 않는다.

Extra job104964는 run3..8의 **30비교**, FFN6입력/23040출력을 통과했다. `structural_build.py --extra`는 선택 origin의 host에서 RUNS/loop offset만 바꾸고 kernel source/PRNG/tolerance를 유지한다. 기대 source manifest에 이 host 변환을 명시하고 실제 조립을 검사한다. [검사](extra-image-verification.json)에서 세 device image가 기존 M03와 동일함을 확인했다. Extra timing은6개 upper median이므로 기본 성능집계에 넣지 않았다. 표준 FFN 입력의 서로 다른 case는 기본3+extra6=9개다.

## 변경된 경계에 대한 진단

Diagnostic source는 동결 M03 파일을 그대로 prefix로 복사하고 private helpers를 호출하는 child module과 root `#[device]` entry만 추가했다. 실제 `project_up_and_gate`, `project_down_tile`, `finish_down` 구현을 호출한다. 원본 몸체를 복사해서 새로 계산하는 모사 kernel이 아니다. HBM 입출력이 추가된 별도 image라 성능 비교에 쓰지 않는다.

C07 독립 FP4표/F64 oracle6cases를 재사용하여 FP4 16codes, row identity, block/column 경계, zero, cancellation, 작은 mixed input을 검사했다. M03의 두 L-half partial 합산을 직접 시험하는7번째 case를 추가했다.

`down x[0]=1, x[1]=15/256, x[7680]=−1, x[7681]=−14/256`, 나머지0, block scale1, 각 row는 일정한 FP4 weight를 쓴다. w=1일 때 cluster별 F32 partial은1.05859375와−1.0546875이고 정답은0.00390625다. 각 partial을 먼저 BF16으로 바꾸면0.0078125가 되어 틀린다. [독립 binary-fraction 검산](cross-cluster-oracle.json).

**Job105005: 7cases×up/gate/down, 총241920개 BF16 출력 모두 독립 oracle과 일치.** Output buffer를 NaN으로 채워 누락 저장도 검사했다. 이는 지정한 binary-fraction case의 정확한 비교이며 일반 BF16 입력 전체나 전체 모델 동치를 뜻하지 않는다. E5M2 세 항의 기존 일반 입력 근사 한계는 남는다. Full FFN extra6가 nonlinear GeGLU와 epilogue 연결을 별도로 검사한다.

## 첫 진단 실패와 구체적인 수정

v1/job104971은첫 launch 전에 `Uncompiled { path: ...device::mlp::validation_probe::upgate }`로 종료했다. Compiler image 이름은 re-export된 `stream_probe::upgate`인데 launch descriptor는원래 nested path를 참조했다. 출력 비교0개이며 M03 수치 실패로 세지 않는다.

[v2 repair](probe-v2-repair.json)는 root `stream_probe`에 실제 `#[device]` wrapper를 두고 private helper 접근용 child module을 일반 함수로 바꿨다. Production prefix와 host oracle은 v1/v2가 동일하다. 최초 소스·build/log/업로드/실패 결과를 `evidence/m03-adoption-probe-20260927-v1`과 관련 JSON에 보존했다. v2도 source/compile/CPU oracle/실제 업로드 archive를 검증했다.

## 채택 검토 범위

추가 parameter tuning을 하지 않고 기존 작은 이득을 회수하는 검증이다. 과거5개 관측과 새 clean1개에서 같은 package가 반복 개선되어5% 탐색 예산 cutoff의 예외 근거가 있다. 채택해도10위권 격차를 해결하는 주력 개선은 아니다. 공식 점수·순위·전체 모델 정확도는 이번 작업에서 측정하지 않았다.

실제 source는 E74 crate + 동결 M03 mlp이고 성능 control은 E104다. M03와 E104의46개 compile inputs 중 차이는 `src/device/shared/mlp.rs` 하나다. 새 runtime/tag/recipe/공식 제출은 이 branch에서 만들지 않았다.

```sh
python3 moa-dev/experiments/candidates/m03-producer-local-ffn/validation_results.py --check
python3 moa-dev/experiments/candidates/m03-producer-local-ffn/validation_verify.py --check
python3 moa-dev/experiments/test_structural_build.py
```

새 재현을 할 때는 기존 label을 덮어쓰지 않는다. 기본 package는 `structural_screen.py`로 재사용하고 extra는 `structural_build.py ... --package --extra`, probe는 `validation_probe.py build/submit/collect NEW-LABEL`을 쓴다. Probe target은 `/tmp/moa-m03-validation-probe/target`로 다른 probe와 격리했다.
