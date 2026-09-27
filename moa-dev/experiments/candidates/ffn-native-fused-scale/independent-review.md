# E128 채택 전 독립 검토

2026-09-27. 동결 source·실제 compiled graph·host/image·업로드 archive·원시 로그와 root의 live source 복사 후 상태를 읽기 전용으로 대조했다. 검토자가 compile/Arena/공식 제출을 실행하거나 동결 verifier·package·결과를 수정하지 않았다. 차단할 구현·provenance 오류를 발견하지 못했다.

## Source와 실행 경로

- Live `furiosa-opt-gemma4-12B`의46개 compile 입력이 `build-e128-six-row-v1-repro.json`과 모두 일치한다. MLP는28,283B / SHA256 `3b94e94376d6a933c613102daa64501b7d802563758034255ef9fddb4864de8d`로 candidate snapshot, exact compile crate, evidence source와 같다.
- `baseline-e127` 대비 MLP의 변경 범위는 상단 설명과 `project_rows_with_scale_contraction`이다. `project_up_and_gate`부터 파일 끝까지 byte-identical이며 helper 앞의 선언도 상단 설명을 제외하면 같다. 따라서 실제 quantizer·FP4→E5M2 weight materialization·up/gate 호출·local gather/BF16·GeGLU·down·epilogue는 보존됐다. `src/ops.rs`를 포함한 다른 compile 입력도 E127과 일치한다.
- 새 helper는 기존 native16-element block dot/F32 term 합 이후 `x_scale Mul0 → block_scale Mul1 → H reduce`를 연결한다. 각 projection의30행을5×6행으로 계산하고 F32 결과를 기존 row30 destination의 서로 다른 범위에 쓴다. Public interface와 첫 BF16 경계를 옮기지 않는다.
- 별도 probe compile 입력은 이 전체 MLP 파일에 diagnostic child wrapper만 붙인 정확한 prefix다. Wrapper는 실제 `prepare_ffn_fp8_split → project_up_and_gate → 기존 local gather/BF16`을 호출한다. Production image와 다른 diagnostic image이며 성능 측정에 포함하지 않는다.

## Emitted graph와 범위

[graph-verification.json](graph-verification.json)과 read-only `verify_graph.py e128-six-row-v1 --check` PASS를 재확인했다. 열 개 실제 TuExec에 native operation unit, 두 개 구분된 VRF operand의 Mul0/Mul1, H reduce, F32 transpose/8B commit이 함께 있다. Up/gate F32 block-partial DM 및 별도 scale/reduce command가 없고, native partial을 VRF로 옮기는 우회도 없다.

실제 IndexAccess 상수와 fetch/commit sequencer를 결합해 slice별 weight offset0/23040/46080/69120/92160B(각23040B), scale offset0/1440/2880/4320/5760B(각1440B), 출력 offset0/24/48/72/96B(각24B)의 전수 coverage·비중첩을 검사했다. Scale은6×240 F32의5,760B VRF이고 native stream의60×4 block 순서와 맞는다. Compiler는 source의 두 activation reload를 하나의 TRF 및 x-scale VRF producer로 합쳤다. 실제 lifetime이 양 projection을 덮고, 같은 주소를 사용하는 block-scale VRF들의 수명은 겹치지 않는다.

Graph 검사는 공개된 logical VRF mapping과 두 VrfId까지 다룬다. Binary vISA read-indexer bitfield를 독립 해독한 것은 아니며 아래 NPU probe가 이 수치 경계의 실행 증거다. HBM 전체 ingress는 변경되지 않은 source/manifest에 묶였고 이 checker에서 새로 전수 packet 열거하지 않았다.

## 실측과 정확도 주장의 범위

| 독립 Arena allocation | Candidate FFN median | E127 앞/뒤 median | Paired 변화율 |
|---|---:|---|---:|
|108339|329451|332115 /332020|−0.787942%|
|108342|329159|331449 /331222|−0.656887%|
|108343|329097|330548 /331329|−0.556448%|

각 allocation 안의 앞/뒤 control 평균(두 값의 median)에 대한 비율을 다시 계산했다. Paired 변화율 중앙값은 **−0.656887%**다. 세 allocation의 실제 host·fixture·runner와 outer package는 같고 QKV/O image도 E127과 같다. QKV/O의 관측 시간 변화를 이 후보의 최적화 효과로 계산하지 않는다. Static FFN은154440→155190, **+0.485625%**로 늘었으며 이 정적 결과와 NPU 감소율은 서로 다른 관측이다.

표준 세 입력을 세 allocation에서 반복해 candidate45비교(FFN9비교)가 PASS했다. 이는 서로 다른 표준 입력9개가 아니다. 별도 job108344의 run3…8 추가6입력에서30비교가 PASS했고 같은 production3images임을 확인했다. 따라서 서로 다른 표준 입력은 **3+6=9개**다. Extra 타이밍은 성능 집계에서 제외한다. Extra 결과의 actual tar member hash 및 raw log의30PASS/실행 ID도 독립 재확인했다.

Job108341은 **8개 별도 boundary case × up/gate ×15360행 =245760개 출력**을 BF16 bitwise로 통과했다(±0 동등, nonfinite0). 실제 host activation fingerprint와 독립 E5M2 모델, CPU oracle/원시 NPU 로그·실제 업로드 archive를 재검증했다. 이8case는 표준 입력 수에 합산하지 않는다. K15/16·63/64·3839, row5/6·29/30·cluster 경계, 별도 up/gate scales, 실제 x-scale2^-7 및 상쇄/작은 값을 검사한 제한된 증거다. General BF16/full-model 동치나 기존 세 항 입력 근사의 제거를 증명하지 않는다.

재확인한 read-only 명령은 `verify_graph.py e128-six-row-v1 --check`, `validation_performance.py --check`, `validation_extra.py verify`, `probe.py --check`, `validation/probe_model_binding.py --check`이며 모두 PASS다.

## 채택 판단에 적용할 범위

동결 후보의 작은 FFN 이득은 세 독립 allocation에서 같은 방향으로 재현됐고 별도 정확도·source/host/image 검증도 통과했다. 제한적 채택을 지지하며 추가 tile tuning을 요구하지 않는다. 제거된 partial write/read 각각29,491,200B는 논리적 데이터량이며 측정된 bus traffic 또는 전체 FFN 병목 비중이 아니다. 이 결과를 whole-FFN architecture 개선이나 상위권 진입 규모의 개선으로 부르지 않는다.

5%는 추가 parameter tuning 예산 기준으로만 적용한다. 기존 performance/screen report의 `adopted:false`는 검증 시점의 기록으로 보존하고 root의 별도 source-build/adoption 기록이 최종 상태를 선언한다. 이 검토 시점에는 새 E128 공식 제출·공식 점수·순위 개선을 확인하지 않았다. Root의 source commit/tag/recipe 전환은 별도의 채택 기록으로 확인해야 한다.
