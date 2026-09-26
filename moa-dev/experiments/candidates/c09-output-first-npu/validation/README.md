# E127: 동결 C09의 제한적 채택 검증

2026-09-27, branch `validate/c09-adoption-20260927`, parent `6375af4`.
**세 독립 allocation에서 모두 개선되고 추가 정확도 검사를 통과해 E127로 채택한다.**
사용자가 작은 개선도 활용할 수 있음을 지적했고, E126의 조기 종료 판단을 재검토했다.
5%는 추가 parameter 탐색 예산 기준이며 채택 최소 개선율이 아니다. 동결 소스를
그대로 검증했으며 새 architecture, tile sweep, 공식 대회 제출은 아니다.
원본 [E126 판정](../decision.json)과 최초 측정은 당시 기록으로 보존한다.

| Arena allocation | E122 before | C09 O | E122 after | 앞뒤 중앙값 대비 |
|---|---:|---:|---:|---:|
| 105820 (E126 원본) | 51237 | 49923 | 51165 | −2.4960% |
| 105902 | 50903 | 48835 | 50652 | −3.8255% |
| 105909 | 51653 | 49180 | 50731 | −3.9303% |

Allocation별 paired 변화율의 중앙값은 **−3.8255%**다. 별도로 각 job 중앙값을
집계하면 candidate49180 / paired control51192 cycles로 **−3.9303%**다.
두 통계를 혼용하지 않는다. 후보의 모든 job 중앙값이 control의 모든 process 중앙값보다
작다. 세 allocation은 독립 실행이지만 동일한 기본3입력을 재사용했다.
Original7-stage batch와 새3-stage batch의 전체 포장은 다르며, 비교 대상 stage의
host/fixture/runner는 byte 동일하다. 실제 업로드 tar inventory도 각각 보존했다.
[성능 원본·재현](performance-summary.md), [독립 재계산](independent-performance-recap.json).

정확도 검증:

- 후보 기본3입력×3allocation: 총45개 표준 출력 비교 PASS.
- 추가 run3..8의6입력, Arena105893: 30개 비교 PASS. 원본 host의 RUNS/offset만
  변경했고 세 device image는 기본 C09 package와 같다. 따라서 서로 다른 표준 입력은
  합계9개다. 추가 입력 timing은 성능 집계에서 제외한다.
- Arena105952: 실제 동결 `project_output`의 pre-RMS BF16 출력, 8개 사례×3840 =
  **30,720개**가 독립 F64 oracle과 bitwise 일치(±0 동등). K255/256/4095,
  stage59/60/119와 cluster1919/1920 경계, zero, 작은 값 상쇄, channel scale 순서를
  검사했다. 조기 BF16 반올림을 넣은 반례는1920개 출력이 달라져 해당 검사의 검출력도
  확인했다. 별도 diagnostic image이며 성능·전체 kernel image 동일성 증거로 사용하지 않는다.

[추가 입력](extra-results.json), [동일 image 검사](extra-image-verification.json),
[경계 검사와 범위](probe-v1/README.md), [경계 검사 결과](probe-results.json).
Probe의 최초 package 경로 검색 오류는 [실패 기록](probe-build-attempt1.json)과
compile 당시 driver를 보존하고 재컴파일 없이 수리했다. 수치 실패가 아니다.

Source와 채택 경로:

- E122의46개 compile input 중 `src/device/sliding/projection.rs` 하나만 바뀐다.
  QKV와 M03 FFN image, public entrypoint, 입력 계약은 유지한다.
- 채택 source SHA `5692369dd5b8f6ddbeecdfaa9289a9683fae18798b920a71775362f306560fe6`,
  exact O image SHA `fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495`.
  최초 C09와 E126에서 측정한 동일 구현이다.
- `adopt.py source`는 [통합 검증](verification.json)을 재검사하고 동결 소스를 설치한다.
  이 source checkpoint에 `baseline-e127` tag를 붙인 뒤 `adopt.py recipe`가
  [E127 recipe](../../../../../baseline/e127.json)와 current pointer를 연결한다.
  `measured-build-c09-adoption.json`은 기존 exact compile/standard package의 파생 연결
  기록이며 새 compile을 주장하지 않는다. 기존 compile/package 기록은 수정하지 않는다.

검증 재실행: 저장소 root에서
`python3 moa-dev/experiments/candidates/c09-output-first-npu/validation_verify.py --check`.
이 명령은 보존한 log/tar/source/image/oracle을 확인하며 새 NPU 실행은 하지 않는다.
검사 범위는 위9개 표준 입력과8개 진단 사례다. 모델 전체·모든 입력의 동치나 공식 점수
개선을 보장하지 않는다. 작은 이득은 회수하고, 큰 구조 개선 탐색은 별도로 이어간다.
