# E126 / frozen C09 first NPU: independent interpretation

Arena105820에서 C09 whole O는 **49,923cycles**, 동시 E122 앞뒤 control median51,201 대비 **−1,278cycles / −2.496045%**였다. 기본57개 비교가 모두 통과했고 실제 업로드 tar의 파일 해시도 screen 기록과 일치한다. 한 allocation·표준3입력의 소폭 positive 결과로 보존하되, 이번5% 탐색 기준 아래이므로 추가 sweep/repeat/채택 없이 종료하는 판단에 동의한다. 이 결과를 공식 점수나 확장 정확도 검증으로 표현하지 않는다.

## 검증 범위

`analyze_profile.py c09-first-npu-v1` 및 `--check`가 통과했다. Control22개 / C09 26개 engine marker의 전체 kind 순서와 source tensor lineage를 별도로 검증했다. 각 side3runs×2clusters를 모두 보존하여 총12개 run-cluster observation, **288개 engine interval**이 확인됐다. 각 run의 두 cluster Task duration 최댓값이 보고된 kernel cycles와 정확히 같다. 독립 raw-span 분석 결과도 `screen.py`의 저장된 per-operation observations와 일치한다.

Source 및 exact image가 원 C09 two-stage-v1과 같음을 확인했다. Candidate O image SHA256는 `fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495`다. 따라서 이번 결과는 새 pipeline 변형이 아니라 **동결 C09의 최초 하드웨어 측정**이다. 전체 benchmark host는 E122 나머지 커널을 유지하며, O-only diagnostic host와 별도로 비교했다.

분석기는 모든 event 차이를 한 launch·한 cluster의 Task clock 안에서 계산한다. 서로 다른 launch/cluster의 절대 timestamp를 빼지 않는다. 같은 run/cluster index끼리의 profile 차이는 독립 실행 사이의 상대 window 비교이며,6개 관측이6개 독립 benchmark allocation을 뜻하지 않는다.

## Whole 결과와 trace를 구분한 수치

표준 benchmark host (`FURIOSA_OPT_PROFILE=info`) 결과:

| Stage | O median | 3run cycles |
|---|---:|---|
| E122 before |51,237|52,702 /50,354 /51,237|
| C09 whole |49,923|50,813 /48,496 /49,923|
| E122 after |51,165|52,336 /49,853 /51,165|

Trace diagnostic의 아래 값은 각 side6개 run-cluster observation의 중앙값이다. Whole 결과와 같은 host/timing boundary로 취급하지 않는다.

| Window | E122 control | C09 |
|---|---:|---:|
| Task→first dot begin |28,945|17,148|
| Task→final dot end |40,953|39,542.5|
| First weight begin→final weight end |26,660|30,073|
| First weight begin→final dot end |38,679|36,802|
| First weight begin→gather end |39,865|38,006|
| Final dot end→gather begin |2|2|
| F32 canonical gather interval |1,199|1,228|
| Final dot end→Task end tail |9,615.5|9,832.5|

첫 dot은 중앙값 차이로11,797cycles 앞당겨졌지만 마지막 dot 종료의 차이는1,410.5cycles였다. 동일 run/cluster-index별 차이를 먼저 계산한 중앙값은 각각−11,638 /−1,566cycles다. 두 통계를 혼용하지 않는다. First-weight→final-dot boundary 차이는 각 관측에서−2,434..−1,444cycles, 차이의 중앙값−2,025cycles였다.

## Pipeline overlap은 실제로 있었지만 공급과 stage 비용도 늘었다

C09의 stage0/1은 각 slice의 서로 다른60행을 계산한다. Stage0 interval을 control의120행 dot과 단순 비교하여 속도 향상률을 만들지 않는다.

| C09 내부 window | Median | Range |
|---|---:|---:|
| Stage0 weight DMA |14,478|14,064–14,626|
| Stage1 weight DMA |13,390.5|12,844–13,698|
| Stage0 lookup/dot/reduce/commit |9,564|9,511–9,831|
| Stage1 lookup/dot/reduce/commit |6,724|6,719–6,731|
| Dot0∩W1 overlap |7,392.5|7,153–7,468|
| W0 end→W1 begin |2,154.5|2,063–2,680|
| Stage1 table DMA |2,044|1,956–2,615|
| Stage1 table DMA end→W1 begin |98|56–105|
| Stage1 table prepare (StoTab) |7,489|7,204–7,561|
| Dot0 end→dot1 begin |6,057|5,579–6,345|

두 dot 모두 모든 관측에서 자기 weight 완료2cycles 뒤에 시작했다. 두 번째 table 준비는 dot1보다 이미5,582–6,348cycles 앞서 끝나므로 dot1의 가장 늦은 관측 prerequisite는 W1이었다. W0 종료와 W1 시작 사이에 반복 table DMA가 위치하며, 해당 gap을 단순한 idle 시간으로 부르면 안 된다.

전체 weight 공급 envelope는26,660→30,073cycles로 길어졌다. Weight interval 합의 진단 중앙값도26,660→27,770cycles, dot interval 합도12,017→16,290cycles다. 이 합들은 resource interval 통계이며 latency나 정확한 연산량 측정이 아니다. 그럼에도 dot0와 W1이 실제로 겹쳐 complete-projection 종료 경계가 앞당겨졌다는 것이 이번 관측의 핵심이다. 더 긴 공급·stage interval과 overlap을 함께 봐야 한다.

동일60행 작업인 dot0와 dot1의 interval이9,564 vs6,724로 다르고, table1 prepare도 첫 table prepare 약5,379.5보다 길었다. 또한 **dot0 끝은 table1 prepare 끝3cycles 뒤**였으며6개 관측 모두 같았다. 이는 동시 실행 중 completion/wait 비용을 더 조사할 수 있는 구체적인 경계다. 다만 marker interval에는 dependency/queue 대기가 들어갈 수 있으므로, 이 패턴만으로 table 충돌·DM contention·DPE 자체 연산 slowdown을 확정하지 않는다. 이번 실행에는 그런 물리 원인을 가르는 resource counters가 없다.

## Static gather handoff 가설은 NPU에서 확인되지 않았다

Compiler schedule에서는 마지막 dot 종료→gather 시작이 C091,206cycles, control0cycles였다. NPU에서는 **양쪽 모두 모든6개 관측에서2cycles**였다. 따라서 정적1,206cycle handoff를 실제 병목으로 설명하거나 그것을 없애면 같은 크기의 이득이 남는다고 계산하면 안 된다.

Canonical gather 자체도 baseline과 비슷했다. 같은 run/cluster-index별 gather duration 차이 중앙값은+3cycles, range−4..+62다. M08의8B fragmentation 회귀와는 다른 실행이다. C09의 mutable-output/IndexWrite와 destination-sync graph 경계는 실제 존재하지만, 이번 trace에서 최종 dot 뒤의 큰 추가 대기는 관측되지 않았다. Destination Scratchpad→ClusterSync→gather 구조 자체는 baseline에도 존재한다.

## 종료 해석

정적+0.71% 결과와 하드웨어−2.50% 결과의 방향이 달랐다. 따라서 원래의 compile-only 판단을 C09의 하드웨어 회귀 증거로 쓰면 안 된다. 반대로 이번 단회 소폭 이득으로 pipeline 전체가 검증되었다거나 상위권에 필요한 큰 개선을 얻었다고 확대하지 않는다.

이번 frozen prototype은 “overlap이 실제로 생기며 modest positive gain을 낼 수 있다”는 선택지로 남긴다. 더 큰 pipeline 설계를 나중에 검토한다면 반복 table 공급, 동시에 실행되는 첫 dot의 completion interval, weight 공급 envelope를 함께 다뤄야 한다는 관측 근거가 생겼다. 이는 다음 실행을 요청하거나 특정 수정의 이득을 약속하는 결론이 아니다. 이번 C09에는 추가 반복·튜닝·채택을 하지 않는다.

근거 파일:

- [원본 수집 결과](e126-c09-first-npu-screen-job-105820-results.json)
- [전체 profile 분석](profile-analysis-c09-first-npu-v1.json)
- [Source/marker 대응](marker-source-map-c09-first-npu-v1.json)
- [재현 분석기](analyze_profile.py)

분석기 SHA256 `f72432472cf1a69efb0343eaa3391f3e195268b6b7450a22f35d5aa3fc80e5e1`; analysis JSON SHA256 `5b78c1a68d7ec2177011f4e62a348068589dc8e8971c7881ed32f29ee5df0b57`; marker-map SHA256 `6aa2a0cb9cc61c1b0db9ec80cb9c4c45913764f956a913d6e091854f82e74804`.
