# E55: 원본 토큰 이력에서 32개 후속 분포 비교

2026-09-22 KST. Arena **job71329**는 42.448초에 exit0으로 완료됐다.
네 prompt의 각8개 checkpoint에서 원본과 E55의 실제 greedy 선택이 **32/32 동일**했고,
전체 **8,388,608개 logits**가 finite였다. E55의 수치 차이는 남아 있으며
정확도 허용기준·합격 판정은 `null`이다. [수집 결과](e55-model-continuation.json).

| Prompt | Checkpoint / 종료 | 전체 checkpoint의 RMSE 범위 | 최대 절대 차이 | 최소 허용-token margin 원본 / E55 |
|---|---|---:|---:|---:|
| `Hi` | 8 / 길이 제한 | 0.146925–0.512598 | 1.6875 | 10.25 / 10.3125 |
| `What is 2 + 2?` | 8 / EOS | 0.074530–0.426843 | 1.0 | 2.5625 / 2.5625 |
| `대한민국의 수도는?` | 8 / 길이 제한 | 0.070894–0.228469 | 0.75 | 4.125 / 3.59375 |
| Python `len([1, 2, 3])` prompt | 8 / 길이 제한 | 0.064635–0.207814 | 0.65625 | 0.875 / 0.875 |

각 값은 동일 prompt의8개 checkpoint에서 구한 범위 또는 극값이다.
전체 평균 RMSE나8개의 독립 prompt를 뜻하지 않는다. 최소 margin0.875가 관측됐지만
동점에 매우 가까운 선택은 만나지 않았다. 산수 case의 마지막 선택은 양쪽 모두
EOS106이었다. EOS를 제외한 completion 수는 `[8,7,8,8]`이며 EOS와 각 case의
마지막 선택 token은 다시 feed하지 않았다.

[host](../../moa-dev/experiments/candidates/model-continuation-e55/probe_model_continuation.rs)는
실제 tokenizer와 chat template으로 prompt14/21/19/28개 token을 얻었다.
공개 `sampling::sample`에 `temperature=1, top_k=1, top_p=1`을 적용하고
EOI/EOA만 suppress했다. EOS는 선택 가능하며, 동점 규칙은 공개 sampler의
`total_cmp` 및 unstable 순서를 그대로 사용한다. CLI의 기본 확률적 sampling과는 다르다.

각 prompt는 새 Workspace/Decode를 사용하고, 해당 prompt의 prefill과 continuation은
같은 cache를 유지한다. E55에는 매번 **원본이 선택한 token**을 다음 입력으로 공급한다.
후보의 다른 선택이나 조기 EOS가 나타나도 원본 이력을 따라 비교하도록 구현했다.
이번 실행에서는 그런 불일치가 없었다. 모델당110회 forward와32회 logits head가
실행됐으며 최대 position34라512-page/1024-ring 경계에는 도달하지 않았다.

각 host는 기존46개 source/Cargo/toolchain 입력에 동일한 진단 host만 추가했다.
[빌드 증거](../../moa-dev/experiments/candidates/model-continuation-e55/evidence.json)에서
실제47개 컴파일 입력과10개 embedded kernel image를 frozen 원본/E55와 각각 대조했다.
따라서 관측을 위해 모델 계산 코드를 바꾸지 않았다. 실제 공개 sampler를 포함한
CPU helper 검사11개도 통과했다. 이 검사는 Device/cache 실행이나 전체 host I/O의
모든 실패 경로를 포괄하지 않는다.

[같은-job runner](../../moa-dev/experiments/candidates/model-continuation-pair-e55/remote_entrypoint.sh)는
먼저 전체 weight SHA `7c2ee232…8ce704b` 및 metadata를 확인하고, 원본이 기록한
32개 reference 파일과 마지막에 발행한 manifest를 E55에 넘겼다. E55는 모든 배열·파일
구조와 token ID를 Device 생성 전에 검증했다. 실행 전후33개 reference 파일의 SHA가
동일했고, 종료 뒤 weight stat/header 및 metadata hash도 동일했다. 종료 뒤 weight
payload 전체를 다시 해시한 것은 아니다. Manifest SHA는
`fbb73037ecd56565e09cbbd6981d7ebe11847e6d01c93d0a71ffbae02d084e3d`다.

[수집기](../../moa-dev/experiments/collect_model_continuation.py)는 실제 다운로드한
업로드 archive의4개 파일과 `job.yaml`, 실행된3개 ELF hash, 원본/E55의 모든
checkpoint 순서·소비 token 이력·EOS·카운터·통계와 최종 완료를 검사했다.
다운로드 archive에는 생성된 reference 배열이 들어 있지 않으므로 로컬에서 배열 오차를
독립 재계산했다고 주장하지 않는다. [수집기 검사](../../moa-dev/experiments/candidates/model-continuation-pair-e55/collector-tests.json)는
합성·변형56개와 실제 양성1개·실제 로그 변형6개, 총63개를 통과했다.

기록된 host 전체 시간은 원본24.537초, E55 10.949초다. 원본의 reference JSON 쓰기와
E55의 파싱·비교 작업이 다르고, 실행 순서도 원본→E55 한 번뿐이다. 이를 모델 처리량
배율이나 공식 점수로 사용하지 않는다. 첫 checkpoint는 기존4-prefix 결과와 같은
RMSE를 보이며, 이번 결과는 그 이후28개 non-EOS token을 입력하는 추가
forward에서 이어지는 분포까지 관측 범위를 넓혔다. 추가 forward에는 EOS를 입력하지
않으며, 산수 case도 EOS를 선택하기 전7개 non-EOS token을 모두 입력했다.

E55를 기준 구현으로 유지한다. 이 결과는 정답이 있는 독립 평가, 더 다양한 prompt,
긴 context, Workspace 재사용 요청, multimodal 입력의 검증을 대신하지 않는다.
공식 제출·점수·순위는 아직 없다. [설계와 소스 추적](../research/model-continuation-diagnostic-design.md),
[재현 패키지](../../moa-dev/experiments/candidates/model-continuation-pair-e55/README.md).

다음 실험은 가장 작은 margin이 나온 Python prompt 하나를 최대40 checkpoint로
연장하는 원본/E55 pair다. 기존28-token prompt가 같다면 최대67forward·position66이며
원본 EOS가 먼저 나오면 거기서 정상 종료한다. 첫8개 checkpoint를 기존 결과와 대조하고
그 이후의 선택·margin·분포를 관측한다. 새 host 구현과 실행은 아직 하지 않았다.
현재 측정한 적재·workspace·decode/read·JSON 비용으로 조립한 예상은 약33초이나
다른 실행의 시간 보장은 아니다. 512-token 문맥은 원본 prefill만 단순 외삽 약66초여서
같은70초 pair에 포함하지 않는다. 임의로 position을 건너뛴 빈 cache 검사는 실제 긴
문맥의 대조를 대신하지 못한다.
