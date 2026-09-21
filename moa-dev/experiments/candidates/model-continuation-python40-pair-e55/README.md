# Python prompt의 최대 40-step 원본/E55 비교

실제 [job71362](../../../../docs/experiments/e55-model-continuation-python40.md)는
14번째 선택 EOS에서 정상 종료했다. 14개 선택은 원본/E55가 같았고3,670,016개 finite
logits를 비교했다. Step9에는 E55의 허용 최댓값2개가 동점인 사례가 관측됐다.
40개 완료나 전체 정확도 PASS로 세지 않는다. [수집기 검사](collector-tests.json)에는
기존63개 회귀와 Python40의40개 검사 결과가 있다.

기존 [32-checkpoint job71329](../../../../docs/experiments/e55-model-continuation.md)에서
가장 작은 허용-token margin0.875가 나온 Python prompt를 길게 이어가는 진단이다.
같은 prompt를 다시 tokenize하고 원본이 고른 token을 E55에 공급한다.
원본 EOS 또는40번째 선택에서 종료하며, E55의 다른 선택이나 조기 EOS는 기록한 뒤
원본 history를 계속 따른다. EOS와 마지막 선택은 다음 입력으로 쓰지 않는다.

[host](../model-continuation-python40-e55/probe_model_continuation.rs)는 기존 continuation
host의 schema, prompt 목록, 생성 한도와 해당 검사만 바꿨다. Async main의 Device,
모델 적재, cache, sampler, reference I/O와 비교 흐름은 byte-identical이다.
실제 prompt가 기존28token과 같으면 guard68≤128, 최대67forward·position66이다.
최대40개 분포의262144개 logits를 모두 비교하며 EOS가 빠르면 분포 수도 줄어든다.
40개 도달이나70초 완료를 가정하지 않는다.

[prepare_package.py](prepare_package.py)는 두 host의 실제47개 컴파일 입력,
기존46개 모델 입력의 보존, 실제 ELF의10개 text image와 이전 frozen 원본/E55 image의
일치를 확인한다. 기존 host 빌드를 덮어쓰지 않고 두 새 host, 동결된 Rust asset
inspector와 runner를 새 package에 복사한다. 기존 package/manifest가 있으면 중단한다.

[runner](remote_entrypoint.sh)는 같은 Arena job 안에서 다음 순서로 실행한다.

1. 실제 실행할3개 ELF와 전체 model/metadata를 해시한다.
2. 원본 `--emit reference`가 checkpoint를 쓰고 마지막에 manifest를 발행한다.
3. Manifest와 실제 존재하는 `case-0-step-0..39.json`의 hash를 남긴다.
4. E55 `--compare reference`가 reference 전체를 Device 생성 전에 검증하고 재생한다.
5. Reference 파일들을 다시 해시하고 weight stat/header·metadata hash를 확인한다.
6. 구조적 완료 marker를 출력하고 실제 종료 코드를 보존한다.

마지막 검사는 weight payload 전체를 다시 해시하지 않는다. Arena에서 다운로드하는
archive는 업로드 package이며 생성된 logits 파일은 들어 있지 않다. 수집기는 audited
host의 전원소 통계·순서·이력·출처를 확인하고, 다운로드한 배열로 오차를 독립 재계산한
것으로 표현하지 않는다. 기존71329의 Python 첫8개와 비교할 때 case index와 filename은
달라지므로 reference 파일 SHA 자체의 일치를 요구하지 않는다. 동일한 자산에서
token/history/선택/legal-choice/전원소 비교 통계의 재현 여부를 따로 기록한다.

빌드·CPU 검사 뒤 저장소 루트에서 준비한다.

```sh
python3 moa-dev/experiments/candidates/model-continuation-python40-pair-e55/prepare_package.py
```

정답률·독립적으로 자유 생성한 답변·긴 context·멀티모달·공식 성능을 검증하는 실험은
아니다. 수치 차이나 token 불일치에 임의 허용오차를 부여하지 않으며 모든 정확도
합격 판정은 null이다. 순위 목표와 공식 평가 필요성은 그대로 유지한다.
