E55의 **공식 제출 CLI 패키징을 로컬에서 실제 실행해 확인한 기록**이다. 설치된 `moa-submitter 1.2.0`이 만든 요청의 허용 소스23개/160,829B가 NPU에서 검증한 E55와 모두 일치한다. 공식 서버에는 제출하지 않았고, 공식 contract·인증·점수 검사는 수행하지 않았다.

[preflight.py](preflight.py)는 기존 Docker 컨테이너 안에서 loopback `127.0.0.1`의 임시 HTTP 수신기를 띄운다. CLI에는 명시적 `--server http://127.0.0.1:<port>`, 일회용 합성 `MOA_TOKEN`, 별도 빈 config 경로를 전달한다. 저장된 인증 파일을 읽거나 쓰지 않고 proxy 환경변수도 자식 프로세스에서 제거한다. 수신기는 정확한 요청을 저장한 뒤 의도적으로 HTTP400과 `LOCAL_PAYLOAD_CAPTURED_NO_OFFICIAL_SUBMISSION`을 반환한다. 따라서 CLI의 nonzero 종료는 이 검사에서 예상한 결과이며, 승인된 submission ID를 만들어내지 않는다.

검사 경로는 다음과 같다.

```text
installed moa-submitter
→ load_token: 합성 환경변수만 사용
→ find_repository: src/ops.rs가 있는 크레이트 검색
→ collect_sources: ops.rs + src/device/의 실제 파일, 경로 정렬
→ 각 파일을 base64 JSON으로 직렬화
→ loopback GET /api/health → POST /api/submissions
→ 요청의 경로·중복·바이트를 원본 소스와 대조하고 HTTP400 반환
→ 실제 요청을 동결 E55 provenance와 별도로 대조
```

| 호출 조건 | 관측 결과 |
|---|---|
| `/workspace`에서 기본 경로 검색 | 정확한23파일 요청, 예상 HTTP400 |
| `/workspace/src/device/sliding`에서 상위 검색 | 같은 요청 바이트, 예상 HTTP400 |
| 빈 작업 폴더에서 `--source /workspace` | 같은 요청 바이트, 예상 HTTP400 |
| 빈 작업 폴더에서 `--source` 생략 | 크레이트 검색 실패, HTTP 요청0개 |

세 POST body는215,789B, SHA256 `f21ae29d56d73a47b10c0613b5e01bc553ee0344b3cc526d88323fb56befd189`로 동일하다. Body 안에는 base64 소스와 경로만 있다. 바이너리·fixture·모델 가중치·`moa-dev/`·인증 정보는 포함하지 않는다. 테스트 전후23개 소스가 같고, 별도 config 디렉터리도 생성되지 않았다. [결과와 파일별 해시](../../../../docs/experiments/e55-official-submit-preflight.json), [측정 소스 provenance](../../../../docs/experiments/e55-fused-qkv-provenance.json), [원시 요청·로그](../../artifacts/official-submit-preflight-e55).

이 검사를 택한 이유는 공식 클라이언트가 `--dry-run`을 제공하지 않고, `submit`의 첫 단계에서 인증 토큰부터 읽기 때문이다. 토큰이 없을 때의 오류만으로는 현재 작업 폴더에서 실제 전송할 파일을 확인할 수 없었다. 클라이언트 코드는 수정하지 않고 로컬 수신기로 전송 경로를 실행했다. `.git` 위치에 의존하지 않으므로 상위 `moa-workspace/`가 Git 루트이고 Docker에는 하위 crate만 `/workspace`로 보이는 현재 구조와도 맞는다. [공식 CLI 구현](https://github.com/micro2026-moa/moa-submitter-cli/blob/ccae5a0cd35b5cd6a9fe985f7e8f7052e29d164a/src/main.rs)의 `find_repository`417–450, `collect_sources`457–504, `walk_device`523–552, `submit`568–610을 설치된 동일 버전 소스와 대조했다.

재현은 실행 중인 개발 컨테이너와 새 출력 경로를 사용한다. 이 명령은 공식 제출 명령이 아니다.

```bash
docker cp moa-dev/experiments/candidates/official-submit-preflight-e55/preflight.py \
  moa2026-dev:/tmp/moa-official-submit-preflight-e55.py
docker exec moa2026-dev python /tmp/moa-official-submit-preflight-e55.py \
  --source /workspace --output /tmp/moa-official-submit-preflight-e55-new
```

스크립트는 출력 디렉터리를 덮어쓰지 않는다. 새 실행은 **그 시점의 현재 소스**를 검사하므로 기존 E55와 같다고 주장하려면 보고서의23개 해시를 동결 provenance와 다시 대조해야 한다. 기록된 E55는 Git commit `d50eb159d9a260c411f2ebdc96adc5e0dc82afcb`다. 공식 평가가 재개되고 실제 인증을 완료한 뒤에는 새 평가 계약과 업로드 직전 소스를 재확인해야 한다. 이 로컬 PASS로 서버의 build/correctness/rank 통과를 대신하지 않는다.
