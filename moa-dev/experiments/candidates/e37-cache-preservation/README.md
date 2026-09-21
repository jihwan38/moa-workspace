# E37 KV cache 비대상 슬롯 보존 검사

Frozen `e37-kernel-specific-rms`에 E31의 cache sentinel host 계측을 그대로 붙인 독립 harness다. **전체 build PASS, 세 device image가 frozen E37 ELF와 byte-identical**임을 검증했다. Root가 제출한 **[job70589](../../../../docs/experiments/e37-cache-preservation-correctness.json)가 exit0으로 완료됐고 원래15개 출력 비교와 cache guard12개가 모두 PASS**했다. Kernel 호출 후 비대상 cache **12,570,624개 BF16 bits**가 전부 일치했고 mismatch는0이었다. 이는 E37 device image를 직접 실행한 결과이며 E31 job70507의 관측을 대신 인용한 것이 아니다.

제출 패키지는 `../../artifacts/e37-cache-preservation-package/`다. 실행 파일 `test_runtime` SHA256은 `aca22dfb587b301cb0767a5fef30a00ed7cd171e69b75a1d1e3328634d1ae279`(3,688,624B)이다. fixture는 E37 원본 SHA `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`, remote entrypoint도 원본 그대로다. 추가 cache upload/readback을 하는 진단 harness이므로 **cycles는 공식 성능·score·rank 판정에 사용하지 않는다.**

## 원본 유지와 빌드 증명

기준 스냅샷은 `../../artifacts/e37-kernel-specific-rms/`다. 원본 `src/bin/test_kernels.rs`가 E31 원본과 동일(SHA `42db4e99ae788020726d530d1cd799e647779b7b9912e593401d6981cffc7516`)함을 먼저 확인했다. 따라서 E31 guard `test_kernels.rs`를 byte-identical하게 재사용했다(SHA `d724d470f30a394fd00f4699d9b23d1f14185d83c62816686bc44c06320e5a3a`).

실제 compile 입력49개 중 변경 파일은 이 host 파일 하나뿐이다. RUNS=3, 원본 seed/position/input 생성, fixture expectation, `compare` 함수와 tolerance, Q/O/FFN kernel launch 인자는 바꾸지 않았다. before sentinel upload/검사와 after 비대상 cache 검사만 추가하고, 원래 target K/V는 BF16 bits를 정확한 F32로 확장해 원래 oracle에 넘긴다. guard source와 최소 diff는 `test_kernels.rs`, `candidate.patch`에 있다.

`verify_images.py`는 실행 파일을 **실행하지 않고** ELF64의 `furiosa_kernels` registry를 파싱한다. kernel 이름·specialization key·image length와 실제 image bytes를 원본 E37 ELF(SHA `5527080606282675b04ad56deae87b3efd704b8e043b260e9dfbd3127c604035`)와 비교했다.

| kernel | bytes | SHA256 |
|---|---:|---|
| QKV |187,648|`91b232f0179b37b126f2ba8755b6125e532364418516948fd98dadcc9d7e77ea`|
| O |88,576|`e9d3402c62fd866f417053e618e7a7af1d5cb60485c04cc292f3a702f5996cde`|
| FFN |204,032|`61ad0c50a03902de9208447951086df930669ded080c94e4c637bffac3c414b6`|

세 image가 모두 정확히 같다. 결과는 `kernel-byte-identity.json`, compile 입력은 `compiled-input-hashes.json`이며 빌드 후 실제 Docker 입력49개를 다시 hash해 불변도 확인했다. `build-manifest.json`에 원본/결과 binary, 입력, build log, identity 근거를 묶었다. E31에서 이미 통과한 순수 Rust checker selftest source도 byte-identical하게 보존했다. 이번 준비에서는 같은 CPU selftest를 재실행하지 않았고, 기존 실행 log/hash를 manifest에 연결했다.

## 검사 범위

각 K/V cache는2,097,152개 BF16이다. 별도 PRNG namespace `cache-preservation/{k|v}/run{run}`와 element index로 생성한 유한·비영 normal BF16 sentinel을 쓴다. 기대 출력 값에 의존하지 않는다. BF16 종류가 유한하므로 일부 값은 반복될 수 있고 모든 가능한 손상을 탐지한다는 보장은 없다.

각 run에서 cache 전체 sentinel을 upload하고 실제 NPU contents 전체를 read-back해 먼저 비교한다. 원래 QKV를 원래 offset으로 호출한 후 다시 전체를 읽고 target2048개를 제외한 양쪽 영역을 BF16 bits로 비교한다. target Q/K/V 정확도와 O/FFN의 원래15개 출력 비교도 계속 수행한다.

| run | target slot | 제외하는 element 범위 | 제외하는 byte 범위 |
|---|---:|---|---|
|0|739|[1513472,1515520)|[3026944,3031040)|
|1|693|[1419264,1421312)|[2838528,2842624)|
|2|552|[1130496,1132544)|[2260992,2265088)|

before에는 target도 포함한다. after에 제외하는 범위는 정확히 target 하나다. 길이·slot·offset bounds를 assert하며 mismatch가 생겨도 전체를 세고 원래 출력 비교를 진행한 뒤 nonzero exit를 반환한다.

통과 판정에는 아래 조건이 모두 필요하다.

- 원래15개 비교 PASS와 전체 exit0.
- before/after×K/V×3run=`CACHE_CHECK`12개가 정확히 한 번씩 존재하고 mismatch0.
- before 각2,097,152개, after 각2,095,104개 검사. after 합계 **12,570,624개 BF16**.
- 마지막 `CACHE_PRESERVATION total_mismatches=0 status=PASS`가 정확히 한 번 존재.
- Arena에서 다시 다운로드한 실제 package의 binary/fixture/runner hash가 동결 package와 일치.

`collect_job.py`는 job70589의 status/log/download만 읽는다. submit/rerun 기능이 없고 새 NPU job을 만들지 않는다. `moa-dev/measure.py`의 원래15개 출력 비교 parser를 그대로 재사용하며 위12개 guard와 실제주소/개수도 별도로 검증한다. 결과는 `docs/experiments/e37-cache-preservation-correctness.json`, raw log는 `moa-dev/logs/e37-cache-preservation-job-70589.log`에 남겼다. 원본 로그의12개 guard coordinate/range/검사 개수와 마지막 mismatch0을 확인했고, 실제 업로드 tar의 binary/fixture/runner hash도 모두 동결 package와 같았다. 이 결과는3개 공식 입력 구성과 해당 sentinel 패턴에 대한 관측이며 모든 입력·모든 손상 형태의 보장을 뜻하지 않는다.

## 재현

E37 스냅샷의 src/Cargo/toolchain/ref/scripts를 Docker의 별도 `/tmp/moa-e37-cache-preservation`에 복사하고 host파일만 이 폴더의 `test_kernels.rs`로 대체한다. dependency target은 `/tmp/moa-e31-cache-preservation/target`을 `cp -a --reflink=auto`로 독립 복사했다. 공유 target을 수정하지 않았으며 fresh-target build라고 주장하지 않는다.

```bash
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-e37-cache-preservation; export CARGO_TARGET_DIR=/tmp/moa-e37-cache-preservation/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-e37-cache-preservation/target/furiosa-opt/kernel; cargo furiosa-opt build --release --locked --bin test_kernels 2>&1 | tee build.log'
python3 moa-dev/experiments/candidates/e37-cache-preservation/verify_images.py moa-dev/experiments/artifacts/e37-kernel-specific-rms/target/release/test_kernels moa-dev/experiments/artifacts/e37-cache-preservation/test_kernels
```

기존 고정toolchain/compiler0.8.1을 사용했고 첫 build가 성공했다. 빌드 log·binary·세 kernel image는 `../../artifacts/e37-cache-preservation/`에 있다. 이 subagent는 live source/Git/NPU 제출을 바꾸지 않았으며 root가 검증된 패키지로job70589를 제출했다. 확인일2026-09-21.
