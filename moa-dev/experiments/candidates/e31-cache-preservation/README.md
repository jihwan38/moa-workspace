# E31 KV cache의 비대상 슬롯 보존 검사

Frozen `e31-switch-gather-rms`의 **device/ops와3개 device kernel binary를 그대로 사용**하는 독립 개발 harness다. 원래 Stage1 target15 comparisons를 유지하면서, QKV가 target slot 외 cache 데이터를 훼손하지 않는지 검사한다. full host build 및 CPU checker selftest PASS이며 **NPU 실행은 아직 하지 않았다.** 이 검사는 기존 출력 정확성 검사를 대신하지 않는다.

실행할 host binary는 `../../artifacts/e31-cache-preservation/test_kernels`, SHA256 `cf7a328218c42fd010e6954df7e79bdc2cacb0e93aa19cb2758e570591330241`(3,694,192 bytes)다. fixture는 frozen E31의 `ref/fixtures.safetensors`, SHA256 `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`를 그대로 사용한다. remote entrypoint도 frozen E31 사본을 사용할 수 있다. NPU 제출은 root가 수행한다.

`src/bin/test_kernels.rs`만 바꿨다. 원래 RUNS=3, seed/position/input 생성, fixture loader·expectation, TESTS, `compare` 함수와 tolerance, Q/O/FFN kernel launch 인자는 그대로다. QKV shim에서 K/V zero initialization을 비영 finite BF16 pattern으로 대체하고 별도 before/after 검사를 추가했다. target K/V를 BF16 bits로 읽은 뒤 정확한 F32 값으로 확장하여 원래 oracle에 넘긴다. Q/O/FFN 비교는 원래대로다.

Sentinel은 기존 host PRNG의 별도 이름 `cache-preservation/{k|v}/run{run}`과 전역 element index로 생성한다. bit magnitude는 `0x3e00..0x3fff`, sign은 PRNG bit에서 가져오므로 모든 값이 유한 normal BF16이며0이 아니다. expected fixture의 출력 값에 의존하지 않는다. K/V·run·위치마다 패턴이 달라지지만 유한 BF16 종류 수 때문에 일부 값은 반복될 수 있다. 모든 possible corruption을 검출한다는 주장은 하지 않는다.

각 run의 순서는 다음과 같다.

1. K/V 각각2,097,152개 BF16(4MiB)를 sentinel로 upload한다.
2. kernel 실행 전에 실제 cache 전체를 read-back하여 host pattern과 BF16 bits를 정확히 비교한다.
3. 원래 QKV kernel을 원래 kv_offset/rope_offset으로 한 번 호출한다.
4. K/V 전체를 다시 read-back하고 target2048개를 제외한 각각2,095,104개를 bitwise 비교한다.
5. 원래 target slot만 잘라 expected K/V 비교에 넘기며 Q/O/FFN 비교도 계속 수행한다.

| run | target slot | target element 범위 | target byte 범위 |
| --- | ---: | --- | --- |
|0|739|[1,513,472,1,515,520)|[3,026,944,3,031,040)|
|1|693|[1,419,264,1,421,312)|[2,838,528,2,842,624)|
|2|552|[1,130,496,1,132,544)|[2,260,992,2,265,088)|

주소 계약은 `width=Ns*Ds=2048`, `offset=slot*width*2`, 각 slot의 first/last element는 `slot*2048`/`(slot+1)*2048-1`이다. 검사에서 제외하는 범위는 정확히 이 target뿐이며 앞쪽 `[0,start)`와 뒤쪽 `[end,total)`를 모두 검사한다. before에는 target도 포함한다. 범위·길이·offset을 assert하고, mismatch가 있으면 처음8개 coordinate/expected bits/actual bits를 출력하되 모든 element를 끝까지 센다. mismatch가 생겨도 기존15개 출력 비교를 계속 수행한 뒤 nonzero exit로 만든다.

기대되는 NPU log는 before/after×K/V×3run=12개의 `CACHE_CHECK`이며 모두 `mismatches=0`이어야 한다. before 각2,097,152개, after 각2,095,104개가 검사된다. 총 after 검사12,570,624개(25,141,248 bytes)다. 마지막 `CACHE_PRESERVATION total_mismatches=0 status=PASS`와 기존15개 `-> PASS`, 전체 exit0을 함께 확인해야 한다. 추가 upload/readback이 포함된 개발 실행이므로 cycles는 성능 판정에 사용하지 않는다.

CPU 검사에서는 **실제 추가한 순수 Rust 함수**를 `checker-selftest.rs`로 추출해 실행했다. run0–2 K/V sentinel의 모든 값이 유한·비영인지 확인하고, slot0/1/511/1023 각각의 target 첫/끝 원소를 손상시켜 after에서 제외·before에서 탐지됨을 검사했다. cache 양끝 및 target 바로 전/후의 손상도 exact 탐지했다. 총20개 의도적1-bit 손상을 정확히 셌다. 이는 checker 자체의 경계 처리 검증이며 장치 side effect 검증 결과가 아니다.

`build-manifest.json`에는49개 실제 compile input hash, 변경된 host 파일, 공식 E31 ELF와 새 ELF에 동일하게 포함된3개 device kernel의 hash/offset, build log와 checker log hash가 있다. full build는 독립 `/tmp/moa-e31-cache-preservation`와 독립 복사 target에서 `cargo furiosa-opt build --release --locked --bin test_kernels`로 수행했다. 빌드 뒤 실제 Docker source49개 hash가 사전 manifest와 같은지 재검사했다.

Raw binary/kernel/log는 `../../artifacts/e31-cache-preservation/`, host source와 최소 diff는 이 폴더의 `test_kernels.rs`, `candidate.patch`다. live main source·Git mutation·NPU는 변경하지 않았다. 추가 source는 제출용 device 구현 변경이 아니라 테스트용 host harness다.
