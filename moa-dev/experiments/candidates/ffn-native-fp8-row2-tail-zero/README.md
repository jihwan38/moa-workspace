# Row2 + tail-only zero — E40 NPU FAIL, 원인 미확정

2026-09-21. 공통 ancestor에서 이미 검증한 두 source diff만 결합했다. 추가 최적화나 수동 conflict resolution은 없다. 첫 exact FFN compile이 통과했고 정적 makespan은 **173,016 cycles**다. **E40 job70552는 FFN run0/1/2 모두 `non-finite device output at 0`으로 실패했다.** QKV96455/O45927은 통과했지만 FFN 보고값432862는 유효한 최적화 성능이 아니며 전체 score도 없다. [공식 harness 실행 기록](../../../../docs/experiments/e40-row2-tail-zero.json)과 [physical-layout audit](physical-layout-audit.md)에 실패와 조사 범위를 기록했다.

| Source | SHA256 |
| --- | --- |
| 공통 ancestor: direct weight commit + direct scale copy | `7c363a7f116ff9ebf4027f6a9957faa5cb3ce74aac85b3c5f7b6579ff628b267` |
| [Row2 parent](../ffn-native-fp8-full256-row-two/README.md) | `bf5e8c153ec6b226136cf0a9887267fedefd562a67b74ef34d47429960dbf602` |
| [Tail-only zero parent](../ffn-native-fp8-tail-zero/README.md) | `07910fa5b11bd23c7da97c7c31c431db6629fe7db95cead2eaf5cdda482f1514` |
| **통합 mlp.rs** | **`28cf2313c22e5065c0e32a09e44dc28f7375991a31602d20e898a6fca25e087f`** |

`diff3 -m mlp.row2.rs mlp.ancestor.rs mlp.tail-zero.rs`가 exit0으로 출력한 결과가 [mlp.rs](mlp.rs)와 byte-identical하다. 세 입력 source도 같은 폴더에 동결했다. [candidate.patch](candidate.patch)는 공통 ancestor 대비 전체 diff다.

## 결합한 두 변경

Row2는 `L=7680*c+60*g+2*r+s`(`c2,g128,r30,s2`)로 HBM weights/scales를 읽는다. Physical slice는 `2*g+s`이며 256개 모두 실제30행을 갖는다. Output은 Switch로 같은 group의 인접 두 slices를 모으고,2행×2slice의 BF16 네 개를8B packet으로 commit한다. 60개의 연속 결과를 기존 UpGateRows로 typed gather한다. Padding을 행 데이터로 재해석하지 않는다.

Tail-only zero는 각 weight/scale dense32 buffer의 첫30행을 실제 값으로 직접 쓰고, 끝2행은 별도의 실제2-row zero tensor를 초기화한 다음 Sub fetch/collect/commit_view로 쓴다. 네 buffer 모두 같은 절차를 쓴다. 전체32행을 먼저0으로 덮던 작업을 줄이지만 모든 cell을 정의한다.

| Slice당 FP8 buffer | 실제30행 write 범위 | 실제 zero tail 범위 |
| --- | ---: | ---: |
| Weight | byte0..115,199 | byte115,200..122,879 |
| Scale | byte0..7,199 | byte7,200..7,679 |

두 write 범위는 겹치지 않으며 전체32행을 빠짐없이 채운다. Global row mapping이 바뀌어도 local row30/31은 실제0이고 output에서는 읽지 않는다.

## Source·좌표·schedule 검증

[verify_combination.py](verify_combination.py)를 실행한 [verification.json](verification.json)은 다음을 확인한다.

- SHA가 고정된 두 parent와 ancestor의 clean3way merge 출력과 candidate가 완전히 동일.
- DPE helper 전체, `feedforward`부터 EOF까지, 모든 cast 수가 ancestor와 동일. 추가 dtype 변환0개.
- Output compact helper는 row2 parent와 byte-identical.
- 15,360행 bijection, canonical row60 group 순서,4-BF16 packet 순서.
- 네 buffer의 실제2-row zero source memset·tail copy·real30 write가 존재하며 모든 byte가 정확히 한 번 정의됨.
- **각 consumer의 compiler dependency ancestry에 zero memset, tail copy, real30 write가 모두 포함**되고 두 write 완료 뒤 소비를 시작함.

| Buffer | zero memset 완료 | tail copy 완료 | real30 write 완료 | 첫 consumer 시작 |
| --- | ---: | ---: | ---: | ---: |
| Up weight | 4,363 | 60,360 | 82,455 | 91,621 |
| Up scale | 1,329 | 4,827 | 84,656 | 99,585 |
| Gate weight | 6,053 | 101,091 | 123,346 | 125,414 |
| Gate scale | 1,655 | 92,085 | 121,178 | 123,951 |

이는 source·integer·compiler schedule 검증이다. Native FP8 hi/lo 근사와 block partial의 BF16 반올림은 parent에서 그대로 물려받았으므로 모든 입력에 대한 동치성을 주장하지 않는다. E35/E38의 NPU 통과도 새 mapping/initialization 조합의 검증을 대신하지 않는다.

## 같은 비교 조건의 정적 결과

| 후보 | makespan | Main union | Sub union | DMA union |
| --- | ---: | ---: | ---: | ---: |
| 공통 direct-commit ancestor | 191,836 | 115,325 | 54,281 | 145,727 |
| Row2만 | 182,868 | 99,061 | 54,281 | 144,055 |
| Tail-only zero만 | 175,459 | 115,325 | 26,905 | 145,727 |
| **결합** | **173,016** | **99,061** | **26,905** | **144,055** |

결합은 ancestor 대비18,820(-9.81%), row2 대비9,852(-5.39%), tail-only 대비2,443(-1.39%) cycles 감소다. Parent의 개별 makespan 감소를 단순 합한 값이 아니다. Main/Sub 작업량 절감은 각각 보존됐지만 의존성·overlap·sync 배치가 달라졌다. Core의 긴 reuse/sync lifetime 합도 달라지며 이를 CPU 산술 시간으로 읽지 않는다.

기존 [E36 safe BF16 FFN](../ffn-parallel-epilogue/README.md)의162,209와 비교하면 이 후보는 여전히 정적 우위가 없다. E40 실행이 정확성에 실패했으므로 현재 후보를 채택하지 않는다.

Scratch는 `/tmp/moa-ffn-row2-tail-zero`, compiler 명령은 `cargo furiosa-opt compile ops::decoder_feedforward --exact`였다. 전용 target/cache를 사용했고 compile1회/오류0회다. [evidence.json](evidence.json)에 source manifest와 schedule SHA를, [candidate.json](candidate.json)에 전체 schedule을 보존했다. Parent schedule들도 이 폴더에 함께 동결했다.

실제 적용 파일은 `mlp.rs`다. `ops.rs`와 `residual.snapshot.rs`는 exact 비교 환경의 참고본이며 최신 다른 커널을 덮어쓰기 위한 파일이 아니다. Live source·Git·NPU를 변경하지 않았다.
