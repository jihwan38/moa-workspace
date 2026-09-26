# E124 — bounds가 검증된 joint QKV 재개

진단 commit `f569de7` 뒤 같은 branch에서 재개한다. Source는 frozen C08 v2이며
E104는 고정 비교군이다. E122 채택 FFN과의 조합은 이 작업에 포함하지 않는다.
E105에서 입증된 K/V weight·scale padded destination DMA 결함만 고친다.
전체 QKV architecture 후보의 정확성 수리이며 새 아키텍처 가족으로 세지 않는다.

HBM input → 기존 sharded pre-RMS → 기존 leader gather/ring broadcast → 공통 BF16 TRF.
Q FP8 weight/scale은 bounds가 정상인 기존 joint-slot DMA로 적재한다.
K/V는 각각 원본 pitch에 대응하는 compact DM(4행/slice)에 HBM을 읽고,
명시적인 raw packet fetch/commit copy로 joint matrix slot2/3에 옮긴다.
BF16 scale도 compact4개/slice를 해당 slot에 복사한다. Padded view를 받는 DM→DM
TDMA로 재조립하면 같은 extent 오류가 발생할 수 있으므로 그 경로는 쓰지 않는다.

모든 joint weight와 scale이 준비된 후 한 FP8→BF16 lookup/DPE stream이
slice당16행을 계산한다. 완전한 F32 dot에 channel scale을 곱한 뒤 첫 BF16 cast를
한다. 검증된 세 output typed gather → 기존 Q/K/V RMS → Q/K RoPE → Q출력/KVcache.
입력 interface·dtype·cache 위치·반올림 경계·정확도 기준은 그대로다.

E104 대비 공통 TRF 하나, lookup setup 하나를 사용하므로 중복 준비 두 개를 없앤다.
Q/K/V weight의 필수 HBM bytes는 기존 E104와 같은30MiB이며, E105의 잘못된75MiB
emitted volume은 비교군의 정상 비용으로 취급하지 않는다. 새로 K/V compact 임시
DM write 후 packet copy read/write가 필요하다. 정상 weight direct-ingress 대비 추가
DM read+write는30MiB, scale은16KiB다. K/V weight 임시 하나는 aggregate7.5MiB,
slice당15KiB다. Joint weight는 slice당60KiB다. 실제 peak는 alias/lifetime/padding을
포함한 compile allocation으로 확인해야 하며 단순 합을 메모리 보장으로 쓰지 않는다.

통합 dot은 세 weight 준비를 모두 기다리고, Q head/RoPE를 K/V 계산과 겹치던 기존
일정의 기회를 잃는다. Copy는 Sub 작업과 DM traffic을 추가하며 lookup setup·input
TRF와 경쟁할 수 있다. 이 tradeoff를 제외하고 table 준비 감소를 speedup으로 잡지 않는다.

첫 gate는 exact compile, 모든 Q/K/V·scale HBM descriptor의 source bounds/coverage,
세 output gather byte 대응, packet-copy의 valid extent/slot offset, unchanged O/FFN
image다. 통과하면 E104→E124→E104 기본3입력 screen 한 allocation만 실행한다.
실패하면 명시된 producer/transfer/consumer 계약 오류만 고치고 원본 시도를 보존한다.
이득이 없으면 종료하며 반복·추가 입력은 root가 실측 후 결정한다.
