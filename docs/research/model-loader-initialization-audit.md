# 모델 초기화·가중치 적재 경로 감사

2026-09-22 KST. E55의 `gemma4` 실행이 첫 모델 적재 로그 전에 종료된 구간을 실제 소스로 분해했다. **아래 호출 수와 byte 수는 코드·고정 shape에서 계산한 값이며 시간·전송량 실측이 아니다.** 장치 초기화, 파일 접근, host 복사, 메모리 할당, 전송 완료 대기 중 어느 단계가 지연됐는지는 아직 확인하지 못했다.

## 관측과 첫 로그의 위치

[Arena job71173 결과](../experiments/e55-arena-model-smoke.json)는 원본 CLI를 변경하지 않은 E55 host, prompt `Hi`, max-new-tokens=1의 실행이다. Scheduler는 70.093479245초 뒤 `timeout`, exit code−1로 종료했다. 보존 로그에는 wrapper의 `MOA_E55_MODEL_SMOKE_BEGIN`과 scheduler 종료 문구만 있고 `model loaded`나 생성 token은 없다. 따라서 모델 정확도·추론 latency·throughput 결과는 아니다.

[src/bin/gemma4.rs](../../furiosa-opt-gemma4-12B/src/bin/gemma4.rs)의 실제 순서는 다음과 같다.

| 위치 | 동작 | 첫 `model loaded` 이전 여부 |
|---|---|---|
| 50–55행 | argument와 max-new-tokens 해석 | 이전 |
| 57행 | `Device::new(ops::embed_token.topology())` | 이전 |
| 58행 | `load_start = Instant::now()` | 이전. **Device::new 시간은 이 타이머 밖** |
| 59행 | `load::load_model(...).await` | 이전 |
| 60행 | `eprintln!("model loaded ...")` | 첫 모델 적재 완료 로그 |
| 62–75행 | 선택적 이미지 변환, tokenizer/chat encoding, `Workspace::new` | 이후 |
| 99행 이후 | `generate`, model kernel launch, sampling | 이후 |

그러므로 first-log 부재로 `Device::new` 완료 여부와 `load_model` 진행 위치를 구분할 수 없다. Tokenizer나 Workspace의 비용은 이 CLI에서 해당 로그 뒤에 있다. 적재 함수에는 Gemma projection/FFN kernel launch가 없으므로 이 결과를 QKV·FFN 계산 timeout으로 해석하지 않는다.

모델 파일 부재와도 구분한다. 앞선 [job71164](../experiments/e55-arena-model-assets.json)는 Arena가 설정한 `RNGD_MODEL_DIR` 아래 `gemma-4-12b-it-NVFP4`에 모델·tokenizer·metadata/template 네 파일이 readable regular file임을 확인했다. 로컬 자산 부재는 이 원격 관측과 별개이며, 해당 probe는 내용·모델 호환성이나 전체 파일 읽기 성공을 검증하지 않았다.

## 확인한 SDK와 장치 초기화

실제 개발 container `moa2026-dev`의 `furiosa-opt-std 0.8.1`, `furiosa-opt-rt 0.8.1` 소스를 읽었다. [고정 Cargo.lock](../../moa-dev/locks/gemma4-Cargo.lock)도 두 버전을 가리킨다. 아래 `std/…`, `rt/…`는 각각 다음 **container 내부 경로**의 약칭이다. Host 파일 링크로 오해하지 않도록 code path로 기록한다.

```text
std = /home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1
rt  = /home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-rt-0.8.1
```

| 소스·심볼 | 확인한 호출/동기화 |
|---|---|
| `std/src/runtime.rs:63`, `Device::new` | `CurrentBackend::open(topology)`가 반환한 뒤 main/sub/TDMA/PDMA context를 만든다. 호출 자체는 `async`가 아니다. |
| `std/src/backend/npu/backend.rs:67`, `Npu::open` | 허용된 chip 설정을 해석한 뒤 `furiosa_opt_rt::Device::builder(...).open()`을 호출한다. 모델 device-function cache는 비어 있는 상태로 시작한다. |
| `rt/src/device/mod.rs:87`, `Builder::open`; 139행 `open_chips` | 노출 chip과 PE 배치를 선택하고 cluster를 연 뒤 `Engine::open`과 allocator 초기화를 완료해야 반환한다. 다른 process가 예약한 후보는 건너뛰지만, 이 감사에서 실제 예약 상태는 조회하지 않았다. |
| `rt/src/device/engine/mod.rs:79`, `Engine::open`; `rt/src/device/dma/mod.rs:34`, `Worker::open` | DMA queue/io_uring과 channel을 열고 `Reserved::open`으로 firmware/cluster를 준비한다. 그 뒤 BAR window를 열고 DMA worker thread를 시작한다. |
| `rt/src/device/engine/reserved.rs:253`, `boot`; 344행 `initialize` | Firmware image staging, boot 응답, cluster identity/설정 응답을 확인한다. 모델 weight나 QKV/FFN image를 실행하는 단계는 아니다. |
| `rt/src/device/dma/queue/mod.rs:115`, `DmaQueue::write` | 초기 firmware staging에는 synchronous write가 쓰인다. 해당 ticket completion이 올 때까지 polling한다. 이 함수 본문에는 별도의 elapsed-time 검사문이 없다. |
| `rt/src/device/engine/link.rs:172`, `Link::boot` | Boot 응답 polling은 별도 deadline을 검사하고 초과 시 오류를 반환한다. 초기화 전체가 무제한이라는 뜻으로 앞선 한 함수의 관찰을 확대하지 않는다. |

`rt/src/device/mod.rs:51–53`의 기본 5초는 **cluster 한 번의 launch 제한**이다. `load_model` 전체를 5초로 제한하거나 70초 이내 완료를 보장하는 설정이 아니다. Arena job의 70초 wall-clock 제한과도 별개의 범위다.

## CPU 적재·변환과 HBM 쓰기 경로

[host/load.rs](../../furiosa-opt-gemma4-12B/src/host/load.rs)의 `load_model`(360–388행)은 `model_dir` 탐색 → `File::open` → `Mmap::map` → `SafeTensors::deserialize` → embedding/final norm → vision/audio → 48-layer loop 순서다. Mmap을 만들었다는 것만으로 모든 payload page를 읽었다는 뜻은 아니다. Tensor staging의 `to_vec()`가 실제 bytes를 접근하므로 파일 page 접근과 host 메모리 복사도 첫 로그 전 비용에 포함될 수 있다. 현재 filesystem/cache 상태나 그 소요 시간은 측정하지 않았다.

| 로더·SDK 위치 | 실제 CPU 처리 |
|---|---|
| `load.rs:88–95` `load_bf16`, 131–139행 `load_f8` | `HostTensor::from_safetensors` 뒤 `to_hbm().await` |
| `std/src/tensor/memory.rs:143–167`, `HostTensor::from_safetensors` | Mapping shape와 byte 길이를 검사하고 `view.data().to_vec()` → `Tensor::from_buf`로 packed bytes를 저장한다. **현재 구현에는 대형 BF16/FP8 tensor를 원소별로 decode 후 repack하는 루프가 없다.** |
| `load.rs:112–128` `load_bf16_column`, 142–159행 `load_f4` | 명시적 dtype/shape 검사 뒤 raw `view.data().to_vec()` → `HostTensor::from_buf`. Packed FP4를 CPU에서 BF16/F32로 dequantize하지 않는다. |
| `std/src/storage/buf.rs:82`, `BufStorage::from_buf` | 받은 byte buffer를 감싸며 원소별 packing을 다시 하지 않는다. |
| `load.rs:170–189`, `reciprocal_global_scale`/`load_global_scale` | Layer당 세 F32 global scale의 0/비finite를 거절하고 역수를 계산한다. 전체 48-layer에서 144개 scalar다. |
| `load.rs:98–109`, `load_scalar_bf16` | Layer scalar 하나를 host에서 읽고 BF16 8개로 반복해 16-byte padded tensor를 만든다. 전체 48개 layer에 적용한다. |
| `load.rs:331–346`, `load_vision` | Position embedding은 host tensor에서 `Vec<bf16>`로 남는다. 이 17,203,200 B는 아래 HBM 쓰기 합계에 들어가지 않는다. |

일반 가중치의 host staging은 pageable `Vec<u8>`다(`std/src/backend/npu/host.rs:31–34`). 현재 loader는 `.pinned()`를 미리 호출하지 않는다. `HostTensor::to_hbm`(`std/src/tensor/memory.rs:93`) → NPU backend의 `to_hbm`(`std/src/backend/npu/backend.rs:212`) → `Function::write/write_into`(`std/src/backend/npu/function.rs:94–125`) → `device.alloc` + runtime `Device::write(...).await`로 이어진다. 각 tensor마다 새 HBM allocation을 만들고 그 쓰기가 완료되어야 loader가 다음 호출로 진행한다.

`load_mlp_projection`의 tuple 안에서도 `load_f4.await` → `load_f8.await` → `load_global_scale.await` 순서다(222–224행). `load_mlp`의 up/gate/down(233–240행), `load_common`(267–271행), 각 layer field 초기화, `load_model`의 48-layer loop도 모두 직렬이다. `async`라는 선언만으로 tensor 간 복사·적재가 병렬화되지 않으며 여기에는 `spawn`, `join!`, 다중 in-flight upload batch가 없다. Runtime이 한 요청의 여러 pair를 동시에 전송할 수 있다는 사실과 이 caller의 직렬 실행을 구분한다.

## HBM 호출 수와 논리 payload 계산

가정은 현재 Rust 축 `H=3840`, `L=15360`, `W=262144`, `LAYERS=48`과 로더에 맞는 checkpoint다. [lib.rs](../../furiosa-opt-gemma4-12B/src/lib.rs), [axes.rs](../../furiosa-opt-gemma4-12B/src/axes.rs), `load.rs`의 호출을 기준으로 센다. 한 chip의 논리 tensor payload이며 allocation 정렬·driver command 분할·재전송·firmware·Workspace·KV cache는 제외한다. 현재 smoke가 이 모든 호출을 수행했다는 주장도 아니다.

| 구간 | 구간당 `to_hbm` | 반복 수 | 논리 bytes 합계 |
|---|---:|---:|---:|
| Embedding table BF16 `[W,H]` | 1 | 1 | 2,013,265,920 |
| Final norm BF16 `[H]` | 1 | 1 | 7,680 |
| Vision: LN1 weight/bias, dense weight/bias, LN2 weight/bias, position norm weight/bias, embedding projection | 9 | 1 | 82,641,408 |
| Audio embedding projection BF16 `[H,640]` | 1 | 1 | 4,915,200 |
| Sliding layer | 24 | 40 | 5,870,982,240 |
| Full-attention layer | 22 | 8 | 1,315,770,592 |
| **합계** | **12 + 40×24 + 8×22 = 1,148** | | **9,287,583,040 B ≈8.650 GiB** |

Layer별 호출 내역:

- MLP는 up/gate/down 각각 packed weight·local scale·global scale의 3회 쓰기, 합계 9회다. Bytes는 `3 × (H×L/2 + H×L/16 + 4) = 99,532,812`다.
- `load_common`은 pre/post FF norm, input norm의 3회와 MLP 9회를 합쳐 12회다.
- Sliding 추가분은 Q/K/V weight 3회, 각 channel scale 3회, Q/K norm 2회, post-attention norm 1회, O weight/scale 2회, layer scalar 1회로 12회, 총 24회다.
- Full 추가분은 Q/K weight 2회, 각 channel scale 2회, Q/K norm 2회, post-attention norm 1회, O weight/scale 2회, layer scalar 1회로 10회, 총 22회다. Full은 V projection weight를 별도로 읽지 않는다.

Byte 계산을 식으로 다시 확인할 수 있다.

```text
MLP = 3 × (3840×15360/2 + 3840×15360/16 + 4)
    = 99,532,812
Sliding = MLP + (4096+2048+2048+4096)×3840
        + (4096+2048+2048+3840)×2 + 4×3840×2 + 2×256×2 + 16
        = 146,774,556 bytes/layer
Full    = MLP + (8192+512+8192)×3840
        + (8192+512+3840)×2 + 4×3840×2 + 2×512×2 + 16
        = 164,471,324 bytes/layer
Vision  = 2×6912×2 + 3840×6912×2 + 3840×2
        + 4×3840×2 + 3840×3840×2
        = 82,641,408
Total   = 2,013,265,920 + 7,680 + 82,641,408 + 4,915,200
        + 40×146,774,556 + 8×164,471,324
        = 9,287,583,040
CPU-only vision position embedding = 1120×2×3840×2 = 17,203,200
```

1,148회를 모두 물리 PDMA request라고 부르면 안 된다. `rt/src/device/engine/mod.rs:45–48,156–168`은 한 write의 합계가 2,048 B 이하면 BAR memory window에 `post`하고 나머지는 DMA thread에 보낸다. 현재 단일 chip shape에서 Sliding은 layer당 작은 write 6회/큰 write 18회, Full은 7회/15회, 선행 12회는 모두 큰 write다. 따라서 **296회가 작은 MMIO 경로, 852회가 큰 PDMA 경로**다. Full의 K channel scale `[512]` BF16이 1,024 B인 점도 반영했다. 이는 SDK 분기 호출 수이며 driver 내부 scatter/page/chunk 단위의 실측이 아니다.

## 전송 완료 대기와 이번 결과의 한계

`rt/src/device/mod.rs:209–215`는 pageable host slice를 전송 기간 동안 pin하고 완료를 기다리는 계약을 명시한다. `Engine::commands`(`rt/src/device/engine/mod.rs:223–259`)는 host backing을 판별하고 chip·memory kind별 scatter command를 만든다. 큰 embedding도 이 경로로 들어가지만, 실제 driver 내부 pin/chunk 비용은 이 감사로 측정하지 않았다.

`Engine::transfer`(215–219행)는 DMA thread queue에 넣은 뒤 reply를 기다린다. `enqueue`(265–276행)는 queue 공간을 기다릴 수 있고, `Worker::run`(`rt/src/device/dma/mod.rs:59–90`)은 DMA completion을 받아 reply를 보낸다. `DmaQueue::complete`(`rt/src/device/dma/queue/mod.rs:96–110`)는 해당 transfer의 모든 command 완료를 모은다. 이 코드 경로의 완료 대기를 읽었다는 사실은 실제 job에서 응답이 유실됐거나 worker가 정체됐다는 증거가 아니다.

정상적으로 필요한 작업은 장치/firmware 준비, 모델 파일·metadata 접근, packed-byte host staging, tensor별 HBM 할당과 쓰기 완료 대기다. 현재 자료로 각각의 시간이나 정상 완료 시간을 추정하지 않는다. 첫 두 구간을 나누려면 기존 모델 소스를 보존한 별도 진단 host의 `Device::new` 전후, `load_model` 전후 marker가 필요하다. 이후 세분화 여부는 그 관측에 따라 정한다. 모델·커널 변경이나 같은 무출력 smoke의 단순 반복만으로 원인을 알아냈다고 기록하지 않는다.

감사에 인용한 repository 파일과 container SDK 경로는 실제 존재를 확인했다. 확인 시 핵심 SDK 파일 SHA256은 다음과 같다.

```text
std/src/runtime.rs                 2c75ed585d0db12b93a1ad66ab18b7acd899b1f50ba00dc4e1b916a304c94dd7
std/src/tensor/memory.rs           fec623ddf8fadd5cf31b32beb747243c2c463c48e4f73bb29bb2b6ef5d1aaff1
std/src/backend/npu/function.rs    391b108bfd7ecf6ea87e19d67e6714fa56d20dc77b1616a5ea1e4e16ddd0126a
rt/src/device/mod.rs              93d82e7614811e77f013c088eae37f342ae7ad09efdf66fa2a5ff11379b90342
rt/src/device/engine/mod.rs        c39a674be0a6cac02df6e0cec3d199e4cf928d66bee70150dfbcd4de46f7e284
rt/src/device/dma/mod.rs           83913ad996e7a94ffd43503f03039ba2e54064b84db18f942a1977ea2a34efeb
rt/src/device/dma/queue/mod.rs     48fdab9dcada9ee49cca7fb92c838c8785ad65eb971d796c414e7c530d9682bd
```

이 감사에서는 위 새 문서 외 소스/SDK/Git 변경, 모델 파일 내용 읽기, 환경·자산 속도 측정, 모델/NPU 실행을 하지 않았다.

후속 [고정-prefix71198](../experiments/e55-arena-model-progress.json)과 [원본/E55 paired71207](../experiments/e55-model-fixed-prefix-pair.md)은 모델 적재와 짧은 text 실행을 완료했다. 이 성공과 측정된 단계별 시간은 위 최초 timeout의 원인을 소급해서 확정하지 않는다. Kernel/runtime 소스를 변경해 적재 문제를 고친 결과도 아니다.
