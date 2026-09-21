# E55 전체 모델 실행 준비 상태

2026-09-21. 현재 채택된 E55의 실제 host 호출과 설치된 의존성을 확인했다. **기록된 Stage1 커널 검사를 통과했고 Arena의 실제 모델 자산도 확인했지만, 첫 CLI 실행은 초기화 구간에서70초 timeout이었고, 후속 고정-prefix 진단은 실제 text 실행과262,144개 finite logits를 확인했다. 이어 원본/E55 logits 전체를 비교했고 최대 차이1.5·top1 동일을 관측했다. 후속 무수정 CLI71395도 Python 입력의 실제 문자열 생성과 Stop/exit0을 완료했다. 정확도 수용 여부·공식 처리량은 아직 검증하지 못했다.** 공식 [README의 Stage2 절](../../furiosa-opt-gemma4-12B/README.md)은 입력 수·prompt/output 길이·반복 횟수·RNGD 구성·정확도 기준·성능 지표를 TBD로 남긴다.

추가 [격리 빌드와 registry 검사](../experiments/e55-model-host-build.json)에서 `gemma4`와 `server` release build가 모두 성공했다. Source/Cargo/toolchain46개를 E55와 대조했으며, 두 ELF는 각각14개 device image를 포함하고 registry 전체가 byte-identical하다. Stage1 세 image도 실측 E55 `test_kernels`의 해당 image와 정확히 같다. 이는 기존 세 커널 밖의 **실제 모델 진입점에 필요한 컴파일·패키징**을 확인한 것이며, 모델 실행 성공을 뜻하지 않는다. [재현 절차](../../moa-dev/experiments/candidates/e55-model-host-build/README.md).

## 실제 실행 경로

[src/bin/gemma4.rs](../../furiosa-opt-gemma4-12B/src/bin/gemma4.rs)의 `main`은 다음 순서로 동작한다.

1. prompt와 선택적 `--image`를 읽고 `GEMMA4_MAX_NEW_TOKENS`를 해석한다. 기본 prompt는 Rust hello-world 요청, 기본 생성 예산은 300이다. 이 CLI에는 sampling seed나 top-k 옵션이 없다.
2. `ops::embed_token.topology()`로 `Device`를 만들고 [host/load.rs](../../furiosa-opt-gemma4-12B/src/host/load.rs)의 `load_model`을 호출한다. 모델 적재에는 실제 HBM 전송이 포함된다. 장치 없이 가중치 파일만 파싱하는 CLI 모드는 없다.
3. 선택적 이미지를 CPU에서 patch로 변환한다. `Tokenizer::new(load::tokenizer_path(&model))`로 모델 디렉터리의 tokenizer를 읽고, `encode_chat(..., false)`로 chat template을 렌더링한다. `splice_multimodal_tokens`가 이미지 soft token 위치를 확장한다. CLI는 audio를 `None`으로 넘기며 audio 입력은 서버의 별도 경로다.
4. [host/runtime.rs](../../furiosa-opt-gemma4-12B/src/host/runtime.rs)의 `Workspace::new`가 sliding/full RoPE, mask, 각 layer의 KV cache, 위치 offset, logits를 만든다. 축은 [axes.rs](../../furiosa-opt-gemma4-12B/src/axes.rs)에 고정되어 있다: hidden 3840, vocabulary 262144, sliding window 1024, full page 512, page 수 64, 최대 위치 32768.
5. [host/generate.rs](../../furiosa-opt-gemma4-12B/src/host/generate.rs)의 `generate`가 새 `Decode` 임시 버퍼를 만들고 prompt를 **토큰별 순차 실행**한다. 일반 토큰은 `Decode::run → embed_token → run_layers`, 이미지/audio soft token은 해당 encoder 뒤 `run_with_embedding → run_layers`로 들어간다. prompt의 마지막 위치만 logits를 계산한다.
6. `run_layers`는 [lib.rs](../../furiosa-opt-gemma4-12B/src/lib.rs)의 48개 layer를 돈다. 로더가 `layer % 6 == 5`를 full로 분류하므로 sliding 40개, full 8개다. Sliding은 QKV/RoPE/cache write → sliding attention → O/post-norm/residual → FFN, full은 QKV/cache write → 유효 page별 attention 누적 → O/post-norm/residual → FFN이다. 마지막에는 `final_norm_and_logits`가 embedding table을 출력 weight로 재사용한다.
7. `read_logits`가 BF16 vocabulary logits를 host로 가져와 F32로 변환한다. [host/sampling.rs](../../furiosa-opt-gemma4-12B/src/host/sampling.rs)의 `sample`이 token을 고르고, decode stream이 문자열을 만든다. 생성 토큰을 다시 전체 48-layer 경로에 넣는 과정이 EOS·stop·예산 종료까지 반복된다.

[src/bin/server.rs](../../furiosa-opt-gemma4-12B/src/bin/server.rs)는 tokenizer를 먼저 만든 뒤 [api/worker.rs](../../furiosa-opt-gemma4-12B/src/api/worker.rs)의 worker를 시작한다. Worker는 동일한 `load_model`, `Workspace`, `generate`를 사용하며 모델과 Workspace를 요청 사이에 보존한다. `begin_decode`는 임시 activation을 새로 만들지만 KV cache 전체를 매 요청 재할당하는 함수가 아니다. 따라서 다중 요청의 mask·위치·cache 재사용도 E2E 검증 항목이다. 이 관찰 자체가 cache 오류를 뜻하지 않는다.

## 로더가 실제로 요구하는 자산

| 자산/설정 | 구현에서 확인한 요구 조건 | 현재 상태와 한계 |
|---|---|---|
| `RNGD_MODEL_DIR` | `load::model_dir`가 환경 변수 자체, 그 아래 `Gemma4-12B-it`, `gemma-4-12b-it-NVFP4` 순으로 `model.safetensors`를 찾는다. | 조사한 host와 개발 container 모두 미설정. 후속 Arena job71164에서는 설정됨을 확인했고 세 번째 후보에 모델 파일이 있었다. |
| 단일 `model.safetensors` | `load_model`이 mmap으로 한 파일을 읽는다. shard index를 처리하거나 Hugging Face에서 다운로드하는 경로는 없다. | 로컬 조사 경로에는 없으나 Arena job71164의 세 번째 후보에는 읽을 수 있는 단일 파일이 있다. 후속71224에서 전체 SHA와 필수 header/type 계약을 확인했다. Tensor 값과 E2E 정확도는 별도 범위다. |
| Text 가중치 전체 | `model.language_model.embed_tokens.weight`, `norm.weight`, 48개 `layers.{i}`의 norm/scalar/attention/MLP를 읽는다. Attention은 FP8 weight와 BF16 `[rows,1]` scale, MLP는 U8 packed FP4 `[rows,columns/2]`, FP8 block scale, F32 global scalar를 요구한다. Global scale은 0/비finite를 거절하고 역수를 HBM에 적재한다. | 기존 fixture에는 이 체크포인트가 없다. BF16 HF 모델의 다른 tensor 이름·dtype·shape를 자동 변환하지 않는다. 전체 shape 검사는 `HostTensor::from_safetensors` 및 `load_f4`/`load_bf16_column`에 있다. |
| Vision/audio 가중치 | `load_model`이 **입력 modality 검사 없이** `load_vision`과 `load_audio`를 호출한다. `model.vision_embedder.*`, `model.embed_vision.embedding_projection.weight`, `model.embed_audio.embedding_projection.weight`가 필요하다. | 현재 로더에서는 **텍스트 요청만 보내도 필수**다. 이미지·WAV 입력 파일 자체는 텍스트 요청에 필요하지 않다. |
| `tokenizer.json` | `load::tokenizer_path_in`은 모델 디렉터리의 이 파일을 반환하고 `Tokenizer::new`는 `HuggingFaceTokenizer::from_file`로 읽는다. | 로컬에는 없으나 Arena job71164에서 tokenizer.json을 확인했다. 실제 encode 성공과 vocabulary·특수 token 계약은 별도 검증 대상이다. |
| Chat template와 tokenizer 메타데이터 | `encode_chat`은 template을 반드시 적용한다. 실제 lock은 `llm-tokenizer 1.7.0`이다. 설치 소스의 `HuggingFaceTokenizer::from_file/from_built_tokenizer`는 sibling `chat_template.json`, `chat_template.jinja` 등 또는 `tokenizer_config.json`의 `chat_template` 문자열을 사용한다. `ChatTemplateState::apply`는 template이 없으면 실패한다. | **template 공급원 하나는 chat 경로에 필수**다. `tokenizer_config.json` 자체를 별도 template이 있어도 항상 필수라고 주장하지 않는다. 올바른 특수 token·chat formatting을 재현하려면 실제 모델의 metadata를 함께 고정해야 한다. |

Rust weight loader는 모델 `config.json`을 읽어 축을 바꾸지 않는다. 모델 구조는 Rust 축·상수·layer 배치에 고정되어 있으므로 외부 자산을 확보하면 먼저 이름·shape·dtype·quantization 계약을 검사해야 한다. 단순히 공개 모델 ID가 같다는 이유만으로 현재 로더와 맞는다고 가정하지 않는다.

자산 조회 범위는 작업 공간의 모델/tokenizer 파일명(빌드·Git·인증 상태 디렉터리 제외), host `/home/jihwan38/.cache/huggingface`, container `/workspace`, `/models`, `/data`, `/home/moa/.cache/huggingface`였다. 실제 지정된 모델 디렉터리가 없었으므로 다른 시스템 전체에 자산이 없다고 확대하지 않는다. 비밀값·인증 파일을 읽거나 출력하지 않았다.

## 기존 검증이 대체하지 못하는 부분

[generate_references.py](../../furiosa-opt-gemma4-12B/scripts/generate_references.py)는 체크포인트 없이 세 커널 입력을 합성하고 출력/checksum만 저장한다. 현재 `ref/fixtures.safetensors`는 202,632 bytes·114 tensors이며 namespace는 run0..2의 `sliding_project_qkv`, `sliding_attention_output`, `decoder_feedforward`뿐이다. [test_kernels.rs](../../furiosa-opt-gemma4-12B/src/bin/test_kernels.rs)도 이 세 커널만 실행한다. 추가 seed 및 경계 검사는 그 범위를 넓혔지만 실제 모델 layer 연결·embedding·attention softmax·full attention·logits·tokenizer·sampling·멀티모달 전처리 검증으로 바뀌지 않는다.

[E55 source/build 감사](../experiments/e55-source-build-audit.json)의 "full host build"는 Stage1 `test_kernels` host의 완전한 빌드를 뜻하며, Gemma E2E 실행을 뜻하지 않는다. 해당 기록의 embedded image는 세 개다. 추가 모델-host 빌드에서는 실제 registry의14개를 확인했다: `ops`10개, vision3개, audio1개이며, host caller가 없는 별도 `ops_vision::layernorm` 선언은 제외된다. CLI에서 audio는 입력할 수 없지만 공통 생성 함수의 branch가 registry에 남아 있고, 서버의 chat `input_audio`에서는 실행 가능하다. `full_attention_page`도 position512부터 필요하므로 짧은 text 한 번이 모든 text 커널을 실행하는 것은 아니다. 컴파일·포함 여부와 실행·정확도 검증을 구분한다.

[scripts/reference/gemma4.py](../../furiosa-opt-gemma4-12B/scripts/reference/gemma4.py)는 전체 text forward도 정의하지만 `__main__`은 `transformers`의 외부 config/tokenizer/weight를 읽는다. 개발 container에는 `transformers`가 없으며 필요한 HF 자산도 없다. `load_hf_weights`가 다른 형식의 HF state를 `strict=False`로 넣는 예제를 현재 NVFP4 체크포인트와 동치인 oracle로 간주할 수 없다. 현재 E2E golden prompt/logits/token trace fixture도 없다.

로컬 자산 없이 가능한 검사는 전체 host의 격리 빌드·registry 조사와 CPU helper의 별도 합성 검사다. 후속 Arena 자산 조회로 원격 모델 실행을 시도할 수 있게 되었다. 후자는 sampling의 금지 token/top-k 규칙, 이미지 patch 좌표, WAV frame 경계 등에는 의미가 있지만 새 테스트가 필요하며 E2E 정확도 근거는 아니다. 기존 Stage1 fixture를 다시 실행하는 것만으로 미검증 E2E 경로가 추가로 검증되지는 않는다.

## Arena 자산 조회와 첫 실제 실행

[Job71164](../experiments/e55-arena-model-assets.json)는 `RNGD_MODEL_DIR`의 세 번째 후보에서 model.safetensors, tokenizer.json, tokenizer_config.json, chat_template.jinja가 모두 regular/readable임을 확인했다. 파일 내용이나 인증 정보를 출력하지 않은 메타데이터 검사다. 기존 로컬 자산 부재를 Arena에도 적용할 수 없다는 점이 확인되었다. [소스와 재현](../../moa-dev/experiments/candidates/arena-model-assets-e55/README.md).

[Job71173](../experiments/e55-arena-model-smoke.json)은 위에서 검증한 E55 gemma4 ELF 그대로 `Hi`, 생성 예산1로 실행했다. 70.093초 뒤 scheduler timeout으로 끝났고 `model loaded` 로그는 없었다. 실제 CLI에서는 `Device::new`와 `load_model` 모두 그 로그보다 앞이므로 이 실행만으로 장치 초기화와 모델 적재 중 어느 쪽인지 구분하지 못한다. 출력 token·E2E 정확도·처리량 결과는 없다. 업로드 archive의 ELF hash는 동결 E55와 같다. [실행 wrapper와 한계](../../moa-dev/experiments/candidates/arena-model-smoke-e55/README.md).

[후속 진단71198](../experiments/e55-arena-model-progress.json)은18.422초에 정상 종료했다. Device 초기화가0.317초에 완료되고 model load가16.058초에 완료된 뒤, 실제 tokenizer로 만든14-token `Hi` prefix의 마지막 위치에서262,144개 finite logits를 얻었다. 기존46개 입력과 kernel image를 보존한 별도 host다. 생성 sampling·멀티모달·position512 page 경계·원본과의 logits 비교는 이 job에서 수행하지 않았다. 한 번의 비통제 실행이므로 최초 timeout의 원인이나 cold-load·공식 처리량을 확정하지 않는다.

[적재 경로 감사](model-loader-initialization-audit.md)는 현재 loader가 논리8.650GiB를1,148개의 직렬 `to_hbm` 호출로 적재함을 코드에서 계산했다. 이는 실측 전송량·시간이 아니다. SDK `HostTensor::from_safetensors`는 shape와 byte 길이를 검사하지만 dtype 자체는 검사하지 않으므로, 이름·shape·dtype의 의미상 호환성도 별도 확인해야 한다. [별도 고정-prefix 진단](../../moa-dev/experiments/candidates/model-fixed-prefix-e55/README.md)의 원본/E55 host는 기존 입력46개를 보존하고 동일 probe를 하나 추가했다. 각각10개 text image와 기존 Stage1 image3개를 확인했다. [실제 paired job71207](../experiments/e55-model-fixed-prefix-pair.md)은 양쪽262,144개 finite logits를 모두 비교해 max abs1.5, mean abs0.194487, RMSE0.251039, bit-equal46,239개와 raw top1 동일을 관측했다. 정확도 허용기준은 미정이며 이 greeting 한 건의 top1 일치로 모델 정확성을 승인하지 않는다.

[자산 식별71224](../experiments/e55-arena-model-identity.json)은 가중치9,304,966,064 bytes의 전체 SHA256 `7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b`와 tokenizer/template 세 파일의 전체 해시를 기록했다. 전체1,389개 tensor 중 필수1,149개의 header 조건과 의도한 dtype이 모두 맞았다. Global scale 값·tensor 수치 자체를 header로 검증한 것은 아니다. Top-level partial은 선택 파일 chat_template.json 부재 때문이며 weight 해시는 완전하다. 앞선 모델 실행 당시의 파일을 소급하여 같은 hash로 확정하지 않는다.

[4-prefix 확장71239](../experiments/e55-model-prefix-suite.md)는 인사·산수·한국어·Python의82개 고정 input token에 대해1,048,576개 logits를 비교했다. 네 raw top1이 모두 같았고 max abs는0.583984375–1.5였다. 각 case는 새 Workspace를 사용했고 같은 job의 전체 weight hash 및 전후 stat/header·metadata hash를 확인했다. 순차1회 prefix/read 비율4.837–4.866은 공식 성능이 아니며, 생성 답변·near-tie·long context·멀티모달 정확도를 평가하지 않았다. [산술 차이와 다음 단일-kernel 대조](e55-model-numerical-differences.md)에 가능한 차이 발생 위치를 정리했다.

[실제 단일-kernel 복원 대조](../experiments/e55-model-single-kernel-controls.md)는 대상 image 하나만 원본으로 바꾸고 나머지9개를 E55와 동일하게 유지한다. Full-O 복원은 네 prefix의1,048,576개 logits 모두 E55와 bit-identical이었다. O/FFN 복원은 입력마다 원본과의 차이가 커지거나 줄었으며 오차의 가산 기여율로 해석하지 않는다. 이후 [생성 토큰 고정 진단71329](../experiments/e55-model-continuation.md)는32개 checkpoint·8,388,608개 finite logits와 모델당110forward를42.448초에 완료했다. 실제 공개 sampler의 greedy 선택32개가 모두 같았고 산수 case는8번째 선택 EOS106으로 끝났다. 최소 허용-token margin0.875, 최대 절대 차이1.6875이며 정확도 판정은 null이다. 원본 토큰만 다음 입력으로 공급하는 짧은 문맥 진단이라 자유 생성·정답률·cache 경계·멀티모달이나 다른 workload의70초 완료를 보장하지 않는다.

[Python 최대40 진단71362](../experiments/e55-model-continuation-python40.md)는 실제로14번째 EOS에서 종료했고14개 선택이 모두 일치했다. 전체3,670,016개 finite logits와41forward를 비교했으며 이전 첫8개의 보고값/자산이 재현됐다. 새 step9에서는 원본 margin0.125가 E55 margin0·tie2로 바뀌어 실제 동점을 관측했다. 원본/E55 모델 커널은 변경하지 않았고 정확도 판정은 null이다. 기존의 짧은 문맥 범위를 넓혔으나 문자열 decode/callback·정답률·cache 경계 검증을 대신하지 않는다.

[무수정 CLI71395](../experiments/e55-model-cli-python40.md)는 기존71173과 같은 ELF로 실제 문자열 `The result of the expression is: **3**`을 출력했다. Prompt28/completion11/Stop이며 asset 검사까지 전체12.284초에 완료했다. 기본 확률적 sampler를 유지하므로 greedy71362와 token sequence 일치를 요구하지 않는다. 같은46개 소스/14개 image의 기존 빌드와 실제 업로드를 연결했다. 실행 전 weight hash가 cache 상태에 영향을 줄 수 있어 이전 timeout의 원인 해결이나 cold-load 성능 개선을 주장하지 않는다.

## 원본/E55 정확도·성능 비교 조건

원본 `8559916`과 E55를 **같은 체크포인트·tokenizer/template·입력 bytes·토큰 ID·SDK/driver·RNGD 구성**으로 실행해야 한다. 모델/metadata 및 두 host와 registry image의 hash를 기록하고, cold load와 이미 적재된 모델의 실행을 분리한다. 원본의 이전 Stage1 수치나 다른 HF BF16 모델을 E55 NVFP4 E2E 대조군으로 대신하지 않는다.

정확도는 생성 문자열보다 **동일 token prefix에서의 logits 비교를 우선**한다. `runtime::Decode::run`은 token·position·`compute_logits`를 받고 `generate::read_logits`는 sampling 직전 값을 제공하므로, 외부 평가 규칙이 허용하는 별도 진단에서 같은 prefix를 두 구현에 공급할 수 있다. 비finite 여부, max/mean 절대 차이, 허용오차 밖 개수, 상위 token 순위와 margin을 기록하고, attention/cache 경계(512 page, 1024 ring)와 여러 요청을 포함한다. Stage1의 세 kernel atol/rtol을 E2E logits 기준으로 임의 승계하지 않는다. 전체 모델 오차 허용 기준은 아직 미정이다.

`generate`는 `rand::rng()`를 사용하고 CLI는 temperature=1/top-k=64/top-p=0.95를 고정한다. [api/schema.rs](../../furiosa-opt-gemma4-12B/src/api/schema.rs)의 `seed`/`logprobs`는 unsupported이며 [api/handlers.rs](../../furiosa-opt-gemma4-12B/src/api/handlers.rs)가 이를 HTTP400으로 거절한다. 비교를 위해 지원되지 않는 seed나 logits API를 몰래 구현하지 않는다.

기존 chat-completions API는 `top_k=1`을 허용한다. 예를 들어 temperature=1, top_p=1, top_k=1이면 `sample`이 금지 token을 제외한 최상위 후보 하나만 남기므로 **finite logits와 유일한 최댓값 아래에서는 RNG draw와 무관한 선택**이 가능하다. 이는 CLI의 기본 동작과 다르며 완료 endpoint는 같은 top-k 입력을 제공하지 않는다. 동률 정렬의 선택과 작은 logits 차이로 인한 argmax 전환까지 원본과 동일하다고 보장하지 않는다. `temperature=0`만 지정하는 방법도 내부에서 `1e-5`로 clamp되므로 명시적인 argmax 모드로 해석하지 않는다. 먼저 고정 prefix logits를 비교한 뒤 이 기존 API 옵션으로 token trace를 비교하는 순서가 적절하다.

Stage2 성능을 실제로 측정할 때는 아래 관측값과 공식 지표를 구분한다.

| 관측값 | 현재 구현 기준으로 분리할 범위 |
|---|---|
| 초기화 | 모델 파일 읽기·가중치 HBM 적재, tokenizer 준비, Workspace/RoPE/mask/cache 할당을 구분한다. |
| TTFT | prompt tokenization/멀티모달 전처리 포함 여부, model load 포함 여부, 첫 token 선택과 첫 표시 문자열 callback 중 어느 지점인지 고정한다. |
| Decode | prompt 뒤 각 step의 model execution, logits host 전송, sampling/디코딩, 출력 전달을 분리하고 실제 생성 token 수·EOS·stop 조건을 함께 기록한다. |
| 상태/메모리 | 동일 prompt 길이·출력 예산·modality에서 KV page/ring 경계, 요청 간 재사용, peak device/host memory를 비교한다. |
| 반복 | 같은 hardware/toolchain와 통제된 동시 작업 조건에서 원본·후보를 반복하고 각 run과 분산을 보존한다. 공식 반복 수나 점수식은 아직 확정하지 않는다. |

현재 CLI의 `ttft` 타이머는 모델·tokenizer·Workspace 준비 **뒤** 시작하며 첫 `on_delta` callback을 기준으로 한다. 따라서 로그의 `prompt_tokens / ttft`는 tokenizer나 모델 적재를 포함한 서비스 처리량도, prompt-only NPU 처리량도 아니다. `decode_elapsed`는 첫 callback까지 시간을 빼지만 분자는 전체 `completion_tokens`이므로 엄밀한 token 간 TPOT와도 구분해야 한다. 이 로그를 가공 없이 공식 Stage2 score로 보고하지 않는다.

최초 준비 감사는 읽기 전용이었다. 이후 추가한 Arena 조회와 실제 모델 실행 시도는 위의 별도 job·결과로 구분한다. 현재 채택된 kernel/runtime 소스와 인증 상태는 변경하지 않았다. Stage2 workload와 전체 모델 검증 결과를 확보하기 전에는 E55의 커널 개선을 모델 전체 성능이나 공식 순위로 환산하지 않는다.
