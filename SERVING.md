# Serving

Running the model as an OpenAI-compatible HTTP server. This is separate from kernel
optimization — if you are here to make the kernels faster, see
[README.md](README.md). For where the server code lives, see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

Access to Furiosa RNGD and a checkpoint is required. Unlike the kernel-optimization loop,
the server has no hardware-free path: it loads weights into HBM and runs the model.

- The toolchain from [README.md](README.md#toolchain).
- A Furiosa RNGD environment, local or through the Arena server.
- A Gemma-4-12B-it checkpoint in safetensors form.

## Starting the server

```sh
./scripts/run_server.sh
```

Start the server with this script rather than executing the built binary directly — the
script goes through `cargo furiosa-opt run`, which compiles the `#[device]` kernels before
launching. A plain `cargo run` builds for the CPU and will not work.

`src/bin/server.rs` reads three environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `RNGD_MODEL_DIR` | *required* | The checkpoint directory, or its parent |
| `GEMMA4_API_ADDR` | `0.0.0.0:8000` | Listen address |
| `GEMMA4_API_KEY` | unset | Bearer token; unset means no auth |

## Batched serving

The supplied harness handles HTTP, tokenization, sampling, streaming, admission and KV
allocation. `host::model::Model::forward` is the host-side calculation boundary:

```rust
pub async fn forward(
    &mut self,
    device: &mut Device,
    inputs: &[Input<'_>],
    positions: &[usize],
    blocks: &[&Blocks],
    workspace: &mut Workspace,
) -> Result<&[Logits], Box<dyn std::error::Error>>;
```

Each row appends one token (`inputs[i].token`, or an image/audio soft token in
`inputs[i].soft`) at `positions[i]` in one request's history. `blocks[i].full` maps logical
full-attention blocks of `Tf` positions to physical blocks in the shared cache;
`blocks[i].sliding` selects that request's sliding-window buffers. The harness retains these
mappings until completion. The model updates KV and returns logits in input order; it does
not sample or finish requests. Rows have disjoint block mappings, and each history before
`positions[i]` has already been written. Prompt tokens are fed one per iteration. A row with
`inputs[i].logits == false` (every prompt token but the last) needs no logits, and its buffer
is stale.

`load::Weights` holds checkpoint tensors. `model::Model` owns the calculation scratch and
dispatches to the existing `runtime::Decode`, then `ops_*` and `device/`.
`runtime::Workspace` holds KV, RoPE, masks and output staging; it is shared by the batch.
There is no single whole-model `#[device]` function in this reference.

The reference calls the existing kernels serially within `forward`, so concurrency adds no
throughput yet. Participants can implement joint batch calculation behind the same caller
contract. Image/audio soft tokens retain their existing representation in `Input`. Host
dispatch can be changed along with kernels; using the supplied scheduler requires no serving
implementation from participants. Everything outside `src/api/` may be changed.

`host::worker` admits waiting requests between iterations, advances every active request by
one token, and reuses completed requests' blocks. With at most `R` active requests, total
prompt and output limit `L`, and block size `S`, the pool contains `R * ceil(L / S)`
full-attention blocks per layer. Blocks are assigned as histories grow. Every admitted
request can finish within its budget, so preemption is unnecessary. The baseline uses
`R = 4`, `L = 8192`, `S = 512` (`Tf`), so 64 blocks per layer; each active request also has
its own sliding-window buffers per sliding layer.

Requests beyond `L` are rejected; requests beyond `R` wait. These are harness limits,
not restrictions imposed by the NPU. A device failure takes the worker offline instead of
reusing potentially incomplete shared state. A single thread still owns the device, the
weights and the workspace, because `Device::new()` is non-`Send`; see `api/worker.rs` before
changing how the server is threaded.

## Sampling

Sampling is fixed server-side at `SamplingConfig::DEFAULT` (temperature 0), which picks
the argmax token (greedy decoding). The API does not read
`temperature`, `top_p` or `top_k`: a request that includes them is still accepted and they
are ignored.

### Fixed output length

A chat completion request with `"ignore_eos": true` always generates exactly
`max_completion_tokens` tokens. The server excludes the end-of-sequence tokens and the
reasoning-channel tokens from sampling, so generation does not stop early and every token
is part of the visible answer. The Stage 2 performance tests set this flag to give every
request the same output length; other requests leave it unset and stop normally.

## CLI

For a quick check without the HTTP layer:

```sh
./scripts/run.sh what is the capital of France
./scripts/run.sh --image path/to.png what is this
```

## Thinking mode

`/v1/chat/completions` can let the model reason before it answers. Either spelling works —
`chat_template_kwargs` (the vLLM/SGLang convention) is checked first:

```jsonc
{ "chat_template_kwargs": { "enable_thinking": true } }   // or:
{ "reasoning_effort": "medium" }   // "none"/"minimal" means off; the template has no levels
```

With neither present thinking is off, matching the chat template's own default. The
reasoning comes back separated from the answer, under the field name the OpenAI-compatible
ecosystem settled on:

```jsonc
{ "choices": [ { "message": {
    "role": "assistant",
    "reasoning_content": "3 × 4 = 12, 3 × 5 = 15, 4 × 5 = 20 …",
    "content": "47\nJustification: …"
} } ] }
```

Streaming puts the same text in `delta.reasoning_content`; a chunk never carries both
fields. The field is omitted entirely when the model did not reason, so a client that knows
nothing about thinking sees an unchanged response shape.

**Budget for it.** Reasoning tokens are drawn from the same `max_tokens` allowance as the
answer, and the model is not brief: a question that answers fine in 200 tokens can easily
spend 500 reasoning first. Too small a budget returns `finish_reason: "length"` with a
populated `reasoning_content` and an **empty** `content`. Roughly 1000 tokens is a sane
floor.

The whole switch is a matter of prompt shape rather than a second decoding path — see
`api::handlers::wants_thinking` for how a request turns thinking on, and `host::generate`
for how the two channels are pulled apart.
