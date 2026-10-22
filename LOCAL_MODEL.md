# Optional local model

The runtime can now generate real text with a local CPU model. Simulation remains
the default and requires none of the model dependencies. This is a small serving
integration, not a full chatbot or a production model server.

The tested checkpoint is [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
a 135M-parameter instruction model published under Apache-2.0. The download helper
pins revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` for reproducibility.
It downloads only configuration, tokenizer, safetensors weights, and the model
card, excluding ONNX variants and training artifacts.

## Install and run

From the repository root, activate the existing virtual environment:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,local]'
python -m llm_runtime.download_model
LLM_MODEL_PATH=.models/smollm2 uvicorn llm_runtime.local:create_local_app \
  --factory --host 127.0.0.1 --port 8000
```

The first download requires internet access and stores approximately 270 MB in the
ignored `.models/smollm2` directory. Model files are not committed. The optional
PyTorch/Transformers installation uses additional disk space. No hosted inference
service or API key is involved. After installation/download, serving uses only
local files; missing files fail startup rather than silently downloading.

The tested setup is Python 3.12, PyTorch 2.14.1, Transformers 4.57.6, macOS on an
Apple M3 with 16 GB RAM. This backend uses float32 CPU inference and two PyTorch
intra-op threads, even on Apple Silicon. It sets that process-wide thread count
at load time. CUDA/MPS, quantization, and device selection are not implemented.
Use one Uvicorn worker: each process would otherwise load its own weights and own
separate queues. Startup loads the model before accepting requests.

In another terminal:

```bash
curl -N -D - http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the capital of France? Answer briefly.","max_tokens":24}'
curl http://127.0.0.1:8000/health
```

The verified answer was `The capital of France is Paris.`. This tiny model can
produce incorrect or incomplete answers; the integration does not establish answer
quality. Inputs are single user instructions, with the tokenizer's chat template
and generation prefix applied automatically. Multi-turn history, sampling, tools,
and OpenAI-compatible chat endpoints are not implemented.

Cancel with the ID from `X-Request-ID`:

```bash
curl -X POST http://127.0.0.1:8000/requests/REQUEST_ID/cancel
```

## What changes in real mode

- Prompt counts use the model tokenizer, including template/special tokens,
  instead of whitespace words. The submitted prompt still has the 8192-character
  API limit. Whitespace prompts are valid instructions encoded by the template.
- The model context limit and logical KV capacity both constrain
  `encoded_prompt_tokens + max_tokens`; an impossible request returns HTTP 422
  before streaming. The downloaded model declares an 8192-token context.
- Prefill chunks feed actual model token IDs and extend a per-request KV cache.
  Decode uses greedy argmax, reuses cached context, and emits one model token per
  selected decode step. The behavior follows [Transformers cached inference](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation).
- Real token events include an additional integer `token_id`. Text pieces can be
  empty for partial UTF-8 sequences or EOS. Clients should concatenate `text` and
  use token indices/counts independently of visible text length.
- EOS stops generation early with `finish_reason: "eos"`; an EOS token is included
  in `generated_tokens` and its text is empty. The output limit still produces
  `"length"`, and accepted cancellation produces `"cancelled"`.
- Tokenization of the bounded prompt runs synchronously using the fast tokenizer.
  Loading and model forwards run on a dedicated serialized CPU worker. Continuous
  scheduling remains, but this is not true tensor batching: forward calls execute
  one at a time behind the existing iteration barrier.
- Active cancellation discards results at a boundary. Cancelling an asyncio task
  cannot terminate native model computation: even repeated shutdown cancellation
  waits for submitted worker operations to finish before releasing state. A stuck
  native operation can therefore delay shutdown.

## Memory and cleanup

The existing reservation manager enforces logical token slots, not physical RAM.
Each admitted request gets separate real KV tensors; no sharing or paging is
implemented. Prefill fills the cache, then each nonfinal decode forward extends
it. The final generated token need not be fed back into the model, so actual KV
length normally ends at `prompt + generated - 1`. Logical accounting deliberately
reserves/counts all output tokens conservatively.

Completion, EOS, cancellation, errors, and shutdown call `release_request()` after
work settles. This drops per-request caches; shutdown also closes the worker and
drops the model reference. The allocator or operating system may retain freed
memory, so zero live tensors does not imply zero process RSS.

`/health` exposes backend/device/dtype, context size, model parameter bytes, and
`kv_tensor_bytes_after_last_step` separately from logical cache counters. The
KV figure is a snapshot after settled worker steps and cleanup, not a live peak
or RAM bound. It excludes model weights, tensor temporaries, output queues, and
library overhead. The local factory uses two active requests; other runtime
settings remain the standard defaults. For experiments, use
`create_app(config, executor_loader=...)` programmatically.

## Verification

```bash
python -m pytest -q
python scripts/verify_local_model.py > benchmarks/local-model-smoke.json
```

The normal suite needs no download. Optional offline tests construct a tiny
random model and compare chunked cached output against `model.generate`; they
skip if local dependencies are absent. Worker and protocol tests still run
without PyTorch and check event-loop responsiveness, repeated cancellation,
startup cleanup, context rejection, and ownership while a native step is gated.

The checkpoint smoke script requires the downloaded model and development
HTTPX dependency. It compares two concurrent API requests with reference greedy
output token IDs, checks canonical decoded text, verifies EOS/length handling,
and gates one returned decode result to check HTTP cancellation and cleanup.
HTTPX's in-process ASGI transport buffers response bodies; this smoke is not a
network streaming benchmark. A separate live Uvicorn/curl check verified real SSE
output and clean shutdown. Simulator benchmark numbers remain separate.

See [the recorded smoke check](benchmarks/local-model-smoke.json) and
[its interpretation](benchmarks/local-model.md) for the hardware, versions,
measured memory snapshots, and limitations.
