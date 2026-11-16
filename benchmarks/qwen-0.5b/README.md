# Larger-model verification: Qwen2.5-0.5B-Instruct

Verified 2026-11-16 on the existing Apple M3 / 16 GB Mac. This tests our own
Transformers/PyTorch runtime with a second, larger checkpoint before implementing
tensor batching. No llama.cpp conversion or external-engine comparison was run.

## Model, settings, and download

- Model: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
  revision `7ae557604adf67be50417f59c2c2f167def9a775`. The
  [pinned model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/README.md)
  identifies its Apache-2.0 license, approximately 0.49B parameters, and Qwen2
  architecture. The original SmolLM2-135M checkpoint uses the Llama architecture.
- Observed parameter count: **494,032,768**, about 3.7 times our 134,515,008-parameter
  SmolLM2 baseline. Float32 parameter storage is **1,976,131,072 bytes** (about
  1.84 GiB), separately from libraries, KV tensors, temporary work, and residency.
- Downloaded approximately 1 GB of BF16 safetensors/tokenizer/config/license files
  into ignored `.models/qwen2.5-0.5b` (953 MiB directory usage). The existing
  `.models/smollm2` is preserved. No new dependency was installed.
- The existing executor loads float32 weights on CPU with two model threads and
  greedy argmax. No GPU, quantization, or shared tensor batching was added.
- Qwen declares a 32,768-token context, but our default 8,192 logical KV slots and
  API character/output limits still apply. This check used short prompts and
  bounded outputs; it does not establish 32K context support or capacity.

Qwen's published generation configuration enables sampling and uses a repetition
penalty of 1.1. Those are not our executor's decoding rules. The reference checks
explicitly set `do_sample=False` and `repetition_penalty=1.0` to compare plain greedy
generation. Unused sampling defaults may produce Transformers stderr warnings.
The native executor's algorithm did not need modification. Both published EOS
IDs (`151645`, `151643`) are handled by its existing EOS-set logic.

## What passed

- **Direct-model agreement:** two native prompt cases produced exactly the same
  output token IDs as direct Transformers generation using the same loaded model
  and decoding settings. EOS and output-limit termination were both observed.
- **Conversation handling:** multi-turn JSON and streaming responses matched the
  direct reference, including prompt/output usage. The name-history example
  returned `Your name is Ada.`
- **Unicode consistency:** the Norwegian fixture matched the direct reference in
  JSON and streaming modes. This verifies faithful execution, not factual truth.
- **Real HTTP:** eight measured smoke requests completed across concurrency 1/2,
  with two excluded warmups and one excluded streaming preflight per setting.
  Final usage and terminal/identity validation passed.
- **Limits:** the one-token output cap produced one token; an API-valid Unicode
  string with too many encoded tokens returned HTTP 422 for KV capacity before
  model execution. Limits are tokenizer-based rather than character estimates.
- **Lifecycle:** queued disconnect removed a waiter while both blockers remained
  active; active disconnect and explicit cancellation recovered and served
  follow-up requests. Final queued/active counts, reserved/used slots, and tracked
  KV tensor bytes were all zero. The temporary HTTP server stopped afterward.
- **Baseline preservation:** the original SmolLM2 native/reference smoke was rerun
  successfully after making the reference settings explicit.
- **Automated tests:** **444 passed**, including seven new download-selection/
  identity/resume tests and four additional Qwen2 reference/EOS cases. The model
  download helper keeps the original default and selects separate directories.

## Actual examples and a quality limitation

The France question returned `The capital of France is Paris.` But Qwen incorrectly
answered that “blåbær” translates to “blue whale.” The direct Transformers reference
produced the **same wrong answer**, and both API formats matched it. The dictionary
meaning is [blueberry](https://dictionary.cambridge.org/dictionary/norwegian-english/blabaer),
not blue whale.

This particular error is reproduced without our serving adapter; it is not
evidence of a corrupted cache or streaming bug. A larger model is not guaranteed
to answer every question correctly. These few examples are not a scored quality
evaluation or evidence that Qwen is broadly better/worse than SmolLM2.

## Memory and timing observations

The native/reference smoke process recorded a **2.12 GiB peak RSS**. In the final
separate HTTP server, RSS samples were approximately **0.91 GiB at readiness** and
**2.01 GiB after checks**. Those latter values are resident snapshots, not peaks;
macOS residency, loading, and temporary allocations differ from parameter bytes.
The model fit and ran on this Mac in these bounded cases. This does not establish
a physical RAM cap or safe capacity for arbitrary contexts/concurrency.

The short France HTTP smoke recorded mean first-content latency of about 320 ms
at concurrency 1 and 469 ms at concurrency 2, with approximately 14.55/17.43 output
tokens/s. Each setting has only four measured requests. These are illustrative
samples, **not a repeated speed comparison**, a capacity estimate, or a batching
benefit. Tokenizers, model computation, output lengths, and host conditions differ
from earlier SmolLM2 measurements. Future batching comparisons must use the same
model/settings on both sides.

## Records and reproduction

- [local-smoke.json](local-smoke.json): native token/reference checks, parameter
  bytes, process peak RSS, cancellation, and final cache state.
- [chat-smoke.json](chat-smoke.json): multi-turn JSON/stream/reference agreement.
- [unicode-reference.json](unicode-reference.json): exact direct/reference/API
  agreement for the incorrect Norwegian example.
- [http-smoke.json](http-smoke.json): model-file/source hashes, pinned identity,
  versions, launch command, raw HTTP receipt samples, examples, errors, and cleanup.
- [smollm2-recheck.json](smollm2-recheck.json): original model reference regression.

Model files remain ignored locally and are not committed. Downloading requires
internet; afterward the model loads from local files only. The helper rejects a
destination owned by another model/revision or unknown existing contents. An
identity marker permits resuming a matching interrupted download; the final
runtime manifest is written after a successful download.

From the repository root, with existing local/dev dependencies:

```bash
.venv/bin/python -m llm_runtime.download_model --model qwen2.5-0.5b
.venv/bin/python scripts/verify_local_model.py --model-path .models/qwen2.5-0.5b
.venv/bin/python scripts/verify_chat_api.py --model-path .models/qwen2.5-0.5b
.venv/bin/python scripts/verify_chat_api.py --model-path .models/qwen2.5-0.5b --unicode-example
.venv/bin/python -m scripts.verify_additional_model \
  --output-dir .experiments/qwen-recheck
```

The HTTP verifier needs port 8020 free and a new output directory. It checks the
selected model identity, launches only that model, and stops its server in
`finally`. To use Qwen through the browser instead, stop an existing port-8000
server with Ctrl-C and run:

```bash
LLM_MODEL_PATH=.models/qwen2.5-0.5b .venv/bin/python -m uvicorn \
  llm_runtime.local:create_local_app --factory --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/**. The API alias remains `local-model`; health reports
the `qwen2` backend model type. Switching back uses `LLM_MODEL_PATH=.models/smollm2`
and a server restart. This is one selected checkpoint per server, not hot swapping
or a multi-model router.

Next: retain both models as baselines for optional tensor batching. Long contexts,
sustained larger-model memory profiling, a factual-quality evaluation, acceleration,
and Qwen comparisons with other inference engines remain future work.
