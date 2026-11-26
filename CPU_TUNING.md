# CPU profiling and optional SDPA attention

The local executor can select either the original `eager` attention implementation
or PyTorch's scaled dot-product attention (`sdpa`). This changes how attention is
calculated, while keeping the model weights, greedy token selection, CPU float32,
two Torch threads, scheduler, and cache/cancellation ownership unchanged.

**Normal startup still uses eager attention and unbatched execution.** SDPA is an
explicit option, independent of the experimental tensor-batching flag. Results
depend on the model, workload, and CPU; see the
[recorded investigation and HTTP comparison](benchmarks/cpu-attention/README.md).
In the checked counting workload, SDPA improved unbatched throughput by about
12–13% for SmolLM2 and 6–7% for Qwen. It did not improve the batched CPU path;
leave batching off when trying the measured unbatched gain.

## Run the optional path

From the project root, with the existing downloaded model:

```bash
LLM_TENSOR_BATCHING=0 LLM_ATTENTION_IMPLEMENTATION=sdpa \
  LLM_MODEL_PATH=.models/qwen2.5-0.5b \
  .venv/bin/python -m uvicorn llm_runtime.local:create_local_app \
  --factory --host 127.0.0.1 --port 8000
```

Use `.models/smollm2` for the original checkpoint. Omit the attention flag or set
it to `eager` to retain the original implementation. Only `eager` and `sdpa` are
accepted; other values fail before loading the model. The loader also verifies
that the requested attention implementation was actually selected and fails
startup on an unexpected Transformers-level fallback. SDPA still lets PyTorch
choose an available kernel for the CPU, dtype, and input shapes.

Change to `LLM_TENSOR_BATCHING=1` to test SDPA with tensor batching. That option remains
off by default. See [TENSOR_BATCHING.md](TENSOR_BATCHING.md) for its cache ownership
and memory tradeoffs. Starting a server reads these flags once; restart to change
them. The browser and Android app need no changes.

`GET /health` includes `executor.attention_implementation`, reporting the loaded
implementation. The direct Python equivalent is:

```python
executor = await LocalModelExecutor.load(
    ".models/qwen2.5-0.5b",
    attention_implementation="sdpa",
    tensor_batching=False,
)
```

No new dependency or model download is needed. This milestone does not add GPU,
MPS, quantization, sampling, repetition penalties, or a thread-count setting.

## What was investigated

The previous milestone showed a two-client throughput regression with padded
tensor batching. A coarse diagnostic located most of the increase inside the
model call. The new investigation uses PyTorch's CPU operator profiler inside
the owning model worker, with shapes recorded, to identify expensive operators.
Profiler instrumentation is kept separate from unprofiled timing samples.

Screening then varies one setting at a time across eager/SDPA attention, one/two/
four Torch threads, and batching off/on. It reuses one loaded checkpoint, clears
request caches between settings, uses the real budget-32 prefill planner, warms
four decode rounds, and measures 16 rounds for one fixed pair. Three repetitions
reverse setting order between repetitions. Generated-token fingerprints are
compared with the baseline. These are diagnostics, not HTTP throughput results.

The chosen production change exposes the standard SDPA option. Thread count stays
at two because the screening did not establish a consistent thread-only benefit
across both models. Matrix operations remain a separate performance target;
choosing SDPA is not a claim that every batching bottleneck has been solved.

## Verification and reproduction

```bash
# Initial operator profiles and small candidate screen; no server or downloads.
.venv/bin/python -m scripts.investigate_cpu_inference \
  --output .experiments/new-cpu-investigation

# Full eager/SDPA HTTP comparison on both existing checkpoints.
.venv/bin/python -m scripts.verify_cpu_attention \
  --output .experiments/new-cpu-attention-run --repetitions 3

# Regression suite; run after timing, not alongside it.
.venv/bin/python -m pytest -q
```

The HTTP runner starts fresh temporary loopback servers on port 8020, refuses an
existing output directory/listener, and stops each owned server in `finally`.
It explicitly sets the requested attention implementation so an exported user
environment variable cannot silently change the baseline. It checks real model
token/text agreement with independent eager Transformers references, multi-turn
history/usage, two-client cancellation/disconnect, waiting removal, and zero
final logical/owned-tensor state.

Timings cover unbatched concurrency one/two and batched concurrency two. Batched
single-client HTTP timings are not included. One preflight stream and one warmup
per timing case are excluded, along with loading, correctness, and lifecycle work.
Each measured run uses four 96-token counting responses. Small fixed-pair screens
are not used as substitutes for these longer HTTP measurements.

Offline tests include strict option validation, unchanged defaults, loader
fallback cleanup, SDPA cached/ragged token agreement against independent eager
models, both scheduling policies, and gated cancellation/shutdown for both
attention implementations. Existing EOS/Unicode/storage/lifecycle tests remain
in the full suite. Checked fixtures do not establish bitwise agreement for every
possible prompt/model configuration or improve factual answer quality.
The final regression run passed **510 Python tests**.

## Primary implementation references

- [Transformers 4.57 attention interface and supported switching](https://huggingface.co/docs/transformers/v4.57.1/en/attention_interface)
- [PyTorch 2.14 scaled dot-product attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch 2.14 operator profiler](https://docs.pytorch.org/docs/2.14/profiler.html)
