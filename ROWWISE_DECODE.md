# Experimental rowwise CPU projections

This option investigates the two-request CPU batching slowdown. Tensor batching
still performs a shared model forward, attention, and KV update, but plain Linear
projections with input shape `[2, 1, hidden]` calculate each request's row separately
before concatenating the results. The short local probe favored two single-row
matrix/vector operations over one two-row matrix operation on this Mac.

This is a narrow CPU experiment, not a replacement for efficient large-batch GPU
kernels. It keeps float32 weights, greedy token selection, two Torch threads,
request scheduling, and owned per-request KV tensors. It changes neither the
browser/Android API nor the downloaded model. Normal startup remains unbatched,
eager attention, and no projection specialization.

The checked two-client HTTP workload improved over ordinary tensor batching by
35–60%. Compared with unbatched execution, SmolLM2 was approximately tied and
Qwen improved by 4–8%, depending on attention. These are bounded measurements,
not a general batching guarantee. See the [full results](benchmarks/rowwise-decode/README.md).

## Try it

With an existing local checkpoint:

```bash
LLM_MODEL_PATH=.models/qwen2.5-0.5b \
  LLM_TENSOR_BATCHING=1 LLM_ROWWISE_DECODE=1 \
  LLM_ATTENTION_IMPLEMENTATION=sdpa \
  .venv/bin/python -m uvicorn llm_runtime.local:create_local_app \
  --factory --host 127.0.0.1 --port 8000
```

Use `.models/smollm2` for the other existing checkpoint, or `eager` attention for a
comparison with the original attention implementation. Restart to change flags.
`LLM_ROWWISE_DECODE` accepts only `0` (default) or `1`; `1` requires
`LLM_TENSOR_BATCHING=1`. Both existing full-attention Llama and Qwen2 models are
supported. The direct Python equivalent is:

```python
executor = await LocalModelExecutor.load(
    ".models/qwen2.5-0.5b", tensor_batching=True,
    rowwise_decode=True, attention_implementation="sdpa",
)
```

`GET /health` reports `executor.rowwise_decode` and the cumulative
`rowwise_decode_linear_calls` count. That count measures projection invocations,
not tokens or model forwards. Shared model calls remain visible in
`batched_forward_calls` and `max_forward_batch_size`.

## Boundaries and ownership

The loader replaces plain `torch.nn.Linear` modules only in its own model with a
small subclass. Weights and biases are the same Parameter objects, with identical
parameter names. Meta-device initialization avoids allocating duplicate weights.
There is no global PyTorch patch and no reference cycle through a bound model
method. Shutdown clears both the model and the executor's list of projections.

The specialized branch requires evaluation mode, autograd disabled, CPU float32,
and exactly two single-token rows. Single-client inputs, larger batches,
multi-token prompt chunks, other dtypes/devices, and training/autograd use normal
Linear calculations. A pair of one-token prefill chunks can also match the shape;
this is a shape-based optimization, not an independently tracked decode phase.
Custom Linear subclasses are left intact.

KV packing, attention masks, logical positions, cache splitting, and cancellation
barriers are unchanged. Requests still own compact independent KV tensors after
each step. Temporary batched allocations remain possible; the logical token
capacity is not a physical RAM cap. Floating-point operation grouping can change
rounding, so tested greedy-token agreement does not guarantee identical answers
for every model and prompt.

## Reproduce the investigation

```bash
# HTTP timing and real-model reference/lifecycle checks; owns loopback port 8020.
.venv/bin/python -m scripts.verify_rowwise_decode \
  --output .experiments/new-rowwise-http --repetitions 3

# Separate fixed-pair model-versus-surrounding timing and operator profiles.
.venv/bin/python -m scripts.profile_rowwise_decode \
  --output .experiments/new-rowwise-profile

# Run regression tests separately from model performance measurements.
.venv/bin/python -m pytest -q
```

Both scripts refuse an existing output directory and use only existing local
models. HTTP settings explicitly pin attention, batching, and projection flags;
exported user flags cannot silently alter comparison modes. Servers are stopped
in `finally`. The HTTP matrix includes unbatched and rowwise concurrency 1/2,
ordinary tensor batching concurrency 2, both attention implementations, both
checkpoints, and three repetitions. Loading, correctness/lifecycle checks, one
preflight and one warmup per timing case are excluded. Each timed run completes
four 96-token counting responses. Profiles and unprofiled HTTP measurements must
run separately.

See [recorded measurements and their limits](benchmarks/rowwise-decode/README.md)
for results. No new models or dependencies are needed.
