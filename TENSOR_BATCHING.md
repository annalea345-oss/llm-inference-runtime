# Optional real tensor batching

The local CPU executor can combine multiple requests into one PyTorch model
forward. This is separate from the existing continuous scheduler: the scheduler
already chooses work for several requests, while tensor batching changes how
that selected work is calculated.

The feature is **opt-in**. The original unbatched path remains the default and
the comparison baseline. It uses the same downloaded model, tokenizer, greedy
argmax, CPU float32 weights, two Torch threads, API, and scheduling limits.
Batching does not change the model's training or add repetition controls.
The [later CPU investigation](CPU_TUNING.md) adds optional SDPA attention
independently of batching; eager attention remains the default and the baseline
for this milestone's original measurements.
The measured two-client prototype was slower for both checkpoints on this Mac;
treat it as experimental. [Recorded results](benchmarks/tensor-batching/README.md)
explain the throughput and latency tradeoffs.

The subsequent [rowwise CPU projection experiment](ROWWISE_DECODE.md) targets
the small matrix operations inside two-request forwards. It requires a separate
flag; the original tensor-batching measurements remain unchanged.

## Run

With an already downloaded checkpoint and the existing virtual environment:

```bash
LLM_TENSOR_BATCHING=1 LLM_MODEL_PATH=.models/qwen2.5-0.5b \
  .venv/bin/python -m uvicorn llm_runtime.local:create_local_app \
  --factory --host 127.0.0.1 --port 8000
```

Use `.models/smollm2` to run the original model. Omit `LLM_TENSOR_BATCHING`, or
set it to `0`, for unbatched execution. Any value other than `0` or `1` fails
startup. Model installation/download instructions remain in [LOCAL_MODEL.md](LOCAL_MODEL.md).
No new model or dependency is required for batching.

The browser, Android app, and Terminal clients use the same API as before. To
exercise batching, generate from two clients at once. The local factory still
admits at most two active requests; the flag does not increase that capacity.

`GET /health` exposes these executor fields:

- `tensor_batching`: whether the optional path is enabled.
- `forward_calls`: attempted model calls through the executor since load.
- `batched_forward_calls`: those calls with more than one row.
- `max_forward_batch_size`: largest observed call size.
- `kv_tensor_bytes_after_last_step`: owned per-request KV tensors after the most
  recently settled step. It excludes model weights and temporary batch tensors.

Enabled does not mean every call has multiple rows. Single-client work, a shared
budget that selects only one step, mixed prefill/decode work, and EOS/output-cap
rows can leave a singleton or require no new forward. Counters describe actual
attempted forward shapes, including a call that subsequently fails.

## Architecture and ownership

```mermaid
flowchart LR
    A[Scheduler selects request steps] --> B[Group by phase]
    B --> C[Pad inputs and caches; mask padding]
    C --> D[Shared PyTorch forward]
    D --> E[Copy dense caches for each request]
    E --> F[Wait for execution barrier]
    F --> G[Stream results and release finished requests]
```

1. `ContinuousBatchScheduler` keeps its admission, fairness, token budgets, and
   reservation rules. Its execution barrier optionally calls `execute_batch()`
   with the selected steps. Executors without that method still use individual
   `execute()` calls. Results must match the input order and length; a malformed
   result or a shared exception fails the affected selected work safely.
2. `LocalModelExecutor.execute_batch()` submits one operation to the existing
   serial model worker when enabled. The unbatched setting retains the original
   individual `execute()` calls, including observation hooks.
3. Within that operation, compatible prefill steps share a forward; decode steps
   that need another prediction share a separate forward. Phase groups follow
   their first appearance in the scheduler plan. EOS and output-cap rows return
   their final token without spending another model forward.
4. Temporary tensors left-pad past caches and new input chunks independently.
   A combined attention mask excludes both padding regions. Explicit position
   IDs use each request's logical token positions; physical cache positions index
   the temporary padded sequence. This supports different prompt, cache, and
   prefill-chunk lengths, including a new row alongside a partly prefilled row.
5. After the forward, the executor removes padding from each output cache and
   creates dense per-request tensors. Splitting copies storage: a surviving
   request never retains a view of a cancelled peer's batched storage. All output
   cache candidates are built before committing the group to request state.
6. The scheduler consumes results and releases completed/cancelled state only
   after the shared worker operation settles. Repeated shutdown cancellation
   cannot free tensors that PyTorch is still using. Waiting cancellations still
   remove waiting work immediately, including while a batch forward is busy.

An invalid row is isolated before calculation. A failed model forward fails every
participating row in that phase group; unrelated completed rows or other phase
groups can still return their results. Failed-batch exception frames are detached
before returning them so they do not retain batch tensors. Error type/message is
logged, while HTTP responses retain the existing generic execution-error behavior.

Supported model architectures are full-attention **Llama** and **Qwen2**, covering
our SmolLM2-135M-Instruct and Qwen2.5-0.5B-Instruct checkpoints. Enabling this path
on another architecture or a sliding-attention configuration fails explicitly.
It uses the installed Transformers 4.57.x `DynamicCache` interface. There is no
shared persistent batch cache, paged allocator, prefix sharing, quantization,
CUDA/MPS backend, or change to greedy decoding in this milestone.

Logical KV capacity is still an admission limit in tokens, **not a physical RAM
cap**. Padding, packed caches, split copies, attention, and logits consume extra
temporary memory. Owned cache bytes returning to zero establishes request-state
cleanup; it does not require process RSS to return to its startup value.
Scheduler token budgets count real request tokens; padding adds compute beyond
that logical count.

## Tests and measurements

```bash
.venv/bin/python -m pytest -q
```

Offline tests use tiny random Llama/Qwen2 models and independent `.generate()`
references. They cover ragged prefill/decode, shared multi-row decode forwards,
late admission, mixed phases, both policies, budgets including one, empty versus
existing past caches, EOS/output-cap row removal, exact token/text agreement,
independent cache storage, failure/recovery, weak-reference cleanup, waiting and
active cancellation, and repeated shutdown while a shared forward is blocked.
The optional hook, bad return shape/type, and strict factory flag have separate
tests. Existing simulator/API/browser tests remain part of the full suite.
The recorded final run passed **485 Python tests**, including 41 new cases.

The reproducible real-model runner uses only local files and starts one temporary
loopback server on port 8020 at a time, stopping it in `finally`:

```bash
.venv/bin/python -m scripts.verify_tensor_batching \
  --output .experiments/my-tensor-batching-run --repetitions 3
```

The output directory must be new. By default it checks both downloaded models;
use `--models qwen` or `--models smollm2` for one. It records source/model hashes,
direct Transformers references, real HTTP token/text agreement, paired active
cancellation/disconnect with a surviving peer, queued disconnect with both slots
busy, zero final request/KV state, and timing runs at concurrency one/two. Warmups,
preflight, correctness, and lifecycle work are excluded from measured samples.

See [the recorded CPU results](benchmarks/tensor-batching/README.md) for the actual
tradeoffs. Timing one prompt at concurrency two is not a general model-quality,
long-context, many-client, or GPU-performance result. The supplied counting prompt
is deliberately long enough to exercise repeated decode steps; it is not an
answer-quality test. Avoid running model tests or other CPU-heavy tools during a
timing run. Small repeated samples still reflect host load and thermal conditions.

## Implementation references

- [Transformers 4.57 cache/attention-mask explanation](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
- [DynamicCache implementation at installed version 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/cache_utils.py)
- [Qwen2 positions, masks, and cache handling at 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2/modeling_qwen2.py)
