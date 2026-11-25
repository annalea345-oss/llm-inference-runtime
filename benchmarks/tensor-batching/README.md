# CPU tensor batching: correctness and tradeoffs

This milestone adds genuine multi-request PyTorch forwards to **our own
runtime**. It compares the new optional path against our existing unbatched
executor on the same Mac, using the same model/settings. It is not a new
llama.cpp comparison or a comparison between the quality of two models.

The implementation is correct for the checked cases, but the measured two-client
CPU throughput is lower. **Unbatched execution remains the default.** See
[activation and architecture](../../TENSOR_BATCHING.md).

## Primary comparison

| Model | Clients | Unbatched tokens/s | Batched tokens/s | Throughput change | First content off/on (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M | 1 | 68.50 | 69.02 | +0.8% | 68.4 / 68.4 |
| SmolLM2-135M | 2 | 68.99 | 45.83 | -33.6% | 125.5 / 146.1 |
| Qwen2.5-0.5B | 1 | 26.22 | 26.03 | -0.7% | 225.4 / 240.1 |
| Qwen2.5-0.5B | 2 | 26.34 | 20.58 | -21.9% | 416.3 / 378.4 |

Values are medians of three complete runs per model/mode/concurrency setting.
Throughput is aggregate output tokens per second across all clients, verified by
terminal usage. First-content values are medians of each run's client-received
p50, not model-internal token timestamps. Small single-client differences are
not evidence of a useful speedup. First-content and total throughput can move in
different directions.

The primary run contains **96 measured completed requests across 24 runs**:
two models × two execution modes × concurrency one/two × three repetitions ×
four requests. Every measured request produced 96 output tokens (9,216 total).
References, lifecycle work, server loading, one preflight stream, and one warmup
per timing run are excluded from those counts and elapsed times.

## Exactly what was run

- Same Apple M3 / 16 GB Mac, CPU float32 weights, eager attention, two Torch
  intra-op threads, one server/worker process, round-robin scheduling, token
  budget 32, maximum two active requests, 8,192 logical KV slots.
- **SmolLM2-135M-Instruct** (Llama), revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`, and
  **Qwen2.5-0.5B-Instruct** (Qwen2), revision
  `7ae557604adf67be50417f59c2c2f167def9a775`. These are the existing downloaded
  checkpoints; model/tokenizer/config/license file hashes are in each reference
  record. No model was downloaded, converted, quantized, or trained.
- Greedy argmax throughout. Direct Transformers references explicitly use
  `do_sample=False` and `repetition_penalty=1.0`, matching the runtime rather than
  Qwen's published sampling/repetition defaults. No answer-quality feature or
  decoding-policy change is included.
- The measured prompt is `Count from 1 to 500. Include every number.`, with
  `max_tokens=96`, closed-loop HTTP streaming at concurrency one/two. This keeps
  decoding active; it does not test factual accuracy or resolve repetition.
- A fresh loopback Uvicorn server on port 8020 for each mode/repetition. Mode
  order alternates off/on and on/off between repetitions. Correctness/lifecycle
  checks run outside timing. All owned servers stop in `finally`.
- The published run was performed without concurrent project tests or profiling.
  Earlier exploratory trials remain ignored in `.experiments/`; they are not
  included in these primary numbers. Other host activity and thermal behavior
  remain uncontrolled, so this is a small local experiment, not a general ranking.

## Correctness and lifecycle evidence

For each model and mode, the first repetition checked three unequal native
prompts (France, counting, Norwegian), two simultaneous identical counting
prompts, and a multi-turn JSON chat history. The **20 native responses** matched
direct Transformers token IDs and canonical decoded text; **four history
responses** matched reference text and generated-token usage. EOS/output caps
were checked against the references. A wrong answer from the model is not counted
as factual success merely because both paths agree.

Enabled mode's health counters show actual multi-row forwards, with maximum
batch size two. Disabled mode stays at one. Timing-run counter deltas also verify
that the enabled two-client runs actually used multi-row model calls; this is not
just concurrent scheduling of individual forwards.

Real HTTP checks cover eight paired active-cancel/disconnect scenarios: one
client stops or disconnects while its peer completes normally. Four queued
disconnect checks remove a waiter with both active slots still occupied. Every
settled correctness, timing, and lifecycle check ends with zero active/queued
requests, zero logical reserved/used tokens, and zero owned KV tensor bytes.
Offline tests additionally gate a shared forward and verify that repeated
shutdown cancellation cannot release state prematurely.

## Diagnostic: where the CPU time went

The separate [decode diagnostic](decode-profile.json) times 24 decode rounds for
one fixed pair, after four warmup rounds and the real budget-32 prefill plan. It
splits executor wall time into time inside `model.forward` and surrounding work.
The latter includes cache packing/splitting, token decoding, input assembly, and
worker/async overhead; it is not an exclusive measurement of cache copying.

| Model | Mode | Executor total (ms/pair) | Inside model.forward | Other executor work |
| --- | --- | ---: | ---: | ---: |
| smollm2 | off | 29.47 | 28.91 | 0.56 |
| smollm2 | on | 48.89 | 45.55 | 3.34 |
| qwen | off | 72.71 | 72.06 | 0.65 |
| qwen | on | 87.21 | 85.04 | 2.16 |

Each row above represents a **pair of tokens**: two separate model calls when
off, one two-row model call when on. `model.forward` includes its own KV updates
and attention calculations. Generated-token fingerprints match between modes.
The timing wrapper is removed before unloading each model to avoid retaining it
through a bound-method closure.

This diagnostic supports slower CPU model-call shapes as the larger contributor
in this particular setup, with additional surrounding executor cost. It does
not identify a specific BLAS/operator bottleneck, prove how a GPU would behave,
or substitute for the end-to-end comparison. Persistent batch-cache ownership,
CPU-kernel/backend choices, and reduced packing costs are future experiments.

## Memory and limits

The split caches have independent storage and no retained padding; tests inspect
storage sizes/pointers and weak-reference collection after injected failures.
That protects per-request cleanup but adds temporary copies. Readiness and
post-run process RSS samples are in the timing reports. These are **not peak RSS**
or a physical memory cap. Model weights, padding, attention/logits, allocator
retention, and the OS affect RSS independently of live request KV ownership.
Across both modes and fresh processes, post-run samples ranged from 0.86–1.05 GiB
for SmolLM2 and 1.98–2.28 GiB for Qwen. These ranges include model/process memory
and are not isolated measurements of batching's temporary memory overhead.

This experiment covers two short-context checkpoints, one repeated timing
prompt, and at most two active clients. It does not establish long-context
memory safety for arbitrary workloads, many-client scaling, GPU speed, or
universal bitwise agreement for every possible prompt/configuration. API 26/35,
Android rendering, and browser UI are not changed by this milestone.

## Reproduce and inspect

```bash
.venv/bin/python -m scripts.verify_tensor_batching \
  --output .experiments/new-tensor-batching-run --repetitions 3
.venv/bin/python -m scripts.profile_tensor_batching \
  --output .experiments/new-decode-profile.json --steps 24
.venv/bin/python -m pytest -q
```

Run these sequentially. The runner refuses an existing output directory or an
occupied port 8020. It requires existing model files and dependencies and never
downloads them. Run on a trusted local machine with no competing model tests.

- [Primary manifest, settings, hashes, and summary](raw/manifest.json)
- [SmolLM2 direct reference fixtures](raw/smollm2-reference.json)
- [Qwen direct reference fixtures](raw/qwen-reference.json)
- Per-mode/per-repetition `*-c1.json` / `*-c2.json`: individual receipt timings,
  usage, counters, RSS samples, and final ownership state.
- `*-correctness.json` / `*-lifecycle.json`: reference agreement, cancellation,
  disconnect, peer completion, and cleanup. Server logs record startup/shutdown.

The implementation is available for study and further optimization. Current CPU
evidence supports keeping it optional rather than replacing the unbatched path.
The final regression run passed **485 Python tests**. The manifest's base commit
identifies the branch starting point; executed-source hashes identify the tested
implementation, which was benchmarked before the feature commit was created.
