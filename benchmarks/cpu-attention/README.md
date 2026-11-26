# CPU operator investigation and attention comparison

This phase investigates the previous CPU batching slowdown and tests one optional
change: **SDPA attention instead of eager attention**, with Torch threads fixed
at two. It uses our own runtime and the existing SmolLM2-135M-Instruct and
Qwen2.5-0.5B-Instruct checkpoints. No new model, dependency, GPU backend, or
external engine was introduced. Normal eager/unbatched defaults are retained.

See [activation and API details](../../CPU_TUNING.md).

## What the operator profiles show

CPU profiling runs inside the existing model worker and records operator shapes
for four decode rounds, after a budget-32 prefill and four warmup rounds. Each
profile represents the same pair of requests, producing eight tokens in total.
The largest entries are matrix multiplications (`aten::mm`), with `aten::addmm`
also significant for Qwen. Selected shapes identify model projections:

| Model | Input rows | Weight shape | Interpretation from model dimensions |
| --- | --- | --- | --- |
| SmolLM2 | 1 or 2 × 576 | 576 × 1536 | MLP gate/up projection |
| SmolLM2 | 1 or 2 × 1536 | 1536 × 576 | MLP down projection |
| SmolLM2 | 1 or 2 × 576 | 576 × 49152 | vocabulary output head |
| Qwen | 1 or 2 × 896 | 896 × 4864 | MLP gate/up projection |
| Qwen | 1 or 2 × 4864 | 4864 × 896 | MLP down projection |
| Qwen | 1 or 2 × 896 | 896 × 151936 | vocabulary output head |

For example, SmolLM2's shape-specific gate/up `aten::mm` entry has about
27.21 ms self CPU time across the unbatched profile versus 38.91 ms batched;
Qwen's corresponding entry has 102.23 versus 133.96 ms. These are instrumented
aggregate operator times, **not end-to-end latency estimates**. Shape-based
module attribution is an inference from the architecture; no operator call
stacks were collected. The profiles locate the expensive operator families but
do not establish a specific BLAS implementation or hardware fault as the cause.

The profiles support the earlier observation that the slowdown is substantially
inside the model's matrix calculations, with cache copying adding surrounding
cost. Attention is one component, not the entire bottleneck.

## Small unprofiled screen

The screen tests eager/SDPA × one/two/four threads × batching off/on, with 16
measured decode rounds after four warmup rounds, three repetitions, and setting
order reversed between repetitions. Both requests begin together in this fixed
pair probe; HTTP arrivals and longer growing contexts are different conditions.
All 72 setting/repetition samples matched the baseline generated-token fingerprint.

Selected median pair times (milliseconds for two tokens):

| Model | Setting | Unbatched | Batched |
| --- | --- | ---: | ---: |
| SmolLM2 | eager, 2 threads | 27.61 | 41.06 |
| SmolLM2 | eager, 1 thread | 26.72 | 35.60 |
| SmolLM2 | SDPA, 2 threads | 24.74 | 36.59 |
| Qwen | eager, 2 threads | 71.61 | 89.73 |
| Qwen | eager, 1 thread | 70.02 | 91.54 |
| Qwen | SDPA, 2 threads | 67.71 | 87.30 |

One-thread results are mixed across models and execution modes, and four threads
did not establish a consistent benefit. Thread settings remain unchanged. SDPA
was selected for an end-to-end comparison as a standard supported option with
promising diagnostic results. This screen alone does not justify a serving-speed
claim or a default change.

## Primary HTTP comparison

| Model | Batching | Clients | Eager tokens/s | SDPA tokens/s | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2-135M | off | 1 | 68.75 | 77.26 | +12.4% |
| SmolLM2-135M | off | 2 | 68.66 | 77.77 | +13.3% |
| SmolLM2-135M | on | 2 | 45.70 | 42.60 | -6.8% |
| Qwen2.5-0.5B | off | 1 | 26.41 | 27.88 | +5.5% |
| Qwen2.5-0.5B | off | 2 | 26.22 | 28.01 | +6.8% |
| Qwen2.5-0.5B | on | 2 | 21.19 | 21.06 | -0.6% |

SDPA improved unbatched throughput in these samples: about 12–13% for SmolLM2
and 6–7% for Qwen. It did not improve the batched path: SmolLM2's median fell
6.8%, while Qwen was nearly unchanged. The retained change is an explicit option
for the useful unbatched case, not a solution to the larger batching regression.
The default stays eager/unbatched; no automatic policy switches are introduced.

Each value is the median of three complete runs; throughput aggregates all
clients' verified output tokens. The run contains **144 measured completions
across 36 runs**, each producing 96 output tokens (13,824 measured tokens).
For each model and attention implementation, unbatched execution is measured at
concurrency one/two and batched execution at concurrency two. Batched
single-client HTTP timing is intentionally not included.

The prompt is `Count from 1 to 500. Include every number.`. Loading, correctness,
lifecycle, one preflight stream, and one warmup per timing case are excluded.
This is a repeated counting workload, not an answer-quality benchmark. First
content timings and individual samples are in the raw reports.

Each model/repetition/attention/batching combination starts a fresh loopback
server on port 8020. Setting order reverses between repetitions. Python tests,
profiling, and model experiments were not run concurrently with these timing
runs. Other host activity and thermals remain uncontrolled. Small differences
should not be generalized to all workloads or hardware. In filenames, `single`
means the unbatched executor; `c1`/`c2` indicate client counts.

The candidate screen and HTTP results are separate evidence. A gain in the small
fixed pair does not guarantee a gain in longer requests with real arrivals,
different cache alignment, and growing context. This phase does not establish
that tensor batching is the fastest CPU execution path.

## Correctness, cancellation, and memory

The direct reference always uses eager attention, CPU float32, greedy decoding,
and `repetition_penalty=1.0`. Each first-repetition combination checks three
unequal native prompts (France/counting/Norwegian), an identical counting pair,
and multi-turn JSON chat. Across both models and all four settings, **40 native
responses** match reference token IDs/text and **eight history responses** match
reference text/usage. Reference agreement is not a factual-quality score.

SDPA is also checked in eight paired real-HTTP active cancel/disconnect scenarios
with the peer completing, and four queued disconnect scenarios with both active
slots occupied. Every settled correctness, timing, and lifecycle check ends with
zero active/queued requests, logical reserved/used tokens, and owned KV tensor
bytes. Health verifies the actual requested attention implementation and genuine
multi-row calls where batching is enabled.

The candidate uses the same independent dense cache splits and settled worker
barrier as the previous phase. RSS values are readiness/post-run samples, not
peaks or a physical memory limit. No many-client, long-context, GPU/MPS,
arbitrary-model, or universal bitwise-output guarantee is established here.

## Records and reproduction

- [HTTP manifest, settings, source fingerprints, and summary](raw/manifest.json)
- [SmolLM2 operator/screening data](screening/smollm2.json)
- [Qwen operator/screening data](screening/qwen.json)
- [Verification summary](verification.json)
- `raw/*-correctness.json` and `raw/*-lifecycle.json`: reference and cleanup checks.
- `raw/*-c1.json` / `raw/*-c2.json`: per-request timings, usage, backend/counters,
  RSS samples, and settled state. Logs record server startup/shutdown.

The screening ran against merged main before adding the optional loader flag;
it dynamically switched attention with the Transformers interface and changed
thread count only inside its worker. Production math/caching was otherwise the
baseline. HTTP source fingerprints identify the later implementation with the
new explicit option. Old benchmark datasets remain unchanged.

```bash
.venv/bin/python -m scripts.investigate_cpu_inference \
  --output .experiments/new-operator-screen
.venv/bin/python -m scripts.verify_cpu_attention \
  --output .experiments/new-attention-http --repetitions 3
.venv/bin/python -m pytest -q
```

Run sequentially with existing model files and dependencies. Output paths must
be new; the HTTP runner refuses an occupied port and closes owned servers in
`finally`. No downloads occur. See [CPU_TUNING.md](../../CPU_TUNING.md) for the
optional server command and unchanged defaults.
