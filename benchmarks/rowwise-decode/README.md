# Rowwise CPU projection investigation

This follows the [CPU operator/SDPA study](../cpu-attention/README.md). That study
found higher costs in matrix projections for two-row model calls. This phase
changes the grouping of those calculations, while keeping a genuine shared
model forward, batched attention, and batched KV updates.

## Scope

- Apple M3 Mac, 16 GB RAM; CPU float32, two Torch intra-op threads.
- Existing HuggingFaceTB/SmolLM2-135M-Instruct, revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Existing Qwen/Qwen2.5-0.5B-Instruct, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Python/Torch/Transformers/platform versions, source hashes, model file hashes,
  timestamps, load samples, and settings are in the raw manifests/references.
- No model/dependency download, external-engine rerun, Android build, GPU/MPS,
  quantization, sampling, model training, or answer-quality improvement.

`single` means the original unbatched executor; it can serve one or two concurrent
HTTP clients. `batch` means ordinary padded tensor batching. `rowwise` means that
same shared model/cache path, with a plain Linear specialization for CPU float32
input shape `[2,1,hidden]`. Larger batches and other shapes use normal Linear.
One-token prefill chunks may also match; the branch is shape-based.

## HTTP protocol

[Raw HTTP data](raw/manifest.json) covers three repetitions of both models,
both eager/SDPA attention implementations, and all three projection variants.
Unbatched and rowwise variants cover concurrency one and two; ordinary batching
covers concurrency two. That is **60 measured runs, 240 completed requests, and
23,040 generated tokens**. Each run completes four 96-token counting responses.
Loading, correctness and lifecycle work, one preflight, and one warmup per timing
case are excluded.

Each model/repetition/attention/variant gets a fresh process, for **36 temporary
servers**. Setting order reverses on alternate repetitions. Modes are explicitly
set in the child environment and verified through health; c2 batched cases must
increase real batched-forward counters, and c2 rowwise cases must additionally
increase rowwise Linear counters. c1 rowwise cases must not use that specialization.
All modes keep two Torch threads. Host load/thermals are not controlled; medians
of three repetitions are observations, not statistical guarantees.

Median completed output tokens/second, concurrency two:

| Model | Attention | Unbatched | Ordinary batching | Rowwise batching | Rowwise vs ordinary batch | Rowwise vs unbatched |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M | eager | 69.16 | 46.45 | 68.30 | +47.0% | -1.2% |
| SmolLM2-135M | sdpa | 70.88 | 45.23 | 72.15 | +59.5% | +1.8% |
| Qwen2.5-0.5B | eager | 27.03 | 21.71 | 29.22 | +34.6% | +8.1% |
| Qwen2.5-0.5B | sdpa | 28.42 | 20.79 | 29.50 | +41.9% | +3.8% |

Rowwise improves on ordinary tensor batching by **35–60%** in these medians.
The practical comparison with the existing unbatched executor is smaller:
SmolLM2 is approximately tied (-1.2% eager, +1.8% SDPA), while Qwen gains
8.1% eager and 3.8% SDPA. The ordinary batching regression is substantially
reduced, not a universal speedup over the original runtime.

Single-client medians (unbatched → rowwise option, tokens/second):

- SmolLM2, eager: 66.40 → 64.31 (-3.1%).
- SmolLM2, sdpa: 72.53 → 73.96 (+2.0%).
- Qwen, eager: 26.81 → 26.46 (-1.3%).
- Qwen, sdpa: 28.43 → 28.37 (-0.2%).

No specialized calls occur in these c1 runs, but the installed wrapper still
takes its fallback branch. Small c1 differences and run-to-run variation are
another reason not to promote the option by default. First-content latency
quantiles are recorded separately in the manifest/raw runs; throughput alone
does not establish better responsiveness for every case.

Compare modes within the same attention implementation to attribute projection
changes. A comparison of rowwise+SDPA against unbatched+eager also changes attention.
The original default remains eager, unbatched, and rowwise disabled. These results
are specific to this workload and CPU and do not justify a universal default.

## Model and cache costs

The [separate worker diagnostic](profiles/manifest.json) uses the real budget-32
prefill planner and a fixed pair, followed by four warmup rounds and 32 measured
decode rounds. Three repetitions reverse setting order. Timing samples are
separate from four-round PyTorch operator profiles with shape recording. The
diagnostic reuses one model with the projection wrappers installed, disabling
the specialization for its single/batch baselines; the HTTP baselines load ordinary
Linear modules without wrappers.

Each timing records total pair wall time, wall time inside `model.forward`, and
the remaining surrounding executor time. The remainder includes cache packing
and splitting, input assembly, token decoding, and counters; it does **not** isolate
cache copies. It excludes HTTP and asynchronous worker submission overhead. No
production cache-layout or cache-ownership change is made in this milestone.

Median of per-repetition mean milliseconds per pair. Columns are independently
aggregated medians, so they need not sum exactly.

| Model | Attention | Variant | Pair wall ms | Model wall ms | Surrounding wall ms |
| --- | --- | --- | ---: | ---: | ---: |
| SmolLM2 | eager | single | 27.38 | 27.14 | 0.25 |
| SmolLM2 | eager | batch | 38.03 | 35.38 | 2.61 |
| SmolLM2 | eager | rowwise | 22.44 | 19.69 | 2.74 |
| SmolLM2 | sdpa | single | 23.84 | 23.63 | 0.21 |
| SmolLM2 | sdpa | batch | 36.06 | 33.33 | 2.72 |
| SmolLM2 | sdpa | rowwise | 19.34 | 16.78 | 2.56 |
| Qwen | eager | single | 69.52 | 69.11 | 0.39 |
| Qwen | eager | batch | 85.47 | 83.78 | 1.72 |
| Qwen | eager | rowwise | 62.94 | 61.19 | 1.75 |
| Qwen | sdpa | single | 66.93 | 66.56 | 0.37 |
| Qwen | sdpa | batch | 83.32 | 81.53 | 1.79 |
| Qwen | sdpa | rowwise | 58.64 | 56.87 | 1.77 |

The short probe locates most of the recovered time inside the model call.
The surrounding batched overhead remains approximately 2.6–2.7 ms for SmolLM2
and 1.7–1.8 ms for Qwen in this fixed pair. It includes cache work but does not
isolate a cache-layout bottleneck. Rowwise changes no cache layout.

The larger fixed-pair gain, particularly for SmolLM2, does not fully carry over
to HTTP throughput. Longer growing caches and different arrivals are additional
workload differences; their separate contributions were not isolated.

For example, the eager four-round operator profiles show the following gate/up
projection entries (module attribution is inferred from dimensions):

| Model | Ordinary-batch matrix shapes | Self CPU ms | Rowwise matrix shapes | Self CPU ms |
| --- | --- | ---: | --- | ---: |
| SmolLM2 | `[2,576] × [576,1536]` | 41.19 | `[1,576] × [576,1536]` | 16.65 |
| Qwen | `[2,896] × [896,4864]` | 129.66 | `[1,896] × [896,4864]` | 86.84 |

The rowwise entries include more calls; values are cumulative self CPU sums over
the same four pair rounds, not per-call times. Both still appear as `aten::mm`;
the narrower call shape does not by itself establish a particular BLAS routine.

The specialization calls ordinary PyTorch functional Linear once per request row,
then concatenates the two results. Operator names and input shapes establish the
changed grouping. They do not prove which internal BLAS kernel ran or a hardware
fault. Profiled self CPU sums are not HTTP throughput or uninstrumented wall times.
The fixed-pair diagnostic has shorter contexts and steadier arrivals than the
HTTP test; it must not substitute for HTTP results.

## Correctness, cancellation, and memory

First-repetition checks cover all **12 model/attention/variant configurations**.
Three unequal native prompts and one identical counting pair per configuration
give **60 exact token-ID/text matches** against independent eager Transformers
references. Twelve multi-turn JSON checks match reference text and token usage.
These fixtures include English, Norwegian/Unicode, counting, and conversation
history. They do not prove bitwise equivalence for all inputs or repair factual
errors made by small models.

Rowwise candidate lifecycle checks cover each model/attention combination: **eight
paired explicit-cancel/disconnect scenarios**, with a surviving peer completing,
and **four queued disconnects** while both active slots stay occupied. All settled
correctness, timing, and lifecycle states have zero active/queued requests,
reserved/used logical tokens, and owned KV bytes. Readiness/post-run RSS samples
are not peak measurements or physical memory guarantees.

Offline regression coverage includes activation/default/invalid options, parameter
identity and state names, bias/no-bias and shape/dtype/autograd/training fallbacks,
ragged greedy agreement for both architectures/attention options, both scheduling
policies and token budgets, compact independent cache ownership, EOS/output caps,
failed-forward cleanup/recovery, shared-decode cancellation/repeated shutdown, and
weak-reference release of specialized modules on unload. No scheduler, admission,
API, or client behavior changed.

The final regression suite passed **570 Python tests** (60 additional cases).
Post-run RSS samples were **0.64–1.01 GiB** for SmolLM2 and **1.96–2.30 GiB**
for Qwen. All 36 owned HTTP server logs end in graceful application shutdown.
All 10/11 model-file hashes match the preceding phase. Existing model files,
older benchmark data, local IDE files, and the user’s untracked `error.log`
are preserved. No temporary model server remains running.

## Files and reproduction

- [Activation, implementation boundaries, and commands](../../ROWWISE_DECODE.md)
- [HTTP measurements, references, lifecycle proofs, and server logs](raw/manifest.json)
- [Unprofiled pair timings and shape-aware operator profiles](profiles/manifest.json)
- [Final verification](verification.json)

Run the two investigation scripts separately, then the regression suite. Both
scripts refuse an existing output directory. The HTTP runner refuses an existing
port-8020 listener and stops every owned server in `finally`. Earlier benchmark
files are preserved; this dataset records a new comparison rather than rewriting
historical baselines.
