# CPU comparison v1 raw records

See the [human-readable report](../cpu-comparison.md) for interpretation and limits.
The dataset contains 80 timing runs / 640 measured requests plus separate
lifecycle, RSS, and native reservation-pressure observations.

- [manifest.json](manifest.json): exact workload, prompt tokens, source/artifact
  hashes, commands, package versions, host, repetition order, and run window.
- [summary.json](summary.json): reproducible per-run aggregate distributions and
  compact lifecycle observations. It does not replace raw sample counts.
- `repetition-N-ENGINE-WORKLOAD-cC.json`: excluded preflight/warmup, measured
  samples, actual token usage, submission offsets, timings, and class statistics.
- `repetition-N-ENGINE-state.json`: state and RSS outside the timing interval.
- [lifecycle-runtime.json](lifecycle-runtime.json) and
  [lifecycle-llama_cpp.json](lifecycle-llama_cpp.json): active/queued disconnect,
  follow-up recovery, five memory observation waves; runtime explicit cancel.
- [admission-pressure-runtime.json](admission-pressure-runtime.json): separate
  two-request token-reservation pressure and explicit-cancellation check.

`runtime` is our serialized PyTorch CPU executor; `llama_cpp` is the pinned
native CPU build described in [the compatibility report](../llama-cpp.md).
The comparison uses the explicit `per_chunk` timestamp policy for llama.cpp and
the default stable policy for our runtime. Timing tests have no concurrent RSS
polling; the memory timeline is a separate session. This is closed-loop load.

Large artifacts and server logs stay under ignored `.experiments/`. Dataset JSON
was copied unchanged from completed experiment directories. The comparison's
project base commit precedes the experiment code; recorded source hashes identify
the tested driver/module/client. The separately rerun admission probe records its
own source hash. Reproduce the summary from the repository root:

```bash
.venv/bin/python scripts/summarize_engine_comparison.py \
  benchmarks/cpu-comparison-v1 --output .experiments/cpu-summary-recheck.json
```
