# Retained-memory v1 records

Read the [report](../retained-memory.md) for interpretation, method, and limits.

- [manifest.json](manifest.json): completed protocol, checkpoint/source hashes,
  platform/packages, and session order; two normal runs plus one diagnostic run.
- [normal-1.json](normal-1.json), [normal-2.json](normal-2.json): 30 waves of
  four completed HTTP requests, idle state/RSS snapshots, and five idle samples.
- [diagnostic.json](diagnostic.json): weak object counts, native/Python counters,
  trace differences, explicit GC/allocator probes, recovery, and final cleanup.
- [diagnostic-shutdown.json](diagnostic-shutdown.json): state after the original
  model/worker shutdown while the child still existed for a final observation.
- [summary.json](summary.json): reproducible aggregate observations for 364
  completed workload requests, including four recovery requests.
- [native-api.json](native-api.json): installed Apple SDK header hash, hardware,
  and allocator API source/semantics.
- [memory-profile.png](memory-profile.png): normal/diagnostic idle RSS and separate
  diagnostic native in-use and post-load traced Python curves before interventions.
- [rendering.json](rendering.json): optional plot interpreter/version and source,
  input, and output hashes.

The base commit is merged main before adding the experiment. Source hashes identify
the tested uncommitted driver/diagnostic code. Normal production modules are
unchanged. Optional analysis/plot code is committed alongside these records.
Logs and the initial small pilot remain in ignored `.experiments/`. Raw JSON was
copied from completed sessions without changing the observations.

From the repository root, regenerate the summary without running models:

```bash
.venv/bin/python -m scripts.summarize_memory_profile \
  benchmarks/retained-memory-v1 --output .experiments/memory-summary-recheck.json
```

The normal server exposes no `/__memory` controls. Those belong only to the
explicitly launched private experiment factory and refuse active/queued work.
