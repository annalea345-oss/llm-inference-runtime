# Pinned llama.cpp CPU compatibility check

Verified **2026-11-09** on the existing Apple M3 / 16 GB development host.
This is a conversion and HTTP compatibility milestone, **not a comparative speed
benchmark**. It leaves the project's inference backend unchanged.

Follow-up: the [repeated CPU comparison](cpu-comparison.md) now tests longer and
mixed workloads. It found changing `created` timestamps in longer external streams,
handled by an explicit client policy, and records comparative results separately
from this original short compatibility check.

## What passed

- Built upstream commit `a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5` from its
  source archive. Metal, Accelerate, BLAS, and OpenMP were disabled explicitly.
- Converted our existing SmolLM2 checkpoint to float32 GGUF. Every one of its
  **272 tensors** matches the original values widened to float32, after accounting
  for the HF-to-GGML query/key RoPE layout permutation. No quantization was used.
- Verified BOS `1`, EOS `2`, padding `2`, and the embedded chat template against
  the checkpoint. Four fixtures (single turn, system instruction, multiple turns,
  and Unicode) produced exactly matching rendered prompts and prompt token IDs
  through Transformers and llama.cpp's `/apply-template` plus `/tokenize`.
- Our portable benchmark passed required-usage preflight and completed **eight
  measured smoke requests**, four each at client concurrency 1 and 2, with two
  excluded warmups and one excluded preflight per setting. Each measured response
  reported 40 prompt tokens and 8 completion tokens, finishing on EOS.
- A separate non-streaming request answered: `The capital of France is Paris.`
  This is a smoke example, not an answer-quality evaluation.
- Final `/slots` showed both slots idle. Metrics showed zero processing/deferred
  requests and zero cached-prefix tokens across the server lifetime. This verifies
  an idle server and no reported prefix hits, not physical memory returning to zero.
  The temporary server was stopped afterward.

The exact source/archive, executable and shared-library hashes, build options,
compiler, and launch command are in [llama-cpp-build.json](llama-cpp-build.json).
Checkpoint/tokenizer/GGUF hashes, prompt fixtures, raw HTTP samples, server
properties, final slots, and metrics are in [llama-cpp-smoke.json](llama-cpp-smoke.json).
Their project base commit identifies the checkout before this milestone;
the verifier hash identifies the tested script.

## What was downloaded, built, and stored

The original checkpoint remains under ignored `.models/smollm2`. Its stored
weights are bfloat16; both our existing runtime and this conversion widen them
to float32 for execution. The new GGUF is approximately **538 MB**.

Ignored `.experiments/` contains the pinned MIT-licensed llama.cpp source archive
and extracted source, an isolated tools environment, compiled CPU binaries,
the converted GGUF, and temporary logs/help text. Nothing from that directory is
committed. No second checkpoint was downloaded, and no project dependency or
server backend was replaced. The source license remains in the extracted tree.

The only installed experiment tools were CMake 4.1.3 and SentencePiece 0.2.1.
Conversion used the existing project PyTorch/Transformers/NumPy plus upstream
`gguf-py` directly from the source tree. This differs from upstream's pinned
conversion requirements; successful conversion and exact tensor verification
establish this specific combination, not general support for other models.

## Reproduce the setup

Requires the existing project `.venv`, downloaded checkpoint, Apple command-line
compiler tools, and enough disk space. Run from the repository root. First obtain
and safely extract the [pinned source archive](https://codeload.github.com/ggml-org/llama.cpp/tar.gz/a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5)
under `.experiments/`; verify its SHA-256 against the build manifest. The extracted
directory is named `llama.cpp-a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5`.

```bash
.venv/bin/python -m venv --system-site-packages .experiments/tools-venv
.experiments/tools-venv/bin/python -m pip install cmake==4.1.3 sentencepiece==0.2.1

GIT_CEILING_DIRECTORIES="$PWD/.experiments" .experiments/tools-venv/bin/cmake \
  -S .experiments/llama.cpp-a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5 \
  -B .experiments/llama-build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
.experiments/tools-venv/bin/cmake --build .experiments/llama-build --target llama-server -j 2

PYTHONPATH=.experiments/tools-venv/lib/python3.12/site-packages .venv/bin/python \
  .experiments/llama.cpp-a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5/convert_hf_to_gguf.py \
  .models/smollm2 --outtype f32 --outfile .experiments/smollm2-f32.gguf
```

The `PYTHONPATH` uses the verified Python 3.12 layout; adjust it for another Python
minor version. Existing project packages provide the rest of the converter's
dependencies. The tools environment is separate from the project's `.venv`.

The Git ceiling prevents an archive build from borrowing this parent project's
commit as its own version. The binary consequently reports `b0-unknown`; the
archive and binary/library hashes establish provenance instead. No upstream
source patch was needed. The upstream build guide confirms that Metal is enabled
by default on Macs; see the [pinned build documentation](https://github.com/ggml-org/llama.cpp/blob/a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5/docs/build.md).

Start this separate server in one terminal:

```bash
.experiments/llama-build/bin/llama-server \
  --model .experiments/smollm2-f32.gguf --alias local-model \
  --host 127.0.0.1 --port 8018 --offline \
  --threads 2 --threads-batch 2 --threads-http 2 \
  --ctx-size 8192 --parallel 2 --kv-unified --kv-unified-per-slot 8192 \
  --batch-size 32 --ubatch-size 32 \
  --n-gpu-layers 0 --no-kv-offload --cache-type-k f32 --cache-type-v f32 \
  --flash-attn off --cache-ram 0 --no-cache-prompt --no-cache-idle-slots \
  --jinja --metrics --slots
```

Wait for its listening message. From another terminal:

```bash
.venv/bin/python scripts/verify_llama_cpp.py \
  --source-dir .experiments/llama.cpp-a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5 \
  --gguf .experiments/smollm2-f32.gguf \
  --output .experiments/llama-cpp-recheck.json
```

The verifier only checks existing artifacts and an already running local server;
it does not install, download, build, or launch. It fails on mismatched weights,
templates/tokens, failed streams, absent usage, or non-idle final slots. Use Ctrl-C
in the server terminal when finished. The normal project start command continues
to run our own engine; this experiment does not route the browser to llama.cpp.

## Limits and next experiment

This configuration aligns CPU execution, float32 weights/KV, two model threads,
8192-token context settings, two active slots, and disabled prefix reuse. The
server has a shared KV pool and retains its allocated storage while idle; our
logical slot cap does not enforce an equivalent physical memory limit. Different
kernels, libraries, HTTP stacks, admission rules, prefill chunking, and actual
tensor batching remain experimental variables. Equal token IDs do not establish
bitwise-identical logits or every possible continuation.

We have not measured a speed advantage over the current runtime. The samples
are single short smoke runs; do not compare their timings with last week's HTTP
baseline to infer a winner. No sustained memory, cancellation/disconnect,
overload-recovery, fairness, or long-context experiment was run here. No upstream
missing feature or bug is claimed.

Next: run both engines with versioned short/long/mixed workloads, repeated runs,
matched settings, and alternating order on an otherwise idle host. Add bounded
cancellation/recovery and memory observations. Follow the
[comparison protocol](../research/engine-comparison/EXPERIMENT_PLAN.md) before
choosing a runtime improvement such as real tensor batching or decision traces.
