"""Offline operator profiling and candidate screening; no server or downloads.

Profiler samples are separate from unprofiled timings. Reuses one loaded model
per checkpoint, clearing request caches between settings. Normal defaults remain
unchanged. Run without competing model workloads.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter

from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.local_executor import LocalModelExecutor
from llm_runtime.protocol import ExecutionStep, InferenceRequest
from scripts.verify_tensor_batching import MODELS, COUNT


def prepare(executor, steps):
    requests = [InferenceRequest(COUNT, steps + 16) for _ in range(2)]
    for r in requests:
        r.prompt_token_ids = executor.tokenize(COUNT)
        r.phase = 'prefill'
    planner = ContinuousBatchScheduler(executor, RuntimeConfig(max_active_requests=2))
    planner._active.extend(requests)
    while any(r.phase == 'prefill' for r in requests):
        planned = planner._plan()
        results = run_steps(executor, planned)
        assert not any(isinstance(r, BaseException) for r in results)
        for step in planned:
            r = step.request
            if step.phase == 'prefill':
                r.prefilled_tokens += step.token_count
                if r.prefilled_tokens == r.prompt_token_count:
                    r.phase = 'decode'
            else:
                r.generated_tokens += 1
    return requests


def run_steps(executor, steps):
    return (executor._execute_batch(tuple(steps)) if executor.tensor_batching
            else [executor._execute(s) for s in steps])


def advance(executor, requests):
    results = run_steps(executor, [ExecutionStep(r, 'decode', r.generated_tokens, 1) for r in requests])
    assert not any(isinstance(r, BaseException) for r in results)
    for r in requests:
        r.generated_tokens += 1
    return [r.token_id for r in results]


def sample(executor, threads, attention, batched, steps, operators=False):
    torch = executor.torch
    torch.set_num_threads(threads)
    executor.model.set_attn_implementation(attention)
    executor.tensor_batching = batched
    requests = prepare(executor, steps)
    try:
        for _ in range(4):
            advance(executor, requests)
        ids = []
        timings = []
        events = None
        if operators:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                for _ in range(steps):
                    ids.append(advance(executor, requests))
            events = [{'operator': e.key, 'shapes': e.input_shapes, 'calls': e.count,
                       'self_cpu_ms': e.self_cpu_time_total / 1000,
                       'inclusive_cpu_ms': e.cpu_time_total / 1000}
                      for e in sorted(prof.key_averages(group_by_input_shape=True),
                                      key=lambda e: e.self_cpu_time_total, reverse=True)[:30]]
        else:
            for _ in range(steps):
                started = perf_counter()
                ids.append(advance(executor, requests))
                timings.append((perf_counter() - started) * 1000)
        return {'threads': threads, 'attention': attention, 'tensor_batching': batched,
                'steps': steps, 'mean_pair_ms': statistics.mean(timings) if timings else None,
                'generated_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                'operators': events}
    finally:
        for r in requests:
            executor.release_request(r.request_id)
        assert executor.cache_bytes == executor.active_sequences == 0


async def main_async(args):
    if args.output.exists():
        raise ValueError('output must be new')
    args.output.mkdir(parents=True)
    for name in args.models:
        executor = await LocalModelExecutor.load(str(MODELS[name]))
        try:
            results = {'model': name, 'created_at': datetime.now(timezone.utc).isoformat(),
                       'torch': executor.torch.__version__, 'scope': 'fixed-pair worker-thread CPU diagnostic; not HTTP throughput',
                       'operator_profiles': [], 'screening': []}
            for batched in (False, True):
                results['operator_profiles'].append(await executor.worker.run(sample, executor, 2, 'eager', batched, 4, True))
            expected = None
            settings = [(t, a, b) for a in ('eager', 'sdpa') for t in (2, 1, 4) for b in (False, True)]
            for rep in range(3):
                for threads, attention, batched in (settings if rep % 2 == 0 else list(reversed(settings))):
                    row = await executor.worker.run(sample, executor, threads, attention, batched, 16)
                    expected = expected or row['generated_ids_sha256']
                    row['matches_baseline_ids'] = row['generated_ids_sha256'] == expected
                    row['repetition'] = rep + 1
                    results['screening'].append(row)
                print(f'{name}: screening repetition {rep + 1} complete', flush=True)
            summary = []
            for threads, attention, batched in settings:
                rows = [r for r in results['screening'] if (r['threads'], r['attention'], r['tensor_batching']) == (threads, attention, batched)]
                summary.append({'threads': threads, 'attention': attention, 'tensor_batching': batched,
                    'median_pair_ms': statistics.median(r['mean_pair_ms'] for r in rows),
                    'all_ids_match': all(r['matches_baseline_ids'] for r in rows)})
            results['summary'] = summary
            (args.output / f'{name}.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps({'model': name, 'summary': summary}, indent=2), flush=True)
        finally:
            await executor.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == '__main__':
    main()
