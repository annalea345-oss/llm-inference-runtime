"""Small CPU-only diagnostic separating model.forward wall time from other work.

Run after timing comparisons, not concurrently with them. No downloads or server.
This is a diagnostic of one fixed pair of requests, not an end-user benchmark.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter

from llm_runtime.local_executor import LocalModelExecutor
from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.protocol import ExecutionStep, InferenceRequest
from scripts.verify_tensor_batching import MODELS, COUNT


async def profile(path, enabled, steps):
    executor = await LocalModelExecutor.load(str(path), tensor_batching=enabled)
    try:
        requests = [InferenceRequest(COUNT, steps + 6) for _ in range(2)]
        for r in requests:
            r.prompt_token_ids = executor.tokenize(COUNT)
        n = requests[0].prompt_token_count
        # Use the real budget-32 planner to establish the same cache alignment.
        planner = ContinuousBatchScheduler(executor, RuntimeConfig(max_active_requests=2))
        for r in requests:
            r.phase = 'prefill'
            planner._active.append(r)
        while any(r.phase == 'prefill' for r in requests):
            planned = planner._plan()
            results = await executor.execute_batch(tuple(planned))
            assert not any(isinstance(r, BaseException) for r in results)
            for step in planned:
                r = step.request
                if step.phase == 'prefill':
                    r.prefilled_tokens += step.token_count
                    if r.prefilled_tokens == r.prompt_token_count:
                        r.phase = 'decode'
                else:
                    r.generated_tokens += 1
        initial_offsets = [r.generated_tokens for r in requests]
        original = executor.model.forward
        calls = []
        def timed(**kwargs):
            started = perf_counter()
            result = original(**kwargs)
            calls.append({'seconds': perf_counter() - started,
                          'rows': kwargs['input_ids'].shape[0]})
            return result
        executor.model.forward = timed
        measured = []
        ids = [[], []]
        for i in range(steps + 4):
            calls.clear()
            started = perf_counter()
            results = await executor.execute_batch(tuple(ExecutionStep(r, 'decode', r.generated_tokens, 1)
                                                        for r in requests))
            elapsed = perf_counter() - started
            assert not any(isinstance(r, BaseException) for r in results)
            for row, result in zip(ids, results):
                row.append(result.token_id)
            for r in requests:
                r.generated_tokens += 1
            if i >= 4:
                model_seconds = sum(c['seconds'] for c in calls)
                measured.append({'executor_wall_ms': elapsed * 1000,
                    'model_forward_wall_ms': model_seconds * 1000,
                    'other_executor_wall_ms': (elapsed - model_seconds) * 1000,
                    'model_call_rows': [c['rows'] for c in calls]})
        for request in requests:
            executor.release_request(request.request_id)
        assert executor.active_sequences == executor.cache_bytes == 0
        return {'tensor_batching': enabled, 'steps': steps, 'warmup_decode_rounds': 4,
            'prompt_tokens': n, 'initial_decode_offsets': initial_offsets,
            'model_forward_includes': 'entire CPU model.forward, including attention and its KV updates',
            'other_includes': 'token decoding, cache pack/split, input assembly, counters, worker/async overhead',
            'mean_ms': {key: sum(r[key] for r in measured) / steps for key in
                ('executor_wall_ms', 'model_forward_wall_ms', 'other_executor_wall_ms')},
            'generated_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'samples': measured, 'final_cache_bytes': executor.cache_bytes}
    finally:
        # Remove the timing closure before unloading: it captures a bound method
        # and otherwise forms a model -> closure -> model reference cycle.
        if executor.model is not None:
            executor.model.__dict__.pop('forward', None)
        await executor.aclose()


async def main_async(args):
    if args.steps < 1 or args.steps > 128:
        raise ValueError('steps must be in 1..128')
    if args.output.exists():
        raise ValueError('output must be a new file')
    results = {}
    for name in args.models:
        rows = [await profile(MODELS[name], enabled, args.steps) for enabled in (False, True)]
        assert rows[0]['generated_ids_sha256'] == rows[1]['generated_ids_sha256']
        results[name] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'scope': 'fixed pair, budget-32 prefill then steady decode; CPU diagnostic only',
        'results': results}, indent=2) + '\n')
    print(json.dumps({name: [r['mean_ms'] for r in rows] for name, rows in results.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == '__main__':
    main()
