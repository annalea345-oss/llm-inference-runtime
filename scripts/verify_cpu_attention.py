"""Fresh-process HTTP comparison: eager versus SDPA at fixed two Torch threads.

Both existing checkpoints, identical greedy settings. No downloads. Single-client
timings cover unbatched execution; two-client timings cover batching off/on.
Correctness and candidate lifecycle checks are excluded from timing.
"""

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics

import httpx

from llm_runtime.http_benchmark import benchmark
from scripts.compare_engines import digest, rss_bytes, write_json, disconnect_queued
from scripts.verify_tensor_batching import (BASE, ROOT, MODELS, COUNT, reference,
    server, health, clean, correctness, paired_cancellation)


async def main_async(args):
    if args.repetitions < 1:
        raise ValueError('repetitions must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    settings = [('eager', False), ('sdpa', False), ('eager', True), ('sdpa', True)]
    manifest = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': version('torch'), 'transformers': version('transformers'),
        'source_sha256': {name: digest(ROOT / name) for name in [
            'src/llm_runtime/local_executor.py', 'src/llm_runtime/local.py',
            'src/llm_runtime/batching.py', 'src/llm_runtime/http_benchmark.py',
            'scripts/verify_tensor_batching.py', 'scripts/verify_cpu_attention.py']},
        'protocol': {'threads': 2, 'dtype': 'float32', 'device': 'cpu', 'requests_per_run': 4,
            'prompt': COUNT, 'max_tokens': 96, 'repetitions': args.repetitions,
            'order': 'fresh process for each model/repetition/attention/batching; reverse settings each odd repetition',
            'timing_cases': 'unbatched concurrency 1 and 2; batched concurrency 2 only',
            'excluded': 'reference/correctness/lifecycle, loading, one preflight and one warmup per timing case',
            'rss_scope': 'readiness/post-run samples; not peaks'}}
    write_json(args.output / 'manifest.json', manifest)
    reports = []
    for model_name in args.models:
        refs = await reference(MODELS[model_name])
        write_json(args.output / f'{model_name}-reference.json', refs)
        for rep in range(args.repetitions):
            ordered = settings if rep % 2 == 0 else list(reversed(settings))
            for attention, batched in ordered:
                label = f'{model_name}-r{rep + 1}-{attention}-{"batch" if batched else "single"}'
                async with server(MODELS[model_name], batched, args.output / f'{label}.log',
                                  attention_implementation=attention) as process:
                    async with httpx.AsyncClient(timeout=120, trust_env=False,
                        limits=httpx.Limits(max_connections=8, max_keepalive_connections=8)) as client:
                        ready = await health(client)
                        assert ready['executor']['attention_implementation'] == attention
                        ready_rss = await rss_bytes(process.pid)
                        if rep == 0:
                            proof = await correctness(client, refs['cases'])
                            proof['executor'] = (await health(client))['executor']
                            write_json(args.output / f'{label}-correctness.json', proof)
                        for concurrency in ((2,) if batched else (1, 2)):
                            before = (await health(client))['executor']
                            report = await benchmark(BASE, requests=4, concurrency=concurrency,
                                warmup=1, max_tokens=96, prompt=COUNT, timeout=120, client=client)
                            assert report['completed'] == report['usage_verified_requests'] == 4
                            assert all(r['usage']['completion_tokens'] == len(refs['cases'][1]['token_ids'])
                                       for r in report['samples'])
                            report.update(model_name=model_name, repetition=rep + 1,
                                attention_implementation=attention, tensor_batching=batched,
                                executor_before=before, executor_after=(await health(client))['executor'],
                                final=await clean(client), readiness_rss_bytes=ready_rss,
                                post_run_rss_bytes=await rss_bytes(process.pid), host_load=os.getloadavg())
                            if batched:
                                assert report['executor_after']['batched_forward_calls'] > before['batched_forward_calls']
                            write_json(args.output / f'{label}-c{concurrency}.json', report)
                            reports.append(report)
                            print(f'{label} c{concurrency}: {report["completed_tokens_per_second"]:.2f} tokens/s', flush=True)
                        if rep == 0 and attention == 'sdpa':
                            lifecycle = {'paired_cancel': await paired_cancellation(client, True),
                                'paired_disconnect': await paired_cancellation(client, False),
                                'queued_disconnect': await disconnect_queued(client, 'runtime', {'prompt': COUNT}),
                                'final': await clean(client)}
                            assert lifecycle['queued_disconnect']['queue_removal_observed']
                            assert lifecycle['queued_disconnect']['blockers_still_active_at_removal'] == 2
                            write_json(args.output / f'{label}-lifecycle.json', lifecycle)
    summary = []
    for model_name in args.models:
        for batched, concurrency in [(False, 1), (False, 2), (True, 2)]:
            row = {'model': model_name, 'tensor_batching': batched, 'concurrency': concurrency}
            for attention in ('eager', 'sdpa'):
                selected = [r for r in reports if r['model_name'] == model_name
                    and r['tensor_batching'] is batched and r['workload']['concurrency'] == concurrency
                    and r['attention_implementation'] == attention]
                row[attention] = {'median_tokens_per_second': statistics.median(r['completed_tokens_per_second'] for r in selected),
                    'median_first_content_p50_ms': statistics.median(r['latency_ms']['first_content']['p50'] for r in selected),
                    'runs': len(selected)}
            row['throughput_ratio'] = row['sdpa']['median_tokens_per_second'] / row['eager']['median_tokens_per_second']
            summary.append(row)
    manifest.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
                    measured_requests=sum(r['completed'] for r in reports), measured_runs=len(reports), summary=summary)
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == '__main__':
    main()
