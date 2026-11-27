"""Fresh-process HTTP comparison of ordinary and rowwise CPU projections.

Existing local models only. Uses the same eager references, HTTP workload,
owned-server cleanup, and lifecycle checks as the previous CPU investigation.
"""

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import platform
import statistics
import json

import httpx

from llm_runtime.http_benchmark import benchmark
from scripts.compare_engines import digest, rss_bytes, write_json, disconnect_queued
from scripts.verify_tensor_batching import (BASE, ROOT, MODELS, COUNT, reference,
    server, health, clean, correctness, paired_cancellation)

VARIANTS = {'single': (False, False), 'batch': (True, False), 'rowwise': (True, True)}


async def main_async(args):
    if args.repetitions < 1:
        raise ValueError('repetitions must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    settings = [(a,v) for a in ('eager','sdpa') for v in VARIANTS]
    manifest = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': version('torch'), 'transformers': version('transformers'),
        'source_sha256': {name: digest(ROOT / name) for name in [
            'src/llm_runtime/local_executor.py', 'src/llm_runtime/local.py',
            'src/llm_runtime/cpu_linear.py', 'src/llm_runtime/batching.py',
            'src/llm_runtime/http_benchmark.py', 'scripts/verify_tensor_batching.py',
            'scripts/verify_rowwise_decode.py']},
        'protocol': {'threads': 2, 'dtype': 'float32', 'device': 'cpu',
            'requests_per_run': 4, 'prompt': COUNT, 'max_tokens': 96,
            'repetitions': args.repetitions,
            'order': 'fresh process per model/repetition/attention/variant; reverse setting order on alternate repetitions',
            'timing_cases': 'unbatched and rowwise c1/c2; ordinary tensor batching c2',
            'excluded': 'loading, references, correctness/lifecycle, one preflight and one warmup per timing case',
            'rss_scope': 'readiness and post-run samples, not peaks'}}
    write_json(args.output / 'manifest.json', manifest)
    reports = []
    for model_name in args.models:
        refs = await reference(MODELS[model_name])
        write_json(args.output / f'{model_name}-reference.json', refs)
        for rep in range(args.repetitions):
            for attention, variant in (settings if rep % 2 == 0 else settings[::-1]):
                batched, rowwise = VARIANTS[variant]
                label = f'{model_name}-r{rep+1}-{attention}-{variant}'
                async with server(MODELS[model_name], batched, args.output / f'{label}.log',
                    attention_implementation=attention, rowwise_decode=rowwise) as process:
                    async with httpx.AsyncClient(timeout=120, trust_env=False,
                        limits=httpx.Limits(max_connections=8,max_keepalive_connections=8)) as client:
                        ready = await health(client)
                        ready_rss = await rss_bytes(process.pid)
                        if rep == 0:
                            proof = await correctness(client, refs['cases'])
                            proof['executor'] = (await health(client))['executor']
                            write_json(args.output / f'{label}-correctness.json', proof)
                        for concurrency in ((2,) if variant == 'batch' else (1,2)):
                            before = (await health(client))['executor']
                            report = await benchmark(BASE, requests=4, concurrency=concurrency,
                                warmup=1, max_tokens=96, prompt=COUNT, timeout=120, client=client)
                            assert report['completed'] == report['usage_verified_requests'] == 4
                            assert all(r['usage']['completion_tokens'] == len(refs['cases'][1]['token_ids'])
                                       for r in report['samples'])
                            after = (await health(client))['executor']
                            if batched and concurrency == 2:
                                assert after['batched_forward_calls'] > before['batched_forward_calls']
                            if rowwise and concurrency == 2:
                                assert after['rowwise_decode_linear_calls'] > before['rowwise_decode_linear_calls']
                            if not rowwise or concurrency == 1:
                                assert after['rowwise_decode_linear_calls'] == before['rowwise_decode_linear_calls']
                            report.update(model_name=model_name, repetition=rep+1,
                                attention_implementation=attention, variant=variant,
                                executor_before=before, executor_after=after,
                                final=await clean(client), readiness=ready,
                                readiness_rss_bytes=ready_rss, post_run_rss_bytes=await rss_bytes(process.pid),
                                host_load=os.getloadavg())
                            write_json(args.output / f'{label}-c{concurrency}.json',report)
                            reports.append(report)
                            print(f'{label} c{concurrency}: {report["completed_tokens_per_second"]:.2f} tokens/s',flush=True)
                        if rep == 0 and rowwise:
                            lifecycle = {'paired_cancel': await paired_cancellation(client,True),
                                'paired_disconnect': await paired_cancellation(client,False),
                                'queued_disconnect': await disconnect_queued(client,'runtime',{'prompt':COUNT}),
                                'final': await clean(client)}
                            assert lifecycle['queued_disconnect']['queue_removal_observed']
                            assert lifecycle['queued_disconnect']['blockers_still_active_at_removal'] == 2
                            write_json(args.output / f'{label}-lifecycle.json',lifecycle)
    summary = []
    for name in args.models:
        for attention in ('eager','sdpa'):
            for variant in VARIANTS:
                for concurrency in ((2,) if variant=='batch' else (1,2)):
                    rows = [r for r in reports if r['model_name']==name
                        and r['attention_implementation']==attention and r['variant']==variant
                        and r['workload']['concurrency']==concurrency]
                    summary.append({'model':name,'attention':attention,'variant':variant,
                        'concurrency':concurrency,'runs':len(rows),
                        'median_tokens_per_second':statistics.median(r['completed_tokens_per_second'] for r in rows),
                        'median_first_content_p50_ms':statistics.median(r['latency_ms']['first_content']['p50'] for r in rows)})
    manifest.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),
        measured_requests=sum(r['completed'] for r in reports),measured_runs=len(reports),summary=summary)
    write_json(args.output / 'manifest.json',manifest)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__=='__main__':
    main()
