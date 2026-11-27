"""Offline pair diagnostic: model calls versus surrounding cache/executor work.

Run separately from HTTP measurements. Profiles use instrumentation and are not
throughput results. All calls stay inside the owning serial model worker.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter

from llm_runtime.local_executor import LocalModelExecutor
from scripts.investigate_cpu_inference import prepare, advance
from scripts.verify_tensor_batching import MODELS, ROOT
from scripts.compare_engines import digest


def sample(executor, variant, attention, steps, operators=False):
    executor.tensor_batching = variant != 'single'
    for layer in executor._rowwise_layers:
        layer.enabled = variant == 'rowwise'
    executor.model.set_attn_implementation(attention)
    requests = prepare(executor, steps)
    original = executor.model.forward
    calls = []
    def timed(**kwargs):
        started = perf_counter()
        result = original(**kwargs)
        calls.append((perf_counter() - started) * 1000)
        return result
    executor.model.forward = timed
    ids, timings = [], []
    try:
        for _ in range(4):
            advance(executor, requests)
        if operators:
            torch = executor.torch
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                for _ in range(4):
                    ids.append(advance(executor,requests))
            events = sorted(prof.key_averages(group_by_input_shape=True),
                            key=lambda e:e.self_cpu_time_total,reverse=True)
            return {'variant':variant, 'attention':attention, 'profiled_rounds':4,
                'operators':[{'operator':e.key,'shapes':e.input_shapes,'calls':e.count,
                    'self_cpu_ms':e.self_cpu_time_total/1000} for e in events],
                'generated_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        for _ in range(steps):
            calls.clear()
            started = perf_counter()
            ids.append(advance(executor,requests))
            total=(perf_counter()-started)*1000
            model=sum(calls)
            timings.append({'pair_wall_ms':total, 'model_wall_ms':model,
                'surrounding_wall_ms':total-model, 'model_calls':len(calls)})
        return {'variant':variant,'attention':attention,'steps':steps,
            'mean_ms':{k:statistics.mean(t[k] for t in timings) for k in
                ('pair_wall_ms','model_wall_ms','surrounding_wall_ms')},
            'generated_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'samples':timings}
    finally:
        executor.model.__dict__.pop('forward',None)
        for r in requests: executor.release_request(r.request_id)
        assert executor.cache_bytes == executor.active_sequences == 0


async def main_async(args):
    if not 1 <= args.steps <= 128:
        raise ValueError('steps must be in 1..128')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={'started_at':datetime.now(timezone.utc).isoformat(),
        'scope':'fixed pair, budget-32 prefill, four warmup rounds; CPU worker diagnostic, not HTTP',
        'surrounding_scope':'cache pack/split, token decoding, input assembly and counters; not an isolated cache timing',
        'threads':2,'dtype':'float32','repetitions':3,
        'source_sha256':{p:digest(ROOT/p) for p in [
            'scripts/profile_rowwise_decode.py','scripts/investigate_cpu_inference.py',
            'src/llm_runtime/local_executor.py','src/llm_runtime/cpu_linear.py']}}
    for name in args.models:
        ex=await LocalModelExecutor.load(str(MODELS[name]),tensor_batching=True,rowwise_decode=True)
        try:
            settings=[(v,a) for a in ('eager','sdpa') for v in ('single','batch','rowwise')]
            rows=[]
            for rep in range(3):
                for variant,attention in (settings if rep%2==0 else settings[::-1]):
                    row=await ex.worker.run(sample,ex,variant,attention,args.steps)
                    row['repetition']=rep+1
                    rows.append(row)
                print(f'{name}: diagnostic repetition {rep+1} complete',flush=True)
            expected=rows[0]['generated_ids_sha256']
            assert all(r['generated_ids_sha256']==expected for r in rows)
            profiles=[await ex.worker.run(sample,ex,v,a,args.steps,True) for v,a in settings]
            assert len({p['generated_ids_sha256'] for p in profiles})==1
            summary=[{'variant':v,'attention':a,'median_ms':{k:statistics.median(
                r['mean_ms'][k] for r in rows if r['variant']==v and r['attention']==a)
                for k in ('pair_wall_ms','model_wall_ms','surrounding_wall_ms')}} for v,a in settings]
            result={'model':name,'timings':rows,'operator_profiles':profiles,'summary':summary,
                'all_timing_ids_match':True,'all_profile_ids_match':True,'final_cache_bytes':ex.cache_bytes}
            (args.output/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(summary,indent=2),flush=True)
        finally:
            await ex.aclose()
    manifest['finished_at']=datetime.now(timezone.utc).isoformat()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__=='__main__':
    main()
