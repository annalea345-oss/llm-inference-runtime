"""Closed-loop, mixed-prompt HTTP experiments; no server or model lifecycle."""

import asyncio
from time import perf_counter

from llm_runtime.http_benchmark import sample, summarize, validate_options


def validate_workloads(document):
    if not isinstance(document, dict) or document.get('schema_version') != 1:
        raise ValueError('expected workload schema_version 1')
    cases, workloads = document.get('cases'), document.get('workloads')
    if not isinstance(cases, list) or not cases or not isinstance(workloads, list) or not workloads:
        raise ValueError('nonempty cases and workloads are required')
    lookup = {}
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get('id'), str)
                or not case['id'] or case['id'] in lookup
                or not isinstance(case.get('class'), str) or not case['class']):
            raise ValueError('case IDs/classes must be nonempty and IDs unique')
        validate_options('http://127.0.0.1', 1, 1, 0, case.get('max_tokens'), case.get('prompt'), 30)
        lookup[case['id']] = case
    names = set()
    for spec in workloads:
        if (not isinstance(spec, dict) or not isinstance(spec.get('id'), str)
                or not spec['id'] or spec['id'] in names):
            raise ValueError('workload IDs must be nonempty and unique')
        names.add(spec['id'])
        pattern, sizes = spec.get('pattern'), spec.get('concurrencies')
        if (not isinstance(pattern, list) or not pattern
                or any(not isinstance(k, str) or k not in lookup for k in pattern)):
            raise ValueError('workload pattern must reference known cases')
        if (not isinstance(sizes, list) or not sizes or any(type(c) is not int or c < 1 for c in sizes)
                or len(set(sizes)) != len(sizes)):
            raise ValueError('concurrencies must be distinct positive integers')
        if (type(spec.get('requests')) is not int or spec['requests'] < max(sizes)
                or spec['requests'] % len(pattern)):
            raise ValueError('requests must cover concurrency and complete pattern cycles')
    return lookup


async def pattern_workload(client, base_url, model, *, cases, pattern, requests,
                           concurrency, timeout=30, expected_prompt_tokens=None, clock=perf_counter,
                           created_policy='stable'):
    """Workers assign deterministic case indices; arrivals depend on completion."""
    rows = [None] * requests
    indices = iter(range(requests))
    started = clock()

    async def worker():
        for index in indices:
            case = cases[pattern[index % len(pattern)]]
            submitted = (clock() - started) * 1000
            row = await sample(client, base_url, model, index=index, prompt=case['prompt'],
                               max_tokens=case['max_tokens'], timeout=timeout, clock=clock,
                               created_policy=created_policy)
            row.update(case_id=case['id'], request_class=case['class'],
                       requested_max_tokens=case['max_tokens'], client_submitted_ms=submitted)
            if (row['status'] == 'completed' and expected_prompt_tokens is not None
                    and row['usage']['prompt_tokens'] != expected_prompt_tokens[case['id']]):
                row.update(status='error', message='server prompt usage differs from verified tokenizer')
            rows[index] = row

    await asyncio.gather(*(worker() for _ in range(min(requests, concurrency))))
    return rows, clock() - started


def summarize_classes(rows, elapsed):
    # A class denominator is the whole mixed run, not the sum of request latencies.
    return {name: {k: v for k, v in summarize(
                [r for r in rows if r['request_class'] == name], elapsed).items() if k != 'samples'}
            for name in sorted({r['request_class'] for r in rows})}


async def run_workload(client, base_url, model, *, cases, spec, concurrency,
                       expected_prompt_tokens, timeout=30, clock=perf_counter, created_policy='stable'):
    options = dict(cases=cases, pattern=spec['pattern'], concurrency=concurrency,
                   expected_prompt_tokens=expected_prompt_tokens, timeout=timeout, clock=clock,
                   created_policy=created_policy)
    # Each case in the pattern is exercised; warmups also cover all live workers.
    warmup = max(len(spec['pattern']), concurrency)
    preflight, _ = await pattern_workload(client, base_url, model, requests=1, **options)
    if preflight[0]['status'] != 'completed':
        raise ValueError(f'comparison preflight failed: {preflight}')
    warmed, _ = await pattern_workload(client, base_url, model, requests=warmup, **options)
    if any(r['status'] != 'completed' for r in warmed):
        raise ValueError(f'comparison warmup failed: {warmed}')
    rows, elapsed = await pattern_workload(client, base_url, model, requests=spec['requests'], **options)
    return {'schema_version': 1, 'workload_id': spec['id'], 'concurrency': concurrency,
            'arrival_mode': 'closed_loop', 'created_policy': created_policy, 'preflight_samples': preflight,
            'warmup_samples': warmed, 'by_class': summarize_classes(rows, elapsed),
            **summarize(rows, elapsed)}
