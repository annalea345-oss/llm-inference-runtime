import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from llm_runtime import comparison


def document():
    return json.loads(Path('benchmarks/cpu-workloads-v1.json').read_text())


def row(index, status='completed'):
    return {'index': index, 'status': status, 'usage': {'prompt_tokens': 40, 'completion_tokens': 8},
            'timings_ms': {'headers': 1, 'first_content': 2, 'completion': 3, 'content_delta_gaps': []}}


@pytest.mark.parametrize('change', ['duplicate_case', 'bad_reference', 'incomplete_pattern',
                                    'duplicate_workload', 'bad_prompt', 'bad_concurrency', 'bad_cap'])
def test_invalid_workloads_rejected(change):
    doc = document()
    if change == 'duplicate_case':
        doc['cases'].append(deepcopy(doc['cases'][0]))
    elif change == 'bad_reference':
        doc['workloads'][0]['pattern'] = ['absent']
    elif change == 'incomplete_pattern':
        doc['workloads'][-1]['requests'] = 9
    elif change == 'duplicate_workload':
        doc['workloads'].append(deepcopy(doc['workloads'][0]))
    elif change == 'bad_prompt':
        doc['cases'][0]['prompt'] = ''
    elif change == 'bad_concurrency':
        doc['workloads'][0]['concurrencies'] = [True]
    elif change == 'bad_cap':
        doc['cases'][0]['max_tokens'] = 513
    with pytest.raises(ValueError):
        comparison.validate_workloads(doc)


async def test_mixed_workers_preserve_cases_and_bound_concurrency(monkeypatch):
    cases = comparison.validate_workloads(document())
    gate, full = asyncio.Event(), asyncio.Event()
    active, peak, observed = 0, 0, {}

    async def sample(client, url, model, **options):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        observed[options['index']] = (options['prompt'], options['max_tokens'])
        if active == 2:
            full.set()
        await gate.wait()
        active -= 1
        return row(options['index'])

    monkeypatch.setattr(comparison, 'sample', sample)
    task = asyncio.create_task(comparison.pattern_workload(
        None, 'http://localhost', 'model', cases=cases, pattern=['short', 'decode'],
        requests=6, concurrency=2))
    try:
        await asyncio.wait_for(full.wait(), 1)
        assert active == peak == 2
    finally:
        gate.set()
    rows, elapsed = await asyncio.wait_for(task, 1)
    assert elapsed > 0 and peak == 2
    assert [r['index'] for r in rows] == list(range(6))
    assert [r['case_id'] for r in rows] == ['short', 'decode'] * 3
    for i in range(6):
        case = cases[['short', 'decode'][i % 2]]
        assert observed[i] == (case['prompt'], case['max_tokens'])
    assert all(r['client_submitted_ms'] >= 0 for r in rows)


async def test_prompt_usage_mismatch_is_not_successful(monkeypatch):
    async def sample(*args, **kwargs):
        return row(kwargs['index'])
    monkeypatch.setattr(comparison, 'sample', sample)
    cases = comparison.validate_workloads(document())
    rows, elapsed = await comparison.pattern_workload(None, 'http://localhost', 'model',
        cases=cases, pattern=['short'], requests=1, concurrency=1,
        expected_prompt_tokens={'short': 41})
    assert rows[0]['status'] == 'error'
    assert comparison.summarize(rows, elapsed)['completed_tokens_per_second'] is None


async def test_preflight_and_warmup_excluded_from_measured_elapsed(monkeypatch):
    cases = comparison.validate_workloads(document())
    calls = []
    clock_value = 0

    def clock():
        return clock_value

    async def sample(*args, **options):
        nonlocal clock_value
        calls.append(options)
        clock_value += 1
        return row(options['index'])
    monkeypatch.setattr(comparison, 'sample', sample)
    report = await comparison.run_workload(None, 'http://localhost', 'model', cases=cases,
        spec={'id': 'mixed', 'pattern': ['short', 'decode'], 'requests': 4}, concurrency=2,
        expected_prompt_tokens={'short': 40, 'decode': 40}, clock=clock)
    assert len(calls) == 7  # One preflight, two warmups, four measured.
    assert report['elapsed_seconds'] == 4
    assert report['completed_tokens_per_second'] == 8
    assert report['by_class']['short']['completed'] == 2
    assert report['by_class']['short']['completed_tokens_per_second'] == 4
    assert len(report['samples']) == 4


@pytest.mark.parametrize('failing_call', [1, 2])
async def test_failed_preflight_or_warmup_prevents_measured_work(monkeypatch, failing_call):
    calls = []

    async def sample(*args, **options):
        calls.append(options)
        return row(options['index'], 'error' if len(calls) == failing_call else 'completed')
    monkeypatch.setattr(comparison, 'sample', sample)
    cases = comparison.validate_workloads(document())
    with pytest.raises(ValueError, match='preflight|warmup'):
        await comparison.run_workload(None, 'http://localhost', 'model', cases=cases,
            spec={'id': 'short', 'pattern': ['short'], 'requests': 8}, concurrency=1,
            expected_prompt_tokens={'short': 40})
    assert len(calls) == failing_call
