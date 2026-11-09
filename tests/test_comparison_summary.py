from copy import deepcopy
import json

import pytest

from scripts.summarize_engine_comparison import load_reports, summarize_groups


def report(repetition, latency, throughput):
    row = {'index': 0, 'status': 'completed', 'case_id': 'short', 'request_class': 'short',
           'requested_max_tokens': 24, 'usage_status': 'verified', 'finish_reason': 'stop',
           'usage': {'prompt_tokens': 40, 'completion_tokens': 8}}
    summary = {'latency_ms': {'first_content': {'mean': latency, 'p95': latency},
                              'completion': {'mean': latency * 2}}}
    return {'engine': 'runtime', 'workload_id': 'short', 'concurrency': 1,
            'repetition': repetition, 'samples': [row], 'completed_tokens_per_second': throughput,
            'by_class': {'short': summary}, **summary}


def test_summary_reports_distribution_of_runs_and_actual_work():
    reports = [report(1, 10, 20), report(2, 30, 40), report(3, 100, 50)]
    result = summarize_groups(reports)[0]
    assert result['run_count'] == result['completed'] == 3
    assert result['per_run_mean_first_content_ms']['p50'] == 30
    assert result['per_run_mean_first_content_ms']['mean'] == pytest.approx(140 / 3)
    assert result['per_run_tokens_per_second']['p50'] == 40
    assert result['by_class']['short']['output_token_counts'] == {'8': 3}


def manifest():
    return {'status': 'completed', 'repetitions': 1,
            'workloads': {'cases': [{'id': 'short', 'max_tokens': 24}],
                          'workloads': [{'id': 'short', 'requests': 1, 'pattern': ['short'], 'concurrencies': [1]}]},
            'prompts': {'short': {'token_ids': [1] * 40}}}


@pytest.mark.parametrize('issue', ['incomplete', 'missing', 'duplicate', 'usage', 'case', 'count'])
def test_incomplete_or_inconsistent_artifacts_cannot_produce_headlines(tmp_path, issue):
    data = manifest()
    if issue == 'incomplete':
        data['status'] = 'running'
    (tmp_path / 'manifest.json').write_text(json.dumps(data))
    for engine in ('runtime', 'llama_cpp'):
        row = report(1, 10, 20)
        row['engine'] = engine
        if engine == 'runtime':
            if issue == 'missing':
                continue
            if issue == 'usage':
                row['samples'][0]['usage']['prompt_tokens'] = 41
            elif issue == 'case':
                row['samples'][0]['case_id'] = 'other'
            elif issue == 'count':
                row['samples'].append(deepcopy(row['samples'][0]))
            elif issue == 'duplicate':
                (tmp_path / 'repetition-1-runtime-duplicate-c1.json').write_text(json.dumps(row))
        (tmp_path / f'repetition-1-{engine}-short-c1.json').write_text(json.dumps(row))
    with pytest.raises(ValueError):
        load_reports(tmp_path)
