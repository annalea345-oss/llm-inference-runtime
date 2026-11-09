"""Summarize a completed compare_engines.py directory without rerunning servers."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from llm_runtime.http_benchmark import distribution


def summarize_groups(reports):
    groups = defaultdict(list)
    for report in reports:
        groups[(report['engine'], report['workload_id'], report['concurrency'])].append(report)
    result = []
    for (engine, workload, concurrency), runs in sorted(groups.items()):
        rows = [r for run in runs for r in run['samples']]
        completed = [r for r in rows if r['status'] == 'completed']
        classes = {}
        for name in sorted({r['request_class'] for r in rows}):
            selected = [r for r in completed if r['request_class'] == name]
            classes[name] = {
                'completed': len(selected),
                'prompt_token_counts': sorted({r['usage']['prompt_tokens'] for r in selected}),
                'output_token_counts': dict(sorted(Counter(str(r['usage']['completion_tokens']) for r in selected).items())),
                'finish_reasons': dict(Counter(r['finish_reason'] for r in selected)),
                'per_run_mean_first_content_ms': distribution([
                    run['by_class'][name]['latency_ms']['first_content']['mean'] for run in runs
                    if name in run['by_class'] and run['by_class'][name]['latency_ms']['first_content']['mean'] is not None]),
                'per_run_p95_first_content_ms': distribution([
                    run['by_class'][name]['latency_ms']['first_content']['p95'] for run in runs
                    if name in run['by_class'] and run['by_class'][name]['latency_ms']['first_content']['p95'] is not None]),
            }
        result.append({
            'engine': engine, 'workload_id': workload, 'concurrency': concurrency,
            'run_count': len(runs), 'measured_requests': len(rows), 'completed': len(completed),
            'failures': dict(Counter(r['status'] for r in rows if r['status'] != 'completed')),
            'per_run_mean_first_content_ms': distribution([r['latency_ms']['first_content']['mean']
                for r in runs if r['latency_ms']['first_content']['mean'] is not None]),
            'per_run_p95_first_content_ms': distribution([r['latency_ms']['first_content']['p95']
                for r in runs if r['latency_ms']['first_content']['p95'] is not None]),
            'per_run_mean_completion_ms': distribution([r['latency_ms']['completion']['mean']
                for r in runs if r['latency_ms']['completion']['mean'] is not None]),
            'per_run_tokens_per_second': distribution([r['completed_tokens_per_second']
                for r in runs if r['completed_tokens_per_second'] is not None]),
            'created_timestamp_changes': sum(r.get('stream_metadata', {}).get('created_changes', 0) for r in rows),
            'by_class': classes,
        })
    return result


def load_reports(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('comparison has not completed; refuse a partial headline summary')
    expected = {(engine, repetition, w['id'], concurrency)
        for engine in ('runtime', 'llama_cpp') for repetition in range(1, manifest['repetitions'] + 1)
        for w in manifest['workloads']['workloads'] for concurrency in w['concurrencies']}
    reports = [json.loads(p.read_text()) for p in sorted(directory.glob('repetition-*-c[0-9]*.json'))]
    observed = [(r['engine'], r['repetition'], r['workload_id'], r['concurrency']) for r in reports]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError('missing, duplicate, or unexpected timing runs')
    specs = {w['id']: w for w in manifest['workloads']['workloads']}
    cases = {c['id']: c for c in manifest['workloads']['cases']}
    for report in reports:
        spec = specs[report['workload_id']]
        if len(report['samples']) != spec['requests']:
            raise ValueError('wrong measured sample count')
        for index, row in enumerate(report['samples']):
            case = cases[spec['pattern'][index % len(spec['pattern'])]]
            if row['index'] != index or row['case_id'] != case['id'] or row['requested_max_tokens'] != case['max_tokens']:
                raise ValueError('sample ordering/case does not match versioned workload')
            if row['status'] == 'completed':
                if (row['usage_status'] != 'verified' or row['usage']['prompt_tokens'] != len(manifest['prompts'][case['id']]['token_ids'])
                        or row['usage']['completion_tokens'] > case['max_tokens']):
                    raise ValueError('unverified or mismatched token usage')
    return manifest, reports


def summarize_directory(directory):
    manifest, reports = load_reports(directory)
    lifecycle = {e: json.loads((directory / f'lifecycle-{e}.json').read_text()) for e in ('runtime', 'llama_cpp')}
    observations = {}
    for engine, data in lifecycle.items():
        rss = [r['rss_bytes'] for r in data['memory_timeline']]
        observations[engine] = {
            'active_disconnect_idle_ms': data['active_disconnect']['observed_idle_after_ms'],
            'queued_disconnect': {k: v for k, v in data['queued_disconnect'].items()
                if k not in ('followup', 'removal_observations', 'idle_observations_after_all_disconnects')},
            'rss_at_readiness_bytes': data['rss_at_readiness_bytes'],
            'observed_rss_peak_bytes': max(rss),
            'settled_wave_rss_bytes': [w['settled_rss_bytes'] for w in data['sustained_waves']],
            'final_rss_bytes': data['final_rss_bytes'], 'final_state': data['final_state'],
        }
        if 'explicit_cancel' in data:
            observations[engine]['explicit_cancel_idle_ms'] = data['explicit_cancel']['observed_idle_after_ms']
    return {'schema_version': 1, 'scope': 'Per-run descriptive statistics; no confidence intervals or scheduler-only attribution',
            'started_at': manifest['started_at'], 'finished_at': manifest['finished_at'],
            'timing_runs': len(reports), 'measured_requests': sum(len(r['samples']) for r in reports),
            'groups': summarize_groups(reports), 'lifecycle': observations}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize_directory(args.directory), indent=2, allow_nan=False) + '\n')
