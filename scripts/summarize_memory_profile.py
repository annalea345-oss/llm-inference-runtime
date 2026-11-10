"""Summarize a completed bounded retained-memory experiment; no server actions."""

import argparse
import json
from pathlib import Path


def span(values):
    return {'first': values[0], 'last': values[-1], 'min': min(values), 'max': max(values),
            'last_minus_first': values[-1] - values[0]}


def summarize(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    expected = [f'normal-{i + 1}' for i in range(manifest['normal_repetitions'])] + ['diagnostic']
    if manifest['status'] != 'completed' or manifest['sessions'] != expected:
        raise ValueError('refuse to summarize an incomplete memory experiment')
    result = {'schema_version': 1, 'scope': 'Bounded idle-wave memory observations; no long-term leak guarantee',
              'sessions': {}, 'completed_workload_requests': 0}
    pattern = ['prefill', 'short', 'decode', 'short']
    for label in expected:
        run = json.loads((directory / f'{label}.json').read_text())
        if not run['server_stopped'] or len(run['waves']) != manifest['waves_per_session']:
            raise ValueError('missing waves or child cleanup')
        waves = run['waves']
        for index, wave in enumerate(waves):
            if wave['wave'] != index + 1 or [r['case_id'] for r in wave['requests']] != pattern:
                raise ValueError('wrong wave index or workload pattern')
            if any(value != 0 for value in wave['state'].values()):
                raise ValueError('wave did not release request ownership')
            if any(r['status'] != 'completed' for r in wave['requests']):
                raise ValueError('failed workload request')
        rss = [w['rss_bytes'] for w in waves]
        summary = {'waves': len(waves), 'completed_requests': len(waves) * 4,
                   'rss_at_readiness_bytes': run['at_readiness']['rss_bytes'],
                   'rss_after_waves_bytes': span(rss),
                   'rss_last_10_waves_bytes': span(rss[-10:]),
                   'rss_after_idle_bytes': run['idle_observations'][-1]['rss_bytes'],
                   'final_rss_bytes': run['final']['rss_bytes'],
                   'all_idle_ownership_counters_zero': True}
        if run['diagnostic']:
            diagnostics = [w['diagnostic'] for w in waves]
            if any(d['weak_objects'][kind]['alive'] for d in diagnostics
                   for kind in ('released_sequences', 'released_kv_tensors')):
                raise ValueError('observed released cache object remained alive')
            summary.update(
                released_sequence_objects_observed=diagnostics[-1]['weak_objects']['released_sequences']['observed'],
                released_kv_tensor_wrappers_observed=diagnostics[-1]['weak_objects']['released_kv_tensors']['observed'],
                released_cache_objects_alive_at_idle=0,
                completed_request_objects_alive_at_idle=span([d['weak_objects']['requests']['alive'] for d in diagnostics]),
                python_traced_current_bytes=span([d['python_traced_current_bytes'] for d in diagnostics]),
                native_malloc_in_use_bytes=span([d['native_malloc']['size_in_use'] for d in diagnostics]),
                gc_probe={'rss_before_bytes': run['before_gc']['rss_bytes'], 'rss_after_bytes': run['after_gc']['rss_bytes'],
                          'objects_collected': run['gc_action']['snapshot']['gc_collected_by_probe']},
                allocator_probe={'reported_released_bytes': run['relief_action']['allocator_reported_released_bytes'],
                                 'rss_before_bytes': run['after_gc']['rss_bytes'], 'rss_after_bytes': run['after_relief']['rss_bytes']},
                python_allocation_diff=run['trace_diff']['python_allocation_diff'],
                model_alive_after_shutdown=run['after_model_shutdown']['model_alive'],
                model_parameter_objects_alive_after_shutdown=run['after_model_shutdown']['model_parameters_alive'],
                request_objects_alive_after_shutdown=run['after_model_shutdown']['weak_objects']['requests']['alive'],
                native_malloc_in_use_after_model_shutdown_bytes=run['after_model_shutdown']['native_malloc']['size_in_use'])
            if (run['after_model_shutdown']['model_alive']
                    or run['after_model_shutdown']['model_parameters_alive']
                    or any(r['status'] != 'completed' for r in run['recovery_wave']['requests'])):
                raise ValueError('model lifetime or post-intervention recovery check failed')
            result['completed_workload_requests'] += 4  # Recovery wave after allocator probe.
        result['sessions'][label] = summary
        result['completed_workload_requests'] += summary['completed_requests']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.directory), indent=2, allow_nan=False) + '\n')
