import json
from pathlib import Path

import pytest

from scripts.summarize_memory_profile import summarize


DATA = Path('benchmarks/retained-memory-v1')


def test_archived_profile_summary_reproduces_all_requests_and_lifetimes():
    result = summarize(DATA)
    assert result == json.loads((DATA / 'summary.json').read_text())
    assert result['completed_workload_requests'] == 364
    diagnostic = result['sessions']['diagnostic']
    assert diagnostic['released_sequence_objects_observed'] == 120
    assert diagnostic['released_kv_tensor_wrappers_observed'] == 7200
    assert diagnostic['released_cache_objects_alive_at_idle'] == 0
    assert not diagnostic['model_alive_after_shutdown']
    assert diagnostic['model_parameter_objects_alive_after_shutdown'] == 0


def test_unfinished_profile_cannot_produce_summary(tmp_path):
    manifest = json.loads((DATA / 'manifest.json').read_text())
    manifest['status'] = 'running'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incomplete'):
        summarize(tmp_path)


@pytest.mark.parametrize('problem', ['waves', 'pattern', 'request_failed', 'ownership', 'child_alive'])
def test_invalid_wave_or_cleanup_evidence_is_rejected(tmp_path, problem):
    (tmp_path / 'manifest.json').write_text((DATA / 'manifest.json').read_text())
    data = json.loads((DATA / 'normal-1.json').read_text())
    if problem == 'waves':
        data['waves'].pop()
    elif problem == 'pattern':
        data['waves'][0]['requests'][0]['case_id'] = 'other'
    elif problem == 'request_failed':
        data['waves'][0]['requests'][0]['status'] = 'error'
    elif problem == 'ownership':
        data['waves'][0]['state']['registered'] = 1
    elif problem == 'child_alive':
        data['server_stopped'] = False
    (tmp_path / 'normal-1.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summarize(tmp_path)
