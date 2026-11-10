"""Experiment observer isolation: no real allocator interventions in unit tests."""

import gc
from types import SimpleNamespace

import pytest

from llm_runtime.api import create_app
from llm_runtime.protocol import InferenceRequest
from scripts.memory_diagnostics import MemoryProbe, WeakAudit, require_idle


def test_weak_audit_does_not_keep_requests_alive_or_accumulate_dead_references():
    audit = WeakAudit()
    for _ in range(1000):
        value = InferenceRequest('test', 1)
        audit.observe('requests', value)
    del value
    gc.collect()
    result = audit.snapshot()
    assert result['requests'] == {'observed': 1000, 'alive': 0}
    assert audit.references['requests'] == []


@pytest.mark.parametrize('active,queued', [(1, 0), (0, 1)])
@pytest.mark.parametrize('operation', ['gc', 'relieve', 'start_trace', 'anchor', 'diff'])
def test_mutating_diagnostics_refuse_live_work_before_any_intervention(active, queued, operation):
    # Construct without installing hooks or loading a model/native library.
    probe = object.__new__(MemoryProbe)
    probe.scheduler = SimpleNamespace(active=[object()] * active, queued_requests=queued)
    with pytest.raises(RuntimeError, match='idle'):
        probe.operate(operation)


def test_idle_guard_accepts_empty_scheduler():
    require_idle(SimpleNamespace(active=(), queued_requests=0))


def test_normal_server_has_no_memory_controls():
    app = create_app()
    assert not any(getattr(route, 'path', '').startswith('/__memory') for route in app.routes)


def test_audit_reports_a_live_reference_then_compacts_it_after_collection():
    audit = WeakAudit()

    class Cycle:
        pass

    value = Cycle()
    value.self = value
    audit.observe('released_sequences', value)
    assert audit.snapshot()['released_sequences']['alive'] == 1
    del value
    gc.collect()
    assert audit.snapshot()['released_sequences'] == {'observed': 1, 'alive': 0}


def test_unknown_diagnostic_operation_is_rejected():
    probe = object.__new__(MemoryProbe)
    probe.scheduler = SimpleNamespace(active=(), queued_requests=0)
    with pytest.raises(ValueError, match='unknown'):
        probe.operate('not-an-operation')
