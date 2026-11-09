import socket
import subprocess

import httpx
import pytest

from scripts import compare_engines as runner


class Probe:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def connect_ex(self, address):
        return 1


class Process:
    def __init__(self, stalls=False):
        self.stalls = stalls
        self.signals = []
        self.killed = False
        self.waited = False

    def poll(self):
        return None

    def send_signal(self, signal):
        self.signals.append(signal)

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if timeout is not None and self.stalls:
            raise subprocess.TimeoutExpired('fake-server', timeout)
        self.waited = True
        return 0


@pytest.mark.parametrize('stalls', [False, True])
async def test_managed_server_reaps_process_when_experiment_fails(monkeypatch, tmp_path, stalls):
    process = Process(stalls)
    original_client = httpx.AsyncClient
    # Restore socket before pytest-asyncio creates its teardown event loop.
    with monkeypatch.context() as patches:
        patches.setattr(socket, 'socket', lambda: Probe())
        patches.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: process)
        patches.setattr(runner, 'commands', lambda: {'runtime': ['fake-server']})
        patches.setattr(runner.httpx, 'AsyncClient', lambda **kwargs: original_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))))
        with pytest.raises(ValueError, match='experiment failed'):
            async with runner.server('runtime', tmp_path, 'test'):
                raise ValueError('experiment failed')
    assert process.signals == [runner.signal.SIGINT]
    assert process.killed == stalls
    assert process.waited


async def test_timestamp_policy_is_explicit_for_external_engine():
    assert runner.created_policy('runtime') == 'stable'
    assert runner.created_policy('llama_cpp') == 'per_chunk'


async def test_absent_engine_queue_metrics_are_not_fabricated():
    def handler(request):
        return httpx.Response(200, text='llamacpp:requests_processing 0\n')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(KeyError, match='requests_deferred'):
            await runner.snapshot(client, 'llama_cpp')
