"""Repeat real HTTP workloads in normal and private instrumented CPU servers.

macOS experiment only; no production factory changes, downloads, or dependencies.
Run from the repository root: python -m scripts.profile_runtime_memory --help.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
from time import perf_counter

import httpx

from llm_runtime.comparison import pattern_workload, validate_workloads
from scripts.compare_engines import ROOT, digest, rss_bytes, write_json


BASE = 'http://127.0.0.1:8021'


@asynccontextmanager
async def managed_server(output, label, diagnostic):
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 8021)) == 0:
            raise RuntimeError('memory experiment port 8021 already has a listener')
    command = [str(ROOT / '.venv/bin/python'), '-m', 'uvicorn',
               'scripts.profile_memory_app:create_profile_app' if diagnostic else 'llm_runtime.local:create_local_app',
               '--factory', '--host', '127.0.0.1', '--port', '8021', '--no-access-log']
    env = {**os.environ, 'LLM_MODEL_PATH': str(ROOT / '.models/smollm2'),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'LLM_PROFILE_SHUTDOWN_RECORD': str(output / f'{label}-shutdown.json')}
    process = None
    with (output / f'{label}.log').open('w') as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            async with httpx.AsyncClient(timeout=2, trust_env=False, follow_redirects=False) as client:
                started = perf_counter()
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'profile child failed; inspect {label}.log')
                    try:
                        response = await client.get(BASE + '/health')
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if perf_counter() - started > 60:
                        raise TimeoutError('memory profile startup timeout')
                    await asyncio.sleep(0.1)
            yield process, command
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    await asyncio.to_thread(process.wait, 15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait)


async def json_request(client, method, path):
    response = await client.request(method, BASE + path)
    response.raise_for_status()
    return response.json()


async def observe(client, pid, diagnostic):
    health = await json_request(client, 'GET', '/health')
    if health['active_requests'] or health['queued_requests'] or health['request_states']:
        raise ValueError('memory snapshot requires completed/idle workload')
    state = {'active': health['active_requests'], 'queued': health['queued_requests'],
             'registered': len(health['request_states']), 'reserved_tokens': health['kv_cache']['reserved_tokens'],
             'used_tokens': health['kv_cache']['used_tokens'],
             'kv_tensor_bytes': health['executor']['kv_tensor_bytes_after_last_step']}
    if any(state.values()):
        raise ValueError(f'request ownership not fully released: {state}')
    data = {'state': state, 'host_load_average': os.getloadavg()}
    if diagnostic:
        data['diagnostic'] = await json_request(client, 'GET', '/__memory/snapshot')
        probe = data['diagnostic']
        if (probe['active_sequences'] or probe['requests_registered']
                or probe['weak_objects']['released_sequences']['alive']
                or probe['weak_objects']['released_kv_tensors']['alive']):
            raise ValueError('released sequence/cache objects remain alive')
    data['rss_bytes'] = await rss_bytes(pid)
    return data


async def wave(client, cases, counts):
    rows, elapsed = await pattern_workload(client, BASE, 'local-model', cases=cases,
        pattern=['prefill', 'short', 'decode', 'short'], requests=4, concurrency=2,
        expected_prompt_tokens=counts, timeout=120)
    if any(r['status'] != 'completed' for r in rows):
        raise ValueError(f'memory workload failed: {[(r["case_id"], r["message"]) for r in rows]}')
    return {'elapsed_seconds': elapsed, 'requests': [
        {key: row[key] for key in ('case_id', 'status', 'usage', 'requested_max_tokens')} for row in rows]}


async def run_session(output, label, diagnostic, cases, counts, waves):
    result = {'label': label, 'diagnostic': diagnostic, 'waves': [], 'idle_observations': []}
    async with managed_server(output, label, diagnostic) as (process, command):
        result['command'] = command
        async with httpx.AsyncClient(timeout=120, trust_env=False, follow_redirects=False,
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=2)) as client:
            result['at_readiness'] = await observe(client, process.pid, diagnostic)
            if diagnostic:
                await json_request(client, 'POST', '/__memory/start_trace')
            for index in range(waves):
                requests = await wave(client, cases, counts)
                observation = await observe(client, process.pid, diagnostic)
                result['waves'].append({'wave': index + 1, **requests, **observation})
                if diagnostic and index == 0:
                    await json_request(client, 'POST', '/__memory/anchor')
                write_json(output / f'{label}.json', result)
                print(f'{label}: wave {index + 1}/{waves}; RSS {observation["rss_bytes"] / 1024**2:.1f} MiB', flush=True)
            for _ in range(5):
                await asyncio.sleep(1)
                result['idle_observations'].append(await observe(client, process.pid, diagnostic))
            if diagnostic:
                result['before_gc'] = await observe(client, process.pid, diagnostic)
                result['trace_diff'] = await json_request(client, 'POST', '/__memory/diff')
                result['gc_action'] = await json_request(client, 'POST', '/__memory/gc')
                result['after_gc'] = await observe(client, process.pid, diagnostic)
                result['relief_action'] = await json_request(client, 'POST', '/__memory/relieve')
                await asyncio.sleep(1)
                result['after_relief'] = await observe(client, process.pid, diagnostic)
                result['recovery_wave'] = {**await wave(client, cases, counts),
                                           **await observe(client, process.pid, diagnostic)}
            result['final'] = await observe(client, process.pid, diagnostic)
    result['server_stopped'] = True
    if diagnostic:
        result['after_model_shutdown'] = json.loads((output / f'{label}-shutdown.json').read_text())
    write_json(output / f'{label}.json', result)
    return result


async def run(args):
    if sys.platform != 'darwin':
        raise RuntimeError('this profiling protocol uses macOS RSS/malloc APIs')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((ROOT / 'benchmarks/cpu-comparison-v1/manifest.json').read_text())
    for name, expected in previous['model_files_sha256'].items():
        if digest(ROOT / '.models/smollm2' / name) != expected:
            raise ValueError('checkpoint changed since baseline')
    document = json.loads((ROOT / 'benchmarks/cpu-workloads-v1.json').read_text())
    cases = validate_workloads(document)
    counts = {name: len(value['token_ids']) for name, value in previous['prompts'].items()}
    manifest = {'schema_version': 1, 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'project_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'platform': platform.platform(), 'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'uvicorn', 'httpx')},
        'source_sha256': {name: digest(ROOT / name) for name in (
            'scripts/profile_runtime_memory.py', 'scripts/profile_memory_app.py', 'scripts/memory_diagnostics.py',
            'src/llm_runtime/local_executor.py', 'src/llm_runtime/batching.py', 'src/llm_runtime/api.py',
            'src/llm_runtime/http_benchmark.py', 'src/llm_runtime/comparison.py')},
        'model_files_sha256': previous['model_files_sha256'],
        'workload_sha256': digest(ROOT / 'benchmarks/cpu-workloads-v1.json'),
        'waves_per_session': args.waves, 'requests_per_wave': 4, 'concurrency': 2,
        'request_deadline_seconds': 120,
        'normal_repetitions': args.normal_repetitions, 'sessions': [],
        'native_api_reference': 'https://github.com/apple-oss-distributions/libmalloc/blob/main/include/malloc/malloc.h',
        'notes': ['Normal sessions have no weak observers, tracemalloc, forced GC, or allocator interventions.',
                  'Diagnostic session is instrumented and cannot be used as a performance comparison.',
                  'malloc_zone_statistics(NULL) sums zones; pressure_relief(NULL,0) is best effort for this child only.',
                  'Python tracing excludes most native tensor allocations; native malloc counters are not RSS.',
                  'Weak tensor wrappers dying does not independently account for all native storage.',
                  'Snapshots are at idle between bounded waves; no claim about transient peak memory.']}
    write_json(output / 'manifest.json', manifest)
    try:
        for i in range(args.normal_repetitions):
            label = f'normal-{i + 1}'
            await run_session(output, label, False, cases, counts, args.waves)
            manifest['sessions'].append(label)
        await run_session(output, 'diagnostic', True, cases, counts, args.waves)
        manifest['sessions'].append('diagnostic')
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'manifest.json', manifest)
        raise
    manifest.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat())
    write_json(output / 'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--waves', type=int, default=30)
    parser.add_argument('--normal-repetitions', type=int, default=2)
    args = parser.parse_args()
    if not 2 <= args.waves <= 100 or not 0 <= args.normal_repetitions <= 3:
        parser.error('waves must be 2..100 and normal-repetitions 0..3')
    asyncio.run(run(args))
