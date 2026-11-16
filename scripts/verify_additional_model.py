"""Targeted real HTTP checks for an explicitly downloaded pinned model.

Run from the repository root with python -m scripts.verify_additional_model.
No downloads/installations. Starts one local server on port 8020 and stops it.
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
from time import perf_counter

import httpx

from llm_runtime.download_model import MODELS
from llm_runtime.http_benchmark import benchmark
from scripts.compare_engines import (BASE, ROOT, digest, disconnect_active, disconnect_queued,
                                    rss_bytes, snapshot, write_json)


@asynccontextmanager
async def model_server(path, output):
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 8020)) == 0:
            raise RuntimeError('verification port 8020 already has a listener')
    command = [str(ROOT / '.venv/bin/python'), '-m', 'uvicorn',
               'llm_runtime.local:create_local_app', '--factory', '--host', '127.0.0.1',
               '--port', '8020', '--no-access-log']
    env = {**os.environ, 'LLM_MODEL_PATH': str(path), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    process = None
    with (output / 'server.log').open('w') as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            async with httpx.AsyncClient(timeout=2, trust_env=False, follow_redirects=False) as client:
                started = perf_counter()
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('model startup failed; inspect server.log')
                    try:
                        response = await client.get(BASE + '/health')
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if perf_counter() - started > 90:
                        raise TimeoutError('model startup deadline exceeded')
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


async def verify(args):
    model_id, revision, default_path = MODELS[args.model]
    path = (args.model_path or ROOT / default_path).resolve()
    expected = {'model_id': model_id, 'revision': revision}
    if json.loads((path / 'runtime_model.json').read_text()) != expected:
        raise ValueError('checkpoint identity does not match selected pinned model')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {'checked_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Larger-model CPU HTTP compatibility smoke, not an engine speed or answer-quality comparison',
        'project_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'model': expected, 'platform': platform.platform(), 'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'httpx', 'uvicorn')},
        'model_files_sha256': {p.name: digest(p) for p in sorted(path.iterdir()) if p.is_file()},
        'source_sha256': {name: digest(ROOT / name) for name in (
            'scripts/verify_additional_model.py', 'scripts/compare_engines.py',
            'scripts/verify_local_model.py', 'scripts/verify_chat_api.py',
            'src/llm_runtime/download_model.py', 'src/llm_runtime/local_executor.py',
            'src/llm_runtime/api.py', 'src/llm_runtime/chat.py', 'src/llm_runtime/batching.py',
            'src/llm_runtime/http_benchmark.py')}, 'http_runs': []}
    async with model_server(path, output) as (process, command):
        record['server_command'] = command
        record['model_path_environment'] = str(path)
        async with httpx.AsyncClient(timeout=60, trust_env=False, follow_redirects=False,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8)) as client:
            health = (await client.get(BASE + '/health')).json()
            if health['executor']['backend'] != 'transformers' or health['executor']['dtype'] != 'float32':
                raise ValueError('unexpected backend')
            record['initial_health'] = health
            record['rss_at_readiness_bytes'] = await rss_bytes(process.pid)
            for concurrency in (1, 2):
                report = await benchmark(BASE, requests=4, concurrency=concurrency,
                                         warmup=2, max_tokens=32, timeout=60, client=client)
                if report['preflight']['status'] != 'passed' or report['completed'] != 4:
                    raise ValueError(f'HTTP smoke failed: {report}')
                record['http_runs'].append(report)
            body = {'model': 'local-model', 'messages': [{'role': 'user', 'content':
                'What does blåbær mean in English? Answer briefly.'}], 'max_tokens': 32, 'temperature': 0}
            response = await client.post(BASE + '/v1/chat/completions', json=body)
            response.raise_for_status()
            record['unicode_example'] = response.json()
            response = await client.post(BASE + '/v1/chat/completions', json={
                **body, 'messages': [{'role': 'user', 'content': 'What is the capital of France?'}], 'max_tokens': 1})
            response.raise_for_status()
            capped = response.json()
            if capped['usage']['completion_tokens'] != 1 or capped['choices'][0]['finish_reason'] != 'length':
                raise ValueError('one-token output cap failed')
            record['one_token_limit'] = capped
            response = await client.post(BASE + '/v1/chat/completions', json={
                **body, 'messages': [{'role': 'user', 'content': '\U00010348' * 4096}]})
            if response.status_code != 422:
                raise ValueError('expected oversized token-context rejection')
            record['oversized_input'] = {'http_status': response.status_code, 'body': response.json(),
                                         'content': 'U+10348 repeated 4096 times'}
            cases = json.loads((ROOT / 'benchmarks/cpu-workloads-v1.json').read_text())['cases']
            case = next(c for c in cases if c['id'] == 'decode')
            record['queued_disconnect'] = await disconnect_queued(client, 'runtime', case)
            if (not record['queued_disconnect']['queue_removal_observed']
                    or record['queued_disconnect']['blockers_still_active_at_removal'] != 2):
                raise ValueError('queued disconnect did not preserve active blockers')
            record['active_disconnect'] = await disconnect_active(client, 'runtime', case)
            record['explicit_cancel'] = await disconnect_active(client, 'runtime', case, explicit=True)
            record['final_state'] = await snapshot(client, 'runtime')
            if any(record['final_state'].values()):
                raise ValueError('request/KV state remains after verification')
            record['rss_after_checks_bytes'] = await rss_bytes(process.pid)
    record['server_stopped'] = True
    write_json(output / 'http-smoke.json', record)
    print('Verified real HTTP at concurrency 1/2, Unicode, output/context limits, disconnects, cancellation, and zero final KV.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, default='qwen2.5-0.5b')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    asyncio.run(verify(parser.parse_args()))
