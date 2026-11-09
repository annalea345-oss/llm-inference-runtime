"""Run the pinned local CPU comparison; see benchmarks/cpu-comparison.md.

Starts one loopback server at a time, stops it in finally, and checkpoints each
result. Requires pre-existing artifacts from the llama.cpp compatibility phase.
No installation, model download, or inference backend modification.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from time import perf_counter

import httpx

from llm_runtime.comparison import pattern_workload, run_workload, validate_workloads
from llm_runtime.http_benchmark import sample, sse_frames


ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8020'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def commands():
    build = json.loads((ROOT / 'benchmarks/llama-cpp-build.json').read_text())
    llama = build['launch_command'][:]
    llama[llama.index('--port') + 1] = '8020'
    return {'runtime': [str(ROOT / '.venv/bin/python'), '-m', 'uvicorn',
                        'llm_runtime.local:create_local_app', '--factory',
                        '--host', '127.0.0.1', '--port', '8020', '--no-access-log'],
            'llama_cpp': llama}


async def get_json(client, path):
    response = await client.get(BASE + path)
    response.raise_for_status()
    return response.json()


async def snapshot(client, engine):
    if engine == 'runtime':
        data = await get_json(client, '/health')
        return {'active': data['active_requests'], 'queued': data['queued_requests'],
                'reserved_tokens': data['kv_cache']['reserved_tokens'],
                'used_tokens': data['kv_cache']['used_tokens'],
                'kv_tensor_bytes': data['executor']['kv_tensor_bytes_after_last_step']}
    response = await client.get(BASE + '/metrics')
    response.raise_for_status()
    metrics = {}
    for line in response.text.splitlines():
        if line and not line.startswith('#'):
            key, value = line.rsplit(' ', 1)
            metrics[key] = float(value)
    # Required gauges must exist; absent counters are not silently treated as zero.
    return {'active': metrics['llamacpp:requests_processing'],
            'queued': metrics['llamacpp:requests_deferred'],
            'cached_prefix_tokens_total': metrics['llamacpp:prompt_tokens_cached_total']}


async def wait_state(client, engine, predicate, timeout=10):
    started = perf_counter()
    observations = []
    while True:
        state = await snapshot(client, engine)
        observations.append({'elapsed_ms': (perf_counter() - started) * 1000, **state})
        if predicate(state):
            return observations
        if perf_counter() - started > timeout:
            raise TimeoutError(f'{engine} state did not settle: {state}')
        await asyncio.sleep(0.02)


def idle(state):
    return state['active'] == state['queued'] == 0


def created_policy(engine):
    return 'per_chunk' if engine == 'llama_cpp' else 'stable'


async def rss_bytes(pid):
    process = await asyncio.create_subprocess_exec('ps', '-o', 'rss=', '-p', str(pid),
                                                  stdout=asyncio.subprocess.PIPE)
    output, _ = await process.communicate()
    if process.returncode:
        raise RuntimeError('server process disappeared during RSS observation')
    return int(output.strip()) * 1024  # macOS ps reports RSS in KiB.


@asynccontextmanager
async def server(engine, output, label):
    # Never attach an experiment to a pre-existing listener on the chosen port.
    import socket
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 8020)) == 0:
            raise RuntimeError('comparison port 8020 already has a listener')
    env = {**os.environ, 'LLM_MODEL_PATH': str(ROOT / '.models/smollm2'),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    log_path = output / f'{label}-{engine}.log'
    process = None
    with log_path.open('w') as log:
        try:
            process = subprocess.Popen(commands()[engine], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            async with httpx.AsyncClient(timeout=2, trust_env=False, follow_redirects=False) as client:
                started = perf_counter()
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'{engine} startup failed; inspect {log_path}')
                    try:
                        response = await client.get(BASE + '/health')
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if perf_counter() - started > 60:
                        raise TimeoutError('server startup deadline exceeded')
                    await asyncio.sleep(0.1)
            yield process
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    await asyncio.to_thread(process.wait, 15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait)


async def verify_prompts(client, cases, expected):
    for name, case in cases.items():
        messages = [{'role': 'user', 'content': case['prompt']}]
        response = await client.post(BASE + '/apply-template', json={'messages': messages})
        response.raise_for_status()
        prompt = response.json()['prompt']
        response = await client.post(BASE + '/tokenize', json={
            'content': prompt, 'add_special': False, 'parse_special': True})
        response.raise_for_status()
        if response.json()['tokens'] != expected[name]['token_ids'] or prompt != expected[name]['rendered_prompt']:
            raise ValueError(f'comparison prompt mismatch: {name}')


async def measure(output, document, manifest, repetitions):
    cases = validate_workloads(document)
    counts = {k: len(v['token_ids']) for k, v in manifest['prompts'].items()}
    for repetition in range(repetitions):
        engines = ['runtime', 'llama_cpp'] if repetition % 2 == 0 else ['llama_cpp', 'runtime']
        for engine in engines:
            label = f'repetition-{repetition + 1}'
            async with server(engine, output, label) as process:
                async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False,
                    limits=httpx.Limits(max_connections=2, max_keepalive_connections=2)) as client:
                    if engine == 'llama_cpp':
                        await verify_prompts(client, cases, manifest['prompts'])
                    baseline = await snapshot(client, engine)
                    before_rss = await rss_bytes(process.pid)
                    specs = document['workloads'] if repetition % 2 == 0 else document['workloads'][::-1]
                    for spec in specs:
                        sizes = spec['concurrencies'] if repetition % 2 == 0 else spec['concurrencies'][::-1]
                        for concurrency in sizes:
                            report = await run_workload(client, BASE, 'local-model', cases=cases,
                                spec=spec, concurrency=concurrency, expected_prompt_tokens=counts,
                                created_policy=created_policy(engine))
                            report.update(engine=engine, repetition=repetition + 1,
                                checked_at=datetime.now(timezone.utc).isoformat(),
                                host_load_average=os.getloadavg())
                            name = f'{label}-{engine}-{spec["id"]}-c{concurrency}.json'
                            write_json(output / name, report)
                            print(f'{name}: {report["completed"]}/{spec["requests"]} completed', flush=True)
                            if report['completed'] != spec['requests']:
                                raise ValueError(f'measured failures; inspect {name}')
                    final = (await wait_state(client, engine, idle))[-1]
                    write_json(output / f'{label}-{engine}-state.json', {
                        'baseline': baseline, 'final': final, 'rss_before_bytes': before_rss,
                        'rss_after_bytes': await rss_bytes(process.pid)})


async def lifecycle(output, document, manifest):
    # Run lifecycle/RSS polling separately; it does not perturb the timing runs.
    cases = validate_workloads(document)
    counts = {k: len(v['token_ids']) for k, v in manifest['prompts'].items()}
    for engine in ('runtime', 'llama_cpp'):
        async with server(engine, output, 'lifecycle') as process:
            async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False,
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=8)) as client:
                result = {'engine': engine, 'rss_at_readiness_bytes': await rss_bytes(process.pid),
                          'initial_state': await snapshot(client, engine)}
                result['active_disconnect'] = await disconnect_active(client, engine, cases['decode'])
                result['queued_disconnect'] = await disconnect_queued(client, engine, cases['decode'])
                if engine == 'runtime':
                    result['explicit_cancel'] = await disconnect_active(client, engine, cases['decode'], explicit=True)
                timeline, done = [], asyncio.Event()
                started = perf_counter()

                async def monitor():
                    while not done.is_set():
                        timeline.append({'elapsed_ms': (perf_counter() - started) * 1000,
                            'rss_bytes': await rss_bytes(process.pid), **await snapshot(client, engine)})
                        try:
                            await asyncio.wait_for(done.wait(), 0.1)
                        except asyncio.TimeoutError:
                            pass

                task = asyncio.create_task(monitor())
                waves = []
                try:
                    for wave in range(5):
                        rows, elapsed = await pattern_workload(client, BASE, 'local-model', cases=cases,
                            pattern=['prefill', 'short', 'decode', 'short'], requests=4, concurrency=2,
                            expected_prompt_tokens=counts, created_policy=created_policy(engine))
                        if any(r['status'] != 'completed' for r in rows):
                            write_json(output / f'lifecycle-{engine}-failed-wave.json', rows)
                            raise ValueError(f'sustained observation wave failed: {[(r["case_id"], r["message"]) for r in rows]}')
                        state = (await wait_state(client, engine, idle))[-1]
                        waves.append({'wave': wave + 1, 'elapsed_seconds': elapsed, 'samples': rows,
                                      'settled_state': state, 'settled_rss_bytes': await rss_bytes(process.pid)})
                    await asyncio.sleep(1)
                finally:
                    done.set()
                    await task
                result.update(memory_timeline=timeline, sustained_waves=waves,
                              final_state=await snapshot(client, engine),
                              final_rss_bytes=await rss_bytes(process.pid))
                if engine == 'runtime' and (result['final_state']['reserved_tokens'] or result['final_state']['kv_tensor_bytes']):
                    raise ValueError('runtime logical/tensor memory not released')
                write_json(output / f'lifecycle-{engine}.json', result)
                print(f'lifecycle {engine}: completed', flush=True)


def payload(case):
    return {'model': 'local-model', 'messages': [{'role': 'user', 'content': case['prompt']}],
            'max_tokens': 512, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}}


async def disconnect_active(client, engine, case, explicit=False):
    async with client.stream('POST', BASE + '/v1/chat/completions', json=payload(case)) as response:
        response.raise_for_status()
        frames = sse_frames(response.aiter_lines())
        async for _, data in frames:
            if data != '[DONE]' and any(c.get('delta', {}).get('content')
                                       for c in json.loads(data).get('choices', [])):
                break
        before = await snapshot(client, engine)
        if before['active'] < 1:
            raise ValueError('generation finished before active disconnect could be tested')
        acted = perf_counter()
        if explicit:
            request_id = response.headers['x-request-id']
            cancelled = await client.post(BASE + f'/requests/{request_id}/cancel')
            if cancelled.status_code != 202:
                raise ValueError('explicit cancel was not accepted')
            # Keep the stream open to establish cancellation rather than disconnect cleanup.
            terminal_body = [dict(event=event, data=data) async for event, data in frames]
        else:
            terminal_body = None
    observations = await wait_state(client, engine, idle)
    elapsed = (perf_counter() - acted) * 1000
    followup = await sample(client, BASE, 'local-model', index=0,
        prompt='What is the capital of France? Answer briefly.', max_tokens=24, timeout=30,
        created_policy=created_policy(engine))
    if followup['status'] != 'completed':
        raise ValueError('follow-up after cancellation failed')
    return {'action': 'explicit_cancel' if explicit else 'close_stream_after_first_content',
            'before': before, 'observed_idle_after_ms': elapsed, 'state_observations': observations,
            'terminal_body_after_cancel': terminal_body, 'followup': followup}


async def disconnect_queued(client, engine, case):
    release = asyncio.Event()

    async def hold():
        async with client.stream('POST', BASE + '/v1/chat/completions', json=payload(case)) as response:
            response.raise_for_status()
            await release.wait()

    blockers, third = [], None
    result = {}
    try:
        blockers = [asyncio.create_task(hold()) for _ in range(2)]
        await wait_state(client, engine, lambda s: s['active'] == 2)
        third = asyncio.create_task(hold())
        queued = await wait_state(client, engine, lambda s: s['active'] == 2 and s['queued'] >= 1)
        result['queued_before_disconnect'] = queued[-1]
        acted = perf_counter()
        third.cancel()
        await asyncio.gather(third, return_exceptions=True)
        try:
            removed = await wait_state(client, engine, lambda s: s['queued'] == 0, timeout=2)
            result.update(queue_removal_observed=True, removal_observations=removed,
                          blockers_still_active_at_removal=removed[-1]['active'],
                          observed_queue_removal_after_ms=(perf_counter() - acted) * 1000)
        except TimeoutError as exc:
            result.update(queue_removal_observed=False, detail=str(exc))
    finally:
        # Always disconnect remaining streams, even if the observation timed out.
        for task in blockers + ([third] if third else []):
            task.cancel()
        await asyncio.gather(*blockers, *([third] if third else []), return_exceptions=True)
    result['idle_observations_after_all_disconnects'] = await wait_state(client, engine, idle)
    followup = await sample(client, BASE, 'local-model', index=0, prompt='What is a token?', max_tokens=24, timeout=30,
                            created_policy=created_policy(engine))
    if followup['status'] != 'completed':
        raise ValueError('queued disconnect recovery failed')
    result['followup'] = followup
    return result


async def main_async(args):
    from transformers import AutoTokenizer
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    document = json.loads(args.workloads.read_text())
    cases = validate_workloads(document)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / '.models/smollm2', local_files_only=True, trust_remote_code=False)
    previous = json.loads((ROOT / 'benchmarks/llama-cpp-smoke.json').read_text())
    build = json.loads((ROOT / 'benchmarks/llama-cpp-build.json').read_text())
    if digest(ROOT / '.experiments/smollm2-f32.gguf') != previous['gguf_sha256']:
        raise ValueError('GGUF changed since compatibility verification')
    for name, expected in build['executables_and_libraries_sha256'].items():
        if digest(ROOT / '.experiments/llama-build/bin' / name) != expected:
            raise ValueError(f'external engine build changed: {name}')
    for name, expected in previous['model_files_sha256'].items():
        if digest(ROOT / '.models/smollm2' / name) != expected:
            raise ValueError(f'checkpoint changed: {name}')
    manifest = {
        'schema_version': 1, 'started_at': datetime.now(timezone.utc).isoformat(),
        'status': 'running', 'project_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'runner_sha256': digest(Path(__file__)),
        'comparison_module_sha256': digest(ROOT / 'src/llm_runtime/comparison.py'),
        'http_client_sha256': digest(ROOT / 'src/llm_runtime/http_benchmark.py'),
        'workloads_sha256': digest(args.workloads), 'workloads': document,
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'httpx', 'uvicorn')},
        'python': platform.python_version(), 'platform': platform.platform(),
        'hardware': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize'], text=True).strip(),
        'repetitions': args.repetitions, 'commands': commands(),
        'created_timestamp_policies': {e: created_policy(e) for e in commands()},
        'model_files_sha256': previous['model_files_sha256'], 'gguf_sha256': previous['gguf_sha256'],
        'llama_build_manifest_sha256': digest(ROOT / 'benchmarks/llama-cpp-build.json'),
        'prompts': {name: {'rendered_prompt': tokenizer.apply_chat_template(
            [{'role': 'user', 'content': c['prompt']}], tokenize=False, add_generation_prompt=True),
            'token_ids': tokenizer.apply_chat_template([{'role': 'user', 'content': c['prompt']}],
                tokenize=True, add_generation_prompt=True)} for name, c in cases.items()},
        'notes': ['One server process at a time; timing runs have no concurrent RSS/health polling.',
                  'Closed-loop clients; arrival times differ with service time.',
                  'All preflight/warmup excluded; model loading excluded; no prefix reuse configured.',
                  'OS RSS observation is not live tensor memory or an enforced physical memory limit.',
                  'Same-host whole-engine comparison; kernels, libraries, and scheduling differ.']}
    write_json(output / 'manifest.json', manifest)
    try:
        await measure(output, document, manifest, args.repetitions)
        await lifecycle(output, document, manifest)
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'manifest.json', manifest)
        raise
    manifest.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat())
    write_json(output / 'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workloads', type=Path, default=ROOT / 'benchmarks/cpu-workloads-v1.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=5)
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        parser.error('repetitions must be positive')
    asyncio.run(main_async(arguments))
