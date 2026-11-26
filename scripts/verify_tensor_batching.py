"""Compare optional tensor batching with this runtime's unbatched CPU path.

Uses existing local checkpoints only. Owns one temporary loopback server on 8020
at a time and stops it in finally. Timing, correctness, and lifecycle checks are
separate. Output directory must be new; checkpoints are written after each run.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import signal
import socket
import statistics
import subprocess
from time import perf_counter

import httpx

from llm_runtime.http_benchmark import benchmark, sse_frames
from llm_runtime.local_executor import LocalModelExecutor
from scripts.compare_engines import (BASE, ROOT, digest, disconnect_queued, rss_bytes,
                                    snapshot, wait_state, idle, write_json)


COUNT = 'Count from 1 to 500. Include every number.'
CASES = [
    {'id': 'france', 'messages': [{'role': 'user', 'content': 'What is the capital of France? Answer briefly.'}], 'max_tokens': 24},
    {'id': 'count', 'messages': [{'role': 'user', 'content': COUNT}], 'max_tokens': 96},
    {'id': 'unicode', 'messages': [{'role': 'user', 'content': 'Hva betyr blåbær? Svar kort.'}], 'max_tokens': 32},
    {'id': 'history', 'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant. Answer briefly.'},
        {'role': 'user', 'content': 'My name is Ada.'},
        {'role': 'assistant', 'content': 'Hello, Ada.'},
        {'role': 'user', 'content': 'What is my name?'}], 'max_tokens': 24},
]
MODELS = {'smollm2': ROOT / '.models/smollm2', 'qwen': ROOT / '.models/qwen2.5-0.5b'}


async def reference(path):
    executor = await LocalModelExecutor.load(str(path))
    try:
        def generate():
            result = []
            with executor.torch.inference_mode():
                for case in CASES:
                    ids = executor.tokenize_messages(case['messages'])
                    output = executor.model.generate(
                        executor.torch.tensor([ids]),
                        attention_mask=executor.torch.ones(1, len(ids), dtype=executor.torch.long),
                        max_new_tokens=case['max_tokens'], do_sample=False,
                        repetition_penalty=1.0, use_cache=True,
                        pad_token_id=executor.tokenizer.eos_token_id,
                    )[0, len(ids):].tolist()
                    result.append({**case, 'prompt_token_ids': list(ids), 'token_ids': output,
                        'text': executor.tokenizer.decode(output, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False),
                        'finish_reason': 'eos' if output[-1] in executor.eos_ids else 'length'})
            return result
        return {'manifest': json.loads((path / 'runtime_model.json').read_text()),
                'model_files_sha256': {p.name: digest(p) for p in path.iterdir() if p.is_file()},
                'executor': executor.info, 'cases': await executor.worker.run(generate)}
    finally:
        await executor.aclose()


@asynccontextmanager
async def server(path, enabled, log_path, *, attention_implementation='eager'):
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 8020)) == 0:
            raise RuntimeError('port 8020 already has a listener; refusing to attach')
    command = [str(ROOT / '.venv/bin/python'), '-m', 'uvicorn', 'llm_runtime.local:create_local_app',
               '--factory', '--host', '127.0.0.1', '--port', '8020', '--no-access-log']
    process = None
    with log_path.open('w') as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'LLM_MODEL_PATH': str(path), 'LLM_TENSOR_BATCHING': str(int(enabled)),
                     'LLM_ATTENTION_IMPLEMENTATION': attention_implementation,
                     'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
            started = perf_counter()
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'server startup failed: {log_path}')
                    try:
                        response = await client.get(BASE + '/health')
                        if response.status_code == 200:
                            assert response.json()['executor']['tensor_batching'] is enabled
                            assert response.json()['executor']['attention_implementation'] == attention_implementation
                            break
                    except httpx.TransportError:
                        pass
                    if perf_counter() - started > 90:
                        raise TimeoutError('server readiness deadline exceeded')
                    await asyncio.sleep(0.1)
            yield process
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    await asyncio.to_thread(process.wait, 20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait)


async def health(client):
    response = await client.get(BASE + '/health')
    response.raise_for_status()
    return response.json()


async def clean(client):
    await wait_state(client, 'runtime', idle, timeout=30)
    state = await snapshot(client, 'runtime')
    assert state['reserved_tokens'] == state['used_tokens'] == state['kv_tensor_bytes'] == 0
    return state


async def correctness(client, references):
    async def native(case):
        response = await client.post(BASE + '/generate', json={
            'prompt': case['messages'][0]['content'], 'max_tokens': case['max_tokens']})
        response.raise_for_status()
        frames = [(kind, json.loads(data)) async for kind, data in sse_frames(response.aiter_lines())]
        assert frames[-1][0] == 'done', frames
        assert [data['token_id'] for kind, data in frames if kind == 'token'] == case['token_ids']
        assert ''.join(data['text'] for kind, data in frames if kind == 'token') == case['text']
        assert frames[-1][1]['finish_reason'] == case['finish_reason']
        return {'case': case['id'], 'token_ids_match': True, 'text_match': True,
                'terminal': frames[-1][1]}

    rows = await asyncio.gather(*(native(c) for c in references[:3]))
    # Identical counting prompts exercise the exact shape used by timing runs.
    identical = await asyncio.gather(native(references[1]), native(references[1]))
    case = references[3]
    response = await client.post(BASE + '/v1/chat/completions', json={
        'model': 'local-model', 'messages': case['messages'], 'max_tokens': case['max_tokens'],
        'temperature': 0, 'stream': False})
    response.raise_for_status()
    chat = response.json()
    assert chat['choices'][0]['message']['content'] == case['text']
    assert chat['usage']['completion_tokens'] == len(case['token_ids'])
    return {'ragged_native_requests': rows, 'identical_native_pair': identical,
            'history_json_matches': True, 'history_usage': chat['usage'], 'final': await clean(client)}


async def paired_cancellation(client, explicit):
    ready = [asyncio.Future(), asyncio.Future()]
    gates = [asyncio.Event(), asyncio.Event()]

    async def hold(index):
        async with client.stream('POST', BASE + '/v1/chat/completions', json={
            'model': 'local-model', 'messages': [{'role': 'user', 'content': COUNT}],
            'max_tokens': 256, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}}) as response:
            response.raise_for_status()
            frames = sse_frames(response.aiter_lines())
            body = []
            async for _, data in frames:
                body.append(data)
                if data != '[DONE]' and any(c.get('delta', {}).get('content') for c in json.loads(data).get('choices', [])):
                    ready[index].set_result(response.headers['x-request-id'])
                    break
            if not ready[index].done():
                raise RuntimeError('stream ended before any text')
            await gates[index].wait()
            body.extend([data async for _, data in frames])
            return body

    tasks = [asyncio.create_task(hold(i)) for i in range(2)]
    try:
        ids = await asyncio.wait_for(asyncio.gather(*ready), 30)
        before = await health(client)
        assert before['active_requests'] == 2
        if explicit:
            response = await client.post(BASE + f'/requests/{ids[0]}/cancel')
            assert response.status_code == 202
        else:
            tasks[0].cancel()
            await asyncio.gather(tasks[0], return_exceptions=True)
        for gate in gates:
            gate.set()
        if explicit:
            cancelled = await asyncio.wait_for(tasks[0], 30)
            assert cancelled[-1] == '[DONE]'
            assert any(json.loads(d).get('error', {}).get('code') == 'request_cancelled'
                       for d in cancelled if d != '[DONE]')
        peer = await asyncio.wait_for(tasks[1], 90)
        assert peer[-1] == '[DONE]'
        assert any(json.loads(d).get('usage') for d in peer if d != '[DONE]')
        return {'action': 'explicit_cancel' if explicit else 'disconnect',
                'active_before': before['active_requests'], 'peer_completed': True,
                'executor_before': before['executor'], 'final': await clean(client)}
    finally:
        for gate in gates:
            gate.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def main_async(args):
    if args.repetitions < 1:
        raise ValueError('repetitions must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': version('torch'), 'transformers': version('transformers'),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: digest(ROOT / name) for name in [
            'src/llm_runtime/local_executor.py', 'src/llm_runtime/batching.py',
            'src/llm_runtime/local.py', 'src/llm_runtime/http_benchmark.py',
            'scripts/verify_tensor_batching.py']},
        'protocol': {'requests_per_run': 4, 'max_tokens': 96, 'concurrency': [1, 2],
            'repetitions': args.repetitions, 'prompt': COUNT,
            'scope': 'loopback HTTP receipt timings; CPU float32/eager, two Torch threads, max active two',
            'order': 'fresh server per mode/repetition; alternate off/on order each repetition',
            'excluded': 'model loading, reference/correctness/lifecycle checks, one preflight and one warmup per timing run',
            'rss': 'readiness and post-run RSS samples only; not peak RSS or a physical memory cap'}}
    write_json(output / 'manifest.json', manifest)
    reports = []
    for name in args.models:
        path = MODELS[name]
        refs = await reference(path)
        write_json(output / f'{name}-reference.json', refs)
        print(f'{name}: direct Transformers references ready', flush=True)
        for rep in range(args.repetitions):
            for enabled in ([False, True] if rep % 2 == 0 else [True, False]):
                label = f'{name}-r{rep + 1}-{"on" if enabled else "off"}'
                async with server(path, enabled, output / f'{label}.log') as process:
                    async with httpx.AsyncClient(timeout=120, trust_env=False,
                        limits=httpx.Limits(max_connections=8, max_keepalive_connections=8)) as client:
                        readiness_rss = await rss_bytes(process.pid)
                        if rep == 0:
                            verified = await correctness(client, refs['cases'])
                            verified['executor'] = (await health(client))['executor']
                            assert verified['executor']['max_forward_batch_size'] == (2 if enabled else 1)
                            write_json(output / f'{label}-correctness.json', verified)
                        for concurrency in (1, 2):
                            before = (await health(client))['executor']
                            report = await benchmark(BASE, requests=4, concurrency=concurrency,
                                warmup=1, max_tokens=96, prompt=COUNT, timeout=120, client=client)
                            assert report['completed'] == report['usage_verified_requests'] == 4
                            assert all(row['usage']['completion_tokens'] == len(refs['cases'][1]['token_ids'])
                                       for row in report['samples'])
                            report.update(model_name=name, tensor_batching=enabled, repetition=rep + 1,
                                executor_before=before, executor_after=(await health(client))['executor'],
                                final=await clean(client), readiness_rss_bytes=readiness_rss,
                                after_run_rss_bytes=await rss_bytes(process.pid), host_load=os.getloadavg())
                            if enabled and concurrency == 2:
                                assert report['executor_after']['batched_forward_calls'] > before['batched_forward_calls']
                            write_json(output / f'{label}-c{concurrency}.json', report)
                            reports.append(report)
                            print(f'{label} c{concurrency}: {report["completed_tokens_per_second"]:.2f} tokens/s', flush=True)
                        if rep == 0:
                            lifecycle = {'paired_cancel': await paired_cancellation(client, True),
                                'paired_disconnect': await paired_cancellation(client, False),
                                'queued_disconnect': await disconnect_queued(client, 'runtime', {'prompt': COUNT}),
                                'final': await clean(client)}
                            assert lifecycle['queued_disconnect']['queue_removal_observed']
                            assert lifecycle['queued_disconnect']['blockers_still_active_at_removal'] == 2
                            write_json(output / f'{label}-lifecycle.json', lifecycle)
    summary = []
    for name in args.models:
        for concurrency in (1, 2):
            item = {'model': name, 'concurrency': concurrency}
            for enabled in (False, True):
                selected = [r for r in reports if r['model_name'] == name
                            and r['tensor_batching'] is enabled and r['workload']['concurrency'] == concurrency]
                item['on' if enabled else 'off'] = {
                    'median_tokens_per_second': statistics.median(r['completed_tokens_per_second'] for r in selected),
                    'median_first_content_p50_ms': statistics.median(r['latency_ms']['first_content']['p50'] for r in selected),
                    'runs': len(selected)}
            item['throughput_ratio'] = item['on']['median_tokens_per_second'] / item['off']['median_tokens_per_second']
            summary.append(item)
    manifest.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
                    measured_requests=sum(r['completed'] for r in reports), measured_runs=len(reports), summary=summary)
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--repetitions', type=int, default=3)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == '__main__':
    main()
