"""Measure a running local server: python -m llm_runtime.http_benchmark --help."""

import argparse
import asyncio
import ipaddress
import json
import math
import sys
from time import perf_counter
from urllib.parse import urlsplit


def validate_options(base_url, requests, concurrency, warmup, max_tokens, prompt, timeout):
    parsed = urlsplit(base_url)
    try:
        local = parsed.hostname == 'localhost' or ipaddress.ip_address(parsed.hostname).is_loopback
    except (ValueError, TypeError):
        local = False
    if (parsed.scheme not in ('http', 'https') or not local or parsed.username is not None
            or parsed.password is not None or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment):
        raise ValueError('base URL must address localhost or a loopback IP, without credentials or a path')
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError('invalid base URL port') from exc
    if any(type(n) is not int or n < 1 for n in (requests, concurrency)):
        raise ValueError('requests and concurrency must be positive integers')
    if type(warmup) is not int or warmup < 0:
        raise ValueError('warmup must be a nonnegative integer')
    if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
        raise ValueError('max-tokens must be between 1 and 512')
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 8192:
        raise ValueError('prompt must contain between 1 and 8192 characters')
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')


async def sse_frames(lines):
    """Parse SSE line boundaries independently of HTTP transport chunks."""
    data, event = [], 'message'
    async for line in lines:
        if not line:
            if data:
                yield event, '\n'.join(data)
            data, event = [], 'message'
            continue
        if line.startswith(':'):
            continue
        field, _, value = line.partition(':')
        if value.startswith(' '):
            value = value[1:]
        if field == 'data':
            data.append(value)
            if sum(map(len, data)) > 1_048_576:
                raise ValueError('SSE frame exceeds one megabyte')
        elif field == 'event':
            event = value
    # SSE dispatch requires a blank line; EOF does not complete a partial frame.
    if data:
        raise ValueError('stream ended with an incomplete SSE frame')


def _usage(value):
    if not isinstance(value, dict):
        raise ValueError('missing usage object')
    keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    if any(type(value.get(k)) is not int or value[k] < 0 for k in keys):
        raise ValueError('invalid usage token counts')
    if value['total_tokens'] != value['prompt_tokens'] + value['completion_tokens']:
        raise ValueError('inconsistent usage token counts')
    return {k: value[k] for k in keys}


async def sample(client, base_url, model, *, index, prompt, max_tokens, timeout,
                 clock=perf_counter, usage_mode='required'):
    """Time receipt of headers/content/[DONE], with one total request deadline."""
    started = clock()
    row = {'index': index, 'status': 'error', 'http_status': None, 'message': None,
           'usage': None, 'usage_status': 'unverified', 'content_deltas': 0, 'finish_reason': None,
           'timings_ms': {'headers': None, 'first_content': None, 'completion': None,
                          'content_delta_gaps': []}}

    async def consume():
        payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                   'max_tokens': max_tokens, 'temperature': 0, 'stream': True}
        if usage_mode != 'disabled':
            payload['stream_options'] = {'include_usage': True}
        async with client.stream('POST', base_url + '/v1/chat/completions', json=payload,
                                 follow_redirects=False) as response:
            row['timings_ms']['headers'] = (clock() - started) * 1000
            row['http_status'] = response.status_code
            row['request_id'] = response.headers.get('x-request-id')
            if response.status_code != 200:
                row['status'] = ('rejected' if 400 <= response.status_code < 500
                                 or response.status_code == 503 else 'error')
                row['message'] = f'HTTP {response.status_code}'
                return
            if response.headers.get('content-type', '').split(';')[0].strip() != 'text/event-stream':
                raise ValueError('expected text/event-stream response')
            terminal, usage, last_content, identity, done_at = None, None, None, None, None
            async for event, data in sse_frames(response.aiter_lines()):
                if done_at is not None:
                    raise ValueError('unexpected SSE data after [DONE]')
                if event == 'error':
                    raise ValueError('server sent an SSE error event')
                if data == '[DONE]':
                    if terminal is None or (usage is None and usage_mode == 'required'):
                        raise ValueError('[DONE] requires a valid finish reason and usage')
                    if usage is not None and usage['completion_tokens'] > max_tokens:
                        raise ValueError('usage exceeds requested output limit')
                    # Record user-visible completion now, then consume the HTTP
                    # ending so HTTPX can return this connection to its pool.
                    done_at = clock()
                    continue
                frame = json.loads(data)
                if not isinstance(frame, dict) or 'error' in frame:
                    raise ValueError('invalid or error stream frame')
                if (frame.get('object') != 'chat.completion.chunk'
                        or not isinstance(frame.get('id'), str) or not frame['id']
                        or type(frame.get('created')) is not int
                        or frame.get('model') != model):
                    raise ValueError('invalid chunk metadata')
                current_identity = (frame['id'], frame['created'], frame['model'])
                if identity is not None and identity != current_identity:
                    raise ValueError('stream chunk identity changed')
                identity = current_identity
                choices = frame.get('choices')
                if choices == []:
                    if terminal is None or usage is not None:
                        raise ValueError('unexpected usage frame')
                    usage = _usage(frame.get('usage'))
                    continue
                if terminal is not None or not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError('unexpected stream choices')
                choice = choices[0]
                if (not isinstance(choice, dict) or type(choice.get('index')) is not int
                        or choice['index'] != 0):
                    raise ValueError('unexpected choice index')
                delta = choice.get('delta')
                if not isinstance(delta, dict):
                    raise ValueError('missing content delta')
                if 'role' in delta and delta['role'] != 'assistant':
                    raise ValueError('invalid delta role')
                content = delta.get('content')
                if content is not None and not isinstance(content, str):
                    raise ValueError('invalid content delta')
                if content:
                    received = clock()
                    if last_content is None:
                        row['timings_ms']['first_content'] = (received - started) * 1000
                    else:
                        row['timings_ms']['content_delta_gaps'].append((received - last_content) * 1000)
                    last_content = received
                    row['content_deltas'] += 1
                reason = choice.get('finish_reason')
                if reason is not None:
                    if reason not in ('stop', 'length'):
                        raise ValueError('unsupported finish reason')
                    terminal = reason
                # Some servers attach usage to the finish chunk instead of a
                # separate choices=[] chunk. Only terminal usage is supported.
                if frame.get('usage') is not None:
                    if terminal is None or usage is not None:
                        raise ValueError('unexpected usage before finish or duplicate usage')
                    usage = _usage(frame['usage'])
            if done_at is None:
                raise ValueError('stream ended without [DONE]')
            row.update(status='completed', usage=usage, finish_reason=terminal,
                       usage_status='verified' if usage is not None else 'not_reported')
            row['timings_ms']['completion'] = (done_at - started) * 1000

    try:
        await asyncio.wait_for(consume(), timeout)
    except asyncio.TimeoutError:
        row.update(status='timeout', message='total request deadline exceeded')
    except Exception as exc:
        # Includes HTTPX transport/read timeouts, without importing it at module load.
        row.update(status='timeout' if 'timeout' in type(exc).__name__.lower() else 'error',
                   message=f'{type(exc).__name__}: {exc}')
    row['elapsed_ms'] = (clock() - started) * 1000
    return row


async def workload(client, base_url, model, *, requests, concurrency, prompt, max_tokens,
                   timeout, clock=perf_counter, usage_mode='required'):
    rows = [None] * requests
    indices = iter(range(requests))

    async def worker():
        for index in indices:
            rows[index] = await sample(client, base_url, model, index=index, prompt=prompt,
                                       max_tokens=max_tokens, timeout=timeout, clock=clock,
                                       usage_mode=usage_mode)
    await asyncio.gather(*(worker() for _ in range(min(requests, concurrency))))
    return rows


def distribution(values):
    ordered = sorted(values)
    return {'count': len(ordered), 'mean': sum(ordered) / len(ordered) if ordered else None,
            **{f'p{p}': ordered[math.ceil(p / 100 * len(ordered)) - 1] if ordered else None
               for p in (50, 95, 99)}}


def summarize(rows, elapsed):
    completed = [r for r in rows if r['status'] == 'completed']
    verified = [r for r in completed if r['usage'] is not None]
    return {'elapsed_seconds': elapsed,
            **{name: sum(r['status'] == status for r in rows)
               for name, status in (('completed', 'completed'), ('rejected', 'rejected'),
                                    ('errors', 'error'), ('timeouts', 'timeout'))},
            'completed_requests_per_second': len(completed) / elapsed if elapsed > 0 else None,
            'usage_verified_requests': len(verified),
            'usage_missing_requests': len(completed) - len(verified),
            'completed_tokens_per_second': (
                sum(r['usage']['completion_tokens'] for r in verified) / elapsed
                if completed and len(verified) == len(completed) and elapsed > 0 else None),
            'latency_ms': {name: distribution([r['timings_ms'][name] for r in completed
                                              if r['timings_ms'][name] is not None])
                           for name in ('headers', 'first_content', 'completion')},
            'content_delta_gap_ms': distribution([gap for r in completed
                                                 for gap in r['timings_ms']['content_delta_gaps']]),
            'samples': rows}


def validate_profile(profile, model, readiness_path, usage_mode, preflight_only):
    if profile not in ('runtime', 'portable'):
        raise ValueError('profile must be runtime or portable')
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError('model must be a nonempty string')
    if usage_mode not in ('required', 'optional', 'disabled'):
        raise ValueError('usage-mode must be required, optional, or disabled')
    if type(preflight_only) is not bool:
        raise ValueError('preflight-only must be a boolean')
    if readiness_path is not None:
        if not isinstance(readiness_path, str):
            raise ValueError('readiness-path must be a local absolute path')
        parsed = urlsplit(readiness_path)
        if (not readiness_path.startswith('/') or readiness_path.startswith('//')
                or parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
                or '\\' in readiness_path or any(ord(c) <= 32 or ord(c) == 127 for c in readiness_path)):
            raise ValueError('readiness-path must be a local absolute path without query or fragment')


async def preflight(client, base_url, *, profile, model, readiness_path, prompt,
                    max_tokens, timeout, usage_mode, clock=perf_counter):
    """Exercise one real stream before measuring; never silently retry/fallback."""
    result = {'status': 'failed', 'stage': 'readiness', 'message': None,
              'readiness': {'path': readiness_path, 'status': 'skipped'},
              'model': None, 'health': None, 'sample': None}
    try:
        if readiness_path is not None:
            result['readiness']['status'] = 'failed'
            response = await asyncio.wait_for(
                client.get(base_url + readiness_path, follow_redirects=False), timeout)
            result['readiness']['http_status'] = response.status_code
            response.raise_for_status()
            try:
                body = response.json()
            except ValueError:
                body = response.text[:4096]
            result['readiness'].update(status='passed', body=body)
            if profile == 'runtime':
                result['health'] = body
        result['stage'] = 'model_selection'
        if model is not None and profile == 'portable':
            # Explicit portable selection also supports servers without discovery.
            record = {'id': model}
            result['model_selection'] = 'explicit; discovery skipped'
        else:
            response = await asyncio.wait_for(
                client.get(base_url + '/v1/models', follow_redirects=False), timeout)
            response.raise_for_status()
            body = response.json()
            records = body.get('data') if isinstance(body, dict) else None
            if (not isinstance(records, list) or not records
                    or any(not isinstance(r, dict) or not isinstance(r.get('id'), str)
                           or not r['id'].strip() for r in records)):
                raise ValueError('invalid model discovery; expected nonempty data with model IDs')
            selected = [r for r in records if model is None or r['id'] == model]
            if len(selected) != 1:
                raise ValueError('select exactly one advertised model with --model')
            record = selected[0]
            result['model_selection'] = 'discovery'
        result['model'] = record
        result['stage'] = 'stream'
        row = await sample(client, base_url, record['id'], index=-1, prompt=prompt,
                           max_tokens=max_tokens, timeout=timeout, usage_mode=usage_mode, clock=clock)
        result['sample'] = row
        if row['status'] != 'completed':
            raise ValueError(f"stream preflight {row['status']}: {row['message']}")
        result.update(status='passed', stage='complete')
    except Exception as exc:
        result['message'] = f'{type(exc).__name__}: {exc}'
    return result


async def benchmark(base_url='http://127.0.0.1:8000', *, requests=8, concurrency=2,
                    warmup=2, max_tokens=24,
                    prompt='What is the capital of France? Answer briefly.', timeout=30,
                    client=None, clock=perf_counter, profile='runtime', model=None,
                    readiness_path=None, usage_mode='required', preflight_only=False):
    validate_options(base_url, requests, concurrency, warmup, max_tokens, prompt, timeout)
    validate_profile(profile, model, readiness_path, usage_mode, preflight_only)
    if readiness_path is None and profile == 'runtime':
        readiness_path = '/health'
    base_url = base_url.rstrip('/')
    if client is None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("HTTP benchmark requires HTTPX; install pip install '.[dev]'") from exc
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False,
                                     limits=httpx.Limits(max_connections=concurrency,
                                                         max_keepalive_connections=concurrency)) as owned:
            return await benchmark(base_url, requests=requests, concurrency=concurrency,
                                   warmup=warmup, max_tokens=max_tokens, prompt=prompt,
                                   timeout=timeout, client=owned, clock=clock, profile=profile,
                                   model=model, readiness_path=readiness_path, usage_mode=usage_mode,
                                   preflight_only=preflight_only)
    checked = await preflight(client, base_url, profile=profile, model=model,
                              readiness_path=readiness_path, prompt=prompt, max_tokens=max_tokens,
                              timeout=timeout, usage_mode=usage_mode, clock=clock)
    report = {
        'schema_version': 2,
        'scope': 'external HTTP streaming; client receipt times, not model token timings',
        'base_url': base_url, 'profile': profile, 'model': checked['model'],
        'health': checked['health'], 'preflight': checked, 'preflight_only': preflight_only,
        'workload': {'requests': requests, 'concurrency': concurrency, 'warmup': warmup,
                     'prompt': prompt, 'max_tokens': max_tokens, 'timeout_seconds': timeout,
                     'arrival_mode': 'closed_loop', 'usage_mode': usage_mode},
        'connections': 'shared client pool; metadata and preflight/warmup connections reused; no retries',
        'notes': ['Metadata, one preflight stream, and warmups excluded from measured elapsed time.',
                  'Preflight uses the workload prompt and may warm server prefix caches.',
                  'Readiness checks only validate HTTP success; liveness is not measured.',
                  'Content deltas can contain multiple tokens or arrive together in network buffers.',
                  'Token throughput is null if any completed stream lacks verified terminal usage.']}
    if checked['status'] != 'passed' or preflight_only:
        return {**report, **summarize([], 0)}
    model = checked['model']['id']
    options = dict(concurrency=concurrency, prompt=prompt, max_tokens=max_tokens,
                   timeout=timeout, clock=clock, usage_mode=usage_mode)
    if warmup:
        warmed = await workload(client, base_url, model, requests=warmup, **options)
        if any(r['status'] != 'completed' for r in warmed):
            raise ValueError('warmup failed; no measured workload was run')
    started = clock()
    rows = await workload(client, base_url, model, requests=requests, **options)
    elapsed = clock() - started
    return {**report, **summarize(rows, elapsed)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--base-url', default='http://127.0.0.1:8000')
    result.add_argument('--requests', type=int, default=8)
    result.add_argument('--concurrency', type=int, default=2)
    result.add_argument('--warmup', type=int, default=2)
    result.add_argument('--max-tokens', type=int, default=24)
    result.add_argument('--prompt', default='What is the capital of France? Answer briefly.')
    result.add_argument('--timeout', type=float, default=30)
    result.add_argument('--profile', choices=('runtime', 'portable'), default='runtime',
                        help='runtime collects /health and discovers models; portable skips health by default')
    result.add_argument('--model', help='model ID; portable selection skips model discovery')
    result.add_argument('--readiness-path', help='local readiness endpoint; portable default: no readiness GET')
    result.add_argument('--usage-mode', choices=('required', 'optional', 'disabled'), default='required',
                        help='required/optional request final usage; disabled omits stream_options')
    result.add_argument('--preflight-only', action='store_true', help='check one stream; do not benchmark')
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        report = asyncio.run(benchmark(**vars(arguments)))
    except Exception as exc:
        print(f'HTTP benchmark failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    if report.get('preflight', {}).get('status') == 'failed':
        return 1
    if arguments.preflight_only:
        return 0
    return 0 if report['completed'] == arguments.requests else 1


if __name__ == '__main__':
    raise SystemExit(main())
