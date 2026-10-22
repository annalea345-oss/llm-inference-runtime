"""Closed-loop in-process benchmark: python -m llm_runtime.benchmark --help."""

import argparse
import asyncio
from dataclasses import asdict, replace
import json
import math
import platform
from time import perf_counter

from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import SimulatedExecutor
from llm_runtime.kv_cache import RequestTooLargeError
from llm_runtime.scheduler import FCFSScheduler, QueueFullError, RuntimeUnavailableError


def distribution(values: list[float]) -> dict:
    """Nearest-rank percentiles; empty populations have null statistics."""
    ordered = sorted(values)
    return {
        'count': len(ordered),
        'mean': sum(ordered) / len(ordered) if ordered else None,
        **{f'p{p}': ordered[math.ceil(p / 100 * len(ordered)) - 1] if ordered else None
           for p in (50, 95, 99)},
    }


async def workload(scheduler, requests: int, concurrency: int, prompt: str | list[str],
                   output_pattern: list[int], timeout: float) -> list[dict]:
    results = [None] * requests
    indices = iter(range(requests))

    async def worker():
        for index in indices:
            max_tokens = output_pattern[index % len(output_pattern)]
            request_prompt = prompt[index % len(prompt)] if isinstance(prompt, list) else prompt
            prompt_tokens = max(1, len(request_prompt.split()))
            try:
                request = scheduler.submit(request_prompt, max_tokens)
            except (QueueFullError, RuntimeUnavailableError, RequestTooLargeError) as exc:
                results[index] = dict(index=index, status='rejected', message=str(exc),
                                      generated_tokens=0, max_tokens=max_tokens,
                                      prompt_tokens=prompt_tokens)
                continue
            tokens = 0
            status = 'error'
            message = None
            try:
                async def consume():
                    nonlocal tokens, status, message
                    while True:
                        event = await request.events.get()
                        if event.kind == 'token':
                            tokens += 1
                        else:
                            status = 'completed' if event.kind == 'done' else 'error'
                            message = event.data.get('message')
                            return
                await asyncio.wait_for(consume(), timeout)
                if status == 'completed' and tokens != max_tokens:
                    status, message = 'error', 'unexpected output token count'
            except asyncio.TimeoutError:
                request.abandoned = True
                status, message = 'timeout', 'request deadline exceeded'
            results[index] = dict(index=index, request_id=request.request_id,
                                  status=status, message=message,
                                  generated_tokens=tokens, max_tokens=max_tokens,
                                  prompt_tokens=prompt_tokens,
                                  timings=request.timings.snapshot())

    await asyncio.gather(*(worker() for _ in range(min(concurrency, requests))))
    return results


async def benchmark(mode: str, config: RuntimeConfig, *, requests: int,
                    concurrency: int, prompt: str, output_pattern: list[int],
                    warmup: int = 2, timeout: float = 30,
                    prompt_pattern: list[str] | None = None) -> dict:
    if mode not in ('fcfs', 'continuous'):
        raise ValueError('unknown scheduler')
    if requests < 1 or concurrency < 1 or warmup < 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('invalid benchmark size or timeout')
    prompts = [prompt] if prompt_pattern is None else prompt_pattern
    if not prompts or any(not 1 <= len(p) <= 8192 for p in prompts) or not output_pattern or any(
        type(n) is not int or not 1 <= n <= 512 for n in output_pattern
    ):
        raise ValueError('invalid workload')
    executor = SimulatedExecutor(config)
    scheduler = (ContinuousBatchScheduler(executor, config) if mode == 'continuous'
                 else FCFSScheduler(executor, config.queue_capacity))
    await scheduler.start()
    try:
        if warmup:
            warmed = await workload(scheduler, warmup, concurrency, prompts, output_pattern, timeout)
            if any(r['status'] != 'completed' for r in warmed):
                raise ValueError('warmup failed; reduce concurrency or increase capacity/timeout')
        started = perf_counter()
        rows = await workload(scheduler, requests, concurrency, prompts, output_pattern, timeout)
        elapsed = perf_counter() - started
    finally:
        await scheduler.stop()
    completed = [r for r in rows if r['status'] == 'completed']
    return {
        'scheduler': mode,
        'config': asdict(config),
        'elapsed_seconds': elapsed,
        'completed': len(completed),
        'rejected': sum(r['status'] == 'rejected' for r in rows),
        'errors': sum(r['status'] == 'error' for r in rows),
        'timeouts': sum(r['status'] == 'timeout' for r in rows),
        'completed_requests_per_second': len(completed) / elapsed,
        'completed_tokens_per_second': sum(r['generated_tokens'] for r in completed) / elapsed,
        'latency_ms': {
            name: distribution([r['timings'][name] for r in completed
                                if r['timings'][name] is not None])
            for name in ('queue_wait_ms', 'time_to_first_token_ms', 'end_to_end_ms')
        },
        'inter_token_ms': distribution([gap for r in completed
                                        for gap in r['timings']['inter_token_ms']]),
        'requests': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheduler', choices=['fcfs', 'continuous', 'both', 'policies'], default='both')
    parser.add_argument('--requests', type=int, default=20)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--prompt-words', type=int, nargs='+', default=[8],
                        help='deterministic repeating prompt-length pattern')
    parser.add_argument('--scheduling-policy', choices=['round_robin', 'decode_first'],
                        default='round_robin')
    parser.add_argument('--prefill-token-budget', type=int, default=8)
    parser.add_argument('--output-tokens', type=int, nargs='+', default=[2, 16],
                        help='deterministic repeating output-length pattern')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--queue-capacity', type=int, default=64)
    parser.add_argument('--max-active-requests', type=int, default=8)
    parser.add_argument('--token-budget', type=int, default=32)
    parser.add_argument('--kv-cache-capacity-tokens', type=int, default=8192)
    parser.add_argument('--prefill-delay', type=float, default=0.05)
    parser.add_argument('--token-delay', type=float, default=0.03)
    args = parser.parse_args()
    try:
        if any(not 1 <= n <= 4096 for n in args.prompt_words):
            raise ValueError('each prompt-words value must be between 1 and 4096')
        if any(not math.isfinite(v) or v < 0 for v in (args.prefill_delay, args.token_delay)):
            raise ValueError('delays must be finite and nonnegative')
        config = RuntimeConfig(queue_capacity=args.queue_capacity,
                               max_active_requests=args.max_active_requests,
                               token_budget=args.token_budget,
                               scheduling_policy=args.scheduling_policy,
                               prefill_token_budget=args.prefill_token_budget,
                               kv_cache_capacity_tokens=args.kv_cache_capacity_tokens,
                               prefill_delay=args.prefill_delay, token_delay=args.token_delay)

        async def run():
            if args.scheduler == 'policies':
                runs = [('continuous', replace(config, scheduling_policy=policy))
                        for policy in ('round_robin', 'decode_first')]
            else:
                modes = ['fcfs', 'continuous'] if args.scheduler == 'both' else [args.scheduler]
                runs = [(mode, config) for mode in modes]
            return [await benchmark(mode, run_config, requests=args.requests,
                                    concurrency=args.concurrency, prompt='x ' * args.prompt_words[0],
                                    prompt_pattern=['x ' * n for n in args.prompt_words],
                                    output_pattern=args.output_tokens, warmup=args.warmup,
                                    timeout=args.timeout) for mode, run_config in runs]

        reports = asyncio.run(run())
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({
        'schema_version': 2,
        'scope': 'in-process simulated scheduler; no HTTP or GPU measurement',
        'python': platform.python_version(), 'platform': platform.platform(),
        'config': asdict(config), 'workload': vars(args), 'runs': reports,
    }, indent=2, allow_nan=False))
    if any(r['rejected'] or r['errors'] or r['timeouts'] for r in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
