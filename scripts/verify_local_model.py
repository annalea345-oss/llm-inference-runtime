"""Offline checkpoint smoke check; install .[dev,local] and download first."""

import argparse
import asyncio
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import json
import platform
import resource
import sys
from time import perf_counter

import httpx

from llm_runtime.api import create_app
from llm_runtime.config import RuntimeConfig
from llm_runtime.local_executor import LocalModelExecutor


async def verify(path):
    started = perf_counter()
    executor = await LocalModelExecutor.load(path)
    load_seconds = perf_counter() - started
    config = RuntimeConfig(max_active_requests=2, token_budget=8)
    app = create_app(config, executor)
    prompts = ['What is the capital of France? Answer briefly.', 'What color is the sky on a clear day?']
    max_tokens = 24
    peak_cache = 0
    entered, release = asyncio.Event(), asyncio.Event()
    execute = executor.execute

    async def observe(step):
        nonlocal peak_cache
        result = await execute(step)
        peak_cache = max(peak_cache, executor.cache_bytes)
        if step.request.prompt == 'cancellation check' and step.phase == 'decode':
            entered.set()
            await release.wait()
        return result

    executor.execute = observe
    async with app.router.lifespan_context(app):
        def reference(prompt):
            ids = executor.tokenize(prompt)
            torch = executor.torch
            with torch.inference_mode():
                output = executor.model.generate(
                    torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    max_new_tokens=max_tokens, do_sample=False, use_cache=True,
                    pad_token_id=executor.tokenizer.eos_token_id,
                )[0, len(ids):].tolist()
            return output

        expected = [await executor.worker.run(reference, p) for p in prompts]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://local') as client:
            responses = await asyncio.gather(*(client.post('/generate', json={
                'prompt': p, 'max_tokens': max_tokens,
            }) for p in prompts))
            cases = []
            for prompt, response, ids in zip(prompts, responses, expected):
                assert response.status_code == 200
                events = [json.loads(frame.splitlines()[1][6:])
                          for frame in response.text.strip().split('\n\n')]
                terminal, pieces = events[-1], events[:-1]
                assert [e['token_id'] for e in pieces] == ids
                text = ''.join(e['text'] for e in pieces)
                assert text == executor.tokenizer.decode(ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
                assert terminal['finish_reason'] in ('eos', 'length')
                cases.append(dict(prompt=prompt, output=text, generated_tokens=len(ids),
                                  reference_token_ids_match=True, terminal=terminal))
            scheduler = app.state.scheduler
            assert executor.active_sequences == executor.cache_bytes == 0
            request = scheduler.submit('cancellation check', max_tokens)
            try:
                await asyncio.wait_for(entered.wait(), 30)
                health = (await client.get('/health')).json()
                assert health['active_requests'] == 1
                assert health['executor']['backend'] == 'transformers'
                response = await client.post(f'/requests/{request.request_id}/cancel')
                assert response.status_code == 202
                assert executor.active_sequences == 1
                release.set()
                event = await asyncio.wait_for(request.events.get(), 30)
                assert event.kind == 'done' and event.data['finish_reason'] == 'cancelled'
                assert request.generated_tokens == 0
                await asyncio.wait_for(scheduler.queue.join(), 30)
            finally:
                release.set()
            assert executor.active_sequences == executor.cache_bytes == 0
            assert not scheduler._requests
        return dict(scope='real local CPU model; in-process ASGI; no network latency measurement',
                    python=platform.python_version(), platform=platform.platform(),
                    torch=executor.torch.__version__,
                    transformers=version('transformers'),
                    model_manifest=json.loads((Path(path) / 'runtime_model.json').read_text()),
                    config=asdict(config), executor=executor.info,
                    load_seconds=load_seconds,
                    peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                    (1 if sys.platform == 'darwin' else 1024),
                    peak_observed_kv_tensor_bytes=peak_cache,
                    cancellation_verified=True, final_cache_bytes=executor.cache_bytes,
                    requests=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='.models/smollm2')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(verify(args.model_path)), indent=2))


if __name__ == '__main__':
    main()
