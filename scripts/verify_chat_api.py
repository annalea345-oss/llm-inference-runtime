"""Offline real-checkpoint chat adapter check; requires .[dev,local]."""

import argparse
import asyncio
from importlib.metadata import version
import json
from pathlib import Path
import platform

import httpx

from llm_runtime.api import create_app
from llm_runtime.config import RuntimeConfig
from llm_runtime.local_executor import LocalModelExecutor


async def verify(path):
    executor = await LocalModelExecutor.load(path)
    app = create_app(RuntimeConfig(max_active_requests=2, token_budget=8), executor)
    messages = [dict(role='system', content='You are a helpful assistant. Answer briefly.'),
                dict(role='user', content='My name is Ada.'),
                dict(role='assistant', content='Hello, Ada.'),
                dict(role='user', content='What is my name?')]
    limit = 32
    async with app.router.lifespan_context(app):
        def reference():
            tokenizer, torch = executor.tokenizer, executor.torch
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            with torch.inference_mode():
                output = executor.model.generate(
                    torch.tensor([inputs]), attention_mask=torch.ones(1, len(inputs), dtype=torch.long),
                    max_new_tokens=limit, do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )[0, len(inputs):].tolist()
            text = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return len(inputs), output, text

        prompt_count, expected_ids, expected_text = await executor.worker.run(reference)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
            models = (await client.get('/v1/models')).json()
            model_id = models['data'][0]['id']
            body = dict(model=model_id, messages=messages, max_completion_tokens=limit, temperature=0)
            complete = await client.post('/v1/chat/completions', json=body)
            streamed = await client.post('/v1/chat/completions', json={
                **body, 'stream': True, 'stream_options': {'include_usage': True},
            })
            assert complete.status_code == streamed.status_code == 200
            result = complete.json()
            assert result['choices'][0]['message']['content'] == expected_text
            expected_usage = dict(prompt_tokens=prompt_count, completion_tokens=len(expected_ids),
                                  total_tokens=prompt_count + len(expected_ids))
            assert result['usage'] == expected_usage
            frames = streamed.text.strip().split('\n\n')
            assert frames[-1] == 'data: [DONE]'
            chunks = [json.loads(frame[6:]) for frame in frames[:-1]]
            assert len({chunk['id'] for chunk in chunks}) == 1
            assert len({chunk['created'] for chunk in chunks}) == 1
            text = ''.join(chunk['choices'][0]['delta'].get('content', '')
                           for chunk in chunks if chunk['choices'])
            assert text == expected_text
            assert chunks[-1]['choices'] == [] and chunks[-1]['usage'] == expected_usage
            assert chunks[-2]['choices'][0]['finish_reason'] == result['choices'][0]['finish_reason']
            scheduler = app.state.scheduler
            await scheduler.queue.join()
            assert executor.cache_bytes == executor.active_sequences == 0
            assert not scheduler._requests
            return dict(scope='real checkpoint chat adapter; buffered in-process ASGI, not network latency',
                        python=platform.python_version(), platform=platform.platform(),
                        torch=version('torch'), transformers=version('transformers'),
                        model_manifest=json.loads((Path(path) / 'runtime_model.json').read_text()),
                        model_id=model_id, messages=messages,
                        reference_text=expected_text, reference_token_ids=expected_ids,
                        json_matches_reference=True, stream_matches_reference=True,
                        json_response=result, stream_chunk_count=len(chunks),
                        final_live_cache_bytes=executor.cache_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='.models/smollm2')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(verify(args.model_path)), indent=2))


if __name__ == '__main__':
    main()
