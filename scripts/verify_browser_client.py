"""Exercise the real browser UI against a running local server (Chrome installed)."""

import argparse
import asyncio
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import time

import httpx
from playwright.async_api import async_playwright, expect

from llm_runtime.http_benchmark import validate_options


async def verify(base_url, output_dir, channel):
    validate_options(base_url, 1, 1, 0, 1, 'check', 30)
    base_url = base_url.rstrip('/')
    output_dir.mkdir(parents=True, exist_ok=True)
    failures, submitted = [], []
    async with httpx.AsyncClient(base_url=base_url, trust_env=False) as client, async_playwright() as playwright:
        health = (await client.get('/health')).json()
        browser = await playwright.chromium.launch(channel=channel, headless=True)
        try:
            context = await browser.new_context(viewport={'width': 1440, 'height': 1040})
            page = await context.new_page()
            page.on('pageerror', lambda error: failures.append(str(error)))
            page.on('request', lambda request: submitted.append(request.post_data_json)
                    if request.method == 'POST' and request.url.endswith('/v1/chat/completions') else None)
            await page.goto(base_url)
            await expect(page.locator('#send')).to_be_enabled(timeout=20000)
            await page.locator('.settings summary').click()

            async def send(target, prompt):
                await target.locator('#prompt').fill(prompt)
                await target.locator('#send').click()

            async def finished(target):
                await expect(target.locator('.assistant.pending')).to_have_count(0, timeout=60000)

            await page.locator('#max-tokens').fill('24')
            await send(page, 'What is the capital of France? Answer briefly.')
            await finished(page)
            await expect(page.locator('.message-footer').last).to_contain_text('Finished')
            answer = await page.locator('.assistant .message-text').last.inner_text()
            assert answer
            await page.screenshot(path=str(output_dir / 'desktop.png'), full_page=True)
            await send(page, 'What was the answer you just gave? Answer briefly.')
            await finished(page)
            assert len(submitted[-1]['messages']) == 3
            assert submitted[-1]['messages'][1] == {'role': 'assistant', 'content': answer}
            await page.locator('#new-chat').click()
            await expect(page.locator('#welcome')).to_be_visible()
            assert await page.locator('.turn').count() == 0

            await page.locator('#max-tokens').fill('512')
            await send(page, 'List the numbers from one to one hundred with a short sentence about each. Continue the list without an introduction.')
            await expect(page.locator('.assistant.pending .message-text')).not_to_be_empty(timeout=30000)
            second = await context.new_page()
            second.on('pageerror', lambda error: failures.append(str(error)))
            await second.goto(base_url)
            await expect(second.locator('#send')).to_be_enabled(timeout=20000)
            await second.locator('.settings summary').click()
            await second.locator('#max-tokens').fill('24')
            await send(second, 'What is the capital of France? Answer briefly.')
            overlap = 0
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                snapshot = (await client.get('/health')).json()
                overlap = max(overlap, len(snapshot['request_states']))
                if overlap >= 2:
                    break
                await asyncio.sleep(0.02)
            assert overlap >= 2, 'did not observe concurrent requests'
            await finished(second)
            await expect(page.locator('#stop')).to_be_visible()
            await page.locator('#stop').click()
            await finished(page)
            await expect(page.locator('.message-footer').last).to_contain_text('Stopped')
            await page.locator('#max-tokens').fill('24')
            await send(page, 'What is the capital of France? Answer briefly.')
            await finished(page)
            assert len(submitted[-1]['messages']) == 1, 'cancelled turn leaked into future context'

            # These transport edge cases use explicit browser-side fixtures, not model output.
            model = await second.locator('#model').input_value()
            malicious = '<img src=x onerror="window.injected=1">'
            stage = {'value': 'literal'}
            fixture_bodies = []

            async def fixture(route):
                fixture_bodies.append(route.request.post_data_json)
                envelope = dict(id='chatcmpl-fixture', object='chat.completion.chunk', created=1, model=model)
                def chunk(delta, reason=None):
                    return {**envelope, 'choices': [dict(index=0, delta=delta, finish_reason=reason)]}
                text = malicious if stage['value'] == 'literal' else 'partial text'
                chunks = [chunk({'role': 'assistant'}), chunk({'content': text})]
                if stage['value'] != 'truncated':
                    chunks += [chunk({}, 'stop'), {**envelope, 'choices': [], 'usage': dict(prompt_tokens=2, completion_tokens=1, total_tokens=3)}]
                body = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks)
                if stage['value'] != 'truncated':
                    body += 'data: [DONE]\n\n'
                await route.fulfill(status=200, content_type='text/event-stream', body=body,
                                    headers={'X-Request-ID': 'fixture'})

            await second.route('**/v1/chat/completions', fixture)
            await second.locator('#new-chat').click()
            await send(second, 'Show literal text.')
            await finished(second)
            assert await second.locator('.assistant .message-text').inner_text() == malicious
            assert await second.locator('.conversation img').count() == 0
            assert await second.evaluate('window.injected === undefined')
            await second.locator('#new-chat').click()
            stage['value'] = 'truncated'
            await send(second, 'This request will be truncated.')
            await finished(second)
            await expect(second.locator('#notice')).to_contain_text('closed before')
            stage['value'] = 'complete'
            await send(second, 'A fresh request after failure.')
            await finished(second)
            assert len(fixture_bodies[-1]['messages']) == 1, 'failed turn leaked into future context'
            await second.unroute('**/v1/chat/completions', fixture)

            await second.route('**/health', lambda route: route.fulfill(status=503, body='unavailable'))
            await expect(second.locator('#snapshot-note')).to_contain_text('stale', timeout=10000)
            await expect(second.locator('#send')).to_be_disabled()
            await second.unroute('**/health')
            await expect(second.locator('#send')).to_be_enabled(timeout=10000)
            for width in (390, 320):
                await second.set_viewport_size({'width': width, 'height': 844})
                assert await second.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), f'horizontal overflow at {width}px'
                await expect(second.locator('#prompt')).to_be_visible()
            await second.screenshot(path=str(output_dir / 'mobile.png'), full_page=True)
            deadline = time.monotonic() + 15
            while True:
                after = (await client.get('/health')).json()
                if not after['request_states']:
                    break
                assert time.monotonic() < deadline, 'runtime failed to drain'
                await asyncio.sleep(0.05)
            assert after['kv_cache']['reserved_tokens'] == after['kv_cache']['used_tokens'] == 0
            if after['executor']['backend'] == 'transformers':
                assert after['executor']['kv_tensor_bytes_after_last_step'] == 0
            assert not failures, failures
            return dict(scope='real Chrome against local HTTP; fixture cases marked separately',
                        playwright=version('playwright'), browser=browser.version,
                        backend=health['executor']['backend'], first_answer=answer,
                        history_replay_verified=True, observed_concurrent_requests=overlap,
                        stop_and_cancelled_context_verified=True, new_chat_verified=True,
                        fixture_checks=['literal HTML rendering', 'truncated stream rejection',
                                        'failed turn omitted from history', 'stale health and recovery'],
                        mobile_widths=[390, 320], screenshots=['desktop.png', 'mobile.png'],
                        final_reserved_tokens=0, final_used_tokens=0, browser_errors=failures)
        finally:
            await browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--output-dir', type=Path, default=Path(tempfile.gettempdir()) / 'llm-runtime-browser-qa')
    parser.add_argument('--channel', choices=['chrome', 'chromium'], default='chrome')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(verify(args.base_url, args.output_dir, args.channel)), indent=2))


if __name__ == '__main__':
    main()
