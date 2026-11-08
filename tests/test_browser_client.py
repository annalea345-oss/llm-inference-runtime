"""Packaged browser resources and safe live telemetry used by the UI."""

import asyncio
from dataclasses import asdict
from html.parser import HTMLParser

import httpx
import pytest

from llm_runtime.api import create_app
from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import SimulatedExecutor


class AssetLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        value = attrs.get('src') if tag == 'script' else attrs.get('href') if tag == 'link' else None
        if value and value.startswith('/assets/'):
            self.paths.add(value)


async def test_browser_entry_and_referenced_assets_are_served_from_the_app():
    app = create_app(RuntimeConfig(prefill_delay=0, token_delay=0))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://test') as client:
            response = await client.get('/')
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('text/html')
            links = AssetLinks()
            links.feed(response.text)
            assert any(path.endswith('.js') for path in links.paths)
            assert any(path.endswith('.css') for path in links.paths)
            for path in links.paths | {'/assets/app.js', '/assets/style.css', '/assets/stream.js'}:
                asset = await client.get(path)
                assert asset.status_code == 200, path
                assert asset.content
                if path.endswith('.css'):
                    assert asset.headers['content-type'].startswith('text/css')
                elif path.endswith('.js'):
                    assert asset.headers['content-type'].split(';')[0] in (
                        'text/javascript', 'application/javascript')
            assert (await client.get('/assets/missing.js')).status_code == 404
            assert (await client.get('/assets/../config.py')).status_code == 404


@pytest.mark.parametrize('policy', ['round_robin', 'decode_first'])
@pytest.mark.parametrize('phase', ['prefill', 'decode'])
async def test_health_reports_safe_active_pending_and_queued_states_through_cancellation(policy, phase):
    entered, release = asyncio.Event(), asyncio.Event()
    config = RuntimeConfig(scheduling_policy=policy, max_active_requests=2,
                           queue_capacity=3, kv_cache_capacity_tokens=5,
                           token_budget=2, prefill_token_budget=2,
                           prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            if (step.request.prompt == 'PRIVATE-PROMPT owner' and step.phase == phase
                    and step.offset == (1 if phase == 'decode' else 0)):
                entered.set()
                await release.wait()
            return await super().execute(step)

    app = create_app(config, Executor(config))
    async with app.router.lifespan_context(app):
        scheduler = app.state.scheduler
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://test') as client:
            initial = (await client.get('/health')).json()
            assert initial['config'] == asdict(config)
            assert initial['request_states'] == []
            owner = scheduler.submit('PRIVATE-PROMPT owner', 3)
            pending = scheduler.submit('PRIVATE-PENDING', 1)
            queued = scheduler.submit('PRIVATE-QUEUED', 1)
            try:
                await asyncio.wait_for(entered.wait(), 2)
                assert scheduler._pending is pending
                response = await client.get('/health')
                health = response.json()
                assert all(secret not in response.text for secret in
                           ('PRIVATE-PROMPT', 'PRIVATE-PENDING', 'PRIVATE-QUEUED'))
                assert health['active_requests'] == 1
                assert health['active_request_ids'] == [owner.request_id]
                assert health['queued_requests'] == 2
                states = {state['request_id']: state for state in health['request_states']}
                assert set(states) == {owner.request_id, pending.request_id, queued.request_id}
                expected_keys = {'request_id', 'phase', 'prompt_tokens', 'prefilled_tokens',
                                 'generated_tokens', 'max_tokens', 'cancel_requested',
                                 'reserved_tokens', 'used_tokens'}
                assert all(set(state) == expected_keys for state in states.values())
                generated = int(phase == 'decode')
                used = 3 if phase == 'decode' else 0
                assert states[owner.request_id] == {
                    'request_id': owner.request_id, 'phase': phase, 'prompt_tokens': 2,
                    'prefilled_tokens': 2 if phase == 'decode' else 0,
                    'generated_tokens': generated, 'max_tokens': 3,
                    'cancel_requested': False, 'reserved_tokens': 5, 'used_tokens': used,
                }
                for waiting in (pending, queued):
                    assert states[waiting.request_id] == {
                        'request_id': waiting.request_id, 'phase': 'waiting', 'prompt_tokens': 1,
                        'prefilled_tokens': 0, 'generated_tokens': 0, 'max_tokens': 1,
                        'cancel_requested': False, 'reserved_tokens': 0, 'used_tokens': 0,
                    }
                assert sum(s['reserved_tokens'] for s in states.values()) == health['kv_cache']['reserved_tokens']
                assert sum(s['used_tokens'] for s in states.values()) == health['kv_cache']['used_tokens']
                # State readers get snapshots, not mutable live request objects.
                snapshot = scheduler.request_states
                snapshot[0]['generated_tokens'] = 999
                assert owner.generated_tokens == generated
                assert all(s['generated_tokens'] != 999 for s in scheduler.request_states)

                assert (await client.post(f'/requests/{pending.request_id}/cancel')).status_code == 202
                after_waiting = (await client.get('/health')).json()
                assert after_waiting['queued_requests'] == 1
                assert {s['request_id'] for s in after_waiting['request_states']} == {
                    owner.request_id, queued.request_id}

                assert (await client.post(f'/requests/{owner.request_id}/cancel')).status_code == 202
                after_active = (await client.get('/health')).json()
                owner_state = next(s for s in after_active['request_states']
                                   if s['request_id'] == owner.request_id)
                assert owner_state['cancel_requested'] is True
                assert owner_state['reserved_tokens'] == 5
                assert owner_state['used_tokens'] == used
                release.set()
                await asyncio.wait_for(scheduler.queue.join(), 2)
                assert owner.phase == pending.phase == 'cancelled'
                assert queued.phase == 'finished'
                empty = (await client.get('/health')).json()
                assert empty['request_states'] == []
                assert empty['active_requests'] == empty['queued_requests'] == 0
                assert empty['kv_cache']['reserved_tokens'] == empty['kv_cache']['used_tokens'] == 0
                assert empty['config'] == asdict(config)
                await scheduler.stop()
                stopped = (await client.get('/health')).json()
                assert stopped['status'] == 'unavailable'
                assert stopped['request_states'] == []
            finally:
                release.set()
