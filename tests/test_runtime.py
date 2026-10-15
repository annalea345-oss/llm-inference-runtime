import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from llm_runtime.api import create_app
from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import SimulatedExecutor
from llm_runtime.protocol import InferenceRequest
from llm_runtime.scheduler import FCFSScheduler, QueueFullError, RuntimeUnavailableError


async def collect(request):
    events = []
    while True:
        event = await asyncio.wait_for(request.events.get(), timeout=2)
        events.append(event)
        if event.kind in ("done", "error"):
            return events


@asynccontextmanager
async def client_for(app):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            yield client


class GatedExecutor:
    """Explicit synchronization avoids correctness assertions based on timing."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.order = []

    async def generate(self, request):
        self.order.append(request.prompt)
        self.started.set()
        await self.release.wait()
        yield request.prompt


async def test_simulation_is_deterministic():
    executor = SimulatedExecutor(RuntimeConfig(prefill_delay=0, token_delay=0))
    request = InferenceRequest("hello world", 3)
    assert [token async for token in executor.generate(request)] == [
        "hello ", "world ", "hello ",
    ]


async def test_fcfs_runs_one_request_to_completion():
    executor = GatedExecutor()
    scheduler = FCFSScheduler(executor, queue_capacity=2)
    await scheduler.start()
    try:
        first = scheduler.submit("first", 1)
        await asyncio.wait_for(executor.started.wait(), 2)
        second = scheduler.submit("second", 1)
        third = scheduler.submit("third", 1)
        assert executor.order == ["first"]
        assert second.events.empty()
        with pytest.raises(QueueFullError):
            scheduler.submit("overflow", 1)
        executor.release.set()
        for request in (first, second, third):
            events = await collect(request)
            assert [event.kind for event in events] == ["token", "done"]
        assert executor.order == ["first", "second", "third"]
        await asyncio.wait_for(scheduler.queue.join(), 2)
    finally:
        await scheduler.stop()


async def test_shutdown_terminates_active_and_waiting_requests():
    executor = GatedExecutor()
    scheduler = FCFSScheduler(executor, 2)
    with pytest.raises(RuntimeUnavailableError):
        scheduler.submit("before start", 1)
    await scheduler.start()
    first = scheduler.submit("active", 1)
    await asyncio.wait_for(executor.started.wait(), 2)
    second = scheduler.submit("waiting", 1)
    await scheduler.stop()
    assert (await collect(first))[-1].kind == "error"
    assert (await collect(second))[-1].kind == "error"
    assert not scheduler.running
    await asyncio.wait_for(scheduler.queue.join(), 2)
    await scheduler.stop()  # Idempotent cleanup.
    with pytest.raises(RuntimeUnavailableError):
        scheduler.submit("after stop", 1)


async def test_abandoned_waiting_request_is_skipped():
    executor = GatedExecutor()
    scheduler = FCFSScheduler(executor, 2)
    await scheduler.start()
    try:
        first = scheduler.submit("first", 1)
        await asyncio.wait_for(executor.started.wait(), 2)
        abandoned = scheduler.submit("abandoned", 1)
        abandoned.abandoned = True
        last = scheduler.submit("last", 1)
        executor.release.set()
        await collect(first)
        await collect(last)
        assert executor.order == ["first", "last"]
    finally:
        await scheduler.stop()


async def test_executor_error_does_not_kill_worker():
    class FailingExecutor:
        async def generate(self, request):
            if request.prompt == "fail":
                raise RuntimeError("private failure details")
            yield "ok "

    scheduler = FCFSScheduler(FailingExecutor(), 2)
    await scheduler.start()
    try:
        failed = scheduler.submit("fail", 1)
        good = scheduler.submit("good", 1)
        events = await collect(failed)
        assert events[-1].data["message"] == "model execution failed"
        assert (await collect(good))[-1].kind == "done"
        assert scheduler.running
    finally:
        await scheduler.stop()


async def test_http_stream_contract_and_health():
    app = create_app(RuntimeConfig(prefill_delay=0, token_delay=0))
    async with client_for(app) as client:
        assert (await client.get("/health")).json()["status"] == "ok"
        response = await client.post(
            "/generate", json={"prompt": "hello world", "max_tokens": 3},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = response.text.strip().split("\n\n")
        kinds = [frame.splitlines()[0] for frame in frames]
        assert kinds == ["event: token"] * 3 + ["event: done"]
        payloads = [json.loads(frame.splitlines()[1][6:]) for frame in frames]
        assert [p["text"] for p in payloads[:-1]] == ["hello ", "world ", "hello "]
        assert [p["index"] for p in payloads[:-1]] == [0, 1, 2]
        assert payloads[-1]["generated_tokens"] == 3
        assert {p["request_id"] for p in payloads} == {response.headers["x-request-id"]}
    assert not app.state.scheduler.running


@pytest.mark.parametrize("body", [
    {}, {"prompt": ""}, {"prompt": "x", "max_tokens": 0},
    {"prompt": "x", "max_tokens": 513}, {"prompt": "x", "max_tokens": 1.5},
    {"prompt": "x", "max_tokens": True}, {"prompt": "x" * 8193},
])
async def test_http_validation(body):
    async with client_for(create_app()) as client:
        assert (await client.post("/generate", json=body)).status_code == 422


async def test_http_overload_returns_503():
    executor = GatedExecutor()
    app = create_app(RuntimeConfig(queue_capacity=1), executor)
    async with client_for(app) as client:
        scheduler = app.state.scheduler
        scheduler.submit("active", 1)
        await asyncio.wait_for(executor.started.wait(), 2)
        scheduler.submit("waiting", 1)
        response = await client.post("/generate", json={"prompt": "overflow"})
        assert response.status_code == 503
        assert response.json()["detail"] == "waiting queue is full"


async def test_stream_delivers_token_before_generation_finishes():
    # httpx's ASGI transport buffers bodies. Exercise the ASGI send interface
    # directly so this test detects an accidentally buffered implementation.
    class PausingExecutor:
        async def generate(self, request):
            yield "first "
            await release.wait()
            yield "second "

    release = asyncio.Event()
    first_chunk = asyncio.Event()
    frames = []
    app = create_app(executor=PausingExecutor())
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": b'{"prompt":"x","max_tokens":2}'}
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            frames.append(message["body"])
            first_chunk.set()

    async with app.router.lifespan_context(app):
        task = asyncio.create_task(app({
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/generate", "raw_path": b"/generate", "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234), "server": ("test", 80),
        }, receive, send))
        try:
            await asyncio.wait_for(first_chunk.wait(), 2)
            assert b"first " in frames[0]
            assert not task.done()
            assert not release.is_set()
            release.set()
            await asyncio.wait_for(task, 2)
            assert b"event: done" in b"".join(frames)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
