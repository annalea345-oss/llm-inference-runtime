import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import ModelExecutor, SimulatedExecutor
from llm_runtime.kv_cache import RequestTooLargeError
from llm_runtime.protocol import GenerationRequest, InferenceRequest
from llm_runtime.scheduler import QueueFullError, RuntimeUnavailableError
from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.chat import install_chat_routes


class GenerationResponse(StreamingResponse):
    def __init__(self, request: InferenceRequest, content: AsyncIterator[str]) -> None:
        self.request = request
        super().__init__(
            content, media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "X-Request-ID": request.request_id},
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Covers header sends and body sends, even before iteration starts.
            self.request.abandoned = True


def create_app(
    config: RuntimeConfig | None = None, executor: ModelExecutor | None = None,
    *, executor_loader: Callable[[], Awaitable[ModelExecutor]] | None = None,
) -> FastAPI:
    if executor is not None and executor_loader is not None:
        raise ValueError("provide executor or executor_loader, not both")
    config = config if config is not None else RuntimeConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        selected = (await executor_loader() if executor_loader is not None else
                    executor if executor is not None else SimulatedExecutor(config))
        scheduler = None
        try:
            scheduler = ContinuousBatchScheduler(selected, config)
            app.state.scheduler = scheduler
            await scheduler.start()
            yield
        finally:
            try:
                if scheduler is not None:
                    await scheduler.stop()
            finally:
                close = getattr(selected, "aclose", None)
                if callable(close):
                    await close()

    app = FastAPI(title="LLM Inference Runtime", lifespan=lifespan)
    web_directory = Path(__file__).with_name("web")
    app.mount("/assets", StaticFiles(directory=web_directory), name="assets")

    @app.get("/", include_in_schema=False)
    async def browser_client():
        return FileResponse(web_directory / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health():
        scheduler = app.state.scheduler
        return {
            "status": "ok" if scheduler.running else "unavailable",
            "queued_requests": scheduler.queued_requests,
            "config": asdict(scheduler.config),
            "request_states": scheduler.request_states,
            "executor": getattr(scheduler.executor, "info", {"backend": "custom"}),
            "kv_cache": {
                "capacity_tokens": scheduler.kv_cache.capacity_tokens,
                "reserved_tokens": scheduler.kv_cache.reserved_tokens,
                "used_tokens": scheduler.kv_cache.used_tokens,
                "available_tokens": scheduler.kv_cache.available_tokens,
            },
            "active_request_ids": [r.request_id for r in scheduler.active],
            "active_requests": len(scheduler.active),
            "active_request_id": (
                scheduler.active[0].request_id if scheduler.active else None
            ),
        }

    @app.post("/requests/{request_id}/cancel", status_code=202)
    async def cancel(request_id: str):
        try:
            accepted = app.state.scheduler.cancel(request_id)
        except RuntimeUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not accepted:
            raise HTTPException(status_code=404, detail="request is not live")
        return {"request_id": request_id, "status": "cancellation_requested"}

    @app.post("/generate")
    async def generate(body: GenerationRequest):
        try:
            request = app.state.scheduler.submit(body.prompt, body.max_tokens)
        except (QueueFullError, RuntimeUnavailableError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RequestTooLargeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        async def stream():
            while True:
                event = await request.events.get()
                yield f"event: {event.kind}\ndata: {json.dumps(event.data)}\n\n"
                if event.kind in ("done", "error"):
                    return

        return GenerationResponse(request, stream())

    install_chat_routes(app, GenerationResponse)
    return app


app = create_app()
