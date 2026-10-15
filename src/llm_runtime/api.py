import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import ModelExecutor, SimulatedExecutor
from llm_runtime.protocol import GenerationRequest
from llm_runtime.scheduler import FCFSScheduler, QueueFullError, RuntimeUnavailableError


def create_app(
    config: RuntimeConfig | None = None, executor: ModelExecutor | None = None,
) -> FastAPI:
    config = config if config is not None else RuntimeConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler = FCFSScheduler(
            executor if executor is not None else SimulatedExecutor(config),
            config.queue_capacity,
        )
        app.state.scheduler = scheduler
        await scheduler.start()
        try:
            yield
        finally:
            await scheduler.stop()

    app = FastAPI(title="LLM Inference Runtime", lifespan=lifespan)

    @app.get("/health")
    async def health():
        scheduler = app.state.scheduler
        return {
            "status": "ok" if scheduler.running else "unavailable",
            "queued_requests": scheduler.queue.qsize(),
            "active_request_id": (
                scheduler.active.request_id if scheduler.active else None
            ),
        }

    @app.post("/generate")
    async def generate(body: GenerationRequest):
        try:
            request = app.state.scheduler.submit(body.prompt, body.max_tokens)
        except (QueueFullError, RuntimeUnavailableError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        async def stream():
            try:
                while True:
                    event = await request.events.get()
                    yield f"event: {event.kind}\ndata: {json.dumps(event.data)}\n\n"
                    if event.kind in ("done", "error"):
                        return
            finally:
                # StreamingResponse cancels this generator on client disconnect.
                # The worker skips queued abandoned work or stops at a token boundary.
                request.abandoned = True

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "X-Request-ID": request.request_id},
        )

    return app


app = create_app()
