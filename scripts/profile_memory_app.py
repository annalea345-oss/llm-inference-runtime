"""Private loopback experiment factory; normal llm_runtime factories are unchanged."""

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import HTTPException

from llm_runtime.local import create_local_app
from scripts.compare_engines import write_json
from scripts.memory_diagnostics import MemoryProbe


def create_profile_app():
    app = create_local_app()
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        probe = None
        try:
            async with original_lifespan(app):
                probe = MemoryProbe(app.state.scheduler)
                app.state.memory_probe = probe
                yield
        finally:
            if probe is not None:
                # Original model/worker cleanup has finished, but the child
                # process still exists for one final lifetime/allocator snapshot.
                write_json(Path(os.environ['LLM_PROFILE_SHUTDOWN_RECORD']), probe.snapshot())

    app.router.lifespan_context = lifespan

    @app.get('/__memory/snapshot')
    async def snapshot():
        try:
            return app.state.memory_probe.snapshot()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/__memory/{operation}')
    async def operate(operation: str):
        try:
            return app.state.memory_probe.operate(operation)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app
