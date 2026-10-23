"""Text-only Chat Completions adapter over the existing request scheduler."""

import asyncio
import json
import logging
from time import time
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_runtime.kv_cache import RequestTooLargeError
from llm_runtime.scheduler import ChatUnsupportedError, QueueFullError, RuntimeUnavailableError

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    role: Literal['system', 'user', 'assistant']
    content: str = Field(min_length=1, max_length=8192)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    model: str = Field(min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=512)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=512)
    temperature: float | None = Field(default=None, ge=0, le=0)
    n: int = Field(default=1, ge=1, le=1)
    stream_options: StreamOptions | None = None

    @model_validator(mode='after')
    def supported_conversation(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError('use max_tokens or max_completion_tokens, not both')
        if self.stream_options is not None and not self.stream:
            raise ValueError('stream_options requires stream=true')
        if sum(len(m.content) for m in self.messages) > 8192:
            raise ValueError('total message content exceeds 8192 characters')
        turns = self.messages[1:] if self.messages[0].role == 'system' else self.messages
        if not turns or turns[-1].role != 'user':
            raise ValueError('conversation must end with a user message')
        for index, message in enumerate(turns):
            if message.role != ('user' if index % 2 == 0 else 'assistant'):
                raise ValueError('expected alternating user/assistant turns after an optional initial system')
        return self

    @property
    def output_limit(self):
        return self.max_completion_tokens or self.max_tokens or 16


class ChatError(Exception):
    def __init__(self, status: int, message: str, code: str, *, param=None):
        self.status = status
        self.payload = {'error': {
            'message': message,
            'type': 'server_error' if status >= 500 else 'invalid_request_error',
            'param': param, 'code': code,
        }}


def _terminal_error(event):
    if event.kind == 'error':
        shutdown = event.data['message'] == 'runtime is shutting down'
        return ChatError(503 if shutdown else 500, event.data['message'],
                         'runtime_unavailable' if shutdown else 'execution_failed')
    if event.data['finish_reason'] == 'cancelled':
        return ChatError(409, 'request was cancelled', 'request_cancelled')
    return None


def _usage(request, terminal):
    return {'prompt_tokens': request.prompt_token_count,
            'completion_tokens': terminal.data['generated_tokens'],
            'total_tokens': request.prompt_token_count + terminal.data['generated_tokens']}


async def _complete(request, http_request):
    """Collect before sending headers; listen for nonstream HTTP disconnects."""
    async def consume():
        pieces = []
        while True:
            event = await request.events.get()
            if event.kind != 'token':
                return ''.join(pieces), event
            pieces.append(event.data['text'])

    async def disconnected():
        while True:
            message = await http_request.receive()
            if message['type'] == 'http.disconnect':
                return

    consume_task = asyncio.create_task(consume())
    disconnect_task = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait([consume_task, disconnect_task],
                                     return_when=asyncio.FIRST_COMPLETED)
        if consume_task in done:
            return consume_task.result()
        disconnect_task.result()
        raise ChatError(499, 'client disconnected', 'client_disconnected')
    finally:
        request.abandoned = True
        for task in (consume_task, disconnect_task):
            task.cancel()
        await asyncio.gather(consume_task, disconnect_task, return_exceptions=True)


def install_chat_routes(app: FastAPI, stream_response) -> None:
    available_since = int(time())

    def model_record():
        executor = app.state.scheduler.executor
        return {'id': getattr(executor, 'model_id', 'runtime'), 'object': 'model',
                'created': available_since, 'owned_by': 'llm-runtime'}

    @app.exception_handler(ChatError)
    async def chat_error(_request, exc):
        return JSONResponse(exc.payload, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        if not request.url.path.startswith('/v1/'):
            return await request_validation_exception_handler(request, exc)
        error = exc.errors()[0]
        param = '.'.join(str(p) for p in error['loc'] if p != 'body') or None
        rejected = ChatError(400, error['msg'], 'invalid_request', param=param)
        return JSONResponse(rejected.payload, status_code=400)

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [model_record()]}

    @app.get('/v1/models/{model_id}')
    async def model(model_id: str):
        record = model_record()
        if model_id != record['id']:
            raise ChatError(404, 'model is not served by this runtime', 'model_not_found', param='model')
        return record

    @app.post('/v1/chat/completions')
    async def chat(body: ChatRequest, http_request: Request):
        if body.model != model_record()['id']:
            raise ChatError(404, 'model is not served by this runtime', 'model_not_found', param='model')
        scheduler = app.state.scheduler
        try:
            request = scheduler.submit_chat([m.model_dump() for m in body.messages], body.output_limit)
        except (QueueFullError, RuntimeUnavailableError) as exc:
            raise ChatError(503, str(exc), 'runtime_unavailable') from exc
        except RequestTooLargeError as exc:
            raise ChatError(422, str(exc), 'context_capacity_exceeded') from exc
        except ChatUnsupportedError as exc:
            raise ChatError(501, str(exc), 'chat_not_supported') from exc
        except Exception as exc:
            logger.exception('Conversation preparation failed')
            raise ChatError(500, 'conversation preparation failed', 'preparation_failed') from exc
        created = int(time())
        completion_id = f'chatcmpl-{request.request_id}'

        def payload(choices, *, chunk=False):
            return {'id': completion_id, 'object': 'chat.completion.chunk' if chunk else 'chat.completion',
                    'created': created, 'model': body.model, 'choices': choices}

        def reason(terminal):
            return 'stop' if terminal.data['finish_reason'] == 'eos' else 'length'

        if not body.stream:
            text, terminal = await _complete(request, http_request)
            error = _terminal_error(terminal)
            if error is not None:
                raise error
            result = payload([{'index': 0, 'message': {'role': 'assistant', 'content': text},
                               'finish_reason': reason(terminal)}])
            result['usage'] = _usage(request, terminal)
            return JSONResponse(result, headers={'X-Request-ID': request.request_id})

        include_usage = body.stream_options is not None and body.stream_options.include_usage

        def chunk(delta, finish_reason=None):
            result = payload([{'index': 0, 'delta': delta, 'finish_reason': finish_reason}], chunk=True)
            if include_usage:
                result['usage'] = None
            return f'data: {json.dumps(result)}\n\n'

        async def stream():
            yield chunk({'role': 'assistant', 'content': ''})
            while True:
                event = await request.events.get()
                if event.kind == 'token':
                    if event.data['text']:
                        yield chunk({'content': event.data['text']})
                    continue
                error = _terminal_error(event)
                if error is not None:
                    yield f'data: {json.dumps(error.payload)}\n\n'
                else:
                    yield chunk({}, reason(event))
                    if include_usage:
                        result = payload([], chunk=True)
                        result['usage'] = _usage(request, event)
                        yield f'data: {json.dumps(result)}\n\n'
                yield 'data: [DONE]\n\n'
                return

        return stream_response(request, stream())
