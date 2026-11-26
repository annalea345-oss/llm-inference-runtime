"""Optional CPU Transformers executor; downloads are a separate explicit step."""

from dataclasses import dataclass, field
import asyncio
import logging
from pathlib import Path
from typing import Any

from llm_runtime.protocol import DecodeResult, ExecutionStep
from llm_runtime.worker import SerialWorker

logger = logging.getLogger(__name__)


@dataclass
class _Sequence:
    cache: Any = None
    next_token: int | None = None
    prefilled: int = 0
    generated: list[int] = field(default_factory=list)
    emitted_text: str = ''


class LocalModelExecutor:
    """One CPU model, serialized forwards, independent per-request KV tensors.

    Logical token reservations remain admission accounting, not a physical RAM
    limit. release_request() runs after that request's execute() has settled.
    """

    model_id = 'local-model'

    def __init__(self, model, tokenizer, torch_module, worker: SerialWorker, *,
                 tensor_batching: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.worker = worker
        self.context_limit = int(model.config.max_position_embeddings)
        eos = model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        self._states: dict[str, _Sequence] = {}
        self._cache_bytes = 0
        self._model_type = model.config.model_type
        self._attention_implementation = model.config._attn_implementation
        self._parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        if type(tensor_batching) is not bool:
            raise ValueError('tensor_batching must be a boolean')
        if tensor_batching and (self._model_type not in ('llama', 'qwen2') or
                any(kind != 'full_attention' for kind in getattr(model.config, 'layer_types', []) or []) or
                getattr(model.config, 'use_sliding_window', False) or
                (self._model_type == 'llama' and getattr(model.config, 'sliding_window', None))):
            raise ValueError('tensor batching supports full-attention Llama and Qwen2 models only')
        self.tensor_batching = tensor_batching
        self._forward_calls = 0
        self._batched_forward_calls = 0
        self._max_forward_batch_size = 0

    @classmethod
    async def load(cls, model_path: str, *, tensor_batching: bool = False,
                   attention_implementation: str = 'eager') -> 'LocalModelExecutor':
        if attention_implementation not in ('eager', 'sdpa'):
            raise ValueError('attention_implementation must be eager or sdpa')
        worker = SerialWorker()

        def load():
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("install the optional local dependencies: pip install '.[local]'") from exc
            path = Path(model_path).expanduser().resolve()
            if not path.is_dir():
                raise ValueError('model path must be an existing local directory')
            torch.set_num_threads(2)
            tokenizer = AutoTokenizer.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False,
                use_safetensors=True, dtype=torch.float32,
                attn_implementation=attention_implementation,
            ).to('cpu').eval()
            if model.config._attn_implementation != attention_implementation:
                raise RuntimeError('model did not select the requested attention implementation')
            return cls(model, tokenizer, torch, worker, tensor_batching=tensor_batching)

        try:
            return await worker.run(load)
        except BaseException:
            await worker.aclose()
            raise

    @property
    def info(self) -> dict:
        return {'backend': 'transformers', 'device': 'cpu', 'dtype': 'float32',
                'tokenization': 'model', 'context_limit': self.context_limit,
                'model': self._model_type, 'parameter_bytes': self._parameter_bytes,
                'kv_tensor_bytes_after_last_step': self.cache_bytes,
                'tensor_batching': self.tensor_batching,
                'attention_implementation': (self.model.config._attn_implementation
                    if self.model is not None else self._attention_implementation),
                'forward_calls': self._forward_calls,
                'batched_forward_calls': self._batched_forward_calls,
                'max_forward_batch_size': self._max_forward_batch_size}

    @property
    def active_sequences(self) -> int:
        return len(self._states)

    @property
    def cache_bytes(self) -> int:
        """Latest settled-step snapshot; excludes weights and temporaries."""
        return self._cache_bytes

    def _measure_cache(self) -> int:
        total = 0
        for state in self._states.values():
            if state.cache is not None:
                for keys, values in state.cache:
                    total += keys.numel() * keys.element_size()
                    total += values.numel() * values.element_size()
        return total

    def tokenize(self, prompt: str) -> tuple[int, ...]:
        # Bounded HTTP prompts use the fast tokenizer on the owning event loop.
        # Applying the template makes this endpoint a single-turn instruction.
        return self.tokenize_messages([{'role': 'user', 'content': prompt}])

    def tokenize_messages(self, messages: list[dict[str, str]]) -> tuple[int, ...]:
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True,
            add_generation_prompt=True,
        )
        if not ids:
            raise ValueError('model tokenizer returned an empty prompt')
        return tuple(ids)

    async def execute(self, step: ExecutionStep) -> DecodeResult | None:
        return await self.worker.run(self._execute, step)

    async def execute_batch(self, steps: tuple[ExecutionStep, ...]) -> list:
        if not self.tensor_batching:
            # Preserve the original path, including execute() observation hooks.
            return await asyncio.gather(*(self.execute(s) for s in steps), return_exceptions=True)
        return await self.worker.run(self._execute_batch, steps)

    def _record_forward(self, size: int) -> None:
        self._forward_calls += 1
        self._batched_forward_calls += size > 1
        self._max_forward_batch_size = max(self._max_forward_batch_size, size)

    def _forward(self, ids: tuple[int, ...] | list[int], state: _Sequence) -> None:
        inputs = self.torch.tensor([ids], dtype=self.torch.long, device='cpu')
        with self.torch.inference_mode():
            self._record_forward(1)
            output = self.model(input_ids=inputs, past_key_values=state.cache,
                                use_cache=True, return_dict=True)
        state.cache = output.past_key_values
        state.next_token = int(output.logits[0, -1].argmax().item())

    def _execute(self, step: ExecutionStep) -> DecodeResult | None:
        try:
            return self._execute_step(step)
        finally:
            self._cache_bytes = self._measure_cache()

    def _execute_step(self, step: ExecutionStep) -> DecodeResult | None:
        request = step.request
        if request.prompt_token_ids is None:
            raise ValueError('local execution requires model-tokenized prompt IDs')
        if step.phase == 'prefill':
            state = self._states.setdefault(request.request_id, _Sequence())
            if step.offset != state.prefilled or step.token_count < 1:
                raise ValueError('invalid local prefill offset')
            chunk = request.prompt_token_ids[step.offset:step.offset + step.token_count]
            if len(chunk) != step.token_count:
                raise ValueError('prefill exceeds encoded prompt')
            self._forward(chunk, state)
            state.prefilled += step.token_count
            return None
        state = self._states[request.request_id]
        if (state.prefilled != request.prompt_token_count or step.token_count != 1
                or step.offset != len(state.generated) or state.next_token is None):
            raise ValueError('invalid local decode step')
        result, token, final = self._decode_token(step, state)
        if not final:
            self._forward([token], state)
        return result

    def _decode_token(self, step: ExecutionStep, state: _Sequence) -> tuple[DecodeResult, int, bool]:
        request = step.request
        token = state.next_token
        state.generated.append(token)
        eos = token in self.eos_ids
        final = eos or len(state.generated) == request.max_tokens
        text = self.tokenizer.decode(state.generated, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        # Byte-level tokens may decode to U+FFFD until subsequent bytes arrive.
        # Keep incomplete suffixes buffered, but flush the canonical final decode.
        stable = text if final else text.rstrip('\ufffd')
        if not stable.startswith(state.emitted_text):
            raise RuntimeError('tokenizer changed already emitted text')
        piece = stable[len(state.emitted_text):]
        state.emitted_text = stable
        return DecodeResult(piece, token, 'eos' if eos else None), token, final

    def _forward_batch(self, chunks: list, states: list[_Sequence]) -> None:
        """Temporary left padding, logical RoPE positions, then owned dense caches.

        Past and new tokens are padded independently. Attention masks exclude
        both kinds of padding; splitting removes the intervening padded gap.
        No row retains a view into another request's batched tensor storage.
        """
        if len(states) == 1:
            self._forward(chunks[0], states[0])
            return
        from transformers import DynamicCache

        torch = self.torch
        lengths = [0 if s.cache is None else s.cache.get_seq_length() for s in states]
        past_width, width = max(lengths), max(map(len, chunks))
        with torch.inference_mode():
            inputs = torch.zeros((len(states), width), dtype=torch.long)
            positions = torch.zeros_like(inputs)
            mask = torch.zeros((len(states), past_width + width), dtype=torch.long)
            for row, (chunk, length) in enumerate(zip(chunks, lengths)):
                count = len(chunk)
                inputs[row, -count:] = torch.tensor(chunk, dtype=torch.long)
                positions[row, -count:] = torch.arange(length, length + count)
                mask[row, past_width - length:past_width] = 1
                mask[row, past_width + width - count:] = 1
            cache = None
            if past_width:
                template = next(s.cache for s in states if s.cache is not None)
                layers = []
                for layer, (keys, values) in enumerate(template):
                    rows = []
                    for state, length in zip(states, lengths):
                        if length:
                            k, v = state.cache[layer]
                            rows.append((torch.nn.functional.pad(k, (0, 0, past_width - length, 0)),
                                         torch.nn.functional.pad(v, (0, 0, past_width - length, 0))))
                        else:
                            shape = (1, keys.shape[1], past_width, keys.shape[3])
                            rows.append((keys.new_zeros(shape), values.new_zeros(shape)))
                    layers.append((torch.cat([r[0] for r in rows], dim=0),
                                   torch.cat([r[1] for r in rows], dim=0)))
                cache = DynamicCache.from_legacy_cache(tuple(layers))
            self._record_forward(len(states))
            output = self.model(input_ids=inputs, attention_mask=mask,
                                position_ids=positions,
                                cache_position=torch.arange(past_width, past_width + width),
                                past_key_values=cache, use_cache=True, return_dict=True)
            tokens = output.logits[:, -1].argmax(dim=-1).tolist()
            # Construct all candidates before committing any of the rows.
            caches = []
            for row, (length, chunk) in enumerate(zip(lengths, chunks)):
                owned = []
                for keys, values in output.past_key_values:
                    def extract(tensor):
                        return torch.cat((tensor[row:row + 1, :, past_width - length:past_width, :],
                                          tensor[row:row + 1, :, past_width + width - len(chunk):, :]),
                                         dim=2).contiguous()
                    owned.append((extract(keys), extract(values)))
                caches.append(DynamicCache.from_legacy_cache(tuple(owned)))
            for state, cache, token in zip(states, caches, tokens):
                state.cache, state.next_token = cache, int(token)

    @staticmethod
    def _batch_error(exc: Exception) -> Exception:
        # Returned exceptions must not keep failed batch tensors alive via frames.
        logger.error('Batched execution failed (%s): %s', type(exc).__name__, str(exc))
        exc.__traceback__ = exc.__context__ = exc.__cause__ = None
        return exc

    def _execute_batch(self, steps: tuple[ExecutionStep, ...]) -> list:
        if len({s.request.request_id for s in steps}) != len(steps):
            raise ValueError('duplicate request in execution batch')
        results = [None] * len(steps)
        groups = {}
        try:
            for index, step in enumerate(steps):
                try:
                    request = step.request
                    if request.prompt_token_ids is None:
                        raise ValueError('local execution requires model-tokenized prompt IDs')
                    if step.phase == 'prefill':
                        state = self._states.setdefault(request.request_id, _Sequence())
                        chunk = request.prompt_token_ids[step.offset:step.offset + step.token_count]
                        if step.offset != state.prefilled or step.token_count < 1 or len(chunk) != step.token_count:
                            raise ValueError('invalid local prefill offset or count')
                    elif step.phase == 'decode':
                        state = self._states[request.request_id]
                        if (state.prefilled != request.prompt_token_count or step.token_count != 1
                                or step.offset != len(state.generated) or state.next_token is None):
                            raise ValueError('invalid local decode step')
                        result, token, final = self._decode_token(step, state)
                        results[index] = result
                        if final:
                            continue  # EOS/output-cap rows need no additional model forward.
                        chunk = [token]
                    else:
                        raise ValueError('invalid local phase')
                    groups.setdefault(step.phase, []).append((index, step, state, chunk))
                except Exception as exc:
                    results[index] = self._batch_error(exc)
            for group in groups.values():
                try:
                    self._forward_batch([r[3] for r in group], [r[2] for r in group])
                    for _, step, state, _ in group:
                        if step.phase == 'prefill':
                            state.prefilled += step.token_count
                except Exception as exc:
                    failure = self._batch_error(exc)
                    for index, _, _, _ in group:
                        results[index] = failure
            return results
        finally:
            self._cache_bytes = self._measure_cache()

    def release_request(self, request_id: str) -> None:
        if self._states.pop(request_id, None) is not None:
            self._cache_bytes = self._measure_cache()

    async def aclose(self) -> None:
        try:
            await self.worker.aclose()
        finally:
            self._states.clear()
            self._cache_bytes = 0
            self.model = None
