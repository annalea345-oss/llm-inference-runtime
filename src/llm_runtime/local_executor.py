"""Optional CPU Transformers executor; downloads are a separate explicit step."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_runtime.protocol import DecodeResult, ExecutionStep
from llm_runtime.worker import SerialWorker


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

    def __init__(self, model, tokenizer, torch_module, worker: SerialWorker) -> None:
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
        self._parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    @classmethod
    async def load(cls, model_path: str) -> 'LocalModelExecutor':
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
                attn_implementation='eager',
            ).to('cpu').eval()
            return cls(model, tokenizer, torch, worker)

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
                'kv_tensor_bytes_after_last_step': self.cache_bytes}

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
        ids = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=True,
            add_generation_prompt=True,
        )
        if not ids:
            raise ValueError('model tokenizer returned an empty prompt')
        return tuple(ids)

    async def execute(self, step: ExecutionStep) -> DecodeResult | None:
        return await self.worker.run(self._execute, step)

    def _forward(self, ids: tuple[int, ...] | list[int], state: _Sequence) -> None:
        inputs = self.torch.tensor([ids], dtype=self.torch.long, device='cpu')
        with self.torch.inference_mode():
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
        if not final:
            self._forward([token], state)
        return DecodeResult(piece, token, 'eos' if eos else None)

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
