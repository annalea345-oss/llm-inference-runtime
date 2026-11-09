"""Verify an existing SmolLM2 float32 conversion and running llama.cpp server.

Does not download, build, launch, or install anything. See benchmarks/llama-cpp.md.
Requires the project's local/dev extras and the pinned upstream gguf-py sources.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from llm_runtime.http_benchmark import benchmark, validate_options


FIXTURES = [
    ('single_turn', [{'role': 'user', 'content': 'What is the capital of France? Answer briefly.'}]),
    ('system', [{'role': 'system', 'content': 'Answer in one sentence.'},
                {'role': 'user', 'content': 'What is a token?'}]),
    ('multi_turn', [{'role': 'user', 'content': 'Name a city in Norway.'},
                    {'role': 'assistant', 'content': 'Oslo.'},
                    {'role': 'user', 'content': 'Name another.'}]),
    ('unicode', [{'role': 'user', 'content': 'Explain blåbær, café, and 日本語. 🙂'}]),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_conversion(model_dir, gguf_path, source_dir):
    # Use the GGUF reader from the exact source tree used for conversion.
    sys.path.insert(0, str(source_dir / 'gguf-py'))
    import gguf
    import numpy as np
    from safetensors import safe_open

    config = json.loads((model_dir / 'config.json').read_text())
    if config.get('architectures') != ['LlamaForCausalLM'] or not config.get('tie_word_embeddings'):
        raise ValueError('this verifier is scoped to the tied-embedding SmolLM2 checkpoint')
    reader = gguf.GGUFReader(str(gguf_path))
    tensors = {t.name: t for t in reader.tensors}
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, config['num_hidden_layers'])
    checked = set()
    # The pinned checkpoint has one safetensors file, not an arbitrary model layout.
    with safe_open(model_dir / 'model.safetensors', framework='pt', device='cpu') as weights:
        for name in weights.keys():
            target = names.get_name(name, try_suffixes=('.weight', '.bias'))
            if target is None or target in checked or target not in tensors:
                raise ValueError(f'unmapped, duplicate, or missing weight: {name}')
            expected = weights.get_tensor(name).float().numpy()
            if name.endswith(('q_proj.weight', 'k_proj.weight')):
                heads = config['num_attention_heads'] if name.endswith('q_proj.weight') else config['num_key_value_heads']
                # HF and GGML use different RoPE channel layouts for Q and K.
                expected = expected.reshape(heads, 2, expected.shape[0] // heads // 2,
                                            *expected.shape[1:]).swapaxes(1, 2).reshape(expected.shape)
            tensor = tensors[target]
            if tensor.tensor_type != gguf.GGMLQuantizationType.F32 or not np.array_equal(tensor.data, expected):
                raise ValueError(f'weight precision or value mismatch: {name}')
            checked.add(target)
    if checked != set(tensors):
        raise ValueError('unexpected extra GGUF tensors')

    def scalar(key):
        field = reader.fields[key]
        return int(field.parts[field.data[0]][0])

    for key, value in [('bos_token_id', config['bos_token_id']),
                       ('eos_token_id', config['eos_token_id']),
                       ('padding_token_id', config['pad_token_id'])]:
        if scalar('tokenizer.ggml.' + key) != value:
            raise ValueError(f'special token mismatch: {key}')
    field = reader.fields['tokenizer.chat_template']
    template = bytes(field.parts[field.data[0]]).decode('utf-8')
    tokenizer_config = json.loads((model_dir / 'tokenizer_config.json').read_text())
    if template != tokenizer_config['chat_template']:
        raise ValueError('embedded template differs from checkpoint')
    return {'status': 'passed', 'float32_tensors_verified': len(checked),
            'qk_rope_layout_transform': 'HF rows reshaped by head; pair axis exchanged',
            'special_token_ids_verified': {'bos': config['bos_token_id'], 'eos': config['eos_token_id'],
                                           'pad': config['pad_token_id']},
            'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest()}


async def verify_server(base_url, model_dir):
    import httpx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    fixtures = []
    async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False) as client:
        for name, messages in FIXTURES:
            expected_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            expected_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            response = await client.post(base_url + '/apply-template', json={'messages': messages})
            response.raise_for_status()
            actual_text = response.json()['prompt']
            response = await client.post(base_url + '/tokenize', json={
                'content': actual_text, 'add_special': False, 'parse_special': True})
            response.raise_for_status()
            actual_ids = response.json()['tokens']
            if actual_text != expected_text or actual_ids != expected_ids:
                raise ValueError(f'chat template/token mismatch: {name}')
            fixtures.append({'name': name, 'messages': messages, 'rendered_prompt': actual_text,
                             'token_ids': actual_ids, 'status': 'passed'})
    reports = []
    for concurrency in (1, 2):
        report = await benchmark(base_url, profile='portable', model='local-model',
                                 readiness_path='/health', requests=4, concurrency=concurrency,
                                 warmup=2, max_tokens=24)
        if report['preflight']['status'] != 'passed' or report['completed'] != 4:
            raise ValueError(f'benchmark compatibility failed: {report}')
        if report['usage_verified_requests'] != 4:
            raise ValueError('missing usage in external engine smoke')
        reports.append(report)
    async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False) as client:
        response = await client.get(base_url + '/props')
        response.raise_for_status()
        properties = response.json()
        if properties.get('model_ftype') != 'all F32' or properties.get('total_slots') != 2:
            raise ValueError('expected float32 model and two server slots')
        response = await client.post(base_url + '/v1/chat/completions', json={
            'model': 'local-model', 'messages': FIXTURES[0][1], 'temperature': 0,
            'max_tokens': 24, 'stream': False})
        response.raise_for_status()
        answer = response.json()
        if not answer['choices'][0]['message']['content']:
            raise ValueError('empty non-streaming smoke response')
        response = await client.get(base_url + '/slots')
        response.raise_for_status()
        slots = response.json()
        if len(slots) != 2 or any(s['is_processing'] or s['n_ctx'] != 8192 for s in slots):
            raise ValueError('server slots not idle or unexpected context configuration')
        response = await client.get(base_url + '/metrics')
        response.raise_for_status()
        metrics = response.text
    return {'prompt_fixtures': fixtures, 'http_smoke': reports, 'server_properties': properties,
            'nonstreaming_answer': answer, 'final_slots': slots, 'final_metrics': metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--model-dir', default=Path('.models/smollm2'), type=Path)
    parser.add_argument('--gguf', required=True, type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:8018')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    validate_options(args.base_url, 4, 2, 2, 24, 'verification', 30)
    record = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Pinned llama.cpp CPU compatibility smoke, not comparative performance evidence',
        'project_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'verifier_sha256': sha256(Path(__file__)),
        'platform': platform.platform(), 'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'httpx')},
        'model_files_sha256': {p.name: sha256(p) for p in sorted(args.model_dir.iterdir()) if p.is_file()},
        'gguf_sha256': sha256(args.gguf),
        'conversion': verify_conversion(args.model_dir, args.gguf, args.source_dir),
    }
    record.update(asyncio.run(verify_server(args.base_url.rstrip('/'), args.model_dir)))
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(f"Verified {record['conversion']['float32_tensors_verified']} tensors, "
          f"{len(record['prompt_fixtures'])} prompt fixtures, and both HTTP concurrency settings.")


if __name__ == '__main__':
    main()
