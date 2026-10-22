"""Download only the pinned SmolLM2 checkpoint needed by the local backend."""

import argparse
import json
import os
from pathlib import Path

MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'
MODEL_REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default='.models/smollm2')
    args = parser.parse_args()
    # Plain HTTP downloads keep cache placement predictable in restricted envs.
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.error("install local dependencies first: pip install '.[local]'")
    destination = Path(args.destination).expanduser().resolve()
    snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_dir=str(destination),
        cache_dir=str(destination / '.download-cache'), token=False,
        allow_patterns=['config.json', 'generation_config.json', 'model.safetensors',
                        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                        'vocab.json', 'merges.txt', 'README.md'],
    )
    (destination / 'runtime_model.json').write_text(json.dumps(
        {'model_id': MODEL_ID, 'revision': MODEL_REVISION}, indent=2) + '\n')
    print(f'Downloaded {MODEL_ID}@{MODEL_REVISION} to {destination}')


if __name__ == '__main__':
    main()
