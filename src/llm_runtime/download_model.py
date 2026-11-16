"""Download an explicitly selected, pinned local checkpoint; SmolLM2 is the default."""

import argparse
import json
import os
from pathlib import Path

MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'
MODEL_REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac'
MODELS = {
    'smollm2-135m': (MODEL_ID, MODEL_REVISION, '.models/smollm2'),
    'qwen2.5-0.5b': ('Qwen/Qwen2.5-0.5B-Instruct',
                    '7ae557604adf67be50417f59c2c2f167def9a775', '.models/qwen2.5-0.5b'),
}


def prepare_destination(destination, model_id, revision):
    identity = {'model_id': model_id, 'revision': revision}
    manifest = destination / 'runtime_model.json'
    marker = destination / '.runtime_download.json'
    if manifest.exists() or marker.exists():
        recorded = json.loads((manifest if manifest.exists() else marker).read_text())
        if recorded != identity:
            raise ValueError('destination belongs to another model/revision; choose a separate directory')
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError('destination is nonempty and has no runtime identity; choose a separate directory')
    destination.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    return identity


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, default='smollm2-135m')
    parser.add_argument('--destination', help='separate model directory; defaults depend on --model')
    args = parser.parse_args(argv)
    model_id, revision, default_destination = MODELS[args.model]
    # Plain HTTP downloads keep cache placement predictable in restricted envs.
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.error("install local dependencies first: pip install '.[local]'")
    destination = Path(args.destination or default_destination).expanduser().resolve()
    try:
        identity = prepare_destination(destination, model_id, revision)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    snapshot_download(
        model_id, revision=revision, local_dir=str(destination),
        cache_dir=str(destination / '.download-cache'), token=False,
        allow_patterns=['config.json', 'generation_config.json', 'model.safetensors',
                        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                        'vocab.json', 'merges.txt', 'README.md', 'LICENSE'],
    )
    (destination / 'runtime_model.json').write_text(json.dumps(
        identity, indent=2) + '\n')
    print(f'Downloaded {model_id}@{revision} to {destination}')


if __name__ == '__main__':
    main()
