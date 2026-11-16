import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from llm_runtime import download_model as download


@pytest.mark.parametrize('key', list(download.MODELS))
def test_download_selects_pinned_model_and_separate_default_folder(monkeypatch, tmp_path, key):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda *args, **kwargs: calls.append((args, kwargs))))
    download.main(['--model', key])
    model_id, revision, folder = download.MODELS[key]
    assert calls[0][0] == (model_id,)
    assert calls[0][1]['revision'] == revision
    assert calls[0][1]['local_dir'] == str((tmp_path / folder).resolve())
    assert calls[0][1]['token'] is False
    assert 'model.safetensors' in calls[0][1]['allow_patterns']
    assert 'LICENSE' in calls[0][1]['allow_patterns']
    assert json.loads(Path(folder, 'runtime_model.json').read_text()) == {'model_id': model_id, 'revision': revision}


def test_default_download_preserves_original_checkpoint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda *args, **kwargs: calls.append((args, kwargs))))
    download.main([])
    assert calls[0][0] == (download.MODEL_ID,)
    assert calls[0][1]['revision'] == download.MODEL_REVISION
    assert Path('.models/smollm2/runtime_model.json').exists()


@pytest.mark.parametrize('kind', ['manifest', 'marker', 'unknown_contents'])
def test_existing_model_or_unknown_directory_is_not_overwritten(tmp_path, kind):
    if kind == 'manifest':
        (tmp_path / 'runtime_model.json').write_text(json.dumps({'model_id': 'existing', 'revision': 'old'}))
    elif kind == 'marker':
        (tmp_path / '.runtime_download.json').write_text(json.dumps({'model_id': 'existing', 'revision': 'old'}))
    else:
        (tmp_path / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='separate directory'):
        download.prepare_destination(tmp_path, 'new', 'new-revision')


def test_interrupted_download_resumes_only_with_same_identity(monkeypatch, tmp_path):
    destination = tmp_path / 'model'
    def fail(*args, **kwargs):
        raise ConnectionError('download interrupted')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=fail))
    args = ['--model', 'qwen2.5-0.5b', '--destination', str(destination)]
    with pytest.raises(ConnectionError):
        download.main(args)
    assert not (destination / 'runtime_model.json').exists()
    model_id, revision, _ = download.MODELS['qwen2.5-0.5b']
    assert download.prepare_destination(destination, model_id, revision)['model_id'] == model_id
    with pytest.raises(ValueError):
        download.prepare_destination(destination, download.MODEL_ID, download.MODEL_REVISION)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=lambda *args, **kwargs: None))
    download.main(args)
    assert (destination / 'runtime_model.json').exists()
