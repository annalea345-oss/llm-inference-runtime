"""Run browser parser tests with Node when available; no npm dependencies."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_browser_stream_parser():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed only for the optional JavaScript parser tests')
    version = subprocess.run([node, '--version'], capture_output=True, text=True, check=True).stdout.strip()
    if int(version.lstrip('v').split('.')[0]) < 20:
        pytest.skip('Optional JavaScript parser tests require Node 20 or newer')
    script = Path(__file__).with_name('browser_stream.test.mjs')
    result = subprocess.run([node, '--test', str(script)], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
