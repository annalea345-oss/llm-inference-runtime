"""Targeted real CPU disconnect regression; run as python -m scripts.verify_disconnect_cleanup.

Uses the prior comparison's queued/active probes without rerunning performance
benchmarks or launching llama.cpp. Requires the existing model/local dependencies.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import httpx

from scripts.compare_engines import (BASE, ROOT, commands, digest, disconnect_active,
                                    disconnect_queued, server, snapshot, write_json)


async def verify(output):
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((ROOT / 'benchmarks/cpu-comparison-v1/manifest.json').read_text())
    for name, expected in prior['model_files_sha256'].items():
        if digest(ROOT / '.models/smollm2' / name) != expected:
            raise ValueError(f'checkpoint changed: {name}')
    workloads = json.loads((ROOT / 'benchmarks/cpu-workloads-v1.json').read_text())
    case = next(c for c in workloads['cases'] if c['id'] == 'decode')
    record = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Targeted CPU disconnect cleanup regression; not a throughput or RSS experiment',
        'project_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'uvicorn', 'starlette')},
        'source_sha256': {p: digest(ROOT / p) for p in (
            'src/llm_runtime/api.py', 'src/llm_runtime/batching.py', 'src/llm_runtime/chat.py',
            'scripts/compare_engines.py', 'scripts/verify_disconnect_cleanup.py')},
        'model_files_sha256': prior['model_files_sha256'],
        'server_command': commands()['runtime'], 'queued_disconnect_trials': [], 'active_disconnect_trials': [],
    }
    async with server('runtime', output, 'disconnect-regression'):
        async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8)) as client:
            response = await client.get(BASE + '/health')
            response.raise_for_status()
            record['initial_health'] = response.json()
            if response.json()['executor']['backend'] != 'transformers':
                raise ValueError('expected the real model backend')
            for index in range(3):
                queued = await disconnect_queued(client, 'runtime', case)
                record['queued_disconnect_trials'].append(queued)
                if not queued['queue_removal_observed'] or queued['blockers_still_active_at_removal'] != 2:
                    raise ValueError(f'queued disconnect did not reclaim its slot with both blockers active: {queued}')
                record['active_disconnect_trials'].append(await disconnect_active(client, 'runtime', case))
                print(f'disconnect trial {index + 1}: queue reclaimed with both blockers active', flush=True)
            explicit = await disconnect_active(client, 'runtime', case, explicit=True)
            terminal = explicit['terminal_body_after_cancel']
            if not any('request_cancelled' in event['data'] for event in terminal):
                raise ValueError('explicit cancellation no longer produced its terminal error')
            record['explicit_cancel'] = explicit
            record['final_state'] = await snapshot(client, 'runtime')
            if any(record['final_state'][k] for k in ('active', 'queued', 'reserved_tokens', 'used_tokens', 'kv_tensor_bytes')):
                raise ValueError('final state still owns requests or tracked memory')
    record['server_stopped'] = True
    write_json(output / 'disconnect-cleanup.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(verify(arguments.output_dir))
