"""Bounded native runtime KV admission/cancellation probe, separate from timing.

Two individually valid requests each reserve over half the logical token capacity.
The second must wait for memory despite an available active slot. No deliberate
host-RAM exhaustion; starts/stops only the local runtime on comparison port 8020.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import httpx

from scripts.compare_engines import BASE, commands, idle, rss_bytes, server, snapshot, wait_state, write_json
from llm_runtime.http_benchmark import sample


async def probe(output):
    output.mkdir(parents=True, exist_ok=False)
    prompt = ' x' * 3700
    payload = {'model': 'local-model', 'messages': [{'role': 'user', 'content': prompt}],
               'max_tokens': 512, 'temperature': 0, 'stream': True,
               'stream_options': {'include_usage': True}}
    result = {'scope': 'Native logical token-reservation pressure; not a physical RAM limit comparison',
              'prompt': prompt, 'max_tokens': 512, 'request_count': 2,
              'checked_at': datetime.now(timezone.utc).isoformat(),
              'project_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).with_name('compare_engines.py').read_bytes()).hexdigest(),
              'server_command': commands()['runtime']}
    async with server('runtime', output, 'admission') as process:
        async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False) as client:
            responses = []
            result['initial_rss_bytes'] = await rss_bytes(process.pid)
            try:
                for _ in range(2):
                    response = await client.send(client.build_request('POST', BASE + '/v1/chat/completions', json=payload), stream=True)
                    response.raise_for_status()
                    responses.append(response)
                observations = await wait_state(client, 'runtime', lambda s: s['active'] == 1 and s['queued'] == 1)
                health = await client.get(BASE + '/health')
                health.raise_for_status()
                result['pressure_health'] = health.json()
                result['pressure_observations'] = observations
                result['pressure_rss_bytes'] = await rss_bytes(process.pid)
                capacity = health.json()['kv_cache']['capacity_tokens']
                reserved = observations[-1]['reserved_tokens']
                if not capacity / 2 < reserved <= capacity:
                    raise ValueError('did not establish expected reservation pressure')
                acted = perf_counter()
                waiting_id = responses[1].headers['x-request-id']
                response = await client.post(BASE + f'/requests/{waiting_id}/cancel')
                if response.status_code != 202:
                    raise ValueError('waiting cancellation not accepted')
                result['waiting_cancel_observations'] = await wait_state(client, 'runtime', lambda s: s['queued'] == 0)
                result['waiting_cancel_observed_ms'] = (perf_counter() - acted) * 1000
                response = await client.post(BASE + f'/requests/{responses[0].headers["x-request-id"]}/cancel')
                if response.status_code != 202:
                    raise ValueError('active cancellation not accepted')
                result['idle_after_cancel'] = await wait_state(client, 'runtime', idle)
            finally:
                for response in responses:
                    await response.aclose()
            result['final_state'] = await snapshot(client, 'runtime')
            if result['final_state']['reserved_tokens'] or result['final_state']['kv_tensor_bytes']:
                raise ValueError('reservations/tensors not released after cancellation')
            result['followup'] = await sample(client, BASE, 'local-model', index=0,
                prompt='What is the capital of France? Answer briefly.', max_tokens=24, timeout=30)
            if result['followup']['status'] != 'completed':
                raise ValueError('follow-up request failed')
            result['final_rss_bytes'] = await rss_bytes(process.pid)
    write_json(output / 'admission-pressure-runtime.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(probe(args.output_dir))
