import json
import subprocess
import sys

import pytest

from llm_runtime.benchmark import benchmark
from llm_runtime.config import RuntimeConfig


@pytest.mark.parametrize('policy', ['round_robin', 'decode_first'])
async def test_mixed_prompt_benchmark_records_identical_workload_and_policy(policy):
    config = RuntimeConfig(scheduling_policy=policy, token_budget=2,
                           prefill_token_budget=1, prefill_delay=0, token_delay=0)
    report = await benchmark('continuous', config, requests=6, concurrency=3,
                             prompt='unused', prompt_pattern=['x', 'x y z'],
                             output_pattern=[3, 1], warmup=2)
    assert report['config']['scheduling_policy'] == policy
    assert report['completed'] == 6
    assert report['errors'] == report['timeouts'] == report['rejected'] == 0
    assert [r['prompt_tokens'] for r in report['requests']] == [1, 3] * 3
    assert [r['generated_tokens'] for r in report['requests']] == [3, 1] * 3


async def test_empty_prompt_pattern_is_rejected():
    with pytest.raises(ValueError, match='invalid workload'):
        await benchmark('continuous', RuntimeConfig(), requests=1, concurrency=1,
                        prompt='x', prompt_pattern=[], output_pattern=[1])


def test_cli_compares_both_policies_and_records_each_config():
    result = subprocess.run([
        sys.executable, '-m', 'llm_runtime.benchmark', '--scheduler', 'policies',
        '--requests', '4', '--concurrency', '2', '--warmup', '0',
        '--prompt-words', '1', '4', '--output-tokens', '2', '1',
        '--prefill-delay', '0', '--token-delay', '0', '--prefill-token-budget', '1',
    ], text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['schema_version'] == 2
    assert [r['config']['scheduling_policy'] for r in report['runs']] == ['round_robin', 'decode_first']
    for run in report['runs']:
        assert run['completed'] == 4
        assert [r['prompt_tokens'] for r in run['requests']] == [1, 4, 1, 4]
        assert [r['generated_tokens'] for r in run['requests']] == [2, 1, 2, 1]
