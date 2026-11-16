"""Optional report rendering; requires matplotlib in the rendering interpreter."""

import argparse
import json
from pathlib import Path


def plot(directory, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('cannot plot an incomplete memory experiment')
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    for label in manifest['sessions']:
        session = json.loads((directory / f'{label}.json').read_text())
        waves = session['waves']
        x = [w['wave'] for w in waves]
        axes[0].plot(x, [w['rss_bytes'] / 2**20 for w in waves],
                     '--' if session['diagnostic'] else '-', label=label)
        if session['diagnostic']:
            axes[1].plot(x, [w['diagnostic']['native_malloc']['size_in_use'] / 2**20 for w in waves], color='C2')
            axes[2].plot(x, [w['diagnostic']['python_traced_current_bytes'] / 2**20 for w in waves], color='C2')
    axes[0].set_title('Idle RSS after repeated four-request waves')
    axes[0].set_ylabel('RSS (MiB)')
    axes[0].legend()
    axes[1].set_title('Diagnostic process: native malloc in-use counter (not RSS)')
    axes[1].set_ylabel('MiB')
    axes[2].set_title('Diagnostic process: Python allocations traced after model load')
    axes[2].set_ylabel('MiB (excludes most native tensor storage)')
    for axis in axes:
        axis.set_xlabel('Wave (4 completed requests, concurrency 2)')
        axis.grid(alpha=0.25)
    fig.suptitle('Retained-memory profiling — bounded CPU experiment')
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.directory, args.output)
