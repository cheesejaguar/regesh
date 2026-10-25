#!/usr/bin/env python3
"""Create a standalone four-panel comparison from completed recorded studies."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/papers')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.results), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('preserve previous figures: output must be empty')
    sources = {}
    def read(relative):
        path = root / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    home = read('homeostatic/summary.json')
    happy = read('happiness/summary.json')
    chain = read('chain_emotion/summary.json')
    emotion = read('emotionprompt/live/summary.json')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'regesh-papers-v1'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    blue, orange, gray = '#22639d', '#bc6419', '#737a80'
    ax = axes[0, 0]
    labels = {'published_inverse_beta': 'Learning (assumed inverse β)',
              'no_learning': 'No learning',
              'beta_temperature_sensitivity': 'Learning (β as temperature)'}
    for row, color in zip(home['cohorts'], [blue, gray, orange]):
        path = root / f'homeostatic/cohort_{row["condition"]}_moments.npz'
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as archive:
            values = archive['moments']
        ax.plot(values[:, 0], values[:, 3], color=color, label=labels[row['condition']])
    ax.set(title='A. Homeostatic regulation: 100,000 agents per condition',
           xlabel='Actions per agent', ylabel='Mean absolute deviation from setpoint')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    offsets = {'fixed_compare': -.15, 'dynamic_compare': .15}
    settings = ['stationary_spread', 'stationary_close', 'sudden_change', 'constant_change']
    named = {(r['setting'], r['condition']): r for r in happy['conditions']}
    for condition, color in [('fixed_compare', blue), ('dynamic_compare', orange)]:
        for i, setting in enumerate(settings):
            key = (setting, condition)
            if key not in named:
                continue
            contrast = next(r for r in happy['contrasts'] if r['setting'] == setting and
                            r['condition_a'] == condition and r['condition_b'] == 'objective')
            mean = contrast['paired_difference_mean'] / 5000
            ci = np.array(contrast['paired_seed_block_ci95']) / 5000
            x = i + offsets[condition]
            ax.errorbar(x, mean, yerr=np.array([[mean-ci[0]], [ci[1]-mean]]),
                        fmt='o', color=color, capsize=3,
                        label=condition.replace('_', ' ') if i == 0 else None)
            current, baseline = named[key], named[(setting, 'objective')]
            if 'published_mean' in current and 'published_mean' in baseline:
                delta = (current['published_mean'] - baseline['published_mean']) / 5000
                ax.scatter(x, delta, marker='x', color=color, s=65,
                           label='Published contrast (×)' if condition == 'fixed_compare' and i == 0 else None)
    ax.axhline(0, color=gray, linewidth=.8)
    ax.set_xticks(range(4), ['Stationary\nspread', 'Stationary\nclose', 'Sudden\nchange', 'Constant\nchange'])
    ax.set(title='B. Comparison rewards versus ordinary objective rewards',
           ylabel='Cumulative reward difference / 5,000 actions')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    data = chain['experiment_1']['conditions']
    means = np.array([r['accuracy'] for r in data])
    lower, upper = np.array([r['wilson_95_ci'] for r in data]).T
    ax.bar(range(3), means, color=[gray, blue, orange], width=.6)
    ax.errorbar(range(3), means, yerr=np.array([means-lower, upper-means]),
                fmt='none', color='#202020', capsize=4)
    ax.scatter(range(3), [r['published_correct']/42 for r in data], marker='x',
               color='#202020', zorder=4, label='Published counts (×)')
    ax.set_xticks(range(3), ['No memory', 'Memory', 'Appraisal'])
    ax.set(title='C. Chain-of-Emotion: original 42 author-scored items',
           ylabel='Accuracy; Wilson 95% intervals', ylim=(0, 1))
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    task_names = list(emotion['per_task'])
    x = np.arange(len(task_names))
    for condition, offset, color in [('baseline', -.18, blue), ('emotionprompt_EP02', .18, orange)]:
        ax.bar(x+offset, [emotion['per_task'][task][condition] for task in task_names],
               width=.34, color=color, label=condition.replace('_', ' '))
    ax.set_xticks(x, [name.replace('_word_letter', ' letter') for name in task_names])
    ax.set(title='D. EmotionPrompt: new Qwen pilot, 8 pairs per task',
           ylabel='Normalized exact-match accuracy', ylim=(0, 1.05))
    ax.legend(fontsize=8)
    fig.suptitle('Published experiments and scoped replications — distinct endpoints', fontsize=15)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'paper_comparisons.png', dpi=180)
    svg = output / 'paper_comparisons.svg'
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    (output / 'manifest.json').write_text(json.dumps({'source_sha256': sources,
        'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'note': 'No pooled score; panel B conditions also differ in published exploration/learning settings. Panel C reuses original observations; panel D changes model and task scope.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
