"""Scoped replication of the published EmotionPrompt EP02 intervention.

The scorer follows Instruction Induction's Apache-2.0 evaluate.py at
f7304ae0a35239c9819a08f93fd58426f6b4e4c1; provenance and its license
accompany the selected, unchanged execution examples. This is not an exact
reproduction of the paper's unavailable raw model generations or aggregate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import string

import numpy as np

from regesh.providers import LocalChatModel, ProviderError


def normalize_prediction(prediction):
    """Original benchmark normalization, including its first-line/period rule."""
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split('\n')[0]
    prediction = prediction.split('.')[0]
    prediction = prediction.lower()
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(str.maketrans('', '', string.punctuation))
    return prediction


def score(prediction, gold):
    if not isinstance(prediction, str) or not isinstance(gold, str):
        raise ValueError('prediction and original key must be text')
    return {
        'normalized_exact_match': int(normalize_prediction(prediction) == normalize_prediction(gold)),
        'strict_exact_match': int(prediction.strip() == gold.strip()),
    }


def messages(item, condition, config):
    if condition not in config['conditions']:
        raise ValueError('unknown condition')
    instruction = item['instruction']
    if condition != 'baseline':
        instruction += ' ' + config['suffix']
    return [{'role': 'user', 'content': f"{instruction}\n\nInput: {item['input']}\nOutput:"}]


def exact_mcnemar(baseline_only, emotion_only):
    """Two-sided exact binomial McNemar test on discordant paired outcomes."""
    if min(baseline_only, emotion_only) < 0:
        raise ValueError('counts must be nonnegative')
    n = baseline_only + emotion_only
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(baseline_only, emotion_only) + 1)) / 2**n) if n else 1.


def analyze(rows, config):
    by_id = defaultdict(dict)
    for row in rows:
        if row['condition'] not in config['conditions'] or row['condition'] in by_id[row['id']]:
            raise ValueError('unknown or duplicate condition')
        by_id[row['id']][row['condition']] = row
    if len(by_id) != config['items'] or len(rows) != config['calls']:
        raise ValueError('incomplete attempted experiment')
    if any(set(pair) != set(config['conditions']) for pair in by_id.values()):
        raise ValueError('incomplete pair')
    pairs = [by_id[k] for k in sorted(by_id)]
    result = {'schema_version': 1, 'pairs': len(pairs), 'calls': len(rows), 'conditions': {}}
    for condition in config['conditions']:
        values = [p[condition] for p in pairs]
        token_counts = Counter()
        for row in values:
            token_counts.update(row.get('usage', {}))
        result['conditions'][condition] = {
            'n': len(values),
            **{metric: sum(r[metric] for r in values) / len(values) for metric in ('normalized_exact_match', 'strict_exact_match')},
            'provider_failures': sum(bool(r.get('error')) for r in values),
            'finish_reasons': dict(Counter(r.get('finish_reason', 'provider_error') for r in values)),
            'actual_tokens': dict(token_counts),
            'elapsed_seconds': sum(r.get('elapsed_seconds', 0.) for r in values),
        }
    deltas = np.array([p['emotionprompt_EP02']['normalized_exact_match'] - p['baseline']['normalized_exact_match'] for p in pairs])
    task_positions = defaultdict(list)
    for i, pair in enumerate(pairs):
        task_positions[pair['baseline']['task']].append(i)
    rng = np.random.default_rng(config['seed'])
    boot = []; task_boot = []
    positions = list(task_positions.values())
    for _ in range(config['bootstrap_samples']):
        # Primary uncertainty retains the declared fixed four-task composition.
        sampled = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in positions])
        boot.append(float(deltas[sampled].mean()))
        chosen = rng.integers(0, len(positions), len(positions))
        task_boot.append(float(np.mean([deltas[positions[i]].mean() for i in chosen])))
    baseline_only = int(np.sum(deltas == -1)); emotion_only = int(np.sum(deltas == 1))
    result['effect'] = {'emotion_minus_baseline': float(deltas.mean()),
        'paired_within_task_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(),
        'task_cluster_bootstrap_95ci': np.quantile(task_boot, [.025, .975]).tolist(),
        'baseline_only_correct': baseline_only, 'emotion_only_correct': emotion_only,
        'both_correct': sum(p['baseline']['normalized_exact_match'] and p['emotionprompt_EP02']['normalized_exact_match'] for p in pairs),
        'exact_mcnemar_p': exact_mcnemar(baseline_only, emotion_only),
        'caution': 'Only four task clusters; intervals do not establish broad task transfer. Model/prompt differ from paper.'}
    result['per_task'] = {}
    for task, ids in task_positions.items():
        result['per_task'][task] = {'pairs': len(ids), **{c: float(np.mean([pairs[i][c]['normalized_exact_match'] for i in ids])) for c in config['conditions']}}
    result['published_reference'] = config['published_reference']
    result['limitations'] = [
        'Single 1.5B quantized coding model instead of published Llama2/ChatGPT/GPT4; one decode per item.',
        'Four tasks, eight fixed examples each, and EP02 only instead of all24 tasks and multiple stimuli.',
        'No released authors generation code/raw responses found; prompt delimiter and benchmark-source normalization are disclosed reconstruction choices.',
        'Emotion condition has extra prompt tokens; generation caps and information are matched, actual token totals differ; no length-matched neutral control.',
        'Static language conditioning; no persistent developing affect or subjective experience is tested.',
        'Possible pretraining exposure; a null small replication does not overturn the published multi-model result.',
        'Elapsed time may include concurrent CPU contention and is descriptive only.',
    ]
    return result


def run(config_path, items_path, output, model):
    config_path = Path(config_path); items_path = Path(items_path); output = Path(output)
    config = json.loads(config_path.read_text())
    payload = items_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != config['items_sha256']:
        raise ValueError('locked item checksum mismatch')
    items = [json.loads(line) for line in payload.decode().splitlines()]
    if len(items) != config['items'] or len({i['id'] for i in items}) != len(items):
        raise ValueError('invalid item count or duplicate id')
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be empty; preserve previous evidence')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_bytes(config_path.read_bytes())
    (output / 'items.jsonl').write_bytes(payload)
    (output / 'runner.py').write_bytes(Path(__file__).read_bytes())
    manifest = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'items_sha256': config['items_sha256'],
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_reference': 'results/runtime/model.json (historical pinned download/server identity, not this run accounting)',
        'model': model.model, 'max_calls': config['calls'], 'max_completion_tokens': config['calls'] * config['max_tokens'],
        'model_calls_completed': 0, 'complete': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    with (output / 'calls.jsonl').open('w') as calls, (output / 'rows.jsonl').open('w') as rowfile:
        for i, item in enumerate(items):
            order = config['conditions'] if i % 2 == 0 else list(reversed(config['conditions']))
            for condition in order:
                prompt = messages(item, condition, config)
                try:
                    generated = model.complete(prompt, seed=config['seed'] + i, max_tokens=config['max_tokens'])
                    text = generated.text; record = generated.record; error = None
                except ProviderError as exc:
                    text = ''; record = exc.record; error = record.get('error_type', 'ProviderError')
                response = record.get('response', {})
                choice = response.get('choices', [{}])[0]
                row = {'id': item['id'], 'task': item['task'], 'condition': condition, 'gold': item['gold'],
                    'prediction': text, **(score(text, item['gold']) if not error else {'normalized_exact_match': 0, 'strict_exact_match': 0}),
                    'error': error, 'finish_reason': choice.get('finish_reason'),
                    'usage': {k: v for k, v in response.get('usage', {}).items() if isinstance(v, int)},
                    'elapsed_seconds': record.get('elapsed_seconds', 0.)}
                calls.write(json.dumps({'id': item['id'], 'condition': condition, 'record': record}, ensure_ascii=False) + '\n'); calls.flush()
                rowfile.write(json.dumps(row, ensure_ascii=False) + '\n'); rowfile.flush(); rows.append(row)
                print(f"{len(rows)}/{config['calls']} {item['id']} {condition}", flush=True)
    summary = analyze(rows, config)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest.update({'finished_utc': datetime.now(timezone.utc).isoformat(), 'model_calls_completed': len(rows), 'complete': True,
        'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.name != 'manifest.json'}})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/reproductions/emotionprompt.json')
    p.add_argument('--items', default='results/papers/emotionprompt/protocol/items.jsonl')
    p.add_argument('--output', required=True)
    p.add_argument('--base-url', required=True); p.add_argument('--model', required=True)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    model = LocalChatModel(args.base_url, args.model, timeout=300, temperature=config['temperature'])
    summary = run(args.config, args.items, args.output, model)
    print(json.dumps(summary['effect'], indent=2))


if __name__ == '__main__':
    main()
