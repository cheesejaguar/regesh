"""Offline audit against pinned upstream scoring, without any model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import runpy


def audit(directory, metric_source):
    directory = Path(directory); metric_source = Path(metric_source)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('completed run required')
    for name, expected in manifest['artifact_sha256'].items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('artifact integrity failure')
    config = json.loads((directory / 'config.json').read_text())
    if hashlib.sha256(metric_source.read_bytes()).hexdigest() != config['source_evaluate_sha256']:
        raise ValueError('upstream metric integrity failure')
    original_em = runpy.run_path(str(metric_source))['get_em_score']
    rows = [json.loads(x) for x in (directory / 'rows.jsonl').read_text().splitlines()]
    calls = [json.loads(x) for x in (directory / 'calls.jsonl').read_text().splitlines()]
    items = [json.loads(x) for x in (directory / 'items.jsonl').read_text().splitlines()]
    summary = json.loads((directory / 'summary.json').read_text())
    if len(calls) != config['calls'] or len(rows) != config['calls'] or len(items) != config['items']:
        raise ValueError('wrong attempt count')
    lookup = {item['id']: (i, item) for i, item in enumerate(items)}
    if len(lookup) != len(items):
        raise ValueError('duplicate input id')
    by_id = defaultdict(dict); usage = defaultdict(Counter); hits = Counter()
    for row, call in zip(rows, calls):
        if (row['id'], row['condition']) != (call['id'], call['condition']):
            raise ValueError('call/row order mismatch')
        condition = row['condition']; i, item = lookup[row['id']]
        if condition not in config['conditions'] or condition in by_id[row['id']]:
            raise ValueError('duplicate or unknown condition')
        record = call['record']; request = record['request']
        instruction = item['instruction'] + (' ' + config['suffix'] if condition != 'baseline' else '')
        expected_prompt = [{'role': 'user', 'content': instruction + '\n\nInput: ' + item['input'] + '\nOutput:'}]
        if request['messages'] != expected_prompt or request['seed'] != config['seed'] + i:
            raise ValueError('prompt or seed mismatch')
        if request['max_tokens'] != config['max_tokens'] or request['temperature'] != config['temperature']:
            raise ValueError('generation budget mismatch')
        if row.get('error'):
            prediction = ''; correct = 0; strict = 0
        else:
            response = json.loads(record['raw_response'])
            if response != record['response']:
                raise ValueError('raw/parsed response mismatch')
            prediction = response['choices'][0]['message']['content']
            correct = int(original_em(prediction, item['gold']))
            strict = int(prediction.strip() == item['gold'].strip())
            expected_usage = {k: v for k, v in response.get('usage', {}).items() if isinstance(v, int)}
            if row['usage'] != expected_usage:
                raise ValueError('token accounting mismatch')
            usage[condition].update(expected_usage)
        if row['prediction'] != prediction or row['gold'] != item['gold'] or row['task'] != item['task']:
            raise ValueError('prediction/input mismatch')
        if row['normalized_exact_match'] != correct or row['strict_exact_match'] != strict:
            raise ValueError('score differs from original metric')
        by_id[row['id']][condition] = correct; hits[condition] += correct
    for condition in config['conditions']:
        if summary['conditions'][condition]['normalized_exact_match'] != hits[condition] / config['items']:
            raise ValueError('aggregate mismatch')
        if summary['conditions'][condition]['actual_tokens'] != dict(usage[condition]):
            raise ValueError('aggregate token mismatch')
    if any(set(p) != set(config['conditions']) for p in by_id.values()):
        raise ValueError('incomplete pair')
    deltas = [p['emotionprompt_EP02'] - p['baseline'] for p in by_id.values()]
    loss = deltas.count(-1); gain = deltas.count(1); n = loss + gain
    exact_p = min(1., 2 * sum(math.comb(n, k) for k in range(min(loss, gain) + 1)) / 2**n) if n else 1.
    if summary['effect']['emotion_minus_baseline'] != sum(deltas) / len(deltas) or summary['effect']['exact_mcnemar_p'] != exact_p:
        raise ValueError('paired effect mismatch')
    return {'schema_version': 1, 'run_complete': True, 'audited_calls': len(calls), 'pairs': len(by_id),
        'model_calls': 0, 'network_calls': 0, 'upstream_metric_sha256': config['source_evaluate_sha256'],
        'all_recorded_scores_match_unmodified_original_evaluator': True,
        'all_prompts_seeds_budgets_and_token_totals_match': True,
        'condition_correct_counts': dict(hits), 'exact_mcnemar_p': exact_p,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--metric-source', default='results/papers/emotionprompt/protocol/upstream_evaluate.py.txt')
    p.add_argument('--output', required=True)
    args = p.parse_args(); output = Path(args.output)
    if output.exists():
        raise ValueError('preserve existing evidence; choose a new output')
    result = audit(args.run, args.metric_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
