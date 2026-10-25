#!/usr/bin/env python3
"""Independently count recorded paper results; never call a model or simulator.

This audit deliberately does not import the experiment implementations. It
checks retained file hashes and recomputes headline statistics from raw arrays
and trial records. Chain-of-Emotion's independently checked analysis is recorded
by its separate reviewer because its upstream data remain in an external cache.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import string

import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-10):
        raise ValueError(f'recorded statistic mismatch: {actual!r} vs {expected!r}')


def verify_manifest(directory):
    manifest = load(directory / 'manifest.json')
    mapping = manifest.get('files', manifest.get('artifact_sha256'))
    if not mapping:
        raise ValueError(f'no artifact hashes: {directory}')
    for filename, expected in mapping.items():
        target = directory / filename
        if target.resolve().parent != directory.resolve():
            raise ValueError('manifest must reference direct artifacts')
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f'checksum mismatch: {target}')
    return len(mapping)


def homeostatic(directory):
    summary = load(directory / 'summary.json')
    config = load(directory / 'config.json')
    checks = []
    for condition in summary['cohorts']:
        name = condition['condition']
        files = sorted(directory.glob(f'cohort_{name}_[0-9][0-9].npz'))
        if len(files) != len(config['cohort_seeds']):
            raise ValueError('missing independent cohort batch')
        final, maxima, visits = [], [], 0
        for path in files:
            with np.load(path, allow_pickle=False) as arrays:
                final.extend(arrays['final_states'].tolist())
                maxima.extend(arrays['max_abs'].tolist())
                visits += int(arrays['visits'].sum())
        close(len(final), condition['agents'])
        close(statistics.mean(final), condition['final_mean'])
        close(statistics.pstdev(final), condition['final_sd'])
        close(statistics.mean(abs(value) for value in final), condition['final_mean_absolute'])
        close(max(maxima), condition['maximum_absolute_ever'])
        close(visits, len(final) * config['cohort_steps'])
        checks.append({'condition': name, 'independent_agents': len(final),
                       'executed_actions': visits, 'maximum_absolute_ever': max(maxima),
                       'final_mean_absolute': statistics.mean(abs(v) for v in final)})
    return checks


def happiness(directory):
    summary = load(directory / 'summary.json')
    raw = rows(directory / 'rounds.jsonl.gz')
    groups = defaultdict(list)
    identities = set()
    for record in raw:
        identity = tuple(record[k] for k in ('setting', 'condition', 'seed_block', 'round'))
        if identity in identities:
            raise ValueError('duplicate simulated round')
        identities.add(identity)
        groups[(record['setting'], record['condition'])].append(record)
    checks = []
    for condition in summary['conditions']:
        key = (condition['setting'], condition['condition'])
        records = groups.pop(key)
        values = [r['objective_reward'] for r in records]
        close(len(values), condition['rounds'])
        close(statistics.mean(values), condition['objective_reward_mean'])
        close(statistics.stdev(values), condition['objective_reward_sd'])
        checks.append({'setting': key[0], 'condition': key[1], 'rounds': len(values),
                       'objective_reward_mean': statistics.mean(values),
                       'action_disagreements_with_fixed': sum(
                           r['action_disagreements_with_fixed'] for r in records)})
    if groups:
        raise ValueError('unreported simulated conditions')
    return checks


def upstream_normalize(value):
    # Independent transcription of the archived Apache-2.0 benchmark scorer.
    value = value.replace(' and ', ' ').replace('Sentence 1:', ' ').replace('Sentence 2:', ' ')
    value = value.strip().split('\n')[0]
    value = value.split('.')[0].lower().replace('-', ' ')
    return ''.join(character for character in value if character not in string.punctuation)


def emotionprompt(directory):
    summary = load(directory / 'summary.json')
    config = load(directory / 'config.json')
    raw = rows(directory / 'rows.jsonl')
    calls = rows(directory / 'calls.jsonl')
    if len(raw) != config['calls'] or len(calls) != len(raw):
        raise ValueError('incomplete model run')
    totals = defaultdict(lambda: {'n': 0, 'correct': 0, 'strict': 0})
    identities = set()
    for record, call in zip(raw, calls):
        identity = (record['id'], record['condition'])
        if identity in identities or identity != (call['id'], call['condition']):
            raise ValueError('duplicate or reordered response')
        identities.add(identity)
        response = call['record'].get('response', {})
        if not record.get('error'):
            if record['prediction'] != response['choices'][0]['message']['content']:
                raise ValueError('scored text differs from raw provider response')
        correct = int(not record.get('error') and upstream_normalize(record['prediction']) == upstream_normalize(record['gold']))
        strict = int(not record.get('error') and record['prediction'].strip() == record['gold'].strip())
        close(correct, record['normalized_exact_match'])
        close(strict, record['strict_exact_match'])
        group = totals[record['condition']]
        group['n'] += 1
        group['correct'] += correct
        group['strict'] += strict
    for name, group in totals.items():
        close(group['correct'] / group['n'], summary['conditions'][name]['normalized_exact_match'])
        close(group['strict'] / group['n'], summary['conditions'][name]['strict_exact_match'])
    close(totals['emotionprompt_EP02']['correct'] / totals['emotionprompt_EP02']['n'] -
          totals['baseline']['correct'] / totals['baseline']['n'],
          summary['effect']['emotion_minus_baseline'])
    return dict(totals)


def chain_emotion(directory, cache=None):
    manifest = load(directory / 'manifest.json')
    if hashlib.sha256((directory / 'summary.json').read_bytes()).hexdigest() != manifest['summary_sha256']:
        raise ValueError('Chain-of-Emotion summary checksum mismatch')
    source = {}
    for name, specification in manifest['sources'].items():
        path = Path(cache) / name if cache else Path(manifest['source_paths'][name])
        payload = path.read_bytes()
        if len(payload) != specification['bytes'] or hashlib.sha256(payload).hexdigest() != specification['sha256']:
            raise ValueError('Chain-of-Emotion source checksum mismatch')
        with path.open(encoding='utf-8-sig', newline='') as handle:
            source[name] = list(csv.DictReader(handle))
    summary = load(directory / 'summary.json')
    steu = source['experiment_1_STEU.csv']
    for condition in summary['experiment_1']['conditions']:
        column = {'No Memory': 'No Memory Score', 'Memory': 'Memory Score',
                  'Appraisal Prompts': 'Appraisal Score'}[condition['condition']]
        close(sum(int(r[column]) for r in steu), condition['correct'])
        close(len(steu), condition['n'])
    # Independently check published and reanalysed means, without using the
    # implementation's statistical functions or altering its missing-row rule.
    liwc = source['experiment_2_LIWC_Output.csv']
    metrics = ['WC', 'Authentic', 'Tone', 'Affect', 'emo_pos', 'emo_neg']
    def complete(row):
        try:
            return all(math.isfinite(float(row[key])) for key in metrics)
        except ValueError:
            return False
    liwc = [r for r in liwc if complete(r)]
    close(len(liwc), summary['experiment_2']['complete_metric_rows'])
    condition_codes = {'No Memory': '0', 'Memory': '1', 'Chain of Emotion': '2'}
    for metric in summary['experiment_2']['metrics']:
        for name, group in metric['conditions'].items():
            values = [float(r[metric['metric']]) for r in liwc if r['Condition'] == condition_codes[name]]
            close(len(values), group['n'])
            close(statistics.mean(values), group['mean'])
            close(statistics.stdev(values), group['sd'])
    user = source['experiment_3_user_study.csv']
    close(len(user), summary['experiment_3']['rows'])
    close(len({r['id'] for r in user}), summary['experiment_3']['participants'])
    return {'source_files_verified': len(source), 'steu_items': len(steu),
            'complete_liwc_rows': len(liwc), 'user_rows': len(user),
            'participants': len({r['id'] for r in user}),
            'checks': 'Original score counts and all six LIWC condition means/SD agree; statistical inference separately reviewed.'}


def audit(root, chain_cache=None):
    root = Path(root)
    paths = {'homeostatic': root / 'homeostatic', 'happiness': root / 'happiness',
             'emotionprompt': root / 'emotionprompt/live'}
    return {'schema_version': 1, 'mode': 'record-only; zero simulation/model/network calls',
            'artifact_hashes_verified': {k: verify_manifest(p) for k, p in paths.items()},
            'homeostatic': homeostatic(paths['homeostatic']),
            'happiness': happiness(paths['happiness']),
            'chain_emotion': chain_emotion(root / 'chain_emotion', chain_cache),
            'emotionprompt': emotionprompt(paths['emotionprompt'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/papers')
    parser.add_argument('--output', required=True)
    parser.add_argument('--chain-cache', help='Directory with the three checksum-verified OSF CSVs')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('preserve prior audit: output must not exist')
    result = audit(args.results, args.chain_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('Recorded paper aggregates and artifact hashes agree.')


if __name__ == '__main__':
    main()
