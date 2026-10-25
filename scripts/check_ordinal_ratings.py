#!/usr/bin/env python3
"""Exploratory ordinal sensitivity for all 12 released Chain-of-Emotion ratings.

This additional analysis was chosen after inspecting the ANOVA results. It is
not a reproduction of an analysis in the paper and adds no new participants.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from scipy.stats import friedmanchisquare


def analyze(source, manifest):
    spec = manifest['sources']['experiment_3_user_study.csv']
    payload = source.read_bytes()
    if len(payload) != spec['bytes'] or hashlib.sha256(payload).hexdigest() != spec['sha256']:
        raise ValueError('source does not match released data')
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = [c for c in reader.fieldnames if c not in {'age', 'gender', 'id', 'group'}]
        rows = list(reader)
    if len(columns) != 12:
        raise ValueError('expected all 12 original rating items')
    grouped = {}
    for row in rows:
        key = (row['id'], row['group'])
        if key in grouped:
            raise ValueError('duplicate participant condition')
        grouped[key] = row
    people = sorted({row['id'] for row in rows})
    conditions = ['No Memory', 'Memory', 'Chain of Emotion']
    if len(people) != 30 or len(rows) != 90 or any((p, c) not in grouped for p in people for c in conditions):
        raise ValueError('expected complete original 30-participant repeated measures')
    records = []
    for column in columns:
        samples = [[float(grouped[(person, condition)][column]) for person in people]
                   for condition in conditions]
        statistic, p = friedmanchisquare(*samples)
        records.append({'item': column, 'friedman_chi_square': float(statistic),
                        'df': 2, 'p': float(p),
                        'kendalls_w': float(statistic) / (len(people) * 2)})
    maximum = 0.
    for rank, index in enumerate(sorted(range(12), key=lambda i: records[i]['p'])):
        maximum = max(maximum, min(1., records[index]['p'] * (12-rank)))
        records[index]['holm_p_12_items'] = maximum
    return {'schema_version': 1, 'source_sha256': spec['sha256'],
            'scope': 'Exploratory post-ANOVA sensitivity; all 12 ordinal rating items, all 30 paired participants; no new model or human observations.',
            'method': 'Tie-corrected Friedman rank test; asymptotic chi-square p values; Holm across 12 items; Kendall W effect size.',
            'limitations': 'Small existing sample, approximate p values, correlated outcomes; not a fresh preregistered confirmation.',
            'participants': len(people), 'rows': len(rows), 'items': records,
            'significant_after_holm': [r['item'] for r in records if r['holm_p_12_items'] < .05]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--manifest', default='results/papers/chain_emotion/manifest.json', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve prior analysis: output must not exist')
    result = analyze(args.source, json.loads(args.manifest.read_text()))
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'surviving_items': result['significant_after_holm']}, indent=2))


if __name__ == '__main__':
    main()
