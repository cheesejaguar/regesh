"""Exploratory post-outcome arithmetic content check; primary scores unchanged.

Motivated after observing correct full equations rejected by the original exact
metric. Accepts only a complete integer or an exactly checked binary equation;
does not infer answers from prose and never evaluates generated code.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def arithmetic_content(prediction, item):
    if item['task'] not in {'sum', 'diff'}:
        raise ValueError('arithmetic tasks only')
    a, b = map(int, item['input'].split())
    expected = a + b if item['task'] == 'sum' else a - b
    if expected != int(item['gold']):
        raise ValueError('original arithmetic key mismatch')
    text = prediction.strip()
    if re.fullmatch(r'-?\d+', text):
        return {'correct': int(text) == expected, 'form': 'integer'}
    match = re.fullmatch(r'(-?\d+)\s*([+-])\s*(-?\d+)\s*=\s*(-?\d+)', text)
    if match:
        x, operator, y, z = match.groups()
        correct = (int(x) == a and int(y) == b and operator == ('+' if item['task'] == 'sum' else '-')
            and int(z) == expected)
        return {'correct': correct, 'form': 'complete_binary_equation'}
    return {'correct': False, 'form': 'unparsed_no_prose_or_code_extraction'}


def analyze_content(directory):
    directory = Path(directory)
    items = {r['id']: r for r in map(json.loads, (directory / 'items.jsonl').read_text().splitlines())}
    rows = list(map(json.loads, (directory / 'rows.jsonl').read_text().splitlines()))
    result = {'schema_version': 1, 'analysis_type': 'Exploratory post-outcome diagnostic, not preregistered primary result',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'motivation': 'After outcomes were inspected, correct baseline subtraction equations were observed to fail original exact match.',
        'rule': 'Complete integer, or complete equation with original operands, required operation and arithmetically correct original key. No prose/code extraction. Non-arithmetic rows retain frozen original metric.',
        'primary_metric_unchanged': True, 'model_calls': 0, 'conditions': {}, 'rows': [],
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256': {p: hashlib.sha256((directory / p).read_bytes()).hexdigest() for p in ('items.jsonl', 'rows.jsonl')}}
    by_condition = defaultdict(list)
    for row in rows:
        checked = (arithmetic_content(row['prediction'], items[row['id']]) if row['task'] in {'sum', 'diff'}
            else {'correct': bool(row['normalized_exact_match']), 'form': 'original_non_arithmetic_metric'})
        if row.get('error'): checked['correct'] = False
        detail = {'id': row['id'], 'task': row['task'], 'condition': row['condition'],
            'primary_correct': row['normalized_exact_match'], **checked}
        result['rows'].append(detail); by_condition[row['condition']].append(detail)
    for condition, values in by_condition.items():
        arithmetic = [r for r in values if r['task'] in {'sum', 'diff'}]
        result['conditions'][condition] = {'n': len(values), 'primary_correct': sum(r['primary_correct'] for r in values),
            'exploratory_content_correct': sum(r['correct'] for r in values),
            'exploratory_content_accuracy': sum(r['correct'] for r in values) / len(values),
            'arithmetic_n': len(arithmetic), 'arithmetic_content_correct': sum(r['correct'] for r in arithmetic),
            'correct_equations_rejected_by_primary': sum(r['correct'] and not r['primary_correct'] and r['form'] == 'complete_binary_equation' for r in values)}
    result['interpretation_limit'] = 'Selected after looking at the results. Useful for diagnosing format sensitivity, not confirmatory evidence for an effect. No new significance testing or primary-score replacement.'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); output = Path(args.output)
    if output.exists(): raise ValueError('choose a new output')
    result = analyze_content(args.run)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['conditions'], indent=2))


if __name__ == '__main__':
    main()
