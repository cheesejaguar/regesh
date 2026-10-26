"""Record-only audit of the pair-bonding results: verifies manifest hashes and recomputes
platform summary means from raw run records with the standard library only.

Run: python scripts/audit_bonding.py --output results/bonding/audit.json
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(directory):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if not manifest_path.exists():
        return {'directory': str(directory), 'status': 'no manifest'}
    manifest = json.loads(manifest_path.read_text())
    hashes = None
    for key in ('files', 'artifact_sha256', 'file_sha256', 'outputs', 'artifacts'):
        if isinstance(manifest.get(key), dict):
            hashes = manifest[key]; break
    if hashes is None:
        return {'directory': str(directory), 'status': 'manifest without recognised hash map'}
    mismatches = []; checked = 0
    for name, expected in hashes.items():
        if not isinstance(expected, str) or len(expected) != 64:
            continue
        target = directory / name
        if not target.exists():
            mismatches.append({'file': name, 'problem': 'missing'}); continue
        checked += 1
        if sha256(target) != expected:
            mismatches.append({'file': name, 'problem': 'hash mismatch'})
    return {'directory': str(directory), 'status': 'ok' if not mismatches else 'mismatch', 'checked': checked, 'mismatches': mismatches}


def recompute_platform(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    report = {'attachment': {}, 'dyad': {}, 'max_abs_difference': 0.}
    with gzip.open(directory / 'attachment_runs.jsonl.gz', 'rt') as f:
        att = [json.loads(line) for line in f]
    for key, block in summary['attachment']['conditions'].items():
        q, kind = key.split('|'); q = float(q[2:]); bonded = kind != 'reference'
        variant = kind.split(':', 1)[1] if ':' in kind else None
        rows = [r for r in att if r['responsiveness'] == q and r['bonded'] == bonded and (not bonded or r.get('variant') == variant)]
        for endpoint, stats in block.items():
            if stats is None:
                continue
            values = [r[endpoint] for r in rows if r.get(endpoint) is not None]
            mean = statistics.fmean(values)
            diff = abs(mean - stats['mean'])
            report['attachment'][f'{key}:{endpoint}'] = {'recomputed': mean, 'summary': stats['mean'], 'n': len(values)}
            report['max_abs_difference'] = max(report['max_abs_difference'], diff)
    with gzip.open(directory / 'dyad_runs.jsonl.gz', 'rt') as f:
        dy = [json.loads(line) for line in f]
    for key, block in summary['dyad']['conditions'].items():
        partner, policy = key.split('|')
        rows = [r for r in dy if r['partner'] == partner and r['policy'] == policy]
        for endpoint, stats in block.items():
            if stats is None:
                continue
            values = [r[endpoint] for r in rows if r.get(endpoint) is not None]
            mean = statistics.fmean(values)
            report['dyad'][f'{key}:{endpoint}'] = {'recomputed': mean, 'summary': stats['mean'], 'n': len(values)}
            report['max_abs_difference'] = max(report['max_abs_difference'], abs(mean - stats['mean']))
    report['agree_within_1e-9'] = report['max_abs_difference'] < 1e-9
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', nargs='*', default=['results/bonding/platform', 'results/bonding/platform_v2'])
    parser.add_argument('--papers', nargs='*', default=['results/papers/attachment_robot', 'results/papers/marital_dynamics',
                                                        'results/papers/coordination_power', 'results/papers/llm_convergence/live'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = {'schema_version': 1, 'manifests': [verify_manifest(d) for d in [*args.platform, *args.papers]]}
    result['platform_recomputation'] = {d: recompute_platform(d) for d in args.platform if Path(d, 'summary.json').exists()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({m['directory']: m['status'] for m in result['manifests']}, indent=2))
    for d, rec in result['platform_recomputation'].items():
        print(d, 'means agree:', rec['agree_within_1e-9'])


if __name__ == '__main__':
    main()
