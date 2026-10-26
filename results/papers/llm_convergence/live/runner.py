"""Scoped real-model replication of "LLMs overconverge to their users" (Blevins 2026).

Blevins, *Accommodation Goes Both Ways* (arXiv:2605.29278, Table 2) reports mean
LIWC function-word convergence .068 for ChatGPT versus .035 for its WildChat users
and .036 for DailyDialog human pairs, and noun convergence .442 versus .141. We
replay the measurement with a small open model replying to DailyDialog contexts:
the asymmetric convergence of Danescu-Niculescu-Mizil & Lee (2011) / Eq. 1 of
Danescu-Niculescu-Mizil et al. (2012), implemented in regesh.bonding.synchrony
with an authored open function-word lexicon (not LIWC; absolute levels differ),
a shuffled-response baseline after French, D'Mello & von der Wense (2024), LSM
(Ireland et al. 2011) and a WordNet-free content-word proxy for the noun scores.

Steps: ``prepare`` selects and locks items before any model call, ``run`` makes
the fixed number of recorded calls, ``analyze`` recomputes the summary from rows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import warnings
import zipfile

import numpy as np

from regesh.bonding.synchrony import (CATEGORIES, COORDINATION_MARKERS, LEXICON, CoordinationCounts,
                                      accommodation_assessment, lsm, tokenize)
from regesh.providers import LocalChatModel, ProviderError

DEFAULT_CONFIG = 'configs/reproductions/llm_convergence.json'
DEFAULT_ITEMS = 'results/papers/llm_convergence/protocol/items.jsonl'
RESPONSE_SETS = ('human', 'plain', 'bounded')
_CONTRACTION = re.compile(r"(\w) ' (s|t|m|d|ll|re|ve|em|clock)\b", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.!?;:])")
_LABEL = re.compile(r"^(?:[AB]|speaker ?[12AB]|second speaker|first speaker|user|assistant|you|me)\s*:\s*", re.IGNORECASE)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- data

def normalize_text(text):
    """Undo DailyDialog's tokenized spacing so the lexicon sees ordinary contractions.

    "Let ' s go , don ' t you ?" -> "Let's go, don't you?". Applied identically to
    contexts, human replies and model replies before any measurement.
    """
    if not isinstance(text, str):
        raise ValueError('text must be a string')
    text = text.replace('’', "'").replace('‘', "'")
    text = _CONTRACTION.sub(lambda m: m.group(1) + "'" + m.group(2), text)
    text = _SPACE_BEFORE_PUNCT.sub(r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def parse_eou_lines(lines):
    """Original release format: one dialogue per line, utterances separated by __eou__."""
    dialogues = []
    for line in lines:
        turns = [t.strip() for t in line.split('__eou__')]
        turns = [t for t in turns if t]
        if turns:
            dialogues.append(turns)
    return dialogues


def load_original_zip(zip_path):
    """Train split from the original nested archive (train.zip -> train/dialogues_train.txt)."""
    with zipfile.ZipFile(zip_path) as outer:
        inner_name = next((n for n in outer.namelist() if n.endswith('train.zip')), None)
        if inner_name is None:
            raise ValueError('train.zip not found inside the DailyDialog archive')
        with outer.open(inner_name) as handle, zipfile.ZipFile(handle) as inner:
            text_name = next((n for n in inner.namelist() if n.endswith('dialogues_train.txt')), None)
            if text_name is None:
                raise ValueError('dialogues_train.txt not found inside train.zip')
            lines = inner.read(text_name).decode('utf-8').splitlines()
    return parse_eou_lines(lines), {'format': 'original nested zip', 'member': f'{inner_name}:{text_name}'}


def load_parlai_json(path):
    """ParlAI mirror format: JSON lines with a "dialogue" list of {"text": ...} turns."""
    dialogues = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            turns = [t['text'].strip() for t in record['dialogue']]
            turns = [t for t in turns if t]
            if turns:
                dialogues.append(turns)
    return dialogues, {'format': 'ParlAI JSON lines mirror', 'member': Path(path).name}


def load_dialogues(cache):
    """Raw train-split dialogues plus provenance (archive sha256) from the local cache."""
    cache = Path(cache)
    original = cache / 'ijcnlp_dailydialog.zip'
    if original.exists() and zipfile.is_zipfile(original):
        dialogues, info = load_original_zip(original)
        return dialogues, {**info, 'archive': original.name, 'archive_sha256': sha256_file(original)}
    mirror = cache / 'parlai' / 'dailydialog.tar.gz'
    train = cache / 'parlai' / 'train.json'
    if train.exists():
        dialogues, info = load_parlai_json(train)
        source = {**info, 'archive': 'parlai/dailydialog.tar.gz', 'train_json_sha256': sha256_file(train)}
        if mirror.exists():
            source['archive_sha256'] = sha256_file(mirror)
        return dialogues, source
    raise FileNotFoundError('no DailyDialog archive in cache; see docs/reproductions/llm_convergence.md')


# ------------------------------------------------------------------------ selection

def select_items(dialogues, config):
    """Deterministic pre-outcome item selection; never inspects any model output.

    Dialogues inside the turn range are normalized, exact duplicates are dropped
    (first occurrence kept), the cut point k comes from SHA256(seed:cut:text), the
    target u1 is turn k and the human reply u2 is turn k+1 (both >= min_tokens),
    and the first ``items`` candidates by SHA256(seed:select:index:text) are kept.
    """
    seed = config['seed']
    low, high = config['turn_range']
    cuts = list(config['cut_points'])
    seen = set()
    candidates = []
    for index, raw_turns in enumerate(dialogues):
        turns = [normalize_text(t) for t in raw_turns]
        turns = [t for t in turns if t]
        if not low <= len(turns) <= high:
            continue
        key = ' __eou__ '.join(turns)
        if key in seen:
            continue
        seen.add(key)
        cut_digest = hashlib.sha256(f'{seed}:cut:{key}'.encode()).hexdigest()
        k = cuts[int(cut_digest, 16) % len(cuts)]
        if k + 1 > len(turns):
            continue
        u1, u2 = turns[k - 1], turns[k]
        if len(tokenize(u1)) < config['min_tokens'] or len(tokenize(u2)) < config['min_tokens']:
            continue
        order = hashlib.sha256(f'{seed}:select:{index}:{key}'.encode()).hexdigest()
        candidates.append((order, {'id': f'dailydialog-train:{index}', 'source_index': index, 'turns': len(turns),
                                   'k': k, 'context': turns[:k], 'u1': u1, 'u2_human': u2, 'order_sha256': order}))
    candidates.sort(key=lambda c: c[0])
    items = [c[1] for c in candidates[:config['items']]]
    if len(items) != config['items']:
        raise ValueError(f'only {len(items)} eligible items for {config["items"]} requested')
    return items


def prepare(cache, config_path, items_path):
    config = json.loads(Path(config_path).read_text())
    dialogues, source = load_dialogues(cache)
    items = select_items(dialogues, config)
    items_path = Path(items_path)
    items_path.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in items)
    items_path.write_text(payload, encoding='utf-8')
    source.update({'split': 'train', 'dialogues': len(dialogues), 'items': len(items),
                   'items_sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest(),
                   'cut_points': dict(sorted(Counter(i['k'] for i in items).items()))})
    (items_path.parent / 'source.json').write_text(json.dumps(source, indent=2) + '\n')
    return source


# -------------------------------------------------------------------------- prompts

def messages(item, condition, config):
    """System prompt plus the context as alternating turns; the target u1 is the final user message.

    Roles are assigned relative to the cut: the speaker of turn k is ``user`` and the
    replying speaker is ``assistant``. For even k the replier opened the dialogue, so
    the first context message carries the assistant role.
    """
    if condition not in config['conditions']:
        raise ValueError('unknown condition')
    system = config['system_prompt']
    if condition == 'bounded':
        system += ' ' + config['bounded_sentence']
    k = item['k']
    if len(item['context']) != k or item['context'][-1] != item['u1']:
        raise ValueError('context must be turns 1..k with u1 as turn k')
    out = [{'role': 'system', 'content': system}]
    for j, turn in enumerate(item['context'], start=1):
        out.append({'role': 'user' if (k - j) % 2 == 0 else 'assistant', 'content': turn})
    return out


def measured_reply(text, error=None):
    """Prespecified cleanup: first non-empty line, leading speaker label and quotes removed, normalized."""
    if error or not isinstance(text, str):
        return ''
    for line in text.splitlines():
        line = _LABEL.sub('', line.strip()).strip().strip('"“”').strip()
        if line:
            return normalize_text(line)
    return ''


# ------------------------------------------------------------------------- analysis

def feature_flags(texts, feature_sets):
    """Boolean matrix (features, texts): whether each text exhibits each feature set."""
    token_sets = [set(tokenize(t)) for t in texts]
    return np.array([[not words.isdisjoint(tokens) for tokens in token_sets] for words in feature_sets], dtype=bool)


def convergence_from_flags(a, b):
    """Eq. 1 per feature over the last axis; NaN where the target never exhibits the feature.

    Identical to CoordinationCounts.coordination for (features, n) inputs, and applies
    to stacked resamples of shape (features, resamples, n).
    """
    n = a.shape[-1]
    a_sum = a.sum(-1)
    b_sum = b.sum(-1)
    ab_sum = (a & b).sum(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        conv = ab_sum / a_sum - b_sum / n
    return np.where(a_sum > 0, conv, np.nan)


def aggregated_3(conv):
    """Mean over defined features (D-N-M et al. 2012 aggregation 3), along axis 0."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(conv, axis=0)


def resample_aggregates(a, b_sets, index_matrix, chunk=1000):
    """aggregated_3 for every row of index_matrix (R, n) and every response set."""
    out = {name: [] for name in b_sets}
    for start in range(0, len(index_matrix), chunk):
        idx = index_matrix[start:start + chunk]
        a_s = a[:, idx]
        for name, b in b_sets.items():
            out[name].append(aggregated_3(convergence_from_flags(a_s, b[:, idx])))
    return {name: np.concatenate(parts) for name, parts in out.items()}


def content_vocabulary(texts, size, min_length):
    """Most frequent non-function tokens of the target utterances; ties broken alphabetically."""
    function_words = frozenset().union(*LEXICON.values())
    counts = Counter(t for text in texts for t in tokenize(text) if len(t) >= min_length and t not in function_words)
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:size]]


def _interval(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {'mean': None, 'ci95': [None, None], 'defined': 0}
    return {'mean': float(finite.mean()), 'ci95': np.quantile(finite, [.025, .975]).tolist(), 'defined': int(len(finite))}


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
    pairs = sorted(by_id.values(), key=lambda p: p['plain']['index'])
    n = len(pairs)
    u1 = [normalize_text(p['plain']['u1']) for p in pairs]
    replies = {'human': [normalize_text(p['plain']['u2_human']) for p in pairs],
               'plain': [measured_reply(p['plain']['reply'], p['plain'].get('error')) for p in pairs],
               'bounded': [measured_reply(p['bounded']['reply'], p['bounded'].get('error')) for p in pairs]}
    markers = tuple(config['markers'])
    all_categories = tuple(CATEGORIES)
    vocabulary = content_vocabulary(u1, config['content_proxy']['vocabulary_size'], config['content_proxy']['min_word_length'])
    feature_sets = {'markers': [LEXICON[m] for m in markers], 'all_categories': [LEXICON[c] for c in all_categories],
                    'content': [frozenset([w]) for w in vocabulary]}
    a = {name: feature_flags(u1, fs) for name, fs in feature_sets.items()}
    b = {name: {s: feature_flags(replies[s], fs) for s in RESPONSE_SETS} for name, fs in feature_sets.items()}

    result = {'schema_version': 1, 'items': n, 'calls': len(rows), 'markers': list(markers),
              'lexicon': 'regesh.bonding.synchrony.LEXICON (authored open list; not LIWC)',
              'content_vocabulary': vocabulary, 'response_sets': {}}
    for s in RESPONSE_SETS:
        counts = CoordinationCounts(markers)
        counts_all = CoordinationCounts(all_categories)
        for target, reply in zip(u1, replies[s]):
            counts.add(target, reply)
            counts_all.add(target, reply)
        summary = counts.summary()
        point = aggregated_3(convergence_from_flags(a['markers'], b['markers'][s]))
        if summary['aggregated_3'] is None or not np.isclose(point, summary['aggregated_3']):
            raise RuntimeError('vectorized convergence disagrees with CoordinationCounts')
        content_conv = convergence_from_flags(a['content'], b['content'][s])
        non_empty = [i for i, r in enumerate(replies[s]) if tokenize(r)]
        lengths = [len(tokenize(r)) for r in replies[s]]
        entry = {'n': n, 'coordination': summary, 'coordination_all_categories': counts_all.summary(),
                 'accommodation_assessment': accommodation_assessment(summary['aggregated_3']),
                 'content_proxy_mean': float(aggregated_3(content_conv)),
                 'content_proxy_defined_words': int(np.isfinite(content_conv).sum()),
                 'lsm_mean_all_items': float(np.mean([lsm(t, r) for t, r in zip(u1, replies[s])])),
                 'lsm_mean_non_empty': float(np.mean([lsm(u1[i], replies[s][i]) for i in non_empty])) if non_empty else None,
                 'reply_tokens_mean': float(np.mean(lengths)), 'reply_tokens_median': float(np.median(lengths)),
                 'empty_replies': n - len(non_empty), 'empty_fraction': (n - len(non_empty)) / n}
        if s != 'human':
            values = [p[s] for p in pairs]
            token_counts = Counter()
            for row in values:
                token_counts.update(row.get('usage', {}))
            entry.update({'provider_failures': sum(bool(r.get('error')) for r in values),
                          'finish_reasons': dict(Counter(r.get('finish_reason') or 'provider_error' for r in values)),
                          'actual_tokens': dict(token_counts),
                          'completion_tokens_mean': token_counts['completion_tokens'] / n if token_counts else None,
                          'elapsed_seconds': sum(r.get('elapsed_seconds', 0.) for r in values),
                          'multi_line_outputs': sum(1 for r in values if isinstance(r.get('reply'), str) and len(r['reply'].strip().splitlines()) > 1)})
        result['response_sets'][s] = entry

    # Shuffled-pair baseline (French et al. 2024): replies re-paired with other items' u1.
    rng = np.random.default_rng(config['shuffle_seed'])
    perms = np.array([rng.permutation(n) for _ in range(config['shuffle_permutations'])])
    result['shuffled_baseline'] = {'permutations': int(len(perms)), 'seed': config['shuffle_seed'],
                                   'markers': {}, 'content_proxy': {}}
    for name, key in (('markers', 'markers'), ('content_proxy', 'content')):
        shuffled = resample_aggregates(a[key], b[key], perms)
        for s in RESPONSE_SETS:
            true_value = float(aggregated_3(convergence_from_flags(a[key], b[key][s])))
            band = _interval(shuffled[s])
            result['shuffled_baseline'][name][s] = {'true': true_value, 'shuffled_mean': band['mean'],
                                                    'shuffled_95_band': band['ci95'],
                                                    'true_minus_shuffled_mean': true_value - band['mean'] if band['mean'] is not None else None,
                                                    'true_above_band': bool(band['ci95'][1] is not None and true_value > band['ci95'][1])}

    # Paired item bootstrap: the same resampled items for all three response sets.
    rng = np.random.default_rng(config['bootstrap_seed'])
    idx = rng.integers(0, n, (config['bootstrap_samples'], n))
    result['bootstrap'] = {'resamples': int(config['bootstrap_samples']), 'seed': config['bootstrap_seed'], 'markers': {}, 'content_proxy': {}}
    contrasts = (('plain', 'human'), ('bounded', 'human'), ('plain', 'bounded'))
    for name, key in (('markers', 'markers'), ('content_proxy', 'content')):
        boot = resample_aggregates(a[key], b[key], idx)
        block = {s: _interval(boot[s]) for s in RESPONSE_SETS}
        for x, y in contrasts:
            point = float(aggregated_3(convergence_from_flags(a[key], b[key][x])) - aggregated_3(convergence_from_flags(a[key], b[key][y])))
            block[f'{x}_minus_{y}'] = {'point': point, **_interval(boot[x] - boot[y])}
        result['bootstrap'][name] = block

    ref = config['published_reference']
    marker_means = {k: float(np.mean([ref['per_marker'][k][m] for m in markers])) for k in ref['per_marker']}
    gap = result['bootstrap']['markers']['plain_minus_human']
    result['comparison'] = {
        'published_liwc_mean_llm_minus_user': ref['liwc_mean']['wildchat_llm'] - ref['liwc_mean']['wildchat_user'],
        'published_eight_marker_means_derived': marker_means,
        'published_eight_marker_llm_minus_user_derived': marker_means['wildchat_llm'] - marker_means['wildchat_user'],
        'measured_plain_minus_human': gap['point'], 'measured_plain_minus_human_ci95': gap['ci95'],
        'same_direction_as_published': bool(gap['point'] > 0),
        'ci_excludes_zero': bool(gap['ci95'][0] is not None and (gap['ci95'][0] > 0 or gap['ci95'][1] < 0)),
        'published_noun_llm_vs_user': [ref['noun_mean']['wildchat_llm'], ref['noun_mean']['wildchat_user']],
        'measured_content_proxy_plain_vs_human': [result['response_sets']['plain']['content_proxy_mean'],
                                                  result['response_sets']['human']['content_proxy_mean']],
        'bounded_minus_plain_markers': result['bootstrap']['markers']['plain_minus_bounded']['point'] * -1,
        'bounded_empty_fraction': result['response_sets']['bounded']['empty_fraction'],
        'warning': 'Different model (1.5B coder vs ChatGPT), corpus (DailyDialog contexts vs WildChat), lexicon (open list vs LIWC) and noun proxy; direction and contrasts are comparable, absolute levels are not.'}
    result['published_reference'] = ref
    result['limitations'] = [
        'Qwen2.5-Coder-1.5B-Instruct Q4_K_M on CPU instead of ChatGPT (GPT-3.5/4) in WildChat; one greedy decode per item.',
        'Authored open function-word lexicon, not the LIWC 2007 dictionary; per-category absolute levels differ.',
        'Human replies are the real next DailyDialog turn; model replies answer the same u1 after a 2-4 turn context, not organic multi-turn WildChat use.',
        'Content-word proxy uses the top-100 frequent non-function tokens of u1 texts without POS tagging or WordNet filtering, so it is not Blevins noun-lemma convergence.',
        'Only reply length, finish reason and empty-reply counts measure output quality; coherence and appropriateness are unmeasured.',
        'Original data host is a parked domain; the ParlAI mirror of the train split was used and hashed (see config).',
        'One aggregated speaker per response set; no per-speaker macro average and no per-turn trajectory.',
    ]
    return result


# ------------------------------------------------------------------------------ run

def _complete_lines(path):
    """Parsed JSON lines; a truncated trailing line from an interrupted write is dropped."""
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            break
    return records


def _resume_state(output, config_bytes, payload):
    """Validate an interrupted run directory and return its completed rows and call records."""
    if (output / 'config.json').read_bytes() != config_bytes or (output / 'items.jsonl').read_bytes() != payload:
        raise ValueError('resume requires the identical frozen config and items')
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest.get('complete'):
        raise ValueError('run already complete; preserve previous evidence')
    rows = _complete_lines(output / 'rows.jsonl')
    done = {(r['id'], r['condition']) for r in rows}
    records = [c for c in _complete_lines(output / 'calls.jsonl') if (c['id'], c['condition']) in done]
    if len(records) != len(rows):
        raise ValueError('calls.jsonl and rows.jsonl disagree; cannot resume')
    (output / 'calls.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in records), encoding='utf-8')
    (output / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    manifest.setdefault('resumed_utc', []).append(datetime.now(timezone.utc).isoformat())
    return manifest, rows


def run(config_path, items_path, output, model, resume=False):
    config_path = Path(config_path); items_path = Path(items_path); output = Path(output)
    config = json.loads(config_path.read_text())
    payload = items_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != config['items_sha256']:
        raise ValueError('locked item checksum mismatch')
    items = [json.loads(line) for line in payload.decode('utf-8').splitlines()]
    if len(items) != config['items'] or len({i['id'] for i in items}) != len(items):
        raise ValueError('invalid item count or duplicate id')
    if config['calls'] != config['items'] * len(config['conditions']):
        raise ValueError('call ceiling must equal items times conditions')
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError('output must be empty; preserve previous evidence')
        manifest, rows = _resume_state(output, config_path.read_bytes(), payload)
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'config.json').write_bytes(config_path.read_bytes())
        (output / 'items.jsonl').write_bytes(payload)
        (output / 'runner.py').write_bytes(Path(__file__).read_bytes())
        manifest = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
                    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    'items_sha256': config['items_sha256'],
                    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'model_reference': 'results/runtime/model.json (historical pinned download/server identity, not this run accounting)',
                    'model': model.model, 'max_calls': config['calls'],
                    'max_completion_tokens': config['calls'] * config['max_tokens'],
                    'model_calls_completed': 0, 'complete': False}
        rows = []
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    done = {(r['id'], r['condition']) for r in rows}
    with (output / 'calls.jsonl').open('a') as calls, (output / 'rows.jsonl').open('a') as rowfile:
        for i, item in enumerate(items):
            order = config['conditions'] if i % 2 == 0 else list(reversed(config['conditions']))
            for condition in order:
                if (item['id'], condition) in done:
                    continue
                prompt = messages(item, condition, config)
                try:
                    generated = model.complete(prompt, seed=config['seed'] + i, max_tokens=config['max_tokens'])
                    text = generated.text; record = generated.record; error = None
                except ProviderError as exc:
                    text = ''; record = exc.record; error = record.get('error_type', 'ProviderError')
                response = record.get('response', {}) or {}
                choice = (response.get('choices') or [{}])[0]
                row = {'id': item['id'], 'index': i, 'k': item['k'], 'condition': condition, 'u1': item['u1'],
                       'u2_human': item['u2_human'], 'reply': text, 'reply_measured': measured_reply(text, error),
                       'error': error, 'finish_reason': choice.get('finish_reason'),
                       'usage': {k: v for k, v in (response.get('usage') or {}).items() if isinstance(v, int)},
                       'elapsed_seconds': record.get('elapsed_seconds', 0.)}
                calls.write(json.dumps({'id': item['id'], 'condition': condition, 'record': record}, ensure_ascii=False) + '\n'); calls.flush()
                rowfile.write(json.dumps(row, ensure_ascii=False) + '\n'); rowfile.flush(); rows.append(row)
                print(f"{len(rows)}/{config['calls']} {item['id']} k={item['k']} {condition} "
                      f"{row['finish_reason'] or error} {row['elapsed_seconds']:.1f}s", flush=True)
    summary = analyze(rows, config)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest.update({'finished_utc': datetime.now(timezone.utc).isoformat(), 'model_calls_completed': len(rows), 'complete': True,
                     'artifact_sha256': {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.name != 'manifest.json'}})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return summary


def analyze_run(run_dir, output=None):
    """Recompute the summary from recorded rows without any model call."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    rows = [json.loads(line) for line in (run_dir / 'rows.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    summary = analyze(rows, config)
    if output:
        Path(output).write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='select and write locked items from the cached DailyDialog train split')
    prep.add_argument('--cache', required=True)
    prep.add_argument('--config', default=DEFAULT_CONFIG)
    prep.add_argument('--items', default=DEFAULT_ITEMS)
    live = sub.add_parser('run', help='make the fixed number of recorded model calls')
    live.add_argument('--config', default=DEFAULT_CONFIG)
    live.add_argument('--items', default=DEFAULT_ITEMS)
    live.add_argument('--output', required=True, help='new or empty directory')
    live.add_argument('--base-url', required=True); live.add_argument('--model', required=True)
    live.add_argument('--timeout', type=float, default=300)
    live.add_argument('--resume', action='store_true', help='continue an interrupted run with identical config/items')
    ana = sub.add_parser('analyze', help='recompute summary.json from a recorded run')
    ana.add_argument('--run', required=True); ana.add_argument('--output')
    args = p.parse_args(argv)
    if args.command == 'prepare':
        print(json.dumps(prepare(args.cache, args.config, args.items), indent=2))
    elif args.command == 'run':
        config = json.loads(Path(args.config).read_text())
        model = LocalChatModel(args.base_url, args.model, timeout=args.timeout, temperature=config['temperature'])
        summary = run(args.config, args.items, args.output, model, resume=args.resume)
        print(json.dumps({'bootstrap_markers': summary['bootstrap']['markers'], 'comparison': summary['comparison']}, indent=2))
    else:
        summary = analyze_run(args.run, args.output)
        print(json.dumps({'bootstrap_markers': summary['bootstrap']['markers'], 'comparison': summary['comparison']}, indent=2))


if __name__ == '__main__':
    main()
