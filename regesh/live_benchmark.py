"""Bounded same-model numeric design generation with executable adjudication.

This runner makes live calls only when explicitly supplied a model adapter. It
measures a six-integer toy design grammar, not open-ended creative engineering.
Use --help; there is no endpoint discovery or automatic provider selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np

from .benchmark import ACTIONS, DesignTask, FAMILIES, Policy, REFERENCE, SEMANTICS, SearchEnvironment
from .learning import CONTEXT_NAMES, DirectValuePredictor, RidgePredictor
from .providers import LocalChatModel, ModelResult, ProviderError

CONDITIONS = ('memory_planning', 'static_persona', 'turn_sentiment', 'heuristic',
              'hybrid', 'ordinary_adaptive', 'unlabelled_hybrid')
TEST_SEEDS = (800, 801, 802)
_INVALID = (-1,) * 6
CONTROLS = {
    'diverge': {'exploration': .9, 'verification': .25, 'planning_horizon': 3},
    'refine': {'exploration': .25, 'verification': .6, 'planning_horizon': 6},
    'combine': {'exploration': .55, 'verification': .45, 'planning_horizon': 5},
    'verify': {'exploration': .1, 'verification': .95, 'planning_horizon': 8},
}


def parse_design(text):
    """Strict one-object schema: prose, booleans, floats, and extra keys fail."""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'design'}:
        raise ValueError('Expected exactly one design field')
    design = value['design']
    if not isinstance(design, list) or len(design) != 6:
        raise ValueError('Expected six integers')
    if any(type(x) is not int or not 0 <= x <= 3 for x in design):
        raise ValueError('Design levels must be integers in [0,3]')
    return tuple(design)


class RecordedCallsModel:
    """Exact recorded-response replay; verifies requests and never calls a server."""
    identifier = 'recorded-model-calls-v1'

    def __init__(self, calls_path):
        self.rows = [json.loads(line) for line in Path(calls_path).read_text().splitlines()]
        self.position = 0

    def complete(self, messages, *, seed=0, max_tokens=192, recorder=None):
        if self.position >= len(self.rows):
            raise ValueError('Recorded calls exhausted')
        row = self.rows[self.position]
        if (row['messages'] != messages or row['model_seed'] != seed
                or row['max_tokens'] != max_tokens):
            raise ValueError('Replay request differs from recorded request')
        self.position += 1
        record = row['model_records'][-1]['record']
        if recorder:
            recorder('model_call', record)
        if row['error'] == 'provider_error':
            raise ProviderError('Replaying recorded provider failure', record)
        return ModelResult(row['text'], record)


class _LiveTask(DesignTask):
    def probe(self, design, step, revised):
        if design == _INVALID:
            self.evaluations += 1
            return {'quality': 0.0, 'utility': 0.0, 'feasible': False,
                    'resource': 0.0, 'invalid_attempt': True}
        return super().probe(design, step, revised)


class _LiveEnvironment(SearchEnvironment):
    def __init__(self, family, seed, budget):
        super().__init__(family, seed, budget)
        self.task = _LiveTask(family, seed)
        self.proposal = None

    def candidate(self, action, memory=True):
        return self.proposal if self.proposal is not None else _INVALID

    def consume(self, action, proposal):
        """An invalid attempt consumes a slot without creating candidate evidence."""
        self.proposal = proposal
        self.prepare()
        incumbent = self.state.incumbent
        _, _, _, trace = self.step(action)
        if proposal is None:
            self.state.estimates.pop(','.join(map(str, _INVALID)), None)
            self.state.incumbent = self.best() if self.state.estimates else incumbent
            self.candidates.pop()
            trace['design'] = None
        return trace


def _messages(env, condition, action):
    # Source is given equally to every condition. Scoring is independently run by
    # the harness; no model claims are accepted as measured success or evidence.
    equations = inspect.getsource(DesignTask.evaluate)
    context = dict(zip(CONTEXT_NAMES, env.context()))
    # Neutral names preserve all numeric information for the ordinary comparator.
    context['signed_error_average'] = context.pop('mood')
    history = [{'step': h['step'], 'design': h['design'], 'observation': h['observation'],
                'revised': h['revised']} for h in env.state.history]
    payload = {
        'family': env.task.family, 'coordinate_semantics': SEMANTICS[env.task.family],
        'goal': 'Maximize independently checked feasible quality across workload scenarios.',
        'design_schema': {'design': 'six integers, each from 0 through 3'},
        'reference_corpus': REFERENCE, 'incumbent': env.state.incumbent,
        'observed_history': history, 'operational_context': context,
        'task_parameters': {'demand': env.task.demand, 'resource_limit': env.task.resource_limit},
        'current_revision': env.phase,
        'constraint_change': 'After half the attempts the resource limit is multiplied by 0.85.',
        'workload_distribution': 'Uniform scenario in [0.65,1.35]; actual future draws are hidden.',
        'scoring_implementation': equations,
        'remaining_attempts': env.budget-env.state.step,
        'control': {'strategy': action, **CONTROLS[action]},
    }
    persona = ''
    if condition == 'static_persona':
        persona = ' Maintain a fixed optimistic, enthusiastic persona throughout the trial.'
    elif condition == 'turn_sentiment':
        last = history[-1]['observation']['quality'] if history else .5
        persona = (' For this turn, use disappointed caution.' if last < .35
                   else ' For this turn, use positive confidence.')
    instruction = (
        'Return exactly one JSON object with one key, design, containing six integers in [0,3]. '
        'Do not include explanation or Markdown. Use only the supplied evidence. '
        'The control strategy regulates your search: diverge seeks a different configuration; '
        'refine adjusts the incumbent; combine recombines evidence-supported configurations; '
        'verify repeats the incumbent to check another workload. You may choose the actual design. '
        'All claims are checked by the harness. Never invent an observed result.' + persona)
    return [{'role': 'system', 'content': instruction},
            {'role': 'user', 'content': json.dumps(payload, sort_keys=True, allow_nan=False)}]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def run_live_benchmark(model, output, *, weights_dir, seeds=TEST_SEEDS,
                       families=FAMILIES, conditions=CONDITIONS, budget=8,
                       max_tokens=192, recorder=None):
    """Run an explicit adapter with bounded calls; fake-adapter runs stay labeled.

    `recorder`, e.g. Agent.record_artifact, receives each exact model audit record.
    One model call and one probe slot are charged per attempt, including malformed
    output and ProviderError. There are no repair calls or hidden model judges.
    """
    seeds, families, conditions = list(seeds), list(families), list(conditions)
    if (type(budget) is not int or not 2 <= budget <= 64
            or type(max_tokens) is not int or not 1 <= max_tokens <= 8192):
        raise ValueError('budget must be 2..64 and max_tokens 1..8192')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 800 for s in seeds):
        raise ValueError('Live held-out seeds must be unique integers >=800')
    if not families or len(set(families)) != len(families) or any(f not in FAMILIES for f in families):
        raise ValueError('Unknown or duplicate task family')
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError('Unknown or duplicate condition')
    output, weights_dir = Path(output), Path(weights_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; preserve prior experiments')
    counter_path, ordinary_path = weights_dir/'counterfactual_weights.json', weights_dir/'ordinary_weights.json'
    counterfactual, ordinary = RidgePredictor.load(counter_path), DirectValuePredictor.load(ordinary_path)
    for fitted in (counterfactual, ordinary):
        used = set(fitted.metadata.get('training_seeds', []))
        if used.intersection(seeds):
            raise ValueError('Live and generic training seed partitions overlap')
    output.mkdir(parents=True, exist_ok=True)
    config = {'schema_version': 1, 'seeds': seeds, 'families': families, 'conditions': conditions,
              'budget': budget, 'max_tokens_per_call': max_tokens,
              'maximum_model_calls': len(seeds)*len(families)*len(conditions)*budget,
              'maximum_completion_tokens': len(seeds)*len(families)*len(conditions)*budget*max_tokens,
              'counterfactual_sha256': _sha(counter_path), 'ordinary_sha256': _sha(ordinary_path),
              'model_adapter': getattr(model, 'identifier', type(model).__name__),
              'model': getattr(model, 'model', None),
              'execution_mode': ('live' if isinstance(model, LocalChatModel) else
                                 'recorded response replay' if isinstance(model, RecordedCallsModel) else
                                 'injected adapter; verify provenance'),
              'python': platform.python_version(), 'numpy': np.__version__,
              'scope': 'same-model numeric toy-design generation; no open-ended creativity claim'}
    _write(output/'config.json', config)
    results = []
    started = time.perf_counter()
    with (output/'model_calls.jsonl').open('w') as calls, (output/'traces.jsonl').open('w') as traces, (output/'results.jsonl').open('w') as results_file:
        for seed in seeds:
            for family in families:
                for condition in conditions:
                    env = _LiveEnvironment(family, seed, budget)
                    policy = Policy(condition, counterfactual, ordinary, seed)
                    valid, invalid, errors = 0, 0, 0
                    prompt_tokens = completion_tokens = 0
                    missing_usage = False
                    request_characters = 0
                    trial_seconds = time.perf_counter()
                    for turn in range(budget):
                        action = policy.action(env)
                        messages = _messages(env, condition, action)
                        request_characters += sum(len(m['content']) for m in messages)
                        records = []

                        def capture(kind, record):
                            records.append({'kind': kind, 'record': record})
                            if recorder:
                                recorder(kind, record)

                        proposal, text, error = None, None, None
                        try:
                            response = model.complete(messages, seed=seed*1000+turn,
                                                      max_tokens=max_tokens, recorder=capture)
                            text = response.text
                            if not records:
                                capture('model_call', response.record)
                            try:
                                proposal = parse_design(text)
                            except (ValueError, TypeError, json.JSONDecodeError):
                                error = 'invalid_design_schema'
                        except ProviderError as exc:
                            errors += 1
                            error = 'provider_error'
                            if not records:
                                capture('model_call', exc.record)
                        if proposal is None:
                            invalid += 1
                        else:
                            valid += 1
                        usage = records[-1]['record'].get('response', {}).get('usage', {}) if records else {}
                        for key in ('prompt_tokens', 'completion_tokens'):
                            if not isinstance(usage.get(key), int):
                                missing_usage = True
                        prompt_tokens += usage.get('prompt_tokens') or 0
                        completion_tokens += usage.get('completion_tokens') or 0
                        row = {'seed': seed, 'family': family, 'condition': condition, 'turn': turn,
                               'messages': messages, 'model_seed': seed*1000+turn, 'max_tokens': max_tokens,
                               'text': text, 'error': error,
                               'valid_design': proposal, 'model_records': records}
                        calls.write(json.dumps(row, allow_nan=False)+'\n'); calls.flush()
                        trace = env.consume(action, proposal)
                        trace.update(seed=seed, family=family, condition=condition,
                                     valid_design=proposal is not None, model_error=error)
                        traces.write(json.dumps(trace, allow_nan=False)+'\n'); traces.flush()
                    if valid:
                        scored = env.task.adjudicate(env.state.incumbent)
                    else:
                        scored = dict(quality=0.0, usefulness=0.0, constraint_compliance=0.0, worst_case_feasible=False)
                    original = bool(valid and tuple(env.state.incumbent) not in REFERENCE)
                    result = {'seed': seed, 'family': family, 'condition': condition, **scored,
                        'final_design': list(env.state.incumbent) if valid else None,
                        'benchmark_original': original,
                        'useful_original': bool(original and scored['quality'] >= .45 and scored['constraint_compliance'] >= .9),
                        'valid_attempts': valid, 'invalid_attempts': invalid, 'provider_errors': errors,
                        'model_calls': budget, 'probe_slots': env.task.evaluations,
                        'actual_valid_design_probes': valid, 'predictor_evaluations': policy.predictions,
                        'completion_token_upper_budget': budget*max_tokens,
                        'reported_prompt_tokens': None if missing_usage else prompt_tokens,
                        'reported_completion_tokens': None if missing_usage else completion_tokens,
                        'request_characters': request_characters,
                        'seconds': time.perf_counter()-trial_seconds,
                        'initial_snapshot': env.initial_snapshot,
                        'final_regulation': {k: asdict(env.state)[k] for k in ('mood','load','expected_gain')},
                        'adjudication_evaluations': 128 if valid else 64,
                        'execution_mode': config['execution_mode']}
                    results.append(result)
                    results_file.write(json.dumps(result, allow_nan=False)+'\n'); results_file.flush()
    summary = {'schema_version': 1, 'trials': len(results), 'seconds': time.perf_counter()-started,
               'model_calls': sum(r['model_calls'] for r in results), 'means': {},
               'paired_quality_vs_hybrid': {}, 'execution_mode': config['execution_mode'],
               'scope': config['scope'], 'uncertainty': 'seed-cluster bootstrap; exploratory small-sample intervals'}
    for condition in conditions:
        rows = [r for r in results if r['condition'] == condition]
        summary['means'][condition] = {key: float(np.mean([r[key] for r in rows]))
            for key in ('quality','constraint_compliance','useful_original','valid_attempts','invalid_attempts','request_characters')}
    if 'hybrid' in conditions:
        hybrid = {(r['seed'], r['family']): r['quality'] for r in results if r['condition']=='hybrid'}
        rng = np.random.default_rng(8811)
        for condition in conditions:
            differences = np.asarray([np.mean([hybrid[(s,f)]-next(r['quality'] for r in results
                if r['seed']==s and r['family']==f and r['condition']==condition) for f in families]) for s in seeds])
            bootstrap = np.mean(rng.choice(differences, (2000,len(seeds))), axis=1)
            summary['paired_quality_vs_hybrid'][condition] = {
                'difference': float(differences.mean()), 'ci95': [float(np.quantile(bootstrap,.025)), float(np.quantile(bootstrap,.975))],
                'seed_clusters': len(seeds)}
    _write(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='Explicit API prefix, e.g. http://localhost:8000/v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--weights', required=True, help='Directory with both trained predictor JSON artifacts')
    parser.add_argument('--output', required=True, help='New or empty output directory')
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=192)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(TEST_SEEDS))
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    args = parser.parse_args()
    model = LocalChatModel(base_url=args.base_url, model=args.model)
    summary = run_live_benchmark(model, args.output, weights_dir=args.weights,
                                seeds=args.seeds, families=args.families,
                                budget=args.budget, max_tokens=args.max_tokens)
    print(json.dumps({'trials': summary['trials'], 'model_calls': summary['model_calls'],
                      'execution_mode': summary['execution_mode'], 'output': args.output}, indent=2))


if __name__ == '__main__':
    main()
