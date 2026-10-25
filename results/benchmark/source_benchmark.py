"""Executable, procedural idea-search benchmark; this does NOT measure LLM creativity.

Run: python -m regesh.benchmark --output artifacts/benchmark
Every condition gets the same candidate operators, observations, reference motifs,
32 actual simulator evaluations, and a fresh snapshot. Predictions cost arithmetic,
not extra oracle calls. All design choices are discrete mechanism choices.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
import platform
from datetime import datetime, timezone
import numpy as np
from .learning import ACTIONS, RidgePredictor, DirectValuePredictor

FAMILIES = ('software', 'engineering', 'scientific')
# Explicit finite reference corpus, available to EVERY condition.
REFERENCE = ((1, 0, 1, 1, 1, 0), (2, 1, 0, 2, 0, 1),
             (0, 2, 2, 0, 2, 2), (1, 1, 1, 1, 1, 1))
SEMANTICS = {
    'software': ['cache', 'compression', 'replication', 'batching', 'retry', 'adaptive_eviction'],
    'engineering': ['storage', 'insulation', 'redundancy', 'demand_shift', 'reserve', 'feedback'],
    'scientific': ['sample_size', 'blocking', 'replication', 'factorial_design', 'controls', 'sequential_testing'],
}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_config(config=None):
    if config is not None:
        return dict(config)
    return json.loads((Path(__file__).parent / 'data/benchmark.json').read_text())


@dataclass
class SearchState:
    """A complete per-trial initial snapshot; trained weights are separately fixed."""
    schema_version: int = 1
    step: int = 0
    mood: float = 0.0
    load: float = 0.0
    uncertainty: float = 1.0
    recent_gain: float = 0.0
    expected_gain: float = 0.1
    failures: float = 0.0
    since_progress: int = 0
    action_expectations: list = field(default_factory=lambda: [0.1] * 4)
    estimates: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    incumbent: tuple = REFERENCE[3]


class DesignTask:
    """Three independently checkable toy systems with resource/robustness constraints.

    Training sees individual scenarios. Final adjudication uses a fixed independent
    64-scenario suite and is blind to condition. No language judge is involved.
    """
    def __init__(self, family, seed):
        if family not in FAMILIES:
            raise ValueError(family)
        self.family, self.seed = family, int(seed)
        self.rng = np.random.default_rng(seed)
        self.demand = float(self.rng.uniform(0.8, 1.2))
        self.resource_limit = float(self.rng.uniform(1.0, 1.35))
        self.probes = self.rng.uniform(.65, 1.35, 256)
        self.test_cases = np.random.default_rng(seed + 900000).uniform(.65, 1.35, 64)
        self.evaluations = 0

    def evaluate(self, design, scenario, revised=False):
        a, b, c, d, e, f = np.asarray(design, dtype=float) / 3
        demand = self.demand * scenario
        limit = self.resource_limit * (.85 if revised else 1.0)
        if self.family == 'software':
            # Caching/compression help different workloads; batching costs latency.
            service = (.35 + .8*a + .4*b + .35*d + .25*f - .16*a*b) / demand
            robustness = .4 + .45*c + .28*e + .17*f - .12*d
            resource = .25 + .65*a + .35*c + .18*b + .22*e + .12*d + .08*f
            latency = demand / (.35 + .65*a + .4*b + .2*f) + .45*d + .15*e
            feasible = resource <= limit and robustness >= .57 and latency <= 2.4
            utility = .55*min(service, 1.6)/1.6 + .3*min(robustness, 1) + .15*(1-min(resource/1.8, 1))
        elif self.family == 'engineering':
            supply = (.42 + .5*a + .38*b + .34*d + .2*f) / demand
            robustness = .38 + .48*c + .36*e + .14*a - .1*d
            resource = .28 + .5*a + .3*b + .3*c + .16*d + .3*e + .1*f
            feasible = resource <= limit and robustness >= .6 and supply >= .65
            utility = .5*min(supply, 1.6)/1.6 + .35*min(robustness, 1) + .15*(1-min(resource/1.8, 1))
        else:
            # Power improves through blocking and replication, costs through samples.
            power = (.3 + .7*a + .4*b + .35*c + .18*d) / demand
            identification = .35 + .35*e + .3*d + .25*b - .18*f*(1-e)
            resource = .2 + .62*a + .28*b + .4*c + .25*d + .22*e - .18*f
            feasible = resource <= limit and identification >= .65 and power >= .68
            utility = .55*min(power, 1.6)/1.6 + .3*min(identification, 1) + .15*(1-min(resource/1.8, 1))
        utility = float(np.clip(utility, 0, 1))
        return {'quality': utility if feasible else .25*utility,
                'utility': utility, 'feasible': bool(feasible), 'resource': float(resource)}

    def probe(self, design, step, revised):
        self.evaluations += 1
        return self.evaluate(design, self.probes[step], revised)

    def adjudicate(self, design, revised=True):
        rows = [self.evaluate(design, c, revised) for c in self.test_cases]
        return {'quality': float(np.mean([r['quality'] for r in rows])),
                'usefulness': float(np.mean([r['utility'] for r in rows])),
                'constraint_compliance': float(np.mean([r['feasible'] for r in rows])),
                'worst_case_feasible': all(r['feasible'] for r in rows)}


class SearchEnvironment:
    def __init__(self, family, seed, budget):
        self.task = DesignTask(family, seed)
        self.state = SearchState()
        self.seed, self.budget = seed, budget
        self.phase = False
        self.initial_snapshot = asdict(self.state)
        self.candidates = []
        self.pre_revision = None

    def estimate(self, design):
        key = ','.join(map(str, design))
        row = self.state.estimates.get(key)
        if row is None:
            return 0.0, 1.0, 0.0
        # Pseudocount: no false certainty from one successful scenario.
        mean = row['sum'] / row['n']
        uncertainty = 1 / math.sqrt(row['n'] + 1)
        return mean, uncertainty, row['feasible'] / row['n']

    def best(self):
        if not self.state.estimates:
            return self.state.incumbent
        # Same evidence-focused selection for all policies; no affect rewriting reward.
        key = max(self.state.estimates, key=lambda k: self.state.estimates[k]['sum'] / self.state.estimates[k]['n'])
        return tuple(map(int, key.split(',')))

    def prepare(self):
        s = self.state
        if s.step == self.budget//2 and not self.phase:
            self.pre_revision = self.task.adjudicate(s.incumbent)
            self.phase = True
            s.estimates = {}
            s.uncertainty = 1.0

    def context(self):
        self.prepare()
        s = self.state
        quality, uncertainty, feasible = self.estimate(s.incumbent)
        return [quality, uncertainty, feasible, s.load, s.mood, s.recent_gain,
                s.expected_gain, s.failures, min(s.since_progress/10, 1),
                s.step/self.budget, float(self.phase), self.task.demand/1.2,
                self.task.resource_limit/1.35, *s.action_expectations]

    def candidate(self, action, memory=True):
        # Coupled potential proposals: same seed/step/action yields same random bits.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.state.step, ACTIONS.index(action), 871]))
        best = np.asarray(self.state.incumbent)
        if action == 'verify':
            return tuple(int(x) for x in best)
        if action == 'diverge':
            return tuple(int(x) for x in rng.integers(0, 4, 6))
        if action == 'refine':
            candidate = best.copy()
            j = int(rng.integers(0, 6))
            candidate[j] = (candidate[j] + rng.choice([-1, 1])) % 4
            return tuple(int(x) for x in candidate)
        if self.state.estimates:
            # Candidate retrieval coupling preserves factual evidence, adds a
            # state-dependent activation term. Ablation removes only that term.
            def activation(k):
                row = self.state.estimates[k]
                return row['sum']/row['n'] + (.18*self.state.mood*row.get('surprise', 0) if memory else 0)
            evidence = sorted(self.state.estimates, key=activation, reverse=True)
            donor = np.asarray(tuple(map(int, evidence[int(rng.integers(0, min(4, len(evidence))))].split(','))))
        else:
            donor = np.asarray(REFERENCE[int(rng.integers(0, len(REFERENCE)))])
        mask = rng.integers(0, 2, 6).astype(bool)
        return tuple(int(x) for x in np.where(mask, best, donor))

    def step(self, action, memory=True, frozen=False, shuffle_rng=None):
        s = self.state
        self.prepare()
        before = self.context()
        design = self.candidate(action, memory)
        old_mean, old_uncertainty, _ = self.estimate(s.incumbent)
        result = self.task.probe(design, s.step, self.phase)
        key = ','.join(map(str, design))
        row = s.estimates.setdefault(key, {'sum': 0.0, 'n': 0, 'feasible': 0})
        row['sum'] += result['quality']; row['n'] += 1; row['feasible'] += result['feasible']
        s.incumbent = self.best()
        new_mean, new_uncertainty, _ = self.estimate(s.incumbent)
        gain = new_mean - old_mean
        surprise = result['quality'] - s.action_expectations[ACTIONS.index(action)]
        row['surprise'] = float(surprise)
        old_load = s.load
        # Distinct resource dynamics: verification/restorative review reduces burden;
        # high effort is not itself distress; appraisal uses mismatch to progress.
        next_load = float(np.clip(.9*s.load + {'diverge': .12, 'refine': .07, 'combine': .09, 'verify': -.08}[action], 0, 1))
        if not frozen:
            s.load = next_load
            s.uncertainty = new_uncertainty
            s.recent_gain = .5*s.recent_gain + .5*gain
            s.expected_gain = .85*s.expected_gain + .15*gain
            s.failures = .85*s.failures + .15*(not result['feasible'])
            s.mood = float(np.clip(.85*s.mood + .15*surprise, -1, 1))
            i = ACTIONS.index(action)
            s.action_expectations[i] = .8*s.action_expectations[i] + .2*result['quality']
            s.since_progress = 0 if gain > .01 else s.since_progress + 1
            if shuffle_rng is not None:
                # Marginal values preserved within this trial; temporal alignment destroyed.
                past = [h['after_regulation'] for h in s.history]
                if past:
                    shuffled = past[int(shuffle_rng.integers(len(past)))]
                    s.mood, s.load, s.recent_gain, s.expected_gain = shuffled
        after_regulation = [s.mood, s.load, s.recent_gain, s.expected_gain]
        s.step += 1
        trace = {'step': s.step, 'action': action, 'design': list(design), 'observation': result,
                 'context': before, 'after_regulation': after_regulation,
                 'gain': gain, 'prediction_error': surprise, 'revised': self.phase}
        s.history.append(trace)
        self.candidates.append(tuple(design))
        # Learned outcome predictions; targets generated by actual transitions only.
        transition = [gain, old_uncertainty-new_uncertainty, next_load-old_load, float(result['feasible'])]
        # Ordinary controller learns direct control value with 288 active parameters.
        direct_value = (gain + .04*float(result['feasible'])
            + (.03+.08*before[1])*(old_uncertainty-new_uncertainty)
            - (.02+.08*before[3])*(next_load-old_load)
            + .015*max(0,-before[4])*(action=='diverge'))
        ordinary = direct_value
        return before, transition, ordinary, trace


class Policy:
    def __init__(self, condition, counterfactual, ordinary, seed):
        self.condition, self.counterfactual, self.ordinary = condition, counterfactual, ordinary
        self.shuffle_rng = np.random.default_rng(seed + 7100)
        self.predictions = 0

    def action(self, env):
        s, c = env.state, env.context()
        condition = self.condition
        # Every condition spends the same four prediction calls per decision.
        model = self.ordinary if condition == 'ordinary_adaptive' else self.counterfactual
        predictions = [model.predict(c, action) for action in ACTIONS]
        self.predictions += len(ACTIONS)
        self.last_predictions = {a: (p.tolist() if hasattr(p, 'tolist') else float(p)) for a,p in zip(ACTIONS,predictions)}
        if condition == 'memory_planning':
            return ('diverge', 'refine', 'combine', 'verify')[s.step % 4]
        if condition == 'static_persona':
            # Fixed optimistic/exploratory persona translated to search controls.
            return ('diverge', 'combine', 'diverge', 'refine', 'verify')[s.step % 5]
        if condition == 'turn_sentiment':
            last = s.history[-1]['observation']['quality'] if s.history else .5
            return 'diverge' if last < .35 else ('verify' if s.step % 3 == 2 else 'refine')
        if condition == 'heuristic':
            if s.load > .65: return 'verify'
            if s.since_progress > 3: return 'diverge'
            return 'combine' if s.mood > .05 else 'refine'
        if condition == 'ordinary_adaptive':
            scores = [float(p) for p in predictions]
        elif condition == 'no_counterfactual':
            scores = [float(p[0]) for p in predictions]
        else:
            # Fixed before experiments; no tuning against held-out test scores.
            # Predicted uncertainty and burden matter conditional on current state.
            scores = [float(p[0] + (.03 + .08*c[1])*p[1]
                       - (.02 + .08*s.load)*p[2] + .04*p[3]
                       + .015*max(0, -s.mood)*(a == 'diverge'))
                      for a, p in zip(ACTIONS, predictions)]
        return ACTIONS[int(np.argmax(scores))]


def train_models(config, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    contexts, actions, transitions, direct = [], [], [], []
    training_file = output / 'training.jsonl'
    started = time.perf_counter()
    with training_file.open('w') as stream:
        for seed in config['training_seeds']:
            for family in config['training_families']:
                for episode in range(config['training_episodes_per_seed']):
                    trial_seed = seed*100 + episode
                    env = SearchEnvironment(family, trial_seed, config['budget'])
                    rng = np.random.default_rng(trial_seed + 123)
                    for _ in range(config['budget']):
                        action = ACTIONS[int(rng.integers(4))]
                        context, target, direct_target, trace = env.step(action)
                        contexts.append(context); actions.append(action)
                        transitions.append(target); direct.append(direct_target)
                        stream.write(json.dumps({'partition': 'train', 'family': family,
                            'seed': seed, 'episode': episode, **trace,
                            'transition_target': target, 'direct_target': direct_target}) + '\n')
    metadata = {'training_families': config['training_families'],
                'training_seeds': config['training_seeds'], 'context_size': len(contexts[0]),
                'generic_training': True, 'test_adaptation': 'per-trial SearchState only'}
    counterfactual = RidgePredictor(ridge=config['ridge'], metadata=dict(metadata)).fit(contexts, actions, transitions)
    ordinary = DirectValuePredictor(ridge=config['ridge'], metadata=dict(metadata)).fit(contexts, actions, direct)
    counterfactual.save(output / 'counterfactual_weights.json')
    ordinary.save(output / 'ordinary_weights.json')
    dump(output / 'training_summary.json', {'samples': len(contexts),
         'training_oracle_evaluations': len(contexts), 'seconds': time.perf_counter()-started,
         'counterfactual_parameters': counterfactual.weights.size,
         'ordinary_parameters': ordinary.weights.size})
    return counterfactual, ordinary


def trial(condition, family, seed, budget, counterfactual, ordinary):
    env = SearchEnvironment(family, seed, budget)
    policy = Policy(condition, counterfactual, ordinary, seed)
    started = time.perf_counter()
    traces = []
    predicted_feasibility = []
    for _ in range(budget):
        action = policy.action(env)
        # Diagnostics do NOT affect action and make no oracle calls.
        forecast = float(np.clip(counterfactual.predict(env.context(), action)[3], 0, 1))
        _, _, _, trace = env.step(action, memory=condition != 'no_memory',
             frozen=condition == 'frozen',
             shuffle_rng=policy.shuffle_rng if condition == 'shuffled' else None)
        trace['predicted_feasibility'] = forecast
        trace['controller_predictions'] = policy.last_predictions
        trace['executed_reason'] = {'controller': condition, 'action': action, 'policy': 'declared numerical decision rule; see source manifest'}
        predicted_feasibility.append((forecast-float(trace['observation']['feasible']))**2)
        traces.append(trace)
    scored = env.task.adjudicate(env.state.incumbent)
    canonical = tuple(env.state.incumbent)
    original = canonical not in REFERENCE
    result = {'condition': condition, 'family': family, 'seed': seed, **scored,
        'benchmark_original': original,
        'useful_original': bool(original and scored['quality'] >= .45 and scored['constraint_compliance'] >= .9),
        'canonical_design_diversity': len(set(env.candidates)),
        'mechanism_diversity': len({tuple(v>0 for v in candidate) for candidate in env.candidates}),
        'mechanism_original': tuple(v>0 for v in canonical) not in {tuple(v>0 for v in r) for r in REFERENCE},
        'post_criticism_gain': scored['quality'] - env.pre_revision['quality'],
        'brier': float(np.mean(predicted_feasibility)),
        'oracle_evaluations': env.task.evaluations,
        'predictor_evaluations': policy.predictions,
        'diagnostic_predictor_evaluations': budget,
        'tokens': 0, 'llm_calls': 0, 'external_tools': 0,
        'seconds': time.perf_counter()-started,
        'final_design': list(canonical),
        'initial_snapshot': env.initial_snapshot,
        'final_regulation': {'mood': env.state.mood, 'load': env.state.load,
                             'expected_gain': env.state.expected_gain},
        'adjudication_evaluations': 128,
        'reference_corpus_size': len(REFERENCE)}
    return result, traces


def summarize(results, config):
    metrics = ['quality', 'usefulness', 'constraint_compliance', 'benchmark_original',
               'useful_original', 'mechanism_diversity', 'canonical_design_diversity', 'mechanism_original', 'post_criticism_gain', 'brier',
               'oracle_evaluations', 'predictor_evaluations', 'seconds']
    summary = {'means': {}, 'family_means': {}, 'paired_quality_vs_hybrid': {},
               'uncertainty_method': 'paired bootstrap of whole seeds, averaging all families per seed',
               'claim_scope': 'procedural simulator only; no LLM creativity or published novelty evidence'}
    for condition in config['conditions']:
        rows = [r for r in results if r['condition'] == condition]
        summary['means'][condition] = {m: float(np.mean([r[m] for r in rows])) for m in metrics}
        summary['family_means'][condition] = {family: float(np.mean([r['quality'] for r in rows if r['family']==family])) for family in config['test_families']}
    hybrid = {(r['family'],r['seed']):r for r in results if r['condition']=='hybrid'}
    rng = np.random.default_rng(config['bootstrap_seed'])
    for condition in config['conditions']:
        differences = []
        for seed in config['test_seeds']:
            rows = [r for r in results if r['condition']==condition and r['seed']==seed]
            differences.append(np.mean([hybrid[(r['family'],seed)]['quality']-r['quality'] for r in rows]))
        differences = np.asarray(differences)
        boot = np.mean(rng.choice(differences, (config['bootstrap_samples'],len(differences))), axis=1)
        summary['paired_quality_vs_hybrid'][condition] = {'hybrid_minus_condition':float(np.mean(differences)),
            'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))], 'seed_clusters':len(differences)}
    return summary


def run_benchmark(output_dir, config=None):
    config = read_config(config)
    # Refuse split leakage and repeated trial IDs rather than silently continuing.
    partitions = [set(config[k]) for k in ('training_seeds','development_seeds','test_seeds')]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
        raise ValueError('Train/development/test seeds overlap')
    if config['held_out_family'] in config['training_families']:
        raise ValueError('Held-out family appears in training')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; preserve prior experiments')
    output.mkdir(parents=True, exist_ok=True)
    dump(output/'config.json', config)
    manifest = {'schema_version': 1, 'study_status': 'exploratory_corrected_v1',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'python': platform.python_version(), 'numpy': np.__version__, 'source_sha256': {}}
    for filename in ('benchmark.py', 'learning.py'):
        content = (Path(__file__).parent/filename).read_bytes()
        manifest['source_sha256'][filename] = hashlib.sha256(content).hexdigest()
        (output/('source_'+filename)).write_bytes(content)
    dump(output/'manifest.json', manifest)
    counterfactual, ordinary = train_models(config, output)
    weight_hash = hashlib.sha256((output/'counterfactual_weights.json').read_bytes()).hexdigest()
    results, development = [], []
    with (output/'traces.jsonl').open('w') as traces_file:
        for partition, seeds, families in [
            ('development', config['development_seeds'], config['development_families']),
            ('test', config['test_seeds'], config['test_families'])]:
            for seed in seeds:
                for family in families:
                    for condition in config['conditions']:
                        result, traces = trial(condition, family, seed, config['budget'], counterfactual, ordinary)
                        result.update(partition=partition, weight_sha256=weight_hash)
                        (results if partition=='test' else development).append(result)
                        for trace in traces:
                            traces_file.write(json.dumps({'partition':partition, 'seed':seed,
                                 'family':family, 'condition':condition, **trace})+'\n')
    with (output/'results.jsonl').open('w') as stream:
        for row in results: stream.write(json.dumps(row)+'\n')
    with (output/'development.jsonl').open('w') as stream:
        for row in development: stream.write(json.dumps(row)+'\n')
    summary = summarize(results, config)
    summary.update(test_trials=len(results), development_trials=len(development),
                   budget=config['budget'], weight_sha256=weight_hash)
    dump(output/'summary.json', summary)
    write_report(output, summary)
    return summary


def write_report(output, summary):
    lines = ['# Procedural search experiment', '',
        'These are executed synthetic search results, not measurements of a live language model.', '',
        '| Condition | Quality | Constraint compliance | Useful novel fraction | Included-mechanism diversity |',
        '|---|---:|---:|---:|---:|']
    for name, m in summary['means'].items():
        lines.append(f"| {name} | {m['quality']:.4f} | {m['constraint_compliance']:.4f} | {m['useful_original']:.4f} | {m['mechanism_diversity']:.2f} |")
    lines += ['', 'Differences are hybrid minus comparator; 95% paired seed-cluster bootstrap intervals.', '',
              '| Comparator | Difference | 95% interval |', '|---|---:|---|']
    for name, d in summary['paired_quality_vs_hybrid'].items():
        lo,hi = d['ci95']
        lines.append(f"| {name} | {d['hybrid_minus_condition']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ['', 'The unlabelled hybrid is exactly the same computation. Any identical performance is expected; labels are not evidence of an emotion-specific benefit.',
       'Static persona and sentiment conditions are numerical analogues only; they do not test prompt conditioning on an LLM.',
       'Originality means absence from four explicitly listed canonical designs. It does not establish published novelty, and distance is not used as a creativity score.',
       'The held-out scientific family is a modest formula transfer test; all families share a six-knob design grammar.',
       'Mechanism diversity counts distinct subsets of enabled knobs (64 possible), while canonical design diversity includes their levels (4096 possible). The four-design originality corpus has weak coverage.',
       'Brier is common outcome-model calibration on policy-selected actions, not independent policy calibration. Internal burden is a regulator cost, not an externally enforced resource constraint.',
       'Structural corrections followed an exploratory pilot whose raw files were overwritten; see docs/exploratory_pilot.md. All results are exploratory. Coefficients were not optimized against held-out scores; intervals are not multiplicity-corrected.',
       'See docs/benchmark_protocol.md for the equations, limitations, and live-model follow-up.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/benchmark')
    parser.add_argument('--config')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else None
    result = run_benchmark(args.output, config)
    print(json.dumps({'test_trials':result['test_trials'],
          'hybrid_quality':result['means']['hybrid']['quality'],
          'ordinary_quality':result['means']['ordinary_adaptive']['quality'],
          'output':args.output}, indent=2))

if __name__ == '__main__':
    main()
