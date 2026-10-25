"""Small, serializable CPU models for experimental counterfactual regulation.

These weights are generic offline training, not an individual agent's history.
All adaptation within a trial lives in SearchState in benchmark.py.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ACTIONS = ('diverge', 'refine', 'combine', 'verify')


def action_features(context, action):
    """Shared information/capacity: block-linear action-conditioned features."""
    x = np.asarray([1.0, *context], dtype=float)
    out = np.zeros(len(ACTIONS) * len(x))
    i = ACTIONS.index(action)
    out[i * len(x):(i + 1) * len(x)] = x
    return out


class RidgePredictor:
    """Four-head linear model, fitted only on the declared training partition."""
    def __init__(self, weights=None, ridge=1.0, metadata=None):
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.ridge = float(ridge)
        self.metadata = metadata or {}

    def fit(self, contexts, actions, targets):
        x = np.vstack([action_features(c, a) for c, a in zip(contexts, actions)])
        y = np.asarray(targets, dtype=float)
        self.weights = np.linalg.solve(x.T @ x + self.ridge * np.eye(x.shape[1]), x.T @ y)
        self.metadata.update(samples=len(x), parameters=int(self.weights.size), heads=int(y.shape[1]))
        return self

    def predict(self, context, action):
        if self.weights is None:
            raise ValueError('Predictor has not been trained')
        return action_features(context, action) @ self.weights

    def save(self, path):
        Path(path).write_text(json.dumps({'schema_version': 1, 'ridge': self.ridge,
            'weights': self.weights.tolist(), 'metadata': self.metadata}, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data['schema_version'] != 1:
            raise ValueError('Unsupported predictor schema')
        return cls(data['weights'], data['ridge'], data['metadata'])

CONTEXT_NAMES = ('quality', 'uncertainty', 'feasible_rate', 'load', 'mood',
    'recent_gain', 'expected_gain', 'failures', 'stagnation', 'budget_fraction',
    'revised', 'demand', 'resource_limit', 'expected_diverge', 'expected_refine',
    'expected_combine', 'expected_verify')


class CounterfactualRegulator:
    """Reusable platform adapter for the trained experimental component.

    Input is a named operational context. Missing task measurements use declared
    defaults; simulator-trained weights outside the simulator are unvalidated.
    The selected action regulates exploration, retrieval, verification, and horizon.
    """
    DEFAULTS = dict(zip(CONTEXT_NAMES, [0, 1, 0, 0, 0, 0, .1, 0, 0, 0, 0, 1, 1, .1, .1, .1, .1]))

    def __init__(self, predictor):
        self.predictor = predictor

    @classmethod
    def load(cls, path):
        return cls(RidgePredictor.load(path))

    def select(self, context):
        values = {**self.DEFAULTS, **context}
        vector = [float(values[name]) for name in CONTEXT_NAMES]
        predictions = {a: self.predictor.predict(vector, a).tolist() for a in ACTIONS}
        scores = {a: float(p[0] + (.03 + .08*values['uncertainty'])*p[1]
                  - (.02 + .08*values['load'])*p[2] + .04*p[3]
                  + .015*max(0, -values['mood'])*(a == 'diverge'))
                  for a,p in predictions.items()}
        action = max(ACTIONS, key=lambda a: scores[a])
        return action, predictions, scores

    def actuation_context(self, context):
        action, predictions, scores = self.select(context)
        controls = {
            'diverge': {'exploration': .9, 'verification': .25, 'retrieval_bias': -.2, 'planning_horizon': 3},
            'refine': {'exploration': .25, 'verification': .6, 'retrieval_bias': .1, 'planning_horizon': 6},
            'combine': {'exploration': .55, 'verification': .45, 'retrieval_bias': .6, 'planning_horizon': 5},
            'verify': {'exploration': .1, 'verification': .95, 'retrieval_bias': 0., 'planning_horizon': 8},
        }[action]
        return {**controls, 'learning_rate': .15, 'strategy': action,
                'predicted_consequences': predictions, 'action_scores': scores,
                'provenance': 'ridge-counterfactual-regulator-v1',
                'validation_scope': 'procedural design-search simulator'}


def nonlinear_action_features(context, action):
    """72 terms per action, 288 active direct-value parameters in total."""
    c = np.asarray(context, dtype=float)
    if len(c) != 17:
        raise ValueError('Direct comparator requires the same 17 observed inputs')
    local = np.concatenate(([1.0], c, c*c, c*c*c, c*np.roll(c, 1),
                            [c[1]*c[3], c[4]*c[8], c[9]*c[10]]))
    out = np.zeros(4*len(local))
    i = ACTIONS.index(action)
    out[i*len(local):(i+1)*len(local)] = local
    return out


class DirectValuePredictor:
    """Structurally different adaptive controller: scalar contextual action value.

    All 288 parameters affect selection; nonlinear polynomial features match the
    outcome model's parameter budget. No named affect terms are required, though
    the same signed prediction-error history is accessible as numeric input.
    """
    def __init__(self, weights=None, ridge=1.0, metadata=None):
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.ridge, self.metadata = ridge, metadata or {}

    def fit(self, contexts, actions, targets):
        x = np.vstack([nonlinear_action_features(c,a) for c,a in zip(contexts,actions)])
        self.weights = np.linalg.solve(x.T@x+self.ridge*np.eye(x.shape[1]),x.T@np.asarray(targets))
        self.metadata.update(samples=len(x), parameters=int(self.weights.size), heads=1,
                             features='raw, squares, cubes, adjacent products, three cross-products')
        return self

    def predict(self, context, action):
        return float(nonlinear_action_features(context,action)@self.weights)

    def save(self, path):
        Path(path).write_text(json.dumps({'schema_version':1, 'ridge':self.ridge,
             'weights':self.weights.tolist(), 'metadata':self.metadata},indent=2)+'\n')

    @classmethod
    def load(cls, path):
        data=json.loads(Path(path).read_text())
        return cls(data['weights'],data['ridge'],data['metadata'])
