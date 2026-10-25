"""Meaningful checks for leakage, budgets, exact controls, and serialization."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from regesh.learning import RidgePredictor, DirectValuePredictor, CounterfactualRegulator
from regesh.benchmark import ACTIONS, read_config, train_models, trial, run_benchmark, SearchEnvironment


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.config = read_config()
        cls.config.update(training_seeds=[10], development_seeds=[20], test_seeds=[30],
                          training_episodes_per_seed=2, budget=12, bootstrap_samples=20)
        cls.model, cls.ordinary = train_models(cls.config, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_labels_removed_exactly_match(self):
        a, ta = trial('hybrid', 'scientific', 30, 12, self.model, self.ordinary)
        b, tb = trial('unlabelled_hybrid', 'scientific', 30, 12, self.model, self.ordinary)
        for row in ta + tb:
            row['executed_reason']['controller'] = 'label-redacted'
        self.assertEqual(ta, tb)
        self.assertEqual(a['final_design'], b['final_design'])
        self.assertEqual(a['quality'], b['quality'])

    def test_equal_actual_budgets_and_isolation(self):
        for condition in self.config['conditions']:
            result, traces = trial(condition, 'scientific', 30, 12, self.model, self.ordinary)
            self.assertEqual(result['oracle_evaluations'], 12)
            self.assertEqual(result['predictor_evaluations'], 48)
            self.assertEqual(len(traces), 12)
            self.assertEqual(result['initial_snapshot']['history'], [])
            self.assertEqual(result['initial_snapshot']['mood'], 0)

    def test_active_parameter_counts_and_serialization(self):
        self.assertEqual(self.model.weights.size, self.ordinary.weights.size)
        self.assertEqual(self.model.weights.size, 288)
        x = SearchEnvironment('software', 30, 12).context()
        for filename, cls, original in [
            ('counterfactual_weights.json', RidgePredictor, self.model),
            ('ordinary_weights.json', DirectValuePredictor, self.ordinary)]:
            loaded = cls.load(Path(self.tmp.name)/filename)
            for action in ACTIONS:
                np.testing.assert_array_equal(loaded.predict(x, action), original.predict(x, action))

    def test_saved_training_model_is_fixed_during_test(self):
        before = self.model.weights.copy()
        trial('hybrid', 'scientific', 30, 12, self.model, self.ordinary)
        np.testing.assert_array_equal(before, self.model.weights)

    def test_constraint_revision_is_visible_before_choice(self):
        env = SearchEnvironment('software', 30, 12)
        for _ in range(6): env.step('diverge')
        context = env.context()
        self.assertEqual(context[10], 1.0)
        self.assertIsNotNone(env.pre_revision)
        self.assertEqual(len(env.state.estimates), 0)

    def test_counterfactual_adapter(self):
        regulator = CounterfactualRegulator.load(Path(self.tmp.name)/'counterfactual_weights.json')
        controls = regulator.actuation_context({'load': .9, 'uncertainty': .8})
        self.assertIn(controls['strategy'], ACTIONS)
        self.assertEqual(set(controls['predicted_consequences']), set(ACTIONS))
        self.assertTrue(0 <= controls['exploration'] <= 1)

    def test_split_overlap_rejected(self):
        config = dict(self.config)
        config['test_seeds'] = config['training_seeds']
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): run_benchmark(directory, config)

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'retained.json'
            marker.write_text('prior evidence')
            with self.assertRaisesRegex(ValueError,'preserve prior experiments'):
                run_benchmark(directory,self.config)
            self.assertEqual(marker.read_text(),'prior evidence')

    def test_reproducible_procedural_trace(self):
        _, a = trial('hybrid', 'engineering', 30, 12, self.model, self.ordinary)
        _, b = trial('hybrid', 'engineering', 30, 12, self.model, self.ordinary)
        self.assertEqual(a, b)

if __name__ == '__main__': unittest.main()
