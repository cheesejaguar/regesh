"""Live-runner protocol checks with a fake adapter; no LLM performance evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from regesh.learning import DirectValuePredictor, RidgePredictor
from regesh.live_benchmark import RecordedCallsModel, parse_design, run_live_benchmark
from regesh.providers import ModelResult, ProviderError


class FakeDesignModel:
    identifier = 'deterministic-test-adapter; not a live model'
    model = 'fake-design-v1'

    def __init__(self, mixed=False, invalid=False):
        self.calls = []
        self.mixed, self.invalid = mixed, invalid

    def complete(self, messages, *, seed=0, max_tokens=192, recorder=None):
        self.calls.append({'messages': messages, 'seed': seed, 'max_tokens': max_tokens})
        request = {'messages': messages, 'seed': seed, 'max_tokens': max_tokens}
        if self.mixed and seed % 1000 == 2:
            record = {'request': request, 'error_type': 'TestProviderFailure'}
            if recorder:
                recorder('model_call', record)
            raise ProviderError('test failure', record)
        text = ('not JSON' if self.invalid or (self.mixed and seed % 1000 == 1)
                else '{"design":[1,1,2,1,1,0]}')
        record = {'request': request, 'response': {'usage': {'prompt_tokens': 123, 'completion_tokens': 20}},
                  'raw_response': text, 'mode': 'fake deterministic adapter'}
        if recorder:
            recorder('model_call', record)
        return ModelResult(text, record)


class LiveProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.weights = self.root/'weights'
        self.weights.mkdir()
        metadata = {'training_seeds': [100], 'training_families': ['software','engineering']}
        RidgePredictor(np.zeros((72,4)), metadata=metadata).save(self.weights/'counterfactual_weights.json')
        DirectValuePredictor(np.zeros(288), metadata=metadata).save(self.weights/'ordinary_weights.json')

    def tearDown(self):
        self.temp.cleanup()

    def run_fake(self, model, **kwargs):
        return run_live_benchmark(model, self.root/'run', weights_dir=self.weights,
                                  seeds=[800], families=['scientific'], budget=4, **kwargs)

    def test_strict_schema_and_no_silent_repair(self):
        self.assertEqual(parse_design('{"design":[0,1,2,3,2,1]}'), (0,1,2,3,2,1))
        for text in ('{"design":[true,1,2,3,2,1]}', '{"design":[0.0,1,2,3,2,1]}',
                     '{"design":[4,1,2,3,2,1]}', '{"design":[0,1]}',
                     '{"design":[0,1,2,3,2,1],"reason":"ok"}', '```json\n{}\n```'):
            with self.subTest(text=text), self.assertRaises((ValueError,TypeError)):
                parse_design(text)

    def test_errors_and_malformed_outputs_consume_budget_and_are_recorded(self):
        model, recorded = FakeDesignModel(mixed=True), []
        summary = self.run_fake(model, conditions=['hybrid','ordinary_adaptive'],
                                recorder=lambda kind, record: recorded.append((kind,record)))
        rows = [json.loads(x) for x in (self.root/'run/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(model.calls), 8)
        self.assertEqual(len(recorded), 8)
        self.assertEqual(summary['model_calls'], 8)
        for row in rows:
            self.assertEqual(row['model_calls'], 4)
            self.assertEqual(row['probe_slots'], 4)
            self.assertEqual(row['valid_attempts'], 2)
            self.assertEqual(row['invalid_attempts'], 2)
            self.assertEqual(row['provider_errors'], 1)
            self.assertIsNone(row['reported_prompt_tokens'])
        calls = [json.loads(x) for x in (self.root/'run/model_calls.jsonl').read_text().splitlines()]
        self.assertEqual(calls[1]['text'], 'not JSON')
        self.assertEqual(calls[2]['error'], 'provider_error')
        self.assertIn('raw_response', calls[0]['model_records'][0]['record'])
        traces = [json.loads(x) for x in (self.root/'run/traces.jsonl').read_text().splitlines()]
        self.assertIsNone(traces[1]['design'])
        self.assertTrue(traces[2]['revised'])
        self.assertIn('injected adapter', summary['execution_mode'])

    def test_no_label_clone_has_identical_prompt_information_and_results(self):
        model = FakeDesignModel()
        self.run_fake(model, conditions=['hybrid','unlabelled_hybrid'])
        self.assertEqual(model.calls[:4], model.calls[4:])
        rows = [json.loads(x) for x in (self.root/'run/results.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['quality'], rows[1]['quality'])
        self.assertEqual(rows[0]['final_design'], rows[1]['final_design'])
        payload = json.loads(model.calls[0]['messages'][1]['content'])
        self.assertIn('scoring_implementation', payload)
        self.assertIn('reference_corpus', payload)
        self.assertIn('signed_error_average', payload['operational_context'])

    def test_all_invalid_is_zero_quality_without_rewarded_fallback(self):
        self.run_fake(FakeDesignModel(invalid=True), conditions=['memory_planning'])
        row = json.loads((self.root/'run/results.jsonl').read_text())
        self.assertEqual(row['quality'], 0)
        self.assertIsNone(row['final_design'])
        self.assertFalse(row['benchmark_original'])
        self.assertEqual(row['actual_valid_design_probes'], 0)

    def test_recorded_replay_consumes_no_live_provider_and_reproduces_outputs(self):
        self.run_fake(FakeDesignModel(mixed=True), conditions=['hybrid'])
        recorded = RecordedCallsModel(self.root/'run/model_calls.jsonl')
        summary = run_live_benchmark(recorded, self.root/'replay', weights_dir=self.weights,
                                    seeds=[800], families=['scientific'], budget=4,
                                    conditions=['hybrid'])
        self.assertEqual(recorded.position, 4)
        original = json.loads((self.root/'run/results.jsonl').read_text())
        replay = json.loads((self.root/'replay/results.jsonl').read_text())
        self.assertEqual(original['final_design'], replay['final_design'])
        self.assertEqual(original['quality'], replay['quality'])
        self.assertEqual(summary['execution_mode'], 'recorded response replay')

    def test_preserves_existing_output(self):
        self.run_fake(FakeDesignModel(), conditions=['hybrid'])
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            self.run_fake(FakeDesignModel(), conditions=['hybrid'])

    def test_fixed_seed_partition_and_budget_validation(self):
        for seeds in ([100], [800,800], []):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                run_live_benchmark(FakeDesignModel(), self.root/'run', weights_dir=self.weights,
                                   seeds=seeds, families=['software'], conditions=['hybrid'])
        with self.assertRaises(ValueError):
            self.run_fake(FakeDesignModel(), conditions=['unknown'])


if __name__ == '__main__':
    unittest.main()
