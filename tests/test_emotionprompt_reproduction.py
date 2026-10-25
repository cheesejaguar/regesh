import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from regesh.providers import ModelResult
from regesh.reproductions.emotionprompt import analyze, exact_mcnemar, messages, normalize_prediction, run, score
from regesh.reproductions.emotionprompt_check import audit
from regesh.reproductions.emotionprompt_content import arithmetic_content


class EmotionPromptTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_prediction('  A. explanation\nB'), 'a')
        self.assertEqual(score('A.', 'a')['normalized_exact_match'], 1)
        self.assertEqual(score('The answer is a', 'a')['normalized_exact_match'], 0)
        self.assertEqual(score('```a```', 'a')['normalized_exact_match'], 1)

    def test_negative_number_benchmark_quirk_retained(self):
        self.assertEqual(normalize_prediction('-7'), ' 7')
        self.assertNotEqual(normalize_prediction('-7'), normalize_prediction('7'))

    def test_prompt_only_intervention(self):
        c = {'conditions': ['baseline', 'emotionprompt_EP02'], 'suffix': 'This is very important to my career.'}
        item = {'instruction': 'Sum the two given numbers.', 'input': '2 3'}
        a = messages(item, 'baseline', c)[0]['content']; b = messages(item, 'emotionprompt_EP02', c)[0]['content']
        self.assertEqual(b.replace(' ' + c['suffix'], ''), a)
        with self.assertRaises(ValueError): messages(item, 'wrong', c)

    def test_mcnemar(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.)
        self.assertEqual(exact_mcnemar(0, 5), .0625)
        self.assertEqual(exact_mcnemar(1, 1), 1.)

    def test_exploratory_arithmetic_checks_operands_and_operator(self):
        item = {'task': 'diff', 'input': '138 48', 'gold': '90'}
        self.assertTrue(arithmetic_content('138 - 48 = 90', item)['correct'])
        self.assertTrue(arithmetic_content('90', item)['correct'])
        self.assertFalse(arithmetic_content('100 - 10 = 90', item)['correct'])
        self.assertFalse(arithmetic_content('138 + 48 = 90', item)['correct'])
        self.assertFalse(arithmetic_content('The answer is 90', item)['correct'])
        self.assertFalse(arithmetic_content('```python\nprint(90)\n```', item)['correct'])

    def test_complete_pairs_and_zero_delta(self):
        c = {'items': 2, 'calls': 4, 'conditions': ['baseline', 'emotionprompt_EP02'], 'seed': 7,
            'bootstrap_samples': 20, 'published_reference': {}}
        rows = [{'id': str(i), 'task': str(i), 'condition': condition, 'normalized_exact_match': i,
            'strict_exact_match': i, 'usage': {'completion_tokens': 2}} for i in range(2) for condition in c['conditions']]
        result = analyze(rows, c)
        self.assertEqual(result['effect']['paired_within_task_bootstrap_95ci'], [0., 0.])
        self.assertEqual(result['conditions']['baseline']['actual_tokens']['completion_tokens'], 4)
        with self.assertRaises(ValueError): analyze(rows[:-1], c)
        with self.assertRaises(ValueError): analyze(rows + [rows[0]], c)

    def test_locked_items_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); cfg = d / 'config.json'; items = d / 'items.jsonl'; out = d / 'out'
            cfg.write_text(json.dumps({'items_sha256': 'wrong'})); items.write_text('{}\n')
            with self.assertRaises(ValueError): run(cfg, items, out, None)

    def test_recorded_end_to_end_with_fixture(self):
        class Model:
            model = 'unit-test-fixture'
            def complete(self, prompt, *, seed, max_tokens):
                response = {'choices': [{'message': {'content': 'a'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 1}}
                return ModelResult('a', {'request': {'messages': prompt, 'seed': seed, 'max_tokens': max_tokens, 'temperature': 0},
                    'response': response, 'raw_response': json.dumps(response), 'elapsed_seconds': 0.})
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); cfg = d / 'config.json'; items = d / 'items.jsonl'; out = d / 'out'
            item = {'id': 'first:1', 'task': 'first', 'instruction': 'Extract the first letter.',
                'input': 'ant', 'gold': 'a'}
            items.write_text(json.dumps(item) + '\n')
            metric = d / 'fixture_metric.py'
            metric.write_text('def get_em_score(prediction, gold):\n    return prediction == gold\n')
            c = {'items': 1, 'calls': 2, 'conditions': ['baseline', 'emotionprompt_EP02'], 'seed': 7,
                'bootstrap_samples': 10, 'published_reference': {}, 'max_tokens': 3, 'suffix': 'A stimulus.',
                'items_sha256': hashlib.sha256(items.read_bytes()).hexdigest(), 'temperature': 0,
                'source_evaluate_sha256': hashlib.sha256(metric.read_bytes()).hexdigest()}
            cfg.write_text(json.dumps(c))
            summary = run(cfg, items, out, Model())
            self.assertEqual(summary['conditions']['baseline']['normalized_exact_match'], 1.)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertTrue(manifest['complete']); self.assertEqual(manifest['model_calls_completed'], 2)
            self.assertEqual(len((out / 'calls.jsonl').read_text().splitlines()), 2)
            self.assertEqual(audit(out, metric)['model_calls'], 0)
            with self.assertRaises(ValueError): run(cfg, items, out, Model())
            (out / 'rows.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError): audit(out, metric)


if __name__ == '__main__':
    unittest.main()
