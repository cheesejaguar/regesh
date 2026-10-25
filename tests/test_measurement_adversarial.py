"""Independent adversarial checks for measurement validity, not model ability."""
import importlib.util
import unittest
from regesh.psych_eval import evaluate_answers, make_variants, parse_answer


def row(identifier='q', **changes):
    value={'id':identifier,'pair_id':identifier,'variant':'base','gold':'A',
           'choices':{'A':'supported answer','B':'unsupported answer','C':'another answer'},
           'answer':'A','probabilities':{'A':.8,'B':.1,'C':.1},'dataset':'external-fixture',
           'source':{'url':'https://example.invalid/measurement-fixture'},'question':'Fixture only.'}
    value.update(changes)
    return value


class MeasurementAdversarialTests(unittest.TestCase):
    def test_decision_and_argmax_are_distinct_observables(self):
        result=evaluate_answers([row(answer='B',probabilities={'A':.6,'B':.1,'C':.3})],bootstrap_samples=10)
        self.assertEqual(result['accuracy'],0.)
        self.assertEqual(result['argmax_accuracy'],1.)
        self.assertEqual(result['answer_argmax_disagreement_rate'],1.)
        self.assertAlmostEqual(result['expected_calibration_error'],.1)
        self.assertAlmostEqual(result['argmax_expected_calibration_error'],.4)

    def test_invalid_attempts_cannot_disappear_from_accuracy_or_coverage(self):
        result=evaluate_answers([row(),row('failed',answer=None,probabilities=None,error='invalid_schema')],bootstrap_samples=10)
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['answer_coverage'],.5)
        self.assertEqual(result['probability_coverage'],.5)
        self.assertEqual(result['invalid_or_failed'],1)

    def test_consistent_wrong_answer_is_not_joint_correctness(self):
        first=row(answer='B')
        second=row('q:permuted',pair_id='q',variant='permuted',
                   choices={'A':'unsupported answer','B':'another answer','C':'supported answer'},
                   gold='C',answer='A',probabilities={'A':.8,'B':.1,'C':.1})
        result=evaluate_answers([first,second],bootstrap_samples=10)
        paired=result['paired_robustness']['permuted']
        self.assertEqual(paired['semantic_answer_invariance'],1.)
        self.assertEqual(paired['joint_accuracy'],0.)
        self.assertEqual(result['independent_clusters'],1)

    def test_abstention_distribution_is_not_an_answer(self):
        result=evaluate_answers([row(answer=None)],bootstrap_samples=10)
        self.assertEqual(result['accuracy'],0.)
        self.assertEqual(result['answer_coverage'],0.)
        self.assertEqual(result['probability_coverage'],1.)
        self.assertIsNone(result['expected_calibration_error'])
        self.assertIsNotNone(result['multiclass_brier'])

    def test_duplicate_semantic_choices_are_rejected_before_permutation(self):
        with self.assertRaises(ValueError):
            make_variants([row(choices={'A':'same','B':'same'},gold='A')])

    def test_unsupported_nonfinite_confidence_rejected(self):
        for text in ('{"answer":"A","probabilities":{"A":NaN,"B":0,"C":0}}',
                     '{"answer":"A","probabilities":{"A":true,"B":0,"C":0}}'):
            with self.assertRaises(ValueError):parse_answer(text,('A','B','C'))
