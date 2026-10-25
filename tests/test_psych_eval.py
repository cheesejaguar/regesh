"""Scoring/robustness protocol fixtures, not psychological validation data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from regesh import Agent, Event
from regesh.providers import ModelResult, ProviderError
from regesh.psych_eval import (evaluate_answers,evaluate_history,make_variants,
                               parse_answer,run_qa_eval,validate_item)


def item(identifier='q'):
    return {'id':identifier,'dataset':'test-fixture','question':'Fixture only?',
            'choices':{'A':'one','B':'two'},'gold':'A',
            'source':{'url':'https://example.invalid/test-fixture','license':'test-only'}}


class PsychScoreChecks(unittest.TestCase):
    def test_perfect_answers_have_proper_scores(self):
        rows=[{**item(str(i)),'answer':'A','probabilities':{'A':1.,'B':0.}} for i in range(4)]
        result=evaluate_answers(rows,bootstrap_samples=20)
        self.assertEqual(result['accuracy'],1.)
        self.assertEqual(result['multiclass_brier'],0.)
        self.assertEqual(result['log_loss'],0.)
        self.assertEqual(result['expected_calibration_error'],0.)
        self.assertEqual(result['accuracy_ci95'],[1.,1.])
    def test_abstentions_and_errors_are_not_dropped(self):
        rows=[{**item('a'),'answer':'A','probabilities':{'A':1.,'B':0.}},
              {**item('b'),'answer':None,'probabilities':{'A':.5,'B':.5}},
              {**item('c'),'answer':None,'probabilities':None,'error':'invalid_answer_schema'}]
        result=evaluate_answers(rows,bootstrap_samples=20)
        self.assertAlmostEqual(result['accuracy'],1/3)
        self.assertAlmostEqual(result['answer_coverage'],1/3)
        self.assertAlmostEqual(result['probability_coverage'],2/3)
        self.assertEqual(result['abstentions'],1)
        self.assertEqual(result['invalid_or_failed'],1)
    def test_decision_and_probability_disagreement_is_visible(self):
        r={**item(),'answer':'B','probabilities':{'A':.9,'B':.1}}
        result=evaluate_answers([r],bootstrap_samples=20)
        self.assertEqual(result['accuracy'],0.)
        self.assertEqual(result['argmax_accuracy'],1.)
        self.assertEqual(result['answer_argmax_disagreement_rate'],1.)
        self.assertAlmostEqual(result['expected_calibration_error'],.1)
        self.assertAlmostEqual(result['argmax_expected_calibration_error'],.1)
    def test_miscalibrated_wrong_confidence_is_penalized(self):
        r={**item(),'answer':'B','probabilities':{'A':0.,'B':1.}}
        result=evaluate_answers([r],bootstrap_samples=20)
        self.assertEqual(result['multiclass_brier'],2.)
        self.assertEqual(result['expected_calibration_error'],1.)
        self.assertGreater(result['log_loss'],20.)
    def test_balanced_metrics_report_class_imbalance(self):
        rows=[]
        for i,gold in enumerate(['A']*9+['B']):
            rows.append({**item(str(i)),'gold':gold,'answer':'A','probabilities':{'A':1.,'B':0.}})
        result=evaluate_answers(rows,bootstrap_samples=20)
        self.assertEqual(result['accuracy'],.9)
        self.assertEqual(result['balanced_accuracy'],.5)
        self.assertLess(result['macro_f1'],result['accuracy'])
    def test_macro_f1_includes_predicted_only_class(self):
        rows=[{**item(str(i)),'gold':'A','answer':answer,'probabilities':{'A':.5,'B':.5}}
              for i,answer in enumerate(['A','A','B'])]
        result=evaluate_answers(rows,bootstrap_samples=20)
        self.assertAlmostEqual(result['macro_f1'],.4)
        self.assertAlmostEqual(result['balanced_accuracy'],2/3)
        self.assertEqual(result['macro_f1_label_universe'],'gold labels union valid non-abstained predicted labels')

    def test_permutation_preserves_gold_meaning(self):
        base,changed=make_variants([item()])
        self.assertNotEqual(base['gold'],changed['gold'])
        self.assertEqual(base['choices'][base['gold']],changed['choices'][changed['gold']])
        for r in (base,changed):r.update(answer=r['gold'],probabilities={k:float(k==r['gold']) for k in r['choices']})
        result=evaluate_answers([base,changed],bootstrap_samples=20)
        self.assertEqual(result['independent_clusters'],1)
        self.assertEqual(result['paired_robustness']['permuted']['semantic_answer_invariance'],1.)
    def test_position_bias_is_not_semantic_consistency(self):
        rows=make_variants([item()])
        for r in rows:r.update(answer='A',probabilities={'A':1.,'B':0.})
        result=evaluate_answers(rows,bootstrap_samples=20)
        self.assertEqual(result['paired_robustness']['permuted']['semantic_answer_invariance'],0.)
        self.assertEqual(result['paired_robustness']['permuted']['joint_accuracy'],0.)
    def test_strict_probability_schema_and_no_boolean_labels(self):
        bad=['{"answer":"A","probabilities":{"A":true,"B":false}}',
             '{"answer":"A","probabilities":{"A":0.8,"B":0.8}}',
             '{"answer":"C","probabilities":{"A":1,"B":0}}',
             '{"answer":"A","probabilities":{"A":NaN,"B":0}}',
             '{"answer":"A","probabilities":{"A":1,"B":0},"reason":"prose"}']
        for value in bad:
            with self.subTest(value=value),self.assertRaises(ValueError):parse_answer(value,['A','B'])
    def test_source_and_unambiguous_choices_required(self):
        bad=item();bad.pop('source')
        with self.assertRaises(ValueError):validate_item(bad)
        bad=item();bad['choices']['B']='one'
        with self.assertRaises(ValueError):validate_item(bad)
    def test_invalid_response_consumes_call_and_is_retained(self):
        class Model:
            identifier='fixture'
            def __init__(self):self.calls=0
            def complete(self,messages,**kwargs):
                self.calls+=1;return ModelResult('not JSON',{'request':messages,'raw_response':'not JSON'})
        model=Model()
        with tempfile.TemporaryDirectory() as temp:
            result=run_qa_eval(model,[item()],temp,bootstrap_samples=20)
            self.assertEqual(model.calls,1)
            self.assertEqual(result['invalid_or_failed'],1)
            self.assertEqual(result['accuracy'],0.)
            record=json.loads(Path(temp,'responses.jsonl').read_text())
            self.assertEqual(record['response_text'],'not JSON')
            self.assertEqual(record['model_records'][0]['record']['raw_response'],'not JSON')
    def test_robustness_variants_share_generation_seed(self):
        class Model:
            def __init__(self):self.seeds=[]
            def complete(self,messages,**kwargs):
                self.seeds.append(kwargs['seed'])
                return ModelResult('{"answer":"A","probabilities":{"A":1,"B":0}}',{})
        model=Model()
        with tempfile.TemporaryDirectory() as temp:
            run_qa_eval(model,make_variants([item()],persona=True),temp,bootstrap_samples=20)
        self.assertEqual(len(set(model.seeds)),1)

    def test_provider_error_is_retained_without_retry(self):
        class Model:
            def complete(self,*args,**kwargs):raise ProviderError('fixture',{'http_status':503})
        with tempfile.TemporaryDirectory() as temp:
            result=run_qa_eval(Model(),[item()],temp,bootstrap_samples=20)
            self.assertEqual(result['model_calls'],1)
            row=json.loads(Path(temp,'responses.jsonl').read_text())
            self.assertEqual(row['error'],'provider_error')
            self.assertEqual(row['model_records'][0]['record']['http_status'],503)
    def test_output_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,'important').write_text('retained')
            with self.assertRaises(ValueError):run_qa_eval(object(),[item()],temp)
            self.assertEqual(Path(temp,'important').read_text(),'retained')
    def test_history_metrics_measure_recall_integrity_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            a=Agent(Path(temp)/'history.sqlite','test')
            a.observe(Event('e','checker','partner','goal','task','2026-10-25T00:00:00Z',data={'goal_delta':{'quality':-.8}}))
            a.record_outcome('e',-.8,partner_correct=True)
            a.retrieve(limit=1,reactivate=True);a.advance_time(500.)
            metrics=evaluate_history(a.trace());a.close()
            self.assertEqual(metrics['recall_evidence_leak_operations'],0)
            self.assertEqual(metrics['expectation_updates'],1)
            self.assertEqual(metrics['source_reliability_forecasts'],1)
            self.assertEqual(metrics['source_reliability_brier'],.25)
            self.assertLess(metrics['post_cessation_residual_ratio'],.01)
            self.assertTrue(metrics['finite'])
    def test_fabricated_learning_is_detected_by_history_measure(self):
        with tempfile.TemporaryDirectory() as temp:
            a=Agent(Path(temp)/'history.sqlite','test')
            a.observe(Event('e','checker','partner','goal','task','2026-10-25T00:00:00Z'))
            a.retrieve(limit=1,reactivate=True);trace=a.trace();a.close()
        changed=copy.deepcopy(trace)
        changed[-1]['state']['expectations']['task']={'n':1}
        self.assertEqual(evaluate_history(changed)['recall_evidence_leak_operations'],1)


if __name__=='__main__':unittest.main()
