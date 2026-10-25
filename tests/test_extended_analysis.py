"""Definition tests for secondary analyses; no real model or live outcomes used."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from regesh.extended_analysis import extract_choice, psych_axes, software_axes, read_jsonl, run_analysis


def qa_row(identifier='q',**changes):
    value={'id':identifier,'pair_id':identifier,'variant':'base','dataset':'fixture',
        'question':'Fixture only','source':{'url':'https://example.invalid/fixture'},
        'choices':{'A':'supported','B':'unsupported'},'gold':'A',
        'answer':None,'probabilities':None,'error':'invalid_answer_schema',
        'response_text':'{"answer":"A","probabilities":{"A":0.8}}'}
    value.update(changes);return value


def software_fixture():
    config={'tasks':['task'],'seeds':[910,911],'conditions':['hybrid','ordinary_adaptive'],'attempts':2}
    rows=[];traces=[]
    for condition in config['conditions']:
        good=condition=='hybrid'
        for seed in config['seeds']:
            evaluation={'all_passed':good,'pass_rate':1. if good else .5,'format_valid':True,
                'resource_limit_compliant':True,'line_events':100 if good else 1,'peak_working_bytes':400,
                'seconds':.01,'fourfold_input_line_ratio':4.,
                'cases':[{'case':0,'passed':True,'error':None},{'case':1,'passed':good,'error':None}],
                'static':{'ast_sha256':'valid-'+condition,'operation_signature':['get'],
                    'source_characters':30,'source_lines':2,'ast_nodes':8,'cyclomatic_proxy':1}}
            first=deepcopy(evaluation);first['all_passed']=not good;first['pass_rate']=.5 if good else 1.
            first['cases'][1]['passed']=not good
            rows.append({'condition':condition,'task':'task','seed':seed,'evaluation':evaluation,
                'retrospective_evaluation':first,'final_hidden_pass_rate':evaluation['pass_rate'],
                'prompt_tokens':200 if good else 300,'completion_tokens':100,'model_calls':2,
                'feedback_tool_calls':1,'final_checker_calls':1,'retrospective_checker_calls':1,
                'requested_completion_token_ceiling':1024,'seconds':1.,'confidence':.8 if good else None,
                'source_error':None})
            for attempt in (0,1):
                traces.append({'condition':condition,'task':'task','seed':seed,'attempt':attempt,
                    'action':'verify' if good and attempt else 'refine','error':None})
    return rows,traces,config


class ConservativeChoiceTests(unittest.TestCase):
    def test_completed_choice_is_independent_of_probability_schema(self):
        parsed=extract_choice('{"answer":"B","probabilities":{"A":"invalid"},"extra":true}',('A','B'))
        self.assertEqual(parsed['answer'],'B')
        self.assertEqual(parsed['status'],'choice')

    def test_no_prose_guessing_or_truncated_json_repair(self):
        for text in ('The answer is A.','{"answer":"A"','Here is {"answer":"A"}',
                     '{"answer":"A"} trailing prose','[{"answer":"A"}]'):
            self.assertIsNone(extract_choice(text,('A','B'),allow_fence=True)['answer'])

    def test_exact_fence_is_a_separate_flagged_diagnostic(self):
        text='```json\n{"answer":"A"}\n```'
        self.assertIsNone(extract_choice(text,('A','B'))['answer'])
        parsed=extract_choice(text,('A','B'),allow_fence=True)
        self.assertEqual(parsed['answer'],'A');self.assertTrue(parsed['wrapper'])
        self.assertIsNone(extract_choice(text+' explanation',('A','B'),allow_fence=True)['answer'])

    def test_ambiguous_duplicate_answer_and_illegal_values_rejected(self):
        for text in ('{"answer":"A","answer":"B"}','{"answer":true}',
                     '{"answer":1}','{"answer":"C"}','{"answer":"A","p":NaN}'):
            self.assertIsNone(extract_choice(text,('A','B'))['answer'])
        self.assertEqual(extract_choice('{"answer":null}',('A','B'))['status'],'abstention')

    def test_secondary_score_cannot_rewrite_strict_primary_failure(self):
        row=qa_row();before=deepcopy(row);result=psych_axes([row])
        self.assertEqual(result['strict_primary_recomputed']['accuracy'],0.)
        secondary=result['secondary_diagnostics']['direct_completed_json_choice']
        self.assertEqual(secondary['accuracy_all_attempts'],1.)
        self.assertEqual(secondary['primary_rejected_but_choice_extractable'],1)
        self.assertEqual(row,before)
        self.assertEqual(result['per_variant']['base']['strict_primary']['attempts'],1)

    def test_all_attempt_denominator_and_position_bias_remain_visible(self):
        result=psych_axes([qa_row(),qa_row('failed',response_text=None,error='provider_error')])
        secondary=result['secondary_diagnostics']['direct_completed_json_choice']
        self.assertEqual(secondary['accuracy_all_attempts'],.5)
        self.assertEqual(secondary['extractable_choice_coverage'],.5)
        self.assertEqual(secondary['accuracy_conditional_on_extraction'],1.)
        self.assertEqual(secondary['position_diagnostics']['choice_frequencies_all_attempts']['A'],.5)


class SoftwareAxesTests(unittest.TestCase):
    def test_action_treatment_is_counted_at_paired_attempt_level(self):
        rows,traces,config=software_fixture();result=software_axes(rows,traces,config)
        value=result['action_disagreement']
        self.assertEqual(value['paired_action_opportunities'],4)
        self.assertEqual(value['different_actions'],2)
        self.assertEqual(value['disagreement_rate'],.5)
        self.assertEqual(value['per_attempt']['0']['disagreement_rate'],0.)

    def test_same_hidden_revision_exposes_repairs_and_regressions(self):
        rows,traces,config=software_fixture();result=software_axes(rows,traces,config)
        h=result['conditions']['hybrid']['revision'];o=result['conditions']['ordinary_adaptive']['revision']
        self.assertEqual(h['mean_same_hidden_delta'],.5);self.assertEqual(h['complete_repairs'],2)
        self.assertEqual(h['repaired_cases'],2);self.assertEqual(h['regressed_cases'],0)
        self.assertEqual(o['mean_same_hidden_delta'],-.5);self.assertEqual(o['complete_regressions'],2)
        self.assertEqual(o['regressed_cases'],2)

    def test_fast_wrong_program_is_not_efficiency_eligible(self):
        rows,traces,config=software_fixture();result=software_axes(rows,traces,config)
        efficiency=result['conditions']['ordinary_adaptive']['correct_only_efficiency']
        self.assertEqual(efficiency['eligible_trials'],0)
        self.assertIsNone(efficiency['line_events']['mean'])

    def test_token_normalization_and_pareto_are_explicit(self):
        rows,traces,config=software_fixture();result=software_axes(rows,traces,config)
        cost=result['conditions']['hybrid']['realized_cost']
        self.assertEqual(cost['reported_total_tokens'],600)
        self.assertAlmostEqual(cost['hidden_pass_fraction_units_per_1000_total_tokens'],1000*2/600)
        self.assertEqual(result['descriptive_pareto']['dominated_by']['ordinary_adaptive'],['hybrid'])

    def test_missing_usage_prevents_full_cost_frontier_comparison(self):
        rows,traces,config=software_fixture();rows[-1]['prompt_tokens']=None
        result=software_axes(rows,traces,config)
        self.assertEqual(result['conditions']['ordinary_adaptive']['realized_cost']['reported_token_coverage'],.5)
        self.assertNotIn('ordinary_adaptive',result['descriptive_pareto']['points'])

    def test_incomplete_or_duplicate_trials_and_actions_are_rejected(self):
        rows,traces,config=software_fixture()
        with self.assertRaisesRegex(ValueError,'trials incomplete'):software_axes(rows[:-1],traces,config)
        with self.assertRaisesRegex(ValueError,'duplicate software trace'):software_axes(rows,traces+[traces[0]],config)
        with self.assertRaisesRegex(ValueError,'traces incomplete'):software_axes(rows,traces[:-1],config)

    def test_revision_case_identity_must_match(self):
        rows,traces,config=software_fixture();rows[0]['retrospective_evaluation']['cases'][0]['case']=99
        with self.assertRaisesRegex(ValueError,'revision suites differ'):software_axes(rows,traces,config)


class ArtifactTests(unittest.TestCase):
    def test_plain_and_gzip_jsonl_are_interchangeable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'rows.jsonl';content=json.dumps(qa_row())+'\n';p.write_text(content)
            plain=read_jsonl(p)
            with gzip.open(str(p)+'.gz','wt') as stream:stream.write(content)
            p.unlink();self.assertEqual(read_jsonl(p),plain)

    def test_completed_psych_only_analysis_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study';root.mkdir();out=Path(tmp)/'analysis'
            (root/'config.json').write_text('{"items":1}')
            (root/'summary.json').write_text('{"attempts":1,"accuracy":0.0}')
            with gzip.open(root/'responses.jsonl.gz','wt') as stream:stream.write(json.dumps(qa_row())+'\n')
            result=run_analysis(out,psych=root)
            self.assertIn('psych',result);self.assertNotIn('software',result)
            self.assertEqual(json.loads((out/'manifest.json').read_text())['model_calls'],0)
            with self.assertRaisesRegex(ValueError,'preserve evidence'):run_analysis(out,psych=root)

    def test_partial_study_is_not_silently_analyzed_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'partial';root.mkdir()
            with self.assertRaisesRegex(ValueError,'completion marker'):run_analysis(Path(tmp)/'out',psych=root)
