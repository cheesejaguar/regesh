"""Frozen supplemental axes for completed studies; never invokes a model.

Definitions were fixed before inspecting real correctness outputs. Strict primary
QA scoring is retained; conservative extractable choices are separate diagnostics.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import numpy as np
from .psych_eval import evaluate_answers
from .schema import canonical

DEFINITIONS_FROZEN_UTC='2026-10-25T16:29:00Z'
BOOTSTRAP_SEED=9041
BOOTSTRAP_SAMPLES=2000


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def _mean(values):
    values=[float(v) for v in values if _finite(v)]
    return float(np.mean(values)) if values else None


def _describe(values):
    values=[float(v) for v in values if _finite(v)]
    if not values:return {'n':0,'mean':None,'median':None,'q25':None,'q75':None}
    return {'n':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),
            'q25':float(np.quantile(values,.25)),'q75':float(np.quantile(values,.75))}


def _fmt(value):
    return f'{value:.4f}' if _finite(value) else 'NA'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl_path(path):
    path=Path(path)
    if path.is_file():return path
    compressed=Path(str(path)+'.gz')
    if compressed.is_file():return compressed
    raise FileNotFoundError(path)


def read_jsonl(path):
    path=_jsonl_path(path)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _confidence_metrics(rows):
    answered=[]
    for r in rows:
        p=r.get('confidence')
        if p is not None:
            if not _finite(p) or not 0<=p<=1:raise ValueError('invalid software confidence')
            answered.append((float(p),bool(r['evaluation']['all_passed'])))
    bins=[];ece=0.
    for i in range(10):
        values=[(p,y) for p,y in answered if min(int(p*10),9)==i]
        if values:
            p=float(np.mean([p for p,_ in values]));y=float(np.mean([y for _,y in values]))
            bins.append({'lower':i/10,'upper':(i+1)/10,'n':len(values),'confidence':p,'all_tests_success':y})
            ece+=len(values)/len(answered)*abs(p-y)
    return {'coverage':len(answered)/len(rows),
            'brier_all_tests':_mean([(p-y)**2 for p,y in answered]),
            'expected_calibration_error':ece if answered else None,'bins':bins,
            'scope':'self-reported full-suite success probability; ten-bin small-sample descriptive calibration'}


def software_axes(rows,traces,config):
    """Source-schema-only supplemental definitions; all failures remain counted."""
    if not rows:raise ValueError('empty software study')
    conditions=config['conditions']
    expected={(task,seed,condition) for task in config['tasks'] for seed in config['seeds'] for condition in conditions}
    keys=[(r['task'],r['seed'],r['condition']) for r in rows]
    if len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('software trials incomplete or duplicated')
    trace_map={}
    for trace in traces:
        key=(trace['task'],trace['seed'],trace['condition'],trace['attempt'])
        if key in trace_map:raise ValueError('duplicate software trace')
        trace_map[key]=trace
    expected_traces={(*key,attempt) for key in expected for attempt in range(config['attempts'])}
    if set(trace_map)!=expected_traces:raise ValueError('software action traces incomplete')
    result={'trials':len(rows),'conditions':{},'action_disagreement':{},
            'scope':'supplemental descriptive axes; primary task-cluster endpoint remains unchanged'}
    for condition in conditions:
        subset=[r for r in rows if r['condition']==condition]
        revisions=[];repairs=regressions=case_repairs=case_regressions=case_pairs=0
        complete_revision=0
        for r in subset:
            first=r.get('retrospective_evaluation');last=r['evaluation']
            if first is None:continue
            complete_revision+=1
            revisions.append(last['pass_rate']-first['pass_rate'])
            repairs+=not first['all_passed'] and last['all_passed']
            regressions+=first['all_passed'] and not last['all_passed']
            first_cases={x['case']:bool(x['passed']) for x in first.get('cases',[])}
            last_cases={x['case']:bool(x['passed']) for x in last.get('cases',[])}
            if first_cases.keys()!=last_cases.keys():raise ValueError('revision suites differ')
            for key in first_cases:
                case_pairs+=1
                case_repairs+=not first_cases[key] and last_cases[key]
                case_regressions+=first_cases[key] and not last_cases[key]
        correct=[r for r in subset if r['evaluation']['all_passed']]
        token_rows=[r for r in subset if type(r.get('prompt_tokens')) is int and r['prompt_tokens']>=0
                    and type(r.get('completion_tokens')) is int and r['completion_tokens']>=0]
        tokens=sum(r['prompt_tokens']+r['completion_tokens'] for r in token_rows)
        completions=sum(r['completion_tokens'] for r in token_rows)
        quality=sum(r['final_hidden_pass_rate'] for r in token_rows)
        errors=Counter();attempt_errors=Counter();functional_wrong=0
        for r in subset:
            if r.get('source_error'):errors['source:'+r['source_error']]+=1
            for case in r['evaluation'].get('cases',[]):
                if case.get('error'):errors['case:'+str(case['error'])]+=1
                elif not case['passed']:functional_wrong+=1
        for t in traces:
            if t['condition']==condition and t.get('error'):attempt_errors[t['error']]+=1
        valid=[r for r in subset if r['evaluation']['format_valid']]
        statics=[r['evaluation']['static'] for r in valid]
        result['conditions'][condition]={
            'trials':len(subset),
            'hidden_pass_fraction_mean':_mean([r['final_hidden_pass_rate'] for r in subset]),
            'full_suite_success_fraction':_mean([float(r['evaluation']['all_passed']) for r in subset]),
            'revision':{'coverage':complete_revision/len(subset),'mean_same_hidden_delta':_mean(revisions),
                'improved_fraction':sum(x>0 for x in revisions)/complete_revision if complete_revision else None,
                'tied_fraction':sum(x==0 for x in revisions)/complete_revision if complete_revision else None,
                'worsened_fraction':sum(x<0 for x in revisions)/complete_revision if complete_revision else None,
                'complete_repairs':repairs,'complete_regressions':regressions,
                'paired_case_observations':case_pairs,'repaired_cases':case_repairs,'regressed_cases':case_regressions},
            'correct_only_efficiency':{'eligible_trials':len(correct),'eligible_fraction':len(correct)/len(subset),
                **{key:_describe([r['evaluation'].get(key) for r in correct]) for key in
                   ('line_events','peak_working_bytes','seconds','fourfold_input_line_ratio')},
                'qualification':'Conditional on full correctness; sets of tasks may differ. Tracing/output included; no asymptotic proof.'},
            'realized_cost':{'reported_token_coverage':len(token_rows)/len(subset),'reported_prompt_tokens':sum(r['prompt_tokens'] for r in token_rows),
                'reported_completion_tokens':completions,'reported_total_tokens':tokens,
                'mean_total_tokens_reported_subset':tokens/len(token_rows) if token_rows else None,
                'hidden_pass_fraction_units_per_1000_total_tokens':1000*quality/tokens if tokens else None,
                'hidden_pass_fraction_units_per_1000_completion_tokens':1000*quality/completions if completions else None,
                'model_calls':sum(r['model_calls'] for r in subset),'feedback_tool_calls':sum(r['feedback_tool_calls'] for r in subset),
                'final_checker_calls':sum(r['final_checker_calls'] for r in subset),
                'retrospective_checker_calls':sum(r.get('retrospective_checker_calls',0) for r in subset),
                'requested_completion_token_ceiling':sum(r['requested_completion_token_ceiling'] for r in subset),
                'trial_latency_seconds':_describe([r['seconds'] for r in subset]),
                'qualification':'Equal ceilings do not imply equal actual tokens; token-normalized ratios are descriptive across heterogeneous tasks.'},
            'errors':{'final_source_or_case_counts':dict(errors),'attempt_protocol_error_counts':dict(attempt_errors),
                'functional_wrong_cases_without_runtime_error':functional_wrong,
                'resource_execution_compliance_fraction':_mean([float(r['evaluation']['resource_limit_compliant']) for r in subset]),
                'qualification':'Overlapping categories and unequal case counts; not a summed score.'},
            'confidence':_confidence_metrics(subset),
            'implementation_proxies':{'valid_source_fraction':len(valid)/len(subset),
                'distinct_ast_hashes':len({s['ast_sha256'] for s in statics if s.get('ast_sha256')}),
                'distinct_operation_signatures':len({tuple(s.get('operation_signature',[])) for s in statics}),
                **{k:_describe([s.get(k) for s in statics]) for k in ('source_characters','source_lines','ast_nodes','cyclomatic_proxy')},
                'qualification':'AST/signature diversity and complexity are surface/implementation proxies, not useful research novelty.'}}
    if len(conditions)==2:
        a,b=conditions;different=0;per_attempt={};pairs=0
        for attempt in range(config['attempts']):
            diffs=0;n=0
            for task in config['tasks']:
                for seed in config['seeds']:
                    n+=1
                    diffs+=trace_map[(task,seed,a,attempt)]['action']!=trace_map[(task,seed,b,attempt)]['action']
            pairs+=n;different+=diffs
            per_attempt[str(attempt)]={'complete_pairs':n,'different_actions':diffs,'disagreement_rate':diffs/n}
        result['action_disagreement']={'paired_action_opportunities':pairs,'different_actions':different,
            'disagreement_rate':different/pairs,'complete_pair_coverage':1.,'per_attempt':per_attempt,
            'qualification':'Treatment-strength diagnostic; later histories may differ because of prior actions. No mediation estimate.'}
    points={}
    for condition,value in result['conditions'].items():
        if value['realized_cost']['reported_token_coverage']==1.:
            points[condition]={'hidden_pass_fraction':value['hidden_pass_fraction_mean'],
                'full_suite_success':value['full_suite_success_fraction'],
                'mean_total_tokens':value['realized_cost']['mean_total_tokens_reported_subset']}
    dominated={name:[] for name in points}
    for name,p in points.items():
        for other,q in points.items():
            if other==name:continue
            weak=q['hidden_pass_fraction']>=p['hidden_pass_fraction'] and q['full_suite_success']>=p['full_suite_success'] and q['mean_total_tokens']<=p['mean_total_tokens']
            strict=q['hidden_pass_fraction']>p['hidden_pass_fraction'] or q['full_suite_success']>p['full_suite_success'] or q['mean_total_tokens']<p['mean_total_tokens']
            if weak and strict:dominated[name].append(other)
    result['descriptive_pareto']={'complete_token_conditions':list(points),'points':points,
        'dominated_by':dominated,'frontier':[n for n in points if not dominated[n]],
        'qualification':'Descriptive means only, identical task/seed coverage required; no uncertainty or scalar superiority claim.'}
    return result


class _PairsObject(dict):
    def __init__(self,pairs):
        super().__init__(pairs);self.pairs=pairs


def extract_choice(text,labels,*,allow_fence=False):
    """No gold argument, probability repair, prose guessing or truncation repair."""
    if not isinstance(text,str):return {'answer':None,'status':'no_response','wrapper':False}
    raw=text.strip();wrapped=False
    if raw.startswith('```'):
        if not allow_fence:return {'answer':None,'status':'fence_not_allowed','wrapper':False}
        lines=raw.splitlines()
        if len(lines)<3 or lines[0] not in {'```','```json'} or lines[-1]!='```':
            return {'answer':None,'status':'invalid_fence','wrapper':False}
        raw='\n'.join(lines[1:-1]);wrapped=True
    try:
        value=json.loads(raw,object_pairs_hook=_PairsObject,
                         parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (ValueError,TypeError):return {'answer':None,'status':'invalid_json','wrapper':wrapped}
    if not isinstance(value,dict):return {'answer':None,'status':'not_object','wrapper':wrapped}
    if sum(k=='answer' for k,_ in value.pairs)>1:
        return {'answer':None,'status':'duplicate_answer','wrapper':wrapped}
    if 'answer' not in value:return {'answer':None,'status':'missing_answer','wrapper':wrapped}
    if value['answer'] is None:return {'answer':None,'status':'abstention','wrapper':wrapped}
    if type(value['answer']) is not str or value['answer'] not in labels:
        return {'answer':None,'status':'illegal_answer','wrapper':wrapped}
    return {'answer':value['answer'],'status':'choice','wrapper':wrapped}


def _choice_summary(rows,extracted):
    n=len(rows);answered=[i for i,x in enumerate(extracted) if x['answer'] is not None]
    hits=[bool(x['answer']==r['gold']) for r,x in zip(rows,extracted)]
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[str(r.get('pair_id',r['id']))].append(i)
    blocks=list(groups.values());rng=np.random.default_rng(BOOTSTRAP_SEED);boot=[]
    for _ in range(BOOTSTRAP_SAMPLES):
        sample=[i for j in rng.integers(len(blocks),size=len(blocks)) for i in blocks[j]]
        boot.append(float(np.mean([hits[i] for i in sample])))
    label_set=sorted({k for r in rows for k in r['choices']})
    answer_counts=Counter(extracted[i]['answer'] for i in answered)
    gold_counts=Counter(rows[i]['gold'] for i in answered)
    answer_frequencies={k:answer_counts[k]/len(answered) for k in label_set} if answered else None
    gold_frequencies={k:gold_counts[k]/len(answered) for k in label_set} if answered else None
    position={'choices_on_answered_subset':answer_frequencies,'gold_on_same_answered_subset':gold_frequencies,
              'total_variation_distance':.5*sum(abs(answer_frequencies[k]-gold_frequencies[k]) for k in label_set) if answered else None,
              'gold_frequencies_all_attempts':{k:sum(r['gold']==k for r in rows)/n for k in label_set},
              'choice_frequencies_all_attempts':{k:answer_counts[k]/n for k in label_set},
              'qualification':'Option labels are positions, not emotions. Difficulty and extraction coverage can confound preference.'}
    return {'attempts':n,'extractable_choice_coverage':len(answered)/n,'accuracy_all_attempts':sum(hits)/n,
            'accuracy_conditional_on_extraction':sum(hits)/len(answered) if answered else None,
            'accuracy_ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            'independent_question_clusters':len(blocks),'parse_status_counts':dict(Counter(x['status'] for x in extracted)),
            'wrapped_choices':sum(x['answer'] is not None and x['wrapper'] for x in extracted),
            'primary_rejected_but_choice_extractable':sum(x['answer'] is not None and bool(r.get('error')) for r,x in zip(rows,extracted)),
            'position_diagnostics':position}


def psych_axes(rows):
    if not rows:raise ValueError('empty QA study')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate QA item')
    primary=evaluate_answers(rows,bootstrap_samples=BOOTSTRAP_SAMPLES,seed=BOOTSTRAP_SEED)
    result={'strict_primary_recomputed':primary,'secondary_diagnostics':{},'per_variant':{},
            'scope':'Keyed emotional-understanding knowledge and response reliability; not affect-controller causality, personality or experience.'}
    for allow_fence,name in ((False,'direct_completed_json_choice'),(True,'exact_json_fence_choice')):
        extracted=[extract_choice(r.get('response_text'),r['choices'],allow_fence=allow_fence) for r in rows]
        result['secondary_diagnostics'][name]=_choice_summary(rows,extracted)
    for variant in sorted({r.get('variant','base') for r in rows}):
        subset=[r for r in rows if r.get('variant','base')==variant]
        result['per_variant'][variant]={'strict_primary':evaluate_answers(subset,bootstrap_samples=BOOTSTRAP_SAMPLES,seed=BOOTSTRAP_SEED)}
        for allow_fence,name in ((False,'direct_completed_json_choice'),(True,'exact_json_fence_choice')):
            result['per_variant'][variant][name]=_choice_summary(subset,[extract_choice(r.get('response_text'),r['choices'],allow_fence=allow_fence) for r in subset])
    result['limitations']=['Strict primary errors remain errors; secondary extraction never rewrites them.',
        'No prose or truncated-JSON repair and no gold-directed parsing.',
        'Variants share source-question clusters and are not independent questions.',
        'Human-author keys can be culturally or normatively contingent; pretraining contamination is unknown.',
        'Small coding-specialized quantized-model scores may reflect reading or schema-following as well as social knowledge.']
    return result


def run_analysis(output_dir,*,software=None,psych=None):
    if software is None and psych is None:raise ValueError('provide at least one study')
    output=Path(output_dir)
    if output.exists() and any(output.iterdir()):raise ValueError('Output directory must be empty; preserve evidence')
    result={};inputs={}
    if software is not None:
        root=Path(software)
        if not (root/'summary.json').is_file():raise ValueError('software study has no completion marker')
        config=json.loads((root/'config.json').read_text())
        rows=read_jsonl(root/'results.jsonl');traces=read_jsonl(root/'traces.jsonl')
        result['software']=software_axes(rows,traces,config)
        for name in ('config.json','summary.json','results.jsonl','traces.jsonl'):
            p=_jsonl_path(root/name) if name.endswith('.jsonl') else root/name
            inputs[str(p)]=_sha(p)
    if psych is not None:
        root=Path(psych)
        if not (root/'summary.json').is_file():raise ValueError('QA study has no completion marker')
        config=json.loads((root/'config.json').read_text());rows=read_jsonl(root/'responses.jsonl')
        if len(rows)!=config['items']:raise ValueError('QA study incomplete')
        result['psych']=psych_axes(rows)
        original=json.loads((root/'summary.json').read_text())
        if original['attempts']!=len(rows) or not math.isclose(original['accuracy'],result['psych']['strict_primary_recomputed']['accuracy'],abs_tol=1e-12):
            raise ValueError('primary QA summary does not match raw attempts')
        for name in ('config.json','summary.json','responses.jsonl'):
            p=_jsonl_path(root/name) if name.endswith('.jsonl') else root/name
            inputs[str(p)]=_sha(p)
    output.mkdir(parents=True,exist_ok=True)
    for name,data in result.items():
        (output/(name+'_axes.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    manifest={'schema_version':1,'definitions_frozen_utc':DEFINITIONS_FROZEN_UTC,
        'analysis_executed_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':_sha(__file__),
        'input_sha256':inputs,'model_calls':0,'supplemental_only':True,
        'psych_scorer_sha256':_sha(Path(__file__).with_name('psych_eval.py')) if psych is not None else None,
        'python':platform.python_version(),'numpy':np.__version__,
        'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_samples':BOOTSTRAP_SAMPLES}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Supplemental multi-axis analysis','',
        'Definitions were fixed before inspecting real correctness outputs. Strict primary results remain authoritative; this analysis makes no model calls.','']
    if 'software' in result:
        value=result['software'];a=value['action_disagreement']
        lines+=['Software: '+str(value['trials'])+' complete trials.',
            f"Paired action disagreement: {a.get('different_actions')} / {a.get('paired_action_opportunities')} opportunities ({a.get('disagreement_rate'):.4f}).",'',
            '| Condition | Hidden pass fraction | Same-suite revision | Total reported tokens | Confidence coverage |',
            '|---|---:|---:|---:|---:|']
        for condition,v in value['conditions'].items():
            lines.append(f"| {condition} | {v['hidden_pass_fraction_mean']:.4f} | {_fmt(v['revision']['mean_same_hidden_delta'])} | {v['realized_cost']['reported_total_tokens']} | {v['confidence']['coverage']:.4f} |")
        lines+=['','Correct-only efficiency, error types, implementation proxies and a descriptive cost-quality frontier are in software_axes.json. No scalar creativity or psychological score is constructed.','']
    if 'psych' in result:
        value=result['psych'];p=value['strict_primary_recomputed'];base=value['per_variant'].get('base',{}).get('strict_primary',{})
        lines += [f"QA strict primary: {p['accuracy']:.4f} accuracy over {p['attempts']} attempts; base-only {base.get('accuracy')} over {base.get('attempts')} original questions.",'',
            '| Secondary diagnostic | All-attempt accuracy | Extractable coverage |',
            '|---|---:|---:|']
        for name,v in value['secondary_diagnostics'].items():
            lines.append(f"| {name} | {v['accuracy_all_attempts']:.4f} | {v['extractable_choice_coverage']:.4f} |")
        lines+=['','These diagnostic choices do not replace schema-primary failures. Confidence distribution quality and response validity are separate from extractable answer knowledge; neither establishes subjective emotion.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--software');parser.add_argument('--psych');parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run_analysis(args.output,software=args.software,psych=args.psych)
    print(json.dumps({'output':args.output,'studies':list(result),'model_calls':0},indent=2))


if __name__=='__main__':main()
