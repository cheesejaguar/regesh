"""Provider-free reconstruction of a recorded software-controller experiment.

Actor replay consumes recorded model responses and PUBLIC feedback only. Optional
independent checker re-execution starts after every actor trajectory is verified.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import hashlib
import types
import json
from pathlib import Path
import sys
from .learning import RidgePredictor, DirectValuePredictor
from .schema import canonical


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_blob(path):
    path=Path(path)
    if path.exists():return path.read_bytes()
    with gzip.open(str(path)+'.gz','rb') as stream:return stream.read()


def records(directory,name):
    path=Path(directory)/name
    if path.exists():
        with path.open() as stream:return [json.loads(line) for line in stream if line.strip()]
    path=Path(str(path)+'.gz')
    with gzip.open(path,'rt') as stream:return [json.loads(line) for line in stream if line.strip()]


def read_json(directory,name):
    path=Path(directory)/name
    if path.exists():return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz','rt') as stream:return json.load(stream)


def require_equal(observed,expected,label):
    if canonical(observed)!=canonical(expected):raise ValueError('Replay mismatch: '+label)


def frozen_module(directory):
    """Load repository-owned source captured at execution, after manifest checks."""
    directory=Path(directory);manifest=read_json(directory,'source_manifest.json')
    for name,expected in manifest.items():
        path=directory/('source_'+name)
        if hashlib.sha256(read_blob(path)).hexdigest()!=expected:raise ValueError('Frozen source integrity failure: '+name)
    # The archived evaluator imports the task definitions from the installed
    # package. Require exact task identity, never silently replay changed tasks.
    from . import software_tasks
    if file_sha(software_tasks.__file__)!=manifest['software_tasks.py']:
        raise ValueError('Installed task source differs from frozen task source')
    source=directory/'source_software_eval.py'
    body=read_blob(source)
    name='regesh._software_replay_frozen_'+hashlib.sha256(body).hexdigest()[:16]
    module=types.ModuleType(name);module.__file__=str(source);module.__package__='regesh';sys.modules[name]=module
    exec(compile(body,str(source),'exec'),module.__dict__)
    return module,manifest


def stable_evaluation(value):
    """Timing/allocator details may vary; functional and line counts should not."""
    if isinstance(value,list):return [stable_evaluation(v) for v in value]
    if isinstance(value,dict):return {k:stable_evaluation(v) for k,v in value.items()
        if k not in {'seconds','peak_bytes','peak_working_bytes'}}
    return value


def replay_software_experiment(run_dir,output_dir,*,weights_dir='results/benchmark',recheck=False):
    directory=Path(run_dir);output=Path(output_dir)
    if output.exists() and any(output.iterdir()):raise ValueError('Output directory must be empty; preserve replay evidence')
    cfg=read_json(directory,'config.json');module,manifest=frozen_module(directory)
    weights=Path(weights_dir);counter_path=weights/'counterfactual_weights.json';ordinary_path=weights/'ordinary_weights.json'
    counter_blob=read_blob(counter_path);ordinary_blob=read_blob(ordinary_path)
    if hashlib.sha256(counter_blob).hexdigest()!=cfg['counterfactual_sha256'] or hashlib.sha256(ordinary_blob).hexdigest()!=cfg['ordinary_sha256']:
        raise ValueError('Frozen weight integrity failure')
    cw=json.loads(counter_blob);ow=json.loads(ordinary_blob)
    counter=RidgePredictor(cw['weights'],cw['ridge'],cw['metadata']);ordinary=DirectValuePredictor(ow['weights'],ow['ridge'],ow['metadata'])
    calls=records(directory,'calls.jsonl');traces=records(directory,'traces.jsonl');results=records(directory,'results.jsonl')
    if len(calls)!=len(traces) or len(calls)!=len(results)*cfg['attempts']:raise ValueError('Incomplete recorded trial/call counts')
    def trial_key(r):return r['task'],r['seed'],r['condition']
    def call_key(r):return (*trial_key(r),r['attempt'])
    call_map={call_key(r):r for r in calls};trace_map={call_key(r):r for r in traces}
    if len(call_map)!=len(calls) or len(trace_map)!=len(traces):raise ValueError('Duplicate recorded call identity')
    result_map={trial_key(r):r for r in results}
    expected_trials={(t,s,c) for t in cfg['tasks'] for s in cfg['seeds'] for c in cfg['conditions']}
    if set(result_map)!=expected_trials or len(result_map)!=len(results):raise ValueError('Missing or repeated trial identity')
    verified=[];stored_sources=[];phase_rows=[]
    for result in results:
        task,seed,condition=trial_key(result);state=module.TrialState()
        require_equal(asdict(state),result['initial_snapshot'],f'{task}/{seed}/{condition}: initial snapshot')
        require_equal(result['weights'],{'counterfactual_sha256':cfg['counterfactual_sha256'],
            'ordinary_sha256':cfg['ordinary_sha256']},'per-trial weight references')
        sources=[];diagnostic_rate=None;last_error=None;usage_rows=[];last_static={}
        for attempt in range(cfg['attempts']):
            key=task,seed,condition,attempt
            call=call_map[key];trace=trace_map[key]
            action,reason=module.choose(condition,state,cfg['attempts'],counter,ordinary)
            require_equal(action,trace['action'],f'{key}: chosen strategy')
            require_equal(reason,trace['structured_reason'],f'{key}: predictions/scores/context')
            require_equal(module.messages(task,state,action,cfg['attempts']),call['messages'],f'{key}: exact prompt')
            require_equal(seed*100+attempt,call['model_seed'],f'{key}: model seed')
            require_equal(cfg['max_tokens'],call['max_tokens'],f'{key}: completion ceiling')
            record=call['records'][-1]['record'] if call['records'] else {}
            if 'request' in record:
                request=record['request']
                require_equal(request.get('messages'),call['messages'],f'{key}: provider request messages')
                require_equal(request.get('seed'),call['model_seed'],f'{key}: provider seed')
                require_equal(request.get('max_tokens'),call['max_tokens'],f'{key}: provider ceiling')
                require_equal(request.get('model'),cfg['model'],f'{key}: provider model')
                require_equal(request.get('temperature'),cfg['generation_temperature'],f'{key}: provider temperature')
            source='';error=None;static={}
            if call['error']=='provider_error':error='provider_error'
            else:
                choices=record.get('response',{}).get('choices')
                if choices:
                    require_equal(choices[0]['message']['content'],call['text'],f'{key}: provider text')
                try:source,static=module.parse_source(call['text'])
                except (ValueError,SyntaxError,TypeError) as exc:error='invalid_source_'+type(exc).__name__
            require_equal(error,call['error'],f'{key}: parse/provider status')
            require_equal(error,trace['error'],f'{key}: trace status');last_error=error;last_static=static;sources.append(source)
            usage_rows.append(record.get('response',{}).get('usage',{}))
            if attempt==0:
                feedback=trace['feedback']
                if feedback.get('diagnostic') is not True or feedback.get('task')!=task:
                    raise ValueError('Replay refuses nonpublic feedback in actor context')
                expected_inputs=list(module.TASKS[task].public_inputs)+list(module.TASKS[task].diagnostic_inputs)
                require_equal([case['input'] for case in feedback['cases']],expected_inputs,f'{key}: public tool input identities')
                visible={k:feedback[k] for k in ('passed','total','pass_rate','resource_limit_compliant','feedback_scope')}
                visible['cases']=[{k:case[k] for k in ('input','expected','actual','passed','error')} for case in feedback['cases']]
                state.history.append({'source':source,'feedback':visible});state.update(action,feedback)
                diagnostic_rate=feedback['pass_rate']
            elif trace['feedback'] is not None:raise ValueError('Unexpected tool feedback after final proposal')
            require_equal(asdict(state),trace['state_after'],f'{key}: full adaptive state')
            phase_rows.append({'task':task,'seed':seed,'condition':condition,'attempt':attempt,'strategy':action,
               'exact_prompt_verified':True,'recorded_response_verified':True,'full_adaptive_state_verified':True})
        require_equal(sources[-1],result['final_source'],f'{task}/{seed}/{condition}: final candidate')
        require_equal(last_error,result['source_error'],f'{task}/{seed}/{condition}: final source status')
        require_equal(diagnostic_rate,result['initial_diagnostic_pass_rate'],'initial diagnostic score')
        require_equal(cfg['attempts'],result['model_calls'],'trial call budget')
        require_equal(1,result['feedback_tool_calls'],'public tool budget')
        require_equal(1,result['final_checker_calls'],'final evaluator budget')
        require_equal(1,result['retrospective_checker_calls'],'retrospective evaluator budget')
        require_equal(result['evaluation']['pass_rate'],result['final_hidden_pass_rate'],'recorded final score consistency')
        require_equal(result['retrospective_evaluation']['pass_rate'],result['initial_hidden_pass_rate_retrospective'],'recorded initial hidden score consistency')
        require_equal(result['final_hidden_pass_rate']-result['initial_hidden_pass_rate_retrospective'],result['revision_delta_same_hidden_suite'],'same-suite revision score')
        require_equal(last_static.get('confidence'),result['confidence'],'parsed confidence')
        complete_usage=all(isinstance(u.get('prompt_tokens'),int) and isinstance(u.get('completion_tokens'),int) for u in usage_rows)
        for token_name in ('prompt_tokens','completion_tokens'):
            expected=sum(u[token_name] for u in usage_rows) if complete_usage else None
            require_equal(expected,result[token_name],'reported '+token_name)

        stored_sources.append((result,sources))
        verified.append({'task':task,'seed':seed,'condition':condition,'phases':cfg['attempts'],'verified':True})
    # Independent executable checks occur only after ALL actor contexts are closed.
    rechecks=[]
    if recheck:
        for result,sources in stored_sources:
            for label,source,recorded in [('initial',sources[0],result['retrospective_evaluation']),
                                          ('final',sources[-1],result['evaluation'])]:
                fresh=module.evaluate_candidate(source,result['task'],diagnostic=False,config=cfg)
                matches=canonical(stable_evaluation(fresh))==canonical(stable_evaluation(recorded))
                rechecks.append({'task':result['task'],'seed':result['seed'],'condition':result['condition'],'candidate':label,
                     'stable_functional_and_line_results_match':matches,'recorded_pass_rate':recorded['pass_rate'],
                     'fresh_pass_rate':fresh['pass_rate'],'fresh_evaluation':fresh})
    output.mkdir(parents=True,exist_ok=True)
    summary={'schema_version':1,'mode':'record-only actor replay; optional separate evaluator recheck',
       'model_calls':0,'live_provider_calls':0,'network_calls':0,'actor_trials_verified':len(verified),
       'actor_phases_verified':len(phase_rows),'conditions_verified':sorted({v['condition'] for v in verified}),
       'all_actor_trajectories_verified':True,'recorded_public_tool_results_consumed':len(verified),
       'hidden_checker_calls_during_actor_replay':0,'independent_recheck_calls':len(rechecks),
       'all_independent_rechecks_match':all(r['stable_functional_and_line_results_match'] for r in rechecks) if recheck else None,
       'independent_recheck_mismatches':sum(not r['stable_functional_and_line_results_match'] for r in rechecks),
       'source_sha256':manifest,'weights_sha256':{'counterfactual':cfg['counterfactual_sha256'],'ordinary':cfg['ordinary_sha256']},
       'scope':'exact recorded-response trajectory closure; not fresh-model determinism or replication across model versions'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,rows in [('verified_trials.jsonl',verified),('verified_phases.jsonl',phase_rows),('independent_rechecks.jsonl',rechecks)]:
        with (output/name).open('w') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--weights',default='results/benchmark')
    parser.add_argument('--recheck',action='store_true');args=parser.parse_args()
    print(json.dumps(replay_software_experiment(args.run,args.output,weights_dir=args.weights,recheck=args.recheck),indent=2))

if __name__=='__main__':main()
