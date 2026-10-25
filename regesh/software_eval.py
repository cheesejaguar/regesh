"""Frozen-controller same-model software design with independent executable checks.

No live call occurs on import. Generated code is restricted pure Python and run
in a fresh, resource-limited child; this is defense in depth, not an OS sandbox
for arbitrary hostile Python. Hidden test answers never enter model prompts.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
import numpy as np
from .learning import ACTIONS, CONTEXT_NAMES, CounterfactualRegulator, RidgePredictor, DirectValuePredictor
from .providers import LocalChatModel, ProviderError
from .software_tasks import TASKS, hidden_cases, scaling_cases, metamorphic_cases, oracle

DEFAULT_CONFIG={"schema_version":1,"study_status":"prospective_transfer_experiment_v1", "tasks":list(TASKS),
 "seeds":[910,911,912,913],"conditions":["hybrid","ordinary_adaptive"],"attempts":2,"max_tokens":512,
 "generation_temperature":.6,"max_source_characters":24000,"case_cpu_seconds":3,"case_wall_seconds":5,
 "case_memory_mib":128,"line_event_limit":150000,"bootstrap_samples":2000,"bootstrap_seed":6211,
 "weights_frozen":True,"held_out_scope":"new executable task families; pretrained-model contamination unknown"}
SAFE_BUILTINS=('abs','all','any','bool','dict','enumerate','filter','float','int','isinstance','len','list',
 'map','max','min','next','range','reversed','round','set','sorted','str','sum','tuple','zip','ValueError')
DENIED_NAMES={'open','exec','eval','compile','input','help','breakpoint','globals','locals','vars','getattr','setattr',
 'delattr','__import__','memoryview','object','type','super'}
ALLOWED_METHODS={'append','extend','pop','remove','clear','copy','count','index','insert','reverse','sort','items',
 'keys','values','get','setdefault','update','popitem','add','discard','union','intersection','difference','issubset',
 'issuperset','split','join','strip','lstrip','rstrip','startswith','endswith'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def parse_source(text):
    if not isinstance(text,str):raise ValueError('model response must be text')
    text=text.strip()
    if text.startswith('```'):
        lines=text.splitlines()
        if lines[0] not in {'```','```python','```py'} or lines[-1]!='```':raise ValueError('invalid code fence')
        text='\n'.join(lines[1:-1])
    confidence=None
    match=re.search(r'^\s*#\s*confidence\s*:\s*([0-9.]+)\s*$',text,re.M|re.I)
    if match:
        confidence=float(match.group(1))
        if not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('confidence outside [0,1]')
    tree=ast.parse(text)
    if not any(isinstance(n,ast.FunctionDef) and n.name=='solve' for n in tree.body):raise ValueError('missing solve(data)')
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal,ast.AsyncFunctionDef,ast.Await,ast.Yield,ast.YieldFrom)):
            raise ValueError('only synchronous pure functions; no imports/classes/global mutation')
        if isinstance(node,ast.Name) and (node.id.startswith('__') or node.id in DENIED_NAMES):raise ValueError('forbidden name')
        if isinstance(node,ast.Attribute) and node.attr not in ALLOWED_METHODS:raise ValueError('forbidden attribute')
    for node in tree.body:
        if not isinstance(node,(ast.FunctionDef,ast.Assign,ast.AnnAssign,ast.Expr)):
            raise ValueError('unsupported module statement')
        if isinstance(node,ast.Expr) and not isinstance(node.value,ast.Constant):raise ValueError('module calls forbidden')
    normalized=ast.dump(tree,include_attributes=False)
    metrics={'source_characters':len(text),'source_lines':len(text.splitlines()),'ast_nodes':sum(1 for _ in ast.walk(tree)),
             'cyclomatic_proxy':1+sum(isinstance(n,(ast.If,ast.For,ast.While,ast.IfExp,ast.comprehension,ast.ExceptHandler)) for n in ast.walk(tree)),
             'ast_sha256':hashlib.sha256(normalized.encode()).hexdigest(),
             'operation_signature':sorted(set(n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute))),
             'confidence':confidence}
    return text,metrics


_WORKER=r'''
import sys,json,resource,time,tracemalloc
p=json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_AS,(p['memory']*1024*1024,p['memory']*1024*1024))
resource.setrlimit(resource.RLIMIT_CPU,(p['cpu'],p['cpu']+1))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
resource.setrlimit(resource.RLIMIT_NOFILE,(16,16))
resource.setrlimit(resource.RLIMIT_NPROC,(0,0))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import builtins
safe={k:getattr(builtins,k) for k in p['builtins']}
ns={'__builtins__':safe}
results=[]
try:
 exec(compile(p['source'],'<candidate>','exec'),ns,ns)
except BaseException as e:
 print(json.dumps({'setup_error':type(e).__name__,'cases':[]}));sys.exit(0)
for case in p['cases']:
 count=[0]
 def tracer(frame,event,arg):
  if frame.f_code.co_filename=='<candidate>' and event=='line':
   count[0]+=1
   if count[0]>p['line_limit']:raise RuntimeError('line_budget')
  return tracer
 tracemalloc.start();started=time.perf_counter()
 try:
  sys.settrace(tracer)
  result=ns['solve'](case)
  sys.settrace(None)
  encoded=json.dumps(result,allow_nan=False)
  if len(encoded)>500000:raise ValueError('output_limit')
  item={'output':result,'error':None}
 except BaseException as e:
  sys.settrace(None);item={'output':None,'error':type(e).__name__}
 item.update(line_events=count[0],seconds=time.perf_counter()-started,peak_bytes=tracemalloc.get_traced_memory()[1])
 tracemalloc.stop();results.append(item)
body=json.dumps({'cases':results},allow_nan=False)
if len(body)>1000000:body=json.dumps({'setup_error':'output_limit','cases':[]})
sys.stdout.write(body)
'''


def run_source(source,cases,config=None):
    cfg={**DEFAULT_CONFIG,**(config or {})}
    if len(source)>cfg['max_source_characters']:return {'setup_error':'source_limit','cases':[]}
    try:source,_=parse_source(source)
    except (ValueError,SyntaxError,TypeError) as exc:return {'setup_error':'validation_'+type(exc).__name__,'cases':[]}
    payload={'source':source,'cases':cases,'memory':cfg['case_memory_mib'],'cpu':cfg['case_cpu_seconds'],
             'line_limit':cfg['line_event_limit'],'builtins':SAFE_BUILTINS}
    try:
        with tempfile.TemporaryDirectory(prefix='regesh-code-') as directory:
            proc=subprocess.run([sys.executable,'-I','-S','-c',_WORKER],input=json.dumps(payload),text=True,
                capture_output=True,cwd=directory,timeout=cfg['case_wall_seconds'],env={'PYTHONHASHSEED':'0'})
        if proc.returncode:return {'setup_error':'worker_exit_'+str(proc.returncode),'cases':[]}
        result=json.loads(proc.stdout)
        return result
    except subprocess.TimeoutExpired:return {'setup_error':'wall_timeout','cases':[]}
    except (ValueError,OSError):return {'setup_error':'worker_protocol_error','cases':[]}


def typed_equal(left,right):
    """JSON integer contracts do not accept booleans or floating substitutes."""
    if type(left) is not type(right):return False
    if isinstance(left,list):return len(left)==len(right) and all(typed_equal(a,b) for a,b in zip(left,right))
    if isinstance(left,dict):return left.keys()==right.keys() and all(typed_equal(left[k],right[k]) for k in left)
    return left==right


def evaluate_candidate(source,task_id,*,diagnostic=False,config=None):
    """Condition-blind independent checker; hidden results are never tool feedback."""
    task=TASKS[task_id]
    cases=(list(task.public_inputs)+list(task.diagnostic_inputs)) if diagnostic else hidden_cases(task_id)+metamorphic_cases(task_id)+scaling_cases(task_id)
    try:_,static=parse_source(source)
    except (ValueError,SyntaxError,TypeError):static={}
    executed=run_source(source,cases,config)
    rows=[]
    for index,data in enumerate(cases):
        observed=executed.get('cases',[])[index] if index<len(executed.get('cases',[])) else {'error':executed.get('setup_error','missing_output'),'output':None,'line_events':0,'peak_bytes':0,'seconds':0.}
        expected=oracle(task_id,data)
        passed=observed.get('error') is None and typed_equal(observed.get('output'),expected)
        row={'case':index,'passed':passed,'error':observed.get('error'),'line_events':observed['line_events'],'peak_bytes':observed['peak_bytes'],'seconds':observed['seconds']}
        if diagnostic:row.update(input=data,expected=expected,actual=observed.get('output'))
        rows.append(row)
    n=len(rows);passed=sum(r['passed'] for r in rows)
    base_count=len(hidden_cases(task_id)) if not diagnostic else len(rows)
    labels=(['boundary']*3+['randomized']*9) if task_id=='toposort' else ['boundary' if i%4 in (0,1) else 'randomized' for i in range(base_count)]
    boundary=[r for r,label in zip(rows[:base_count],labels) if label=='boundary'] if not diagnostic else rows
    randomized=[r for r,label in zip(rows[:base_count],labels) if label=='randomized'] if not diagnostic else rows
    metamorphic=rows[base_count:base_count+4] if not diagnostic else []
    scaling=rows[-2:] if not diagnostic else []
    ratio=(scaling[1]['line_events']/max(1,scaling[0]['line_events'])) if scaling else None
    result={'task':task_id,'diagnostic':diagnostic,'passed':passed,'total':n,'pass_rate':passed/n,
       'all_passed':passed==n,'format_valid':bool(static),'execution_valid':not bool(executed.get('setup_error')),
       'boundary_pass_rate':sum(r['passed'] for r in boundary)/len(boundary) if boundary else None,
       'randomized_pass_rate':sum(r['passed'] for r in randomized)/len(randomized) if randomized else None,
       'metamorphic_pass_rate':sum(r['passed'] for r in metamorphic)/len(metamorphic) if metamorphic else None,
       'resource_limit_compliant':not bool(executed.get('setup_error')) and not any(r['error'] for r in rows),
       'line_events':sum(r['line_events'] for r in rows),'peak_working_bytes':max(r['peak_bytes'] for r in rows),
       'seconds':sum(r['seconds'] for r in rows),'fourfold_input_line_ratio':ratio,
       'scaling_correct':all(r['passed'] for r in scaling) if scaling else None,
       'static':static,'error_counts':{error:sum(r['error']==error for r in rows) for error in sorted({str(r['error']) for r in rows if r['error']})},
       'cases':rows}
    # Compute honesty: a fast wrong program is never considered efficient quality.
    result['efficiency_eligible']=bool(result['all_passed'])
    if diagnostic:result['feedback_scope']='public examples plus one diagnostic case; hidden tests withheld'
    return result


@dataclass
class TrialState:
    quality:float=0.
    uncertainty:float=1.
    feasible_rate:float=0.
    load:float=0.
    mood:float=0.
    recent_gain:float=0.
    expected_gain:float=.1
    failures:float=0.
    stagnation:float=0.
    step:int=0
    expectations:list=field(default_factory=lambda:[.1]*4)
    history:list=field(default_factory=list)

    def context(self,budget):
        return [self.quality,self.uncertainty,self.feasible_rate,self.load,self.mood,self.recent_gain,
          self.expected_gain,self.failures,self.stagnation,self.step/budget,0.,1.,1.,*self.expectations]

    def update(self,action,observed):
        quality=observed['pass_rate'];gain=quality-self.quality
        error=quality-self.expectations[ACTIONS.index(action)]
        self.quality=quality;self.uncertainty=1/math.sqrt(self.step+2)
        self.feasible_rate=float(observed['resource_limit_compliant'])
        self.load=float(np.clip(.9*self.load+{'diverge':.12,'refine':.07,'combine':.09,'verify':-.08}[action],0,1))
        self.mood=.85*self.mood+.15*error;self.recent_gain=.5*self.recent_gain+.5*gain
        self.expected_gain=.85*self.expected_gain+.15*gain;self.failures=.85*self.failures+.15*(quality<1.)
        self.stagnation=0. if gain>.01 else min(1.,self.stagnation+.1)
        index=ACTIONS.index(action);self.expectations[index]=.8*self.expectations[index]+.2*quality
        self.step+=1


def choose(condition,state,budget,counterfactual,ordinary):
    context=state.context(budget)
    if condition=='hybrid':
        action,predictions,scores=CounterfactualRegulator(counterfactual).select(dict(zip(CONTEXT_NAMES,context)))
    elif condition=='ordinary_adaptive':
        predictions={a:ordinary.predict(context,a) for a in ACTIONS};scores=predictions
        action=max(ACTIONS,key=lambda a:scores[a])
    else:raise ValueError('unknown condition')
    return action,{'context':context,'predictions':predictions,'scores':scores,'predictor_evaluations':4}


def messages(task_id,state,action,attempts):
    task=TASKS[task_id]
    neutral=dict(zip(CONTEXT_NAMES,state.context(attempts)));neutral['signed_error_average']=neutral.pop('mood')
    payload={'specification':task.specification,'public_examples':task.public_examples(),
       'allowed_builtin_names':SAFE_BUILTINS,'allowed_method_names':sorted(ALLOWED_METHODS),
       'search_strategy':action,'operational_state':neutral,'previous_attempts':state.history,
       'strategy_meaning':{'diverge':'consider a different data structure or algorithm',
          'refine':'repair edge cases and simplify the current implementation',
          'combine':'combine useful parts of the earlier implementation with a complementary technique',
          'verify':'check boundaries, ordering and state-update invariants before returning the implementation'}[action],
       'remaining_attempts':attempts-state.step}
    system=('Write a correct, lightweight Python implementation of solve(data). Output Python source only, '
      'with an optional first line # confidence: 0.75 expressing probability all specification tests pass. '
      'No explanation or imports. Use only pure builtins and ordinary list/dict/set methods. '
      'No classes, file/network access, dunder attributes, eval or exec. Include helper functions if useful. '
      'When there is no prior candidate, create an initial implementation. Follow the supplied search strategy. '
      'Hidden tests independently check correctness and resource use. Do not claim an unobserved test passed.')
    return [{'role':'system','content':system},{'role':'user','content':json.dumps(payload,sort_keys=True,allow_nan=False)}]


def run_software_experiment(model,output_dir,*,weights_dir='results/benchmark',config=None):
    cfg={**DEFAULT_CONFIG,**(config or {})}
    if cfg['attempts']!=2:raise ValueError('prospective protocol fixes two candidate calls and one feedback tool')
    if len(set(cfg['seeds']))!=len(cfg['seeds']) or any(type(s)is not int or s<900 for s in cfg['seeds']):raise ValueError('unique held-out seeds >=900 required')
    if set(cfg['conditions'])!={'hybrid','ordinary_adaptive'} or len(cfg['conditions'])!=2:raise ValueError('exact paired conditions required')
    if not cfg['tasks'] or len(set(cfg['tasks']))!=len(cfg['tasks']) or any(t not in TASKS for t in cfg['tasks']):raise ValueError('known unique tasks required')
    output=Path(output_dir)
    if output.exists() and any(output.iterdir()):raise ValueError('Output directory must be empty; preserve evidence')
    weights=Path(weights_dir);cp=weights/'counterfactual_weights.json';op=weights/'ordinary_weights.json'
    counterfactual=RidgePredictor.load(cp);ordinary=DirectValuePredictor.load(op)
    expected_temperature=cfg['generation_temperature']
    observed_temperature=getattr(model,'temperature',None)
    if observed_temperature is not None and observed_temperature!=expected_temperature:raise ValueError('model temperature differs from declared protocol')
    config_record={**cfg,'model_adapter':getattr(model,'identifier',type(model).__name__),'model':getattr(model,'model',None),
      'temperature_observed':observed_temperature,'counterfactual_sha256':sha(cp),'ordinary_sha256':sha(op),
      'maximum_model_calls':len(cfg['tasks'])*len(cfg['seeds'])*2*cfg['attempts'],
      'maximum_completion_tokens':len(cfg['tasks'])*len(cfg['seeds'])*2*cfg['attempts']*cfg['max_tokens'],
      'tool_calls_per_trial':1,'final_hidden_checker_per_trial':1,'retrospective_checker_per_trial':1,
      'execution_mode':'live' if isinstance(model,LocalChatModel) else 'injected adapter; verify provenance',
      'python':platform.python_version(),'numpy':np.__version__}
    output.mkdir(parents=True,exist_ok=True);dump(output/'config.json',config_record)
    for path in (__file__,str(Path(__file__).with_name('software_tasks.py'))):
        p=Path(path);(output/('source_'+p.name)).write_bytes(p.read_bytes())
    dump(output/'source_manifest.json',{Path(p).name:sha(p) for p in (__file__,str(Path(__file__).with_name('software_tasks.py')))})
    before_counter=counterfactual.weights.copy();before_ordinary=ordinary.weights.copy();results=[]
    with (output/'calls.jsonl').open('w') as calls,(output/'traces.jsonl').open('w') as traces,(output/'results.jsonl').open('w') as rows:
        for task_index,task_id in enumerate(cfg['tasks']):
            for seed in cfg['seeds']:
                order=cfg['conditions'] if (seed+task_index)%2==0 else list(reversed(cfg['conditions']))
                for condition in order:
                    state=TrialState();initial=asdict(state);initial_score=None;first_source='';last_source='';last_static={};last_record={};usage_rows=[];tool_count=0;started=time.perf_counter()
                    for attempt in range(cfg['attempts']):
                        action,reason=choose(condition,state,cfg['attempts'],counterfactual,ordinary)
                        prompt=messages(task_id,state,action,cfg['attempts']);model_seed=seed*100+attempt
                        captured=[]
                        def capture(kind,record):captured.append({'kind':kind,'record':record})
                        error=None;text=None;source='';static={}
                        try:
                            response=model.complete(prompt,seed=model_seed,max_tokens=cfg['max_tokens'],recorder=capture)
                            text=response.text
                            if not captured:capture('model_call',response.record)
                            try:source,static=parse_source(text)
                            except (ValueError,SyntaxError,TypeError) as exc:error='invalid_source_'+type(exc).__name__
                        except ProviderError as exc:
                            error='provider_error'
                            if not captured:capture('model_call',exc.record)
                        call={'task':task_id,'seed':seed,'condition':condition,'attempt':attempt,'messages':prompt,'model_seed':model_seed,
                              'max_tokens':cfg['max_tokens'],'text':text,'error':error,'records':captured}
                        calls.write(json.dumps(call,allow_nan=False)+'\n');calls.flush()
                        usage=captured[-1]['record'].get('response',{}).get('usage',{}) if captured else {}
                        usage_rows.append(usage)
                        if attempt==0:
                            first_source=source
                            feedback=evaluate_candidate(source,task_id,diagnostic=True,config=cfg);tool_count+=1;initial_score=feedback['pass_rate']
                            visible_feedback={k:feedback[k] for k in ('passed','total','pass_rate','resource_limit_compliant','feedback_scope')}
                            visible_feedback['cases']=[{k:case[k] for k in ('input','expected','actual','passed','error')} for case in feedback['cases']]
                            state.history.append({'source':source,'feedback':visible_feedback})
                            state.update(action,feedback)
                        else:
                            # Only final checker sees hidden cases; its result cannot select a candidate.
                            feedback=None;last_source=source;last_static=static;last_record=call
                        traces.write(json.dumps({'task':task_id,'seed':seed,'condition':condition,'attempt':attempt,'action':action,
                             'structured_reason':reason,'feedback':feedback,'state_after':asdict(state),'error':error},allow_nan=False)+'\n');traces.flush()
                    evaluated=evaluate_candidate(last_source,task_id,diagnostic=False,config=cfg)
                    retrospective=evaluate_candidate(first_source,task_id,diagnostic=False,config=cfg)
                    confidence=last_static.get('confidence');correct=float(evaluated['all_passed'])
                    all_usage=all(isinstance(u.get('prompt_tokens'),int) and isinstance(u.get('completion_tokens'),int) for u in usage_rows)
                    result={'task':task_id,'seed':seed,'condition':condition,'initial_snapshot':initial,
                      'weights':{'counterfactual_sha256':config_record['counterfactual_sha256'],'ordinary_sha256':config_record['ordinary_sha256']},
                      'model_calls':cfg['attempts'],'feedback_tool_calls':tool_count,'final_checker_calls':1,'retrospective_checker_calls':1,
                      'prompt_tokens':sum(u['prompt_tokens'] for u in usage_rows) if all_usage else None,
                      'completion_tokens':sum(u['completion_tokens'] for u in usage_rows) if all_usage else None,
                      'requested_completion_token_ceiling':cfg['attempts']*cfg['max_tokens'],
                      'initial_diagnostic_pass_rate':initial_score,'final_hidden_pass_rate':evaluated['pass_rate'],
                      'revision_delta_different_suites':evaluated['pass_rate']-initial_score,
                      'initial_hidden_pass_rate_retrospective':retrospective['pass_rate'],
                      'revision_delta_same_hidden_suite':evaluated['pass_rate']-retrospective['pass_rate'],
                      'retrospective_evaluation':retrospective,
                      'confidence':confidence,'brier_all_tests':None if confidence is None else (confidence-correct)**2,
                      'seconds':time.perf_counter()-started,'final_source':last_source,'evaluation':evaluated,
                      'execution_mode':config_record['execution_mode'],'source_error':last_record.get('error')}
                    results.append(result);rows.write(json.dumps(result,allow_nan=False)+'\n');rows.flush()
    if not np.array_equal(before_counter,counterfactual.weights) or not np.array_equal(before_ordinary,ordinary.weights):raise AssertionError('frozen model mutated')
    summary=summarize(results,cfg);summary['frozen_weights_verified']=True;summary['execution_mode']=config_record['execution_mode']
    dump(output/'summary.json',summary);write_report(output,summary)
    return summary


def summarize(results,config):
    summary={'trials':len(results),'means':{},'paired_final_pass_rate':{},'scope':'held-out authored executable software design; no published originality claim',
             'primary_uncertainty':'paired bootstrap of task means; generation seeds clustered within tasks',
             'multiple_comparisons':'exploratory; intervals not multiplicity corrected'}
    for condition in config['conditions']:
        rows=[r for r in results if r['condition']==condition]
        confidence=[r for r in rows if r['confidence'] is not None]
        summary['means'][condition]={
          'final_hidden_pass_rate':float(np.mean([r['final_hidden_pass_rate'] for r in rows])),
          'all_tests_passed':float(np.mean([r['evaluation']['all_passed'] for r in rows])),
          'format_valid':float(np.mean([r['evaluation']['format_valid'] for r in rows])),
          'boundary_pass_rate':float(np.mean([r['evaluation']['boundary_pass_rate'] for r in rows])),
          'randomized_pass_rate':float(np.mean([r['evaluation']['randomized_pass_rate'] for r in rows])),
          'metamorphic_pass_rate':float(np.mean([r['evaluation']['metamorphic_pass_rate'] for r in rows])),
          'revision_delta_same_hidden_suite':float(np.mean([r['revision_delta_same_hidden_suite'] for r in rows])),
          'resource_limit_compliance':float(np.mean([r['evaluation']['resource_limit_compliant'] for r in rows])),
          'initial_diagnostic_pass_rate':float(np.mean([r['initial_diagnostic_pass_rate'] for r in rows])),
          'confidence_coverage':len(confidence)/len(rows),'brier_all_tests':float(np.mean([r['brier_all_tests'] for r in confidence])) if confidence else None,
          'mean_line_events_correct_only':float(np.mean([r['evaluation']['line_events'] for r in rows if r['evaluation']['all_passed']])) if any(r['evaluation']['all_passed'] for r in rows) else None,
          'mean_peak_bytes_correct_only':float(np.mean([r['evaluation']['peak_working_bytes'] for r in rows if r['evaluation']['all_passed']])) if any(r['evaluation']['all_passed'] for r in rows) else None,
          'source_ast_diversity':len({r['evaluation']['static'].get('ast_sha256') for r in rows if r['evaluation']['format_valid']}),
          'operation_signature_diversity':len({tuple(r['evaluation']['static'].get('operation_signature',[])) for r in rows if r['evaluation']['format_valid']}),
          'model_calls':sum(r['model_calls'] for r in rows),'feedback_tool_calls':sum(r['feedback_tool_calls'] for r in rows),
          'reported_prompt_tokens':sum(r['prompt_tokens'] for r in rows) if all(r['prompt_tokens'] is not None for r in rows) else None,
          'reported_completion_tokens':sum(r['completion_tokens'] for r in rows) if all(r['completion_tokens'] is not None for r in rows) else None,
          'seconds':sum(r['seconds'] for r in rows)}
    paired=[];per_task={}
    for task in config['tasks']:
        differences=[]
        for seed in config['seeds']:
            h=next(r for r in results if r['task']==task and r['seed']==seed and r['condition']=='hybrid')
            o=next(r for r in results if r['task']==task and r['seed']==seed and r['condition']=='ordinary_adaptive')
            differences.append(h['final_hidden_pass_rate']-o['final_hidden_pass_rate'])
        paired.append(float(np.mean(differences)));per_task[task]={'difference':float(np.mean(differences)),'pairs':len(differences)}
    rng=np.random.default_rng(config['bootstrap_seed']);boot=np.mean(rng.choice(paired,(config['bootstrap_samples'],len(paired))),axis=1)
    summary['paired_final_pass_rate']={'hybrid_minus_ordinary':float(np.mean(paired)),'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
                                     'task_clusters':len(paired),'pairs':len(config['tasks'])*len(config['seeds']),'per_task':per_task}
    return summary


def write_report(output,summary):
    lines=['# Executed frozen-controller software design experiment','','Mode: '+summary['execution_mode']+'.',
           'Hidden executable checks are independent of controller condition. Source code, feedback and all model records are retained.','',
           '| Condition | Hidden pass rate | All tests pass | Boundary | Generated cases | Resource execution |',
           '|---|---:|---:|---:|---:|---:|']
    for condition,m in summary['means'].items():
        lines.append(f"| {condition} | {m['final_hidden_pass_rate']:.4f} | {m['all_tests_passed']:.4f} | {m['boundary_pass_rate']:.4f} | {m['randomized_pass_rate']:.4f} | {m['resource_limit_compliance']:.4f} |")
    d=summary['paired_final_pass_rate'];lines+=['',f"Hybrid minus ordinary final pass rate: {d['hybrid_minus_ordinary']:.4f}, 95% task-cluster bootstrap [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}].",
      '', 'Frozen generic weights were verified unchanged. Context adaptation is within trial only. Equal call/tool ceilings do not imply equal realized tokens; actual usage is reported.',
      f"The transferred controllers were trained on a different toy grammar; this study tests that unvalidated transfer, not an optimized software policy. {d['task_clusters']} authored tasks and {d['pairs']//d['task_clusters']} seeds are exploratory.",
      'Reported source/operation diversity and static complexity are proxies, not research originality. Resource measurements include returned output and tracing overhead; asymptotic storage bounds are not formally proved.',
      'Initial source is retrospectively checked on the same hidden suite only after final generation; that improvement metric cannot influence selection. Public-to-hidden score difference is separately labeled confounded. No human psychological, biological or subjective-experience claim follows.']
    (Path(output)/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',required=True);p.add_argument('--model',required=True)
    p.add_argument('--weights',default='results/benchmark');p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--timeout',type=float,default=120)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text()) if args.config else DEFAULT_CONFIG
    model=LocalChatModel(args.base_url,args.model,timeout=args.timeout,temperature=cfg['generation_temperature'])
    print(json.dumps(run_software_experiment(model,args.output,weights_dir=args.weights,config=cfg),indent=2))

if __name__=='__main__':main()
