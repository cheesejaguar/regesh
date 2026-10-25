"""Protocol fixtures, not model-performance evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from regesh.providers import ModelResult
from regesh.software_eval import DEFAULT_CONFIG,parse_source,run_source,evaluate_candidate,run_software_experiment,typed_equal
from regesh.software_tasks import TASKS,hidden_cases,oracle

SOLUTIONS={
'lru':'''def solve(data):
 c={}; out=[]
 for op in data['ops']:
  k=op[1]
  if op[0]=='get':
   out.append(c.get(k))
   if k in c: c[k]=c.pop(k)
  else:
   c.pop(k,None); c[k]=op[2]
   if len(c)>data['capacity']: c.pop(list(c)[0])
 return {'reads':out,'keys':list(c)}
''',
'ttl':'''def solve(data):
 c={};out=[]
 for op in data['ops']:
  t,k=op[1:3]
  c={k:v for k,v in c.items() if v[1]>t}
  if op[0]=='put':
   if data['ttl']>0: c[k]=(op[3],t+data['ttl'])
  else: out.append(c.get(k,(None,))[0])
 return {'reads':out,'keys':sorted(c)}
''',
'batching':'''def solve(data):
 out=[];batch=[];total=0
 for w in data['weights']:
  if batch and total+w>data['limit']:out.append(batch);batch=[];total=0
  batch.append(w);total+=w
 if batch:out.append(batch)
 return out
''',
'dedup':'''def solve(data):
 last={};out=[]
 for t,k in data['events']:
  last={x:v for x,v in last.items() if v>t-data['window']}
  if k not in last:out.append([t,k]);last[k]=t
 return out
''',
'toposort':'''def solve(data):
 n=data['n']; edges=set(tuple(e) for e in data['edges']); deg=[0]*n;out=[];nexts=[[] for _ in range(n)]
 for a,b in edges:deg[b]+=1;nexts[a].append(b)
 ready=[i for i in range(n) if deg[i]==0]
 while ready:
  ready.sort();a=ready.pop(0);out.append(a)
  for b in nexts[a]:
   deg[b]-=1
   if deg[b]==0:ready.append(b)
 return out if len(out)==n else []
''',
'rolling':'''def solve(data):
 out=[];w=[];total=0
 for v in data['values']:
  w.append(v);total+=v
  if len(w)>data['window']:total-=w.pop(0)
  out.append({'count':len(w),'sum':total,'min':min(w),'max':max(w)})
 return out
'''}


class CodeChecks(unittest.TestCase):
 def test_reference_fixtures_pass_all_independent_suites(self):
  for task,source in SOLUTIONS.items():
   with self.subTest(task=task):
    grade=evaluate_candidate(source,task)
    self.assertTrue(grade['all_passed'],grade)
    self.assertTrue(grade['resource_limit_compliant'])
 def test_incorrect_shortcut_fails_hidden_cases(self):
  result=evaluate_candidate('def solve(data):\n return []','batching')
  self.assertLess(result['pass_rate'],1)
  self.assertFalse(result['efficiency_eligible'])
 def test_external_access_and_dunder_rejected(self):
  for source in ['import os\ndef solve(data):return []','def solve(data):\n return data.__class__','def solve(data):\n return open("/etc/passwd").read()']:
   with self.assertRaises(ValueError):parse_source(source)
 def test_nontermination_has_resource_failure(self):
  result=run_source('def solve(data):\n while True: pass',[{}],{'line_event_limit':100})
  self.assertIsNotNone(result['cases'][0]['error'])
 def test_confidence_and_code_fence(self):
  source,metrics=parse_source('```python\n# confidence: 0.75\ndef solve(data):\n return []\n```')
  self.assertEqual(metrics['confidence'],.75)
  self.assertTrue(source.startswith('# confidence'))
 def test_exact_output_types_do_not_accept_bool_or_float(self):
  self.assertFalse(typed_equal([True],[1]));self.assertFalse(typed_equal({"x":1.0},{"x":1}))
 def test_oracles_have_task_boundary_cases(self):
  self.assertEqual(oracle('ttl',{'ttl':0,'ops':[['put',0,1,9],['get',0,1]]}),{'reads':[None],'keys':[]})
  self.assertEqual(oracle('dedup',{'window':2,'events':[[0,1],[1,1],[2,1]]}),[[0,1],[2,1]])
  self.assertEqual(oracle('toposort',{'n':2,'edges':[[0,1],[0,1]]}),[0,1])


class ConstantModel:
 identifier='constant-software-fixture-v1';temperature=.6
 def complete(self,messages,*,seed,max_tokens,recorder=None):
  record={'request':{'messages':messages,'seed':seed,'max_tokens':max_tokens},
          'response':{'usage':{'prompt_tokens':200,'completion_tokens':30}}}
  if recorder:recorder('model_call',record)
  return ModelResult(SOLUTIONS['batching'],record)


class RunnerChecks(unittest.TestCase):
 def test_frozen_paired_runner_budget_logging_and_no_hidden_leak(self):
  config={**DEFAULT_CONFIG,'tasks':['batching'],'seeds':[910],'bootstrap_samples':10}
  with tempfile.TemporaryDirectory() as tmp:
   summary=run_software_experiment(ConstantModel(),tmp,config=config)
   self.assertTrue(summary['frozen_weights_verified'])
   self.assertEqual(summary['paired_final_pass_rate']['hybrid_minus_ordinary'],0)
   rows=[json.loads(s) for s in (Path(tmp)/'results.jsonl').read_text().splitlines()]
   self.assertEqual(len(rows),2)
   for row in rows:
    self.assertEqual(row['model_calls'],2);self.assertEqual(row['feedback_tool_calls'],1)
    self.assertEqual(row['final_hidden_pass_rate'],1.)
   calls=[json.loads(s) for s in (Path(tmp)/'calls.jsonl').read_text().splitlines()]
   for c in calls:
    prompt=' '.join(m['content'] for m in c['messages'])
    self.assertNotIn('62017',prompt)
    self.assertNotIn('def oracle',prompt)
   with self.assertRaisesRegex(ValueError,'preserve evidence'):
    run_software_experiment(ConstantModel(),tmp,config=config)
