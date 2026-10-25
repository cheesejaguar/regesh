"""Authored held-out pure-Python design tasks with independent executable oracles.

Candidate prompts expose specifications and public examples, never hidden cases
or oracle source. These are transparent engineering tests, not published novelty.
"""
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class SoftwareTask:
    id: str
    specification: str
    public_inputs: tuple
    diagnostic_inputs: tuple

    def public_examples(self):
        return [{'input':x,'output':oracle(self.id,x)} for x in self.public_inputs]


TASKS = {
 'lru': SoftwareTask('lru',
  'Implement solve(data), a bounded LRU cache simulator. data={"capacity":positive_int,"ops":[["put",key,value] or ["get",key]]}. Return {"reads":[value or null for each get],"keys":[cached keys from least to most recently used]}. Put updates count as use; successful get counts as use; a miss does not change order. Evict least recent only when inserting past capacity. Keys and values are integers. Use O(capacity) storage, never retain all historical operations in extra state.',
  ({'capacity':2,'ops':[['put',1,10],['put',2,20],['get',1],['put',3,30],['get',2]]}, {'capacity':1,'ops':[['get',0],['put',1,2],['put',1,3],['get',1]]}),
  ({'capacity':2,'ops':[['put',1,5],['put',2,6],['put',1,7],['put',3,8],['get',2],['get',1]]},)),
 'ttl': SoftwareTask('ttl',
  'Implement solve(data), a TTL cache. data={"ttl":nonnegative_int,"ops":[["put",time,key,value] or ["get",time,key]]}; times never decrease. A put replaces the entry and sets expiration=time+ttl. At each operation remove entries with expiration<=time. Get returns current value or null, never renews TTL. Return {"reads":[one value/null per get],"keys":[sorted unexpired keys after the last operation]}. Keys/values are integers. Remove expired state; no historical log.',
  ({'ttl':3,'ops':[['put',0,1,7],['get',2,1],['get',3,1]]}, {'ttl':0,'ops':[['put',0,2,8],['get',0,2]]}),
  ({'ttl':2,'ops':[['put',0,1,5],['put',1,1,6],['get',2,1],['get',3,1]]},)),
 'batching': SoftwareTask('batching',
  'Implement solve(data), stable bounded batching. data={"limit":positive_int,"weights":[positive integers <=limit]}. Partition weights in original order into contiguous nonempty batches. Greedily fill each batch until the next weight would exceed limit, then start the next batch. Return the list of batches. Every input occurrence must appear exactly once, including duplicate weights. Empty input returns []. Use linear time and output storage, without sorting.',
  ({'limit':5,'weights':[2,3,1,4,2]}, {'limit':3,'weights':[]}),
  ({'limit':5,'weights':[5,1,1,3,2,2]},)),
 'dedup': SoftwareTask('dedup',
  'Implement solve(data), a windowed acceptance filter. data={"window":nonnegative_int,"events":[[time,key],...]}, times never decrease, integer keys. Accept an event if its key has no previously ACCEPTED event with time>current_time-window. A rejected event does not refresh its key. Return accepted events unchanged as [time,key] pairs in original order. An event exactly window units after its last accepted occurrence is accepted; window=0 accepts every event. State should retain only currently relevant keys, not the full input history beyond output.',
  ({'window':3,'events':[[0,1],[1,1],[3,1],[3,2],[4,1]]}, {'window':0,'events':[[0,1],[0,1]]}),
  ({'window':4,'events':[[0,1],[2,1],[4,1],[4,2],[5,2],[8,1]]},)),
 'toposort': SoftwareTask('toposort',
  'Implement solve(data), deterministic dependency scheduling. data={"n":nonnegative_int,"edges":[[before,after],...]}, node IDs 0..n-1. Duplicate edges count once. Return the lexicographically smallest topological ordering: at each step choose the smallest currently available node. Return [] if any directed cycle exists, including a self-loop. Include isolated nodes. Use storage O(n+unique_edges).',
  ({'n':4,'edges':[[0,2],[1,2],[2,3]]}, {'n':2,'edges':[[0,1],[1,0]]}),
  ({'n':4,'edges':[[2,3],[2,3],[0,3]]},)),
 'rolling': SoftwareTask('rolling',
  'Implement solve(data), bounded rolling statistics. data={"window":positive_int,"values":[integers,...]}. After each value emit {"count":size,"sum":sum,"min":minimum,"max":maximum} for the last at most window values including current. Return the list of records; empty input returns []. Keep O(window) working storage, besides returned output; avoid repeatedly rescanning the entire prefix. Exact integer output is required.',
  ({'window':2,'values':[4,-1,3,3]}, {'window':3,'values':[]}),
  ({'window':1,'values':[4,4,-2,0]},)),
}


def oracle(task_id, data):
    if task_id=='lru':
        c={};reads=[]
        for op in data['ops']:
            key=op[1]
            if op[0]=='get':
                reads.append(c.get(key))
                if key in c:c[key]=c.pop(key)
            else:
                c.pop(key,None);c[key]=op[2]
                if len(c)>data['capacity']:del c[next(iter(c))]
        return {'reads':reads,'keys':list(c)}
    if task_id=='ttl':
        c={};reads=[]
        for op in data['ops']:
            now,key=op[1:3]
            c={k:v for k,v in c.items() if v[1]>now}
            if op[0]=='put':
                if data['ttl']>0:c[key]=(op[3],now+data['ttl'])
            else:reads.append(c.get(key,(None,))[0])
        return {'reads':reads,'keys':sorted(c)}
    if task_id=='batching':
        batches=[];current=[];total=0
        for w in data['weights']:
            if current and total+w>data['limit']:batches.append(current);current=[];total=0
            current.append(w);total+=w
        if current:batches.append(current)
        return batches
    if task_id=='dedup':
        last={};result=[]
        for t,k in data['events']:
            last={key:time for key,time in last.items() if time>t-data['window']}
            if k not in last:result.append([t,k]);last[k]=t
        return result
    if task_id=='toposort':
        n=data['n'];edges=set(map(tuple,data['edges']));done=[]
        while len(done)<n:
            available=[i for i in range(n) if i not in done and not any(b==i and a not in done for a,b in edges)]
            if not available:return []
            done.append(min(available))
        return done
    if task_id=='rolling':
        out=[]
        for i in range(len(data['values'])):
            w=data['values'][max(0,i+1-data['window']):i+1]
            out.append({'count':len(w),'sum':sum(w),'min':min(w),'max':max(w)})
        return out
    raise ValueError(task_id)


def hidden_cases(task_id, seed=62017):
    """Fixed boundary cases plus deterministic generated/metamorphic cases."""
    r=random.Random(seed);cases=[]
    if task_id=='lru':
        for capacity in (1,2,4):
            for count in (0,1,8,32):
                ops=[(['get',r.randrange(7)] if r.random()<.5 else ['put',r.randrange(7),r.randrange(-9,10)]) for _ in range(count)]
                cases.append({'capacity':capacity,'ops':ops})
    elif task_id=='ttl':
        for ttl in (0,1,4):
            for count in (0,1,10,32):
                time=0;ops=[]
                for _ in range(count):
                    time+=r.randrange(3)
                    ops.append(['get',time,r.randrange(5)] if r.random()<.5 else ['put',time,r.randrange(5),r.randrange(-5,6)])
                cases.append({'ttl':ttl,'ops':ops})
    elif task_id=='batching':
        for limit in (1,3,7):
            for count in (0,1,10,40):cases.append({'limit':limit,'weights':[r.randint(1,limit) for _ in range(count)]})
    elif task_id=='dedup':
        for window in (0,1,4):
            for count in (0,1,15,40):
                time=0;events=[]
                for _ in range(count):time+=r.randrange(3);events.append([time,r.randrange(6)])
                cases.append({'window':window,'events':events})
    elif task_id=='toposort':
        cases=[{'n':0,'edges':[]},{'n':1,'edges':[[0,0]]},{'n':3,'edges':[[0,1],[1,2],[2,0]]}]
        for n in (1,3,8):
            for density in (.1,.3,.6):
                edges=[[a,b] for a in range(n) for b in range(a+1,n) if r.random()<density]
                cases.append({'n':n,'edges':edges+edges[:2]})
    elif task_id=='rolling':
        for window in (1,3,20):
            for count in (0,1,12,40):cases.append({'window':window,'values':[r.randint(-20,20) for _ in range(count)]})
    else:raise ValueError(task_id)
    return cases


def scaling_cases(task_id):
    if task_id=='lru':return [{'capacity':8,'ops':[['put',i%19,i] for i in range(n)]} for n in (64,256)]
    if task_id=='ttl':return [{'ttl':8,'ops':[['put',i,i%19,i] for i in range(n)]} for n in (64,256)]
    if task_id=='batching':return [{'limit':7,'weights':[1,2,3,1]*n} for n in (16,64)]
    if task_id=='dedup':return [{'window':8,'events':[[i,i%19] for i in range(n)]} for n in (64,256)]
    if task_id=='toposort':return [{'n':n,'edges':[[i,i+1] for i in range(n-1)]} for n in (16,64)]
    if task_id=='rolling':return [{'window':8,'values':list(range(n))} for n in (64,256)]
    raise ValueError(task_id)


def metamorphic_cases(task_id):
    """Semantics-preserving translations/scalings; independent oracle rechecks."""
    import copy
    base=[copy.deepcopy(case) for case in hidden_cases(task_id)[2:6]]
    for case in base:
        if task_id=='lru':
            for op in case['ops']:
                op[1]+=101
                if op[0]=='put':op[2]=3*op[2]+1
        elif task_id=='ttl':
            for op in case['ops']:op[1]+=100
        elif task_id=='batching':case['limit']*=3;case['weights']=[3*w for w in case['weights']]
        elif task_id=='dedup':
            for event in case['events']:event[0]+=100
        elif task_id=='toposort':case['edges']=list(reversed(case['edges']))+case['edges'][:2]
        elif task_id=='rolling':case['values']=[v+17 for v in case['values']]
    return base
