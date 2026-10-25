#!/usr/bin/env python3
"""Paired memory-journal profiling; snapshot/Python-delta/Rust-delta isolated workers.

No language model or cognitive-controller change is involved. Disk savings from
representation and native CPU changes are reported separately. Run after cargo
build --release --manifest-path native/Cargo.toml for all three backends.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time


def worker(backend,count,workload="append"):
    os.environ['REGESH_MEMORY_BACKEND']=backend
    from regesh import Agent,Event
    from regesh.storage import digest
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'profile.sqlite';agent=Agent(path,'profile')
        started=time.perf_counter()
        for i in range(count):
            agent.observe(Event(str(i),'profile','person','target','task','2026-10-25T00:00:00Z',
                data={'goal_delta':{'quality':(-1 if i%2 else 1)*.3},'text':'evidence.'*100}))
            if workload=='clocked':agent.advance_time(1.)
        write_seconds=time.perf_counter()-started
        snapshot_start=time.perf_counter();snapshot=agent.snapshot();snapshot_seconds=time.perf_counter()-snapshot_start
        agent.store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        size=path.stat().st_size
        # Streaming audit measures reconstruction without deliberately retaining
        # O(n^2) logical snapshots in a Python list.
        started=time.perf_counter();entries=sum(1 for _ in agent.store.iter_entries(agent.id));audit_seconds=time.perf_counter()-started
        result={'backend':backend,'workload':workload,'events':count,'write_seconds':write_seconds,
            'snapshot_seconds':snapshot_seconds,'streaming_audit_seconds':audit_seconds,
            'database_bytes':size,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'logical_digest':digest(agent.get_state()),'journal_hash':snapshot['provenance']['journal_hash'],
            'journal_entries':entries,'python':platform.python_version()}
        agent.close();return result


def run(output,counts=(25,100,250),repeats=3,backends=('snapshot','python','rust'),workloads=('append','clocked')):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('output directory must be empty')
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    raw_stream=(output/'raw.jsonl').open('w')
    for workload in workloads:
        for count in counts:
            for repeat in range(repeats):
                # Rotate execution order rather than always giving one backend a cold run.
                ordered=list(backends[repeat%len(backends):])+list(backends[:repeat%len(backends)])
                for backend in ordered:
                    raw=subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'--worker',backend,'--count',str(count),'--workload',workload],text=True)
                    row=json.loads(raw);row['repeat']=repeat;rows.append(row)
                    raw_stream.write(json.dumps(row)+'\n');raw_stream.flush()
                    print(json.dumps(row),flush=True)
    for workload in workloads:
        for count in counts:
            subset=[r for r in rows if r['events']==count and r['workload']==workload]
            if len({r['logical_digest'] for r in subset})!=1 or len({r['journal_hash'] for r in subset})!=1:
                raise AssertionError('storage backends changed logical history')
    raw_stream.close()
    summary={'schema_version':1,'python':platform.python_version(),'counts':list(counts),'repeats':repeats,
             'backends':list(backends),'workloads':list(workloads),'all_histories_identical':True,'measurements':{},
             'scope':'SQLite journal representation and optional Rust JSON-diff kernel; no agent capability improvement claim'}
    for workload in workloads:
        summary['measurements'][workload]={}
        for count in counts:
            summary['measurements'][workload][str(count)]={}
            for backend in backends:
                subset=[r for r in rows if r['events']==count and r['backend']==backend and r['workload']==workload]
                summary['measurements'][workload][str(count)][backend]={k:statistics.median(r[k] for r in subset) for k in ('write_seconds','snapshot_seconds','streaming_audit_seconds','database_bytes','peak_rss_kib')}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results/memory-profile')
    p.add_argument('--counts',type=int,nargs='+',default=[25,100,250]);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--backends',nargs='+',default=['snapshot','python','rust']);p.add_argument('--worker');p.add_argument('--count',type=int)
    p.add_argument('--workload',choices=['append','clocked'],default='append');p.add_argument('--workloads',nargs='+',default=['append','clocked'])
    a=p.parse_args()
    if a.worker:print(json.dumps(worker(a.worker,a.count,a.workload)))
    else:run(a.output,a.counts,a.repeats,tuple(a.backends),tuple(a.workloads))
