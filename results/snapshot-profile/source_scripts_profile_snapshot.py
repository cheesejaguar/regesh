#!/usr/bin/env python3
"""Isolated original snapshot materialization versus latest-hash snapshot lookup."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def worker(mode,count):
    os.environ['REGESH_MEMORY_BACKEND']='snapshot'
    from regesh import Agent,Event
    from regesh.storage import digest
    with tempfile.TemporaryDirectory() as tmp:
        agent=Agent(Path(tmp)/'s.db','a')
        for i in range(count):
            agent.observe(Event(str(i),'profile','person','target','task','2026-10-25T00:00:00Z',data={'text':'evidence.'*100}))
        baseline_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start=time.perf_counter()
        if mode=='legacy':
            # This is exactly the previous Agent.snapshot lookup behavior.
            state=agent.get_state()
            result={'state':state,'digest':digest(state),
                    'journal_hash':agent.store.entries(agent.id)[-1]['hash']}
        else:
            snapshot=agent.snapshot()
            result={'state':snapshot['state'],'digest':snapshot['digest'],
                    'journal_hash':snapshot['provenance']['journal_hash']}
        row={'mode':mode,'events':count,'snapshot_seconds':time.perf_counter()-start,
             'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'presnapshot_peak_rss_kib':baseline_rss,'digest':result['digest'],'journal_hash':result['journal_hash']}
        agent.close();return row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=['legacy','latest']);p.add_argument('--count',type=int,default=250);p.add_argument('--output',default='results/snapshot-profile')
    a=p.parse_args()
    if a.worker:print(json.dumps(worker(a.worker,a.count)))
    else:
        out=Path(a.output)
        if out.exists() and any(out.iterdir()):raise ValueError('output must be empty')
        out.mkdir(parents=True,exist_ok=True);rows=[]
        for mode in ['legacy','latest']:
            row=json.loads(subprocess.check_output([sys.executable,__file__,'--worker',mode,'--count',str(a.count)],text=True));rows.append(row);print(json.dumps(row),flush=True)
        assert rows[0]['digest']==rows[1]['digest'] and rows[0]['journal_hash']==rows[1]['journal_hash']
        (out/'results.json').write_text(json.dumps({'scope':'single isolated measurement each, not confidence intervals','rows':rows,'logical_results_equal':True},indent=2)+'\n')
