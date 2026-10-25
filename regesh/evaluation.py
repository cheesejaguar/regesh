"""Deterministic core history experiments, distinct from creativity evaluation."""
import json
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from . import Agent, Event


def event(identifier, gain=0., **data):
    return Event(str(identifier),'fixture','partner','project','design','2026-10-25T00:00:00Z',
                 data={'goal_delta':{'quality':gain},**data})


def run_core_experiments(output_dir):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    measures={"schema_version":1,"evidence_class":"deterministic controller experiments",
              "seed":0,"live_provider_calls":0,"checks":{},"trajectories":{}}
    history_records=[]
    with TemporaryDirectory() as tmp:
        db=Path(tmp)/'histories.db'
        a=Agent(db,'history')
        a.observe(event('first',-.7,quality=-.7,control=.2))
        a.act('verify'); a.record_outcome('first',-.7,partner_correct=True,control=.2)
        before=a.snapshot();a.close();a=Agent(db,'history')
        measures['checks']['restart_exact']=before['state']==a.get_state()
        class Forbidden:
            def appraise(self,*args):raise AssertionError('provider must not run')
        a.provider=Forbidden()
        replay=a.replay('replayed');measures['checks']['replay_exact']=replay.get_state()==a.get_state();replay.close()
        a.close();a=Agent(db,'history')
        before_n=a.get_state()['expectations']['design']['n']
        a.observe(Event(**{**event('recall',-.7).to_dict(),'kind':'recall','causal_links':['first']}))
        measures['checks']['recall_no_new_learning']=a.get_state()['expectations']['design']['n']==before_n
        expectation0=a.appraise(event('probe',quality=-.7))['expected_quality']
        for j in range(10):
            e=event('learn'+str(j),-.7,quality=-.7,control=.2);a.observe(e);a.record_outcome(e.id,-.7,partner_correct=True,control=.2)
        expectation1=a.appraise(event('probe',quality=-.7))['expected_quality']
        measures['development']={'expected_quality_before':expectation0,'after':expectation1,
            'surprise_after':a.appraise(event('probe',quality=-.7))['surprise'],
            'partner_factual_reliability':a.get_state()['relationships']['partner']['reliability']}
        a.observe(event('correction',.8,quality=.8,control=.9));a.reappraise('first',evidence_ids=['correction'])
        measures['checks']['reappraisal_preserves_observation']=a.get_state()['events']['first']['data']['quality']==-.7
        measures['checks']['reappraisal_changes_interpretation']=a.get_state()['memory']['first']['interpretations'][-1]['appraisal']['gain']==.8
        # Numeric interventions, same evidence and agent history.
        frozen=a.snapshot();a.intervene({'mood':{'valence':-.9,'arousal':.9}},'negative intervention')
        negative=a.actuation_context();a.restore(frozen)
        a.intervene({'mood':{'valence':.9,'arousal':.1}},'positive intervention')
        positive=a.actuation_context()
        measures['control_intervention']={'negative':negative,'positive':positive,
           'changed_processes':[k for k in ('exploration','verification','retrieval_bias') if negative[k]!=positive[k]]}
        history_records.extend({"agent":a.id,**entry} for entry in a.trace())
        a.close()
        # Perturbed histories: selective retrieval plus reactivation, then cessation.
        for coupling in (True,False):
            for initial in (-.8,.8):
                name=f'coupling={coupling},initial={initial}'
                a=Agent(db,name,config={'memory_coupling':coupling})
                for j,gain in enumerate((-.8,.8,-.4,.4)):
                    a.observe(event('m'+str(j),gain,quality=gain))
                a.intervene({'mood':{'valence':initial}},'initial perturbation')
                rows=[]
                for step in range(60):
                    selected=a.retrieve('design',limit=1,reactivate=True) if step<30 else []
                    a.advance_time(1.)
                    s=a.get_state()
                    rows.append({'step':step,'valence':s['mood']['valence'],'arousal':s['mood']['arousal'],
                        'selected':[m['observation']['id'] for m in selected],
                        'factual_occurrences':len(s['events']), 'expectation_updates':sum(e['n'] for e in s['expectations'].values())})
                measures['trajectories'][name]=rows
                history_records.extend({"agent":a.id,**entry} for entry in a.trace())
                a.close()
        measures['stability']={}
        for coupling in (True,False):
            low=measures['trajectories'][f'coupling={coupling},initial=-0.8']
            high=measures['trajectories'][f'coupling={coupling},initial=0.8']
            measures['stability'][str(coupling)]={
                'separation_during_feedback':abs(low[29]['valence']-high[29]['valence']),
                'separation_after_cessation':abs(low[-1]['valence']-high[-1]['valence']),
                'negative_final':low[-1]['valence'],'positive_final':high[-1]['valence'],
                'max_final_10_step_change':max(abs(r['valence']-rows[j-1]['valence']) for rows in (low,high) for j,r in enumerate(rows) if j>=50),
                'scope':'finite synthetic trajectories, not global stability proof'}
    with gzip.open(out/'histories.jsonl.gz','wt') as stream:
        for record in history_records: stream.write(json.dumps(record)+'\n')
    (out/'core_experiments.json').write_text(json.dumps(measures,indent=2)+'\n')
    return measures


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/core')
    args=p.parse_args();result=run_core_experiments(args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='trajectories'},indent=2))
