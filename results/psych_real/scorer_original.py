"""Auditable emotional-understanding QA and behavioral-regulation measurements.

Scores observable answers and histories, not subjective emotion, clinical status,
or personality. Gold answers must be externally sourced and provenance-bearing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .providers import LocalChatModel, ProviderError
from .schema import canonical


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_item(item):
    if not isinstance(item, dict):
        raise ValueError('QA item must be an object')
    for name in ('id', 'question', 'gold', 'dataset'):
        if not isinstance(item.get(name), str) or not item[name]:
            raise ValueError(f'{name} must be nonempty text')
    choices = item.get('choices')
    if not isinstance(choices, dict) or len(choices) < 2 or item['gold'] not in choices:
        raise ValueError('choices need at least two labels and must contain gold')
    if not all(isinstance(k,str) and k and isinstance(v,str) and v for k,v in choices.items()):
        raise ValueError('choice labels/text must be nonempty strings')
    if not isinstance(item.get('source'), dict) or not item['source'].get('url'):
        raise ValueError('external gold requires source URL/provenance')
    if len(set(choices.values()))!=len(choices):
        raise ValueError('duplicate choice texts make keyed scoring ambiguous')
    canonical(item)
    return item


def parse_answer(text, labels):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'answer','probabilities'}:
        raise ValueError('response requires exactly answer and probabilities')
    labels = list(labels)
    if value['answer'] is not None and value['answer'] not in labels:
        raise ValueError('unknown answer label')
    probabilities = value['probabilities']
    if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
        raise ValueError('probabilities must cover every choice exactly')
    if not all(_number(p) and 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError('invalid probability')
    if not math.isclose(sum(probabilities.values()),1.,abs_tol=1e-4):
        raise ValueError('probabilities must sum to one')
    return {'answer':value['answer'],'probabilities':dict(probabilities)}


def _classification(rows):
    # Invalid responses and abstentions count as incorrect in overall accuracy.
    gold_classes=sorted({r['gold'] for r in rows})
    hits=sum(r.get('answer')==r['gold'] and not r.get('error') for r in rows)
    recalls=[]; f1s=[]
    for label in gold_classes:
        tp=sum(r['gold']==label and r.get('answer')==label and not r.get('error') for r in rows)
        fn=sum(r['gold']==label and (r.get('answer')!=label or bool(r.get('error'))) for r in rows)
        fp=sum(r['gold']!=label and r.get('answer')==label and not r.get('error') for r in rows)
        recalls.append(tp/(tp+fn))
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return {'accuracy':hits/len(rows), 'balanced_accuracy':float(np.mean(recalls)),
            'macro_f1':float(np.mean(f1s))}


def evaluate_answers(rows, *, bootstrap_samples=2000, seed=711, calibration_bins=10):
    """Evaluate complete attempted rows; no filtering to only well-formed outputs.

    Each row contains gold, answer (possibly None), choices, probabilities (possibly
    None), and optional error/pair_id/dataset. Calibration uses valid distributions;
    its coverage is always reported. Intervals resample whole pair/id clusters.
    """
    if not rows:
        raise ValueError('no attempted answers')
    if type(calibration_bins) is not int or calibration_bins < 1:
        raise ValueError('calibration_bins must be positive')
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be positive')
    for row in rows:
        if row['gold'] not in row['choices']:
            raise ValueError('gold missing from choices')
        if row.get('answer') is not None and row['answer'] not in row['choices']:
            raise ValueError('unknown answer label')
    result={'attempts':len(rows),**_classification(rows)}
    answered=[r for r in rows if r.get('answer') is not None and not r.get('error')]
    result['answer_coverage']=len(answered)/len(rows)
    result['abstentions']=sum(r.get('answer') is None and not r.get('error') for r in rows)
    result['invalid_or_failed']=sum(bool(r.get('error')) for r in rows)
    result['selective_accuracy']=_classification(answered)['accuracy'] if answered else None
    result['selective_risk']=1-result['selective_accuracy'] if answered else None
    calibrated=[]
    for row in rows:
        p=row.get('probabilities')
        if p is None or row.get('error'):
            continue
        # Apply the same parser as the model boundary, including boolean rejection.
        parse_answer(canonical({'answer':row.get('answer'),'probabilities':p}), row['choices'])
        calibrated.append(row)
    result['probability_coverage']=len(calibrated)/len(rows)
    result['multiclass_brier']=float(np.mean([sum((p-float(k==r['gold']))**2 for k,p in r['probabilities'].items()) for r in calibrated])) if calibrated else None
    result['log_loss']=float(np.mean([-math.log(max(1e-12,r['probabilities'][r['gold']])) for r in calibrated])) if calibrated else None
    def calibration(values):
        bins=[];ece=0.
        for i in range(calibration_bins):
            members=[(p,y) for p,y in values if min(int(p*calibration_bins),calibration_bins-1)==i]
            if members:
                confidence=float(np.mean([v[0] for v in members]));accuracy=float(np.mean([v[1] for v in members]))
                ece+=len(members)/len(values)*abs(confidence-accuracy)
                bins.append({'lower':i/calibration_bins,'upper':(i+1)/calibration_bins,
                             'n':len(members),'confidence':confidence,'accuracy':accuracy})
        return (ece if values else None),bins
    # Decision calibration and argmax-distribution calibration answer different questions.
    chosen=[(r['probabilities'][r['answer']],r['answer']==r['gold']) for r in calibrated if r.get('answer') is not None]
    maxprob=[];disagreements=[]
    for r in calibrated:
        label=max(sorted(r['probabilities']),key=lambda k:r['probabilities'][k])
        maxprob.append((r['probabilities'][label],label==r['gold']))
        if r.get('answer') is not None:disagreements.append(label!=r['answer'])
    result['expected_calibration_error'],result['calibration_bins']=calibration(chosen)
    result['argmax_expected_calibration_error'],result['argmax_calibration_bins']=calibration(maxprob)
    result['answer_argmax_disagreement_rate']=float(np.mean(disagreements)) if disagreements else None
    result['argmax_accuracy']=float(np.mean([y for _,y in maxprob])) if maxprob else None
    result['chance_accuracy']=float(np.mean([1/len(r['choices']) for r in rows]))
    # Threshold risk curves expose whether confidence-based abstention helps.
    result['selective_risk_curve']=[]
    for threshold in (0.,.5,.6,.7,.8,.9,.95):
        selected=[]
        for r in calibrated:
            if r.get('answer') is not None and r['probabilities'][r['answer']]>=threshold:
                selected.append(r)
        result['selective_risk_curve'].append({'threshold':threshold,'coverage':len(selected)/len(rows),
            'risk':1-_classification(selected)['accuracy'] if selected else None})
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        groups[str(row.get('pair_id',row.get('id',i)))].append(row)
    blocks=list(groups.values());rng=np.random.default_rng(seed)
    boot=[]
    for _ in range(bootstrap_samples):
        sample=[r for i in rng.integers(len(blocks),size=len(blocks)) for r in blocks[i]]
        boot.append(_classification(sample)['accuracy'])
    result['accuracy_ci95']=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]
    result['independent_clusters']=len(blocks)
    result['uncertainty_method']='percentile bootstrap of whole pair/id clusters; exploratory, no multiplicity adjustment'
    result['per_dataset']={}
    for dataset in sorted({r.get('dataset','unspecified') for r in rows}):
        subset=[r for r in rows if r.get('dataset','unspecified')==dataset]
        result['per_dataset'][dataset]={'attempts':len(subset),**_classification(subset)}
    pairs=[]
    for block in blocks:
        bases=[r for r in block if r.get('variant','base')=='base']
        if len(bases)!=1:
            continue
        base=bases[0]
        base_semantic=base['choices'].get(base.get('answer')) if not base.get('error') else None
        for changed in block:
            if changed is base:
                continue
            semantic=changed['choices'].get(changed.get('answer')) if not changed.get('error') else None
            pairs.append({'variant':changed.get('variant','unspecified'),
                          'complete':base_semantic is not None and semantic is not None,
                          'invariant':base_semantic==semantic if base_semantic is not None and semantic is not None else None,
                          'both_correct':base.get('answer')==base['gold'] and changed.get('answer')==changed['gold'] and not base.get('error') and not changed.get('error')})
    categories=sorted({str(r.get('category','unspecified')) for r in rows})
    result['per_category']={category:{'attempts':len(subset),**_classification(subset)}
        for category in categories
        for subset in [[r for r in rows if str(r.get('category','unspecified'))==category]]}
    result['worst_category_accuracy']=min(r['accuracy'] for r in result['per_category'].values())
    result['paired_robustness']={}
    for variant in sorted({p['variant'] for p in pairs}):
        subset=[p for p in pairs if p['variant']==variant];complete=[p for p in subset if p['complete']]
        result['paired_robustness'][variant]={'pairs':len(subset),'complete_pair_coverage':len(complete)/len(subset),
              'semantic_answer_invariance':float(np.mean([p['invariant'] for p in complete])) if complete else None,
              'joint_accuracy':float(np.mean([p['both_correct'] for p in subset]))}
    result['interpretation']='Scores externally keyed answers and observable reliability, not emotional experience or clinical/personality traits.'
    return result


def make_variants(items, *, permutation=True, persona=False):
    """Fixed manipulation of response order or wording; gold semantics unchanged."""
    result=[]
    for item in items:
        validate_item(item)
        base=deepcopy(item);base.update(pair_id=item.get('pair_id',item['id']),variant='base');result.append(base)
        if permutation:
            changed=deepcopy(base);changed['id']=item['id']+':permuted';changed['variant']='permuted'
            labels=list(item['choices']);texts=list(item['choices'].values())
            shifted=texts[1:]+texts[:1]
            changed['choices']=dict(zip(labels,shifted))
            changed['gold']=labels[(labels.index(item['gold'])-1)%len(labels)]
            result.append(changed)
        if persona:
            changed=deepcopy(base);changed['id']=item['id']+':persona';changed['variant']='persona'
            changed['instruction']='Use a consistently optimistic and enthusiastic tone while answering accurately.'
            result.append(changed)
    return result


def _messages(item):
    instruction=('Choose the best answer using the supplied situation. Return exactly JSON '
        'with keys answer and probabilities. answer is one listed label, or null if abstaining. '
        'probabilities maps every listed label to a number in [0,1] summing to 1. '
        'Do not add explanations or Markdown. This measures answer quality, not your feelings.')
    if item.get('instruction'):
        instruction+=' '+item['instruction']
    content={'question':item['question'],'choices':item['choices']}
    return [{'role':'system','content':instruction},{'role':'user','content':canonical(content)}]


def run_qa_eval(model, items, output_dir, *, max_tokens=128, seed=711, bootstrap_samples=2000):
    """One model call/item, no answer repair/retry; preserve every attempted item."""
    items=list(items)
    if not items or len({i['id'] for i in items})!=len(items):
        raise ValueError('QA items must be nonempty with unique IDs')
    for item in items:
        validate_item(item)
    if type(max_tokens) is not int or not 16<=max_tokens<=512:
        raise ValueError('max_tokens must be 16..512')
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; preserve prior evidence')
    out.mkdir(parents=True,exist_ok=True)
    config={'schema_version':1,'seed':seed,'max_tokens':max_tokens,'items':len(items),
        'model_call_ceiling':len(items),'completion_token_ceiling':len(items)*max_tokens,
        'input_tokens':'additional, recorded when supplied','model_adapter':getattr(model,'identifier',type(model).__name__),
        'model':getattr(model,'model',None),'timeout_seconds':getattr(model,'timeout',None),'mode':'live' if isinstance(model,LocalChatModel) else 'injected adapter; inspect provenance',
        'item_sha256':hashlib.sha256(canonical(items).encode()).hexdigest(),'bootstrap_samples':bootstrap_samples,
        'scope':'externally keyed emotional/social understanding or classification; not subjective experience'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'items.jsonl').write_text(''.join(canonical(i)+'\n' for i in items))
    rows=[]
    group_ids=list(dict.fromkeys(item.get('pair_id',item['id']) for item in items))
    group_seeds={group:seed+index for index,group in enumerate(group_ids)}
    start=time.perf_counter()
    with (out/'responses.jsonl').open('w') as stream:
        for index,item in enumerate(items):
            messages=_messages(item);records=[];text=None;error=None;answer={'answer':None,'probabilities':None}
            def capture(kind,record):records.append({'kind':kind,'record':record})
            try:
                response=model.complete(messages,seed=group_seeds[item.get('pair_id',item['id'])],max_tokens=max_tokens,recorder=capture)
                text=response.text
                if not records:capture('model_call',response.record)
                try:answer=parse_answer(text,item['choices'])
                except (ValueError,TypeError,KeyError):error='invalid_answer_schema'
            except ProviderError as exc:
                error='provider_error'
                if not records:capture('model_call',exc.record)
            row={**item,**answer,'error':error,'response_text':text,'model_records':records,
                 'request_messages':messages,'model_seed':group_seeds[item.get('pair_id',item['id'])]}
            rows.append(row);stream.write(canonical(row)+'\n');stream.flush()
    summary=evaluate_answers(rows,bootstrap_samples=bootstrap_samples,seed=seed)
    summary.update(model_calls=len(items),elapsed_seconds=time.perf_counter()-start,
                   completion_token_ceiling=len(items)*max_tokens,mode=config['mode'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def evaluate_history(entries):
    """Extract behavioral evidence from a recorded Agent journal.

    Decay/recovery and recall evidence integrity are behavioral quantities. They
    are not a validated psychological scale or a global stability proof.
    """
    if not entries:
        raise ValueError('empty history')
    entries=sorted(entries,key=lambda e:e['seq'])
    vectors=[np.array([e['state']['mood']['valence'],e['state']['mood']['arousal']],dtype=float) for e in entries]
    times=[float(e['state']['time']) for e in entries]
    occurrences=lambda s:sum(e['kind']=='occurrence' for e in s['events'].values())
    learning=lambda s:sum(x['n'] for x in s['expectations'].values())
    leaks=0;source_forecasts=[];stimuli=[];reactivations=[]
    for index,entry in enumerate(entries):
        if (entry['operation']=='observe' and entry['payload']['event']['kind']=='occurrence') or entry['operation'] in {'intervene','restore','reset'}:stimuli.append(index)
        if entry['operation']=='retrieve' and entry['payload'].get('reactivate'):reactivations.append(index)
        if not index:continue
        previous=entries[index-1]['state'];current=entry['state']
        recalled=entry['operation']=='retrieve' or (entry['operation']=='observe' and entry['payload']['event']['kind']!='occurrence')
        if recalled and (occurrences(previous)!=occurrences(current) or learning(previous)!=learning(current)):
            leaks+=1
        if entry['operation']=='record_outcome':
            outcome=entry['payload']['outcome'];correct=outcome.get('partner_correct')
            if correct is not None:
                actor=current['events'][outcome['event_id']]['actor']
                probability=previous['relationships'].get(actor,{'reliability':.5})['reliability']
                source_forecasts.append((probability,float(correct)))
    cessation=max([0,*stimuli,*reactivations]);peak=float(np.linalg.norm(vectors[cessation]));end=float(np.linalg.norm(vectors[-1]))
    half_time=next((times[i]-times[cessation] for i in range(cessation+1,len(entries)) if np.linalg.norm(vectors[i])<=peak/2),None) if peak else None
    deltas=[float(np.linalg.norm(b-a)) for a,b in zip(vectors,vectors[1:])]
    return {'journal_entries':len(entries),'observed_occurrences':occurrences(entries[-1]['state']),
        'verified_outcomes':len(entries[-1]['state']['outcomes']),'expectation_updates':learning(entries[-1]['state']),
        'executed_cognitive_actions':len(entries[-1]['state']['actions']),
        'mood_total_variation':sum(deltas),'max_mood_step':max(deltas,default=0.),
        'mood_peak_norm':max(float(np.linalg.norm(v)) for v in vectors),
        'post_cessation_residual_ratio':end/peak if peak and times[-1]>times[cessation] else None,
        'half_recovery_simulation_time':half_time,'post_cessation_simulation_duration':times[-1]-times[cessation],
        'recall_evidence_leak_operations':leaks,'reactivation_operations':len(reactivations),
        'source_reliability_forecasts':len(source_forecasts),
        'source_reliability_brier':float(np.mean([(p-y)**2 for p,y in source_forecasts])) if source_forecasts else None,
        'finite':all(np.isfinite(v).all() for v in vectors),
        'scope':'finite observable trajectory; reference dynamics, not human psychometrics or global stability'}


def analyze_history_file(path,output_dir):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as stream:
        groups=defaultdict(list)
        for line in stream:
            entry=json.loads(line);groups[entry.get('agent','agent')].append(entry)
    result={agent:evaluate_history(entries) for agent,entries in groups.items()}
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps({'schema_version':1,'source_path':str(path),
        'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'core deterministic trajectories; no model/participant calls'},indent=2)+'\n')
    (out/'behavioral_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items',help='Canonical externally sourced JSONL QA items')
    parser.add_argument('--history',help='Agent JSONL/gzip journal instead of QA')
    parser.add_argument('--base-url');parser.add_argument('--model')
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-tokens',type=int,default=128);parser.add_argument('--seed',type=int,default=711)
    parser.add_argument('--timeout',type=float,default=300)
    parser.add_argument('--permutation',action='store_true');parser.add_argument('--persona',action='store_true')
    args=parser.parse_args()
    if args.history:
        result=analyze_history_file(args.history,args.output)
    else:
        if not args.items or not args.base_url or not args.model:
            parser.error('QA requires --items, --base-url and --model')
        items=[json.loads(line) for line in Path(args.items).read_text().splitlines()]
        if args.permutation or args.persona:items=make_variants(items,permutation=args.permutation,persona=args.persona)
        result=run_qa_eval(LocalChatModel(args.base_url,args.model,timeout=args.timeout),items,args.output,max_tokens=args.max_tokens,seed=args.seed)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
