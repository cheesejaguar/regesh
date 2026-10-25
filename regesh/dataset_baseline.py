"""Small human-label GoEmotions recognition reference, independent of agent affect.

Frozen main controllers are never loaded or changed. Only official training rows
fit the model; deterministic ID-hash development rows select temperature and a
single global multilabel threshold. Official test labels are used once for final
scoring. Outputs contain IDs and numerical labels, not original Reddit text.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import time

import numpy as np

TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
FEATURES = 4096
ALPHA = 1.0
TEMPERATURES = (.5, 1., 2., 4., 8.)
THRESHOLDS = tuple(float(x) for x in np.arange(.05, .81, .05))
SEED = 49261


def tokens(text, dimensions=FEATURES):
    import zlib
    return sorted({zlib.crc32(t.encode('utf-8')) % dimensions for t in TOKEN.findall(text.lower())})


def is_development(identifier):
    return int(hashlib.sha256(('regesh-goemotions-dev-v1:'+identifier).encode()).hexdigest()[:8], 16) % 10 == 0


def read_rows(path, classes=28):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            fields=line.rstrip('\n').split('\t')
            if len(fields)!=3: raise ValueError(f'Expected three TSV fields at row {number}')
            text, value, identifier=fields
            labels=tuple(sorted(set(int(v) for v in value.split(','))))
            if not identifier or not labels or any(v<0 or v>=classes for v in labels):
                raise ValueError('Invalid ID or human label')
            yield identifier, text, labels


def matrix(rows, classes):
    result=np.zeros((len(rows), classes),dtype=bool)
    for i,(_,_,labels) in enumerate(rows): result[i,list(labels)]=True
    return result


def sigmoid(logits):
    return 1/(1+np.exp(-np.clip(logits,-40,40)))


@dataclass
class BernoulliReference:
    weights: np.ndarray
    bias: np.ndarray
    prior: np.ndarray
    labels: list[str]
    temperature: float=1.
    threshold: float=.5

    @classmethod
    def fit(cls, rows, labels, dimensions=FEATURES, alpha=ALPHA):
        if not rows: raise ValueError('Training data is empty')
        class_count=np.zeros(len(labels),dtype=float)
        feature_count=np.zeros(dimensions,dtype=float)
        positive=np.zeros((len(labels),dimensions),dtype=float)
        for _,text,gold in rows:
            indices=tokens(text,dimensions)
            feature_count[indices]+=1
            class_count[list(gold)]+=1
            for label in gold: positive[label,indices]+=1
        n=len(rows)
        p1=(positive+alpha)/(class_count[:,None]+2*alpha)
        p0=(feature_count[None,:]-positive+alpha)/(n-class_count[:,None]+2*alpha)
        prior=(class_count+alpha)/(n+2*alpha)
        absent=np.log1p(-p1)-np.log1p(-p0)
        weights=np.log(p1)-np.log(p0)-absent
        bias=np.log(prior)-np.log1p(-prior)+absent.sum(axis=1)
        return cls(weights.astype(np.float32),bias.astype(np.float32),prior.astype(np.float32),list(labels))

    def logits(self, rows):
        result=np.empty((len(rows),len(self.labels)),dtype=np.float32)
        for i,(_,text,_) in enumerate(rows):
            indices=tokens(text,self.weights.shape[1])
            result[i]=self.bias+self.weights[:,indices].sum(axis=1)
        return result

    def probabilities(self, rows):
        return sigmoid(self.logits(rows)/self.temperature)

    def save(self,path):
        np.savez_compressed(path, schema_version=np.asarray(1),weights=self.weights,bias=self.bias,
            prior=self.prior,labels=np.asarray(self.labels,dtype='U32'),
            temperature=np.asarray(self.temperature),threshold=np.asarray(self.threshold))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            if int(data['schema_version'])!=1:raise ValueError('Unsupported baseline schema')
            weights=data['weights'].copy();bias=data['bias'].copy();prior=data['prior'].copy()
            labels=data['labels'].tolist();temperature=float(data['temperature']);threshold=float(data['threshold'])
        if weights.ndim!=2 or weights.shape[0]!=len(labels) or bias.shape!=(len(labels),) or prior.shape!=bias.shape:
            raise ValueError('Invalid model shape')
        if not all(np.isfinite(a).all() for a in (weights,bias,prior)) or temperature<=0 or not 0<threshold<1:
            raise ValueError('Invalid model numbers')
        return cls(weights,bias,prior,labels,temperature,threshold)


def micro_f1(gold,predicted):
    tp=int(np.sum(gold & predicted));fp=int(np.sum(~gold & predicted));fn=int(np.sum(gold & ~predicted))
    return 2*tp/max(1,2*tp+fp+fn)


def select_threshold(probabilities,gold):
    scores=[micro_f1(gold,probabilities>=threshold) for threshold in THRESHOLDS]
    # Deterministic tie break prefers the earlier/smaller declared threshold.
    i=int(np.argmax(scores))
    return THRESHOLDS[i],scores[i]


def calibrate(model,rows,gold):
    logits=model.logits(rows)
    scores=[float(np.mean((sigmoid(logits/temp)-gold)**2)) for temp in TEMPERATURES]
    i=int(np.argmin(scores));model.temperature=TEMPERATURES[i]
    probs=sigmoid(logits/model.temperature)
    model.threshold,f1=select_threshold(probs,gold)
    return {'criterion':'development Brier temperature, then development micro-F1 global threshold',
            'temperature_candidates':list(TEMPERATURES),'temperature_brier':scores,
            'temperature':model.temperature,'threshold_candidates':list(THRESHOLDS),
            'threshold':model.threshold,'development_micro_f1':f1,'development_brier':scores[i]}


def calibration_error(probabilities,gold,bins=10):
    p=np.asarray(probabilities).ravel();y=np.asarray(gold).ravel()
    locations=np.minimum((p*bins).astype(int),bins-1)
    rows=[];ece=0.
    for b in range(bins):
        mask=locations==b;n=int(mask.sum())
        confidence=float(p[mask].mean()) if n else None
        frequency=float(y[mask].mean()) if n else None
        if n:ece+=n/len(p)*abs(confidence-frequency)
        rows.append({'bin':b,'count':n,'mean_probability':confidence,'positive_frequency':frequency})
    return float(ece),rows


def ranking_scores(probabilities,gold):
    """Binary ROC-AUC and average precision with exact handling of score ties."""
    p=np.asarray(probabilities);y=np.asarray(gold,dtype=bool)
    positive=int(y.sum());negative=len(y)-positive
    if not positive or not negative:return None,None
    order=np.argsort(p,kind='stable')[::-1];s=p[order];truth=y[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(truth)[ends].astype(float);fp=(ends+1)-tp
    recall=tp/positive;precision=tp/(tp+fp)
    ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
    tpr=np.r_[0.,recall];fpr=np.r_[0.,fp/negative]
    auc=float(np.trapezoid(tpr,fpr)) if hasattr(np,'trapezoid') else float(np.trapz(tpr,fpr))
    return auc,ap


def evaluate(gold,probabilities,threshold,labels):
    predicted=probabilities>=threshold
    tp=(gold & predicted).sum(axis=0);fp=(~gold & predicted).sum(axis=0);fn=(gold & ~predicted).sum(axis=0)
    support=gold.sum(axis=0)
    precision=np.divide(tp,tp+fp,out=np.zeros(len(labels),dtype=float),where=(tp+fp)!=0)
    recall=np.divide(tp,tp+fn,out=np.zeros(len(labels),dtype=float),where=(tp+fn)!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(len(labels),dtype=float),where=(precision+recall)!=0)
    intersection=(gold & predicted).sum(axis=1);union=(gold | predicted).sum(axis=1)
    jaccard=np.divide(intersection,union,out=np.ones(len(gold),dtype=float),where=union!=0)
    ece,bins=calibration_error(probabilities,gold)
    by_class=[]
    for i,name in enumerate(labels):
        class_ece,_=calibration_error(probabilities[:,i],gold[:,i]);auc,ap=ranking_scores(probabilities[:,i],gold[:,i])
        by_class.append({'label':name,'support':int(support[i]),'predicted_count':int(predicted[:,i].sum()),
          'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),
          'brier':float(np.mean((probabilities[:,i]-gold[:,i])**2)),
          'ece':class_ece,'roc_auc':auc,'average_precision':ap})
    micro_auc,micro_ap=ranking_scores(probabilities.ravel(),gold.ravel())
    result={'threshold':threshold,'samples':len(gold),'labels':len(labels),
       'micro_precision':float(tp.sum()/max(1,(tp+fp).sum())),
       'micro_recall':float(tp.sum()/max(1,(tp+fn).sum())),
       'micro_f1':micro_f1(gold,predicted),'macro_f1':float(f1.mean()),
       'support_weighted_f1':float(np.dot(f1,support)/max(1,support.sum())),
       'hamming_loss':float(np.mean(gold!=predicted)),
       'exact_set_accuracy':float(np.mean(np.all(gold==predicted,axis=1))),
       'mean_sample_jaccard':float(jaccard.mean()),
       'brier':float(np.mean((probabilities-gold)**2)),
       'micro_ece':ece,'macro_class_ece':float(np.mean([r['ece'] for r in by_class])),
       'micro_roc_auc':micro_auc,'micro_average_precision':micro_ap,
       'macro_roc_auc':float(np.mean([r['roc_auc'] for r in by_class if r['roc_auc'] is not None])) if any(r['roc_auc'] is not None for r in by_class) else None,
       'macro_average_precision':float(np.mean([r['average_precision'] for r in by_class if r['average_precision'] is not None])) if any(r['average_precision'] is not None for r in by_class) else None,
       'mean_gold_labels':float(gold.sum(axis=1).mean()),'mean_predicted_labels':float(predicted.sum(axis=1).mean()),
       'empty_prediction_fraction':float(np.mean(predicted.sum(axis=1)==0)),
       'rare_labels_support_lt_50':[r for r in by_class if r['support']<50],
       'calibration_bins':bins,'per_class':by_class}
    return result,predicted


def paired_interval(gold,first,second,samples=1000):
    rng=np.random.default_rng(SEED)
    counts=[]
    for pred in (first,second):
        counts.append(np.asarray([(gold & pred).sum(axis=1),(~gold & pred).sum(axis=1),(gold & ~pred).sum(axis=1)]))
    differences=[]
    for _ in range(samples):
        chosen=rng.integers(0,len(gold),len(gold))
        a,b=[c[:,chosen].sum(axis=1) for c in counts]
        differences.append(2*a[0]/max(1,2*a[0]+a[1]+a[2])-2*b[0]/max(1,2*b[0]+b[1]+b[2]))
    return {'metric':'micro-F1 reference minus constant-prior baseline','bootstrap_seed':SEED,
        'samples':samples,'unit':'comment ID; unknown author/thread clustering not modeled',
        'difference':micro_f1(gold,first)-micro_f1(gold,second),
        'ci95':[float(np.quantile(differences,.025)),float(np.quantile(differences,.975))]}


def read_predictions(path):
    """Read plain or gzip numerical prediction artifacts without loading text."""
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cache,output,catalog_path=None):
    started=time.perf_counter();output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Output directory must be empty; preserve prior evidence')
    cache=Path(cache);catalog_path=Path(catalog_path or Path(__file__).resolve().parent.parent/'datasets/catalog.json')
    catalog=json.loads(catalog_path.read_text());sources=[]
    for item in catalog['downloads']:
        if item['dataset_id']=='goemotions':
            path=cache/item['filename']
            if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Pinned source integrity mismatch')
            sources.append({**item,'path':str(path.resolve())})
    source={r['split']:Path(r['path']) for r in sources}
    if set(source)!={'train','test','labels'}:raise ValueError('Expected pinned train/test/label artifacts')
    labels=source['labels'].read_text().splitlines()
    official_train=list(read_rows(source['train'],len(labels)))
    ids=[r[0] for r in official_train]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate official training IDs')
    fitting=[r for r in official_train if not is_development(r[0])]
    development=[r for r in official_train if is_development(r[0])]
    if not fitting or not development:raise ValueError('Empty fitting or development partition')
    # No official-test labels are loaded before all tuning is completed.
    train_started=time.perf_counter();model=BernoulliReference.fit(fitting,labels)
    training_seconds=time.perf_counter()-train_started
    dev_gold=matrix(development,len(labels));tuning=calibrate(model,development,dev_gold)
    prior_dev=np.tile(model.prior,(len(development),1))
    prior_threshold,prior_dev_f1=select_threshold(prior_dev,dev_gold)
    test=list(read_rows(source['test'],len(labels)));test_ids={r[0] for r in test}
    if len(test_ids)!=len(test) or test_ids & set(ids):raise ValueError('Duplicate or overlapping official-test ID')
    gold=matrix(test,len(labels));test_started=time.perf_counter();probabilities=model.probabilities(test)
    inference_seconds=time.perf_counter()-test_started
    result,predictions=evaluate(gold,probabilities,model.threshold,labels)
    prior_prob=np.tile(model.prior,(len(test),1));baseline,baseline_pred=evaluate(gold,prior_prob,prior_threshold,labels)
    interval=paired_interval(gold,predictions,baseline_pred)
    output.mkdir(parents=True,exist_ok=True)
    model.save(output/'reference.npz')
    (output/'split_ids.json').write_text(json.dumps({'fit':sorted(r[0] for r in fitting),
        'development':sorted(r[0] for r in development),'test':sorted(test_ids)},indent=2)+'\n')
    with gzip.GzipFile(str(output/'predictions.jsonl.gz'),mode='wb',mtime=0) as stream:
        for i,(identifier,_,human) in enumerate(test):
            row={'id':identifier,'gold_label_ids':list(human),
                 'predicted_label_ids':np.flatnonzero(predictions[i]).tolist(),
                 'probabilities':probabilities[i].tolist(),
                 'prior_predicted_label_ids':np.flatnonzero(baseline_pred[i]).tolist(),
                 'prior_probabilities':model.prior.tolist()}
            stream.write((json.dumps(row,separators=(',',':'))+'\n').encode())
    (output/'source_dataset_baseline.py').write_bytes(Path(__file__).read_bytes())
    artifact_hashes={name:sha(output/name) for name in ('reference.npz','split_ids.json','predictions.jsonl.gz','source_dataset_baseline.py')}
    manifest={'schema_version':1,'executed_at':datetime.now(timezone.utc).isoformat(),
      'scope':'human-label text-emotion recognition only; not functional agent affect or creativity',
      'sources':sources,'source_license':'Official card/repository declare Apache-2.0; upstream Reddit rights not independently guaranteed',
      'source_sha256':sha(__file__),'catalog_sha256':sha(catalog_path),'artifact_sha256':artifact_hashes,
      'seed':SEED,'features':FEATURES,'feature_rule':'binary lowercase token presence hashed with CRC32',
      'model':'one-vs-rest Bernoulli Naive Bayes','smoothing_alpha':ALPHA,
      'split_rule':'SHA256(regesh-goemotions-dev-v1:ID) first8 hex modulo10==0 development',
      'counts':{'official_train':len(official_train),'fit':len(fitting),'development':len(development),'test':len(test),'labels':len(labels)},
      'tuning':tuning,'baseline':{'type':'constant smoothed empirical fitting-label priors','threshold':prior_threshold,'development_micro_f1':prior_dev_f1},
      'runtime':{'python':platform.python_version(),'numpy':np.__version__,
        'thread_variables':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'training_seconds':training_seconds,'test_inference_seconds':inference_seconds,
        'total_seconds':time.perf_counter()-started,'process_peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
      'footprint':{'learned_array_bytes':model.weights.nbytes+model.bias.nbytes+model.prior.nbytes,
                   'compressed_model_bytes':(output/'reference.npz').stat().st_size},
      'unchanged_main_controllers':True,'live_model_calls':0,
      'limitations':['Base-model training contamination is irrelevant to this fitted-from-scratch reference but applies to future LLM comparisons.',
       'This evaluates perceived text emotion labels; it does not establish an agent has emotions.',
       'Bernoulli feature independence, hashing collisions and tokenization limit probability calibration.',
       'Micro ECE is dominated by the many negative label decisions; inspect class ECE and rare-label recall.',
       'The IID comment bootstrap may understate uncertainty if unknown authors or threads are correlated.',
       'Threshold and temperature are development-tuned; test numbers were not used to choose them.']}
    for filename,value in [('manifest.json',manifest),('metrics.json',{'reference':result,'constant_prior':baseline,'paired_micro_f1':interval})]:
        (output/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    lines=['# Executed GoEmotions recognition reference','',manifest['scope']+'.','',
       f"Fit {len(fitting)} / development {len(development)} / official test {len(test)} comments; {len(labels)} labels.",
       f"Hashed Bernoulli model: {manifest['footprint']['learned_array_bytes']:,} learned-array bytes; compressed artifact {manifest['footprint']['compressed_model_bytes']:,} bytes.",'',
       '| Metric | Learned reference | Constant fitting-prior |','|---|---:|---:|']
    for key in ('micro_f1','macro_f1','micro_precision','micro_recall','hamming_loss','exact_set_accuracy','mean_sample_jaccard','brier','micro_ece','macro_class_ece','micro_roc_auc','macro_roc_auc','micro_average_precision','macro_average_precision'):
        a="unavailable" if result[key] is None else f"{result[key]:.6f}"
        b="unavailable" if baseline[key] is None else f"{baseline[key]:.6f}"
        lines.append(f"| {key} | {a} | {b} |")
    lines+=['',f"Paired comment bootstrap micro-F1 difference {interval['difference']:.6f}, 95% interval {interval['ci95']}. Unknown author/thread clustering is not modeled.",'',
       'Raw numerical test predictions/IDs/gold labels are losslessly archived; original Reddit text is omitted. Per-class results and calibration bins are in metrics.json.',
       'All tuning uses development only. A prior baseline receives its own development-selected global threshold. Lower Hamming/Brier/ECE is better; higher F1/accuracy/Jaccard/ranking scores is better.',
       'This small recognition baseline is not an appraiser, persistent emotional controller, psychological exam result, or evidence about useful ideas.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return {'counts':manifest['counts'],'reference_micro_f1':result['micro_f1'],'prior_micro_f1':baseline['micro_f1'],
            'reference_macro_f1':result['macro_f1'],'footprint':manifest['footprint'],'output':str(output)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True);parser.add_argument('--output',default='results/goemotions_baseline')
    parser.add_argument('--catalog')
    args=parser.parse_args()
    # Bound this optional reference to one CPU even beside a running model server.
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    print(json.dumps(run(args.cache,args.output,args.catalog),indent=2))

if __name__=='__main__':main()
