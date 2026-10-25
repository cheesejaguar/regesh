#!/usr/bin/env python3
"""Convert pinned, licensed EmoBench files into canonical evaluation-only QA.

No network, model calls, generated labels, or rewriting of the locked live pilot.
The downloads allowlist supplies exact source checksums and revision metadata.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from regesh.psych_eval import validate_item


def convert_record(record, family, artifact, *, cause=False):
    if family not in {'EA','EU'} or cause and family!='EU':
        raise ValueError('unsupported subtask')
    if family=='EA':
        choices=record['choices'];answer=record['label']
        if record['question type'] not in {'Action','Response'}:
            raise ValueError('unknown author question type')
        stem=(f'What is the most appropriate action for {record["subject"]}?'
              if record['question type']=='Action' else
              f'What is the most appropriate response for {record["subject"]}?')
    else:
        choices=record['cause_choices'] if cause else record['emotion_choices']
        answer=record['cause_label'] if cause else record['emotion_label']
        stem=(f'What is the most likely cause of {record["subject"]}\'s emotion at the end of this situation?'
              if cause else f'Which emotion is {record["subject"]} most likely feeling at the end of this situation?')
    if not isinstance(choices,list) or choices.count(answer)!=1:
        raise ValueError('author answer must match exactly one original choice text')
    labels=[chr(65+i) for i in range(len(choices))]
    qid=str(record['qid']);language=record['language'];revision=artifact['revision']
    suffix='-cause' if cause else ''
    value={'id':f'emobench-{family}-{language}-{qid}{suffix}',
        'dataset':f'EmoBench-{family}'+suffix,
        'question':record['scenario']+'\n\n'+stem,
        'choices':dict(zip(labels,choices)),'gold':labels[choices.index(answer)],
        'category':record['category'] if family=='EA' else record['coarse_category'],
        'source':{'url':f'https://github.com/Sahandfer/EmoBench/blob/{revision}/data/{family}.jsonl',
                  'repository':'https://github.com/Sahandfer/EmoBench','commit':revision,
                  'license':artifact['license'],'upstream_qid':qid,
                  'download_sha256':artifact['sha256'],
                  'question_formulation':'Regesh fixed English stem over unchanged author scenario/choices/gold; no model-generated gold'}}
    return validate_item(value)


def select_pilot(items, per_kind=6, seed=711):
    if type(per_kind) is not int or per_kind<1:
        raise ValueError('per_kind must be a positive integer')
    result=[]
    for dataset in sorted({item['dataset'] for item in items}):
        family=dataset.removeprefix('EmoBench-')
        pool=sorted([r for r in items if r['dataset']==dataset],
            key=lambda r:hashlib.sha256(f'{seed}:{family}:{r["source"]["upstream_qid"]}'.encode()).hexdigest())
        categories=sorted({r['category'] for r in pool})
        if per_kind<len(categories) or per_kind>len(pool):
            raise ValueError('pilot size must cover categories and fit available records')
        selected=[next(r for r in pool if r['category']==category) for category in categories]
        for row in pool:
            if len(selected)>=per_kind:break
            if row not in selected:selected.append(row)
        result.extend(selected)
    return result


def prepare(catalog_path,cache_dir,output_dir,*,language='en',include_causes=False,pilot_per_kind=None,seed=711):
    if language!='en':
        raise ValueError('this converter supplies reviewed English question stems only')
    catalog_path=Path(catalog_path);catalog=json.loads(catalog_path.read_text())
    artifacts={r['filename']:r for r in catalog['downloads'] if r['dataset_id']=='emobench'}
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; existing pilot/evidence is preserved')
    items=[];inputs=[];clusters={}
    for family in ('EA','EU'):
        filename=f'emobench-{family}.jsonl';artifact=artifacts[filename]
        path=Path(cache_dir)/filename;payload=path.read_bytes()
        if len(payload)!=artifact['bytes'] or hashlib.sha256(payload).hexdigest()!=artifact['sha256']:
            raise ValueError(f'integrity mismatch for {filename}')
        rows=[json.loads(line) for line in payload.decode().splitlines()]
        rows=[r for r in rows if r['language']==language]
        qids=[str(r['qid']) for r in rows]
        if len(qids)!=len(set(qids)):raise ValueError('duplicate source qid/language')
        for row in rows:
            for cause in ([False,True] if family=='EU' and include_causes else [False]):
                item=convert_record(row,family,artifact,cause=cause)
                items.append(item);clusters[item['id']]=f'emobench-{family}-{language}-{row["qid"]}'
        inputs.append({'filename':filename,'url':artifact['url'],'revision':artifact['revision'],
                       'license':artifact['license'],'sha256':artifact['sha256'],'language_records':len(rows)})
    if pilot_per_kind is not None:items=select_pilot(items,pilot_per_kind,seed)
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items).encode()
    manifest={'schema_version':1,'purpose':'evaluation-only; no training or tuning on locked evaluation items',
        'catalog_sha256':hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        'converter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs':inputs,'language':language,'include_cause_questions':include_causes,
        'selection':('all selected-language records' if pilot_per_kind is None else
                     f'hash rank seed{seed}; first one/category then fill to {pilot_per_kind}/kind'),
        'items':len(items),'counts':dict(Counter(r['dataset'] for r in items)),
        'items_sha256':hashlib.sha256(data).hexdigest(),
        'case_clusters':{r['id']:clusters[r['id']] for r in items},
        'case_cluster_note':'EU emotion and optional cause QA share one scenario; cluster them together for uncertainty estimates.'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'items.jsonl').write_bytes(data)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',default=str(Path(__file__).resolve().parent.parent/'datasets/catalog.json'))
    p.add_argument('--cache',default=str(Path.home()/'.cache/regesh/datasets'))
    p.add_argument('--output',required=True)
    p.add_argument('--include-causes',action='store_true')
    p.add_argument('--pilot-per-kind',type=int);p.add_argument('--seed',type=int,default=711)
    args=p.parse_args()
    result=prepare(args.catalog,args.cache,args.output,include_causes=args.include_causes,
                   pilot_per_kind=args.pilot_per_kind,seed=args.seed)
    print(json.dumps({k:v for k,v in result.items() if k!='case_clusters'},indent=2))


if __name__=='__main__':main()
