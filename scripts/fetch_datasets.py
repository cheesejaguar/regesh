#!/usr/bin/env python3
"""Fetch only pinned, reviewed small artifacts; never execute dataset code.

The catalog is not permission to download every listed resource. Only its explicit
`downloads` allowlist is actionable. SHA-256/size checks are mandatory, existing
conflicting files are preserved, and network responses have a hard byte ceiling.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def fetch(catalog_path, cache, ids=None, verify_only=False):
    catalog=json.loads(Path(catalog_path).read_text())
    cache=Path(cache).expanduser();cache.mkdir(parents=True,exist_ok=True)
    available={r['dataset_id'] for r in catalog['downloads']}
    selected=set(ids or sorted(available))
    if selected-available:
        raise ValueError('No reviewed automatic download for: '+', '.join(sorted(selected-available)))
    rows=[]
    for record in catalog['downloads']:
        if record['dataset_id'] not in selected:continue
        filename=record['filename']
        if Path(filename).name!=filename or not filename:
            raise ValueError('Unsafe manifest filename')
        target=cache/filename
        if target.is_symlink():raise ValueError('Refusing symlink cache target')
        existed=target.exists()
        if existed:
            payload=target.read_bytes()
        elif verify_only:
            raise FileNotFoundError(target)
        else:
            if not record['url'].startswith('https://'):raise ValueError('HTTPS required')
            request=urllib.request.Request(record['url'],headers={'User-Agent':'regesh-dataset-fetch/1'})
            with urllib.request.urlopen(request,timeout=60) as response:
                payload=response.read(record['bytes']+1)
        actual=hashlib.sha256(payload).hexdigest()
        if len(payload)!=record['bytes'] or actual!=record['sha256']:
            raise ValueError(f'Integrity mismatch for {filename}; existing file preserved')
        if not existed:
            fd,name=tempfile.mkstemp(prefix='.regesh-',dir=cache)
            try:
                with os.fdopen(fd,'wb') as stream:stream.write(payload)
                # Same-process execution; avoid replacing any pre-existing user data.
                if target.exists():raise FileExistsError(target)
                os.link(name,target)
            finally:
                Path(name).unlink(missing_ok=True)
        rows.append({**record,'cache_path':str(target.resolve()),'verified':True,
                     'downloaded_this_run':not existed})
    provenance={'schema_version':1,'checked_at':datetime.now(timezone.utc).isoformat(),
                'catalog_sha256':hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(),
                'purpose':'evaluation material; not used for controller training',
                'artifacts':rows}
    if not verify_only:
        (cache/'fetch-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',default=str(Path(__file__).resolve().parent.parent/'datasets/catalog.json'))
    parser.add_argument('--cache',default=str(Path.home()/'.cache/regesh/datasets'))
    parser.add_argument('--ids',nargs='+')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.list:
        data=json.loads(Path(args.catalog).read_text())
        for r in data['datasets']:
            print(f"{r['id']}: {r['name']} [{r['access']}; {r['license']}]")
        return
    result=fetch(args.catalog,args.cache,args.ids,args.verify_only)
    print(json.dumps({'verified_artifacts':len(result['artifacts']),
        'bytes':sum(r['bytes'] for r in result['artifacts']),
        'datasets':sorted({r['dataset_id'] for r in result['artifacts']}),
        'cache':str(Path(args.cache).expanduser())},indent=2))

if __name__=='__main__':main()
