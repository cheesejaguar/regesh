"""Fetch the pinned, Apache-2.0 Qwen CPU model used in the extended experiment.

The 1.12 GB model is cached outside Git; no credential or GPU is required.
The optional server build needs git, a C++ compiler and cmake.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

MODEL_REV='f86cb2c1fa58255f8052cc32aeede1b7482d4361'
MODEL_FILE='qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
MODEL_SHA='cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046'
LLAMA_REV='a55c16e4ee532aa36d55f5756f3ba28a21a2b62b'
SERVER_URL='https://github.com/ggml-org/llama.cpp/releases/download/b11878/llama-b11878-bin-ubuntu-x64.tar.gz'
SERVER_SHA='edadc40f684c3e44c7d1261eb72c993b98e66c3104555cbc43a0b261d773437f'


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--build-server',action='store_true')
    parser.add_argument('--prebuilt-server',action='store_true',help='Fetch verified Linux x64 CPU server b11878')
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True)
    model=args.cache/MODEL_FILE
    source=f'https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/{MODEL_REV}/{MODEL_FILE}'
    if not model.exists():
        temporary=model.with_suffix('.partial')
        with urllib.request.urlopen(source,timeout=120) as response,temporary.open('wb') as stream:
            for chunk in iter(lambda:response.read(4*1024*1024),b''):stream.write(chunk)
        if file_hash(temporary)!=MODEL_SHA:raise ValueError('model checksum mismatch')
        temporary.rename(model)
    if file_hash(model)!=MODEL_SHA:raise ValueError('existing model checksum mismatch')
    print(json.dumps({'model':str(model),'sha256':MODEL_SHA,'source':source,'license':'Apache-2.0'}))
    if args.prebuilt_server:
        import tarfile
        archive=args.cache/'llama-b11878-bin-ubuntu-x64.tar.gz'
        if not archive.exists():
            with urllib.request.urlopen(SERVER_URL,timeout=120) as response,archive.open('wb') as stream:
                for chunk in iter(lambda:response.read(1024*1024),b''):stream.write(chunk)
        if file_hash(archive)!=SERVER_SHA:raise ValueError('server archive checksum mismatch')
        if tuple(map(int,__import__('sys').version.split()[0].split('.')[:2]))<(3,12):
            raise ValueError('prebuilt extraction requires Python3.12+ data-filter support; use source build on3.11')
        with tarfile.open(archive) as stream:stream.extractall(args.cache,filter='data')
        print('Server executable:',args.cache/'llama-b11878/llama-server')
    if args.build_server:
        root=args.cache/'llama.cpp'
        if not root.exists():
            subprocess.run(['git','clone','--no-checkout','https://github.com/ggml-org/llama.cpp.git',str(root)],check=True)
        actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if actual!=LLAMA_REV:subprocess.run(['git','-C',str(root),'checkout',LLAMA_REV],check=True)
        subprocess.run(['cmake','-S',str(root),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release',
            '-DGGML_NATIVE=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF',
            '-DLLAMA_BUILD_UI=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_SERVER=ON'],check=True)
        subprocess.run(['cmake','--build',str(root/'build'),'-j',str(args.jobs),'--target','llama-server'],check=True)
        print('Server executable:',root/'build/bin/llama-server')


if __name__=='__main__':main()
