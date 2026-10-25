"""Optional Rust lossless-delta kernel with a behavior-equivalent Python fallback.

The native backend is opt-in through REGESH_MEMORY_BACKEND=rust or selected by
`auto` when a built library exists. The compact journal is useful without Rust.
"""
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import struct
from .schema import canonical


def _same_scalar(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack('!d', a) == struct.pack('!d', b)
    return a == b


def _same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x,y) for x,y in zip(a,b))
    return _same_scalar(a,b)


def python_diff(old, new):
    patch={'set': [], 'delete': [], 'append': []}
    def walk(a,b,path):
        if isinstance(a,dict) and isinstance(b,dict):
            for key in sorted(a.keys()-b.keys()):patch['delete'].append(path+[key])
            for key in sorted(b):
                if key in a:walk(a[key],b[key],path+[key])
                else:patch['set'].append([path+[key],deepcopy(b[key])])
        elif isinstance(a,list) and isinstance(b,list) and len(b)>len(a) and all(_same(x,y) for x,y in zip(a,b)):
            patch['append'].append([path,deepcopy(b[len(a):])])
        elif not _same(a,b):
            patch['set'].append([path,deepcopy(b)])
    walk(old,new,[])
    return patch


def apply_patch(old, patch):
    """Reconstruct one independent poststate; malformed patches fail explicitly."""
    if not isinstance(patch,dict) or set(patch)!={'set','delete','append'}:
        raise ValueError('invalid journal delta')
    state=deepcopy(old)
    def parent(path):
        node=state
        for key in path[:-1]:node=node[key]
        return node
    for path in patch['delete']:
        if not path:raise ValueError('cannot delete root')
        del parent(path)[path[-1]]
    for path,value in patch['set']:
        if not path:state=deepcopy(value)
        else:parent(path)[path[-1]]=deepcopy(value)
    for path,values in patch['append']:
        node=state
        for key in path:node=node[key]
        if not isinstance(node,list) or not isinstance(values,list):raise ValueError('invalid append delta')
        node.extend(deepcopy(values))
    return state


class DeltaKernel:
    def __init__(self,backend='auto'):
        if backend not in {'auto','python','rust','snapshot'}:
            raise ValueError('unknown memory backend')
        self.library=None
        self.native_fallbacks=0
        self.backend='python' if backend=='auto' else backend
        if backend not in {'auto','rust'}:return
        root=Path(__file__).resolve().parent.parent
        paths=[os.environ.get('REGESH_NATIVE_LIBRARY'),
               root/'native/target/release/libregesh_memory.so',
               root/'native/target/release/libregesh_memory.dylib',
               root/'native/target/release/regesh_memory.dll']
        found=next((Path(p) for p in paths if p and Path(p).is_file()),None)
        if found is None:
            if backend=='rust':raise RuntimeError('Rust memory library missing; cargo build --release --manifest-path native/Cargo.toml')
            return
        self.library=ctypes.CDLL(str(found))
        self.library.regesh_memory_diff.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_char_p,ctypes.c_size_t]
        self.library.regesh_memory_diff.restype=ctypes.c_void_p
        self.library.regesh_memory_free.argtypes=[ctypes.c_void_p]
        self.library.regesh_memory_free.restype=None
        self.backend='rust'

    def diff(self,old,new,*,old_json=None,new_json=None):
        if self.backend!='rust':return python_diff(old,new)
        a=(old_json if old_json is not None else canonical(old)).encode()
        b=(new_json if new_json is not None else canonical(new)).encode()
        pointer=self.library.regesh_memory_diff(a,len(a),b,len(b))
        if not pointer:raise RuntimeError('native delta kernel returned null')
        try:patch=json.loads(ctypes.string_at(pointer))
        finally:self.library.regesh_memory_free(pointer)
        if 'error' in patch:
            if 'recursion limit' in patch['error']:
                # Python-valid deeply nested evidence remains supported. This
                # exceptional fallback is observable for performance accounting.
                self.native_fallbacks+=1
                return python_diff(json.loads(old_json) if old is None and old_json is not None else old,new)
            raise ValueError(patch['error'])
        return patch
