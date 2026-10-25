"""Lossless storage parity, legacy compatibility, and optional Rust FFI checks."""
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from regesh import Agent, Event
from regesh.schema import canonical
from regesh.native_memory import DeltaKernel, python_diff, apply_patch
from regesh.storage import DELTA_MARKER


def stimulus(i):
    return Event(str(i),'fixture','partner','goal','task','2026-10-25T00:00:00Z',
                 data={'goal_delta':{'quality':.3 if i%2 else -.3},'text':'evidence '*80})


def history(path,backend):
    with patch.dict(os.environ,{'REGESH_MEMORY_BACKEND':backend}):
        a=Agent(path,'a')
        for i in range(6):
            a.observe(stimulus(i));a.record_outcome(str(i),.1*i)
        a.act('verify');a.advance_time(2.)
        snap=a.snapshot();trace=a.trace()
        a.close()
    return snap,trace


class CompactMemoryTests(unittest.TestCase):
    def test_lossless_nested_changes_deletions_and_append(self):
        old={'a':[1,2],'b':{'gone':None,'stay':False},'number':1}
        new={'a':[1,2,3],'b':{'stay':True},'number':1.0,'new':{'x':'é'}}
        delta=python_diff(old,new)
        self.assertEqual(canonical(apply_patch(old,delta)),canonical(new))
        self.assertEqual(old['a'],[1,2])
        self.assertEqual(delta['append'],[[['a'],[3]]])
    def test_float_representation_and_signed_zero_preserved(self):
        old={'a':1,'b':-0.0,'big':2**100};new={'a':1.0,'b':0.0,'big':2**100+1}
        result=apply_patch(old,python_diff(old,new))
        self.assertEqual(canonical(result),canonical(new))
    def test_full_and_compact_journals_have_identical_logical_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            full,full_trace=history(Path(tmp)/'full.db','snapshot')
            compact,compact_trace=history(Path(tmp)/'compact.db','python')
            self.assertEqual(full['state'],compact['state'])
            self.assertEqual(full['digest'],compact['digest'])
            self.assertEqual(full['provenance']['journal_hash'],compact['provenance']['journal_hash'])
            for a,b in zip(full_trace,compact_trace):
                a.pop('wall_time');b.pop('wall_time')
                self.assertEqual(a,b)
    def test_mixed_legacy_and_compact_history_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mixed.db'
            with patch.dict(os.environ,{'REGESH_MEMORY_BACKEND':'snapshot'}):
                a=Agent(path,'a');a.observe(stimulus(0));a.close()
            with patch.dict(os.environ,{'REGESH_MEMORY_BACKEND':'python'}):
                a=Agent(path,'a');a.observe(stimulus(1));a.act('verify')
                rows=a.store.db.execute('SELECT state FROM journal WHERE agent=? ORDER BY seq',('a',)).fetchall()
                self.assertNotIn(DELTA_MARKER,json.loads(rows[1][0]))
                self.assertIn(DELTA_MARKER,json.loads(rows[-1][0]))
                copied=a.replay('replayed')
                self.assertEqual(a.get_state(),copied.get_state())
                copied.close();a.close()
    def test_iterator_returns_independent_prior_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Agent(Path(tmp)/'a.db');a.observe(stimulus(0));a.observe(stimulus(1))
            entries=list(a.store.iter_entries('x'))
            self.assertEqual(len(entries[0]['state']['events']),0)
            self.assertEqual(len(entries[1]['state']['events']),1)
            self.assertEqual(len(entries[2]['state']['events']),2)
            entries[2]['state']['events'].clear()
            self.assertEqual(len(entries[1]['state']['events']),1)
            a.close()
    def test_snapshot_does_not_materialize_entire_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Agent(Path(tmp)/'a.db');a.observe(stimulus(0))
            with patch.object(a.store,'entries',side_effect=AssertionError('quadratic trace materialization')):
                self.assertIn('journal_hash',a.snapshot()['provenance'])
            a.close()
    def test_corrupt_patch_fails_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Agent(Path(tmp)/'a.db');a.observe(stimulus(0))
            a.store.db.execute('UPDATE journal SET state=? WHERE agent=? AND seq=1',
                (canonical({DELTA_MARKER:{'set':[[['time'],999]],'delete':[],'append':[]}}),'x'))
            with self.assertRaisesRegex(ValueError,'integrity'):
                a.trace()
            a.close()


class RustKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:cls.kernel=DeltaKernel('rust')
        except RuntimeError:raise unittest.SkipTest('optional Rust library not built')
    def test_native_float_and_integer_parity(self):
        cases=[(1,1.0),(-0.0,0.0),(2**100,2**100+1),(.10000000000000002,.1),
               ({'a':[1,-0.0]}, {'a':[1,0.0,2]})]
        for old,new in cases:
            self.assertEqual(canonical(apply_patch(old,self.kernel.diff(old,new))),canonical(new))
            self.assertEqual(canonical(self.kernel.diff(old,new)),canonical(python_diff(old,new)))
    def test_random_nested_json_parity(self):
        rng=random.Random(19)
        def generate(depth):
            if depth and rng.random()<.6:
                if rng.random()<.5:return {str(i):generate(depth-1) for i in range(rng.randrange(5))}
                return [generate(depth-1) for _ in range(rng.randrange(5))]
            return rng.choice([None,True,False,rng.randrange(-100,100),rng.random(),-0.0,'é\\n\u0000'])
        for _ in range(300):
            old,new=generate(3),generate(3)
            native=self.kernel.diff(old,new)
            self.assertEqual(canonical(native),canonical(python_diff(old,new)))
            self.assertEqual(canonical(apply_patch(old,native)),canonical(new))
    def test_deep_python_valid_evidence_uses_observable_fallback(self):
        old,new=0,1
        for _ in range(150):old={'nested':old};new={'nested':new}
        before=self.kernel.native_fallbacks
        delta=self.kernel.diff(old,new)
        self.assertEqual(canonical(apply_patch(old,delta)),canonical(new))
        self.assertEqual(self.kernel.native_fallbacks,before+1)
        # Store supplies only serialized old input on the Rust fast path.
        serialized=self.kernel.diff(None,new,old_json=canonical(old),new_json=canonical(new))
        self.assertEqual(canonical(apply_patch(old,serialized)),canonical(new))
        self.assertEqual(self.kernel.native_fallbacks,before+2)

    def test_python_escaped_surrogates_roundtrip_with_observable_fallback(self):
        for new in [{'text':'\ud800'},{'\udfff':'value'}]:
            old={};before=self.kernel.native_fallbacks
            delta=self.kernel.diff(old,new)
            self.assertEqual(canonical(apply_patch(old,delta)),canonical(new))
            self.assertEqual(self.kernel.native_fallbacks,before+1)

    def test_native_and_python_agent_histories_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            rust,rt=history(Path(tmp)/'rust.db','rust')
            python,pt=history(Path(tmp)/'python.db','python')
            self.assertEqual(rust['state'],python['state'])
            self.assertEqual(rust['provenance']['journal_hash'],python['provenance']['journal_hash'])
            for a,b in zip(rt,pt):
                a.pop('wall_time');b.pop('wall_time');self.assertEqual(a,b)
