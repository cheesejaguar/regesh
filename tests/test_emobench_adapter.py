"""Verify external-label conversion independently of any model output."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_emobench import convert_record,prepare,select_pilot

ROOT=Path(__file__).resolve().parents[1]


def artifact(payload):
    return {'filename':'emobench-EA.jsonl','dataset_id':'emobench','url':'https://example.invalid/source',
            'revision':'fixture','license':'MIT','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}


def record(qid='1'):
    return {'qid':qid,'language':'en','category':'category','question type':'Action',
            'scenario':'Fixture scenario.','subject':'Alex','choices':['choice one','choice two'],'label':'choice two'}


class EmoBenchAdapterChecks(unittest.TestCase):
    def test_gold_comes_from_exact_external_text(self):
        r=record();a=artifact(b'fixture');converted=convert_record(r,'EA',a)
        self.assertEqual(converted['gold'],'B')
        self.assertEqual(converted['choices'][converted['gold']],r['label'])
        self.assertEqual(converted['source']['download_sha256'],a['sha256'])
        r['label']='fabricated label'
        with self.assertRaises(ValueError):convert_record(r,'EA',a)
    def test_ambiguous_duplicate_gold_is_rejected(self):
        r=record();r['choices'].append(r['label'])
        with self.assertRaises(ValueError):convert_record(r,'EA',artifact(b'x'))
    def test_optional_cause_is_separate_keyed_subtask(self):
        r={'qid':'1','language':'en','coarse_category':'complex','scenario':'A scenario.',
           'subject':'Sam','emotion_choices':['happy','sad'],'emotion_label':'sad',
           'cause_choices':['event one','event two'],'cause_label':'event one'}
        a=artifact(b'x');emotion=convert_record(r,'EU',a);cause=convert_record(r,'EU',a,cause=True)
        self.assertEqual(emotion['gold'],'B');self.assertEqual(cause['gold'],'A')
        self.assertNotEqual(emotion['id'],cause['id'])
        self.assertEqual(emotion['source']['upstream_qid'],cause['source']['upstream_qid'])
    def test_hash_selection_is_order_invariant_and_covers_categories(self):
        items=[convert_record({**record(str(i)),'category':'ab'[i%2]},'EA',artifact(b'x')) for i in range(10)]
        first=select_pilot(items,4);second=select_pilot(list(reversed(items)),4)
        self.assertEqual(first,second)
        self.assertEqual({i['category'] for i in first},{'a','b'})
    def test_locked_pilot_recreated_from_fixture_equivalent_choices(self):
        # Verify complete locked pilot metadata against its declared deterministic selection.
        rows=[json.loads(line) for line in (ROOT/'datasets/emobench_pilot.jsonl').read_text().splitlines()]
        source=json.loads((ROOT/'datasets/emobench_pilot_source.json').read_text())
        self.assertEqual(len(rows),12)
        self.assertEqual(hashlib.sha256((ROOT/'datasets/emobench_pilot.jsonl').read_bytes()).hexdigest(),source['items_sha256'])
        for family,info in source['files'].items():
            subset=[r for r in rows if r['dataset']=='EmoBench-'+family]
            self.assertEqual([r['source']['upstream_qid'] for r in subset],info['selected_qids'])
            self.assertTrue(all(r['source']['download_sha256']==info['file_sha256'] for r in subset))
    def test_integrity_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/'cache';cache.mkdir()
            raw=(json.dumps(record())+'\n').encode();a=artifact(raw)
            (cache/a['filename']).write_bytes(raw+b'corrupt')
            catalog=root/'catalog.json';catalog.write_text(json.dumps({'downloads':[a]}))
            with self.assertRaisesRegex(ValueError,'integrity'):
                prepare(catalog,cache,root/'out')
            preserved=root/'existing';preserved.mkdir();(preserved/'items.jsonl').write_text('old')
            with self.assertRaisesRegex(ValueError,'preserved'):
                prepare(catalog,cache,preserved)
            self.assertEqual((preserved/'items.jsonl').read_text(),'old')


if __name__=='__main__':unittest.main()
