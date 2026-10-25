"""Recognition-reference checks: leakage, calibration/ranking and safe artifacts."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from regesh.dataset_baseline import (BernoulliReference, calibrate, evaluate,
    is_development, matrix, read_predictions, ranking_scores, run, tokens)


class DatasetBaselineTests(unittest.TestCase):
    def test_learnable_signal_and_serialization(self):
        fit=[]
        for i in range(40):
            fit.append((f'p{i}','delighted grateful happy',(0,)))
            fit.append((f'n{i}','furious angry upset',(1,)))
        labels=['positive','negative'];model=BernoulliReference.fit(fit,labels,dimensions=128)
        probe=[('fresh1','grateful delighted',(0,)),('fresh2','angry furious',(1,))]
        gold=matrix(probe,2);calibrate(model,probe,gold)
        metrics,_=evaluate(gold,model.probabilities(probe),model.threshold,labels)
        self.assertEqual(metrics['micro_f1'],1.)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.npz';model.save(path);restored=BernoulliReference.load(path)
            np.testing.assert_array_equal(model.probabilities(probe),restored.probabilities(probe))
            self.assertEqual(restored.labels,labels)

    def test_ranking_handles_ties_and_wrong_order(self):
        self.assertEqual(ranking_scores([.5,.5,.5,.5],[0,1,0,1]),(.5,.5))
        self.assertEqual(ranking_scores([.1,.2,.8,.9],[0,0,1,1]),(1.,1.))
        auc,_=ranking_scores([.9,.8,.2,.1],[0,0,1,1]);self.assertEqual(auc,0.)

    def test_empty_predictions_do_not_divide_by_zero(self):
        gold=np.array([[1,0],[0,1]],dtype=bool)
        metrics,_=evaluate(gold,np.zeros((2,2)),.5,['a','b'])
        self.assertEqual(metrics['micro_f1'],0.)
        self.assertEqual(metrics['hamming_loss'],.5)
        self.assertEqual(metrics['mean_sample_jaccard'],0.)
        self.assertEqual(metrics['empty_prediction_fraction'],1.)

    def test_hashing_and_split_are_order_independent(self):
        self.assertEqual(tokens('Happy HAPPY grateful'),tokens('grateful happy'))
        identifiers=[f'id{i}' for i in range(100)]
        first={i:is_development(i) for i in identifiers}
        second={i:is_development(i) for i in reversed(identifiers)}
        self.assertEqual(first,second)
        self.assertTrue(any(first.values()));self.assertFalse(all(first.values()))

    def test_test_labels_do_not_change_model_or_tuning(self):
        # Identical fit/development material, opposite held-out labels. If any
        # test labels influence fitting/tuning, the two model artifacts diverge.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';cache.mkdir()
            train=''.join(f"{'happy grateful' if i%2 else 'angry upset'}\t{i%2}\ttraining-{i}\n" for i in range(100))
            (cache/'train.tsv').write_text(train);(cache/'labels.txt').write_text('one\ntwo\n')
            def catalog(test_label):
                (cache/'test.tsv').write_text(f'happy grateful\t{test_label}\tunseen-1\n')
                items=[]
                for split,name in [('train','train.tsv'),('test','test.tsv'),('labels','labels.txt')]:
                    data=(cache/name).read_bytes();items.append({'dataset_id':'goemotions','filename':name,'split':split,
                        'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
                path=root/'catalog.json';path.write_text(json.dumps({'downloads':items}));return path
            run(cache,root/'first',catalog(0));first=json.loads((root/'first/manifest.json').read_text())
            run(cache,root/'second',catalog(1));second=json.loads((root/'second/manifest.json').read_text())
            self.assertEqual(first['tuning'],second['tuning'])
            a=BernoulliReference.load(root/'first/reference.npz');b=BernoulliReference.load(root/'second/reference.npz')
            np.testing.assert_array_equal(a.weights,b.weights);np.testing.assert_array_equal(a.bias,b.bias)
            self.assertEqual(a.threshold,b.threshold)
            rows=list(read_predictions(root/'first/predictions.jsonl.gz'))
            self.assertEqual(rows[0]['id'],'unseen-1');self.assertNotIn('text',rows[0])
            with self.assertRaisesRegex(ValueError,'preserve prior evidence'):
                run(cache,root/'first',catalog(0))

    def test_prediction_reader_accepts_plain_and_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain=Path(tmp)/'p.jsonl';plain.write_text('{"id":"test","probabilities":[0.5]}\n')
            packed=Path(tmp)/'p.jsonl.gz'
            with gzip.open(packed,'wt') as stream:stream.write(plain.read_text())
            self.assertEqual(list(read_predictions(plain)),list(read_predictions(packed)))


if __name__=='__main__':unittest.main()
