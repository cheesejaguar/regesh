import tempfile
import unittest
from pathlib import Path

import numpy as np

from regesh.reproductions import chain_emotion as ce


class ChainEmotionReproductionTests(unittest.TestCase):
    def test_holm_is_order_preserving_and_monotone(self):
        self.assertEqual(ce.holm([.02, .001, .4]), [.04, .003, .4])

    def test_wilson_bounds_and_symmetry(self):
        low, high = ce.wilson(21, 42)
        self.assertAlmostEqual(low+high, 1)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)

    def test_repeated_measures_eliminates_subject_intercepts(self):
        values = np.array([[1., 2., 4.], [2., 2., 5.], [4., 5., 5.], [3., 3., 6.]])
        result = ce.repeated_measures(values)
        shifted = ce.repeated_measures(values+np.array([100, -90, 50, 12])[:, None])
        self.assertAlmostEqual(result['F'], shifted['F'], places=8)
        self.assertEqual(result['df'], [2, 6])
        self.assertGreaterEqual(result['greenhouse_geisser_epsilon'], .5)
        self.assertLessEqual(result['greenhouse_geisser_epsilon'], 1)

    def test_repeated_measures_permuted_conditions_unchanged(self):
        values = np.array([[1., 2., 4.], [2., 2., 5.], [4., 5., 5.], [3., 3., 6.]])
        self.assertAlmostEqual(ce.repeated_measures(values)['F'], ce.repeated_measures(values[:, [2, 0, 1]])['F'])

    def test_bootstrap_uses_paired_differences(self):
        result = ce.paired_bootstrap(np.ones(5)*3, np.random.default_rng(1), 100)
        self.assertEqual(result, [3., 3.])

    def test_source_hash_mismatch_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'experiment_1_STEU.csv'
            path.write_text('corrupt')
            with self.assertRaises(ValueError):
                ce.source_files(Path(directory), fetch=False)
            self.assertEqual(path.read_text(), 'corrupt')

    def test_incomplete_user_pairing_rejected(self):
        with self.assertRaises(ValueError):
            ce.analyse_users([], np.random.default_rng(1), 100)

    def test_duplicate_steu_items_rejected(self):
        with self.assertRaises(ValueError):
            ce.analyse_steu([{'STEU': '1'}]*42, np.random.default_rng(1), 100)

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'results'
            output.mkdir()
            (output/'precious').write_text('existing')
            with self.assertRaises(ValueError):
                ce.run(output, Path(directory)/'cache')
            self.assertEqual((output/'precious').read_text(), 'existing')


if __name__ == '__main__':
    unittest.main()
