"""Independent equation, learning-path, and boundary checks for Figure 6."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from regesh.reproductions.homeostatic import (drive, increase_probability,
    q_learning_update, run, simulate)
from regesh.reproductions.homeostatic_analysis import analyze


class HomeostaticReproductionTests(unittest.TestCase):
    def test_original_drive_definition_and_reward(self):
        np.testing.assert_allclose(drive([-16, 0, 16]), [16**(4/3), 0, 16**(4/3)])
        # Moving closer has positive reward; going away has the opposite sign.
        self.assertGreater(drive(4) - drive(3), 0)
        self.assertLess(drive(4) - drive(5), 0)

    def test_softmax_convention_explicit_and_stable(self):
        self.assertEqual(float(increase_probability(0, 0, .05)), .5)
        self.assertAlmostEqual(float(increase_probability(0, 2, .05)),
                               1/(1+np.exp(-.1)))
        self.assertGreater(float(increase_probability(0, 2, .05, "temperature")), .999)
        self.assertEqual(float(increase_probability(1e6, -1e6, .05)), 0)
        with self.assertRaises(ValueError):
            increase_probability(0, 0, .05, "unknown")

    def test_q_update_includes_future_value(self):
        self.assertAlmostEqual(float(q_learning_update(2, 3, 4, .4, .9)), 3.84)

    def test_no_learning_is_actual_random_walk(self):
        result = simulate(1, 15, 55, initial=0, learn=False, examples=1)
        rng = np.random.default_rng(55)
        expected = np.r_[0, np.cumsum(2*(rng.random(15)<.5).astype(int)-1)]
        np.testing.assert_array_equal(result["example_states"][:, 0], expected)
        self.assertEqual(int(result["visits"].sum()), 15)

    def test_agents_have_independent_q_tables_and_saved_replay(self):
        first = simulate(8, 150, 1)
        second = simulate(8, 150, 1)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(np.array_equal(first["final_q"][0], first["final_q"][1]))

    def test_no_reflecting_or_clipping_boundary(self):
        with self.assertRaisesRegex(RuntimeError, "no clipping"):
            simulate(10, 100, 10, initial=0, radius=2, learn=False)

    def test_discounted_reward_identity_includes_terminal_correction(self):
        # S5 holds exactly when the trajectory terminates at the setpoint.
        gamma=.9
        states=np.array([4.,5.,3.,1.,0.])
        deviations=drive(states)
        discounts=gamma**np.arange(4)
        reward_sum=np.sum(discounts*(deviations[:-1]-deviations[1:]))
        drive_sum=np.sum(discounts*deviations[1:])
        self.assertAlmostEqual(reward_sum, deviations[0]+(gamma-1)*drive_sum)

    def test_recorded_config_hash_and_manifest_close_over_raw_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config={"parameters":{"initial":0,"radius":128},
                    "cohort_conditions":[{"name":"published_inverse_beta","parameters":{}}],
                    "cohort_seeds":[1,2], "agents_per_batch":4,
                    "cohort_steps":12,"long_joint_seed":3,
                    "long_replicates":2,"long_steps":12}
            path=root/'config.json';path.write_text(json.dumps(config))
            metadata=root/'sources.json';metadata.write_text(json.dumps({
                "config_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}))
            with contextlib.redirect_stdout(io.StringIO()):
                result=run(path,root/'out',metadata)
            self.assertEqual(result['cohorts'][0]['agents'],8)
            self.assertEqual(result['long']['visits'],24)
            manifest=json.loads((root/'out/manifest.json').read_text())
            for name,digest in manifest['files'].items():
                self.assertEqual(hashlib.sha256((root/'out'/name).read_bytes()).hexdigest(),digest)
            audit=analyze(root/'out',root/'audit.json')
            self.assertEqual(audit['conditions'][0]['independent_summary_recomputation']['n'],8)
            with self.assertRaisesRegex(ValueError,'preserve prior evidence'):
                analyze(root/'out',root/'audit.json')
            with self.assertRaisesRegex(ValueError,'preserve prior evidence'):
                run(path,root/'out',metadata)
            path.write_text(json.dumps({**config,'agents_per_batch':5}))
            with self.assertRaisesRegex(ValueError,'prospective'):
                run(path,root/'mismatched',metadata)
            (root/'out/summary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                analyze(root/'out',root/'tampered-audit.json')


if __name__ == "__main__":
    unittest.main()
