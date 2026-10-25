import json
from pathlib import Path
import tempfile
import unittest

from regesh.bonding import experiment as ex


class PlatformExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path('configs/bonding_platform.json').read_text())
        self.cfg.update(turns=12, seeds=2, bootstrap_samples=50); self.cfg['attachment']['turns'] = 12; self.cfg['dyad']['turns'] = 12

    def test_attachment_runs_are_deterministic_and_paired(self):
        a = ex.simulate_attachment(3, 1.0, True, self.cfg); b = ex.simulate_attachment(3, 1.0, True, self.cfg)
        self.assertEqual(a, b)
        ref = ex.simulate_attachment(3, 1.0, False, self.cfg)
        self.assertEqual(ref['setback_turns'], a['setback_turns'])    # common random numbers
        self.assertIsNone(ref['mean_arousal']); self.assertEqual(ref['requests'], 0)
        self.assertLessEqual(a['cumulative_threat'], ref['cumulative_threat'] + 1e-12)

    def test_dyad_policies_and_identifier_fields(self):
        for policy in self.cfg['dyad']['policies']:
            row = ex.simulate_dyad(1, 'stable_husband_1999', policy, self.cfg)
            self.assertEqual(len(row['trajectory_partner']), self.cfg['turns'])
            self.assertTrue(all(-1 <= v <= 1 for v in row['trajectory_partner']))
        mirror = ex.simulate_dyad(1, 'stable_husband_1999', 'mirror', self.cfg)
        self.assertEqual(mirror['mean_accommodation_distance'], 0.)
        ojive = ex.simulate_dyad(1, 'ojive_mismatch', 'steer', self.cfg)
        self.assertIn('identifier_inertia_error', ojive)

    def test_run_writes_manifest_and_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / 'cfg.json'; cfg_path.write_text(json.dumps(self.cfg))
            out = Path(tmp) / 'out'
            summary = ex.run(cfg_path, out, workers=1)
            self.assertIn('q=1.0|bonded', summary['attachment']['conditions'])
            self.assertIn('divorced_husband_1999|steer-own_mood', summary['dyad']['contrasts'])
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(set(manifest['files']) >= {'config.json', 'source.py', 'summary.json', 'attachment_runs.jsonl.gz', 'dyad_runs.jsonl.gz'}, True)
            with self.assertRaises(ValueError):
                ex.run(cfg_path, out, workers=1)


if __name__ == '__main__':
    unittest.main()
