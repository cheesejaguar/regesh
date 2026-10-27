"""Equation, timing, determinism and provenance checks for the Hiolle et al. (2014) simulation."""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from regesh.bonding.attachment import (ADAPTIVE, PROFILES, arousal_step, behavior, comfort_step,
                                       perception_trace, profile_from_responsiveness, responsiveness_step)
from regesh.reproductions.attachment_robot import run, simulate_run, stimulation

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "reproductions" / "attachment_robot.json"


def load_config():
    return json.loads(CONFIG.read_text())


def cell(environment, profile, caregiver, **overrides):
    return {"name": f"{environment}_{profile}_{caregiver}", "experiment": "adaptive" if profile == "adaptive" else environment,
            "environment": environment, "profile": profile, "caregiver": caregiver, "overrides": overrides}


class AttachmentRobotReproductionTests(unittest.TestCase):
    def test_table_values_and_equations_against_hand_calculations(self):
        needy, independent = PROFILES["needy"], PROFILES["independent"]
        self.assertEqual((needy["tau_h"], needy["beta_h"], independent["tau_h"], independent["beta_h"]), (3, .7, 8, .95))
        self.assertEqual((ADAPTIVE["alpha_ar"], ADAPTIVE["tau_ar"], ADAPTIVE["alpha_resp"], ADAPTIVE["resp0"]), (.8, 3, .03, .5))
        # Eq. 1: a binary sensor sets the trace to one, then decays geometrically.
        self.assertEqual(perception_trace(0., 1, .9), 1.)
        self.assertAlmostEqual(perception_trace(1., 0, .9), .9)
        # Eq. 4 with Table 1: three steps of face plus head contact, then the trace branch.
        c = comfort_step(0., 1., 1., needy["tau_h"], needy["beta_h"])
        self.assertEqual(c, .5)
        c = comfort_step(c, 1., 1., needy["tau_h"], needy["beta_h"])
        self.assertEqual(c, .875)
        c = comfort_step(c, 1., 1., needy["tau_h"], needy["beta_h"])
        self.assertEqual(c, 1.15625)
        self.assertAlmostEqual(comfort_step(c, 0., 0., needy["tau_h"], needy["beta_h"]), .809375)
        self.assertAlmostEqual(comfort_step(0., 1., 1., independent["tau_h"], independent["beta_h"]), 2 / 9)
        # Eq. 3 with Table 1: exponential average without comfort, direct subtraction with it.
        self.assertAlmostEqual(arousal_step(0., .9, 0., needy["tau_ar"], needy["alpha_ar"]), .15)
        self.assertAlmostEqual(arousal_step(.7, .9, .5, needy["tau_ar"], needy["alpha_ar"]), .4)
        self.assertEqual(arousal_step(.2, .9, .5, needy["tau_ar"], needy["alpha_ar"]), 0.)
        self.assertAlmostEqual(arousal_step(.2, .9, .5, needy["tau_ar"], needy["alpha_ar"], floor=None), -.1)
        # Comf exactly at the .1 floor counts as no comfort.
        self.assertAlmostEqual(arousal_step(.3, .9, .1, 5, .6), (1.5 + .9) / 6)
        # Figure 4 thresholds: >= .6 request, > .4 learn, otherwise explore.
        self.assertEqual([behavior(a) for a in (.6, .5, .4, .39)], ["request", "learn", "explore", "explore"])
        # Eq. 5 with Table 2, including the published (1 - Resp) decrease term.
        self.assertAlmostEqual(responsiveness_step(.5, .5, ADAPTIVE["alpha_resp"]), .515)
        self.assertAlmostEqual(responsiveness_step(.5, .05, ADAPTIVE["alpha_resp"]), .485)
        self.assertAlmostEqual(responsiveness_step(.2, 0., ADAPTIVE["alpha_resp"]), .176)
        # Eqs. 6-7 with Table 2 ranges.
        self.assertEqual(profile_from_responsiveness(.5), {"beta_h": .85, "tau_h": 6.5})
        self.assertEqual(profile_from_responsiveness(1.), {"beta_h": .7, "tau_h": 3})
        self.assertEqual(profile_from_responsiveness(0.), {"beta_h": 1., "tau_h": 10})
        # Declared stimulation substitute.
        self.assertEqual(stimulation(1., 0., 7, 2., 0.), 1.)
        self.assertAlmostEqual(stimulation(1., 14., 7, 2., 0.), math.exp(-1))
        self.assertEqual(stimulation(1., 0., 7, 2., .05), 1.)
        self.assertEqual(stimulation(.85, 1e9, 2, 2., -.05), 0.)

    def test_same_seed_is_deterministic(self):
        config = load_config()
        for spec in (cell("complex", "needy", "responsive"), cell("complex", "adaptive", "schedule")):
            first, series_a = simulate_run(config, spec, 31000, keep_series=True)
            second, series_b = simulate_run(config, spec, 31000, keep_series=True)
            self.assertEqual(first, second)
            for key in series_a:
                np.testing.assert_array_equal(series_a[key], series_b[key])
        same_env_a, _ = simulate_run(config, cell("simple", "needy", "responsive"), 31001)
        same_env_b, _ = simulate_run(config, cell("simple", "independent", "responsive"), 31001)
        self.assertEqual(same_env_a["n_objects"], same_env_b["n_objects"])

    def test_caregiver_latency_is_one_second_of_three_steps(self):
        config = load_config()
        record, series = simulate_run(config, cell("simple", "independent", "responsive"), 31002, keep_series=True)
        self.assertGreater(record["responses"], 0)
        onset = int(np.flatnonzero(series["request_onset"])[0])
        active = series["intervention_active"]
        self.assertFalse(active[onset:onset + 3].any())
        self.assertTrue(active[onset + 3:onset + 6].all())
        self.assertFalse(active[onset + 6])
        self.assertTrue((series["comfort"][onset:onset + 3] <= .1).all())
        self.assertGreater(series["comfort"][onset + 3], .1)
        np.testing.assert_array_equal(series["face_trace"][onset + 3:onset + 6], [1., 1., 1.])
        self.assertAlmostEqual(series["face_trace"][onset + 6], .9)
        self.assertAlmostEqual(series["time_s"][onset + 3] - series["time_s"][onset], 1.)

    def test_needy_requests_at_least_as_often_as_independent(self):
        config = load_config()
        totals = {"needy": 0, "independent": 0}
        for seed in (31003, 31004, 31005):
            per_profile = {}
            for profile in totals:
                record, _ = simulate_run(config, cell("simple", profile, "responsive"), seed)
                self.assertLessEqual(record["duration_s"], 600.)
                per_profile[profile] = record["requests"]
                totals[profile] += record["requests"]
            self.assertGreaterEqual(per_profile["needy"], per_profile["independent"])
        self.assertGreater(totals["needy"], totals["independent"])

    def test_non_responsive_caregiver_yields_zero_comfort_and_profiles_coincide(self):
        config = load_config()
        records = {}
        for profile in ("needy", "independent"):
            record, series = simulate_run(config, cell("complex", profile, "nonresponsive"), 31006, keep_series=True)
            self.assertEqual(record["responses"], 0)
            self.assertEqual(record["comfort_peak"], 0.)
            self.assertEqual(float(series["comfort"].max()), 0.)
            self.assertFalse(series["intervention_active"].any())
            self.assertGreater(record["requests"], 0)
            records[profile] = {k: v for k, v in record.items() if k not in {"cell", "profile"}}
        # Table 1 profiles differ only through comfort, so without comfort their runs are identical.
        self.assertEqual(records["needy"], records["independent"])

    def test_literal_trace_condition_makes_beta_h_nearly_inert(self):
        config = load_config()
        sensor, _ = simulate_run(config, cell("simple", "needy", "responsive"), 31007)
        trace, _ = simulate_run(config, cell("simple", "needy", "responsive", comfort_condition="trace"), 31007)
        self.assertEqual(sensor["comfort_duration_max_s"], 3.)
        self.assertGreater(trace["comfort_duration_max_s"], 9.)

    def test_run_writes_closed_manifest_and_refuses_reuse(self):
        config = load_config()
        config["seeds"].update(published_count=[30000, 30001], adaptive_published_count=[30100], bootstrap=[31000, 31001, 31002])
        config["bootstrap_samples"] = 200
        keep = {"simple_needy_responsive", "simple_independent_responsive", "adaptive_literal_intervention3"}
        config["cells"] = [c for c in config["cells"] if c["name"] in keep]
        config["contrasts"] = [c for c in config["contrasts"] if c["a"] in keep and c["b"] in keep]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(path, root / "out")
            self.assertEqual(summary["runs"], 2 * 5 + 4)
            self.assertTrue(summary["pilot_calibration"]["criterion_met_for_independent"])
            targets = {c["target"] for c in summary["published_comparisons"]}
            self.assertIn("requests", targets)
            self.assertIn("adaptive_responsiveness_rises_then_falls", targets)
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(set(manifest["files"]) >= {"config.json", "source.py", "equations_source.py",
                                                         "runs.jsonl.gz", "summary.json"}, True)
            for name, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256((root / "out" / name).read_bytes()).hexdigest(), digest)
            self.assertTrue((root / "out" / "series_simple_needy_responsive.npz").exists())
            with self.assertRaisesRegex(ValueError, "preserve prior measurements"):
                run(path, root / "out")


if __name__ == "__main__":
    unittest.main()
