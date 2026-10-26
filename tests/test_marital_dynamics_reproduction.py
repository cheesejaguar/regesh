"""Hand-calculated checks of the dyadic affect model and the marital_dynamics runner."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from regesh.bonding import dyad
from regesh.reproductions.marital_dynamics import (GROUPS, autoregressive, e4_records, grid_couple, published_couple,
                                                   quadrant, simulate_noisy_series, step_nullclines, _e3_task)


class MaritalDynamicsReproductionTests(unittest.TestCase):
    def test_step_against_hand_calculation(self):
        params = {"a": 1., "r1": .5, "b": -1., "r2": .4,
                  "I_hw": {"form": "ojive", "neg_threshold": -6., "neg_value": -2., "pos_threshold": 8., "pos_value": 3.},
                  "I_wh": {"form": "bilinear", "slope_neg": .5, "slope_pos": .3}}
        # W' = 1 + .5*2 + I_hw(10) = 1 + 1 + 3 ; H' = -1 + .4*10 + .3*2 = -1 + 4 + .6
        self.assertEqual(dyad.step(2., 10., params), (5., 3.6))
        # inside the thresholds the ojive contributes nothing; negative wife score uses the negative slope
        w1, h1 = dyad.step(-4., 3., params)
        self.assertAlmostEqual(w1, 1 + .5 * -4)
        self.assertAlmostEqual(h1, -1 + .4 * 3 + .5 * -4)
        # repair adds its amplitude only at or below the threshold; damping only at or above
        params["I_hw"]["repair"] = {"threshold": -7., "effectiveness": 4.}
        params["I_hw"]["damping"] = {"threshold": 9., "effectiveness": -2.}
        self.assertEqual(dyad.step(0., -7., params)[0], 1 - 2 + 4)
        self.assertEqual(dyad.step(0., 9., params)[0], 1 + 3 - 2)
        self.assertEqual(dyad.step(0., 8.5, params)[0], 1 + 3)

    def test_uninfluenced_steady_state_is_a_over_one_minus_r(self):
        params = {"a": 1., "r1": .5, "b": -1., "r2": .4, "I_hw": {"form": "none"}, "I_wh": {"form": "none"}}
        u = dyad.uninfluenced_steady_states(params)
        self.assertEqual(u["w"], 2.)
        self.assertAlmostEqual(u["h"], -1 / .6)
        final = dyad.simulate(params, 20., -20., 200)[-1]
        self.assertAlmostEqual(final[0], 2.)
        self.assertAlmostEqual(final[1], -1 / .6)
        # the published-group construction inverts u = a/(1-r)
        for group in GROUPS:
            ar = autoregressive(group)
            u = dyad.uninfluenced_steady_states({**ar, "I_hw": {"form": "none"}, "I_wh": {"form": "none"}})
            self.assertAlmostEqual(u["w"], dyad.PUBLISHED_1999[group]["wife"]["uninfluenced"])
            self.assertAlmostEqual(u["h"], dyad.PUBLISHED_1999[group]["husband"]["uninfluenced"])

    def test_symmetric_ojive_gives_three_strict_and_five_completed_steady_states(self):
        thresholds = {"I_hw": {"neg": -6., "pos": 8.}, "I_wh": {"neg": -6., "pos": 8.}}
        params = grid_couple(thresholds, .4, .4, 0., 0., 8, 8)
        fixed = dyad.steady_states(params)
        self.assertEqual(len(fixed), 3)
        self.assertTrue(all(s["stable"] for s in fixed))
        np.testing.assert_allclose([(s["w"], s["h"]) for s in fixed],
                                   [(-8 / .6, -8 / .6), (0., 0.), (8 / .6, 8 / .6)], atol=1e-9)
        res = step_nullclines(params)
        self.assertEqual((res["n_fixed"], res["n_jump_crossings"], res["n_completed"]), (3, 2, 5))
        self.assertEqual(res["types"], ["NN", "ZZ", "PP"])
        # the smooth sigmoid reading of the same couple has five fixed points: three stable, two unstable
        smooth = grid_couple(thresholds, .4, .4, 0., 0., 8, 8, "sigmoid", 1.)
        numeric = dyad.steady_states(smooth)
        self.assertEqual(len(numeric), 5)
        self.assertEqual(sum(s["stable"] for s in numeric), 3)
        # small plateaus leave only the uninfluenced point
        self.assertEqual(step_nullclines(grid_couple(thresholds, .4, .4, 0., 0., 1, 1))["n_completed"], 1)

    def test_constructed_five_piece_couple_has_five_strict_fixed_points(self):
        # Ojive plus repair and damping triggers: five pieces per spouse, hence up to five strict fixed points.
        spec = {"form": "ojive", "neg_threshold": -3., "neg_value": -6., "pos_threshold": 3., "pos_value": 6.,
                "repair": {"threshold": -8., "effectiveness": 4.}, "damping": {"threshold": 8., "effectiveness": -4.}}
        params = {"a": 0., "r1": .5, "b": 0., "r2": .5, "I_hw": spec, "I_wh": spec}
        fixed = dyad.steady_states(params)
        self.assertEqual([(s["w"], s["h"]) for s in fixed], [(-12., -4.), (-4., -12.), (0., 0.), (4., 12.), (12., 4.)])
        self.assertTrue(all(s["stable"] for s in fixed))
        for s in fixed:
            self.assertEqual(dyad.step(s["w"], s["h"], params), (s["w"], s["h"]))
        # a plain three-piece step ojive never has more than three strict fixed points
        rng = np.random.default_rng(5)
        for _ in range(200):
            thresholds = {"I_hw": {"neg": -rng.uniform(2, 10), "pos": rng.uniform(2, 10)}, "I_wh": {"neg": -rng.uniform(2, 10), "pos": rng.uniform(2, 10)}}
            params = grid_couple(thresholds, rng.uniform(0, .9), rng.uniform(0, .9), rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(0, 12), rng.uniform(0, 12))
            self.assertLessEqual(len(dyad.steady_states(params)), 3)

    def test_published_algorithm_recovers_autoregression_exactly_from_partner_zeros(self):
        params = {"a": .8, "r1": .35, "b": -.5, "r2": .6,
                  "I_hw": {"form": "ojive", "neg_threshold": -5., "neg_value": -3., "pos_threshold": 7., "pos_value": 2.},
                  "I_wh": {"form": "ojive", "neg_threshold": -6., "neg_value": -4., "pos_threshold": 8., "pos_value": 3.}}
        # husband scores with zeros at known times; wife follows the model exactly
        h = np.array([0., 5., 0., -9., 0., 8., 0., -2., 0., 11., 0., 0.], float)
        w = np.empty_like(h); w[0] = 3.
        for t in range(len(h) - 1):
            w[t + 1] = params["a"] + params["r1"] * w[t] + dyad.influence(h[t], params["I_hw"])
        fit = dyad.fit_published_algorithm(w, h, zero_tol=0., form="ojive", threshold_grid=np.arange(1, 13, 1.))
        self.assertEqual(fit["w"]["subset"], int(np.sum(h[:-1] == 0)))
        self.assertAlmostEqual(fit["w"]["inertia"], .35, places=9)
        self.assertAlmostEqual(fit["w"]["constant"], .8, places=9)
        self.assertAlmostEqual(fit["w"]["uninfluenced"], .8 / .65, places=9)
        self.assertAlmostEqual(fit["w"]["influence"]["neg_value"], -3., places=9)
        self.assertAlmostEqual(fit["w"]["influence"]["pos_value"], 2., places=9)
        # the wife never scores zero, so the husband's equation cannot be fitted: a measured limitation, not an error
        self.assertIsNone(fit["h"])

    def test_quadrant_rule_and_bilinear_fixed_point_by_hand(self):
        self.assertEqual([quadrant(1, 1), quadrant(1, -1), quadrant(-1, 1), quadrant(0, 0)], ["++", "+-", "-+", "--"])
        params = {"a": .6, "r1": .4, "b": .6, "r2": .4,
                  "I_hw": {"form": "bilinear", "slope_neg": .5, "slope_pos": .3},
                  "I_wh": {"form": "bilinear", "slope_neg": .5, "slope_pos": .3}}
        # symmetric positive fixed point: w = .6 + .4 w + .3 w  ->  w = 2
        fixed = dyad.steady_states(params)
        self.assertEqual(len(fixed), 1)
        self.assertAlmostEqual(fixed[0]["w"], 2.)
        self.assertAlmostEqual(fixed[0]["h"], 2.)
        self.assertTrue(fixed[0]["stable"])

    def test_simulation_and_runner_tasks_are_deterministic(self):
        params = published_couple("stable", 3, 3)
        noise = np.random.default_rng(3).normal(0, 3, (149, 2))
        a = simulate_noisy_series(params, 1, -2, 149, noise, (-24, 24))
        b = simulate_noisy_series(params, 1, -2, noise.shape[0], noise.copy(), (-24, 24))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.dtype, np.int16)
        self.assertTrue(np.all(np.abs(a) <= 24))
        cfg = {"plateau": [3, 3], "noise_sd": [1, 3, 5], "initial_range": [-6, 6], "series_length": 150,
               "threshold_grid": {"start": 1, "stop": 24, "step": 1}, "zero_tol_primary": 0, "zero_tol_sensitivity": 1,
               "zero_tol_sensitivity_noise_sd": [3], "rls": {"lam": .98, "prior": 100, "noise_sd": [3]}, "base_seed": 4300}
        recs1, series1 = _e3_task((cfg, "divorced", 3, [0, 1]))
        recs2, series2 = _e3_task((cfg, "divorced", 3, [0, 1]))
        self.assertEqual(recs1, recs2)
        np.testing.assert_array_equal(series1, series2)
        self.assertEqual(set(recs1[0]["estimates"]), {"published_tol0", "published_tol1", "joint_bilinear", "rls"})
        e4 = {"groups": ["stable"], "slopes": [[.5, .3]], "repair_effectiveness": [3.99], "repair_thresholds": [-4.64],
              "damping_effectiveness": [-3.11], "damping_thresholds": [6], "cycle_check": {"burn_in": 50, "window": 20, "max_period": 10}}
        self.assertEqual(e4_records(e4), e4_records(e4))
        self.assertEqual([r["condition"] for r in e4_records(e4)], ["none", "damping", "repair", "both"])

    def test_runner_refuses_non_empty_output(self):
        from regesh.reproductions.marital_dynamics import run
        config_path = Path(__file__).resolve().parents[1] / "configs" / "reproductions" / "marital_dynamics.json"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"; out.mkdir(); (out / "existing").write_text("x")
            with self.assertRaisesRegex(ValueError, "empty"):
                run(config_path, out)
            config = json.loads(config_path.read_text())
            self.assertEqual(config["published_1999"], dyad.PUBLISHED_1999)
            self.assertEqual(config["e3_recovery"]["replications"], 1000)


if __name__ == "__main__":
    unittest.main()
