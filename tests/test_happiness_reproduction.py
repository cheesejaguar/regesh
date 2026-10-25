import unittest

import numpy as np

from regesh.reproductions.happiness import aspiration, parameters, random_argmax, simulate_block, update


class HappinessReproductionTests(unittest.TestCase):
    def test_update_equations_against_hand_calculation(self):
        q, counts = np.zeros((1, 2)), np.zeros((1, 2), int)
        update(q, counts, np.array([1]), np.array([4.0]), 3.5, None)
        update(q, counts, np.array([1]), np.array([2.0]), 3.5, None)
        self.assertEqual(q[0, 1], -.5)
        update(q, counts, np.array([0]), np.array([2.0]), 3.5, .1)
        self.assertAlmostEqual(q[0, 0], -.15)

    def test_exact_schedules_and_unsupported_cell(self):
        self.assertEqual([aspiration("stationary_spread", "dynamic_compare", t)
                          for t in [0, 49, 50, 100, 150, 200]], [2, 2, 2.5, 2.75, 3, 3.5])
        self.assertEqual(aspiration("constant_change", "dynamic_compare", 100), 2)
        with self.assertRaises(ValueError):
            aspiration("sudden_change", "dynamic_compare", 100)
        self.assertEqual(parameters("constant_change", "objective"), (.1, .8))

    def test_random_ties_never_reverse_unequal_order(self):
        values = np.array([[0, 0, 0], [1, 3, 2], [np.inf, np.inf, 0]])
        self.assertEqual(random_argmax(values, np.array([.9, .9, .9])).tolist(), [2, 1, 1])

    def test_ordinary_optimism_equivalence_and_determinism(self):
        config = {"rounds_per_block": 9, "arms": 10, "steps": 300,
                  "initial_q": 0., "reward_noise_sd": 1., "checkpoint_every": 100}
        for setting in ["stationary_spread", "stationary_close", "sudden_change", "constant_change"]:
            rows, curves = simulate_block(config, setting, 21)
            self.assertEqual((rows, curves), simulate_block(config, setting, 21))
            fixed = [r for r in rows if r["condition"] == "fixed_compare"]
            ordinary = [r for r in rows if r["condition"] == "optimistic_objective"]
            self.assertEqual([r["objective_reward"] for r in fixed], [r["objective_reward"] for r in ordinary])
            self.assertEqual(sum(r["action_disagreements_with_fixed"] for r in ordinary), 0)
            for a, b in zip(fixed, ordinary):
                self.assertAlmostEqual(a["subjective_reward"] - b["subjective_reward"],
                                       -config["steps"] * aspiration(setting, "fixed_compare", 0))


if __name__ == "__main__":
    unittest.main()
