import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from regesh import Agent, Event
from regesh.bonding import attachment, dyad, synchrony
from regesh.storage import digest


def ev(i, actor='user', **data):
    return Event(str(i), 'chat', actor, 'agent', 'conversation', '2026-10-25T00:00:00Z', data=data)


class AttachmentEquations(unittest.TestCase):
    def test_hiolle_equations_by_hand(self):
        # Eq. 4 with Table 1 needy parameters, then trace decay
        self.assertAlmostEqual(attachment.comfort_step(.5, 1, 0, 3, .7), (0.5 * 3 + 1) / 4)
        self.assertAlmostEqual(attachment.comfort_step(.5, 0, 0, 3, .7), .35)
        # Eq. 3: window average without comfort; direct decrement with comfort
        self.assertAlmostEqual(attachment.arousal_step(.5, .8, 0., 5, .6), (5 * .5 + .8) / 6)
        self.assertAlmostEqual(attachment.arousal_step(.5, .8, .4, 5, .6), .5 - .6 * .4)
        self.assertEqual(attachment.arousal_step(.1, 0., .9, 5, .6), 0.)  # floored
        # Eq. 5 keeps the published (1-Resp) decrease; Eqs. 6-7 ranges
        self.assertAlmostEqual(attachment.responsiveness_step(.5, .5, .03), .515)
        self.assertAlmostEqual(attachment.responsiveness_step(.5, 0., .03), .485)
        self.assertEqual(attachment.profile_from_responsiveness(1.), {'beta_h': .7, 'tau_h': 3.})
        self.assertEqual(attachment.profile_from_responsiveness(0.), {'beta_h': 1., 'tau_h': 10.})
        self.assertEqual([attachment.behavior(x) for x in (.7, .6, .5, .4, .1)], ['request', 'request', 'learn', 'explore', 'explore'])


class DyadModel(unittest.TestCase):
    def test_step_and_uninfluenced_steady_state(self):
        p = {'a': .6, 'r1': .4, 'b': -.3, 'r2': .5,
             'I_hw': {'form': 'bilinear', 'slope_neg': .5, 'slope_pos': .2},
             'I_wh': {'form': 'bilinear', 'slope_neg': .5, 'slope_pos': .2}}
        self.assertEqual(dyad.step(2., -4., p), (.6 + .8 - 2., -.3 - 2. + .4))
        u = dyad.uninfluenced_steady_states(p)
        self.assertAlmostEqual(u['w'], 1.); self.assertAlmostEqual(u['h'], -.6)

    def test_ojive_gives_one_three_or_five_fixed_points(self):
        base = lambda a, b, nv, pv: {'a': a, 'r1': .4, 'b': b, 'r2': .4,
            'I_hw': {'form': 'ojive', 'neg_threshold': -6, 'neg_value': nv, 'pos_threshold': 8, 'pos_value': pv},
            'I_wh': {'form': 'ojive', 'neg_threshold': -6, 'neg_value': nv, 'pos_threshold': 8, 'pos_value': pv}}
        self.assertEqual(len(dyad.steady_states(base(0., 0., -5, 5))), 3)
        self.assertEqual(len(dyad.steady_states(base(0., 0., -1, 1))), 1)
        counts = {len(dyad.steady_states(base(a, b, -5, 5))) for a in np.linspace(-6, 6, 25) for b in np.linspace(-6, 6, 25)}
        self.assertLessEqual(max(counts), 3)   # step ojive: each nullcline segment meets at most one other
        self.assertTrue(all(s['stable'] for s in dyad.steady_states(base(0., 0., -5, 5))))
        smooth = {'a': 0., 'r1': .4, 'b': 0., 'r2': .4,
                  'I_hw': {'form': 'sigmoid', 'neg_threshold': -6, 'neg_value': -8, 'pos_threshold': 8, 'pos_value': 8, 'width': 1.},
                  'I_wh': {'form': 'sigmoid', 'neg_threshold': -6, 'neg_value': -8, 'pos_threshold': 8, 'pos_value': 8, 'width': 1.}}
        points = dyad.steady_states(smooth)
        self.assertEqual(len(points), 5)
        self.assertTrue(any(not s['stable'] for s in points))  # smooth ramps add unstable saddles

    def test_published_estimator_exact_on_noiseless_data(self):
        rng = np.random.default_rng(1)
        p = {'a': .5, 'r1': .4, 'b': -.2, 'r2': .3,
             'I_hw': {'form': 'ojive', 'neg_threshold': -6, 'neg_value': -3, 'pos_threshold': 8, 'pos_value': 2},
             'I_wh': {'form': 'ojive', 'neg_threshold': -6, 'neg_value': -3, 'pos_threshold': 8, 'pos_value': 2}}
        h = rng.integers(-12, 13, 200).astype(float); h[::3] = 0.
        w = [2.]
        for t in range(199):
            w.append(p['a'] + p['r1'] * w[-1] + dyad.influence(h[t], p['I_hw']))
        fit = dyad.fit_published_algorithm(np.array(w), h)
        self.assertAlmostEqual(fit['w']['inertia'], .4, places=6); self.assertAlmostEqual(fit['w']['constant'], .5, places=6)
        self.assertAlmostEqual(fit['w']['influence']['neg_value'], -3, places=6)

    def test_recursive_identifier_and_bounded_steering(self):
        rng = np.random.default_rng(0); ident = dyad.RecursiveIdentifier(); p = 0.
        for _ in range(300):
            a = rng.uniform(-1, 1); nxt = .1 + .5 * p + .6 * min(a, 0) + .3 * max(a, 0) + rng.normal(0, .02)
            ident.update(p, a, nxt); p = nxt
        est = ident.params()
        self.assertAlmostEqual(est['inertia'], .5, delta=.05); self.assertAlmostEqual(est['slope_neg'], .6, delta=.08)
        out = dyad.steer(ident, -.2, own_valence=.1, cap=.3)
        self.assertLessEqual(abs(out['expressed_valence'] - .1), .3 + 1e-9)
        fresh = dyad.steer(dyad.RecursiveIdentifier(), -.6, own_valence=0., cap=.5)
        self.assertFalse(fresh['model_trusted']); self.assertTrue(fresh['repair'])


class SynchronyMeasures(unittest.TestCase):
    def test_lsm_identity_and_disjoint(self):
        text = 'I think that we should go to the store now'
        self.assertAlmostEqual(synchrony.lsm(text, text), 1.)
        # Categories absent from both texts count as matched (Ireland et al. 2011 formula), so a
        # content-only reply still scores on the categories the first text did not use.
        self.assertLess(synchrony.lsm(text, 'Quantum chromodynamics predicts asymptotic freedom'), .6)
        self.assertLess(synchrony.lsm(text, 'never not no none nothing nobody nowhere'), synchrony.lsm(text, 'we should go to the store'))

    def test_coordination_matches_conditional_probability(self):
        c = synchrony.CoordinationCounts()
        c.add('do you want the book', 'yes I want the book')
        c.add('why', 'because the cat'); c.add('ok', 'fine')
        s = c.summary()['by_marker']
        self.assertAlmostEqual(s['ppron'], 1 - 1 / 3); self.assertAlmostEqual(s['article'], 1 - 2 / 3)
        self.assertIsNone(s['ipron'])
        rebuilt = synchrony.CoordinationCounts.from_dict(c.to_dict()).summary()
        self.assertEqual(rebuilt['by_marker'], s)

    def test_group_aggregations_handle_undefined_markers(self):
        c1 = synchrony.CoordinationCounts().add('the cat and a dog', 'the dog or a cat').add('plainly', 'the end').summary()
        c2 = synchrony.CoordinationCounts().add('a book', 'a pen').summary()
        g = synchrony.group_coordination([c1, c2])
        self.assertEqual(g['speakers'], 2); self.assertIsNone(g['aggregated_1']); self.assertIsNotNone(g['aggregated_3'])
        self.assertEqual(synchrony.accommodation_assessment(.1)['band'], 'over')
        self.assertEqual(synchrony.accommodation_assessment(.03)['band'], 'within')


class BondedAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.path = Path(self.tmp.name) / 'bond.sqlite'
    def tearDown(self):
        self.tmp.cleanup()

    def test_default_agent_state_is_unchanged_by_bonding_code(self):
        a = Agent(self.path, 'plain')
        try:
            self.assertNotIn('bonds', a.get_state()); self.assertNotIn('bonding', a.get_state()['config'])
            a.observe(ev(1, approval=.5, goal_delta={'quality': -.5}, control=.3))
            self.assertNotIn('bond', a.actuation_context())
            with self.assertRaises(ValueError): a.express('user', valence=.1)
            with self.assertRaises(ValueError): a.act('request')
        finally: a.close()

    def test_buffering_changes_threat_only_and_never_trust(self):
        agent = Agent(self.path, 'b', config={'bonding': True, 'partner': 'user'})
        try:
            for i in range(10):
                agent.observe(ev(i, goal_delta={'quality': .1}, support=.6, partner_affect=.2))
            base = agent.snapshot()
            agent.intervene({'bonds': {'user': {'responsiveness': 1.}}}, 'maximal measured responsiveness')
            strong = agent.actuation_context()['bond']['threat_buffer']
            agent.observe(ev('threat', actor='checker', goal_delta={'quality': -.9}, control=.1, uncertainty=.8))
            buffered = agent.get_state()
            agent.restore(base)
            agent.intervene({'bonds': {'user': {'responsiveness': 0.}}}, 'no measured responsiveness')
            agent.observe(ev('threat', actor='checker', goal_delta={'quality': -.9}, control=.1, uncertainty=.8))
            unbuffered = agent.get_state()
            self.assertGreater(strong, 0.)
            self.assertLess(buffered['affect']['threat']['threat'], unbuffered['affect']['threat']['threat'])
            for key in ('reliability', 'authority', 'familiarity'):
                self.assertEqual(buffered['relationships']['user'][key], unbuffered['relationships']['user'][key])
            self.assertEqual(buffered['internal']['uncertainty'], unbuffered['internal']['uncertainty'])
            self.assertEqual(buffered['expectations'], unbuffered['expectations'])
        finally: agent.close()

    def test_requests_are_budgeted_and_responsiveness_follows_evidence(self):
        agent = Agent(self.path, 'b', config={'bonding': True, 'partner': 'user', 'attention_cost': .5})
        try:
            agent.observe(ev(0, goal_delta={'quality': -.8}, control=.2, uncertainty=.9))
            r0 = agent.get_state()['bonds']['user']['responsiveness']
            agent.act('request', reason={'partner': 'user'})
            with self.assertRaisesRegex(ValueError, 'pending'): agent.act('request', reason={'partner': 'user'})
            agent.observe(ev(1, goal_delta={'quality': 0.}, support=.9))      # answered: responsiveness rises
            s = agent.get_state(); r1 = s['bonds']['user']['responsiveness']
            self.assertGreater(r1, r0); self.assertEqual(s['attachment']['answered'], 1); self.assertEqual(s['bonds']['user']['attention'], .5)
            agent.act('request', reason={'partner': 'user'})                   # attention budget now exhausted
            agent.advance_time(3.)                                             # exceeds request_window: unanswered
            s = agent.get_state(); r2 = s['bonds']['user']['responsiveness']
            self.assertIsNone(s['attachment']['pending']); self.assertEqual(s['bonds']['user']['unanswered'], 1); self.assertLess(r2, r1)
            self.assertGreater(s['bonds']['user']['profile']['tau_h'], agent.get_state()['config']['profile']['tau_h'])  # drifts toward independent
            with self.assertRaisesRegex(ValueError, 'attention'): agent.act('request', reason={'partner': 'user'})
            agent.advance_time(12.)                                            # attention recovers with elapsed time
            self.assertGreater(agent.get_state()['bonds']['user']['attention'], .5)
            self.assertEqual(agent.get_state()['relationships']['user']['reliability'], .5)
        finally: agent.close()

    def test_express_feeds_dyad_coordination_and_reciprocity_and_replays(self):
        agent = Agent(self.path, 'b', config={'bonding': True, 'partner': 'user'})
        try:
            texts = ['I really do not think that this is going to work for us', 'you know, I am just not sure about it at all',
                     'well the plan is fine but the budget is not', 'maybe we should look at the other option instead']
            for i, t in enumerate(texts):
                agent.observe(ev(i, goal_delta={'quality': .1}, partner_affect=-.3 + .2 * i, text=t, disclosure={'depth': .5, 'emotional': True}))
                agent.express('user', valence=.1 * i, text='I think that the plan is fine and we can look at it', disclosure={'depth': .2})
            bond = agent.actuation_context()['bond']
            self.assertEqual(bond['coordination']['exchanges'], 4); self.assertIsNotNone(bond['coordination']['lsm_mean'])
            self.assertEqual(bond['dyad']['updates'], 3); self.assertEqual(agent.get_state()['bonds']['user']['dyad']['expressed_source'], 'express')
            self.assertGreater(bond['reciprocity']['debt'], 0.); self.assertGreater(bond['intimacy'], 0.)
            self.assertIn(bond['coordination']['band'], {'within', 'over', 'under'})
            copy = agent.replay('copy')
            try:
                self.assertEqual(copy.get_state(), agent.get_state()); self.assertEqual(copy.actuation_context(), agent.actuation_context())
            finally: copy.close()
        finally: agent.close()

    def test_schema_validates_bond_evidence_fields(self):
        with self.assertRaises(ValueError): ev(1, support=1.5)
        with self.assertRaises(ValueError): ev(1, partner_affect=-2)
        with self.assertRaises(ValueError): ev(1, disclosure={'depth': 2})
        with self.assertRaises(ValueError): ev(1, disclosure={'depth': .5, 'emotional': 'yes'})
        ev(1, support=.5, partner_affect=-.5, responsiveness=.7, disclosure={'depth': .5})

    def test_arousal_band_modulates_exploration_but_not_verification(self):
        agent = Agent(self.path, 'b', config={'bonding': True, 'partner': 'user'})
        try:
            agent.observe(ev(0, goal_delta={'quality': .1}))
            base = agent.actuation_context()
            agent.intervene({'attachment': {'arousal': .9}}, 'high attachment arousal')
            high = agent.actuation_context()
            self.assertEqual(high['bond']['mode'], 'request'); self.assertTrue(high['bond']['request_support'])
            self.assertLess(high['exploration'], base['exploration']); self.assertEqual(high['verification'], base['verification'])
        finally: agent.close()


if __name__ == '__main__':
    unittest.main()
