"""Independent adversarial history and scientific boundary checks.

These fixtures test controller invariants; they are not evidence about live LLMs.
"""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from regesh.schema import Event


def event(identity='e1', *, kind='occurrence', gain=-0.7, quality=-0.5,
          actor='partner', target='task', task='engineering', approval=0.0):
    return Event(id=identity, source='independent-check', actor=actor, target=target,
                 task=task, timestamp='2026-10-25T00:00:00Z', kind=kind,
                 data={'goal_delta': {'quality': gain}, 'quality': quality,
                       'control': 0.4, 'uncertainty': 0.6, 'approval': approval})


class EventBoundaryChecks(unittest.TestCase):
    def test_nonfinite_confidence_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Event('id', 'source', 'actor', 'target', 'task', 'time', confidence=value)

    def test_nonfinite_nested_evidence_rejected(self):
        with self.assertRaises(ValueError):
            Event('id', 'source', 'actor', 'target', 'task', 'time', data={'nested': {'bad': float('nan')}})

    def test_unknown_event_semantics_rejected(self):
        with self.assertRaises(ValueError):
            event(kind='dream-is-real')


class HistoryChecks(unittest.TestCase):
    def setUp(self):
        from regesh import Agent
        self.Agent = Agent
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / 'agents.sqlite3')
        self.agent = Agent(self.db, agent_id='original')

    def tearDown(self):
        self.tmp.cleanup()

    def test_outcome_identity_is_idempotent(self):
        self.agent.observe(event())
        self.agent.record_outcome('e1', quality=0.6, partner_correct=True)
        before = self.agent.get_state()
        self.agent.record_outcome('e1', quality=0.6, partner_correct=True)
        self.assertEqual(before, self.agent.get_state())

    def test_conflicting_outcome_rolls_back(self):
        self.agent.observe(event())
        self.agent.record_outcome('e1', quality=0.6)
        before = self.agent.get_state()
        with self.assertRaises(ValueError):
            self.agent.record_outcome('e1', quality=-0.6)
        self.assertEqual(before, self.agent.get_state())

    def test_quoted_event_cannot_have_observed_outcome(self):
        self.agent.observe(event(kind='quotation'))
        before = self.agent.get_state()
        with self.assertRaises(ValueError):
            self.agent.record_outcome('e1', quality=1.0)
        self.assertEqual(before, self.agent.get_state())

    def test_postvalidation_evidence_mutation_is_rejected(self):
        mutated = event()
        mutated.data['quality'] = float('nan')
        before = self.agent.get_state()
        with self.assertRaises(ValueError):
            self.agent.observe(mutated)
        self.assertEqual(before, self.agent.get_state())

    def test_invalid_intervention_rolls_back_all_fields(self):
        before = self.agent.get_state()
        with self.assertRaises(ValueError):
            self.agent.intervene({'mood': {'valence': 0.5}, 'internal': {'resources': -1}}, reason='adversarial invalid intervention')
        self.assertEqual(before, self.agent.get_state())

    def test_bad_snapshot_rejected_without_modification(self):
        before = self.agent.get_state()
        snapshot = self.agent.snapshot()
        snapshot['state']['mood']['valence'] = 0.9
        with self.assertRaises(ValueError):
            self.agent.restore(snapshot)
        self.assertEqual(before, self.agent.get_state())

    def test_goal_changes_appraisal_of_identical_evidence(self):
        stimulus = event(gain=0.8)
        positive = self.agent.appraise(stimulus)
        self.agent.intervene({'goals': {'quality': -1}}, reason='goal counterfactual')
        negative = self.agent.appraise(stimulus)
        self.assertGreater(positive['gain'], 0)
        self.assertLess(negative['gain'], 0)

    def test_learning_changes_future_appraisal(self):
        first = self.agent.appraise(event('future', quality=0.8))
        for i in range(5):
            self.agent.observe(event(f'lesson-{i}', gain=0.8, quality=0.8))
            self.agent.record_outcome(f'lesson-{i}', quality=0.8, control=0.9)
        after = self.agent.appraise(event('future', quality=0.8))
        self.assertGreater(after['expected_quality'], first['expected_quality'])
        self.assertLess(after['surprise'], first['surprise'])

    def test_reappraisal_versions_without_changing_observation(self):
        self.agent.observe(event())
        original = copy.deepcopy(self.agent.get_state()['events']['e1'])
        self.agent.observe(event('correction', gain=0.7, quality=0.5))
        before = self.agent.get_state()
        revised = self.agent.reappraise('e1', evidence_ids=['correction'], reason='independent correction')
        state = self.agent.get_state()
        self.assertEqual(original, state['events']['e1'])
        self.assertEqual(before['outcomes'], state['outcomes'])
        self.assertEqual(len(state['memory']['e1']['interpretations']), 2)
        self.assertGreater(revised['gain'], 0)
        self.assertIn('correction', revised['support'])

    def test_causal_state_intervention_changes_multiple_controls(self):
        original = self.agent.actuation_context()
        self.agent.intervene({'mood': {'valence': -0.8, 'arousal': 0.9},
                              'internal': {'resources': 0.2, 'uncertainty': 0.9}},
                             reason='causal test')
        altered = self.agent.actuation_context()
        changed = [k for k in ('exploration', 'verification', 'retrieval_bias', 'planning_horizon') if original[k] != altered[k]]
        self.assertGreaterEqual(len(changed), 3)

    def test_resource_constraint_has_observable_consequence(self):
        self.agent.intervene({'internal': {'resources': 0.01}}, reason='resource depletion')
        before = self.agent.get_state()
        with self.assertRaises(ValueError):
            self.agent.act('explore')
        self.assertEqual(before, self.agent.get_state())
        self.agent.act('recover')
        self.agent.act('explore')
        self.assertGreater(len(self.agent.get_state()['actions']), 0)

    def test_evidence_retrieval_is_unaffected_by_mood(self):
        self.agent.observe(event('negative', gain=-0.8))
        self.agent.observe(event('positive', gain=0.8))
        self.agent.intervene({'mood': {'valence': -0.9}}, reason='negative retrieval context')
        negative_neutral = self.agent.retrieve(limit=2, evidence_only=True)
        negative_affective = self.agent.retrieve(limit=1)
        self.agent.intervene({'mood': {'valence': 0.9}}, reason='positive retrieval context')
        positive_neutral = self.agent.retrieve(limit=2, evidence_only=True)
        positive_affective = self.agent.retrieve(limit=1)
        self.assertEqual([m['observation']['id'] for m in negative_neutral],
                         [m['observation']['id'] for m in positive_neutral])
        self.assertNotEqual(negative_affective[0]['observation']['id'],
                            positive_affective[0]['observation']['id'])

    def test_feedback_reactivation_does_not_fabricate_learning(self):
        for i in range(8):
            self.agent.observe(event(f'feedback-{i}'))
        before = self.agent.get_state()
        for _ in range(24):
            self.agent.advance_time(1)
            self.agent.retrieve(limit=1, reactivate=True)
        after = self.agent.get_state()
        self.assertEqual(before['events'], after['events'])
        self.assertEqual(before['expectations'], after['expectations'])
        self.assertEqual(before['outcomes'], after['outcomes'])
        self.assertEqual(before['relationships']['partner']['reliability'],
                         after['relationships']['partner']['reliability'])
        self.assertGreater(sum(m['retrievals'] for m in after['memory'].values()), 0)
        feedback_valence = abs(after['mood']['valence'])
        self.agent.advance_time(500)
        self.assertLess(abs(self.agent.get_state()['mood']['valence']), feedback_valence * 0.01)

    def test_replay_does_not_call_provider(self):
        self.agent.observe(event())
        self.agent.act('verify')
        self.agent.intervene({'mood': {'valence': 0.8}}, reason='replay intervention')
        self.agent.record_outcome('e1', quality=0.6)
        expected = self.agent.get_state()
        class ForbiddenProvider:
            def appraise(self, *args, **kwargs):
                raise AssertionError('Live provider called during replay')
        self.agent.provider = ForbiddenProvider()
        replayed = self.agent.replay('provider-free')
        self.assertEqual(expected, replayed.get_state())

    def test_duplicate_is_idempotent(self):
        self.agent.observe(event())
        before = self.agent.get_state()
        self.agent.observe(event())
        self.assertEqual(before, self.agent.get_state())

    def test_conflicting_duplicate_rejected_and_atomic(self):
        self.agent.observe(event())
        before = self.agent.get_state()
        with self.assertRaises(ValueError):
            self.agent.observe(event(gain=0.9))
        self.assertEqual(before, self.agent.get_state())

    def test_restart_retains_exact_state(self):
        self.agent.observe(event())
        self.agent.advance_time(1.75)
        expected = self.agent.get_state()
        reopened = self.Agent(self.db, agent_id='original')
        self.assertEqual(expected, reopened.get_state())

    def test_agent_isolation_with_shared_event_identity(self):
        other = self.Agent(self.db, agent_id='other')
        untouched = other.get_state()
        self.agent.observe(event())
        self.assertEqual(untouched, other.get_state())
        other.observe(event(gain=0.9))
        self.assertNotEqual(self.agent.get_state()['mood'], other.get_state()['mood'])

    def test_fork_does_not_alias_parent(self):
        self.agent.observe(event())
        before = self.agent.get_state()
        fork = self.agent.fork('forked')
        self.assertEqual(before, fork.get_state())
        fork.observe(event('fork-only', gain=1.0))
        self.assertEqual(before, self.agent.get_state())

    def test_returned_state_does_not_alias_persisted_state(self):
        self.agent.observe(event())
        returned = self.agent.get_state()
        returned['mood']['valence'] = 999
        self.assertNotEqual(returned, self.agent.get_state())

    def test_snapshot_is_value_and_restore_is_exact(self):
        self.agent.observe(event())
        snapshot = self.agent.snapshot()
        baseline = self.agent.get_state()
        self.agent.observe(event('later', gain=0.9))
        self.agent.restore(snapshot)
        self.assertEqual(baseline, self.agent.get_state())

    def test_replay_consumes_recorded_operations(self):
        self.agent.observe(event())
        self.agent.advance_time(2)
        replayed = self.agent.replay('replayed')
        self.assertEqual(self.agent.get_state(), replayed.get_state())

    def test_recall_does_not_change_factual_history(self):
        self.agent.observe(event())
        before = self.agent.get_state()
        self.agent.observe(event('recall-1', kind='recall'))
        after = self.agent.get_state()
        self.assertEqual(before['expectations'], after['expectations'])
        self.assertEqual(before['relationships'], after['relationships'])
        self.assertEqual(before['outcomes'], after['outcomes'])

    def test_invalid_time_does_not_partially_update(self):
        self.agent.observe(event())
        before = self.agent.get_state()
        for dt in (-1, float('nan'), float('inf')):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                self.agent.advance_time(dt)
            self.assertEqual(before, self.agent.get_state())

    def test_approval_does_not_become_objective_success(self):
        self.agent.observe(event(gain=0, quality=0, approval=1))
        state = self.agent.get_state()
        self.assertFalse(state['outcomes'])
        self.assertEqual(state['expectations'], {})
        self.assertEqual(state['affect']['e1']['satisfaction'], 0)
        self.assertGreater(state['relationships']['partner']['affiliation'], 0)

    def test_mixed_targets_are_preserved(self):
        self.agent.observe(event('success', gain=0.8, target='project-a'))
        self.agent.observe(event('setback', gain=-0.8, target='project-b'))
        affects = self.agent.get_state()['affect']
        self.assertGreater(affects['success']['satisfaction'], 0)
        self.assertGreater(affects['setback']['threat'], 0)
        self.assertEqual(affects['success']['target'], 'project-a')
        self.assertEqual(affects['setback']['target'], 'project-b')

    def test_habituation_and_recovery(self):
        self.agent.observe(event('first'))
        first = self.agent.get_state()['affect']['first']['threat']
        for i in range(12):
            self.agent.observe(event(f'repeat-{i}'))
        late = self.agent.get_state()['affect']['repeat-11']['threat']
        self.assertLess(late, first)
        self.agent.advance_time(200)
        self.agent.observe(event('recovered'))
        recovered = self.agent.get_state()['affect']['recovered']['threat']
        self.assertGreater(recovered, late)

    def test_stressor_cessation_changes_state_not_just_bounds(self):
        for i in range(40):
            self.agent.observe(event(f'stress-{i}'))
        stressed = self.agent.get_state()
        self.agent.advance_time(500)
        recovered = self.agent.get_state()
        self.assertLess(abs(recovered['mood']['valence']), abs(stressed['mood']['valence']) * 0.01)
        self.assertLess(recovered['mood']['arousal'], stressed['mood']['arousal'] * 0.01)
        self.assertEqual(stressed['outcomes'], recovered['outcomes'])
        self.assertEqual(stressed['expectations'], recovered['expectations'])


if __name__ == '__main__':
    unittest.main()
