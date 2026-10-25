import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from regesh import Agent, Event
from regesh.learning import RidgePredictor


class CoreChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'core.sqlite'
        self.agent=Agent(self.path,'a')
    def tearDown(self):
        self.agent.close();self.tmp.cleanup()
    def test_actual_process_restart(self):
        self.agent.observe(Event('e','test','p','g','t','2026-10-25T00:00:00Z'))
        expected=self.agent.get_state()
        code='import json,sys;from regesh import Agent;a=Agent(sys.argv[1],"a");print(json.dumps(a.get_state()));a.close()'
        observed=json.loads(subprocess.check_output([sys.executable,'-c',code,str(self.path)]))
        self.assertEqual(expected,observed)
    def test_recompute_detects_wrong_transition_poststate(self):
        self.agent.advance_time(1.)
        from unittest.mock import patch
        with patch('regesh.agent.advance',lambda s,dt:s.update(time=7.)):
            with self.assertRaisesRegex(ValueError,'recomputed state mismatch'):
                self.agent.replay('incorrect')
    def test_learned_adapter_is_persistent_and_replayable(self):
        weights=np.zeros((72,4));weights[54,0]=1. # select verify in this integration fixture
        self.agent.set_regulator(RidgePredictor(weights,metadata={'fixture':True}))
        self.agent.set_task_context({'quality':.4,'uncertainty':.8})
        self.assertEqual(self.agent.actuation_context()['strategy'],'verify')
        self.agent.act('auto')
        replayed=self.agent.replay('copy')
        try:
            self.assertEqual(self.agent.get_state(),replayed.get_state())
            self.assertEqual(self.agent.actuation_context(),replayed.actuation_context())
            self.assertLess(self.agent.get_state()['internal']['resources'],1.)
        finally:replayed.close()
    def test_allocating_verification_does_not_fabricate_evidence(self):
        before=self.agent.get_state();self.agent.act('verify');after=self.agent.get_state()
        self.assertEqual(before['internal']['uncertainty'],after['internal']['uncertainty'])
        self.assertEqual(before['events'],after['events'])
    def test_independent_operational_conditions_change_control(self):
        original=self.agent.snapshot();base=self.agent.actuation_context()
        for key,value,control in [('reliability',0.,'verification'),('burden',1.,'planning_horizon'),('commitments',40.,'planning_horizon')]:
            self.agent.restore(original);self.agent.intervene({'internal':{key:value}},'isolated internal condition')
            self.assertNotEqual(base[control],self.agent.actuation_context()[control])
    def test_timezone_and_mapping_schema(self):
        with self.assertRaises(ValueError):Event('x','s','a','t','k','2026-01-01')
        with self.assertRaises(ValueError):Event('x','s','a','t','k','2026-01-01T00:00:00Z',data={'goal_delta':[]})

    def test_accurate_and_misleading_criticism_change_future_evidence_weight(self):
        base=self.agent.snapshot()
        branches=[]
        for name,correct in [('accurate',True),('misleading',False)]:
            branch=self.agent.fork(name,base)
            try:
                for step in range(5):
                    e=Event(str(step),'review','reviewer','candidate','criticism','2026-10-25T00:00:00Z',
                            data={'approval':.8,'text':'Claim: the resource bound is exceeded.'})
                    branch.observe(e)
                    branch.record_artifact('tool_result',{'claim_supported':correct,'task_quality':.4})
                    branch.record_outcome(e.id,.4,partner_correct=correct,provenance='independent check fixture')
                probe=Event('probe','review','reviewer','candidate','criticism','2026-10-25T00:00:00Z')
                branches.append((branch.get_state(),branch.appraise(probe)))
            finally:branch.close()
        reliable,misleading=branches
        self.assertEqual(reliable[0]['expectations'],misleading[0]['expectations'])
        self.assertEqual(reliable[0]['relationships']['reviewer']['affiliation'],misleading[0]['relationships']['reviewer']['affiliation'])
        self.assertLess(reliable[1]['uncertainty'],misleading[1]['uncertainty'])
        self.assertEqual(reliable[0]['relationships']['reviewer']['authority'],0.)

    def test_snapshot_and_replay_stream_without_materializing_trace(self):
        from unittest.mock import patch
        for i in range(4):
            self.agent.observe(Event(str(i),'test','p','g','t','2026-10-25T00:00:00Z'))
        expected=self.agent.get_state()
        with patch.object(Agent,'trace',side_effect=AssertionError('full history must not be materialized')):
            self.assertEqual(self.agent.snapshot()['state'],expected)
            copy=self.agent.replay('streamed')
            try:self.assertEqual(copy.get_state(),expected)
            finally:copy.close()
            playback=self.agent.replay('played',mode='playback')
            try:self.assertEqual(playback.get_state(),expected)
            finally:playback.close()
