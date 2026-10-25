"""Replay closure uses stored model text/public evidence, never fresh providers."""
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from regesh.providers import ModelResult
from regesh.software_eval import DEFAULT_CONFIG,run_software_experiment
from regesh.software_replay import replay_software_experiment


class ModelFixture:
    identifier='software-replay-fixture';temperature=.6
    def complete(self,messages,*,seed,max_tokens,recorder=None):
        text='def solve(data):\n return []'
        record={'request':{'model':None,'messages':messages,'seed':seed,'max_tokens':max_tokens,'temperature':.6},
                'response':{'choices':[{'message':{'content':text}}],'usage':{'prompt_tokens':50,'completion_tokens':10}}}
        if recorder:recorder('model_call',record)
        return ModelResult(text,record)


class ReplayChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.run_dir=cls.root/'run'
        config={**DEFAULT_CONFIG,'tasks':['batching'],'seeds':[910],'bootstrap_samples':10}
        run_software_experiment(ModelFixture(),cls.run_dir,config=config)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_actor_replay_uses_zero_models_and_zero_checkers(self):
        with tempfile.TemporaryDirectory() as out:
            with patch('regesh.providers.LocalChatModel.complete',side_effect=AssertionError('provider forbidden')),patch('subprocess.run',side_effect=AssertionError('checker forbidden')):
                summary=replay_software_experiment(self.run_dir,out)
            self.assertEqual(summary['model_calls'],0)
            self.assertEqual(summary['actor_trials_verified'],2)
            self.assertEqual(summary['actor_phases_verified'],4)
            self.assertEqual(summary['hidden_checker_calls_during_actor_replay'],0)
    def test_optional_independent_recheck_matches(self):
        with tempfile.TemporaryDirectory() as out:
            summary=replay_software_experiment(self.run_dir,out,recheck=True)
            self.assertEqual(summary['independent_recheck_calls'],4)
            self.assertTrue(summary['all_independent_rechecks_match'])
    def test_corrupt_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed=Path(tmp)/'run';shutil.copytree(self.run_dir,changed)
            path=changed/'calls.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
            rows[0]['messages'][0]['content']='tampered'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'exact prompt'):
                replay_software_experiment(changed,Path(tmp)/'out')
    def test_corrupt_adaptation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed=Path(tmp)/'run';shutil.copytree(self.run_dir,changed)
            path=changed/'traces.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
            rows[0]['state_after']['quality']=.123
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'adaptive state'):
                replay_software_experiment(changed,Path(tmp)/'out')
    def test_gzip_archived_records_config_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed=Path(tmp)/'run';shutil.copytree(self.run_dir,changed)
            for name in ('calls.jsonl','traces.jsonl','results.jsonl','config.json','source_manifest.json','source_software_eval.py','source_software_tasks.py'):
                path=changed/name
                with gzip.open(str(path)+'.gz','wb') as stream:stream.write(path.read_bytes())
                path.unlink()
            summary=replay_software_experiment(changed,Path(tmp)/'out')
            self.assertTrue(summary['all_actor_trajectories_verified'])
    def test_frozen_source_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed=Path(tmp)/'run';shutil.copytree(self.run_dir,changed)
            source=changed/'source_software_eval.py';source.write_text(source.read_text()+'\n# altered\n')
            with self.assertRaisesRegex(ValueError,'source integrity'):
                replay_software_experiment(changed,Path(tmp)/'out')
