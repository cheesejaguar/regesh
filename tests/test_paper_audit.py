"""The independent audit must reject corrupted or duplicated evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_paper_reproductions import happiness, verify_manifest


class PaperAuditTests(unittest.TestCase):
    def test_changed_evidence_fails_before_numeric_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            evidence = path / 'summary.json'
            evidence.write_text('{"result": 1}')
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            (path / 'manifest.json').write_text(json.dumps({'files': {'summary.json': digest}}))
            self.assertEqual(verify_manifest(path), 1)
            evidence.write_text('{"result": 2}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_manifest(path)

    def test_incomplete_manifest_is_not_integrity_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'no artifact hashes'):
                verify_manifest(path)

    def test_duplicate_simulation_round_is_not_extra_replication(self):
        import gzip
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'summary.json').write_text('{"conditions": []}')
            row = {'setting': 'stationary', 'condition': 'objective',
                   'seed_block': 1, 'round': 0, 'objective_reward': 10}
            with gzip.open(path / 'rounds.jsonl.gz', 'wt') as output:
                output.write((json.dumps(row) + '\n') * 2)
            with self.assertRaisesRegex(ValueError, 'duplicate simulated round'):
                happiness(path)


if __name__ == '__main__':
    unittest.main()
