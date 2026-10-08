import json
import os
import tempfile
import unittest
from pathlib import Path
from dna.board import public_records, render, snapshot
from dna.campaigns import Store
from dna.schema import record


class Board(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'environments.yaml').write_text('schema_version: 1\nenvironments:\n  - {id: local, label: Local, backend: unknown}\n  - {id: remote, label: Remote, backend: cpu}\n')
        self.inv = record('inventory', environment='local', python='3.12.10', packages={'jax': {'version': '0.11.2'}}, repositories=[], runtime={'backend': 'cpu'})
        Store(self.root/'private').add(self.inv)

    def test_support_expiry_and_policy_review_are_separate(self):
        from datetime import date
        from dna.policy import support_windows
        policy = {'schema_version': 1, 'review_by': '2027-01-01', 'python': [{'version': '3.12', 'upstream_end_month': '2028-10', 'jax_guaranteed_through_month': '2027-07'}]}
        self.assertEqual(support_windows(policy, date(2026, 10, 8))[0]['status'], 'Within upstream window')
        self.assertEqual(support_windows(policy, date(2027, 2, 1))[0]['status'], 'Policy review due')
        self.assertEqual(support_windows(policy, date(2028, 11, 1))[0]['status'], 'Upstream end of life')

    def test_inventory_diff_keeps_unknown_runtime_unknown(self):
        from dna.policy import diff_inventories
        other = record('inventory', environment='remote', python='3.13.1', packages={}, repositories=[], runtime={'backend': 'unknown'})
        rows = diff_inventories(self.inv, other)['rows']
        self.assertEqual(rows[0]['status'], 'different')
        backend = next(r for r in rows if r['component'] == 'runtime:backend')
        self.assertEqual(backend['status'], 'unknown')
        self.assertIsNone(next(r for r in rows if r['component'] == 'package:jax')['right'])

    def test_partial_coverage_never_has_whole_board_refresh(self):
        a = snapshot(self.root)
        b = snapshot(self.root)
        self.assertIsNone(a['refreshed_at'])
        self.assertEqual(a['updated'], self.inv['created'])
        self.assertEqual(a['updated'], b['updated'])
        self.assertEqual(a['environments'][1]['status'], 'unknown')

    def test_public_audit_omits_direct_urls_and_private_projects(self):
        obj = record('audit', inventory=self.inv['digest'], declarations=[], rows=[
            {'repository': 'PyAutoArray', 'package': 'jax', 'declared': 'jax @ https://private.invalid/secret.whl', 'status': 'unknown'},
            {'repository': 'PrivateScience', 'package': 'secret', 'observed': '1', 'status': 'compatible'},
        ])
        clean = json.dumps(public_records([obj]))
        self.assertNotIn('private.invalid', clean)
        self.assertNotIn('PrivateScience', clean)
        self.assertNotIn('secret', clean)
        self.assertIn('direct_source', clean)
        self.assertEqual(public_records(public_records([obj])), public_records([obj]))

    def test_standard_collapsed_sections_and_visible_package_versions(self):
        brain = Path(os.environ.get('PYAUTO_BRAIN', Path(__file__).resolve().parents[2] / 'PyAutoBrain'))
        if not brain.is_dir():
            brain = Path(__file__).resolve().parents[1] / 'PyAutoBrain'
        render(self.root, brain, self.root/'site')
        page = (self.root/'site/index.html').read_text()
        self.assertIn('0.11.2', page)
        self.assertIn('Last updated unavailable', page)
        self.assertEqual(page.count('class="board-section"'), 4)
        self.assertNotIn('<details class="board-section" open', page)
        self.assertIn('data-organ="brain"', page)
        self.assertNotIn(str(self.root), page)
        badge = json.loads((self.root/'site/badge.json').read_text())
        self.assertEqual(badge['message'], '1 / 2 environments observed')


if __name__ == '__main__':
    unittest.main()
