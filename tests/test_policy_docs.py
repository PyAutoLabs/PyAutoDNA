import json
import tempfile
import unittest
from pathlib import Path

from dna.board import flag_cost, matrix_grid, policy_cards, public_records
from dna.policy_docs import load_decisions, render_markdown, split_front_matter, validate_policy
from dna.schema import record

FRONT = '''---
policy: jax-cpu
title: JAX CPU policy
status: Recommended, awaiting release.
declared_in: PyAutoNerves
packages: [jax, jaxlib]
recommended:
  jax: ">=0.12.0,<0.13"
  jaxlib: ">=0.12.0,<0.13"
review_date: 2026-12-10
findings:
  - "0.11.2 hangs at 4 cores."
links:
  - {label: "Research", url: "https://github.com/PyAutoLabs/PyAutoNerves/issues/213"}
---
# Heading

Body text.
'''


def audit(declared_jax, created_offset=0):
    rows = [{'repository': 'PyAutoNerves', 'package': p, 'observed': '0.11.2', 'status': 'compatible',
             'declared': f'{p}{declared_jax}'} for p in ('jax', 'jaxlib')]
    return record('audit', inventory='i' * 64, rows=rows, declarations=[])


def research(jax, flags, outcomes):
    stack = record('stack', name=f'jax-{jax}', python='3.13.13', backend='cpu', packages={'jax': jax, 'jaxlib': jax},
                   runtime_flags={'XLA_FLAGS': flags})
    checks = [{'case': case, 'outcome': out, 'cpus_pinned': cores, 'steady_median_call_s': t}
              for case, out, cores, t in outcomes]
    val = record('validation', stack=stack['digest'], environment='ral-cpu', authority='PyAutoDNA',
                 result='fail' if any(o[1] == 'hang' for o in outcomes) else 'pass', checks=checks,
                 evidence='private', inventory='i' * 64)
    return public_records([stack, val])


class PolicyDocs(unittest.TestCase):
    def test_front_matter_and_validation(self):
        fields, body = split_front_matter(FRONT)
        pol = validate_policy(fields)
        self.assertEqual(pol['review_date'], '2026-12-10')
        self.assertEqual(pol['matrix_package'], 'jax')
        self.assertTrue(body.startswith('# Heading'))
        self.assertEqual(split_front_matter('# plain')[0], None)
        with self.assertRaises(ValueError):
            validate_policy(dict(fields, recommended={'jax': '>=0.12'}))  # jaxlib missing
        with self.assertRaises(ValueError):
            validate_policy(dict(fields, links=[{'label': 'x', 'url': 'javascript:alert(1)'}]))
        with self.assertRaises(ValueError):
            validate_policy(dict(fields, surprise=1))

    def test_renderer_escapes_and_links(self):
        out = render_markdown('# T\n\nA <script>x</script> and `<b>` and **bold** '
                              '[ok](https://a.b/c) [bad](javascript:alert(1)) [sib](jax-2026-10-04.md)\n\n'
                              '- one\n  continued\n- two\n\n| a | b |\n|---|---|\n| <i> | 2 |\n\n> quote')
        self.assertNotIn('<script>', out)
        self.assertIn('&lt;script&gt;', out)
        self.assertIn('<code>&lt;b&gt;</code>', out)
        self.assertIn('<strong>bold</strong>', out)
        self.assertIn('href="https://a.b/c"', out)
        self.assertNotIn('javascript:', out)
        self.assertIn('href="https://github.com/PyAutoLabs/PyAutoDNA/blob/main/decisions/jax-2026-10-04.md"', out)
        self.assertIn('<li>one continued</li>', out)
        self.assertIn('<td>&lt;i&gt;</td>', out)
        self.assertIn('<blockquote>', out)
        self.assertIn('<h3>T</h3>', out)

    def test_repo_decisions_parse(self):
        docs = load_decisions(Path(__file__).resolve().parents[1] / 'decisions')
        self.assertTrue(any(d['policy'] and d['policy']['policy'] == 'jax-cpu' for d in docs))


class PolicyCards(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        (Path(temp.name) / 'p.md').write_text(FRONT)
        self.docs = load_decisions(temp.name)
        self.matrix = (research('0.11.2', '', [('fft', 'hang', 4, None), ('fft', 'pass', 16, 0.1)])
                       + research('0.12.0', '', [('fft', 'pass', 4, 0.2), ('fft', 'pass', 16, 0.1)])
                       + research('0.12.0', '--xla_cpu_multi_thread_eigen=false', [('fft', 'pass', 4, 0.3)]))

    def test_not_yet_declared_flags_allowed_hanging_version(self):
        records = public_records([audit('!=0.10.*,!=0.11.0,<0.12.0,>=0.7.0')]) + self.matrix
        card, = policy_cards(records, self.docs)
        self.assertEqual(card['derived'], 'recommended, not yet declared')
        self.assertEqual(card['allowed_hanging'], ['0.11.2'])
        self.assertEqual(card['recommended_hanging'], [])

    def test_adopted_when_declared_matches(self):
        records = public_records([audit('<0.13,>=0.12.0')]) + self.matrix
        card, = policy_cards(records, self.docs)
        self.assertEqual(card['derived'], 'adopted')
        self.assertEqual(card['allowed_hanging'], [])

    def test_not_audited(self):
        card, = policy_cards(self.matrix, self.docs)
        self.assertEqual(card['derived'], 'not audited')

    def test_grid_and_cost(self):
        from dna.board import research_checks
        checks = research_checks(self.matrix)
        grid = matrix_grid(checks)['ral-cpu']
        self.assertEqual(grid['rows']['0.11.2'][('no flag', 4)], [0, 1, 1])
        self.assertEqual(grid['rows']['0.12.0'][('eigen off', 4)], [1, 0, 1])
        cols, cost = flag_cost(checks, 'ral-cpu', '0.12.0')
        self.assertEqual(cost['fft'][('no flag', 16)], 0.1)
        self.assertIn(('eigen off', 4), cols)
        self.assertNotIn('private', json.dumps(public_records(self.matrix)))
