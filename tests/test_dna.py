import json
import tempfile
import unittest
from pathlib import Path
from dna.schema import record, write, validate, digest
from datetime import datetime, timezone, timedelta
from dna.inventory import public_inventory
from dna.policy import audit, compare
from dna.campaigns import Store
from dna.board import snapshot
from dna.upstream import available

class DNA(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.store = Store(self.root / 'private')
        (self.root / 'environments.yaml').write_text('schema_version: 1\nenvironments:\n  - {id: local, label: Local, backend: cpu}\n')
        self.inv = self.store.add(record('inventory', environment='local', python='3.12.9', packages={'jax': {'version': '0.11.1', 'location': '/private'}, 'secret-project': {'version': '1'}}, repositories=[], runtime={'backend':'cpu', 'flags': {'XLA_FLAGS':'secret'}}))
        self.stack = self.store.add(record('stack', name='baseline', python='3.12.9', packages={'jax':'0.11.1'}, backend='cpu'))
    def tearDown(self): self.temp.cleanup()
    def campaign(self):
        return self.store.create('upgrade', self.stack['digest'], self.stack['digest'], ['local'], 'human', '2099-01-01')
    def validation(self, result='pass'):
        return self.store.add(record('validation', stack=self.stack['digest'], environment='local', authority='PyAutoHeart', result=result, checks=['smoke'], evidence='private-artifact', inventory=self.inv['digest']))
    def test_tamper_immutable_invalid(self):
        obj=dict(self.inv, python='9.0')
        with self.assertRaises(ValueError): validate(obj)
        write(self.root/'record.json', self.inv)
        with self.assertRaises(FileExistsError): write(self.root/'record.json', self.inv)
        with self.assertRaises(ValueError): record('inventory', environment='../escape', python='3', packages={}, repositories=[], runtime={})
    def test_forged_transition_import_rejected(self):
        campaign=self.campaign()
        forged=record('promotion',campaign=campaign['digest'],stack=self.stack['digest'],validations=[])
        with self.assertRaises(ValueError):self.store.add(forged)
        with self.assertRaises(ValueError):self.store.adopt(campaign['digest'],'local',self.inv['digest'])
        forged_adopt=record('adoption',campaign=campaign['digest'],environment='local',stack=self.stack['digest'],inventory=self.inv['digest'])
        with self.assertRaises(ValueError):self.store.add(forged_adopt)
    def test_unregistered_campaign(self):
        with self.assertRaises(ValueError):
            self.store.create('bad',self.stack['digest'],self.stack['digest'],['missing'],'human','2099-01-01')
    def test_compare_and_constraints(self):
        self.assertTrue(all(r['status']=='match' for r in compare(self.inv,self.stack)))
        stack=record('stack',name='flags',python='3.12.9',backend='cpu',packages={'jax':'0.11.1'},runtime_flags={'JAX_ENABLE_X64':'1'})
        self.assertEqual(compare(self.inv,stack)[-1]['status'],'unknown')
    def test_audit_real_declarations(self):
        repo=self.root/'repo'; repo.mkdir()
        (repo/'pyproject.toml').write_text('[project]\nrequires-python=">=3.13"\ndependencies=["jax>=0.12", "numpy; python_version < \'3.0\'"]\n')
        rows=audit(self.inv,[repo])
        self.assertEqual([r['status'] for r in rows], ['incompatible','incompatible'])
    def test_override_marker_and_direct_url(self):
        self.assertEqual(audit(self.inv,[],overrides=["jax==999; python_version < '3'"]),[])
        self.assertEqual(audit(self.inv,[],overrides=['jax @ https://example.com/jax.whl'])[0]['status'],'unknown')
    def test_unknown_target_marker(self):
        repo=self.root/'marker';repo.mkdir()
        (repo/'pyproject.toml').write_text('[project]\ndependencies=["jax; implementation_name == \'cpython\'"]\n')
        self.assertEqual(audit(self.inv,[repo])[0]['status'],'unknown')
    def test_gates_and_newer_failure(self):
        campaign=self.campaign()
        with self.assertRaises(ValueError): self.store.adopt(campaign['digest'],'local',self.inv['digest'])
        evidence=self.validation()
        self.store.promote(campaign['digest'],[evidence['digest']])
        self.assertEqual(self.store.adopt(campaign['digest'],'local',self.inv['digest'])['kind'],'adoption')
        self.validation('fail')
        with self.assertRaises(ValueError): self.store.promote(campaign['digest'],[evidence['digest']])
        with self.assertRaises(ValueError): self.store.adopt(campaign['digest'],'local',self.inv['digest'])
        with self.assertRaises(ValueError): self.store.adopt(campaign['digest'],'local',self.inv['digest'],True)
    def test_expired_evidence_and_unknown_backend(self):
        campaign=self.campaign(); evidence=self.validation()
        evidence['created']=(datetime.now(timezone.utc)-timedelta(days=8)).isoformat()
        evidence['digest']=digest(evidence);self.store.add(evidence)
        # Remove newer pass so the stale record is the latest available evidence.
        for path in (self.root/'private/records').glob('*.json'):
            obj=json.loads(path.read_text())
            if obj['kind']=='validation' and obj['digest']!=evidence['digest']:path.unlink()
        with self.assertRaises(ValueError):self.store.promote(campaign['digest'],[evidence['digest']])
        unknown=self.store.add(record('stack',name='unknown',python='3.12.9',backend='unknown',packages={'jax':'0.11.1'}))
        c=self.store.create('unknown',unknown['digest'],self.stack['digest'],['local'],'human','2099-01-01')
        with self.assertRaises(ValueError):self.store.promote(c['digest'],[])
    def test_validation_provenance(self):
        campaign=self.campaign()
        wrong=self.store.add(record('inventory',environment='local',python='3.13.0',packages={'jax':{'version':'0.11.1'}},repositories=[],runtime={'backend':'cpu'}))
        ev=self.store.add(record('validation',stack=self.stack['digest'],environment='local',authority='PyAutoHeart',result='pass',checks=['x'],evidence='x',inventory=wrong['digest']))
        with self.assertRaises(ValueError): self.store.promote(campaign['digest'],[ev['digest']])
    def test_projection_and_unknown(self):
        public=public_inventory(self.inv)
        text=json.dumps(public)
        self.assertNotIn('/private',text);self.assertNotIn('XLA_FLAGS',text);self.assertNotIn('secret-project',text)
        snap=snapshot(self.root);self.assertEqual(snap['environments'][0]['status'],'observed')
        for p in (self.root/'private/records').glob('*.json'): p.unlink()
        self.assertEqual(snapshot(self.root)['environments'][0]['status'],'unknown')
    def test_upstream_yanked_python_prerelease(self):
        data={'releases':{'1.0':[{'requires_python':'>=3.12'}],'2.0':[{'yanked':True}],'3.0':[{'requires_python':'>=3.13'}],'4.0rc1':[{}]}}
        obj=available('jax','3.12.9',lambda _:data)
        self.assertEqual(obj['latest'],'4.0rc1');self.assertEqual(obj['compatible_stable'],'1.0')
    def test_projection_roundtrip_and_private_injection(self):
        public=snapshot(self.root);public['environments'][0]['observed']['location']='/private/injected'
        (self.root/'inventories').mkdir();(self.root/'inventories/public.json').write_text(json.dumps(public))
        for p in (self.root/'private/records').glob('*.json'): p.unlink()
        self.assertNotIn('/private',json.dumps(snapshot(self.root)))

if __name__=='__main__': unittest.main()


class HangMatrix(unittest.TestCase):
    """PyAutoNerves#213: hang-matrix rows become DNA research evidence that can block, never promote."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.store = Store(self.root / 'private')
        (self.root / 'environments.yaml').write_text('schema_version: 1\nenvironments:\n  - {id: local, label: Local, backend: cpu}\n')
    def tearDown(self): self.temp.cleanup()
    def row(self, flags, outcome, case='export:group_mge_p3', hung=None):
        return {'environment': 'local', 'case': case, 'outcome': outcome, 'calls_requested': 100,
                'calls_completed': 100 if outcome == 'pass' else (hung or 0), 'hung_at_call': hung,
                'median_call_s': 0.07, 'stall_cpu_fraction': 0.0 if outcome == 'hang' else None, 'repeat': 0,
                'header': {'jax': '0.11.2', 'jaxlib': '0.11.2', 'python': '3.13.13', 'backend': 'cpu', 'xla_flags': flags}}
    def write_rows(self, rows):
        path = self.root / 'rows.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return path
    def test_groups_and_results(self):
        from dna.matrix import import_matrix
        out = import_matrix(self.store, self.write_rows([
            self.row('', 'hang', hung=2), self.row('', 'pass', case='fft'),
            self.row('--xla_cpu_multi_thread_eigen=false', 'pass'),
            self.row('--xla_cpu_multi_thread_eigen=false', 'pass', case='fft')]))
        results = {o['digest']: o['result'] for o in out if o['kind'] == 'validation'}
        self.assertEqual(sorted(results.values()), ['fail', 'pass'])
        names = sorted(o['name'] for o in out if o['kind'] == 'stack')
        self.assertEqual(names, ['jax-0.11.2-default', 'jax-0.11.2-xla_cpu_multi_thread_eigen'])
        validation = next(r for r in self.store.records('validation') if r['result'] == 'fail')
        self.assertEqual(validation['authority'], 'PyAutoDNA')
        inv = self.store.require(validation['inventory'], 'inventory')
        stack = self.store.require(validation['stack'], 'stack')
        self.assertTrue(all(r['status'] == 'match' for r in compare(inv, stack)))
    def test_research_never_promotes_and_newer_failure_blocks(self):
        from dna.matrix import import_matrix
        import_matrix(self.store, self.write_rows([self.row('', 'pass')]))
        research = self.store.records('validation')[0]
        stack = self.store.require(research['stack'], 'stack')
        campaign = self.store.create('jax-cpu-policy', stack['digest'], stack['digest'], ['local'], 'human', '2099-01-01')
        with self.assertRaises(ValueError): self.store.promote(campaign['digest'], [research['digest']])
        heart = self.store.add(record('validation', stack=stack['digest'], environment='local', authority='PyAutoHeart', result='pass', checks=['smoke'], evidence='heart-run', inventory=research['inventory']))
        self.store.promote(campaign['digest'], [heart['digest']])
        import time; time.sleep(1.1)
        self.store.add(record('validation', stack=stack['digest'], environment='local', authority='PyAutoDNA', result='fail', checks=['hang'], evidence='matrix', inventory=research['inventory']))
        with self.assertRaises(ValueError): self.store.promote(campaign['digest'], [heart['digest']])
    def test_unknown_authority_rejected(self):
        with self.assertRaises(ValueError):
            record('validation', stack='x', environment='local', authority='Someone', result='pass', checks=['c'], evidence='e', inventory='i')
    def test_public_projection_keeps_only_allowlisted_xla_options(self):
        from dna.inventory import public_xla_options
        self.assertEqual(public_xla_options('--xla_cpu_multi_thread_eigen=false --xla_dump_to=/home/me/secret --xla_disable_hlo_passes=constant_folding'),
                         ['--xla_cpu_multi_thread_eigen=false', '--xla_disable_hlo_passes=constant_folding'])
        inv = record('inventory', environment='local', python='3.13.13', packages={'jax': {'version': '0.11.2'}}, repositories=[],
                     runtime={'backend': 'cpu', 'flags': {'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false --xla_dump_to=/home/me/secret'}})
        public = public_inventory(inv)
        self.assertEqual(public['runtime']['xla_options'], ['--xla_cpu_multi_thread_eigen=false'])
        self.assertEqual(public_inventory(dict(inv, runtime=public['runtime']))['runtime']['xla_options'], ['--xla_cpu_multi_thread_eigen=false'])
        self.assertNotIn('secret', json.dumps(public))
