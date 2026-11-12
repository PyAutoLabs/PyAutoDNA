"""Reviewed public projections and the shared organism dashboard."""
import html
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import yaml
from packaging.requirements import Requirement
from .campaigns import Store
from .inventory import public_inventory, public_runtime, PUBLIC_PACKAGES, PUBLIC_REPOSITORIES
from .matrix import CHECK_FIELDS, MATRIX_SOURCE
from .schema import identifier, now, read, timestamp
from .policy_docs import inline, load_decisions, render_markdown, safe_url

URL = 'https://pyautolabs.github.io/PyAutoDNA/'
REPO = 'https://github.com/PyAutoLabs/PyAutoDNA'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def registry(root):
    obj = yaml.safe_load((root / 'environments.yaml').read_text())
    if obj.get('schema_version') != 1 or not isinstance(obj.get('environments'), list):
        raise ValueError('invalid environment registry')
    seen = set()
    for env in obj['environments']:
        identifier(env['id'])
        if env['id'] in seen:
            raise ValueError('duplicate environment')
        seen.add(env['id'])
        if env.get('backend') not in {'unknown', 'cpu', 'cuda'}:
            raise ValueError('invalid environment backend')
        if not isinstance(env.get('label'), str):
            raise ValueError('environment label required')
    return obj['environments']


SIMPLE = __import__('re').compile(r'[A-Za-z0-9_.:/-]{1,100}')


def public_check(check):
    """A validation check reduced to allowlisted fields with simple values."""
    if isinstance(check, str):
        return check if SIMPLE.fullmatch(check) else None
    if not isinstance(check, dict):
        return None
    out = {}
    for key in CHECK_FIELDS:
        value = check.get(key)
        if value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
        elif isinstance(value, str) and SIMPLE.fullmatch(value):
            out[key] = value
    return out if out.get('case') else None


#: XLA options that decide whether the XLA:CPU deadlock can fire (PyAutoNerves#213). Other
#: options (e.g. --xla_disable_hlo_passes=constant_folding) were tested and do not.
HANG_OPTIONS = ('xla_cpu_multi_thread_eigen', 'xla_cpu_enable_concurrency_optimized_scheduler')


def stack_key(packages, xla_options):
    relevant = [o for o in (xla_options or []) if o.lstrip('-').split('=')[0] in HANG_OPTIONS]
    return (packages.get('jax'), tuple(sorted(relevant)))


def jax_cpu_policy(records):
    """Latest DNA research verdict per (stack, environment), with the stack's jax and XLA options."""
    stacks = {r['digest']: r for r in records if r['kind'] == 'stack'}
    research = latest([r for r in records if r['kind'] == 'validation' and r.get('authority') == 'PyAutoDNA'],
                      'validation', lambda r: (r['stack'], r['environment']))
    rows = []
    for r in research.values():
        stack = stacks.get(r['stack'], {})
        options = stack.get('runtime_flags', {}).get('xla_options', [])
        checks = [c for c in r.get('checks', []) if isinstance(c, dict)]
        hangs = [c for c in checks if c.get('outcome') == 'hang']
        rows.append({'jax': stack.get('packages', {}).get('jax', 'Unknown'), 'xla_options': options,
                     'environment': r['environment'], 'result': r['result'], 'created': r['created'],
                     'cells': len(checks), 'passed': sum(c.get('outcome') == 'pass' for c in checks),
                     'hung': len(hangs),
                     'hung_cases': sorted({c['case'] for c in hangs}),
                     'first_hang': min((c['hung_at_call'] for c in hangs if c.get('hung_at_call') is not None), default=None)})
    return sorted(rows, key=lambda x: (x['jax'], x['environment'], ' '.join(x['xla_options'])))


FLAG_LABELS = {'xla_cpu_multi_thread_eigen': 'eigen off', 'xla_cpu_enable_concurrency_optimized_scheduler': 'scheduler off'}


def flag_label(xla_options):
    """The hang-relevant part of a stack's XLA options, as a short column label."""
    names = sorted(FLAG_LABELS[o.lstrip('-').split('=')[0]] for o in (xla_options or [])
                   if o.lstrip('-').split('=')[0] in FLAG_LABELS)
    return ' + '.join(names) if names else 'no flag'


def _version_key(text):
    from packaging.version import InvalidVersion, Version
    try:
        return (0, Version(text))
    except InvalidVersion:
        return (1, text)


def research_checks(records):
    """Every check of the latest DNA research verdict per (stack, environment), flattened with
    its environment, jax version and flag label."""
    stacks = {r['digest']: r for r in records if r['kind'] == 'stack'}
    research = latest([r for r in records if r['kind'] == 'validation' and r.get('authority') == 'PyAutoDNA'],
                      'validation', lambda r: (r['stack'], r['environment']))
    out = []
    for r in research.values():
        stack = stacks.get(r['stack'], {})
        jax = stack.get('packages', {}).get('jax')
        if not jax:
            continue
        label = flag_label(stack.get('runtime_flags', {}).get('xla_options'))
        for c in r.get('checks', []):
            if isinstance(c, dict):
                out.append(dict(c, environment=r['environment'], jax=jax, flag=label))
    return out


def matrix_grid(checks):
    """``{environment: {'columns': [(flag, cores)], 'rows': {jax: {(flag, cores): (passed, hung, total)}}}}``.
    ``cores`` is the pinned core count, or 'all' when the run was not pinned."""
    grid = {}
    for c in checks:
        cores = c.get('cpus_pinned') or 'all'
        env = grid.setdefault(c['environment'], {'columns': set(), 'rows': {}})
        col = (c['flag'], cores)
        env['columns'].add(col)
        cell = env['rows'].setdefault(c['jax'], {}).setdefault(col, [0, 0, 0])
        cell[0] += c.get('outcome') == 'pass'
        cell[1] += c.get('outcome') == 'hang'
        cell[2] += 1
    order = {'no flag': 0, 'eigen off': 1, 'scheduler off': 2}
    for env in grid.values():
        env['columns'] = sorted(env['columns'], key=lambda fc: (order.get(fc[0], 9), fc[0], str(fc[1]).zfill(4)))
    return grid


def flag_cost(checks, environment, jax):
    """Median steady per-call seconds of passing checks: ``(columns, {case: {(flag, cores): s}})``."""
    import statistics
    cells = {}
    for c in checks:
        if c['environment'] != environment or c['jax'] != jax or c.get('outcome') != 'pass':
            continue
        value = c.get('steady_median_call_s') or c.get('median_call_s')
        if isinstance(value, (int, float)):
            cells.setdefault(c['case'], {}).setdefault((c['flag'], c.get('cpus_pinned') or 'all'), []).append(value)
    columns = sorted({col for row in cells.values() for col in row}, key=lambda fc: (fc[0] != 'no flag', fc[0], str(fc[1]).zfill(4)))
    return columns, {case: {col: statistics.median(v) for col, v in row.items()} for case, row in sorted(cells.items())}


def policy_cards(records, docs, checks=None):
    """One card per decision with policy fields: declared (latest audit) vs recommended, derived
    status, and the matrix versions the declared range allows that hang with no flag."""
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    checks = research_checks(records) if checks is None else checks
    hanging = sorted({c['jax'] for c in checks if c['flag'] == 'no flag' and c.get('outcome') == 'hang'}, key=_version_key)
    audits = sorted((r for r in records if r['kind'] == 'audit'), key=lambda r: timestamp(r['created']))
    cards = []
    for doc in docs:
        pol = doc.get('policy')
        if not pol:
            continue
        declared, audited_at = {}, None
        for audit in audits:  # newest audit wins, package by package
            for row in audit.get('rows', []):
                if row.get('repository') == pol['declared_in'] and row.get('package') in pol['packages'] and row.get('declared'):
                    declared[row['package']] = str(Requirement(row['declared']).specifier)
                    audited_at = audit['created']
        rows, same = [], bool(declared)
        for package in pol['packages']:
            rec = SpecifierSet(str(pol['recommended'][package]))
            dec = declared.get(package)
            if dec is None or SpecifierSet(dec) != rec:
                same = False
            rows.append({'package': package, 'declared': dec, 'recommended': str(rec)})
        derived = 'not audited' if not declared else 'adopted' if same else 'recommended, not yet declared'
        allowed_hanging, recommended_hanging = [], []
        mp = pol.get('matrix_package')
        for version in hanging:
            if mp in declared and SpecifierSet(declared[mp]).contains(version, prereleases=True):
                allowed_hanging.append(version)
            if SpecifierSet(str(pol['recommended'][mp])).contains(version, prereleases=True):
                recommended_hanging.append(version)
        cards.append({'policy': pol['policy'], 'title': pol['title'], 'status': pol['status'], 'derived': derived,
                      'declared_in': pol['declared_in'], 'audited_at': audited_at, 'rows': rows,
                      'findings': pol.get('findings', []), 'links': pol.get('links', []),
                      'review_date': pol['review_date'], 'decision': doc['name'],
                      'allowed_hanging': allowed_hanging, 'recommended_hanging': recommended_hanging})
    return cards


def public_records(records):
    result = []
    for obj in records:
        if obj['kind'] == 'inventory':
            continue
        if obj['kind'] == 'availability' and obj.get('name') not in PUBLIC_PACKAGES:
            continue
        allowed = ('digest', 'kind', 'created', 'name', 'python', 'packages', 'backend',
                   'target', 'rollback', 'environments', 'campaign', 'stack', 'environment',
                   'inventory', 'result', 'authority', 'review_date', 'validations', 'latest',
                   'compatible_stable')
        clean = {k: obj[k] for k in allowed if k in obj}
        if obj['kind'] == 'validation' and isinstance(obj.get('checks'), list):
            clean['checks'] = [c for c in (public_check(c) for c in obj['checks']) if c is not None]
        if obj['kind'] == 'stack' and isinstance(obj.get('runtime_flags'), dict):
            flags = obj['runtime_flags']
            # private stacks carry raw flags; already-public ones carry the filtered fields
            runtime = public_runtime({'flags': flags, **{k: flags[k] for k in ('xla_options', 'platforms') if k in flags}})
            clean['runtime_flags'] = {k: runtime[k] for k in ('xla_options', 'platforms') if k in runtime}
        if 'packages' in clean:
            clean['packages'] = {k: v for k, v in clean['packages'].items() if k in PUBLIC_PACKAGES}
        if obj['kind'] == 'audit':
            clean['rows'] = []
            for row in obj.get('rows', []):
                if row.get('repository') not in PUBLIC_REPOSITORIES | {'explicit-ci-override'}:
                    continue
                if row.get('package') not in PUBLIC_PACKAGES | {'declarations'} and row.get('status') != 'ci_override_review':
                    continue
                safe = {k: row[k] for k in ('repository', 'package', 'observed', 'status', 'source', 'line') if k in row}
                if 'declared' in row:
                    if row.get('package') == 'python':
                        safe['declared'] = row['declared']
                    else:
                        req = Requirement(row['declared'])
                        safe['declared'] = req.name + str(req.specifier)
                        if req.url or row.get('direct_source') is True:
                            safe['direct_source'] = True
                clean['rows'].append(safe)
        result.append(clean)
    return result


def snapshot(root, max_age_days=7):
    if max_age_days < 1:
        raise ValueError('max age must be positive')
    envs = registry(root)
    records = Store(root / 'private').records()
    records += [read(p) for folder in ('stacks', 'decisions', 'campaigns')
                for p in sorted((root / folder).glob('*.json'))]
    projection_path = root / 'inventories/public.json'
    projection = json.loads(projection_path.read_text()) if projection_path.exists() else {}
    if projection and projection.get('schema_version') != 1:
        raise ValueError('unsupported public projection schema')
    current = {}
    for obj in records:
        if obj['kind'] == 'inventory' and obj['runtime'].get('source') != MATRIX_SOURCE:
            # a hang-matrix inventory describes a throwaway test interpreter, not the environment
            inv = public_inventory(obj)
            if inv['environment'] not in current or timestamp(inv['created']) > timestamp(current[inv['environment']]['created']):
                current[inv['environment']] = inv
    for old in projection.get('environments', []):
        inv = old.get('observed')
        if not inv:
            continue
        if inv['environment'] != old['id']:
            raise ValueError('public inventory identity mismatch')
        timestamp(inv['created'])
        # Re-apply the allowlist even to committed public data.
        clean = public_inventory(inv)
        if old['id'] not in current or timestamp(clean['created']) > timestamp(current[old['id']]['created']):
            current[old['id']] = clean
    clock = datetime.now(timezone.utc)
    rows = []
    for env in envs:
        inv = current.get(env['id'])
        state = 'unknown' if inv is None else 'stale' if (clock - timestamp(inv['created'])).total_seconds() > max_age_days * 86400 else 'observed'
        rows.append({'id': env['id'], 'label': env['label'], 'role': env.get('role', 'unknown'),
                     'declared_python': env.get('python', 'unknown'), 'declared_backend': env['backend'],
                     'status': state, 'observed': inv})
    published = public_records(projection.get('records', [])) + public_records(records)
    published = list({r['digest']: r for r in published}.values())
    observed_times = [r['observed']['created'] for r in rows if r['observed']]
    refreshed = min(observed_times, key=timestamp) if rows and len(observed_times) == len(rows) else None
    # Retain observation time across re-renders. No receipt means unknown coverage,
    # even though the shared feed requires a timestamp for the generated artifact.
    updated = min(observed_times, key=timestamp) if observed_times else datetime.fromtimestamp(
        (root / 'environments.yaml').stat().st_mtime, timezone.utc).isoformat()
    return {'schema_version': 1, 'updated': updated, 'refreshed_at': refreshed,
            'environments': rows, 'records': published}


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows, label):
    return ('<div class="dna-table" role="region" tabindex="0" aria-label="' + esc(label) + '">'
            '<table><thead><tr>' + ''.join('<th scope="col">' + esc(h) + '</th>' for h in headers) +
            '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in row) + '</tr>' for row in rows) +
            '</tbody></table></div>')


def latest(records, kind, key):
    result = {}
    for obj in records:
        if obj['kind'] == kind:
            name = key(obj)
            if name not in result or timestamp(obj['created']) > timestamp(result[name]['created']):
                result[name] = obj
    return result


def evidence_age(obj, max_age_days):
    age = (datetime.now(timezone.utc) - timestamp(obj['created'])).total_seconds()
    return 'Stale' if age > max_age_days * 86400 else 'Current receipt'


def render_policies(cards):
    """The Policies section: one card per decision that sets a policy."""
    body = '<section><h2 id="policies">Policies</h2><p>What each repository declares, what DNA recommends, and the evidence behind it.</p>'
    if not cards:
        return body + '<p>No decision sets a policy yet.</p></section>'
    for c in cards:
        body += '<div class="dna-card"><h3>' + esc(c['title']) + '</h3>'
        body += '<p><span class="dna-badge">' + esc(c['derived']) + '</span>' + esc(c['status']) + '</p>'
        body += table(['Package', 'Declared in ' + c['declared_in'], 'Recommended'], [
            [r['package'], r['declared'] or 'Not audited', r['recommended']] for r in c['rows']], c['title'] + ' ranges')
        when = ('declared range from the dependency audit of ' + c['audited_at'][:10]) if c['audited_at'] else 'no dependency audit recorded yet'
        body += '<p class="muted">' + esc(when) + '. Review by ' + esc(c['review_date']) + '.</p>'
        if c['allowed_hanging']:
            body += ('<p class="dna-warn">The declared range still allows jax ' + esc(', '.join(c['allowed_hanging']))
                     + ', which hangs in the hang matrix without a flag. The recommended range excludes it.</p>')
        elif c['rows'] and c['derived'] != 'not audited':
            body += '<p>No version the declared range allows hangs in the hang matrix.</p>'
        if c['recommended_hanging']:
            body += '<p class="dna-warn">The recommended range allows jax ' + esc(', '.join(c['recommended_hanging'])) + ', which hangs in the hang matrix.</p>'
        if c['findings']:
            body += '<h4>Findings</h4><ul>' + ''.join('<li>' + inline(str(f)) + '</li>' for f in c['findings']) + '</ul>'
        links = [l for l in c['links'] if safe_url(l['url'])]
        body += ('<p>' + ' · '.join('<a href="' + esc(l['url']) + '">' + esc(l['label']) + '</a>' for l in links)
                 + (' · ' if links else '') + '<a href="#decision-' + esc(c['decision']) + '">Full decision</a></p>')
        body += '</div>'
    return body + '</section>'


def render_matrix(checks):
    """Pass/hang counts per jax version x XLA flag x cores, and per-call cost on the newest version."""
    if not checks:
        return '<p>No hang-matrix checks recorded.</p>'
    body = ''
    for env, grid in sorted(matrix_grid(checks).items()):
        cols = grid['columns']
        head = ['jax'] + [f"{flag}, {cores} cores" if cores != 'all' else f"{flag}, all cores" for flag, cores in cols]
        rows = []
        for jax in sorted(grid['rows'], key=_version_key):
            row = [jax]
            for col in cols:
                cell = grid['rows'][jax].get(col)
                if not cell:
                    row.append('—')
                else:
                    passed, hung, total = cell
                    other = total - passed - hung
                    text = f"{passed}/{total} pass" if not hung else f"{hung}/{total} hang" + (f" · {passed} pass" if passed else '')
                    row.append(text + (f" ({other} not run)" if other else ''))
            rows.append(row)
        body += '<h4>' + esc(env) + ': cells passed or hung</h4>' + table(head, rows, env + ' hang matrix by version, flag and cores')
        newest = max(grid['rows'], key=_version_key)
        cost_cols, cost = flag_cost(checks, env, newest)
        if cost:
            body += '<h4>' + esc(env) + ': per-call time on jax ' + esc(newest) + ' (median, seconds)</h4>'
            body += table(['Reproducer'] + [f"{flag}, {cores} cores" if cores != 'all' else f"{flag}, all cores" for flag, cores in cost_cols],
                          [[case] + [f"{cost[case][col]:.3f}" if col in cost[case] else '—' for col in cost_cols] for case in cost],
                          env + ' per-call time by flag')
    body += '<p class="muted">A cell counts repeated runs of every reproducer; "not run" covers a module that version cannot load. Per-call time is the steady median of passing runs.</p>'
    return body


def render(root, brain, output, max_age_days=7):
    theme = module(brain / 'board/_theme.py', 'dna_theme')
    state_contract = module(brain / 'board/_state.py', 'dna_state')
    board = snapshot(root, max_age_days)
    envs, records = board['environments'], board['records']
    items = [{'id': e['id'], 'severity': 'yellow' if e['status'] == 'stale' else 'info',
              'state': 'stale' if e['status'] == 'stale' else 'unknown' if e['status'] == 'unknown' else 'active',
              'text': f"{e['label']}: {e['status']}", 'url': URL + '#environments'} for e in envs]
    current_inventories = {e['observed']['digest']: e for e in envs if e['observed']}
    mismatches = []
    for audit in latest(records, 'audit', lambda r: r['inventory']).values():
        env = current_inventories.get(audit['inventory'])
        if env and evidence_age(audit, max_age_days) != 'Stale':
            for row in audit.get('rows', []):
                if row.get('status') == 'incompatible':
                    mismatches.append((env, row))
    for index, (env, row) in enumerate(mismatches):
        items.append({'id': 'dependency-' + str(index), 'severity': 'yellow', 'state': 'action_required',
                      'text': f"{env['label']}: {row.get('package')} {row.get('observed')} is outside {row['repository']} declarations",
                      'url': URL + '#compatibility'})
    policy = jax_cpu_policy(records)
    failing = {stack_key({'jax': r['jax']}, r['xla_options']) for r in policy if r['result'] == 'fail'}
    exposed = []
    for e in envs:
        inv = e['observed']
        if inv and inv['runtime'].get('backend') == 'cpu' and 'jax' in inv['packages']:
            key = stack_key({'jax': inv['packages']['jax']['version']}, inv['runtime'].get('xla_options'))
            if key in failing:
                exposed.append(e)
                items.append({'id': 'jax-cpu-' + e['id'], 'severity': 'yellow', 'state': 'action_required',
                              'text': f"{e['label']}: jax {key[0]} with {', '.join(key[1]) or 'no XLA:CPU hang flag'} hangs in the DNA hang matrix",
                              'url': URL + '#jax-cpu-policy'})
    mismatches = mismatches + [(e, None) for e in exposed]
    docs = load_decisions(root / 'decisions') if (root / 'decisions').is_dir() else []
    checks = research_checks(records)
    cards = policy_cards(records, docs, checks)
    open_policies = [c for c in cards if c['allowed_hanging']]
    for card in open_policies:
        items.append({'id': 'policy-' + card['policy'], 'severity': 'yellow', 'state': 'action_required',
                      'text': f"{card['title']}: {card['declared_in']} still allows jax {', '.join(card['allowed_hanging'])}, which hangs in the DNA hang matrix",
                      'url': URL + '#policies'})
    status = 'yellow' if mismatches or open_policies else 'stale' if any(e['status'] == 'stale' for e in envs) else 'grey'
    count = sum(e['status'] == 'observed' for e in envs)
    headline = f'{count} / {len(envs)} environments observed'
    if len(mismatches) > len(exposed):
        headline += f'; {len(mismatches) - len(exposed)} declaration mismatches'
    if exposed:
        headline += f'; {len(exposed)} on a hanging JAX CPU setup'
    if open_policies:
        headline += f"; {len(open_policies)} {'policy' if len(open_policies) == 1 else 'policies'} not yet declared"
    state = dict(schema_version=1, organ='dna', repo='PyAutoLabs/PyAutoDNA', status=status,
                 headline=headline, updated=board['updated'], pages_url=URL, items=items)
    errors = state_contract.validate_state(state)
    if errors:
        raise ValueError('; '.join(errors))
    sections = ['policies', 'environments', 'compatibility', 'decisions', 'campaigns']
    body = theme.hero('dna', 'Dashboard', navigation=[{'href': '#' + s, 'label': s.title()} for s in sections])
    body += theme.orchestration_panel('dna', 'DNA', '',
        'Review PyAutoDNA software stacks and compatibility. Read AGENTS.md and inspect current inventories, '
        'dependency declarations, support decisions and upgrade campaigns. Explain local, RAL and CI drift. '
        'Propose scoped upgrades through Brain and verify linked Heart evidence before promotion. Preserve '
        'running science environments. Collection or refresh never installs packages or submits compute.',
        organ='dna', refreshed_at=board['refreshed_at'], refresh_url=REPO + '/actions/workflows/pages.yml',
        work_links=[{'label': 'DNA', 'href': REPO}, {'label': 'Brain', 'href': 'https://github.com/PyAutoLabs/PyAutoBrain'}])
    body += render_policies(cards)
    body += '<section><h2 id="environments">Environments</h2>'
    body += table(['Environment', 'Profile', 'Observed Python / backend', 'XLA options', 'Last observation'], [
        [e['label'], e['declared_python'] + ' / ' + e['declared_backend'],
         e['observed']['python'] + ' / ' + e['observed']['runtime']['backend'] if e['observed'] else 'Unknown',
         (', '.join(e['observed']['runtime'].get('xla_options') or []) or 'Default') if e['observed'] else 'Unknown',
         e['status'] + ' · ' + e['observed']['created'] if e['observed'] else 'Not collected'] for e in envs], 'Environment observations; scroll for more columns')
    observed = [e for e in envs if e['observed']]
    if observed:
        body += '<h3>Installed packages</h3><p class="muted">Compare the latest recorded inventory for each environment. Swipe or scroll the table to see all columns.</p>'
        packages = sorted(set().union(*(set(e['observed']['packages']) for e in observed)))
        body += table(['Package'] + [e['label'] for e in envs], [
            [name] + [(e['observed']['packages'].get(name, {}).get('version', 'Not installed') if e['observed'] else 'Unknown') for e in envs]
            for name in packages], 'Installed package versions across environments; scroll horizontally')
        for e in observed:
            inv = e['observed']
            body += '<details><summary>' + esc(e['label']) + ' · source identity</summary>'
            body += '<p>Inventory <code>' + esc(inv['digest']) + '</code></p>'
            body += table(['Repository', 'Commit', 'Working tree'], [
                [r['name'], r['sha'], 'Unknown' if r['dirty'] is None else 'Modified' if r['dirty'] else 'Clean'] for r in inv['repositories']], e['label'] + ' source commits') if inv['repositories'] else '<p>No source checkout recorded.</p>'
            body += '<p class="muted">Source commits identify editable checkouts. Package version metadata may retain an older release stamp.</p></details>'
    body += '</section><section><h2 id="compatibility">Compatibility</h2>'
    availability = latest(records, 'availability', lambda r: (r['name'], r['python']))
    support_path = root / 'support.yaml'
    if support_path.exists():
        from .policy import support_windows
        support = yaml.safe_load(support_path.read_text())
        windows = support_windows(support)
        body += '<h3>Python support horizons</h3>'
        body += table(['Python', 'CPython end of support', 'JAX guarantees support through', 'Policy status'], [[r['version'], r['upstream_end_month'], r['jax_guaranteed_through_month'], r['status']] for r in windows], 'Upstream Python support windows')
        body += '<p class="muted">These upstream windows do not certify PyAuto compatibility. Review due ' + esc(support['review_by']) + '. Sources: <a href="https://devguide.python.org/versions/">CPython</a> and <a href="https://docs.jax.dev/en/latest/deprecation.html">JAX</a>.</p>'
    body += '<h3>Upstream releases</h3>'
    if availability:
        body += table(['Package', 'Latest upstream', 'Stable for Python', 'Python', 'Checked', 'Freshness'], [
            [r['name'], r.get('latest') or 'Unknown', r.get('compatible_stable') or 'None found', r['python'], r['created'], evidence_age(r, max_age_days)]
            for r in availability.values()], 'Upstream availability')
        body += '<p class="muted">Python-compatible metadata does not prove platform, GPU or full-stack compatibility. Newer releases are candidates for validation.</p>'
    else:
        body += '<p>No upstream release check recorded.</p>'
    audits = latest(records, 'audit', lambda r: r['inventory'])
    body += '<h3>Declared compatibility</h3>'
    if audits:
        for obj in audits.values():
            env = next((e['label'] for e in envs if e['observed'] and e['observed']['digest'] == obj['inventory']), 'Historical inventory')
            rows = obj.get('rows', [])
            body += '<details><summary>' + esc(env) + ' · dependency audit · ' + esc(obj['created']) + ' · ' + esc(evidence_age(obj, max_age_days)) + '</summary>'
            body += table(['Repository', 'Package / workflow', 'Requirement', 'Installed', 'Result'], [
                [r['repository'], r.get('package', r.get('source', 'Unknown')), 'Direct source (private)' if r.get('direct_source') else r.get('declared', 'Review installation steps'), r.get('observed') or 'Unknown', r['status'].replace('_', ' ')] for r in rows], env + ' dependency audit')
            body += '</details>'
    else:
        body += '<p>No dependency audit recorded. Package requirements remain authoritative in each repository.</p>'
    body += '<h3 id="jax-cpu-policy">JAX CPU policy: hang matrix</h3>'
    if policy:
        body += render_matrix(checks)
        body += '<details><summary>Every recorded verdict</summary>'
        body += table(['jax', 'XLA options', 'Environment', 'Result', 'Reproducers passed', 'Hung (first call)', 'Recorded'], [
            [r['jax'], ', '.join(r['xla_options']) or 'Default', r['environment'], r['result'],
             f"{r['passed']} / {r['cells']}",
             (', '.join(r['hung_cases']) + (f" (call {r['first_hang']})" if r['first_hang'] is not None else '')) if r['hung'] else 'None',
             r['created']] for r in policy], 'JAX version and XLA flag hang matrix')
        body += '</details>'
        body += '<p class="muted">DNA research evidence: each reproducer runs many times in a process that imports only jax. A pass is not a Heart validation and never promotes a stack; a hang blocks one.</p>'
    else:
        body += '<p>No hang-matrix evidence recorded.</p>'
    body += '<h3>Validation and promotion history</h3>'
    validations = latest(records, 'validation', lambda r: (r['stack'], r['environment']))
    if validations:
        body += table(['Stack', 'Environment', 'Authority', 'Latest supplied result', 'Evidence date', 'Freshness'], [[r['stack'][:12], r['environment'], r.get('authority', 'PyAutoHeart'), r['result'], r['created'], evidence_age(r, max_age_days)] for r in validations.values()], 'Supplied validation receipts')
    else:
        body += '<p>No matching Heart validation receipt recorded. No supported stack is inferred from an observed installation.</p>'
    body += '<p class="muted">Supplied validation receipts are records, not a live Heart readiness verdict. Stale evidence must be refreshed before adoption.</p>'
    stacks = {r['digest']: r for r in records if r['kind'] == 'stack'}
    promotions = latest(records, 'promotion', lambda r: r['campaign'])
    if promotions:
        body += table(['Promoted stack', 'Promotion recorded', 'Campaign'], [[stacks.get(r['stack'], {}).get('name', r['stack'][:12]), r['created'], r['campaign'][:12]] for r in promotions.values()], 'Recorded promotions')
        body += '<p class="muted">Promotion history is retained; adoption rechecks current evidence and expiry.</p>'
    else:
        body += '<p>No recommended stack has been promoted.</p>'
    body += '</section><section><h2 id="decisions">Decisions</h2><p>Reasons for version restrictions and when to review them.</p>'
    for doc in docs:
        title = doc['policy']['title'] if doc['policy'] else doc['name'].replace('-', ' ')
        body += ('<details id="decision-' + esc(doc['name']) + '"><summary>' + esc(title) + '</summary><div class="dna-prose">'
                 + render_markdown(doc['body']) + '</div></details>')
    body += '</section><section><h2 id="campaigns">Campaigns</h2>'
    campaigns = [r for r in records if r['kind'] == 'campaign']
    if not campaigns:
        body += '<p>No upgrade campaign recorded. Start a check-in to compare available releases and choose a candidate stack.</p>'
    for campaign in campaigns:
        body += '<details><summary>' + esc(campaign['name']) + '</summary>'
        body += table(['Target stack', 'Rollback stack', 'Environments', 'Review date'], [[stacks.get(campaign['target'], {}).get('name', campaign['target'][:12]), stacks.get(campaign['rollback'], {}).get('name', campaign['rollback'][:12]), ', '.join(campaign['environments']), campaign['review_date']]], 'Upgrade campaign')
        events = [r for r in records if r.get('campaign') == campaign['digest'] and r['kind'] in {'promotion', 'adoption', 'rollback'}]
        body += table(['Action', 'Environment', 'Recorded'], [[r['kind'], r.get('environment', 'Campaign'), r['created']] for r in sorted(events, key=lambda r: timestamp(r['created']))], 'Campaign history') if events else '<p>Candidate awaiting validation.</p>'
        body += '</details>'
    body += '</section>' + theme.boards_footer(theme.board_links('https://pyautolabs.github.io', 'dna'), 'dna')
    css = '.dna-table{overflow-x:auto;margin:1rem 0}.dna-table table{width:100%;border-collapse:collapse}.dna-table th,.dna-table td{padding:.65rem .8rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--border,#d0d7de)}.dna-table th{white-space:nowrap}.dna-table td{min-width:7rem}.dna-prose{line-height:1.6;max-width:72ch}.dna-prose blockquote{margin:1rem 0;padding:.25rem 1rem;border-left:3px solid var(--border,#d0d7de)}.dna-prose pre{overflow-x:auto}.dna-card{border:1px solid var(--border,#d0d7de);border-radius:10px;padding:1rem 1.1rem;margin:1rem 0}.dna-card h3{margin-top:0}.dna-badge{display:inline-block;padding:.1rem .55rem;border-radius:999px;border:1px solid var(--border,#d0d7de);font-size:.85em;margin-right:.4rem}.dna-warn{border-left:4px solid #bf8700;padding:.4rem .8rem;margin:.6rem 0}code{overflow-wrap:anywhere}'
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PyAutoDNA — Software stacks and compatibility</title><style>' + theme.css('dna') + css + '</style></head><body><main>' + body + '</main><script>' + theme.JS + '</script></body></html>'
    page = theme.section_layout(page)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'index.html').write_text(page)
    for name, obj in (('board', board), ('state', state), ('badge', {'schemaVersion': 1, 'label': 'DNA', 'message': headline, 'color': 'lightgrey' if status == 'grey' else 'yellow'})):
        (output / (name + '.json')).write_text(json.dumps(obj, indent=2) + '\n')
    return {'output': str(output), 'status': status, 'headline': headline}
