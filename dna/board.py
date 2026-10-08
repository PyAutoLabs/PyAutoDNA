"""Reviewed public projections and the shared organism dashboard."""
import html
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import yaml
from packaging.requirements import Requirement
from .campaigns import Store
from .inventory import public_inventory, PUBLIC_PACKAGES, PUBLIC_REPOSITORIES
from .schema import identifier, now, read, timestamp

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
        if obj['kind'] == 'inventory':
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
    status = 'yellow' if mismatches else 'stale' if any(e['status'] == 'stale' for e in envs) else 'grey'
    count = sum(e['status'] == 'observed' for e in envs)
    headline = f'{count} / {len(envs)} environments observed'
    if mismatches:
        headline += f'; {len(mismatches)} declaration mismatches'
    state = dict(schema_version=1, organ='dna', repo='PyAutoLabs/PyAutoDNA', status=status,
                 headline=headline, updated=board['updated'], pages_url=URL, items=items)
    errors = state_contract.validate_state(state)
    if errors:
        raise ValueError('; '.join(errors))
    sections = ['environments', 'compatibility', 'decisions', 'campaigns']
    body = theme.hero('dna', 'Dashboard', navigation=[{'href': '#' + s, 'label': s.title()} for s in sections])
    body += theme.orchestration_panel('dna', 'DNA', '',
        'Review PyAutoDNA software stacks and compatibility. Read AGENTS.md and inspect current inventories, '
        'dependency declarations, support decisions and upgrade campaigns. Explain local, RAL and CI drift. '
        'Propose scoped upgrades through Brain and verify linked Heart evidence before promotion. Preserve '
        'running science environments. Collection or refresh never installs packages or submits compute.',
        organ='dna', refreshed_at=board['refreshed_at'], refresh_url=REPO + '/actions/workflows/pages.yml',
        work_links=[{'label': 'DNA', 'href': REPO}, {'label': 'Brain', 'href': 'https://github.com/PyAutoLabs/PyAutoBrain'}])
    body += '<p class="muted">' + esc(headline) + '. Observed versions describe an installation; support requires matching validation evidence.</p>'
    if mismatches:
        body += '<p class="verdict">' + esc('; '.join(f"{env['label']}: {row.get('package')} {row.get('observed')} is outside declared compatibility" for env, row in mismatches)) + '.</p>'
    body += '<section><h2 id="environments">Environments</h2>'
    body += table(['Environment', 'Profile', 'Observed Python / backend', 'Last observation'], [
        [e['label'], e['declared_python'] + ' / ' + e['declared_backend'],
         e['observed']['python'] + ' / ' + e['observed']['runtime']['backend'] if e['observed'] else 'Unknown',
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
    body += '<h3>Validation and promotion history</h3>'
    validations = latest(records, 'validation', lambda r: (r['stack'], r['environment']))
    if validations:
        body += table(['Stack', 'Environment', 'Latest supplied Heart result', 'Evidence date', 'Freshness'], [[r['stack'][:12], r['environment'], r['result'], r['created'], evidence_age(r, max_age_days)] for r in validations.values()], 'Supplied validation receipts')
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
    for path in sorted((root / 'decisions').glob('*.md')):
        body += '<details><summary>' + esc(path.stem.replace('-', ' ')) + '</summary><pre class="dna-prose">' + esc(path.read_text()) + '</pre></details>'
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
    css = '.dna-table{overflow-x:auto;margin:1rem 0}.dna-table table{width:100%;border-collapse:collapse}.dna-table th,.dna-table td{padding:.65rem .8rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--border,#d0d7de)}.dna-table th{white-space:nowrap}.dna-table td{min-width:7rem}.dna-prose{white-space:pre-wrap;font:inherit;line-height:1.6}code{overflow-wrap:anywhere}'
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PyAutoDNA — Software stacks and compatibility</title><style>' + theme.css('dna') + css + '</style></head><body><main>' + body + '</main><script>' + theme.JS + '</script></body></html>'
    page = theme.section_layout(page)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'index.html').write_text(page)
    for name, obj in (('board', board), ('state', state), ('badge', {'schemaVersion': 1, 'label': 'DNA', 'message': headline, 'color': 'lightgrey' if status == 'grey' else 'yellow'})):
        (output / (name + '.json')).write_text(json.dumps(obj, indent=2) + '\n')
    return {'output': str(output), 'status': status, 'headline': headline}
