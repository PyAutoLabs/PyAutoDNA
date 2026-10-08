"""Compare resolved identities and read authoritative project declarations on demand."""
import ast
import re
import tomllib
from pathlib import Path
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from .schema import validate

def compare(inventory, stack):
    validate(inventory); validate(stack)
    rows = []
    for name, expected in stack["packages"].items():
        observed = inventory["packages"].get(canonicalize_name(name), {}).get("version")
        rows.append({"package": name, "expected": expected, "observed": observed,
                     "status": "unknown" if observed is None else "match" if observed == expected else "drift"})
    rows.append({"package": "python", "expected": stack["python"], "observed": inventory["python"],
                 "status": "match" if stack["python"] == inventory["python"] else "drift"})
    backend = inventory["runtime"].get("backend", "unknown")
    rows.append({"package": "backend", "expected": stack["backend"], "observed": backend,
                 "status": "unknown" if "unknown" in (backend, stack["backend"]) else "match" if backend == stack["backend"] else "drift"})
    for wanted in stack.get("repositories", []):
        actual = next((r for r in inventory["repositories"] if r["name"] == wanted["name"]), None)
        state = "unknown" if actual is None or actual.get("dirty") is None else "match" if actual["sha"] == wanted["sha"] and actual["dirty"] == wanted.get("dirty", False) else "drift"
        rows.append({"package": "git:" + wanted["name"], "status": state})
    for flag, expected in stack.get("runtime_flags", {}).items():
        actual = inventory["runtime"].get("flags", {}).get(flag)
        rows.append({"package": "runtime:" + flag, "expected": expected, "observed": actual,
                     "status": "unknown" if actual is None else "match" if actual == expected else "drift"})
    return rows

def declarations(repo):
    repo = Path(repo)
    path = repo / "pyproject.toml"
    if not path.exists():
        return {"source": "pyproject.toml", "status": "unknown", "requirements": [], "python": None}
    data = tomllib.loads(path.read_text())
    project = data.get("project", {})
    return {"source": "pyproject.toml", "status": "declared", "requirements": project.get("dependencies", []),
            "optional": project.get("optional-dependencies", {}), "dynamic": project.get("dynamic", []),
            "python": project.get("requires-python"), "sha": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}

def audit(inventory, repos, extras=(), overrides=()):
    validate(inventory)
    rows = []
    env = {"python_version": ".".join(inventory["python"].split(".")[:2]), "python_full_version": inventory["python"],
           "sys_platform": {"Linux": "linux", "Windows": "win32", "Darwin": "darwin"}.get(inventory.get("platform", {}).get("system"), "unknown"),
           "platform_system": inventory.get("platform", {}).get("system", "unknown"),
           "platform_machine": inventory.get("platform", {}).get("machine", "unknown"), "extra": ""}
    env.update(inventory.get("platform", {}).get("markers", {}))
    for repo in repos:
        declared = declarations(repo)
        name = Path(repo).name
        if declared["status"] == "unknown" or "dependencies" in declared.get("dynamic", []):
            rows.append({"repository": name, "package": "declarations", "status": "unknown", "reason": "missing or dynamic declarations require build metadata"})
        if declared.get("python"):
            rows.append({"repository": name, "package": "python", "declared": declared["python"],
                         "observed": inventory["python"], "status": "compatible" if inventory["python"] in SpecifierSet(declared["python"]) else "incompatible"})
        requirements = list(declared["requirements"])
        for extra in extras:
            requirements.extend(declared.get("optional", {}).get(extra, []))
        for text in requirements:
            req = Requirement(text)
            if req.marker:
                marker_variables = set(re.findall(r"\b(?:os_name|sys_platform|platform_machine|platform_python_implementation|platform_release|platform_system|platform_version|python_version|python_full_version|implementation_name|implementation_version|extra)\b", str(req.marker)))
                if any(env.get(variable, "unknown") == "unknown" for variable in marker_variables):
                    rows.append({"repository": name, "package": req.name, "declared": str(req), "status": "unknown", "reason": "target marker facts not collected"})
                    continue
                if not any(req.marker.evaluate(dict(env, extra=x)) for x in ("", *extras)):
                    continue
            observed = inventory["packages"].get(canonicalize_name(req.name), {}).get("version")
            state = "unknown" if observed is None or req.url else "compatible" if observed in req.specifier else "incompatible"
            rows.append({"repository": name, "package": req.name, "declared": str(req), "observed": observed, "status": state})
        for workflow in sorted((Path(repo) / ".github/workflows").glob("*.y*ml")):
            for line_no, line in enumerate(workflow.read_text().splitlines(), 1):
                if re.search(r'(pip|uv).*install|python-version|jax[^a-z].*[<>=]', line):
                    rows.append({"repository": name, "source": workflow.name, "line": line_no,
                                 "status": "ci_override_review", "reason": "CI installation or interpreter override; inspect authoritative workflow"})
    for override in overrides:
        req = Requirement(override)
        observed = inventory["packages"].get(canonicalize_name(req.name), {}).get("version")
        marker_unknown = False
        if req.marker:
            variables = set(re.findall(r"\b(?:os_name|sys_platform|platform_machine|platform_python_implementation|platform_release|platform_system|platform_version|python_version|python_full_version|implementation_name|implementation_version|extra)\b", str(req.marker)))
            marker_unknown = any(env.get(v, "unknown") == "unknown" for v in variables)
            if not marker_unknown and not any(req.marker.evaluate(dict(env, extra=x)) for x in ("", *extras)):
                continue
        rows.append({"repository": "explicit-ci-override", "package": req.name, "declared": str(req), "observed": observed,
                     "status": "unknown" if observed is None or req.url or marker_unknown else "compatible" if observed in req.specifier else "incompatible",
                     "reason": "override does not change package support declarations"})
    return rows

def command_plan(stack):
    validate(stack)
    return {"python": stack["python"], "backend": stack["backend"], "requires_approval": True,
            "commands": [["python", "-m", "pip", "install", "--dry-run", "--report", "resolver-report.json",
                          *[f"{name}=={version}" for name, version in sorted(stack["packages"].items())]]],
            "runtime_constraints": stack.get("runtime_flags", {}), "repositories": stack.get("repositories", []),
            "backend_note": "CUDA requires a separately reviewed CUDA/JAX plugin and driver resolution; package pins alone do not select a GPU backend.",
            "next": "Use the exact declared Python interpreter. Review resolution, create isolated environment, collect inventory, obtain Heart evidence; DNA executes none of these commands."}


def diff_inventories(left, right):
    """Compare two observed environments without inventing a recommended stack."""
    validate(left)
    validate(right)
    if left['kind'] != 'inventory' or right['kind'] != 'inventory':
        raise ValueError('diff requires two inventory receipts')
    rows = []
    def add(component, a, b, unknown=False):
        rows.append({'component': component, 'left': a, 'right': b,
                     'status': 'unknown' if unknown else 'same' if a == b else 'different'})
    add('python', left['python'], right['python'])
    for name in sorted(set(left['packages']) | set(right['packages'])):
        add('package:' + name, left['packages'].get(name, {}).get('version'), right['packages'].get(name, {}).get('version'))
    for key in ('backend', 'x64'):
        a, b = left['runtime'].get(key), right['runtime'].get(key)
        add('runtime:' + key, a, b, a in (None, 'unknown') or b in (None, 'unknown'))
    add('platform', left.get('platform'), right.get('platform'), not left.get('platform') or not right.get('platform'))
    flags = set(left['runtime'].get('flags', {})) | set(right['runtime'].get('flags', {}))
    for flag in sorted(flags):
        add('flag:' + flag, left['runtime'].get('flags', {}).get(flag), right['runtime'].get('flags', {}).get(flag))
    names = {r['name'] for r in left['repositories'] + right['repositories']}
    for name in sorted(names):
        a = next((r for r in left['repositories'] if r['name'] == name), None)
        b = next((r for r in right['repositories'] if r['name'] == name), None)
        add('source:' + name, {k: a.get(k) for k in ('sha', 'dirty')} if a else None,
            {k: b.get(k) for k in ('sha', 'dirty')} if b else None,
            not a or not b or a.get('dirty') is not False or b.get('dirty') is not False)
    return {'left': left['environment'], 'right': right['environment'], 'rows': rows}


def support_windows(policy, today=None):
    """Curated upstream lifecycle dates, separate from PyAuto validation."""
    from datetime import date
    today = today or date.today()
    if policy.get('schema_version') != 1:
        raise ValueError('unsupported support policy schema')
    review = date.fromisoformat(policy['review_by'])
    result = []
    for row in policy['python']:
        end = row['upstream_end_month']
        date.fromisoformat(end + '-01')
        date.fromisoformat(row['jax_guaranteed_through_month'] + '-01')
        month = today.strftime('%Y-%m')
        state = 'Upstream end of life' if month > end else 'Ends this month' if month == end else 'Policy review due' if today > review else 'Within upstream window'
        result.append(dict(row, status=state))
    return result
