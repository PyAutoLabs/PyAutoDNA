"""Explicit PyPI discovery; availability is not support certification."""
import json
from urllib.request import urlopen
from packaging.version import Version, InvalidVersion
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from .schema import record

def available(package, python, fetch=None):
    package = canonicalize_name(package)
    if fetch is None:
        with urlopen(f'https://pypi.org/pypi/{package}/json', timeout=20) as response:
            data = json.load(response)
    else:
        data = fetch(package)
    versions, compatible = [], []
    for text, files in data.get('releases', {}).items():
        try: version = Version(text)
        except InvalidVersion: continue
        if not files: continue
        versions.append(version)
        if version.is_prerelease or version.is_devrelease: continue
        if any(not f.get('yanked', False) and python in SpecifierSet(f.get('requires_python') or '') for f in files):
            compatible.append(version)
    return record('availability', name=package, python=python,
                  source=f'https://pypi.org/pypi/{package}/json',
                  latest=str(max(versions)) if versions else None,
                  compatible_stable=str(max(compatible)) if compatible else None,
                  interpretation='Python-compatible non-yanked release files; platform/backend/resolver compatibility remains untested')
