"""Strict identities and immutable, content-addressed evidence records."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

KINDS = {"inventory", "stack", "validation", "decision", "campaign", "adoption", "promotion", "rollback", "availability", "audit"}

def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", value):
        raise ValueError("identity must be a simple name, never a path")
    return value

def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be UTC ISO-8601")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp must be UTC")
    if parsed > datetime.now(timezone.utc):
        raise ValueError("future evidence timestamp")
    return parsed

def digest(data):
    clean = {k: v for k, v in data.items() if k != "digest"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def record(kind, **data):
    obj = dict(schema_version=1, kind=kind, created=now(), **data)
    obj["digest"] = digest(obj)
    validate(obj)
    return obj

def validate(obj):
    if not isinstance(obj, dict) or obj.get("schema_version") != 1 or obj.get("kind") not in KINDS:
        raise ValueError("unsupported record schema/kind")
    timestamp(obj.get("created"))
    if obj.get("digest") != digest(obj):
        raise ValueError("record digest mismatch")
    kind = obj["kind"]
    required = {"audit": ["inventory", "rows", "declarations"],
                "availability": ["name", "python", "source", "interpretation"],
                "inventory": ["environment", "python", "packages", "repositories", "runtime"],
                "stack": ["name", "python", "packages", "backend"],
                "validation": ["stack", "environment", "authority", "result", "evidence", "checks", "inventory"],
                "decision": ["name", "rationale", "evidence", "review_date"],
                "campaign": ["name", "target", "rollback", "environments", "owner", "review_date"],
                "promotion": ["campaign", "stack", "validations"],
                "adoption": ["campaign", "environment", "stack", "inventory"],
                "rollback": ["campaign", "environment", "stack", "inventory", "reason"]}[kind]
    for key in required:
        if key not in obj or obj[key] is None:
            raise ValueError(f"{kind}: missing {key}")
    for key in ("environment", "name", "owner"):
        if key in obj:
            identifier(obj[key])
    if kind in {"inventory", "stack"}:
        if not isinstance(obj["packages"], dict) or not isinstance(obj["python"], str):
            raise ValueError("packages must be a mapping; python must be a string")
    if kind == "inventory":
        for name, package in obj["packages"].items():
            identifier(name)
            if not isinstance(package, dict) or not isinstance(package.get("version"), str):
                raise ValueError("package requires a version")
        if not isinstance(obj["repositories"], list) or not isinstance(obj["runtime"], dict):
            raise ValueError("invalid inventory repositories/runtime")
    if kind == "stack":
        if not obj["packages"]:
            raise ValueError("stack must pin at least one package")
        for repo in obj.get("repositories", []):
            identifier(repo["name"])
            if not re.fullmatch(r"[0-9a-f]{40,64}", repo.get("sha", "")) or not isinstance(repo.get("dirty", False), bool):
                raise ValueError("invalid stack git identity")
        if not isinstance(obj.get("runtime_flags", {}), dict):
            raise ValueError("invalid runtime flags")
        from packaging.version import Version
        Version(obj["python"])
        for name, version in obj["packages"].items():
            identifier(name)
            Version(version)
        if obj["backend"] not in {"cpu", "cuda", "unknown"}:
            raise ValueError("invalid backend")
    if kind == "validation":
        if obj["authority"] != "PyAutoHeart" or obj["result"] not in {"pass", "fail", "unknown"}:
            raise ValueError("validation must name Heart and a valid result")
        if not isinstance(obj["checks"], list) or not obj["checks"] or not obj["evidence"]:
            raise ValueError("validation requires checks and evidence")
    if "review_date" in obj:
        datetime.strptime(obj["review_date"], "%Y-%m-%d")
    if kind == "campaign":
        if not isinstance(obj["environments"], list) or not obj["environments"]:
            raise ValueError("campaign requires environments")
        for env in obj["environments"]:
            identifier(env)
        if len(set(obj["environments"])) != len(obj["environments"]):
            raise ValueError("duplicate campaign environment")
    return obj

def read(path):
    return validate(json.loads(Path(path).read_text()))

def write(path, obj):
    validate(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
