"""Append-only, explicit evidence gates. Recording adoption never installs packages."""
from datetime import datetime, timezone
from pathlib import Path
from .policy import compare
from .schema import identifier, read, record, write, timestamp

class Store:
    def __init__(self, root):
        self.root = Path(root)

    def records(self, kind=None):
        return [read(p) for p in sorted((self.root / "records").glob("*.json")) if kind is None or read(p)["kind"] == kind]

    def get(self, digest):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("expected a record digest")
        return read(self.root / "records" / (digest + ".json"))

    def add(self, obj):
        if obj.get("kind") in {"campaign", "promotion", "adoption", "rollback"}:
            raise ValueError("transition records must be created through explicit gated commands")
        return self._persist(obj)

    def _persist(self, obj):
        path = self.root / "records" / (obj["digest"] + ".json")
        if path.exists():
            if read(path) != obj:
                raise ValueError("immutable record conflict")
            return obj
        write(path, obj)
        return obj

    def require(self, digest, kind):
        obj = self.get(digest)
        if obj["kind"] != kind:
            raise ValueError(f"expected {kind}")
        return obj

    def create(self, name, target, rollback, environments, owner, review_date):
        self.require(target, "stack"); self.require(rollback, "stack")
        registry_path = self.root.parent / "environments.yaml"
        if not registry_path.exists():
            raise ValueError("campaign requires an environment registry")
        import yaml
        registered = {e["id"] for e in yaml.safe_load(registry_path.read_text())["environments"]}
        if not set(environments).issubset(registered):
            raise ValueError("campaign environment is not registered")
        return self._persist(record("campaign", name=name, target=target, rollback=rollback,
                               environments=environments, owner=owner, review_date=review_date))

    def _gate(self, obj, validations):
        if datetime.strptime(obj["review_date"], "%Y-%m-%d").date() < datetime.now(timezone.utc).date():
            raise ValueError("campaign review date expired")
        target = self.require(obj["target"], "stack")
        if target["backend"] == "unknown":
            raise ValueError("promotion requires an explicit backend")
        checks = [self.require(d, "validation") for d in validations]
        all_checks = self.records("validation")
        for env in obj["environments"]:
            matching = [c for c in all_checks if c["stack"] == obj["target"] and c["environment"] == env]
            latest = max(matching, key=lambda c: timestamp(c["created"])) if matching else None
            if latest is None or latest["digest"] not in validations or latest["result"] != "pass":
                raise ValueError(f"latest matching Heart evidence must pass: {env}")
            if (datetime.now(timezone.utc) - timestamp(latest["created"])).total_seconds() > 7 * 86400:
                raise ValueError("stale Heart validation")
            inv = self.require(latest["inventory"], "inventory")
            if inv["environment"] != env or any(r["status"] != "match" for r in compare(inv, target)):
                raise ValueError("validation inventory does not match stack")

    def promote(self, campaign, validations):
        obj = self.require(campaign, "campaign")
        self._gate(obj, validations)
        return self._persist(record("promotion", campaign=campaign, stack=obj["target"], validations=validations))

    def adopt(self, campaign, environment, inventory, rollback=False, reason=None):
        obj = self.require(campaign, "campaign")
        if environment not in obj["environments"]:
            raise ValueError("environment outside campaign")
        if not rollback:
            promotions = [p for p in self.records("promotion") if p["campaign"] == campaign]
            if not promotions:
                raise ValueError("campaign is not promoted")
            latest_promotion = max(promotions, key=lambda p: timestamp(p["created"]))
            self._gate(obj, latest_promotion["validations"])
        stack = obj["rollback"] if rollback else obj["target"]
        inv = self.require(inventory, "inventory")
        if (datetime.now(timezone.utc) - timestamp(inv["created"])).total_seconds() > 7 * 86400:
            raise ValueError("stale adoption inventory")
        if inv["environment"] != environment or any(r["status"] != "match" for r in compare(inv, self.require(stack, "stack"))):
            raise ValueError("inventory does not match environment and stack, including runtime")
        if rollback and not reason:
            raise ValueError("rollback requires a reason")
        kwargs = dict(campaign=campaign, environment=environment, stack=stack, inventory=inventory)
        if rollback:
            kwargs["reason"] = reason
        return self._persist(record("rollback" if rollback else "adoption", **kwargs))
