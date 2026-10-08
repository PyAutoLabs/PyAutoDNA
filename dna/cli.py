"""Explicit commands; no collectors run during render, promotion or adoption."""
import argparse
import json
from pathlib import Path
from . import schema
from .campaigns import Store
from .inventory import collect
from .policy import audit, command_plan, compare, diff_inventories

def main(argv=None):
    parser = argparse.ArgumentParser(prog="pyauto-dna")
    parser.add_argument("--root", default=".", help="DNA data repository")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("upstream"); p.add_argument("package"); p.add_argument("--python", required=True)
    p = sub.add_parser("export-public"); p.add_argument("--output", default="inventories/public.json")
    p = sub.add_parser("collect"); p.add_argument("environment"); p.add_argument("--repo", action="append", default=[]); p.add_argument("--probe-backend", action="store_true")
    p = sub.add_parser("import"); p.add_argument("file")
    p = sub.add_parser("stack"); p.add_argument("name"); p.add_argument("--python", required=True); p.add_argument("--backend", choices=["cpu", "cuda", "unknown"], required=True); p.add_argument("--package", action="append", default=[])
    p = sub.add_parser("diff"); p.add_argument("left"); p.add_argument("right")
    p = sub.add_parser("compare"); p.add_argument("inventory"); p.add_argument("stack")
    p = sub.add_parser("audit"); p.add_argument("inventory"); p.add_argument("--repo", action="append", required=True); p.add_argument("--extra", action="append", default=[]); p.add_argument("--override", action="append", default=[])
    p = sub.add_parser("plan"); p.add_argument("stack")
    p = sub.add_parser("campaign"); p.add_argument("name"); p.add_argument("--target", required=True); p.add_argument("--rollback", required=True); p.add_argument("--environment", action="append", required=True); p.add_argument("--owner", required=True); p.add_argument("--review-date", required=True)
    p = sub.add_parser("promote"); p.add_argument("campaign"); p.add_argument("--validation", action="append", required=True)
    for command in ("adopt", "rollback"):
        p = sub.add_parser(command); p.add_argument("campaign"); p.add_argument("environment"); p.add_argument("inventory")
        if command == "rollback": p.add_argument("--reason", required=True)
    p = sub.add_parser("show"); p.add_argument("digest", nargs="?")
    p = sub.add_parser("board"); p.add_argument("--brain", required=True); p.add_argument("--output", default="_site"); p.add_argument("--max-age-days", type=int, default=7)
    args = parser.parse_args(argv)
    store = Store(Path(args.root) / "private")
    try:
        cmd = args.command
        if cmd == "upstream":
            from .upstream import available
            result = store.add(available(args.package, args.python))
        elif cmd == "export-public":
            from .board import snapshot
            result = snapshot(Path(args.root))
            path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
        elif cmd == "collect":
            from .board import registry
            if args.environment not in {e["id"] for e in registry(Path(args.root))}:
                raise ValueError("environment not in registry")
            result = store.add(collect(args.environment, args.repo, args.probe_backend))
        elif cmd == "import": result = store.add(schema.read(args.file))
        elif cmd == "stack":
            packages = {}
            for text in args.package:
                name, sep, version = text.partition("==")
                if not sep: raise ValueError("stack packages require exact name==version")
                from packaging.utils import canonicalize_name
                name = canonicalize_name(name)
                if name in packages: raise ValueError("duplicate package")
                packages[name] = version
            result = store.add(schema.record("stack", name=args.name, python=args.python, backend=args.backend, packages=packages))
        elif cmd == "diff": result = diff_inventories(store.require(args.left, "inventory"), store.require(args.right, "inventory"))
        elif cmd == "compare": result = compare(store.require(args.inventory, "inventory"), store.require(args.stack, "stack"))
        elif cmd == "audit":
            from .policy import declarations
            result = store.add(schema.record("audit", inventory=args.inventory,
                rows=audit(store.require(args.inventory, "inventory"), args.repo, args.extra, args.override),
                declarations=[dict(repository=Path(repo).name, **declarations(repo)) for repo in args.repo]))
        elif cmd == "plan": result = command_plan(store.require(args.stack, "stack"))
        elif cmd == "campaign": result = store.create(args.name, args.target, args.rollback, args.environment, args.owner, args.review_date)
        elif cmd == "promote": result = store.promote(args.campaign, args.validation)
        elif cmd in {"adopt", "rollback"}: result = store.adopt(args.campaign, args.environment, args.inventory, cmd == "rollback", getattr(args, "reason", None))
        elif cmd == "show": result = store.get(args.digest) if args.digest else store.records()
        elif cmd == "board":
            from .board import render
            result = render(Path(args.root), Path(args.brain), Path(args.output), args.max_age_days)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"pyauto-dna: {error}\n")

if __name__ == "__main__":
    main()
