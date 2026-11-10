"""Import a JAX hang-matrix run as DNA research evidence.

The matrix (autolens_profiling ``scripts/misc/jax_version_policy/run_matrix.py``) writes one
JSON row per cell: a reproducer run N times under one jax version and one ``XLA_FLAGS``
setting, in an interpreter that imports only jax. Each row's ``header`` is observed inside
that interpreter, so DNA records it as an inventory of the actual interpreter, not a guess.

Per (environment, jax version, XLA flags) group this writes:

- an ``inventory`` built from the observed header (jax, jaxlib, Python, backend, flags);
- a ``stack`` pinning that Python, jax and jaxlib, with ``runtime_flags`` holding the flags;
- a ``validation`` with ``authority: PyAutoDNA`` (research evidence; never sufficient for
  promotion, see :mod:`dna.campaigns`), one check per reproducer row.

Result: ``fail`` if any reproducer hung or errored, ``pass`` if every row passed, otherwise
``unknown`` (slow stalls, unloadable modules). PyAutoLabs/PyAutoNerves#213.
"""
import json
from collections import defaultdict
from pathlib import Path
from .schema import identifier, record

#: ``runtime.source`` of a matrix inventory: it identifies the test interpreter only and never
#: counts as an observation of the environment itself.
MATRIX_SOURCE = "hang-matrix header"

CHECK_FIELDS = ("case", "outcome", "calls_requested", "calls_completed", "hung_at_call",
                "median_call_s", "steady_median_call_s", "stall_cpu_fraction", "cpus_pinned", "repeat")


def _flag_label(flags):
    names = sorted(token.split("=")[0].lstrip("-") for token in flags.split() if token)
    return "+".join(names) if names else "default"


def result_for(rows):
    outcomes = {r["outcome"] for r in rows}
    if outcomes & {"hang", "error"}:
        return "fail"
    if outcomes == {"pass"}:
        return "pass"
    return "unknown"


def records_from_rows(rows, evidence):
    """Return the inventory/stack/validation records for parsed matrix rows (not persisted)."""
    groups = defaultdict(list)
    for row in rows:
        head = row.get("header")
        if not head:  # a cell that never started cannot say which interpreter it was
            continue
        if head.get("backend") != "cpu":
            raise ValueError("hang-matrix rows must come from the CPU backend")
        groups[(identifier(row["environment"]), head["jax"], head["jaxlib"], head["python"], head["xla_flags"])].append(row)
    out = []
    for (environment, jax, jaxlib, python, flags), cells in sorted(groups.items()):
        runtime_flags = {"XLA_FLAGS": flags, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "True"}
        inventory = record("inventory", environment=environment, python=python,
                           packages={"jax": {"version": jax}, "jaxlib": {"version": jaxlib}},
                           repositories=[], runtime={"flags": runtime_flags, "backend": "cpu", "x64": True,
                                                     "devices": [], "source": MATRIX_SOURCE})
        stack = record("stack", name=f"jax-{jax}-{_flag_label(flags)}", python=python, backend="cpu",
                       packages={"jax": jax, "jaxlib": jaxlib}, runtime_flags=runtime_flags)
        checks = [{k: c.get(k) for k in CHECK_FIELDS} for c in cells]
        validation = record("validation", stack=stack["digest"], environment=environment,
                            authority="PyAutoDNA", result=result_for(cells), checks=checks,
                            evidence=evidence, inventory=inventory["digest"])
        out += [inventory, stack, validation]
    return out


def import_matrix(store, path, evidence=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("no matrix rows")
    objs = records_from_rows(rows, evidence or f"hang-matrix:{Path(path).name}")
    for obj in objs:
        store.add(obj)
    return [{"kind": o["kind"], "digest": o["digest"], **({"name": o["name"]} if "name" in o else {}),
             **({"result": o["result"]} if o["kind"] == "validation" else {})} for o in objs]
