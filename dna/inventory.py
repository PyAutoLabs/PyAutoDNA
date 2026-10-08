"""Read-only collection in the actual interpreter; private receipts are not feeds."""
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from .schema import record

PUBLIC_REPOSITORIES = {"PyAutoBrain", "PyAutoMind", "PyAutoDNA", "PyAutoNerves", "PyAutoFit", "PyAutoArray", "PyAutoGalaxy", "PyAutoLens", "PyAutoCTI", "PyAutoReduce", "PyAutoHeart", "PyAutoHands"}

PUBLIC_PACKAGES = {"anesthetic", "astropy", "matplotlib", "scikit-learn", "dynesty", "nautilus-sampler", "emcee", "zeus-mcmc", "corner", "optax", "blackjax", "numpyro", "tensorflow-probability", "tfp-nightly", "nufftax", "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin", "jax-cuda13-pjrt", "autofit", "autoarray", "autogalaxy", "autolens", "autocti", "autonerves", "jax", "jaxlib", "numpy", "scipy", "numba", "python", "pyautodna", "packaging", "pyyaml"}

FLAGS = ("JAX_ENABLE_X64", "JAX_PLATFORMS", "JAX_PLATFORM_NAME", "XLA_FLAGS", "CUDA_VISIBLE_DEVICES", "PYAUTO_TEST_MODE", "PYAUTO_DISABLE_JAX", "PYAUTO_TEST")

def command(args, cwd=None):
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None

def collect(environment, repositories=(), probe_backend=False):
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        from packaging.utils import canonicalize_name
        name = canonicalize_name(name)
        origin = dist.read_text("direct_url.json")
        packages[name] = {"version": dist.version, "location": str(dist.locate_file("")),
                          "origin": json.loads(origin) if origin else None}
    repos = []
    for path in repositories:
        path = Path(path).resolve()
        sha = command(["git", "rev-parse", "HEAD"], path)
        if not sha:
            raise ValueError(f"not a git checkout: {path}")
        repos.append({"name": path.name, "sha": sha, "dirty": (lambda status: None if status is None else bool(status))(command(["git", "status", "--porcelain"], path)),
                      "path": str(path), "branch": command(["git", "branch", "--show-current"], path)})
    imports = {}
    for name in ("autofit", "autoarray", "autogalaxy", "autolens", "autocti", "autonerves", "jax"):
        try:
            spec = importlib.util.find_spec(name)
            imports[name] = spec.origin if spec else None
        except (ImportError, ValueError):
            imports[name] = None
    runtime = {"flags": {key: os.environ[key] for key in FLAGS if key in os.environ},
               "backend": "unknown", "devices": [], "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])}
    if probe_backend:
        try:
            import jax
            runtime.update(backend={"gpu": "cuda"}.get(jax.default_backend(), jax.default_backend()), devices=[str(d) for d in jax.devices()], x64=bool(jax.config.x64_enabled))
        except Exception as error:
            runtime["probe_error"] = type(error).__name__
    return record("inventory", environment=environment, python=platform.python_version(),
                  executable=sys.executable, platform={"system": platform.system(), "machine": platform.machine(), "markers": __import__("packaging.markers", fromlist=["default_environment"]).default_environment()},
                  packages=packages, repositories=repos, imports=imports, runtime=runtime)

def public_inventory(obj):
    """Construct from allowed fields; never redact a raw object in place."""
    return {"digest": obj["digest"], "environment": obj["environment"], "created": obj["created"],
            "python": obj["python"], "packages": {n: {"version": p["version"]} for n, p in obj["packages"].items() if n in PUBLIC_PACKAGES},
            "repositories": [{k: r[k] for k in ("name", "sha", "dirty")} for r in obj["repositories"] if r["name"] in PUBLIC_REPOSITORIES],
            "runtime": {"backend": obj["runtime"].get("backend", "unknown"), "x64": obj["runtime"].get("x64")}}
