---
policy: jax-cpu
title: JAX CPU policy
status: Recommended 2026-11-11 (human-approved). The PyAutoNerves floor change is in progress, pending a Heart validation of jax 0.12 and a release.
declared_in: PyAutoNerves
packages: [jax, jaxlib]
recommended:
  jax: ">=0.12.0,<0.13"
  jaxlib: ">=0.12.0,<0.13"
matrix_package: jax
review_date: 2026-12-10
findings:
  - "jax 0.11.2 with no XLA flag hangs in compiled CPU likelihoods: on RAL every reproducer hung on its first call at 4 cores (25 of 25 cells), and none hung at 16 cores (0 of 25)."
  - "jax 0.12.0 and 0.12.0.1 with no flag: 0 hangs in 30,000 calls, at 4 and 16 cores."
  - "The eigen flag (`--xla_cpu_multi_thread_eigen=false`) avoids the hang on 0.11.2 but slows pixelized likelihoods by 18 % to about 50 % on 0.12, so it is not a default."
  - "The FFT part of the bug is upstream jax-ml/jax#41265, fixed by openxla/xla#50091 in jax 0.12.0."
links:
  - {label: "Research PyAutoNerves#213", url: "https://github.com/PyAutoLabs/PyAutoNerves/issues/213"}
  - {label: "Floor change PyAutoNerves#215", url: "https://github.com/PyAutoLabs/PyAutoNerves/issues/215"}
  - {label: "Hang matrix scripts", url: "https://github.com/PyAutoLabs/autolens_profiling/tree/main/scripts/misc/jax_version_policy"}
---
# JAX CPU policy: XLA:CPU deadlocks — November 2026

Status: RECOMMENDATION (research, PyAutoNerves#213). Evidence complete; awaiting a Heart
validation of the proposed range before any requirement changes.
Supersedes: [jax-2026-10-04](jax-2026-10-04.md) (review date 2026-11-04 passed).
Owner: PyAutoBrain; validation authority: PyAutoHeart; evidence: PyAutoDNA research records.

## The problem

On the CPU backend, a compiled JAX likelihood sometimes stops for good: a later call of the same
compiled executable never returns and the process sits at 0 % CPU. Since August 2026 this was
patched five times in five places (test-workspace CI flags, Heart release legs, a profiling
driver, a private RAL job script, and a version exclusion for a different LAPACK path). None of
these reached users' installs, and the then "recommended validation baseline", jax 0.11.2,
still hangs.

## Evidence (DNA hang matrix)

Each reproducer runs many times in a process that imports only jax (or, for Delaunay, through
the library with PyAutoNerves' default XLA flags). A stall with no CPU work is a hang.
Laptop pilot, 8 cores, 2026-11-10, one repeat per cell:

| Reproducer | 0.11.2 no flag | 0.11.2 eigen flag | 0.12.0.1 no flag | 0.12.0.1 eigen flag |
|---|---|---|---|---|
| group MGE (data vector) | hang, call 4 | pass, 0.064 s | pass, 0.062 s | pass, 0.064 s |
| multi_galaxy MGE (full) | hang, call 94 | pass, 0.048 s | pass, 0.045 s | pass, 0.040 s |
| group Delaunay (full, library) | hang, call 1 | pass, 4.54 s | pass, 4.30 s | pass, 4.31 s |
| FFT pool (scatter-fed FFTs) | hang, call 1 | pass | pass | pass |

- The scheduler flag (`--xla_cpu_enable_concurrency_optimized_scheduler=false`) also avoids the
  hang on 0.11.2 but costs 1.4-2x per call on MGE; on 0.12.0.1 it is still 1.6-2x slower.
- 0.9.2 and 0.10.2 ran the 0.11.2-exported group MGE module without a hang (20 calls).
- The batched-Cholesky case did not reproduce the October LAPACK hang even on 0.10.2, so it is
  not evidence either way; the 0.10.* / 0.11.0 exclusion rests on its original reproduction.
- Upstream: the FFT-pool deadlock is jax-ml/jax#41265 (regression in 0.11.1), fixed by
  openxla/xla#50091 (merged 2026-10-06), released in jax 0.12.0 (2026-10-19). No upstream issue
  was found for the MGE / Delaunay hang (no thread working, unlike the FFT latch).

RAL repeat matrix (job 415009, 2026-11-11, AMD EPYC 7763 nodes): jax 0.9.2 / 0.11.2 / 0.12.0 /
0.12.0.1 x {no flag, eigen flag, scheduler flag} x {4, 16} cores x 5 repeats, 300 calls per cell,
600 cells, 180,000 calls in all.

| jax | no flag, 4 cores | no flag, 16 cores | eigen flag | scheduler flag |
|---|---|---|---|---|
| 0.11.2 | **hang in 25 / 25 cells, all on the first call** | pass 25 / 25 | pass 50 / 50 | pass 50 / 50 |
| 0.12.0 | pass 25 / 25 | pass 25 / 25 | pass 50 / 50 | pass 50 / 50 |
| 0.12.0.1 | pass 25 / 25 | pass 25 / 25 | pass 50 / 50 | pass 50 / 50 |
| 0.9.2 | pass (fft, group MGE data vector) | pass | pass | pass |

0.9.2 cannot read the three full-likelihood modules exported from 0.11.2 (StableHLO is newer
than it), so it is only partly tested.

What this adds to the pilot:

- On 0.11.2 the hang depends on the cores the process has: always at 4 cores, intermittent at 8
  (laptop), not seen at 16. Small containers, CI runners and laptops are the most exposed; a
  large node can hide the bug.
- Per-call cost of the flags on 0.12.0.1 (steady median):

  | Likelihood | cores | no flag | eigen flag | scheduler flag |
  |---|---|---|---|---|
  | group MGE (full) | 4 / 16 | 0.056 / 0.040 s | 0.055 / 0.040 s | 0.086 / 0.084 s |
  | multi_galaxy MGE (full) | 4 / 16 | 0.037 / 0.025 s | 0.038 / 0.026 s | 0.062 / 0.056 s |
  | multi_galaxy rectangular (full) | 4 / 16 | 1.44 / 0.91 s | 1.71 / 1.35 s | 1.27 / 0.91 s |
  | fft | 4 / 16 | 0.159 / 0.071 s | 0.146 / 0.120 s | 0.139 / 0.113 s |

  The eigen flag is free for MGE but costs 18 % (4 cores) to about 50 % (16 cores) on the
  pixelized likelihood; the scheduler flag costs 1.5-2x on MGE. Neither is free everywhere.

## Recommended policy (pending Heart validation)

1. Require `jax>=0.12.0,<0.13` and `jaxlib>=0.12.0,<0.13` in PyAutoNerves. Both bounds move
   together: jax 0.12.0 itself accepts jaxlib 0.11.2. This replaces the 0.7 floor and the
   0.10.* / 0.11.0 exclusions. Python (>=3.12) is unchanged; numpy's floor becomes 2.2; x86-64
   needs a Haswell (2013) or newer CPU. 0.12.0 and 0.12.0.1 behave the same on Linux.
2. Do not add a default XLA:CPU flag. On 0.12 no reproducer needs one, and the eigen flag slows
   pixelized likelihoods by up to about 50 %. Remove the per-place eigen-flag workarounds once the
   floor is released (CI test workspaces, Heart release legs, the autolens_profiling 2d driver,
   job scripts), each re-checked on 0.12 first.
3. Guard: a scheduled Heart check runs the exported reproducers for 300 calls at 4 cores on
   every jax version the requirement allows (4 cores is where the 0.11.2 hang is certain) and
   writes the result into DNA. The DNA board marks any observed environment whose jax version
   and XLA flags match a hanging setup.

## Open before deciding

- Heart validation of the full library and workspace suites on 0.12 (CPU and GPU, including
  blackjax, optax, nufftax, jaxnnls and the CUDA plugins).
- The CI hang of `multi_dataset/jax_likelihood/delaunay_mge.py`, which hung twice on 0.11.2
  even with the eigen flag set: rerun on 0.12.

Review date: 2026-12-10.
