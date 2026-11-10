# JAX CPU policy: XLA:CPU deadlocks — November 2026

Status: DRAFT (research, PyAutoNerves#213). Awaiting the RAL repeat matrix and a Heart
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

RAL repeat matrix (0.9.2, 0.11.2, 0.12.0, 0.12.0.1 x 3 flag settings x 4 and 16 cores x 5
repeats, 300 calls): PENDING.

## Proposed policy (pending RAL + Heart)

1. Require `jax>=0.12.0,<0.13` and `jaxlib>=0.12.0,<0.13` in PyAutoNerves. Both bounds move
   together: jax 0.12.0 itself accepts jaxlib 0.11.2. This replaces the 0.7 floor and the
   0.10.* / 0.11.0 exclusions. Python (>=3.12) is unchanged; numpy's floor becomes 2.2; x86-64
   needs a Haswell (2013) or newer CPU.
2. Optionally keep `--xla_cpu_multi_thread_eigen=false` as a CPU-only default in PyAutoNerves'
   `jax_wrapper`, never overriding a flag the user set and never on GPU. On 0.12 it measured
   within about 10 % per call either way, so it is cheap insurance against a regression.
3. Guard: a scheduled Heart check runs the exported reproducers for 300 calls on every jax
   version the requirement allows and writes its result into DNA. The DNA board marks any
   observed environment whose jax version and XLA flags match a hanging setup.

## Open before deciding

- RAL repeat matrix (N=5, two core counts).
- Heart validation of the full library and workspace suites on 0.12 (CPU and GPU, including
  blackjax, optax, nufftax, jaxnnls and the CUDA plugins).
- The CI hang of `multi_dataset/jax_likelihood/delaunay_mge.py`, which hung twice on 0.11.2
  even with the eigen flag set: rerun on 0.12.

Review date: 2026-12-10.
