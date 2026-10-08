# PyAutoDNA

Specify. Compare. Evolve.

DNA records software stacks and the environments actually running them. It keeps
upstream availability, package declarations, tested configurations, recommendations
and observations separate. Brain coordinates changes; Heart supplies validation;
Hands owns releases; Nerves owns runtime compatibility. DNA never installs packages
or submits compute during collection, rendering or campaign actions.

Requires Python 3.12+; `python -m pip install -e .`. From this repository:

```bash
python -m dna collect local --repo ../PyAutoBrain
python -m dna board --brain ../PyAutoBrain --output _site
python -m dna show
```

Collection prints a private receipt and stores it in ignored `private/records/`.
Do not upload its console output: it includes paths and distribution origins.
`--probe-backend` explicitly initializes JAX to identify the actual backend; without
it backend remains unknown. Collect on RAL or CI using their actual interpreter,
then transfer the receipt privately and `python -m dna import receipt.json`.
No remote observation is inferred from a declaration or local interpreter.

See [operations](docs/operations.md), [contracts](docs/contracts.md) and the
[JAX incident](decisions/jax-2026-10-04.md). Checked-in environments are unobserved
until a receipt exists. Pages refresh alone cannot observe remote stacks.

The dashboard follows Brain's shared presentation standards and publishes the
standard `state.json`, `board.json` and `badge.json` feeds. Its source includes
reviewed observations of local development and the RAL shared environment; CPU,
GPU, CI and release validation remain unknown until their own receipts arrive.
`support.yaml` records upstream Python/JAX support horizons with source links and
a policy review date. These dates do not certify PyAuto compatibility.
