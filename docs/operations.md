# Operations

All mutable actions append immutable records. A SHA-256 digest identifies a
complete record, including its timestamp. Existing files cannot be overwritten.
Use digest outputs in the following commands (substitute actual 64-digit values):

```bash
python -m dna stack candidate --python 3.12.9 --backend cpu --package jax==0.11.1
python -m dna compare INVENTORY_DIGEST STACK_DIGEST
python -m dna diff LOCAL_INVENTORY_DIGEST RAL_INVENTORY_DIGEST
python -m dna audit INVENTORY_DIGEST --repo ../PyAutoArray --extra jax --override 'jax==0.11.1'
python -m dna plan STACK_DIGEST
python -m dna campaign jax-update --target TARGET_DIGEST --rollback BASELINE_DIGEST --environment local --owner human --review-date 2026-11-04
python -m dna promote CAMPAIGN_DIGEST --validation HEART_VALIDATION_DIGEST
python -m dna adopt CAMPAIGN_DIGEST local INVENTORY_DIGEST
python -m dna rollback CAMPAIGN_DIGEST local BASELINE_INVENTORY_DIGEST --reason 'Regression after adoption'
```

Example versions demonstrate syntax, not recommendations. `plan` emits argument
arrays for a resolver dry run; it executes nothing. The human/Brain creates and
reviews an isolated candidate environment. Heart validation is imported as a
signed-off record (see contracts), matching the exact stack and environment.
Promotion requires successful evidence for every campaign environment and an
unexpired review date. Adoption requires promotion and a matching fresh inventory
identity, Python, package pins and observed backend. Rollback records an actual
matching baseline observation and reason; it does not perform the rollback.
Retain baseline records and environments until the campaign closes.

Audit reads current pyproject declarations, selected extras and environment
markers. Dynamic dependencies and direct URLs stay unknown; build metadata must
resolve them. CI workflow installation/interpreter lines are review findings,
not support policy. Explicit overrides are checked separately. No source
`__version__` is compared to a wheel as an equality gate.

Suggested cadence: collect after any environment change, review drift weekly,
review support at releases and after dependency incidents. This is advice; no
schedule is enabled. Scientific runs retain their original inventory digest.
RAL observations must be collected on RAL under the workspace HPC policy; DNA
never chooses a partition or runs SSH. CI can collect its actual interpreter as
a private artifact; a Pages build interpreter is not a library validation run.

Pages publishing defaults to checked-in declarations and public decisions only.
Private observations need an explicitly reviewed allowlisted board projection
before upload. Never commit private receipts, resolver logs or machine paths.

Explicit upstream discovery and reviewed public publishing:

```bash
python -m dna upstream jax --python 3.12.9
python -m dna export-public --output inventories/public.json
python -m dna board --brain ../PyAutoBrain
```

Upstream receipts show absolute latest and latest stable non-yanked release with
Python-compatible files separately. Wheel platform, backend and full dependency
resolution remain untested. Review the allowlisted public JSON before committing;
Pages loads it and ages its original collection timestamps. Collection alone does
not publish. Exported records omit private evidence prose and owner identities.
For git/runtime constraints, import a stack record with `repositories` entries
(name, sha, dirty) and `runtime_flags` mapping; compare/adoption enforce them.

The initial RAL receipt is a shared-virtualenv observation taken on the login node. It does not certify CPU or GPU compute-node behavior. Raw receipts remain private; the public board includes only approved package names and source identities.
