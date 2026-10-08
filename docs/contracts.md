# Record and feed contracts

Records use schema_version 1, kind, created (UTC ISO8601), and digest. Digests
hash canonical sorted JSON without the digest field. `dna.schema.validate`
is the normative validator. Stacks pin Python/package versions and backend;
inventories keep distribution versions/origins/import paths and separate git
SHA/dirty identities. Public projections exclude all locations, hosts, raw
runtime flags, origins, owners, free-form private reasons and evidence URLs.
Only reviewed decision Markdown is public prose.

A Heart validation record has `stack` (digest), `environment` (registry ID),
`authority: PyAutoHeart`, `result: pass|fail|unknown`, `checks` (nonempty list),
and `evidence` (the private Heart run/artifact reference). Produce it with
`dna.schema.record('validation', ...)`, persist with `dna.schema.write`, and
import through the CLI. DNA checks structure and matching identity; it cannot
cryptographically authenticate the author of an imported Heart receipt. Human
review remains required. No record implies universal compatibility certification.

`board.json` is DNA's v1 public snapshot. `state.json` uses Brain's shared v1
cockpit validator. `badge.json` is a Shields endpoint. Updated timestamps record
render time; each observation separately retains collection time. Unknown is
never green; observed means collected, not supported. Evidence older than the
configured seven-day default is stale. Source declarations remain authoritative
and are read only when audit is explicitly invoked.

Validation requires `inventory` digest pointing to the tested observation.
Promotion selects the latest known exact-target Heart receipt per environment;
newer failures/unknown evidence block older passes. Validation and adoption
observations expire after seven days. Unknown backend blocks promotion.
Imported validation is supplied evidence, not an authenticated release verdict.
Public package/repository names use explicit library/organ allowlists in
`dna.inventory`; add reviewed identities there before publishing other projects.

Imports accept evidence records, never campaign/promotion/adoption/rollback
transitions. Those must pass the explicit command gates. Adoption reruns the
complete promotion gate, including current review date, latest matching evidence,
expiry and tested inventory identity. Target environment marker facts are captured
with packaging's default_environment in that interpreter; missing facts are unknown,
never replaced by the audit host's facts.

`audit` records store the inventory digest, dated compatibility rows and read
pyproject declaration hashes, Python bounds, requirements/extras and dynamic flags.
These are snapshots of authoritative declarations at audit time, not new support
policy. CI findings identify source filenames/line numbers for human review;
DNA does not execute workflow code. Public compatibility rows should retain only
approved package/repository identities and declarative constraints.
