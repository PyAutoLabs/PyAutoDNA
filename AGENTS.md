# PyAutoDNA

Follow workspace AGENTS.md and approved start-dev worktree flow. DNA owns stack
identities, observations, support rationale and upgrade/adoption records. Brain
coordinates, Heart validates, Hands releases, Nerves guards runtime compatibility.
No automatic upgrades, compute submission, schedules or package publication.
Never publish private records. Public feeds are allowlisted projections; retain
unknown/missing/stale evidence. Render with explicit Brain checkout and shared
board/_theme.py and validate state.json with board/_state.py. Test with
`python -m unittest discover -s tests -v` after sourcing task activate.sh.


<!-- repos_sync:map:begin -->
**You are one organ of the PyAuto organism** — an agentic ecosystem for
human-led, natural-language software development. The organs below are
peer repositories; this repo is one of them, not a part of another.
Canonical boundaries live in `PyAutoBrain/ORGANISM.md`; the full body map
(every repo, not just organs) is `PyAutoMind/repos.yaml`.

| Organ | Repo | Role |
|-------|------|------|
| **Brain** | PyAutoBrain | Reasoning/orchestration layer; how work is decomposed and routed; the specialist agents. |
| **Broca** | PyAutoBroca | Assistant evaluation history, upkeep evidence, collection receipts and an operational dashboard. Brain interprets; Mind tracks fixes; public assistants remain independent. |
| **Mind** | PyAutoMind | Intent, goals, priorities, workflow state; every task starts as a markdown prompt here. |
| **Cortex** | PyAutoCortex | The Cortex — what is true in the science: the body map (`projects.yaml`) and one ledger per science project (runs, results, learnings, where to pick up); the science mirror of the Mind. |
| **Memory** | PyAutoMemory | Long-term scientific/software/project knowledge (see science pointer below). |
| **Eyes** | PyAutoEyes | The Eyes — cross-project visualization dashboard over the `<lib>_visualization` repos: their registry, the `gallery/viz_manifest.yaml` read contract and the Pages board linking their PNGs. Renders and copies nothing, never judges figures (the Brain's Eyes conductor does) and never edits plot code. |
| **Ears** | PyAutoEars | The Ears — community listening: read-only public conversation collection, the versioned community snapshot contract, coverage receipts and the dashboard. GitHub stays authoritative; Brain's Community conductor judges and drafts replies; never posts, labels, files issues or exports transcripts. |
| **Heart** | PyAutoHeart | Health/readiness — the authoritative "is it safe to release?" verdict. |
| **Hands** | PyAutoHands | Packaging, tagging, notebook generation, PyPI release execution. |
| **Pulse** | PyAutoPulse | The Pulse — cross-project profiling dashboard over the `<lib>_profiling` repos: campaign intent, instance registry, the versioned `profiling-summary` read contract, ingest receipts and the Pages board. Validates the contract only; never judges, scores or issues verdicts (the Brain's profiling conductor judges). |
| **Insight** | PyAutoInsight | Inference campaign intent, the cross-project inference instance registry, the versioned `inference-summary` read contract, ingest receipts and the evidence dashboard. Projects execute, Cortex records the science, Mind owns task state; never infers scientific acceptance or submits compute. |
| **DNA** | PyAutoDNA | Software-stack specifications, environment inventory receipts, support rationale, upgrade campaigns and adoption history. Brain coordinates, Heart validates readiness, Hands releases, Nerves enforces runtime compatibility; collection never upgrades environments. |
| **Nerves** | PyAutoNerves | The Nerves — the configuration/serialization layer connecting workspace conventions to libraries (layered config, version handshake, test_mode), delivered as the `autonerves` package. |
| **Gut** | PyAutoGut | Lifecycle of condemned self-material (stale branches, stashes, dead code/tests): held as recoverable git refs through a transit window, voided on a sweep. The storage mirror of Memory. |

Call chain (always this order): **Brain → Heart (gate) → Build (execute)**. Brain agents are **conductors** (front-door; a human drives them; they decide *and* act) or **faculties** (read-only opinions the conductors consult; they judge and stop). New capability grows as a faculty, not a new organ, unless it owns state or effects no existing organ can.

Generated from `PyAutoMind/repos.yaml` + `PyAutoBrain/ORGANISM.md`; edit there, then run `python3 PyAutoMind/scripts/repos_sync.py --write`.
<!-- repos_sync:map:end -->

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

<!-- repos_sync:deliverable:begin -->
## Sessions end at their deliverable

A session ends when it reports its deliverable — never arm anything that
outlives the turn to wait for CI, a review or a merge: no `send_later`, no
`subscribe_pr_activity`, no `CronCreate`, no `ScheduleWakeup`, no `/loop`, no
`RemoteTrigger` create/update/run. Judge once, report, stop; the human re-runs
`/prm` (or the batch review) when it is green. Measured: five batch members
armed hourly check-ins on 2026-08-31, and a mobile `/prm` re-armed a 60-minute
`send_later` hourly all night on 2026-09-03 with no task active, draining usage.
<!-- repos_sync:deliverable:end -->

<!-- repos_sync:filing:begin -->
## Where to file

Questions, help with code or an analysis, ideas, bug reports and results from a
user or collaborator — or an agent acting for one — go to
<https://github.com/orgs/PyAutoLabs/discussions> in the matching category
(Help & Questions, Ideas & Proposals, Bugs & Errors, Show and tell;
Announcements is maintainers-only), never to this repo's Issues. An agent never
runs `gh issue create` for such a report: it drafts the title, category and
body and hands them to the human (sessions cannot create Discussions). Only the
development flow — Mind prompt → `/start_dev` → `/create_issue` → one issue per
task → PR — opens issues here. Why: `PyAutoMind/policy/community_surface.md`.
<!-- repos_sync:filing:end -->

<!-- repos_sync:standards:begin -->
## Shared standards

Before changing a shared interface, consult the applicable
[organism standard](https://github.com/PyAutoLabs/PyAutoBrain/blob/main/docs/standards.md)
on demand, identify affected consumers, and validate their adoption. Change
generated guidance at its canonical source and regenerate.

For board changes, follow the applicable sizing, navigation and orchestration
standards and reuse Brain’s shared components. Keep domain data, prompt meaning
and approval boundaries with the board’s owner.
<!-- repos_sync:standards:end -->
