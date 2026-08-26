# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-17: Simplify `publish_job.sh`: a numeric job ID now selects only its
  exact stdout/stderr pair, while an omitted ID stages the `logs/` and
  lightweight `outputs/` parent trees directly. Publisher and README changed.
  Git Bash syntax passed, and all nine copies have matching SHA-256 hashes. No
  notebook rerun, artifact migration, or result-document update is required.
  Deferred integration remains the first real cluster publish.

- 2026-08-16: Add and document the thesis-standard `publish_job.sh` for
  lightweight notebook artifacts. It sources the proxy and fast-forward pulls
  `origin/main` before staging or committing, excludes heavy payloads, and then
  pushes. Checks passed: Bash syntax for all nine project copies and matching
  SHA-256 hashes. No notebook rerun, artifact migration, or LaTeX update is
  required. Deferred integration: exercise the publisher on the cluster once
  this project has artifacts to synchronize.

Maintenance 2026-08-17: direct inspection confirmed that the README matches
the canonical publisher and that no notebook, artifact, or LaTeX contract
changed. Bash syntax passed for all nine byte-identical project copies; this was
repeated because the publisher is the only executable integration boundary.
Notebook smoke and PDF rendering were skipped as inapplicable. The first real
cluster publish remains required before this entry can be resolved.

Maintenance 2026-08-18: direct inspection found no package, notebook,
artifact, or LaTeX contract change beyond the simplified publisher, and the
README already matches its exact-log/full-tree behavior. Git Bash syntax passed
for all nine byte-identical publisher copies; this shared check was repeated
because the script is the only changed executable boundary. Package smoke and
PDF rendering were skipped as inapplicable. The first real cluster publish
remains required before resolution.

Maintenance 2026-08-19: direct inspection found no package, notebook,
artifact, or public-contract change. The README still matches the canonical
publisher, and all nine publisher copies remain byte-identical at SHA-256
`0A9E87E51517B9F5816BB92CDE726B9E383AB6B8A70DC251FEF429BF7B53B45C`.
The unchanged package/notebook checks, Bash syntax, and PDFs were not repeated
because no integration boundary changed. The first real cluster publish
remains required before resolution.

Maintenance 2026-08-20: direct timestamp, repository, artifact, and notebook
inspection found no package, output, log, or public-contract change after the
previous pass. The publisher remains byte-identical across all nine projects
at SHA-256
`0A9E87E51517B9F5816BB92CDE726B9E383AB6B8A70DC251FEF429BF7B53B45C`.
Package/notebook, Bash-syntax, and PDF checks were deliberately skipped because
their inputs and integration boundaries are unchanged. The first real cluster
publish remains required before resolution.

Maintenance 2026-08-24: direct repository, notebook, artifact, README, LaTeX,
and guidance inspection found no code or public-contract change behind the
local value-based guidance update. That guidance-only entry is resolved.
Package/notebook tests and PDF rendering were deliberately skipped as
inapplicable; the first real publisher run remains pending.

Maintenance 2026-08-26: direct repository, notebook, README, LaTeX timestamp,
active-output, and archive-manifest inspection confirmed zero active payloads
and the eight administratively archived files totaling 659,662 bytes. The
archive-only entry is resolved without changing its historical evidence.
Package, notebook, publisher, and PDF checks were not repeated because no
corresponding boundary changed. The first real publisher run remains pending.
