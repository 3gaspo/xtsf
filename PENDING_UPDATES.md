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
