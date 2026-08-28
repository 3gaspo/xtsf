# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-28: Made Git artifact publication tiered: lightweight now omits
  row-level/per-run diagnostics by default, while `--size detailed` adds them;
  both tiers retain the cluster-binary exclusions. Affected contracts:
  `publish_job.sh`, README guidance, and a focused transfer check. Git Bash
  syntax and the focused check passed, and all nine publisher copies were
  byte-identical. No scientific rerun or LaTeX update is required. Deferred
  integration: inspect one detailed publication on DGX.

- 2026-08-27: Added reusable first-order PDP/ALE computations, prediction-error
  summaries and plots, raw-Shapley waterfall plots, named-parameter counts and
  structure, and a seeded synthetic forecasting-window generator. Added a
  Colab-ready tutorial that trains a compact Hugging Face PatchTST and applies
  every helper. Affected contracts: the public `xpc` API, notebook/all optional
  dependencies, README source map and usage, tests, and
  `src/patchtst_explainability.ipynb`. Seven focused helper tests passed,
  including a real PyTorch parameter-count case; dependency-light public import
  and TOML parsing passed; both new plot layouts were visually inspected; and
  the 19-cell notebook passed nbformat validation and Python syntax parsing.
  The reference experiment protocol, existing artifacts, guideline, and
  executive summary are unaffected, so no scientific rerun or LaTeX rebuild is
  required. Deferred integration: execute the complete PatchTST notebook in a
  user-prepared `.[notebook]` environment because the available thesis runtime
  does not contain Transformers.

- 2026-08-27: Added a stable oversized-sample header recording the first UTC
  time and file-size reason that the associated artifact became stale on Git.
  Affected contracts: `publish_job.sh`, README publication guidance, and the
  shared focused publisher regressions. All five publisher checks passed; Git
  Bash syntax and byte parity passed for all nine active publisher copies. No
  scientific rerun or LaTeX change is required. Deferred integration: exercise
  one real oversized publication and inspect the generated header on DGX.

Maintenance 2026-08-27: direct notebook-project, README, guideline/PDF
freshness, repository-status, and placeholder inspection found no affected
scientific or document surface. All nine publisher copies passed Git Bash
syntax and byte-parity checks, and the representative publisher regression
passed. Notebook smoke and PDF rebuild were skipped as inapplicable to the
publisher-only change. A real oversized publication remains pending.

- 2026-08-27: Hardened the thesis-standard publisher against GitHub's
  100 MB file limit. Before staging, each selected non-excluded file above
  100,000,000 bytes is excluded literally and represented by
  `<original>.sample.txt`; text samples contain source metadata and the first
  10% capped at 10,000,000 bytes, while binary samples retain metadata only.
  Affected contracts: publisher, README, shared publication guidance, and the
  five maintained publisher regressions where present. Git Bash syntax passed
  for all nine active copies, all five focused publisher checks passed, and
  both publisher and test copies are byte-identical. No scientific rerun,
  artifact migration, or LaTeX change is required. Deferred integration:
  exercise one real oversized log publication on DGX.

- 2026-08-26: Updated the shared publisher so unscoped publication includes
  paired `logs_selena/` and lightweight `outputs_selena/` trees when present,
  under the existing `*.pt`, `*.npy`, and `*.cbm` exclusions; numeric job-ID
  mode remains standard-log-only and a partial Selena namespace fails closed.
  Publisher and README changed. Bash syntax passed for all 15 maintained
  scripts, all five publisher checks passed, and all nine publisher copies are
  byte-identical. The experiment guideline/PDF is unaffected. No notebook rerun
  or migration is required; exercise the broader scope if Selena artifacts are
  introduced.

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
