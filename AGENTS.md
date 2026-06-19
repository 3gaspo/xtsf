# Repository Guidelines

<!-- edf-project:begin -->

## Project Snapshot

`xpc` is a Python 3.10+ explainability pipeline for time series forecasting
models, centered on Monte Carlo Shapley explanations over temporal `(n, d, f)`
tensors and supporting tabular `(n, f)` arrays. Keep it focused on explicit
feature-group Shapley estimation unless the user asks otherwise.

## Project Structure & Module Organization

- `src/xpc/`: package source for adapters, conditioners, data specs,
  explanations, explainer logic, feature groups, and maskers. Public exports
  live in `src/xpc/__init__.py`.
- `tests/`: lightweight local tests, named by behavior such as
  `test_explainer.py` and `test_maskers_and_conditioners.py`.
- `examples/`: synthetic notebook and `build_synthetic_notebook.py`.
- `docs/`: LaTeX package summary and `render_with_miktex.ps1`.
- Any `*_old/` folder is historical reference material only.

## Build, Test, and Development Commands

```powershell
pip install -e .
pip install -e ".[test]"
$env:PYTHONPATH="src"; python -m unittest discover -s tests -v
$env:PYTHONPATH="src"; pytest
powershell -ExecutionPolicy Bypass -File docs\render_with_miktex.ps1
```

Use the first command for editable installs, the second for test dependencies,
then run either `unittest` or `pytest`. The docs command rebuilds
`docs/package_summary.pdf` with MiKTeX.

## Coding Style & Naming Conventions

Use 4-space indentation and the existing compact, NumPy-oriented style. Classes
use `PascalCase`; functions, variables, methods, and test helpers use
`snake_case`. Keep shape contracts explicit. No formatter or linter is
configured, so match surrounding code and document public APIs.

## Testing Guidelines

Tests use `unittest` style and also run under `pytest`. Name new files
`test_<area>.py` and methods `test_<expected_behavior>`. Cover estimator
behavior, shape handling, error messages, and optional dependency skips. Keep
local validation small; real experiments run on the distant cluster from
another PC.

## Local Execution & Tool Reference

This PC is for coding, static checks, unit tests, and lightweight loading tests.
Do not run heavy training, large inference, full data processing, or
data-producing experiments locally unless asked.

- Python: `C:\Users\Gaspard\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` (Python 3.12.13)
- Torch: not importable from the bundled Python runtime.
- Git: `C:\Program Files\Git\cmd\git.exe` (git version 2.45.0.windows.1);
  not on PATH in the default shell.
- MiKTeX: `C:\Users\Gaspard\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe`
  (MiKTeX-pdfTeX 4.23 / MiKTeX 25.12).

## Commit, PR, and README Guidelines

The user handles Git commits, pulls, pushes, and branch integration unless
explicitly requested. If asked to commit, use short imperative messages such as
`Fix temporal output shape handling`. PRs should summarize motivation, changed
behavior, test commands, and optional dependency impact.

Keep `README.md` synchronized when scripts, configs, data flow, public APIs, or
loading behavior change.

<!-- edf-project:end -->
