# XTSF

XTSF develops a focused Monte Carlo Shapley toolkit, published as the `xpc`
Python package, for NumPy, pandas, sklearn,
PyTorch, script-backed, and custom models. It supports ordinary tabular data
and temporal tensors shaped `(n, d, f)`.

The package intentionally contains one explanation algorithm: Monte Carlo
Shapley estimation over explicit feature-group players. Kernel SHAP,
Permutation SHAP, GAM-specific paths, SMACH-specific paths, generative
conditioning, and estimator-side known contributions are not part of the API.

## Install

```bash
pip install -e .
pip install -e ".[test]"
```

Optional dependencies are split into `pandas`, `torch`, `notebook`, and `all`
extras.

## Project Structure

- `src/xpc/`: package implementation
- `src/xtsf.ipynb`: the single Colab-ready synthetic forecasting notebook
- `src/tests/`: dependency-light unit and smoke tests
- `outputs/`: generated notebook datasets and other experiment artifacts
- `logs/`: runtime logs

The notebook exposes an early `on_drive` switch. In local mode it uses the
project-relative `datasets/` directory; in Drive mode it mounts Google Drive
and changes into the repository's `src/` directory. Its generated synthetic
panel is cached under `outputs/`. It also benchmarks interventional and
conditional Monte Carlo Shapley estimates against the known structural
contributions while varying coalition and mask-sample budgets. A second
benchmark reproduces the original XPC aggregation study: post-hoc aggregation,
Coalitional Shapley, and the two-player Simplified Shapley definition are
compared at a fixed budget under both masking rules, against structural truth
and each mode's exact all-coalition estimand.

## Quick Start

```python
import numpy as np
from xpc import BaselineMasker, ShapleyExplainer

X = np.array([[1.0, 2.0], [3.0, 4.0]])
model = lambda x: x[:, 0] + 2 * x[:, 1]

explanation = ShapleyExplainer(
    model,
    BaselineMasker("mean"),
    n_coalitions=128,
    random_state=0,
)(X)

# (n, H, G): two rows, one output, two feature-group players
print(explanation.values.shape)
```

## Core Abstractions

- `DataSpec`: tabular `(n, f)` data and model output normalization.
- `TimeSeriesTensorSpec`: temporal `(n, d, f)` data. Each `(n, d)` location is
  explained while the model receives its complete sequence.
- `ModelAdapter`: common prediction boundary for NumPy, pandas/sklearn,
  PyTorch, R/scripts, callables, and registered custom classes.
- `Masker`: baseline, random empirical, or conditional empirical masking.
- `Conditioner`: empirical nearest-neighbor, grid-cell, or user-defined
  non-generative conditional sampling.
- `FeatureGroups`: named Shapley players and always-present features.
- `ShapleyExplainer`: Monte Carlo subset estimator returning signed values and
  attached heightened parts by default; pass `heighten=False` to disable.
- `Explanation`: signed values, precomputed contribution ingestion,
  heightening, and reference comparison.

## Temporal Models

```python
from xpc import TimeSeriesTensorSpec

X = np.random.default_rng(0).normal(size=(8, 24, 5))

def model(x):
    # Output shape (n, d, H)
    total = x.sum(axis=-1)
    return np.stack([total, 2 * total], axis=-1)

explanation = ShapleyExplainer(
    model,
    BaselineMasker(0),
    data_spec=TimeSeriesTensorSpec(
        feature_names=["load", "temperature", "wind", "hour", "holiday"]
    ),
    n_coalitions=64,
)(X)

assert explanation.values.shape == (8, 24, 2, 5)
```

For temporal data, XPC duplicates the explained sample's sequence, masks one
time location in each variant, runs the sequence model, and extracts the
corresponding output location.

## Feature Groups

```python
from xpc import FeatureGroups

groups = FeatureGroups(
    {
        "weather": ["temperature", "wind"],
        "calendar": ["hour", "holiday"],
    },
    remaining="individual",  # or "group" / "ignore"
    always_present=["load"],
)
```

Custom groups must be disjoint. Remaining features may become individual
players, one named remaining player, or no player. Always-present features are
included in every coalition and therefore contribute through the base value.

## Masking

```python
from xpc import (
    BaselineMasker,
    ConditionalMasker,
    EmpiricalConditioner,
    GridConditioner,
    RandomMasker,
)

baseline = BaselineMasker("median", background=background)
random = RandomMasker(background)
nearest = ConditionalMasker(background, EmpiricalConditioner(n_neighbors=50))
grid = ConditionalMasker(background, GridConditioner(n_bins=10))
```

Baselines may be `"zero"`, `"mean"`, `"median"`, a scalar, feature vector,
data-shaped/broadcastable tensor, or callable.

## Model Adapters

`adapt_model` accepts callables, fitted objects with `predict`, explicit
adapters, and registered custom model classes.

```python
from xpc import (
    PandasModelAdapter,
    RScriptModelAdapter,
    TorchModelAdapter,
    register_model_adapter,
)

pandas_adapter = PandasModelAdapter(model, ["a", "b"])
torch_adapter = TorchModelAdapter(torch_model, device="cpu")
r_adapter = RScriptModelAdapter("predict.R", output_columns=["prediction"])
```

`ScriptModelAdapter` passes temporary CSV paths through `{input}` and
`{output}` command placeholders. Scripts must write a headered output CSV.

## Precomputed Contributions And Heightening

Known contributions do not enter the estimator. Ingest them explicitly:

```python
from xpc import Explanation

explanation = Explanation.from_contributions(
    signed_contributions,
    targets=positive_targets,
    group_names=["weather", "calendar"],
)

heightened = explanation.heightened
assert np.allclose(heightened.parts.sum(axis=-1), positive_targets)

comparison = explanation.compare(reference_contributions)
print(comparison.mae, comparison.per_group_mae)
```

Heightening is enabled by default for `ShapleyExplainer` and
`Explanation.from_contributions`. It preserves `raw_values`, shifts group
values positive, computes percentages, and creates parts that sum to each
positive target. Use `heighten=False` when only signed values are needed.

## Tests And Notebook

Run the dependency-light suite from the project root with:

```bash
PYTHONPATH=src python -m unittest discover -s src/tests -v
```

PyTorch, sklearn, and R tests skip when their runtimes are unavailable. See
`src/xtsf.ipynb` for the synthetic end-to-end forecasting explanation and
heightened contribution plots. The convergence benchmark exports its repeated
runs, summary, and figure as `outputs/shapley_convergence_runs.csv`,
`outputs/shapley_convergence_summary.csv`, and
`outputs/shapley_convergence.png`. The aggregation benchmark writes
`outputs/aggregation_exact_estimands.csv`,
`outputs/aggregation_mode_runs.csv`,
`outputs/aggregation_mode_summary.csv`, and
`outputs/aggregation_modes.png`.

## Provenance

This project is a streamlined continuation of the original XPC research code.
The historical implementation is retained separately as read-only thesis
archive material; it is a reference, not a second copy of the active package.

## LaTeX documents

`latex/experiment_guideline.tex` is the current theoretical, implementation,
experiment, cache, and artifact specification. `latex/executive_summary.tex`
contains the analyzed convergence and aggregation results without duplicating
the protocol. Their PDFs are kept beside the sources.

## Maintenance workflow

Every project change is recorded in `PENDING_UPDATES.md` with its scope,
affected contracts, focused checks already completed, deferred integration
coverage, documentation impact, and rerun requirements. Routine edits use only
the smallest relevant smoke check. Periodic maintenance verifies pending entries
against the implementation, runs complementary generic lightweight smoke tests,
reconciles this README and the project LaTeX documents, and renders affected
PDFs before resolving the entries.
