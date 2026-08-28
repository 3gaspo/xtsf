# XTSF

XTSF develops a focused Monte Carlo Shapley toolkit, published as the `xpc`
Python package, for NumPy, pandas, sklearn,
PyTorch, script-backed, and custom models. It supports ordinary tabular data
and temporal tensors shaped `(n, d, f)`. Reusable diagnostics add
one-dimensional PDP and ALE curves, prediction-error plots, raw-Shapley
waterfalls, parameter metadata, and seeded synthetic forecasting windows.

The package intentionally contains one explanation algorithm: Monte Carlo
Shapley estimation over explicit feature-group players. Kernel SHAP,
Permutation SHAP, GAM-specific paths, SMACH-specific paths, generative
conditioning, and estimator-side known contributions are not part of the API.

## Install

```bash
pip install -e .
pip install -e ".[test]"
pip install -e ".[notebook]"
```

Optional dependencies are split into `pandas`, `torch`, `notebook`, and `all`
extras. The `notebook` extra includes Matplotlib and Hugging Face Transformers
for the PatchTST demonstration.

## Project Structure

- `src/xpc/adapters.py`, `data.py`, `groups.py`, `maskers.py`, and
  `conditioners.py`: model/data boundaries and coalition construction
- `src/xpc/explainer.py` and `explanation.py`: Monte Carlo Shapley estimation
  and contribution post-processing
- `src/xpc/effects.py` and `diagnostics.py`: PDP, first-order ALE, and error
  computations
- `src/xpc/plots.py`: lazy-Matplotlib effect, error, and waterfall plots
- `src/xpc/inspection.py`: named-parameter counts and structure
- `src/xpc/synthetic.py`: seeded synthetic series and forecasting windows
- `src/xtsf.ipynb`: the Colab-ready reference synthetic Shapley experiment
- `src/patchtst_explainability.ipynb`: compact PatchTST diagnostics tutorial
- `src/tests/`: dependency-light unit and smoke tests
- `outputs/`: generated notebook datasets and other experiment artifacts
- `logs/`: runtime logs

Both notebooks expose an early `on_drive` switch. In local mode they use the
project-relative `datasets/` directory; in Drive mode they mount Google Drive
and add the repository's `src/` directory to the import path. The reference
notebook's generated synthetic panel is cached under `outputs/`. It also
benchmarks interventional and conditional Monte Carlo Shapley estimates
against the known structural contributions while varying coalition and
mask-sample budgets. A second
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
- `PartialDependence` and `AccumulatedLocalEffects`: reusable multi-output
  effect-curve results, separated from plotting.
- `ParameterCounts` and `ParameterInfo`: framework-light named-parameter
  metadata for models exposing `named_parameters()`.
- `SyntheticForecastingData`: seeded series components and supervised windows.

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

## Effect Curves And Diagnostic Plots

PDP and first-order ALE computations accept the same explicit model adapters
as the Shapley estimator and preserve every flattened model output:

```python
from xpc import (
    accumulated_local_effects,
    partial_dependence,
    plot_accumulated_local_effects,
    plot_partial_dependence,
)

pdp = partial_dependence(model, X, "temperature", feature_names=feature_names)
ale = accumulated_local_effects(
    model, X, "temperature", feature_names=feature_names, n_bins=10
)
plot_partial_dependence(pdp, output=0)
plot_accumulated_local_effects(ale, output=0)
```

`plot_prediction_errors` draws targets/predictions, residual order, and a
residual histogram while reporting MAE, RMSE, and bias.
`plot_shapley_waterfall` plots one output of one `Explanation` from its base
value through raw signed group contributions. A distinct reconstruction line
is retained when finite Monte Carlo error leaves a nonzero efficiency residual.
These diagnostics are associational and remain conditional on the supplied
data and masking distribution.

## Parameter Inspection

`parameter_counts(model)` reports total, trainable, and frozen elements and
tensor counts. `parameter_structure(model)` returns each named tensor's name,
shape, element count, trainability, and dtype without copying weight values.
The model must expose `named_parameters()`, as PyTorch modules do; opaque
callable and script adapters remain prediction-only.

## Synthetic Forecasting Data

`make_synthetic_forecasting_data` provides a deterministic univariate series
with level, trend, daily, weekly, interaction, and noise components. It also
returns chronological context and target tensors shaped
`(samples, context, 1)` and `(samples, horizon, 1)`. One `seed` owns the noise
and repeated calls with the same settings are identical.

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
heightened contribution plots. See `src/patchtst_explainability.ipynb` for a
small locally trained PatchTST example covering the synthetic helper, model
structure, error plots, PDP, first-order ALE, and a grouped Shapley waterfall.
The convergence benchmark exports its repeated
runs, summary, and figure as `outputs/shapley_convergence_runs.csv`,
`outputs/shapley_convergence_summary.csv`, and
`outputs/shapley_convergence.png`. The aggregation benchmark writes
`outputs/aggregation_exact_estimands.csv`,
`outputs/aggregation_mode_runs.csv`,
`outputs/aggregation_mode_summary.csv`, and
`outputs/aggregation_modes.png`.

## Publishing artifacts

Run the thesis-standard publisher from the project Git root to synchronize
`main`, commit the complete lightweight `logs/` and `outputs/` trees plus paired
`logs_selena/` and lightweight `outputs_selena/` trees when present, and push:

```bash
bash publish_job.sh
bash publish_job.sh --size detailed
```

The script sources `$HOME/codes/proxy.sh`, fast-forward pulls `origin/main`
before staging, commits only selected artifact paths, and pushes `origin/main`.
Lightweight is the default and omits row-level window/user-date/sample tables
and per-run criterion/example plots; `--size detailed` adds them. Both tiers
exclude `*.pt`, `*.npy`, and `*.cbm`. `PROXY_SCRIPT_PATH` overrides the proxy
location. A numeric job ID publishes only its exact log pair; a partial Selena
namespace fails closed.

Before staging, each selected non-excluded file larger than 100,000,000 bytes
is replaced for publication by `<original>.sample.txt`. Text samples contain
source metadata and the first 10% of content, capped at 10,000,000 bytes;
binary samples contain metadata only. The header retains the first UTC time
when the associated file became stale on Git because of its size. The original
is excluded literally from
both staging and commit selection. `PUBLISH_MAX_FILE_BYTES` and
`PUBLISH_SAMPLE_MAX_BYTES` override the positive byte limits, with the sample
limit required to remain smaller.

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
the smallest relevant smoke check. Brief daily triage compares stored
fingerprints and updates the queue only for new source, artifact, or external
state; unchanged blockers are carried forward. Broad weekly maintenance verifies
changed entries against the implementation, runs complementary lightweight
integration checks, reconciles this README and the project LaTeX documents, and
renders affected PDFs before resolving entries.
