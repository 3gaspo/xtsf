"""Build the synthetic XPC notebook without requiring nbformat."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    markdown(
        """# XPC: synthetic tabular and temporal Monte Carlo Shapley

This notebook exercises the complete public API on deterministic synthetic
models: NumPy and pandas inputs, optional sklearn/PyTorch/R adapters, baseline,
random, empirical, grid, and callable masking, every grouping policy,
precomputed contributions, heightening, and comparison with known signed
contributions.

The examples use deliberately small arrays so the notebook remains quick. Raise
`n_coalitions` and `n_mask_samples` for production estimates."""
    ),
    code(
        """from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np

ROOT = Path.cwd().resolve()
if ROOT.name == "examples":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from xpc import (
    BaselineMasker, CallableConditioner, ConditionalMasker, DataSpec,
    EmpiricalConditioner, Explanation, FeatureGroups, GridConditioner,
    NumpyModelAdapter, PandasModelAdapter, RScriptModelAdapter,
    RandomMasker, ScriptModelAdapter, ShapleyExplainer,
    TimeSeriesTensorSpec, TorchModelAdapter,
)

rng = np.random.default_rng(42)"""
    ),
    markdown(
        """## 1. NumPy tabular baseline explanations

For an additive linear model and a deterministic zero baseline, every marginal
contribution is exact regardless of sampled coalition."""
    ),
    code(
        """X = rng.normal(size=(6, 4))
weights = np.array([1.0, -2.0, 0.5, 3.0])
numpy_model = lambda x: x @ weights

tabular = ShapleyExplainer(
    numpy_model,
    BaselineMasker("zero"),
    data_spec=DataSpec(feature_names=["a", "b", "c", "d"]),
    n_coalitions=16,
    random_state=0,
)(X)

known = X * weights
print("shape:", tabular.values.shape)
print("max contribution error:", np.max(np.abs(tabular.values[:, 0] - known)))
print("max efficiency residual:", np.max(np.abs(tabular.efficiency_residual)))"""
    ),
    markdown("## 2. pandas and optional sklearn adapters"),
    code(
        """try:
    import pandas as pd
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    pandas_model = lambda df: df["a"] - 2 * df["b"] + 0.5 * df["c"] + 3 * df["d"]
    pandas_explanation = ShapleyExplainer(
        PandasModelAdapter(pandas_model, frame.columns),
        BaselineMasker("mean", background=frame),
        data_spec=DataSpec(feature_names=frame.columns),
        n_coalitions=16,
        random_state=1,
    )(frame)
    print("pandas:", pandas_explanation.values.shape)
except ImportError:
    print("pandas is optional; install xpc[pandas]")

try:
    from sklearn.linear_model import LinearRegression
    fitted = LinearRegression().fit(X, numpy_model(X))
    sklearn_explanation = ShapleyExplainer(
        fitted, BaselineMasker(0), n_coalitions=8, random_state=2
    )(X[:2])
    print("sklearn:", sklearn_explanation.values.shape)
except ImportError:
    print("scikit-learn is optional; install xpc[test]")"""
    ),
    markdown(
        """## 3. Grouping modes

Groups are the Shapley players. The examples below cover custom groups,
remaining-as-individual, remaining-as-one-group, and always-present features."""
    ),
    code(
        """feature_names = ["load", "temperature", "wind", "hour"]

individual = FeatureGroups.individual()
custom_plus_individual = FeatureGroups(
    {"weather": ["temperature", "wind"]}, remaining="individual"
)
custom_plus_remaining = FeatureGroups(
    {"weather": ["temperature", "wind"]},
    remaining="group",
    remaining_name="non_weather",
)
always_present = FeatureGroups(
    {"weather": ["temperature", "wind"]},
    remaining="individual",
    always_present=["load"],
)

for groups in [individual, custom_plus_individual, custom_plus_remaining, always_present]:
    resolved = groups.resolve(4, feature_names)
    print(resolved.names, "always:", resolved.always_present)"""
    ),
    markdown("## 4. Random and non-generative conditional maskers"),
    code(
        """background = rng.normal(size=(500, 4))
background[:, 1] = background[:, 0] + rng.normal(scale=0.1, size=500)
points = background[:3] + 0.25

maskers = {
    "random": RandomMasker(background),
    "empirical": ConditionalMasker(
        background, EmpiricalConditioner(n_neighbors=30)
    ),
    "grid": ConditionalMasker(
        background, GridConditioner(n_bins=8, strategy="quantile")
    ),
    "user": ConditionalMasker(
        background,
        CallableConditioner(
            lambda *, reference, n_samples, rng, **_:
                reference[rng.integers(0, len(reference), size=n_samples)]
        ),
    ),
}

for name, masker in maskers.items():
    result = ShapleyExplainer(
        numpy_model,
        masker,
        n_coalitions=12,
        n_mask_samples=20,
        random_state=3,
    )(points)
    print(name, result.values.shape, "mean |residual|", np.abs(result.efficiency_residual).mean())"""
    ),
    markdown(
        """## 5. Synthetic temporal tensor with two horizons

`TimeSeriesTensorSpec` explains all `(n, d)` locations. The model still sees the
full sequence for each perturbation. Values return as `(n, d, H, G)`."""
    ),
    code(
        """n, d, f = 3, 8, 4
temporal_X = rng.normal(size=(n, d, f))
temporal_weights = np.array([1.0, -1.5, 2.0, 0.25])

def temporal_model(x):
    local = x @ temporal_weights
    # A context term demonstrates that the full sequence remains available.
    context = 0.1 * x[:, :, 0].mean(axis=1, keepdims=True)
    horizon_1 = local + context
    return np.stack([horizon_1, 2.0 * horizon_1], axis=-1)

temporal_groups = FeatureGroups(
    {"weather": [1, 2]},
    remaining="individual",
)
temporal = ShapleyExplainer(
    temporal_model,
    BaselineMasker(np.zeros((1, d, f))),
    data_spec=TimeSeriesTensorSpec(
        feature_names=["load", "temperature", "wind", "calendar"]
    ),
    feature_groups=temporal_groups,
    n_coalitions=32,
    random_state=4,
)(temporal_X)

print("values:", temporal.values.shape)
print("base:", temporal.base_values.shape)
print("predictions:", temporal.predictions.shape)
print("groups:", temporal.group_names)"""
    ),
    markdown("## 6. Optional PyTorch temporal adapter"),
    code(
        """try:
    import torch

    class TemporalTorchModel(torch.nn.Module):
        def forward(self, x):
            local = x.sum(dim=-1)
            return torch.stack([local, 2 * local], dim=-1)

    torch_result = ShapleyExplainer(
        TorchModelAdapter(TemporalTorchModel()),
        BaselineMasker(0),
        data_spec=TimeSeriesTensorSpec(),
        n_coalitions=8,
        random_state=5,
    )(temporal_X[:1])
    print("torch:", torch_result.values.shape)
except ImportError:
    print("PyTorch is optional; install xpc[torch]")"""
    ),
    markdown("## 7. Script adapter and optional R adapter"),
    code(
        """try:
    import pandas as pd
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        script = directory / "predict.py"
        script.write_text(
            "import pandas as pd, sys\\n"
            "x = pd.read_csv(sys.argv[1])\\n"
            "pd.DataFrame({'prediction': x.sum(axis=1)}).to_csv(sys.argv[2], index=False)\\n"
        )
        adapter = ScriptModelAdapter(
            [sys.executable, str(script), "{input}", "{output}"],
            feature_names=["a", "b"],
            output_columns=["prediction"],
        )
        print("script predictions:", adapter.predict(np.array([[1.0, 2.0]])))

    if shutil.which("Rscript"):
        with tempfile.TemporaryDirectory() as directory:
            r_script = Path(directory) / "predict.R"
            r_script.write_text(
                "args <- commandArgs(trailingOnly=TRUE)\\n"
                "x <- read.csv(args[1])\\n"
                "write.csv(data.frame(prediction=rowSums(x)), args[2], row.names=FALSE)\\n"
            )
            r_adapter = RScriptModelAdapter(
                r_script, output_columns=["prediction"]
            )
            print("R predictions:", r_adapter.predict(np.array([[1.0, 2.0]])))
    else:
        print("Rscript is optional and was not found.")
except ImportError:
    print("Script CSV adapters require pandas.")"""
    ),
    markdown(
        """## 8. Precomputed signed contributions, heightening, and comparison

Precomputed values enter through `Explanation.from_contributions`, never through
the estimator. Heightening is attached by default, leaves raw values intact,
creates non-negative weights, and allocates positive targets exactly."""
    ),
    code(
        """signed = np.array([
    [3.0, -1.0, 2.0],
    [-2.0, 4.0, 1.0],
])
targets = np.array([20.0, 30.0])
precomputed = Explanation.from_contributions(
    signed,
    targets=targets,
    group_names=["weather", "calendar", "load"],
)
heightened = precomputed.heightened

print("raw preserved:", np.array_equal(heightened.raw_values[:, 0], signed))
print("positive:", np.all(heightened.positive_values >= 0))
print("percentage sums:", heightened.percentages.sum(axis=-1).ravel())
print("part sums:", heightened.parts.sum(axis=-1).ravel())

reference = signed + np.array([[0.1, 0.0, -0.2], [0.0, 0.2, 0.0]])
comparison = precomputed.compare(reference)
print("MAE:", comparison.mae)
print("per-group MAE:", dict(zip(comparison.group_names, comparison.per_group_mae)))"""
    ),
    markdown(
        """## 9. Practical checks

- Increase coalition draws until values stabilize.
- Increase mask samples for random and conditional value estimates.
- Treat grouping and the masking distribution as part of the explanation
  question, not merely tuning parameters.
- Inspect `efficiency_residual`; Monte Carlo conditional explanations generally
  close only approximately.
- Keep signed values for analysis. Use heightening only when a positive target
  decomposition is the desired downstream representation."""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).with_name("synthetic_temporal_toolkit.ipynb")
output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(output)
