"""Universal Monte Carlo Shapley explanations."""

from .adapters import (
    CallableModelAdapter,
    ModelAdapter,
    NumpyModelAdapter,
    PandasModelAdapter,
    RScriptModelAdapter,
    SklearnModelAdapter,
    ScriptModelAdapter,
    TorchModelAdapter,
    adapt_model,
    register_model_adapter,
)
from .conditioners import (
    CallableConditioner,
    Conditioner,
    EmpiricalConditioner,
    GridConditioner,
)
from .data import DataSpec, TimeSeriesTensorSpec
from .diagnostics import ErrorSummary, error_summary
from .effects import (
    AccumulatedLocalEffects,
    PartialDependence,
    accumulated_local_effects,
    partial_dependence,
)
from .explanation import Comparison, Explanation, HeightenedExplanation
from .explainer import ShapleyExplainer
from .groups import FeatureGroups, ResolvedFeatureGroups
from .inspection import (
    ParameterCounts,
    ParameterInfo,
    parameter_counts,
    parameter_structure,
)
from .maskers import BaselineMasker, ConditionalMasker, Masker, RandomMasker
from .plots import (
    plot_accumulated_local_effects,
    plot_partial_dependence,
    plot_prediction_errors,
    plot_shapley_waterfall,
)
from .synthetic import SyntheticForecastingData, make_synthetic_forecasting_data

__all__ = [
    "BaselineMasker",
    "AccumulatedLocalEffects",
    "CallableConditioner",
    "CallableModelAdapter",
    "Comparison",
    "ConditionalMasker",
    "Conditioner",
    "DataSpec",
    "EmpiricalConditioner",
    "ErrorSummary",
    "Explanation",
    "FeatureGroups",
    "GridConditioner",
    "HeightenedExplanation",
    "Masker",
    "ModelAdapter",
    "NumpyModelAdapter",
    "PandasModelAdapter",
    "ParameterCounts",
    "ParameterInfo",
    "PartialDependence",
    "RScriptModelAdapter",
    "RandomMasker",
    "ResolvedFeatureGroups",
    "SklearnModelAdapter",
    "ScriptModelAdapter",
    "ShapleyExplainer",
    "SyntheticForecastingData",
    "TimeSeriesTensorSpec",
    "TorchModelAdapter",
    "adapt_model",
    "accumulated_local_effects",
    "error_summary",
    "make_synthetic_forecasting_data",
    "parameter_counts",
    "parameter_structure",
    "partial_dependence",
    "plot_accumulated_local_effects",
    "plot_partial_dependence",
    "plot_prediction_errors",
    "plot_shapley_waterfall",
    "register_model_adapter",
]
