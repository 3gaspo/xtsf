"""Model-agnostic partial-dependence and accumulated-local-effect curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .adapters import ModelAdapter, adapt_model


@dataclass(frozen=True)
class PartialDependence:
    """One-dimensional partial-dependence values for every model output."""

    feature_name: str
    feature_index: int
    grid: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class AccumulatedLocalEffects:
    """Centered first-order ALE values for every model output."""

    feature_name: str
    feature_index: int
    grid: np.ndarray
    values: np.ndarray
    bin_edges: np.ndarray
    bin_counts: np.ndarray


def _prepare_data(
    data: Any,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"Effect curves expect shape (n, f), received {array.shape}.")
    columns = getattr(data, "columns", None)
    selected_names = (
        feature_names
        if feature_names is not None
        else (() if columns is None else map(str, columns))
    )
    names = tuple(selected_names)
    if not names:
        names = tuple(f"x{i}" for i in range(array.shape[1]))
    if len(names) != array.shape[1]:
        raise ValueError("feature_names must match the final data dimension.")
    return array.astype(np.result_type(array.dtype, float), copy=True), names


def _resolve_feature(feature: int | str, names: tuple[str, ...]) -> int:
    if isinstance(feature, str):
        if feature not in names:
            raise KeyError(f"Unknown feature name: {feature!r}.")
        return names.index(feature)
    index = int(feature)
    if not 0 <= index < len(names):
        raise IndexError(f"Feature index {index} is out of bounds.")
    return index


def _prediction_matrix(model: ModelAdapter, data: np.ndarray) -> np.ndarray:
    predictions = np.asarray(model.predict(data))
    if predictions.ndim == 0 or predictions.shape[0] != len(data):
        raise ValueError("The model must return one prediction row per input row.")
    if predictions.ndim == 1:
        return predictions[:, None]
    return predictions.reshape(len(data), -1)


def _quantile_grid(
    values: np.ndarray,
    count: int,
    quantiles: tuple[float, float],
) -> np.ndarray:
    lower, upper = map(float, quantiles)
    if count < 2:
        raise ValueError("The grid must contain at least two points.")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1.")
    bounds = np.quantile(values, [lower, upper])
    return np.linspace(bounds[0], bounds[1], count)


def partial_dependence(
    model: Any,
    data: Any,
    feature: int | str,
    *,
    feature_names: Sequence[str] | None = None,
    grid: Any | None = None,
    n_points: int = 20,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> PartialDependence:
    """Compute a one-dimensional PDP by replacing one feature for all rows."""

    array, names = _prepare_data(data, feature_names)
    index = _resolve_feature(feature, names)
    feature_grid = (
        _quantile_grid(array[:, index], n_points, quantiles)
        if grid is None
        else np.asarray(grid, dtype=float).reshape(-1)
    )
    if not len(feature_grid):
        raise ValueError("grid must contain at least one value.")
    adapter = adapt_model(model)
    averages = []
    for value in feature_grid:
        perturbed = array.copy()
        perturbed[:, index] = value
        averages.append(np.mean(_prediction_matrix(adapter, perturbed), axis=0))
    return PartialDependence(
        feature_name=names[index],
        feature_index=index,
        grid=feature_grid,
        values=np.asarray(averages),
    )


def accumulated_local_effects(
    model: Any,
    data: Any,
    feature: int | str,
    *,
    feature_names: Sequence[str] | None = None,
    n_bins: int = 10,
) -> AccumulatedLocalEffects:
    """Compute centered first-order ALE using empirical quantile bins."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least two.")
    array, names = _prepare_data(data, feature_names)
    index = _resolve_feature(feature, names)
    edges = np.unique(
        np.quantile(array[:, index], np.linspace(0.0, 1.0, n_bins + 1))
    )
    if len(edges) < 2:
        raise ValueError("ALE requires a feature with at least two distinct values.")
    bin_ids = np.searchsorted(edges[1:-1], array[:, index], side="right")
    adapter = adapt_model(model)
    local_effects: list[np.ndarray] = []
    counts: list[int] = []
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        rows = np.flatnonzero(bin_ids == bin_index)
        counts.append(len(rows))
        lower_data = array[rows].copy()
        upper_data = array[rows].copy()
        lower_data[:, index] = lower
        upper_data[:, index] = upper
        local_effects.append(
            np.mean(
                _prediction_matrix(adapter, upper_data)
                - _prediction_matrix(adapter, lower_data),
                axis=0,
            )
        )
    local = np.asarray(local_effects)
    uncentered = np.cumsum(local, axis=0) - 0.5 * local
    count_array = np.asarray(counts, dtype=int)
    center = np.average(uncentered, axis=0, weights=count_array)
    return AccumulatedLocalEffects(
        feature_name=names[index],
        feature_index=index,
        grid=0.5 * (edges[:-1] + edges[1:]),
        values=uncentered - center,
        bin_edges=edges,
        bin_counts=count_array,
    )
