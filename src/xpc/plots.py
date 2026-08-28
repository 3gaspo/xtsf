"""Lazy-Matplotlib plots for effect curves, errors, and Shapley explanations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .diagnostics import error_summary
from .effects import AccumulatedLocalEffects, PartialDependence
from .explanation import Explanation


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install xpc[notebook] to use plotting helpers.") from error
    return plt


def _curve_axes(ax: Any | None):
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    return ax.figure, ax


def _plot_curve_values(
    ax: Any,
    grid: np.ndarray,
    values: np.ndarray,
    output: int | None,
) -> None:
    if output is None:
        for index in range(values.shape[1]):
            ax.plot(grid, values[:, index], marker="o", label=f"output {index}")
        if values.shape[1] > 1:
            ax.legend()
        return
    if not 0 <= output < values.shape[1]:
        raise IndexError(f"Output index {output} is out of bounds.")
    ax.plot(grid, values[:, output], marker="o", label=f"output {output}")


def plot_partial_dependence(
    result: PartialDependence,
    *,
    output: int | None = 0,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot a precomputed partial-dependence curve."""

    figure, axes = _curve_axes(ax)
    _plot_curve_values(axes, result.grid, result.values, output)
    axes.set_title(f"Partial dependence: {result.feature_name}")
    axes.set_xlabel(result.feature_name)
    axes.set_ylabel("Mean prediction")
    axes.grid(alpha=0.25)
    figure.tight_layout()
    return figure, axes


def plot_accumulated_local_effects(
    result: AccumulatedLocalEffects,
    *,
    output: int | None = 0,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot a precomputed centered first-order ALE curve."""

    figure, axes = _curve_axes(ax)
    _plot_curve_values(axes, result.grid, result.values, output)
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_title(f"Accumulated local effects: {result.feature_name}")
    axes.set_xlabel(result.feature_name)
    axes.set_ylabel("Centered local effect")
    axes.grid(alpha=0.25)
    figure.tight_layout()
    return figure, axes


def _select_output(values: Any, output: int | None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if output is None:
        return array.reshape(-1)
    if array.ndim == 1:
        if output != 0:
            raise IndexError(f"Output index {output} is out of bounds.")
        return array
    if not 0 <= output < array.shape[-1]:
        raise IndexError(f"Output index {output} is out of bounds.")
    return array.reshape(-1, array.shape[-1])[:, output]


def plot_prediction_errors(
    targets: Any,
    predictions: Any,
    *,
    output: int | None = None,
    x: Any | None = None,
    bins: int = 30,
    title: str = "Prediction errors",
) -> tuple[Any, np.ndarray]:
    """Plot predictions, residual sequence, and a residual histogram."""

    truth = _select_output(targets, output)
    predicted = _select_output(predictions, output)
    if truth.shape != predicted.shape:
        raise ValueError("Selected targets and predictions must have the same shape.")
    positions = np.arange(len(truth)) if x is None else np.asarray(x)
    if len(positions) != len(truth):
        raise ValueError("x must match the selected target length.")
    residuals = predicted - truth
    summary = error_summary(truth, predicted)
    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(positions, truth, label="target", linewidth=1.2)
    axes[0].plot(positions, predicted, label="prediction", linewidth=1.2)
    axes[0].set_title("Targets and predictions")
    axes[0].legend()
    axes[1].plot(positions, residuals, color="tab:red", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Residuals (prediction - target)")
    axes[2].hist(residuals, bins=bins, color="tab:red", alpha=0.8)
    axes[2].axvline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("Residual distribution")
    for axis in axes[:2]:
        axis.set_xlabel("Observation")
        axis.grid(alpha=0.2)
    axes[2].set_xlabel("Residual")
    axes[2].set_ylabel("Count")
    figure.suptitle(
        f"{title} | MAE={summary.mae:.3g}, RMSE={summary.rmse:.3g}, "
        f"bias={summary.bias:.3g}"
    )
    figure.tight_layout()
    return figure, axes


def _unit_index(explanation: Explanation, unit: int | tuple[int, ...]) -> tuple[int, ...]:
    index = unit if isinstance(unit, tuple) else (int(unit),)
    expected = explanation.values.ndim - 2
    if len(index) != expected:
        raise ValueError(f"unit must contain {expected} batch indexes.")
    return index


def plot_shapley_waterfall(
    explanation: Explanation,
    *,
    unit: int | tuple[int, ...] = 0,
    output: int = 0,
    max_display: int | None = 12,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot raw signed group contributions from base value to reconstruction."""

    index = _unit_index(explanation, unit)
    if not 0 <= output < explanation.values.shape[-2]:
        raise IndexError(f"Output index {output} is out of bounds.")
    contributions = np.asarray(
        explanation.values[index + (output, slice(None))], dtype=float
    )
    names = np.asarray(explanation.group_names, dtype=object)
    order = np.argsort(-np.abs(contributions))
    contributions = contributions[order]
    names = names[order]
    if max_display is not None:
        if max_display < 1:
            raise ValueError("max_display must be positive or None.")
        if len(contributions) > max_display:
            kept = max_display - 1
            other_count = len(contributions) - kept
            contributions = np.concatenate(
                [contributions[:kept], [np.sum(contributions[kept:])]]
            )
            names = np.concatenate(
                [names[:kept], [f"other {other_count} groups"]]
            )

    base = float(explanation.base_values[index + (output,)])
    prediction = float(explanation.predictions[index + (output,)])
    running = base
    starts = []
    widths = []
    endpoints = []
    for contribution in contributions:
        endpoint = running + contribution
        starts.append(min(running, endpoint))
        widths.append(abs(contribution))
        endpoints.append(endpoint)
        running = endpoint

    figure, axes = _curve_axes(ax)
    positions = np.arange(len(contributions))
    colors = ["tab:red" if value >= 0 else "tab:blue" for value in contributions]
    axes.barh(positions, widths, left=starts, color=colors, alpha=0.85)
    span = max(abs(running - base), np.ptp([base, prediction, running]), 1.0)
    for position, endpoint, contribution in zip(positions, endpoints, contributions):
        axes.text(
            endpoint + np.sign(contribution or 1.0) * 0.01 * span,
            position,
            f"{contribution:+.3g}",
            va="center",
            ha="left" if contribution >= 0 else "right",
            fontsize=9,
        )
    axes.axvline(base, color="gray", linestyle=":", label=f"base={base:.3g}")
    axes.axvline(
        prediction,
        color="black",
        linestyle="--",
        label=f"prediction={prediction:.3g}",
    )
    if not np.isclose(running, prediction):
        axes.axvline(
            running,
            color="tab:purple",
            linestyle="-.",
            label=f"reconstruction={running:.3g}",
        )
    axes.set_yticks(positions, names)
    axes.invert_yaxis()
    axes.set_xlabel("Model output")
    axes.set_title("Raw Shapley contribution waterfall")
    axes.grid(axis="x", alpha=0.2)
    axes.legend()
    figure.tight_layout()
    return figure, axes
