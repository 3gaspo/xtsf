"""Explanation containers and post-processing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np


def _broadcast_outputs(value: Any, target_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if target_shape and target_shape[-1] == 1 and array.shape == target_shape[:-1]:
        array = array[..., None]
    return np.broadcast_to(array, target_shape).copy()


@dataclass(frozen=True)
class Comparison:
    """Error summary against externally supplied reference contributions."""

    mae: float
    rmse: float
    bias: float
    max_absolute_error: float
    correlation: float
    per_group_mae: np.ndarray
    group_names: tuple[str, ...]


@dataclass(frozen=True)
class HeightenedExplanation:
    """Positive post-processing derived from signed contributions."""

    raw_values: np.ndarray
    positive_values: np.ndarray
    percentages: np.ndarray
    parts: np.ndarray
    targets: np.ndarray
    offsets: np.ndarray
    group_names: tuple[str, ...]


@dataclass(frozen=True)
class Explanation:
    """Signed Monte Carlo Shapley values and their prediction context."""

    values: np.ndarray
    base_values: np.ndarray
    predictions: np.ndarray
    group_names: tuple[str, ...]
    feature_groups: tuple[tuple[int, ...], ...] = ()
    feature_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    heightened: HeightenedExplanation | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        base = np.asarray(self.base_values)
        predictions = np.asarray(self.predictions)
        if values.ndim < 2:
            raise ValueError("values must include output and group axes.")
        if values.shape[-1] != len(self.group_names):
            raise ValueError("The final values axis must match group_names.")
        if values.shape[:-1] != base.shape or base.shape != predictions.shape:
            raise ValueError(
                "values must have shape batch + (H, G), while base_values and "
                "predictions must have shape batch + (H,)."
            )

    @classmethod
    def from_contributions(
        cls,
        contributions: Any,
        *,
        targets: Any | None = None,
        base_values: Any = 0.0,
        group_names: Sequence[str] | None = None,
        feature_groups: Sequence[Sequence[int]] = (),
        feature_names: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        heighten: bool = True,
    ) -> "Explanation":
        """Ingest precomputed signed contributions without running an estimator."""

        values = np.asarray(contributions)
        if values.ndim == 1:
            values = values[None, None, :]
        elif values.ndim == 2:
            values = values[:, None, :]
        if values.ndim < 3:
            raise ValueError("contributions must include a final group axis.")
        names = tuple(group_names or (f"group_{i}" for i in range(values.shape[-1])))
        target_shape = values.shape[:-1]
        base = _broadcast_outputs(base_values, target_shape)
        predictions = (
            base + np.sum(values, axis=-1)
            if targets is None
            else _broadcast_outputs(targets, target_shape)
        )
        explanation = cls(
            values=values,
            base_values=base,
            predictions=predictions,
            group_names=names,
            feature_groups=tuple(tuple(group) for group in feature_groups),
            feature_names=tuple(feature_names),
            metadata=dict(metadata or {}, source="precomputed"),
        )
        if heighten:
            return replace(explanation, heightened=explanation.heighten())
        return explanation

    @property
    def efficiency_residual(self) -> np.ndarray:
        return self.predictions - (self.base_values + np.sum(self.values, axis=-1))

    def heighten(
        self,
        *,
        targets: Any | None = None,
        alpha: float = 1.0,
        minimums: Any | None = None,
    ) -> HeightenedExplanation:
        """Shift signed groups positive, normalize them, and allocate targets.

        Offsets are computed per output and group over all batch axes. Raw
        Shapley values remain untouched in this object and in ``raw_values``.
        """

        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        raw = np.asarray(self.values)
        batch_axes = tuple(range(raw.ndim - 2))
        observed_minimums = np.min(raw, axis=batch_axes) if batch_axes else raw
        if minimums is not None:
            observed_minimums = np.broadcast_to(
                np.asarray(minimums), observed_minimums.shape
            )
        offsets = alpha * np.maximum(-observed_minimums, 0.0)
        positive = np.maximum(raw + offsets, 0.0)
        totals = np.sum(positive, axis=-1, keepdims=True)
        percentages = np.divide(
            positive,
            totals,
            out=np.full_like(positive, 1.0 / positive.shape[-1], dtype=float),
            where=totals != 0,
        )
        target_values = (
            self.predictions
            if targets is None
            else _broadcast_outputs(targets, raw.shape[:-1])
        )
        if np.any(target_values < 0):
            raise ValueError("Heightening targets must be non-negative.")
        parts = percentages * target_values[..., None]
        return HeightenedExplanation(
            raw_values=raw.copy(),
            positive_values=positive,
            percentages=percentages,
            parts=parts,
            targets=target_values,
            offsets=offsets,
            group_names=self.group_names,
        )

    def compare(self, reference: Any) -> Comparison:
        """Compare signed values with external reference contributions."""

        if isinstance(reference, Explanation):
            reference_values = reference.values
        else:
            reference_values = np.asarray(reference)
            if reference_values.ndim == self.values.ndim - 1:
                reference_values = np.expand_dims(reference_values, axis=-2)
        reference_values = np.broadcast_to(reference_values, self.values.shape)
        errors = self.values - reference_values
        estimated_flat = self.values.reshape(-1)
        reference_flat = reference_values.reshape(-1)
        if np.std(estimated_flat) == 0 or np.std(reference_flat) == 0:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(estimated_flat, reference_flat)[0, 1])
        group_axes = tuple(range(errors.ndim - 1))
        return Comparison(
            mae=float(np.mean(np.abs(errors))),
            rmse=float(np.sqrt(np.mean(errors**2))),
            bias=float(np.mean(errors)),
            max_absolute_error=float(np.max(np.abs(errors))),
            correlation=correlation,
            per_group_mae=np.mean(np.abs(errors), axis=group_axes),
            group_names=self.group_names,
        )
