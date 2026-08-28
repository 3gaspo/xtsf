"""Prediction-error summaries independent of any plotting backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ErrorSummary:
    """Scalar error diagnostics over equally shaped targets and predictions."""

    mae: float
    rmse: float
    bias: float
    max_absolute_error: float
    correlation: float
    count: int


def error_summary(targets: Any, predictions: Any) -> ErrorSummary:
    """Summarize prediction errors after flattening all supplied dimensions."""

    truth = np.asarray(targets, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    if truth.shape != predicted.shape:
        raise ValueError(
            f"targets and predictions must have the same shape, got "
            f"{truth.shape} and {predicted.shape}."
        )
    truth_flat = truth.reshape(-1)
    predicted_flat = predicted.reshape(-1)
    if not len(truth_flat):
        raise ValueError("At least one target is required.")
    residuals = predicted_flat - truth_flat
    correlation = (
        float("nan")
        if np.std(truth_flat) == 0 or np.std(predicted_flat) == 0
        else float(np.corrcoef(truth_flat, predicted_flat)[0, 1])
    )
    return ErrorSummary(
        mae=float(np.mean(np.abs(residuals))),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        bias=float(np.mean(residuals)),
        max_absolute_error=float(np.max(np.abs(residuals))),
        correlation=correlation,
        count=len(truth_flat),
    )
