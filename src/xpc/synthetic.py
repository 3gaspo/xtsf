"""Deterministic synthetic time series and supervised forecasting windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SyntheticForecastingData:
    """A synthetic series, its components, and context/target windows."""

    time: np.ndarray
    values: np.ndarray
    components: Mapping[str, np.ndarray]
    contexts: np.ndarray
    targets: np.ndarray
    context_length: int
    horizon: int


def make_synthetic_forecasting_data(
    *,
    n_steps: int = 24 * 60,
    context_length: int = 48,
    horizon: int = 12,
    stride: int = 1,
    daily_period: int = 24,
    weekly_period: int = 24 * 7,
    noise_scale: float = 0.2,
    seed: int = 0,
) -> SyntheticForecastingData:
    """Build a univariate trend/seasonality series and chronological windows."""

    if context_length < 1 or horizon < 1 or stride < 1:
        raise ValueError("context_length, horizon, and stride must be positive.")
    if n_steps < context_length + horizon:
        raise ValueError("n_steps must cover at least one context and horizon.")
    if daily_period < 2 or weekly_period < 2:
        raise ValueError("Seasonal periods must be at least two.")

    rng = np.random.default_rng(seed)
    time = np.arange(n_steps, dtype=float)
    level = np.full(n_steps, 10.0)
    trend = 0.004 * time
    daily = 2.0 * np.sin(2.0 * np.pi * time / daily_period)
    daily += 0.6 * np.cos(4.0 * np.pi * time / daily_period)
    weekly = 1.1 * np.sin(2.0 * np.pi * time / weekly_period - 0.7)
    interaction = 0.18 * daily * weekly
    noise = rng.normal(0.0, noise_scale, size=n_steps)
    components = {
        "level": level,
        "trend": trend,
        "daily": daily,
        "weekly": weekly,
        "interaction": interaction,
        "noise": noise,
    }
    values = np.sum(np.stack(tuple(components.values())), axis=0)
    total_length = context_length + horizon
    windows = np.lib.stride_tricks.sliding_window_view(values, total_length)[::stride]
    contexts = windows[:, :context_length, None].copy()
    targets = windows[:, context_length:, None].copy()
    return SyntheticForecastingData(
        time=time,
        values=values,
        components=components,
        contexts=contexts,
        targets=targets,
        context_length=context_length,
        horizon=horizon,
    )
