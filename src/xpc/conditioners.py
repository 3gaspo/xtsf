"""Non-generative conditional samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class Conditioner(ABC):
    """Select replacement rows from an observed reference matrix."""

    @abstractmethod
    def sample(
        self,
        reference: np.ndarray,
        x: np.ndarray,
        observed: Sequence[int],
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return ``n_samples`` empirical replacement rows."""


class EmpiricalConditioner(Conditioner):
    """Sample from nearest empirical neighbors in observed-feature space."""

    def __init__(self, n_neighbors: int = 32) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        self.n_neighbors = n_neighbors

    def sample(
        self,
        reference: np.ndarray,
        x: np.ndarray,
        observed: Sequence[int],
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if len(reference) == 0:
            raise ValueError("Conditional masking requires non-empty reference data.")
        if not observed:
            indexes = rng.integers(0, len(reference), size=n_samples)
            return reference[indexes].copy()
        columns = np.asarray(tuple(observed), dtype=int)
        scale = np.std(reference[:, columns], axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        distances = np.sum(((reference[:, columns] - x[columns]) / scale) ** 2, axis=1)
        count = min(self.n_neighbors, len(reference))
        neighbors = np.argpartition(distances, count - 1)[:count]
        indexes = rng.choice(neighbors, size=n_samples, replace=True)
        return reference[indexes].copy()


class GridConditioner(Conditioner):
    """Condition on matching empirical grid cells, with nearest-neighbor fallback."""

    def __init__(
        self,
        n_bins: int = 10,
        *,
        categorical: Sequence[int] = (),
        strategy: str = "quantile",
        fallback: Conditioner | None = None,
    ) -> None:
        if n_bins < 1:
            raise ValueError("n_bins must be positive.")
        if strategy not in {"quantile", "uniform"}:
            raise ValueError("strategy must be 'quantile' or 'uniform'.")
        self.n_bins = n_bins
        self.categorical = frozenset(int(index) for index in categorical)
        self.strategy = strategy
        self.fallback = fallback or EmpiricalConditioner()

    def _cell(self, values: np.ndarray, reference_column: np.ndarray, index: int) -> np.ndarray:
        if index in self.categorical:
            return values
        if self.strategy == "quantile":
            edges = np.quantile(
                reference_column, np.linspace(0.0, 1.0, self.n_bins + 1)
            )
        else:
            edges = np.linspace(
                np.min(reference_column), np.max(reference_column), self.n_bins + 1
            )
        edges = np.unique(edges)
        if len(edges) <= 1:
            return np.zeros_like(values, dtype=int)
        return np.digitize(values, edges[1:-1], right=False)

    def sample(
        self,
        reference: np.ndarray,
        x: np.ndarray,
        observed: Sequence[int],
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not observed:
            indexes = rng.integers(0, len(reference), size=n_samples)
            return reference[indexes].copy()
        mask = np.ones(len(reference), dtype=bool)
        for index in observed:
            ref_cells = self._cell(reference[:, index], reference[:, index], index)
            x_cell = self._cell(np.asarray([x[index]]), reference[:, index], index)[0]
            mask &= ref_cells == x_cell
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            return self.fallback.sample(reference, x, observed, n_samples, rng)
        indexes = rng.choice(candidates, size=n_samples, replace=True)
        return reference[indexes].copy()


class CallableConditioner(Conditioner):
    """Wrap a user-defined non-generative conditioning function."""

    def __init__(self, function: Callable[..., Any]) -> None:
        self.function = function

    def sample(
        self,
        reference: np.ndarray,
        x: np.ndarray,
        observed: Sequence[int],
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        rows = np.asarray(
            self.function(
                reference=reference,
                x=x,
                observed=tuple(observed),
                n_samples=n_samples,
                rng=rng,
            )
        )
        if rows.shape != (n_samples, reference.shape[1]):
            raise ValueError(
                "A callable conditioner must return shape "
                f"({n_samples}, {reference.shape[1]}), received {rows.shape}."
            )
        return rows
