"""Masking strategies for Monte Carlo coalition values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .conditioners import Conditioner, EmpiricalConditioner
from .data import DataSpec


class Masker(ABC):
    """Create masked feature rows for a coalition."""

    def prepare(self, data: np.ndarray, spec: DataSpec) -> None:
        self.data = data
        self.spec = spec

    def mask(
        self,
        x: np.ndarray,
        present: Sequence[int],
        n_samples: int,
        rng: np.random.Generator,
        *,
        unit: tuple[int, ...],
    ) -> np.ndarray:
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        replacements = np.asarray(
            self.replacements(x, present, n_samples, rng, unit=unit)
        )
        expected = (n_samples, x.shape[0])
        if replacements.shape != expected:
            raise ValueError(
                f"Masker replacements must have shape {expected}, got {replacements.shape}."
            )
        masked = replacements.copy()
        if present:
            indexes = np.asarray(tuple(present), dtype=int)
            masked[:, indexes] = x[indexes]
        return masked

    @abstractmethod
    def replacements(
        self,
        x: np.ndarray,
        present: Sequence[int],
        n_samples: int,
        rng: np.random.Generator,
        *,
        unit: tuple[int, ...],
    ) -> np.ndarray:
        """Return complete replacement rows before present features are restored."""


class BaselineMasker(Masker):
    """Mask with zero/mean/median/scalar/vector/tensor/callable baselines."""

    def __init__(self, baseline: Any = "zero", *, background: Any | None = None) -> None:
        self.baseline = baseline
        self.background = background

    def prepare(self, data: np.ndarray, spec: DataSpec) -> None:
        super().prepare(data, spec)
        source = data if self.background is None else spec.prepare(self.background)
        self.reference = spec.reference_matrix(source)

    def _resolve(self, x: np.ndarray, unit: tuple[int, ...]) -> np.ndarray:
        baseline = self.baseline
        if isinstance(baseline, str):
            if baseline == "zero":
                return np.zeros_like(x, dtype=np.result_type(x, float))
            if baseline == "mean":
                return np.mean(self.reference, axis=0)
            if baseline == "median":
                return np.median(self.reference, axis=0)
            raise ValueError("String baseline must be 'zero', 'mean', or 'median'.")
        if callable(baseline):
            try:
                value = baseline(data=self.data, unit=unit, x=x)
            except TypeError:
                value = baseline(x)
            return np.broadcast_to(np.asarray(value), x.shape).copy()
        value = np.asarray(baseline)
        if value.ndim == 0 or value.shape == x.shape:
            dtype = np.result_type(value, x, float)
            return np.broadcast_to(value, x.shape).astype(dtype, copy=True)
        if value.shape == self.data.shape:
            return np.asarray(value[unit])
        try:
            broadcast = np.broadcast_to(value, self.data.shape)
            return np.asarray(broadcast[unit])
        except ValueError as error:
            raise ValueError(
                "Baseline must be scalar, feature vector, data-shaped tensor, "
                "broadcastable tensor, callable, or zero/mean/median."
            ) from error

    def replacements(
        self,
        x: np.ndarray,
        present: Sequence[int],
        n_samples: int,
        rng: np.random.Generator,
        *,
        unit: tuple[int, ...],
    ) -> np.ndarray:
        return np.repeat(self._resolve(x, unit)[None, :], n_samples, axis=0)


class RandomMasker(Masker):
    """Unconditional empirical masking from random reference rows."""

    def __init__(self, background: Any | None = None) -> None:
        self.background = background

    def prepare(self, data: np.ndarray, spec: DataSpec) -> None:
        super().prepare(data, spec)
        source = data if self.background is None else spec.prepare(self.background)
        self.reference = spec.reference_matrix(source)
        if not len(self.reference):
            raise ValueError("Random masking requires non-empty reference data.")

    def replacements(
        self,
        x: np.ndarray,
        present: Sequence[int],
        n_samples: int,
        rng: np.random.Generator,
        *,
        unit: tuple[int, ...],
    ) -> np.ndarray:
        indexes = rng.integers(0, len(self.reference), size=n_samples)
        return self.reference[indexes].copy()


class ConditionalMasker(Masker):
    """Empirical, grid, or user-defined non-generative conditional masking."""

    def __init__(
        self,
        background: Any,
        conditioner: Conditioner | None = None,
    ) -> None:
        self.background = background
        self.conditioner = conditioner or EmpiricalConditioner()

    def prepare(self, data: np.ndarray, spec: DataSpec) -> None:
        super().prepare(data, spec)
        self.reference = spec.reference_matrix(spec.prepare(self.background))

    def replacements(
        self,
        x: np.ndarray,
        present: Sequence[int],
        n_samples: int,
        rng: np.random.Generator,
        *,
        unit: tuple[int, ...],
    ) -> np.ndarray:
        return self.conditioner.sample(
            self.reference, x, tuple(present), n_samples, rng
        )
