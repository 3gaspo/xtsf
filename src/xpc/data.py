"""Data layout abstractions used by the estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np


def _dataframe_columns(data: Any) -> tuple[str, ...] | None:
    columns = getattr(data, "columns", None)
    if columns is None:
        return None
    return tuple(str(column) for column in columns)


@dataclass
class DataSpec:
    """Specification for independent tabular rows with shape ``(n, f)``."""

    feature_names: Sequence[str] | None = None
    _resolved_feature_names: tuple[str, ...] = field(init=False, default=())

    def prepare(self, data: Any) -> np.ndarray:
        array = np.asarray(data)
        if array.ndim != 2:
            raise ValueError(f"DataSpec expects shape (n, f), received {array.shape}.")
        configured_names = (
            tuple(self.feature_names) if self.feature_names is not None else None
        )
        names = configured_names or _dataframe_columns(data) or ()
        if names and len(names) != array.shape[-1]:
            raise ValueError("feature_names must match the final feature dimension.")
        if not names:
            names = tuple(f"x{i}" for i in range(array.shape[-1]))
        self._resolved_feature_names = names
        return array

    @property
    def resolved_feature_names(self) -> tuple[str, ...]:
        if not self._resolved_feature_names:
            raise RuntimeError("prepare() must be called before reading feature names.")
        return self._resolved_feature_names

    def batch_shape(self, data: np.ndarray) -> tuple[int, ...]:
        return (data.shape[0],)

    def iter_units(self, data: np.ndarray) -> Iterator[tuple[int, ...]]:
        return np.ndindex(self.batch_shape(data))

    def unit_features(self, data: np.ndarray, unit: tuple[int, ...]) -> np.ndarray:
        return np.asarray(data[unit[0]])

    def reference_matrix(self, data: np.ndarray) -> np.ndarray:
        return data.reshape(-1, data.shape[-1])

    def make_model_batch(
        self,
        data: np.ndarray,
        unit: tuple[int, ...],
        masked_rows: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(masked_rows)

    def extract_unit_predictions(
        self,
        predictions: Any,
        unit: tuple[int, ...],
        n_variants: int,
    ) -> np.ndarray:
        output = np.asarray(predictions)
        if output.ndim == 0:
            output = np.repeat(output.reshape(1), n_variants)
        if output.shape[0] != n_variants:
            raise ValueError(
                "The model must return one prediction row per input row; "
                f"received {output.shape} for {n_variants} rows."
            )
        if output.ndim == 1:
            return output[:, None]
        return output.reshape(n_variants, -1)

    def normalize_full_predictions(self, predictions: Any, data: np.ndarray) -> np.ndarray:
        output = np.asarray(predictions)
        n = data.shape[0]
        if output.ndim == 1 and output.shape[0] == n:
            return output[:, None]
        if output.ndim >= 2 and output.shape[0] == n:
            return output.reshape(n, -1)
        raise ValueError(
            f"Tabular model output must start with shape ({n},), received {output.shape}."
        )


@dataclass
class TimeSeriesTensorSpec(DataSpec):
    """Specification for temporal tensors shaped ``(n, d, f)``.

    Each ``(n, d)`` location is explained. Perturbed model batches retain the
    complete sequence, allowing sequence models to use temporal context.
    """

    def prepare(self, data: Any) -> np.ndarray:
        array = np.asarray(data)
        if array.ndim != 3:
            raise ValueError(
                f"TimeSeriesTensorSpec expects shape (n, d, f), received {array.shape}."
            )
        names = tuple(self.feature_names) if self.feature_names is not None else ()
        if names and len(names) != array.shape[-1]:
            raise ValueError("feature_names must match the final feature dimension.")
        if not names:
            names = tuple(f"x{i}" for i in range(array.shape[-1]))
        self._resolved_feature_names = names
        return array

    def batch_shape(self, data: np.ndarray) -> tuple[int, ...]:
        return data.shape[:2]

    def unit_features(self, data: np.ndarray, unit: tuple[int, ...]) -> np.ndarray:
        return np.asarray(data[unit[0], unit[1]])

    def make_model_batch(
        self,
        data: np.ndarray,
        unit: tuple[int, ...],
        masked_rows: np.ndarray,
    ) -> np.ndarray:
        sample, step = unit
        batch = np.repeat(data[sample : sample + 1], len(masked_rows), axis=0)
        batch[:, step, :] = masked_rows
        return batch

    def extract_unit_predictions(
        self,
        predictions: Any,
        unit: tuple[int, ...],
        n_variants: int,
    ) -> np.ndarray:
        output = np.asarray(predictions)
        step = unit[1]
        if output.shape[0] != n_variants:
            raise ValueError(
                "The temporal model must return one prediction sequence per input "
                f"sequence; received {output.shape} for {n_variants} sequences."
            )
        if output.ndim == 2:
            if step >= output.shape[1]:
                raise ValueError("Temporal model output is missing the explained step.")
            return output[:, step, None]
        if output.ndim >= 3:
            if step >= output.shape[1]:
                raise ValueError("Temporal model output is missing the explained step.")
            return output[:, step, ...].reshape(n_variants, -1)
        raise ValueError(
            "Temporal model output must have shape (n, d) or (n, d, H...)."
        )

    def normalize_full_predictions(self, predictions: Any, data: np.ndarray) -> np.ndarray:
        output = np.asarray(predictions)
        n, d = data.shape[:2]
        if output.ndim == 2 and output.shape == (n, d):
            return output[..., None]
        if output.ndim >= 3 and output.shape[:2] == (n, d):
            return output.reshape(n, d, -1)
        raise ValueError(
            "Temporal model output must start with the input batch shape "
            f"({n}, {d}); received {output.shape}."
        )
