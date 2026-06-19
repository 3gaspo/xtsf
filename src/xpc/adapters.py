"""Model adapters with a common NumPy prediction boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any

import numpy as np


class ModelAdapter(ABC):
    """Convert model-specific predictions to NumPy arrays."""

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict on a NumPy model batch."""


class NumpyModelAdapter(ModelAdapter):
    """Adapter for NumPy callables and fitted objects exposing ``predict``."""

    def __init__(self, model: Any) -> None:
        if callable(model):
            self.function = model
        elif callable(getattr(model, "predict", None)):
            self.function = model.predict
        else:
            raise TypeError("Model must be callable or expose a callable predict method.")

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(self.function(data))


CallableModelAdapter = NumpyModelAdapter
SklearnModelAdapter = NumpyModelAdapter


class PandasModelAdapter(ModelAdapter):
    """Adapter for models that require pandas DataFrames."""

    def __init__(self, model: Any, feature_names: Sequence[str]) -> None:
        if callable(model):
            self.function = model
        elif callable(getattr(model, "predict", None)):
            self.function = model.predict
        else:
            raise TypeError("Model must be callable or expose a callable predict method.")
        self.feature_names = tuple(feature_names)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 2:
            raise ValueError("PandasModelAdapter supports tabular 2D model inputs.")
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("Install xpc[pandas] to use PandasModelAdapter.") from error
        frame = pd.DataFrame(data, columns=self.feature_names)
        return np.asarray(self.function(frame))


class TorchModelAdapter(ModelAdapter):
    """Adapter for torch modules or tensor callables, imported lazily."""

    def __init__(
        self,
        model: Any,
        *,
        device: str | None = None,
        dtype: str = "float32",
        output_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.output_transform = output_transform

    def predict(self, data: np.ndarray) -> np.ndarray:
        try:
            import torch
        except ImportError as error:
            raise ImportError("Install xpc[torch] to use TorchModelAdapter.") from error
        dtype = getattr(torch, self.dtype)
        tensor = torch.as_tensor(data, dtype=dtype, device=self.device)
        training = getattr(self.model, "training", None)
        if callable(getattr(self.model, "eval", None)):
            self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)
        if self.output_transform is not None:
            output = self.output_transform(output)
        if training and callable(getattr(self.model, "train", None)):
            self.model.train()
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        return np.asarray(output)


class ScriptModelAdapter(ModelAdapter):
    """Run an external script using temporary CSV input and output files.

    Command tokens may contain ``{input}`` and ``{output}`` placeholders.
    The script must write a headered output CSV.
    """

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        feature_names: Sequence[str] | None = None,
        output_columns: Sequence[str] | None = None,
        timeout: float | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.feature_names = tuple(feature_names) if feature_names else None
        self.output_columns = tuple(output_columns) if output_columns else None
        self.timeout = timeout
        self.environment = environment

    def predict(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 2:
            raise ValueError("CSV script adapters support tabular 2D model inputs.")
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("Install xpc[pandas] to use ScriptModelAdapter.") from error
        names = self.feature_names or tuple(f"x{i}" for i in range(data.shape[1]))
        with tempfile.TemporaryDirectory(prefix="xpc-") as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.csv"
            pd.DataFrame(data, columns=names).to_csv(input_path, index=False)
            command = [
                token.format(input=str(input_path), output=str(output_path))
                for token in self.command
            ]
            subprocess.run(
                command,
                check=True,
                timeout=self.timeout,
                env=self.environment,
                capture_output=True,
                text=True,
            )
            if not output_path.exists():
                raise RuntimeError("The model script did not create its output CSV.")
            output = pd.read_csv(output_path)
            if self.output_columns:
                output = output.loc[:, self.output_columns]
            return output.to_numpy()


class RScriptModelAdapter(ScriptModelAdapter):
    """Convenience adapter for an ``Rscript`` prediction program."""

    def __init__(
        self,
        script: str | Path,
        *,
        rscript: str = "Rscript",
        arguments: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        command = [
            rscript,
            str(script),
            "{input}",
            "{output}",
            *arguments,
        ]
        super().__init__(command, **kwargs)


@dataclass(order=True)
class _Registration:
    priority: int
    model_type: type
    factory: Callable[[Any], ModelAdapter]


_REGISTRY: list[_Registration] = []


def register_model_adapter(
    model_type: type,
    factory: Callable[[Any], ModelAdapter] | type[ModelAdapter],
    *,
    priority: int = 0,
) -> None:
    """Register an adapter factory for custom model classes."""

    _REGISTRY.append(_Registration(priority, model_type, factory))
    _REGISTRY.sort(reverse=True)


def adapt_model(model: Any) -> ModelAdapter:
    """Return an explicit adapter or infer one from the registry/model surface."""

    if isinstance(model, ModelAdapter):
        return model
    for registration in _REGISTRY:
        if isinstance(model, registration.model_type):
            adapter = registration.factory(model)
            if not isinstance(adapter, ModelAdapter):
                raise TypeError("Registered factories must return ModelAdapter instances.")
            return adapter
    return NumpyModelAdapter(model)
