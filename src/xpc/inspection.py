"""Framework-light parameter counts and named parameter structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterInfo:
    """Metadata for one named parameter tensor."""

    name: str
    shape: tuple[int, ...]
    count: int
    trainable: bool
    dtype: str


@dataclass(frozen=True)
class ParameterCounts:
    """Element and tensor counts split by trainability."""

    total: int
    trainable: int
    frozen: int
    tensors: int
    trainable_tensors: int


def parameter_structure(model: Any) -> tuple[ParameterInfo, ...]:
    """Return named parameter shapes without materializing parameter values."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("The model must expose a callable named_parameters method.")
    structure = []
    for name, parameter in named_parameters():
        numel = getattr(parameter, "numel", None)
        if not callable(numel):
            raise TypeError(f"Parameter {name!r} does not expose numel().")
        structure.append(
            ParameterInfo(
                name=str(name),
                shape=tuple(int(value) for value in parameter.shape),
                count=int(numel()),
                trainable=bool(getattr(parameter, "requires_grad", False)),
                dtype=str(getattr(parameter, "dtype", "unknown")),
            )
        )
    return tuple(structure)


def parameter_counts(model: Any) -> ParameterCounts:
    """Count all, trainable, and frozen elements in a named-parameter model."""

    structure = parameter_structure(model)
    trainable = sum(item.count for item in structure if item.trainable)
    total = sum(item.count for item in structure)
    return ParameterCounts(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        tensors=len(structure),
        trainable_tensors=sum(item.trainable for item in structure),
    )
