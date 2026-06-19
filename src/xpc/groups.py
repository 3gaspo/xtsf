"""Feature group definitions for Shapley players."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


Selector = int | str


def _as_tuple(value: Selector | Iterable[Selector]) -> tuple[Selector, ...]:
    if isinstance(value, (int, str)):
        return (value,)
    return tuple(value)


@dataclass(frozen=True)
class ResolvedFeatureGroups:
    names: tuple[str, ...]
    groups: tuple[tuple[int, ...], ...]
    always_present: tuple[int, ...]
    feature_names: tuple[str, ...]

    @property
    def n_players(self) -> int:
        return len(self.groups)


class FeatureGroups:
    """Define disjoint feature groups that act as Shapley players.

    ``remaining`` may be ``"individual"``, ``"group"``, or ``"ignore"``.
    Features marked ``always_present`` are excluded from players and included
    in every coalition.
    """

    def __init__(
        self,
        groups: Mapping[str, Iterable[Selector]] | None = None,
        *,
        remaining: str = "individual",
        remaining_name: str = "remaining",
        always_present: Iterable[Selector] = (),
    ) -> None:
        if remaining not in {"individual", "group", "ignore"}:
            raise ValueError("remaining must be 'individual', 'group', or 'ignore'.")
        self.group_definitions = {
            str(name): _as_tuple(features) for name, features in (groups or {}).items()
        }
        self.remaining = remaining
        self.remaining_name = remaining_name
        self.always_present_definitions = _as_tuple(always_present)

    @classmethod
    def individual(
        cls, *, always_present: Iterable[Selector] = ()
    ) -> "FeatureGroups":
        return cls(always_present=always_present)

    @classmethod
    def one_group(
        cls,
        features: Iterable[Selector],
        *,
        name: str = "group",
        remaining: str = "individual",
        always_present: Iterable[Selector] = (),
    ) -> "FeatureGroups":
        return cls(
            {name: features},
            remaining=remaining,
            always_present=always_present,
        )

    def resolve(
        self, n_features: int, feature_names: Sequence[str] | None = None
    ) -> ResolvedFeatureGroups:
        names = tuple(feature_names or (f"x{i}" for i in range(n_features)))
        if len(names) != n_features:
            raise ValueError("feature_names must match n_features.")
        lookup = {name: index for index, name in enumerate(names)}

        def resolve_selector(selector: Selector) -> int:
            if isinstance(selector, str):
                if selector not in lookup:
                    raise KeyError(f"Unknown feature name: {selector!r}.")
                return lookup[selector]
            index = int(selector)
            if not 0 <= index < n_features:
                raise IndexError(f"Feature index {index} is out of bounds.")
            return index

        always = tuple(
            dict.fromkeys(resolve_selector(item) for item in self.always_present_definitions)
        )
        occupied = set(always)
        resolved_names: list[str] = []
        resolved_groups: list[tuple[int, ...]] = []

        for group_name, selectors in self.group_definitions.items():
            group = tuple(dict.fromkeys(resolve_selector(item) for item in selectors))
            if not group:
                raise ValueError(f"Feature group {group_name!r} is empty.")
            overlap = occupied.intersection(group)
            if overlap:
                raise ValueError(
                    f"Feature group {group_name!r} overlaps another group or "
                    f"always-present features at indexes {sorted(overlap)}."
                )
            occupied.update(group)
            resolved_names.append(group_name)
            resolved_groups.append(group)

        remaining = tuple(index for index in range(n_features) if index not in occupied)
        if self.remaining == "individual":
            for index in remaining:
                resolved_names.append(names[index])
                resolved_groups.append((index,))
        elif self.remaining == "group" and remaining:
            resolved_names.append(self.remaining_name)
            resolved_groups.append(remaining)

        if not resolved_groups:
            raise ValueError("At least one Shapley player is required.")
        return ResolvedFeatureGroups(
            names=tuple(resolved_names),
            groups=tuple(resolved_groups),
            always_present=always,
            feature_names=names,
        )
