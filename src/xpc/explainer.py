"""Monte Carlo Shapley estimator."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import numpy as np

from .adapters import ModelAdapter, adapt_model
from .data import DataSpec
from .explanation import Explanation
from .groups import FeatureGroups, ResolvedFeatureGroups
from .maskers import Masker


class ShapleyExplainer:
    """Estimate signed group Shapley values by Monte Carlo subset sampling."""

    def __init__(
        self,
        model: Any,
        masker: Masker,
        *,
        data_spec: DataSpec | None = None,
        feature_groups: FeatureGroups | None = None,
        n_coalitions: int = 128,
        n_mask_samples: int = 1,
        random_state: int | np.random.Generator | None = None,
        heighten: bool = True,
    ) -> None:
        if n_coalitions < 1:
            raise ValueError("n_coalitions must be positive.")
        if n_mask_samples < 1:
            raise ValueError("n_mask_samples must be positive.")
        self.model: ModelAdapter = adapt_model(model)
        self.masker = masker
        self.data_spec = data_spec or DataSpec()
        self.feature_groups = feature_groups or FeatureGroups.individual()
        self.n_coalitions = n_coalitions
        self.n_mask_samples = n_mask_samples
        self.heighten = heighten
        self.rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

    def _features_for_players(
        self, groups: ResolvedFeatureGroups, players: Iterable[int]
    ) -> tuple[int, ...]:
        features = list(groups.always_present)
        for player in players:
            features.extend(groups.groups[player])
        return tuple(dict.fromkeys(features))

    def _coalition_value(
        self,
        data: np.ndarray,
        unit: tuple[int, ...],
        x: np.ndarray,
        present: tuple[int, ...],
    ) -> np.ndarray:
        masked_rows = self.masker.mask(
            x,
            present,
            self.n_mask_samples,
            self.rng,
            unit=unit,
        )
        model_batch = self.data_spec.make_model_batch(data, unit, masked_rows)
        predictions = self.model.predict(model_batch)
        unit_predictions = self.data_spec.extract_unit_predictions(
            predictions, unit, self.n_mask_samples
        )
        return np.mean(unit_predictions, axis=0)

    def explain(self, data: Any) -> Explanation:
        """Explain every unit and return values shaped ``batch + (H, G)``."""

        array = self.data_spec.prepare(data)
        groups = self.feature_groups.resolve(
            array.shape[-1], self.data_spec.resolved_feature_names
        )
        self.masker.prepare(array, self.data_spec)
        full_predictions = self.data_spec.normalize_full_predictions(
            self.model.predict(array), array
        )
        batch_shape = self.data_spec.batch_shape(array)
        n_outputs = full_predictions.shape[-1]
        values = np.zeros(batch_shape + (n_outputs, groups.n_players), dtype=float)
        base_values = np.zeros(batch_shape + (n_outputs,), dtype=float)

        for unit in self.data_spec.iter_units(array):
            x = self.data_spec.unit_features(array, unit)
            always = self._features_for_players(groups, ())
            base_values[unit] = self._coalition_value(array, unit, x, always)
            for player in range(groups.n_players):
                other_players = [p for p in range(groups.n_players) if p != player]
                deltas = np.zeros((self.n_coalitions, n_outputs), dtype=float)
                for draw in range(self.n_coalitions):
                    size = int(self.rng.integers(0, groups.n_players))
                    selected = (
                        self.rng.choice(other_players, size=size, replace=False).tolist()
                        if size
                        else []
                    )
                    present = self._features_for_players(groups, selected)
                    with_player = self._features_for_players(
                        groups, [*selected, player]
                    )
                    before = self._coalition_value(array, unit, x, present)
                    after = self._coalition_value(array, unit, x, with_player)
                    deltas[draw] = after - before
                values[unit + (slice(None), player)] = np.mean(deltas, axis=0)

        explanation = Explanation(
            values=values,
            base_values=base_values,
            predictions=full_predictions,
            group_names=groups.names,
            feature_groups=groups.groups,
            feature_names=groups.feature_names,
            metadata={
                "algorithm": "monte_carlo_subset",
                "n_coalitions": self.n_coalitions,
                "n_mask_samples": self.n_mask_samples,
                "always_present": groups.always_present,
                "heightening_enabled": self.heighten,
            },
        )
        if self.heighten:
            return replace(explanation, heightened=explanation.heighten())
        return explanation

    __call__ = explain
