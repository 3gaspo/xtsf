import unittest

import numpy as np

from xpc import (
    BaselineMasker,
    FeatureGroups,
    RandomMasker,
    ShapleyExplainer,
    TimeSeriesTensorSpec,
)


class ShapleyExplainerTests(unittest.TestCase):
    def test_linear_tabular_contributions_are_exact_with_baseline(self):
        weights = np.asarray([1.0, -2.0, 3.0])
        data = np.asarray([[2.0, 1.0, 1.0], [1.0, -1.0, 3.0]])
        explanation = ShapleyExplainer(
            lambda x: x @ weights,
            BaselineMasker("zero"),
            n_coalitions=12,
            random_state=7,
        )(data)

        np.testing.assert_allclose(explanation.values[:, 0, :], data * weights)
        np.testing.assert_allclose(explanation.base_values, 0.0)
        np.testing.assert_allclose(explanation.efficiency_residual, 0.0)
        self.assertIsNotNone(explanation.heightened)
        np.testing.assert_allclose(
            explanation.heightened.parts.sum(axis=-1), explanation.predictions
        )

    def test_heightening_can_be_disabled(self):
        data = np.asarray([[2.0, 3.0]])
        explanation = ShapleyExplainer(
            lambda x: x[:, 0] + x[:, 1],
            BaselineMasker(0),
            n_coalitions=2,
            random_state=0,
            heighten=False,
        )(data)

        self.assertIsNone(explanation.heightened)
        np.testing.assert_allclose(
            explanation.values.sum(axis=-1), explanation.predictions
        )
        np.testing.assert_allclose(explanation.efficiency_residual, 0.0)

    def test_group_players_and_always_present_features(self):
        data = np.asarray([[2.0, 3.0, 5.0, 7.0]])
        weights = np.asarray([1.0, 2.0, 4.0, 8.0])
        groups = FeatureGroups(
            {"first_pair": [0, 1]},
            remaining="group",
            remaining_name="third",
            always_present=[3],
        )
        explanation = ShapleyExplainer(
            lambda x: x @ weights,
            BaselineMasker(0),
            feature_groups=groups,
            n_coalitions=8,
            random_state=0,
        )(data)

        self.assertEqual(explanation.group_names, ("first_pair", "third"))
        np.testing.assert_allclose(explanation.values[0, 0], [8.0, 20.0])
        np.testing.assert_allclose(explanation.base_values[0, 0], 56.0)

    def test_random_masking_estimates_background_centered_contributions(self):
        background = np.asarray(
            [[-1.0, -2.0], [1.0, 2.0], [-1.0, 2.0], [1.0, -2.0]]
        )
        data = np.asarray([[2.0, 3.0]])
        explanation = ShapleyExplainer(
            lambda x: x[:, 0] + 2.0 * x[:, 1],
            RandomMasker(background),
            n_coalitions=20,
            n_mask_samples=200,
            random_state=10,
        )(data)
        np.testing.assert_allclose(explanation.values[0, 0], [2.0, 6.0], atol=0.2)

    def test_temporal_tensor_returns_n_d_h_g(self):
        data = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)

        def temporal_model(x):
            total = np.sum(x, axis=-1)
            return np.stack([total, 2.0 * total], axis=-1)

        explanation = ShapleyExplainer(
            temporal_model,
            BaselineMasker(0),
            data_spec=TimeSeriesTensorSpec(feature_names=["load", "weather"]),
            n_coalitions=5,
            random_state=2,
        )(data)

        self.assertEqual(explanation.values.shape, (2, 3, 2, 2))
        self.assertEqual(explanation.predictions.shape, (2, 3, 2))
        np.testing.assert_allclose(explanation.efficiency_residual, 0.0)
        np.testing.assert_allclose(explanation.values[..., 0, :], data)
        np.testing.assert_allclose(explanation.values[..., 1, :], 2.0 * data)

    def test_temporal_tensor_baseline_is_selected_per_location(self):
        data = np.ones((1, 2, 2)) * 5.0
        baseline = np.asarray([[[1.0, 2.0], [3.0, 4.0]]])
        explanation = ShapleyExplainer(
            lambda x: np.sum(x, axis=-1),
            BaselineMasker(baseline),
            data_spec=TimeSeriesTensorSpec(),
            n_coalitions=3,
            random_state=0,
        )(data)
        expected = data - baseline
        np.testing.assert_allclose(explanation.values[..., 0, :], expected)

    def test_irrelevant_features_have_zero_contribution(self):
        data = np.asarray([[2.0, 3.0, 99.0]])
        for heighten in (True, False):
            with self.subTest(heighten=heighten):
                explanation = ShapleyExplainer(
                    lambda x: x[:, 0] + 2.0 * x[:, 1],
                    BaselineMasker(0),
                    n_coalitions=5,
                    random_state=3,
                    heighten=heighten,
                )(data)

                np.testing.assert_allclose(explanation.values[0, 0], [2.0, 6.0, 0.0])
                np.testing.assert_allclose(explanation.efficiency_residual, 0.0)
                if heighten:
                    np.testing.assert_allclose(
                        explanation.heightened.parts[0, 0, -1], 0.0
                    )
                    np.testing.assert_allclose(
                        explanation.heightened.parts.sum(axis=-1),
                        explanation.predictions,
                    )

    def test_additive_group_model_contributions_equal_group_coefficients(self):
        data = np.asarray([[1.0, 0.0, np.pi / 2.0, 123.0]])
        coefficients = np.asarray([2.0, 5.0, 0.0])
        groups = FeatureGroups(
            {
                "quadratic": [0, 1],
                "seasonal": [2],
                "unused": [3],
            },
            remaining="ignore",
        )

        def grouped_gam(x):
            quadratic = x[:, 0] ** 2 + x[:, 1]
            seasonal = np.sin(x[:, 2])
            return coefficients[0] * quadratic + coefficients[1] * seasonal

        for n_coalitions in (1, 3, 9):
            for heighten in (True, False):
                with self.subTest(n_coalitions=n_coalitions, heighten=heighten):
                    explanation = ShapleyExplainer(
                        grouped_gam,
                        BaselineMasker(0),
                        feature_groups=groups,
                        n_coalitions=n_coalitions,
                        random_state=11,
                        heighten=heighten,
                    )(data)

                    np.testing.assert_allclose(explanation.values[0, 0], coefficients)
                    np.testing.assert_allclose(explanation.efficiency_residual, 0.0)
                    if heighten:
                        np.testing.assert_allclose(
                            explanation.heightened.parts.sum(axis=-1),
                            explanation.predictions,
                        )
                        np.testing.assert_allclose(
                            explanation.heightened.parts[0, 0], coefficients
                        )


if __name__ == "__main__":
    unittest.main()
