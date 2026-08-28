import importlib.util
import unittest

import numpy as np

from xpc import (
    Explanation,
    accumulated_local_effects,
    error_summary,
    partial_dependence,
)


class EffectAndDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.data = np.column_stack(
            [np.linspace(-4.0, 4.0, 16), np.linspace(1.0, 2.0, 16)]
        )
        self.model = lambda x: np.column_stack(
            [2.0 * x[:, 0] + 3.0 * x[:, 1], -x[:, 0]]
        )

    def test_partial_dependence_preserves_every_output(self):
        result = partial_dependence(
            self.model,
            self.data,
            "signal",
            feature_names=["signal", "context"],
            grid=[-2.0, 0.0, 2.0],
        )
        expected = np.column_stack(
            [2.0 * result.grid + 3.0 * self.data[:, 1].mean(), -result.grid]
        )
        self.assertEqual(result.feature_name, "signal")
        np.testing.assert_allclose(result.values, expected)

    def test_linear_ale_is_centered_and_has_the_expected_slope(self):
        result = accumulated_local_effects(
            self.model,
            self.data,
            0,
            feature_names=["signal", "context"],
            n_bins=4,
        )
        centered_grid = result.grid - np.average(
            result.grid, weights=result.bin_counts
        )
        np.testing.assert_allclose(result.values[:, 0], 2.0 * centered_grid)
        np.testing.assert_allclose(result.values[:, 1], -centered_grid)
        np.testing.assert_allclose(
            np.average(result.values, axis=0, weights=result.bin_counts), 0.0
        )

    def test_error_summary(self):
        summary = error_summary([1.0, 2.0, 3.0], [2.0, 2.0, 1.0])
        self.assertAlmostEqual(summary.mae, 1.0)
        self.assertAlmostEqual(summary.rmse, np.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(summary.bias, -1.0 / 3.0)
        self.assertEqual(summary.max_absolute_error, 2.0)
        self.assertEqual(summary.count, 3)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "matplotlib is not installed"
    )
    def test_plot_helpers_return_matplotlib_objects(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from xpc import (
            plot_accumulated_local_effects,
            plot_partial_dependence,
            plot_prediction_errors,
            plot_shapley_waterfall,
        )

        pdp = partial_dependence(self.model, self.data, 0, n_points=4)
        ale = accumulated_local_effects(self.model, self.data, 0, n_bins=4)
        figures = [
            plot_partial_dependence(pdp, output=0)[0],
            plot_accumulated_local_effects(ale, output=0)[0],
            plot_prediction_errors(
                np.ones((4, 2)), np.zeros((4, 2)), output=1
            )[0],
        ]
        explanation = Explanation(
            values=np.asarray([[[1.0, -0.5, 2.0]]]),
            base_values=np.asarray([[3.0]]),
            predictions=np.asarray([[5.5]]),
            group_names=("first", "second", "third"),
        )
        figure, axes = plot_shapley_waterfall(explanation)
        figures.append(figure)
        self.assertEqual(len(axes.patches), 3)
        for item in figures:
            plt.close(item)


if __name__ == "__main__":
    unittest.main()
