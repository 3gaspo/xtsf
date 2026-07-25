import unittest

import numpy as np

from xpc import Explanation


class ExplanationTests(unittest.TestCase):
    def test_from_contributions_preserves_signed_values(self):
        contributions = np.asarray([[3.0, -1.0], [2.0, 4.0]])
        explanation = Explanation.from_contributions(
            contributions,
            base_values=10.0,
            group_names=["a", "b"],
        )
        self.assertEqual(explanation.values.shape, (2, 1, 2))
        np.testing.assert_allclose(explanation.values[:, 0], contributions)
        np.testing.assert_allclose(explanation.predictions[:, 0], [12.0, 16.0])
        self.assertIsNotNone(explanation.heightened)

    def test_from_contributions_can_disable_heightening(self):
        explanation = Explanation.from_contributions(
            [[3.0, -1.0]],
            base_values=10.0,
            heighten=False,
        )
        self.assertIsNone(explanation.heightened)

    def test_heightening_produces_positive_parts_that_sum_to_targets(self):
        contributions = np.asarray(
            [[[-2.0, 1.0, 3.0]], [[1.0, -4.0, 2.0]]]
        )
        explanation = Explanation.from_contributions(
            contributions,
            targets=np.asarray([[20.0], [30.0]]),
            group_names=["a", "b", "c"],
        )
        heightened = explanation.heighten()

        np.testing.assert_allclose(heightened.raw_values, contributions)
        self.assertTrue(np.all(heightened.positive_values >= 0))
        np.testing.assert_allclose(np.sum(heightened.percentages, axis=-1), 1.0)
        np.testing.assert_allclose(
            np.sum(heightened.parts, axis=-1), [[20.0], [30.0]]
        )

    def test_heightening_handles_all_zero_values(self):
        explanation = Explanation.from_contributions(
            np.zeros((1, 3)), targets=[9.0]
        )
        heightened = explanation.heighten()
        np.testing.assert_allclose(heightened.percentages, 1.0 / 3.0)
        np.testing.assert_allclose(heightened.parts, 3.0)

    def test_single_output_targets_accept_batch_vectors(self):
        explanation = Explanation.from_contributions(
            np.zeros((2, 3)), targets=np.asarray([9.0, 12.0])
        )
        np.testing.assert_allclose(explanation.predictions[:, 0], [9.0, 12.0])
        np.testing.assert_allclose(
            explanation.heighten(targets=[6.0, 15.0]).parts.sum(axis=-1)[:, 0],
            [6.0, 15.0],
        )

    def test_comparison_to_external_reference(self):
        explanation = Explanation.from_contributions(
            [[1.0, 3.0], [2.0, 4.0]], group_names=["a", "b"]
        )
        comparison = explanation.compare([[0.0, 3.0], [4.0, 4.0]])
        self.assertAlmostEqual(comparison.mae, 0.75)
        np.testing.assert_allclose(comparison.per_group_mae, [1.5, 0.0])
        self.assertEqual(comparison.group_names, ("a", "b"))


if __name__ == "__main__":
    unittest.main()
