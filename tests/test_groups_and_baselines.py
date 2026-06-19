import unittest

import numpy as np

from xpc import BaselineMasker, DataSpec, FeatureGroups


class FeatureGroupsTests(unittest.TestCase):
    def test_custom_groups_remaining_individual_and_always_present(self):
        groups = FeatureGroups(
            {"weather": ["temp", "wind"]},
            remaining="individual",
            always_present=["bias"],
        ).resolve(5, ["temp", "wind", "price", "holiday", "bias"])

        self.assertEqual(groups.names, ("weather", "price", "holiday"))
        self.assertEqual(groups.groups, ((0, 1), (2,), (3,)))
        self.assertEqual(groups.always_present, (4,))

    def test_remaining_features_can_form_one_player(self):
        groups = FeatureGroups(
            {"weather": [0, 1]}, remaining="group", remaining_name="other"
        ).resolve(5)
        self.assertEqual(groups.names, ("weather", "other"))
        self.assertEqual(groups.groups, ((0, 1), (2, 3, 4)))

    def test_overlapping_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            FeatureGroups({"a": [0, 1], "b": [1, 2]}).resolve(3)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.data = np.asarray([[1.0, 2.0], [3.0, 8.0]])
        self.spec = DataSpec(feature_names=["a", "b"])

    def resolve(self, baseline, unit=(0,)):
        data = self.spec.prepare(self.data)
        masker = BaselineMasker(baseline)
        masker.prepare(data, self.spec)
        return masker.mask(
            data[unit[0]], (), 1, np.random.default_rng(0), unit=unit
        )[0]

    def test_named_baselines(self):
        np.testing.assert_allclose(self.resolve("zero"), [0.0, 0.0])
        np.testing.assert_allclose(self.resolve("mean"), [2.0, 5.0])
        np.testing.assert_allclose(self.resolve("median"), [2.0, 5.0])

    def test_scalar_vector_tensor_and_callable_baselines(self):
        np.testing.assert_allclose(self.resolve(4.0), [4.0, 4.0])
        np.testing.assert_allclose(self.resolve([5.0, 6.0]), [5.0, 6.0])
        np.testing.assert_allclose(
            self.resolve(np.asarray([[9.0, 10.0], [11.0, 12.0]])),
            [9.0, 10.0],
        )
        np.testing.assert_allclose(
            self.resolve(lambda *, x, **_: np.zeros_like(x) + 7.0),
            [7.0, 7.0],
        )

    def test_present_features_are_restored(self):
        data = self.spec.prepare(self.data)
        masker = BaselineMasker(0)
        masker.prepare(data, self.spec)
        masked = masker.mask(
            data[0], (1,), 2, np.random.default_rng(0), unit=(0,)
        )
        np.testing.assert_allclose(masked, [[0.0, 2.0], [0.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
