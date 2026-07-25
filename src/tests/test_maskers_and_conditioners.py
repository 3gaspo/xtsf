import unittest

import numpy as np

from xpc import (
    CallableConditioner,
    ConditionalMasker,
    DataSpec,
    EmpiricalConditioner,
    GridConditioner,
    RandomMasker,
)


class MaskerConditionerTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.asarray(
            [[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        )
        self.spec = DataSpec()
        self.data = self.spec.prepare(np.asarray([[2.1, 999.0]]))

    def test_random_masker_only_draws_observed_rows(self):
        masker = RandomMasker(self.reference)
        masker.prepare(self.data, self.spec)
        rows = masker.mask(
            self.data[0], (), 20, np.random.default_rng(3), unit=(0,)
        )
        observed = {tuple(row) for row in self.reference}
        self.assertTrue(all(tuple(row) in observed for row in rows))

    def test_empirical_conditioner_uses_nearest_observed_rows(self):
        masker = ConditionalMasker(
            self.reference, EmpiricalConditioner(n_neighbors=1)
        )
        masker.prepare(self.data, self.spec)
        rows = masker.mask(
            self.data[0], (0,), 5, np.random.default_rng(1), unit=(0,)
        )
        np.testing.assert_allclose(rows[:, 0], 2.1)
        np.testing.assert_allclose(rows[:, 1], 20.0)

    def test_grid_conditioner_matches_cells(self):
        conditioner = GridConditioner(n_bins=2, strategy="uniform")
        rows = conditioner.sample(
            self.reference,
            np.asarray([2.5, 0.0]),
            observed=(0,),
            n_samples=20,
            rng=np.random.default_rng(2),
        )
        self.assertTrue(np.all(rows[:, 0] >= 2.0))

    def test_callable_conditioner(self):
        conditioner = CallableConditioner(
            lambda *, reference, n_samples, **_: np.repeat(
                reference[-1:], n_samples, axis=0
            )
        )
        rows = conditioner.sample(
            self.reference,
            self.data[0],
            observed=(0,),
            n_samples=3,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(rows, np.repeat([[3.0, 30.0]], 3, axis=0))


if __name__ == "__main__":
    unittest.main()
