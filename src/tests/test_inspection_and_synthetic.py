import importlib.util
import unittest

import numpy as np

from xpc import (
    make_synthetic_forecasting_data,
    parameter_counts,
    parameter_structure,
)


class FakeParameter:
    def __init__(self, shape, requires_grad, dtype="float32"):
        self.shape = shape
        self.requires_grad = requires_grad
        self.dtype = dtype

    def numel(self):
        return int(np.prod(self.shape))


class FakeModel:
    def named_parameters(self):
        return iter(
            [
                ("encoder.weight", FakeParameter((3, 4), True)),
                ("encoder.bias", FakeParameter((3,), False)),
            ]
        )


class InspectionAndSyntheticTests(unittest.TestCase):
    def test_parameter_structure_and_counts(self):
        structure = parameter_structure(FakeModel())
        self.assertEqual(structure[0].name, "encoder.weight")
        self.assertEqual(structure[0].shape, (3, 4))
        self.assertEqual(structure[0].count, 12)
        counts = parameter_counts(FakeModel())
        self.assertEqual(counts.total, 15)
        self.assertEqual(counts.trainable, 12)
        self.assertEqual(counts.frozen, 3)
        self.assertEqual(counts.tensors, 2)
        self.assertEqual(counts.trainable_tensors, 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_torch_parameter_counts(self):
        import torch

        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 1))
        model[0].bias.requires_grad_(False)
        counts = parameter_counts(model)
        self.assertEqual(counts.total, 19)
        self.assertEqual(counts.trainable, 16)
        self.assertEqual(counts.frozen, 3)
        self.assertEqual(counts.tensors, 4)

    def test_synthetic_data_is_seeded_and_windowed(self):
        first = make_synthetic_forecasting_data(
            n_steps=80,
            context_length=12,
            horizon=4,
            stride=3,
            seed=9,
        )
        second = make_synthetic_forecasting_data(
            n_steps=80,
            context_length=12,
            horizon=4,
            stride=3,
            seed=9,
        )
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_allclose(
            first.values,
            np.sum(np.stack(tuple(first.components.values())), axis=0),
        )
        self.assertEqual(first.contexts.shape, (22, 12, 1))
        self.assertEqual(first.targets.shape, (22, 4, 1))
        np.testing.assert_array_equal(first.contexts[0, :, 0], first.values[:12])
        np.testing.assert_array_equal(first.targets[0, :, 0], first.values[12:16])


if __name__ == "__main__":
    unittest.main()
