import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

from xpc import (
    BaselineMasker,
    DataSpec,
    NumpyModelAdapter,
    PandasModelAdapter,
    RScriptModelAdapter,
    ShapleyExplainer,
    ScriptModelAdapter,
    TorchModelAdapter,
    adapt_model,
    register_model_adapter,
)


class AdapterTests(unittest.TestCase):
    def test_numpy_callable_and_predict_class(self):
        callable_adapter = NumpyModelAdapter(lambda x: np.sum(x, axis=1))
        np.testing.assert_allclose(callable_adapter.predict([[1, 2]]), [3])

        class Model:
            def predict(self, x):
                return np.asarray(x)[:, 0]

        np.testing.assert_allclose(adapt_model(Model()).predict([[4, 5]]), [4])

    @unittest.skipUnless(
        importlib.util.find_spec("pandas") is not None, "pandas is optional"
    )
    def test_pandas_adapter_preserves_feature_names(self):
        adapter = PandasModelAdapter(
            lambda frame: frame["b"] - frame["a"], ["a", "b"]
        )
        np.testing.assert_allclose(adapter.predict(np.asarray([[2, 7]])), [5])

        import pandas as pd

        frame = pd.DataFrame([[2.0, 7.0]], columns=["a", "b"])
        explanation = ShapleyExplainer(
            adapter,
            BaselineMasker(0),
            data_spec=DataSpec(feature_names=frame.columns),
            n_coalitions=3,
            random_state=0,
        )(frame)
        np.testing.assert_allclose(explanation.values[0, 0], [-2.0, 7.0])

    @unittest.skipUnless(
        importlib.util.find_spec("pandas") is not None, "pandas is optional"
    )
    def test_script_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "predict.py"
            script.write_text(
                "import pandas as pd, sys\n"
                "x = pd.read_csv(sys.argv[1])\n"
                "pd.DataFrame({'prediction': x.sum(axis=1)}).to_csv(sys.argv[2], index=False)\n",
                encoding="utf-8",
            )
            adapter = ScriptModelAdapter(
                [sys.executable, str(script), "{input}", "{output}"],
                feature_names=["a", "b"],
                output_columns=["prediction"],
            )
            np.testing.assert_allclose(
                adapter.predict(np.asarray([[1.0, 2.0], [3.0, 4.0]])).ravel(),
                [3.0, 7.0],
            )

    def test_custom_registered_class(self):
        class CustomModel:
            def run(self, x):
                return np.asarray(x)[:, 0] * 3.0

        register_model_adapter(
            CustomModel, lambda model: NumpyModelAdapter(model.run), priority=10
        )
        np.testing.assert_allclose(
            adapt_model(CustomModel()).predict(np.asarray([[2.0]])), [6.0]
        )

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn") is not None, "scikit-learn is optional"
    )
    def test_sklearn_predict_adapter(self):
        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit([[0.0], [1.0]], [0.0, 2.0])
        np.testing.assert_allclose(adapt_model(model).predict([[3.0]]), [6.0])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None, "torch is optional"
    )
    def test_torch_adapter(self):
        import torch

        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight[:] = torch.tensor([[2.0, 3.0]])
        prediction = TorchModelAdapter(model).predict(np.asarray([[1.0, 2.0]]))
        np.testing.assert_allclose(prediction, [[8.0]])

    @unittest.skipUnless(shutil.which("Rscript"), "Rscript is optional")
    def test_r_script_adapter_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "predict.R"
            script.write_text(
                "args <- commandArgs(trailingOnly=TRUE)\n"
                "x <- read.csv(args[1])\n"
                "write.csv(data.frame(prediction=rowSums(x)), args[2], row.names=FALSE)\n",
                encoding="utf-8",
            )
            adapter = RScriptModelAdapter(script, output_columns=["prediction"])
            np.testing.assert_allclose(adapter.predict([[1.0, 2.0]]), [[3.0]])


if __name__ == "__main__":
    unittest.main()
