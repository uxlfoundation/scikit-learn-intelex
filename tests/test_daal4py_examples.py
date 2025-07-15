# ==============================================================================
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import importlib
import inspect
import os
import sys
import time
import unittest
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from daal4py.sklearn._utils import daal_check_version, get_daal_version

daal_version = get_daal_version()

project_path = Path(__file__).absolute().parent.parent
example_path = project_path / "examples" / "daal4py"
batch_data_path = example_path / "data" / "batch"
distributed_data_path = example_path / "data" / "distributed"
example_data_path = project_path / "tests" / "unittest_data"


def np_load_distributed(
    path: Path, file_name: str, parts: range, delimiter: str = ","
) -> npt.NDArray[np.float64]:
    data = None
    for i in parts:
        new_data = np.loadtxt(str(path / file_name.format(i=i)), delimiter=delimiter)
        data = new_data if data is None else np.append(data, new_data, axis=0)
    assert data is not None, "Empty parts range is not supported"
    return data


# Examples are not part of any module, so we import them by modifying the
# system's path
def import_module_any_path(path: Path) -> ModuleType:
    """Import a module from any path"""
    import_path = str(path.parent)
    sys.path.insert(0, import_path)
    module = importlib.import_module(path.stem)
    sys.path.pop(0)
    return module


readcsv = import_module_any_path(example_path / "readcsv.py")


@dataclass
class Config:
    module_name: str
    result_file_name: str = ""
    result_attribute: Union[str, Callable[..., Any]] = ""
    required_version: Optional[Tuple[Any, ...]] = None
    req_libs: List[str] = field(default_factory=list)
    timeout_cpu_seconds: int = 170
    suspended_on: Optional[Tuple[int, int, int]] = None
    suspended_for_n_days: int = 30

    def is_suspended(self):
        if self.suspended_on is None:
            return False

        suspended_date = datetime(*self.suspended_on)
        time_delta = datetime.now() - suspended_date
        days_passed = time_delta.days

        return days_passed < self.suspended_for_n_days

    def get_missing_dep(self) -> Optional[str]:
        """Returns the name of the missing lib"""
        for module_name in self.req_libs:
            try:
                import_module(module_name)
            except ImportError:
                return module_name

    def check_version(self):
        return self.required_version is None or daal_check_version(self.required_version)


class Base:
    """
    We also use generic functions to test these, they get added later.
    """

    def call_main(self, module: ModuleType) -> Any:
        """To be implemented by inheriting classes"""
        ...

    @classmethod
    def add_test(cls, config: Config):
        """Add a test to the class"""
        test_name = f"test_{config.module_name}"
        if getattr(cls, test_name, None) is not None:
            # already added, do not override
            return

        missing_dep = config.get_missing_dep()

        @unittest.skipUnless(
            config.check_version(),
            f"Minimum required version {config.required_version} (have {daal_version})",
        )
        @unittest.skipUnless(missing_dep is None, f"Missing dependency: {missing_dep}")
        @unittest.skipIf(
            config.is_suspended(),
            f"Test was suspended for {config.suspended_for_n_days} days on {config.suspended_on}",
        )
        def run_test(self):
            start = time.process_time()

            ex = import_module_any_path(example_path / config.module_name)

            if not hasattr(ex, "main"):
                self.skipTest("Missing main function")

            result: Any = self.call_main(ex)
            if config.result_file_name and config.result_attribute:
                testdata = readcsv.np_read_csv(
                    example_data_path / config.result_file_name
                )
                ra = config.result_attribute
                actual = ra(result) if callable(ra) else getattr(result, ra)
                np.testing.assert_allclose(actual, testdata, atol=1e-05)

            duration_seconds = time.process_time() - start
            timeout_msg = (
                "Runtime (in seconds) too long. Test timeout. "
                "Decrease workload or increase `timeout_cpu_seconds`"
            )
            self.assertLessEqual(
                duration_seconds, config.timeout_cpu_seconds, msg=timeout_msg
            )

        setattr(cls, test_name, run_test)


class Daal4pyBase(Base):
    def test_svd(self):
        ex = import_module_any_path(example_path / "svd")

        reference, intermediate = self.call_main(ex)
        result = np.matmul(
            np.matmul(
                intermediate.leftSingularMatrix, np.diag(intermediate.singularValues[0])
            ),
            intermediate.rightSingularMatrix,
        )

        np.testing.assert_allclose(reference, result)

    def test_svd_stream(self):
        ex = import_module_any_path(example_path / "svd_streaming")

        intermediate = self.call_main(ex)
        result = np.matmul(
            np.matmul(
                intermediate.leftSingularMatrix,
                np.diag(intermediate.singularValues[0]),
            ),
            intermediate.rightSingularMatrix,
        )

        reference = np_load_distributed(distributed_data_path, "svd_{i}.csv", range(1, 5))

        np.testing.assert_allclose(reference, result)

    def test_qr(self):
        ex = import_module_any_path(example_path / "qr")

        data, result = self.call_main(ex)
        np.testing.assert_allclose(data, np.matmul(result.matrixQ, result.matrixR))

    def test_qr_stream(self):
        ex = import_module_any_path(example_path / "qr_streaming")

        result = self.call_main(ex)
        data = np_load_distributed(distributed_data_path, "qr_{i}.csv", range(1, 5))
        np.testing.assert_allclose(data, np.matmul(result.matrixQ, result.matrixR))

    def test_svm(self):
        ex = import_module_any_path(example_path / "svm")
        testdata = readcsv.np_read_csv(example_data_path / "svm.csv", range(1))

        decision_result, _, _ = self.call_main(ex)
        left = np.absolute(decision_result - testdata).max()
        right = np.absolute(decision_result.max() - decision_result.min()) * 0.05
        self.assertLess(left, right)


def correlation_distance_getter(result):
    return [
        [np.amin(result.correlationDistance)],
        [np.amax(result.correlationDistance)],
        [np.mean(result.correlationDistance)],
        [np.average(result.correlationDistance)],
    ]


def cosine_distance_getter(result):
    return [
        [np.amin(result.cosineDistance)],
        [np.amax(result.cosineDistance)],
        [np.mean(result.cosineDistance)],
        [np.average(result.cosineDistance)],
    ]


def low_order_moms_getter(result):
    return np.vstack(
        (
            result.minimum,
            result.maximum,
            result.sum,
            result.sumSquares,
            result.sumSquaresCentered,
            result.mean,
            result.secondOrderRawMoment,
            result.variance,
            result.standardDeviation,
            result.variation,
        )
    )


examples = [
    Config("adaboost", required_version=(2020, "P", 0)),
    Config("adagrad_mse", "adagrad_mse.csv", "minimum"),
    Config("association_rules", "association_rules.csv", "confidence"),
    Config("bacon_outlier", "multivariate_outlier.csv", lambda r: r[1].weights),
    Config("brownboost", required_version=(2020, "P", 0)),
    Config("cholesky", "cholesky.csv", "choleskyFactor"),
    Config(
        "correlation_distance", "correlation_distance.csv", correlation_distance_getter
    ),
    Config("cosine_distance", "cosine_distance.csv", cosine_distance_getter),
    Config("covariance_streaming", "covariance.csv", "covariance"),
    Config("covariance", "covariance.csv", "covariance"),
    Config("dbscan", "dbscan.csv", "assignments", (2019, "P", 5)),
    Config(
        "decision_forest_classification_default_dense",
        result_attribute=lambda r: r[1].prediction,
        required_version=(2023, "P", 1),
    ),
    Config(
        "decision_forest_classification_hist",
        result_attribute=lambda r: r[1].prediction,
        required_version=(2023, "P", 1),
    ),
    Config(
        "decision_forest_regression_default_dense",
        "decision_forest_regression_20230101.csv",
        result_attribute=lambda r: r[1].prediction,
        required_version=(2023, "P", 101),
    ),
    Config(
        "decision_forest_regression_hist",
        "decision_forest_regression_20230101.csv",
        result_attribute=lambda r: r[1].prediction,
        required_version=(2023, "P", 101),
    ),
    Config(
        "decision_tree_classification",
        "decision_tree_classification.csv",
        result_attribute=lambda r: r[1].prediction,
    ),
    Config(
        "decision_tree_regression",
        "decision_tree_regression.csv",
        result_attribute=lambda r: r[1].prediction,
    ),
    Config("elastic_net", required_version=((2020, "P", 1), (2021, "B", 105))),
    Config("em_gmm", "em_gmm.csv", lambda r: r.covariances[0]),
    Config("gradient_boosted_classification", timeout_cpu_seconds=480),
    Config("implicit_als", "implicit_als.csv", "prediction"),
    Config("kdtree_knn_classification"),
    Config("kmeans", "kmeans.csv", "centroids"),
    Config("lasso_regression", required_version=(2019, "P", 5)),
    Config("lbfgs_cr_entr_loss", "lbfgs_cr_entr_loss.csv", "minimum"),
    Config("lbfgs_mse", "lbfgs_mse.csv", "minimum"),
    Config(
        "linear_regression_streaming", "linear_regression.csv", lambda r: r[1].prediction
    ),
    Config("linear_regression", "linear_regression.csv", lambda r: r[1].prediction),
    Config("log_reg_binary_dense", "log_reg_binary_dense.csv", lambda r: r[1].prediction),
    Config("logitboost", required_version=(2020, "P", 0)),
    Config("low_order_moms_dense", "low_order_moms_dense.csv", low_order_moms_getter),
    Config("low_order_moms_streaming", "low_order_moms_dense.csv", low_order_moms_getter),
    Config("multivariate_outlier", "multivariate_outlier.csv", lambda r: r[1].weights),
    Config("naive_bayes_streaming", "naive_bayes.csv", lambda r: r[0].prediction),
    Config("naive_bayes", "naive_bayes.csv", lambda r: r[0].prediction),
    Config("normalization_minmax", "normalization_minmax.csv", "normalizedData"),
    Config("normalization_zscore", "normalization_zscore.csv", "normalizedData"),
    Config("pca_transform", "pca_transform.csv", lambda r: r[1].transformedData),
    Config("pca", "pca.csv", "eigenvectors"),
    Config("pivoted_qr", "pivoted_qr.csv", "matrixR"),
    Config("quantiles", "quantiles.csv", "quantiles"),
    Config(
        "ridge_regression_streaming", "ridge_regression.csv", lambda r: r[0].prediction
    ),
    Config("ridge_regression", "ridge_regression.csv", lambda r: r[0].prediction),
    Config("saga", required_version=(2019, "P", 3)),
    Config("sgd_logistic_loss", "sgd_logistic_loss.csv", "minimum"),
    Config("sgd_mse", "sgd_mse.csv", "minimum"),
    Config("stump_classification", required_version=(2020, "P", 0)),
    Config("stump_regression", required_version=(2020, "P", 0)),
    Config("svm_multiclass", "svm_multiclass.csv", lambda r: r[0].prediction),
    Config("univariate_outlier", "univariate_outlier.csv", lambda r: r[1].weights),
]

module_names_with_configs = [cfg.module_name for cfg in examples]

# add all examples that do not have an explicit config
for fname in os.listdir(example_path):
    if fname == "__init__.py":
        continue
    if not fname.endswith(".py"):
        continue
    if "spmd" in fname:
        # spmd examples are done in test_daal4py_spmd_examples.py
        continue
    stem = Path(fname).stem
    if stem in module_names_with_configs:
        continue

    examples.append(Config(stem))

for cfg in examples:
    Daal4pyBase.add_test(cfg)


class TestExNpyArray(Daal4pyBase, unittest.TestCase):
    """
    We run and validate all the examples but read data with numpy,
    so working natively on a numpy arrays.
    """

    def call_main(self, module: ModuleType):
        signature = inspect.signature(module.main)
        if "readcsv" in list(signature.parameters):
            return module.main(readcsv=readcsv.np_read_csv)
        else:
            return module.main()


class TestExPandasDF(Daal4pyBase, unittest.TestCase):
    """
    We run and validate all the examples but read data with pandas,
    so working natively on a pandas DataFrame
    """

    def call_main(self, module: ModuleType):
        signature = inspect.signature(module.main)
        if "readcsv" not in list(signature.parameters):
            self.skipTest("Missing readcsv kwarg support")

        return module.main(readcsv=readcsv.pd_read_csv)


class TestExCSRMatrix(Daal4pyBase, unittest.TestCase):
    """
    We run and validate all the examples but use scipy-sparse-csr_matrix as input data.
    We also let algos use CSR method (some algos ignore the method argument since they
    do not specifically support CSR).
    """

    def call_main(self, module: ModuleType):
        if any(
            module.__name__.startswith(x)
            for x in [
                "adaboost",
                "brownboost",
                "decision_forest",
                "sorting",
                "stump_classification",
            ]
        ):
            self.skipTest("Missing CSR support")

        signature = inspect.signature(module.main)
        parameters = list(signature.parameters)
        if "readcsv" not in parameters:
            self.skipTest("Missing readcsv kwarg support")

        if "naive_bayes" in module.__name__:
            self.skipTest("CSR support in Naive Bayes is buggy")

        if "method" in parameters:
            method = "fastCSR"
            if any(x in module.__name__ for x in ["low_order_moms", "covariance"]):
                method = "singlePassCSR"
            if "implicit_als" in module.__name__:
                method = "defaultDense"
            if "kmeans" in module.__name__:
                method = "randomDense"
            if hasattr(module, "dflt_method"):
                method = module.dflt_method.replace("defaultDense", "fastCSR").replace(
                    "Dense", "CSR"
                )
            return module.main(readcsv=readcsv.csr_read_csv, method=method)
        else:
            return module.main(readcsv=readcsv.csr_read_csv)


if __name__ == "__main__":
    unittest.main()
