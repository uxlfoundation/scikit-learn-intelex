# ===============================================================================
# Copyright 2021 Intel Corporation
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import lstsq
from sklearn.datasets import make_regression

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.tests.utils import _IS_INTEL


# Note: this is arranged as a fixture with a finalizer instead of as a parameter
# 'True' / 'False' in order to undo the changes later so that it doesn't affect
# other tests afterwards. It returns a function instead of making the change
# directly, in order to avoid importing the estimator class before the import test
# itself, but it still needs to import the class inside the the function that it
# returns due to serialization logic in pytest causing differences w.r.t. current
# closure where the function is called.
@pytest.fixture(params=[False, True])
def non_batched_route(request):
    def change_parameters(queue, macro_block):
        from sklearnex.linear_model import LinearRegression

        if request.param and daal_check_version((2025, "P", 500)):
            if queue and queue.sycl_device.is_gpu:
                pytest.skip("Test for CPU-only functionality")
            if macro_block is not None:
                pytest.skip("Parameter combination with no effect")

            non_batched_route.curr_cpu_max_cols_batched = (
                LinearRegression.get_hyperparameters("fit").cpu_max_cols_batched
            )
            non_batched_route.curr_cpu_small_rows_threshold = (
                LinearRegression.get_hyperparameters("fit").cpu_small_rows_threshold
            )
            non_batched_route.curr_cpu_small_rows_max_cols_batched = (
                LinearRegression.get_hyperparameters(
                    "fit"
                ).cpu_small_rows_max_cols_batched
            )
            LinearRegression.get_hyperparameters("fit").cpu_max_cols_batched = 1
            LinearRegression.get_hyperparameters("fit").cpu_small_rows_threshold = 1
            LinearRegression.get_hyperparameters(
                "fit"
            ).cpu_small_rows_max_cols_batched = 1

        elif request.param and not daal_check_version((2025, "P", 500)):
            pytest.skip("Functionality introduced in later versions")

    def restore_params():
        from sklearnex.linear_model import LinearRegression

        if request.param and daal_check_version((2025, "P", 500)):
            LinearRegression.get_hyperparameters("fit").cpu_max_cols_batched = (
                non_batched_route.curr_cpu_max_cols_batched
            )
            LinearRegression.get_hyperparameters("fit").cpu_small_rows_threshold = (
                non_batched_route.curr_cpu_small_rows_threshold
            )
            LinearRegression.get_hyperparameters(
                "fit"
            ).cpu_small_rows_max_cols_batched = (
                non_batched_route.curr_cpu_small_rows_max_cols_batched
            )

    request.addfinalizer(restore_params)
    return change_parameters


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("overdetermined", [False, True])
@pytest.mark.parametrize("multi_output", [False, True])
def test_sklearnex_import_linear(
    dataframe, queue, dtype, macro_block, non_batched_route, overdetermined, multi_output
):
    if (not overdetermined or multi_output) and not daal_check_version((2025, "P", 1)):
        pytest.skip("Functionality introduced in later versions")
    if (
        not overdetermined
        and queue
        and queue.sycl_device.is_gpu
        and not daal_check_version((2025, "P", 200))
    ):
        pytest.skip("Functionality introduced in later versions")

    from sklearnex.linear_model import LinearRegression

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(10, 20) if not overdetermined else (20, 5))
    y = rng.standard_normal(size=(X.shape[0], 3) if multi_output else X.shape[0])

    Xi = np.c_[X, np.ones((X.shape[0], 1))]
    expected_coefs = lstsq(Xi, y)[0]
    expected_intercept = expected_coefs[-1]
    expected_coefs = expected_coefs[: X.shape[1]]
    if multi_output:
        expected_coefs = expected_coefs.T

    linreg = LinearRegression()
    if daal_check_version((2024, "P", 0)) and macro_block is not None:
        hparams = LinearRegression.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block
    non_batched_route(queue, macro_block)

    X = X.astype(dtype=dtype)
    y = y.astype(dtype=dtype)
    y_list = y.tolist()
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    linreg.fit(X, y)

    assert hasattr(linreg, "_onedal_estimator")
    assert "sklearnex" in linreg.__module__

    rtol = 1e-3 if dtype == np.float32 else 1e-5
    assert_allclose(_as_numpy(linreg.coef_), expected_coefs, rtol=rtol)
    assert_allclose(_as_numpy(linreg.intercept_), expected_intercept, rtol=rtol)

    # check that it also works with lists
    if isinstance(X, np.ndarray):
        linreg_list = LinearRegression().fit(X, y_list)
        assert_allclose(linreg_list.coef_, linreg.coef_)
        assert_allclose(linreg_list.intercept_, linreg.intercept_)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_lasso(dataframe, queue):
    from sklearnex.linear_model import Lasso

    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    lasso = Lasso(alpha=0.1).fit(X, y)
    assert "daal4py" in lasso.__module__
    assert_allclose(lasso.intercept_, 0.15)
    assert_allclose(lasso.coef_, [0.85, 0.0])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_elastic(dataframe, queue):
    from sklearnex.linear_model import ElasticNet

    X, y = make_regression(n_features=2, random_state=0)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    elasticnet = ElasticNet(random_state=0).fit(X, y)
    assert "daal4py" in elasticnet.__module__
    assert_allclose(elasticnet.intercept_, 1.451, atol=1e-3)
    assert_allclose(elasticnet.coef_, [18.838, 64.559], atol=1e-3)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_reconstruct_model(dataframe, queue, dtype):
    from sklearnex.linear_model import LinearRegression

    seed = 42
    num_samples = 3500
    num_features, num_targets = 14, 9

    gen = np.random.default_rng(seed)
    intercept = gen.random(size=num_targets, dtype=dtype)
    coef = gen.random(size=(num_targets, num_features), dtype=dtype).T

    X = gen.random(size=(num_samples, num_features), dtype=dtype)
    gtr = X @ coef + intercept[np.newaxis, :]

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    linreg = LinearRegression(fit_intercept=True)
    linreg.coef_ = coef.T
    linreg.intercept_ = intercept

    y_pred = linreg.predict(X)

    tol = 1e-5 if _as_numpy(y_pred).dtype == np.float32 else 1e-7
    assert_allclose(gtr, _as_numpy(y_pred), rtol=tol)
