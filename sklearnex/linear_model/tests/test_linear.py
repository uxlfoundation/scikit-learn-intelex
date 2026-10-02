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

import array_api_strict
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import lstsq

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _assert_in_namespace,
    _convert_to_dataframe,
    assert_allclose_numpy,
    dpnp_available,
    get_dataframes_and_queues,
    torch_available,
)
from onedal.tests.utils._device_selection import (
    is_sycl_device_available,
)

if dpnp_available:
    import dpnp
if torch_available:
    import torch

if sklearn_check_version("1.9"):
    from sklearn.utils._array_api import (
        get_namespace_and_device,
        move_estimator_to,
        move_to,
    )


@pytest.fixture
def hyperparameters(request):
    from sklearnex.linear_model import LinearRegression

    hparams = LinearRegression.get_hyperparameters("fit")

    def restore_hyperparameters():
        LinearRegression.reset_hyperparameters("fit")

    request.addfinalizer(restore_hyperparameters)
    return hparams


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("non_batched_route", [False, True])
@pytest.mark.parametrize("overdetermined", [False, True])
@pytest.mark.parametrize("multi_output", [False, True])
def test_sklearnex_import_linear(
    hyperparameters,
    dataframe,
    queue,
    dtype,
    macro_block,
    non_batched_route,
    overdetermined,
    multi_output,
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
    if macro_block is not None:
        hyperparameters.cpu_macro_block = macro_block
        hyperparameters.gpu_macro_block = macro_block
        if daal_check_version((2025, "P", 500)) and non_batched_route:
            # If the non-batched route is requested, set the parameters to use it
            hyperparameters.cpu_max_cols_batched = 1
            hyperparameters.cpu_small_rows_threshold = 1
            hyperparameters.cpu_small_rows_max_cols_batched = 1

    X = X.astype(dtype=dtype)
    y = y.astype(dtype=dtype)
    y_list = y.tolist()
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    linreg.fit(X, y)

    assert hasattr(linreg, "_onedal_estimator")
    assert "sklearnex" in linreg.__module__

    rtol = 1e-3 if dtype == np.float32 else 1e-5
    _assert_in_namespace(linreg.coef_, dataframe)
    assert_allclose_numpy(linreg.coef_, expected_coefs, rtol=rtol)
    assert_allclose_numpy(linreg.intercept_, expected_intercept, rtol=rtol)

    # check that it also works with lists
    if isinstance(X, np.ndarray):
        linreg_list = LinearRegression().fit(X, y_list)
        assert_allclose(linreg_list.coef_, linreg.coef_)
        assert_allclose(linreg_list.intercept_, linreg.intercept_)


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
    # reconstructed attrs must share X's namespace/device under array_api_dispatch
    linreg.coef_ = _convert_to_dataframe(coef.T, sycl_queue=queue, target_df=dataframe)
    linreg.intercept_ = _convert_to_dataframe(
        intercept, sycl_queue=queue, target_df=dataframe
    )

    y_pred = linreg.predict(X)

    _assert_in_namespace(y_pred, dataframe)
    tol = 1e-5 if _as_numpy(y_pred).dtype == np.float32 else 1e-7
    assert_allclose_numpy(gtr, y_pred, rtol=tol)


@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later sklearn versions",
)
@pytest.mark.parametrize(
    "array_like",
    [array_api_strict.arange(1)]
    + (
        [dpnp.arange(1, device="gpu")]
        if dpnp_available and is_sycl_device_available
        else []
    )
    + (
        [torch.arange(1, device="xpu")]
        if torch_available and is_sycl_device_available
        else []
    ),
)
def test_move_estimator_to_np_to_arrayapi(array_like, with_array_api):
    from sklearnex.linear_model import LinearRegression

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(10, 3))
    y = rng.standard_normal(size=X.shape[0])

    xp, _, device = get_namespace_and_device(array_like)
    X_array_api = move_to(X, xp=xp, device=device)

    model = LinearRegression().fit(X, y)
    model2 = move_estimator_to(model, xp, device)

    assert isinstance(model.coef_, np.ndarray)
    assert not isinstance(model2.coef_, np.ndarray)
    assert model2.coef_.__class__ == array_like.__class__

    pred_np = model.predict(X)
    pred_array_api = model2.predict(X_array_api)

    assert pred_array_api.__class__ == X_array_api.__class__
    pred_array_api = _as_numpy(pred_array_api)
    np.testing.assert_allclose(pred_array_api, pred_np, atol=1e-6)


# TODO: update this once scikit-learn introduces a config option
# to control whether the attributes are always numpy or follow 'X'
@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later sklearn versions",
)
@pytest.mark.parametrize(
    "array_like",
    [array_api_strict.arange(1)]
    + (
        [dpnp.arange(1, device="gpu")]
        if dpnp_available and is_sycl_device_available
        else []
    )
    + (
        [torch.arange(1, device="xpu")]
        if torch_available and is_sycl_device_available
        else []
    ),
)
def test_move_estimator_to_arrayapi_to_np(array_like, with_array_api):
    from sklearnex.linear_model import LinearRegression

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(10, 3))
    y = rng.standard_normal(size=X.shape[0])

    xp, _, device = get_namespace_and_device(array_like)
    X_array_api = move_to(X, xp=xp, device=device)
    y_array_api = move_to(y, xp=xp, device=device)

    xp_np, _, device_np = get_namespace_and_device(X)

    model = LinearRegression().fit(X_array_api, y_array_api)
    model2 = move_estimator_to(model, xp_np, device_np)

    assert not isinstance(model.coef_, np.ndarray)
    assert isinstance(model2.coef_, np.ndarray)

    pred_np = model2.predict(X)
    pred_array_api = model.predict(X_array_api)

    assert isinstance(pred_np, np.ndarray)
    assert pred_array_api.__class__ == X_array_api.__class__
    pred_array_api = _as_numpy(pred_array_api)
    np.testing.assert_allclose(pred_np, pred_array_api, atol=1e-6)
