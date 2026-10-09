# ==============================================================================
# Copyright contributors to the oneDAL project
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
import warnings

import array_api_strict
import numpy as np
import pytest

from daal4py.sklearn._utils import _package_check_version, sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    dpnp_available,
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

    from sklearnex.tests.utils.misc import assert_same_namespace


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("dim_y", [1, 3])
@pytest.mark.parametrize("estimator", ["LinearRegression", "Ridge"])
@pytest.mark.allow_sklearn_fallback
def test_predict_after_fallback(fit_intercept, dim_y, estimator):
    from sklearnex import linear_model

    model = getattr(linear_model, estimator)(fit_intercept=fit_intercept)

    rng = np.random.default_rng(seed=123)
    X = rng.random(size=(10, 3))
    y = rng.random(X.shape[0] if dim_y == 1 else (X.shape[0], dim_y))
    w = rng.standard_gamma(1, size=X.shape[0])

    # Note: weights are not supported by oneDAL, so this should trigger a fallback.
    # If they become supported in the future, this test would need to be adapted
    # to trigger a fallback in some other way
    model.fit(X, y, w)
    assert not hasattr(model, "_onedal_estimator")

    pred = model.predict(X)

    assert hasattr(model, "_onedal_estimator")
    if dim_y == 1:
        expected_pred = X @ model.coef_ + model.intercept_
    else:
        expected_pred = X @ model.coef_.T
        if fit_intercept:
            expected_pred += model.intercept_.reshape((1, -1))

    np.testing.assert_allclose(pred, expected_pred)


# TODO: Extend this test to LinearRegression once scikit-learn adds array API support
@pytest.mark.skipif(
    not sklearn_check_version("1.8"),
    reason="Functionality introduced in later scikit-learn versions.",
)
@pytest.mark.parametrize("fit_intercept", [True, False])
# Note: this is due to a bug that was fixed in later sklearn versions
@pytest.mark.parametrize("dim_y", [1] + ([3] if sklearn_check_version("1.9") else []))
@pytest.mark.parametrize("estimator", ["Ridge"])
@pytest.mark.allow_sklearn_fallback
def test_predict_after_fallback_array_api(
    fit_intercept, dim_y, estimator, with_array_api
):
    from sklearnex import linear_model

    model = getattr(linear_model, estimator)(fit_intercept=fit_intercept)

    rng = np.random.default_rng(seed=123)
    X = rng.random(size=(10, 3))
    y = rng.random(X.shape[0] if dim_y == 1 else (X.shape[0], dim_y))
    w = rng.standard_gamma(1, size=X.shape[0])

    X = array_api_strict.asarray(X.astype(np.float32))
    y = array_api_strict.asarray(y.astype(np.float32))
    w = array_api_strict.asarray(w.astype(np.float32))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(X, y, w)
    assert not hasattr(model, "_onedal_estimator")

    pred = model.predict(X)
    assert pred.__class__ == X.__class__
    assert pred.dtype == array_api_strict.float32

    assert hasattr(model, "_onedal_estimator")
    if dim_y == 1:
        expected_pred = X @ model.coef_ + model.intercept_
    else:
        expected_pred = X @ model.coef_.T
        if fit_intercept:
            expected_pred += array_api_strict.reshape(model.intercept_, (1, -1))

    pred = np.array(pred)
    expected_pred = np.array(pred)
    np.testing.assert_allclose(pred, expected_pred)


# TODO: update this once scikit-learn introduces a config option
# to control whether the attributes are always numpy or follow 'X'
@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later sklearn versions",
)
@pytest.mark.skipif(
    not _package_check_version("2.2", np.__version__),
    reason="Requires more recent NumPy version",
)
@pytest.mark.parametrize("estimator", ["LinearRegression", "Ridge"])
@pytest.mark.parametrize(
    "array_input_like",
    [np.arange(1), array_api_strict.arange(1)]
    + (
        [dpnp.arange(1, device="gpu")]
        if dpnp_available and is_sycl_device_available("gpu")
        else []
    )
    # Note: 'move_to' has issues with Torch inputs
    # in older sklearn versions.
    + (
        [torch.arange(1, device="xpu")]
        if torch_available
        and is_sycl_device_available("gpu")
        and sklearn_check_version("1.10")
        else []
    ),
)
@pytest.mark.parametrize(
    "array_move_like",
    [np.arange(1), array_api_strict.arange(1)]
    + (
        [dpnp.arange(1, device="gpu")]
        if dpnp_available and is_sycl_device_available("gpu")
        else []
    )
    + (
        [torch.arange(1, device="xpu")]
        if torch_available
        and is_sycl_device_available("gpu")
        and sklearn_check_version("1.10")
        else []
    ),
)
def test_move_estimator_to(estimator, array_input_like, array_move_like, with_array_api):
    # TODO: remove this skip once issue in sklearn is fixed:
    # https://github.com/scikit-learn/scikit-learn/issues/35088
    if (
        isinstance(array_move_like, array_api_strict._array_object.Array)
        and dpnp_available
        and is_sycl_device_available("gpu")
        and isinstance(array_input_like, dpnp.ndarray)
    ):
        pytest.skip()
    from sklearnex import linear_model

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(10, 3), dtype=np.float32)
    y = rng.standard_normal(size=X.shape[0], dtype=np.float32)

    xp_in, _, device_in = get_namespace_and_device(array_input_like)
    X_in = move_to(X, xp=xp_in, device=device_in)
    y_in = move_to(y, xp=xp_in, device=device_in)

    xp_move, _, device_move = get_namespace_and_device(array_move_like)
    X_move = move_to(X, xp=xp_move, device=device_move)

    model = getattr(linear_model, estimator)().fit(X_in, y_in)
    model_moved = move_estimator_to(model, xp_move, device_move)

    assert_same_namespace(model.coef_, array_input_like)
    assert_same_namespace(model_moved.coef_, array_move_like)

    pred_orig = model.predict(X_in)
    pred_moved = model_moved.predict(X_move)

    assert_same_namespace(pred_orig, array_input_like)
    assert_same_namespace(pred_moved, array_move_like)
    np.testing.assert_allclose(_as_numpy(pred_moved), _as_numpy(pred_orig), atol=1e-6)
