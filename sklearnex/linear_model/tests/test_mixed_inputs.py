# ===============================================================================
# Copyright Contributors to the oneDAL Project
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
import pandas as pd
import pytest

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    dpnp_available,
    torch_available,
    torch_xpu_available,
)
from onedal.tests.utils._device_selection import is_sycl_device_available

if dpnp_available:
    import dpnp
if torch_available:
    import torch

from sklearnex.linear_model import (
    IncrementalLinearRegression,
    IncrementalRidge,
    LinearRegression,
    Ridge,
)

non_incremental_estimators = [LinearRegression, Ridge]
incremental_estimators = [IncrementalLinearRegression, IncrementalRidge]
all_estimators = non_incremental_estimators + incremental_estimators


def _expected_type(X, X_xp):
    """Predict/attributes follow X, but pandas inputs produce numpy outputs."""
    return np.ndarray if X_xp is pd else X.__class__


def _split_row(arr, start, stop):
    if hasattr(arr, "iloc"):
        return arr.iloc[start:stop]
    if arr.ndim == 1:
        return arr[start:stop]
    return arr[start:stop, ...]


def _fit_model(model, X, y, use_partial_fit):
    if use_partial_fit:
        half = X.shape[0] // 2
        model.partial_fit(_split_row(X, None, half), _split_row(y, None, half))
        model.partial_fit(_split_row(X, half, None), _split_row(y, half, None))
    else:
        model.fit(X, y)


def _generate_data():
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(50, 4))
    y = rng.standard_normal(size=X.shape[0])
    return X, y


def _convert(arr, xp, device=None):
    if xp is pd:
        return pd.DataFrame(arr) if arr.ndim == 2 else pd.Series(arr)
    return xp.asarray(arr) if device is None else xp.asarray(arr, device=device)


def _check_attributes_and_output(model, X, y, X_xp):
    expected = _expected_type(X, X_xp)
    expected_device = getattr(X, "device", None)
    assert isinstance(model.coef_, expected)
    assert isinstance(model.intercept_, (expected, float, np.floating))
    if X_xp is not pd:
        assert getattr(model.coef_, "device", None) == expected_device
        assert getattr(model.intercept_, "device", None) == expected_device

    pred = model.predict(X)
    assert pred.__class__ == expected
    if X_xp is not pd:
        assert getattr(pred, "device", None) == expected_device

    score = model.score(X, y)
    assert isinstance(score, (float, np.floating))


@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later scikit-learn versions.",
)
@pytest.mark.parametrize("X_xp", [np, pd, array_api_strict])
@pytest.mark.parametrize("y_xp", [np, pd, array_api_strict])
@pytest.mark.parametrize("estimator_class", all_estimators)
@pytest.mark.parametrize("use_partial_fit", [False, True])
def test_linreg_mixed_array_namespaces(
    X_xp, y_xp, estimator_class, use_partial_fit, with_array_api
):
    if X_xp is None or y_xp is None:
        pytest.skip("array_api_strict not available")
    if use_partial_fit and estimator_class in non_incremental_estimators:
        pytest.skip("partial_fit only for incremental estimators")

    X, y = _generate_data()
    X = _convert(X, X_xp)
    y = _convert(y, y_xp)

    model = estimator_class()
    _fit_model(model, X, y, use_partial_fit)
    _check_attributes_and_output(model, X, y, X_xp)


@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later scikit-learn versions.",
)
@pytest.mark.parametrize("estimator_class", all_estimators)
@pytest.mark.parametrize("use_partial_fit", [False, True])
def test_error_on_incompatible_namespaces(
    estimator_class, use_partial_fit, with_array_api
):
    if use_partial_fit and estimator_class in non_incremental_estimators:
        pytest.skip("partial_fit only for incremental estimators")

    X, y = _generate_data()
    Xa = array_api_strict.from_dlpack(X)
    ya = array_api_strict.from_dlpack(y)

    model = estimator_class()
    _fit_model(model, X, y, use_partial_fit)

    with pytest.raises(ValueError, match="same namespace"):
        model.predict(Xa)
    with pytest.raises(ValueError, match="same namespace"):
        model.score(Xa, ya)

    model = estimator_class()
    _fit_model(model, Xa, ya, use_partial_fit)

    with pytest.raises(ValueError, match="same namespace"):
        model.predict(X)
    with pytest.raises(ValueError, match="same namespace"):
        model.score(X, y)


@pytest.mark.skipif(
    not is_sycl_device_available("gpu"),
    reason="Test checks GPU-specific functionality.",
)
@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later scikit-learn versions.",
)
@pytest.mark.parametrize(
    "X_xp, X_device",
    ([(torch, "xpu"), (torch, "cpu")] if torch_xpu_available else [])
    + ([(dpnp, "gpu"), (dpnp, "cpu")] if dpnp_available else []),
)
@pytest.mark.parametrize(
    "y_xp, y_device",
    ([(torch, "xpu"), (torch, "cpu")] if torch_xpu_available else [])
    + ([(dpnp, "gpu"), (dpnp, "cpu")] if dpnp_available else [])
    + [(pd, None)],
)
@pytest.mark.parametrize("estimator_class", all_estimators)
@pytest.mark.parametrize("use_partial_fit", [False, True])
def test_linreg_mixed_devices(
    X_xp, y_xp, X_device, y_device, estimator_class, use_partial_fit, with_array_api
):
    if use_partial_fit and estimator_class in non_incremental_estimators:
        pytest.skip("partial_fit only for incremental estimators")

    X, y = _generate_data()
    X = _convert(X, X_xp, X_device)
    y = _convert(y, y_xp, y_device)

    model = estimator_class()
    _fit_model(model, X, y, use_partial_fit)
    _check_attributes_and_output(model, X, y, X_xp)
