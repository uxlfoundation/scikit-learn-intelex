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

from daal4py.sklearn._utils import sklearn_check_version


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
@pytest.mark.parametrize("dim_y", [1, 3])
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
