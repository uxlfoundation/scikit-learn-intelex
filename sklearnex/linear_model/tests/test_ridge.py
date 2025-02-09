# ===============================================================================
# Copyright 2024 Intel Corporation
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
from sklearn.exceptions import NotFittedError

from daal4py.sklearn._utils import daal_check_version
from daal4py.sklearn.linear_model.tests.test_ridge import (
    _test_multivariate_ridge_alpha_shape,
    _test_multivariate_ridge_coefficients,
)
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


def _compute_ridge_coefficients(X, y, alpha, fit_intercept):
    """
    Computes Ridge regression coefficients for single or multiple target variables.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    alpha : float
        Regularization strength.
    fit_intercept : bool
        Whether to calculate the intercept for this model.

    Returns:
    coeffs : ndarray of shape (n_features, n_targets)
        Estimated coefficients for each target variable.
    intercept : ndarray of shape (n_targets,)
        Intercept terms for each target variable.
    """
    n_samples, n_features = X.shape
    X_copy = X.copy()
    y = np.asarray(y)

    # making single-target y also multi-dim array for consistency
    if y.ndim == 1:
        y = y[:, np.newaxis]
    n_targets = y.shape[1]

    if fit_intercept:
        # adding column of ones to X for the intercept term
        X_copy = np.hstack([np.ones((n_samples, 1)), X])
        identity_matrix = np.diag([0] + [1] * n_features)
    else:
        identity_matrix = np.eye(n_features)

    # normal equation: (X^T * X + alpha * I) * w = X^T * y
    A = X_copy.T @ X_copy + alpha * identity_matrix
    b = X_copy.T @ y
    coeffs = np.linalg.solve(A, b)

    if fit_intercept:
        intercept = coeffs[0, :]  # Shape (n_targets,)
        coeffs = coeffs[1:, :]  # Shape (n_features, n_targets)
    else:
        intercept = np.zeros((n_targets,))  # Shape (n_targets,)

    # in case of single target, flattening both coefficients and intercept
    if n_targets == 1:
        coeffs = coeffs.flatten()
        intercept = intercept[0]
    else:
        coeffs = coeffs.T

    return coeffs, intercept


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_ridge(dataframe, queue):
    from sklearnex.linear_model import Ridge

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    ridge_reg = Ridge(alpha=0.5).fit(X_c, y_c)

    if daal_check_version((2024, "P", 600)):
        assert (
            "sklearnex" in ridge_reg.__module__ and "preview" not in ridge_reg.__module__
        )
    else:
        assert "daal4py" in ridge_reg.__module__

    assert_allclose(ridge_reg.intercept_, 3.86, rtol=1e-2)
    assert_allclose(ridge_reg.coef_, [0.91, 1.64], rtol=1e-2)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("sample_size", [100, 1000])
@pytest.mark.parametrize("feature_size", [10, 50])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_coefficients(
    dataframe, queue, sample_size, feature_size, alpha, fit_intercept
):
    from sklearnex.linear_model import Ridge

    X = np.random.rand(sample_size, feature_size)
    y = np.random.rand(sample_size)
    X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    ridge_reg = Ridge(fit_intercept=fit_intercept, alpha=alpha).fit(X_c, y_c)

    coefficients_manual, intercept_manual = _compute_ridge_coefficients(
        X, y, alpha, fit_intercept=fit_intercept
    )

    assert_allclose(ridge_reg.coef_, coefficients_manual, rtol=1e-6, atol=1e-6)
    assert_allclose(ridge_reg.intercept_, intercept_manual, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not daal_check_version((2024, "P", 600)), reason="requires onedal 2024.6.0 or higher"
)
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_ridge_score_before_fit(dataframe, queue):
    from sklearnex.linear_model import Ridge

    sample_count, feature_count = 10, 5

    model = Ridge(fit_intercept=True, alpha=0.5)

    X, y = np.random.rand(sample_count, feature_count), np.random.rand(sample_count)
    X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

    with pytest.raises(NotFittedError):
        model.score(X_c, y_c)


@pytest.mark.skipif(
    not daal_check_version((2024, "P", 600)), reason="requires onedal 2024.6.0 or higher"
)
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_ridge_predict_before_fit(dataframe, queue):
    from sklearnex.linear_model import Ridge

    sample_count, feature_count = 10, 5

    model = Ridge(fit_intercept=True, alpha=0.5)

    X = np.random.rand(sample_count, feature_count)
    X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    with pytest.raises(NotFittedError):
        model.predict(X_c)


@pytest.mark.skip(reason="Forced to use original sklearn for now.")
def test_sklearnex_multivariate_ridge_coefs():
    from sklearnex.linear_model import Ridge

    _test_multivariate_ridge_coefficients(Ridge, random_state=0)


@pytest.mark.skip(reason="Forced to use original sklearn for now.")
def test_sklearnex_multivariate_ridge_alpha_shape():
    from sklearnex.linear_model import Ridge

    _test_multivariate_ridge_alpha_shape(Ridge, random_state=0)


@pytest.mark.skipif(
    not daal_check_version((2025, "P", 100)), reason="requires onedal 2025.1.0 or higher"
)
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("overdetermined", [True, False])
@pytest.mark.parametrize("alpha", [0.00001, 0.1, 1.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_overdetermined_system(
    dataframe, queue, overdetermined, alpha, fit_intercept
):
    from sklearnex.linear_model import Ridge

    if overdetermined:
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
    else:
        X = np.random.rand(10, 100)
        y = np.random.rand(10)

    X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    ridge_reg = Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(X_c, y_c)

    coefficients_manual, intercept_manual = _compute_ridge_coefficients(
        X, y, alpha, fit_intercept=fit_intercept
    )

    assert_allclose(ridge_reg.coef_, coefficients_manual, rtol=1e-6, atol=1e-6)
    assert_allclose(ridge_reg.intercept_, intercept_manual, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_multivariate_ridge_scalar_alpha(dataframe, queue, fit_intercept, alpha):
    from sklearn.datasets import make_regression

    from sklearnex.linear_model import Ridge

    X, y = make_regression(n_samples=10, n_features=3, n_targets=3, random_state=0)
    X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    ridge.fit(X_c, y_c)

    coef_manual, intercept = _compute_ridge_coefficients(X, y, alpha, fit_intercept)
    assert_allclose(ridge.coef_, coef_manual, rtol=1e-6, atol=1e-6)
    assert_allclose(ridge.intercept_, intercept, rtol=1e-6, atol=1e-6)

    predictions = _as_numpy(ridge.predict(X_c))
    predictions_manual = X @ coef_manual.T + intercept
    assert_allclose(predictions, predictions_manual, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_underdetermined_positive_alpha_ridge(dataframe, queue):
    from sklearn.datasets import make_regression

    from sklearnex.linear_model import Ridge

    X, y = make_regression(n_samples=5, n_features=6, n_targets=1, random_state=0)
    alpha = 1.0
    X_c = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_c = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    ridge = Ridge(alpha=alpha, fit_intercept=True).fit(X_c, y_c)

    coef_manual, intercept = _compute_ridge_coefficients(X, y, alpha, fit_intercept=True)

    assert_allclose(ridge.coef_, coef_manual, rtol=1e-6, atol=1e-6)
    assert_allclose(ridge.intercept_, intercept, rtol=1e-6, atol=1e-6)
