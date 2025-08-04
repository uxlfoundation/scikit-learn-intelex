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
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.linear_model import LogisticRegression as _sklearn_LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
    get_queues,
)
from sklearnex import config_context


def prepare_input(X, y, dataframe, queue):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )
    X_train = _convert_to_dataframe(X_train, sycl_queue=queue, target_df=dataframe)
    y_train = _convert_to_dataframe(y_train, sycl_queue=queue, target_df=dataframe)
    X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)
    return X_train, X_test, y_train, y_test


@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues(device_filter_="cpu")
)
def test_sklearnex_multiclass_classification(dataframe, queue):
    from sklearnex.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = prepare_input(X, y, dataframe, queue=queue)

    logreg = LogisticRegression(fit_intercept=True, solver="lbfgs", max_iter=200).fit(
        X_train, y_train
    )

    if daal_check_version((2024, "P", 1)):
        assert "sklearnex" in logreg.__module__
    else:
        assert "daal4py" in logreg.__module__

    y_pred = _as_numpy(logreg.predict(X_test))
    assert accuracy_score(y_test, y_pred) > 0.99


@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(),
)
def test_sklearnex_binary_classification(dataframe, queue):
    from sklearnex.linear_model import LogisticRegression

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = prepare_input(X, y, dataframe, queue=queue)

    logreg = LogisticRegression(fit_intercept=True, solver="newton-cg", max_iter=100).fit(
        X_train, y_train
    )

    if daal_check_version((2024, "P", 1)):
        assert "sklearnex" in logreg.__module__
    else:
        assert "daal4py" in logreg.__module__
    if (
        dataframe != "numpy"
        and queue is not None
        and queue.sycl_device.is_gpu
        and daal_check_version((2024, "P", 1))
    ):
        # fit was done on gpu
        assert hasattr(logreg, "_onedal_estimator")

    y_pred = _as_numpy(logreg.predict(X_test))
    assert accuracy_score(y_test, y_pred) > 0.95


if daal_check_version((2024, "P", 700)):

    @pytest.mark.parametrize("queue", get_queues("gpu"))
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        "dims", [(3007, 17, 0.05), (50000, 100, 0.01), (512, 10, 0.5)]
    )
    def test_csr(queue, dtype, dims):
        from sklearnex.linear_model import LogisticRegression

        n, p, density = dims

        # Create sparse dataset for classification
        X, y = make_classification(n, p, random_state=42)
        X = X.astype(dtype)
        y = y.astype(dtype)
        np.random.seed(2007 + n + p)
        mask = np.random.binomial(1, density, (n, p))
        X = X * mask
        X_sp = csr_matrix(X)

        model = LogisticRegression(fit_intercept=True, solver="newton-cg")
        model_sp = LogisticRegression(fit_intercept=True, solver="newton-cg")

        with config_context(target_offload="gpu:0"):
            model.fit(X, y)
            pred = model.predict(X)
            prob = model.predict_proba(X)
            model_sp.fit(X_sp, y)
            pred_sp = model_sp.predict(X_sp)
            prob_sp = model_sp.predict_proba(X_sp)

        rtol = 2e-4
        assert_allclose(pred, pred_sp, rtol=rtol)
        assert_allclose(prob, prob_sp, rtol=rtol)
        assert_allclose(model.coef_, model_sp.coef_, rtol=rtol)
        assert_allclose(model.intercept_, model_sp.intercept_, rtol=rtol)


# Note: this is adapted from a test in scikit-learn:
# https://github.com/scikit-learn/scikit-learn/blob/9b7a86fb6d45905eec7b7afd01d3ae32643c8180/sklearn/linear_model/tests/test_logistic.py#L1494
# Here we don't always expect them to match exactly due to differences in numerical precision
# and how each library deals with large/small numbers, but oftentimes the results from oneDAL
# end up being better in terms of resulting function values (for the objective function being
# minimized), hence this test will try to look at function values if coefficients aren't
# sufficiently similar.
def test_logistic_regression_is_correct():
    from sklearnex.linear_model import LogisticRegression

    X = np.array([[-1, 0], [0, 1], [1, 1]])
    y = np.array([0, 1, 1])
    C = 3.0
    model_sklearn = _sklearn_LogisticRegression(C=C).fit(X, y)
    model_sklearnex = LogisticRegression(C=C).fit(X, y)

    try:
        np.testing.assert_allclose(model_sklearnex.coef_, model_sklearn.coef_)
        np.testing.assert_allclose(model_sklearnex.intercept_, model_sklearn.intercept_)
    except AssertionError:

        def logistic_model_function(predicted_probabilities, coefs):
            neg_log_likelihood = X.shape[0] * log_loss(y, predicted_probabilities)
            sum_squares_coefs = np.dot(coefs.reshape(-1), coefs.reshape(-1))
            return C * neg_log_likelihood + 0.5 * sum_squares_coefs

        fn_sklearn = logistic_model_function(
            model_sklearn.predict_proba(X)[:, 1], model_sklearn.coef_
        )
        fn_sklearnex = logistic_model_function(
            model_sklearnex.predict_proba(X)[:, 1], model_sklearnex.coef_
        )
        assert fn_sklearnex <= fn_sklearn


def test_multinomial_logistic_regression_is_correct():
    from sklearnex.linear_model import LogisticRegression

    X = np.array([[-1, 0], [0, 1], [1, 1]])
    y = np.array([2, 1, 0])
    C = 3.0
    model_sklearn = _sklearn_LogisticRegression(C=C, multi_class="multinomial").fit(X, y)
    model_sklearnex = LogisticRegression(C=C, multi_class="multinomial").fit(X, y)

    try:
        np.testing.assert_allclose(model_sklearnex.coef_, model_sklearn.coef_)
        np.testing.assert_allclose(model_sklearnex.intercept_, model_sklearn.intercept_)
    except AssertionError:

        def logistic_model_function(predicted_probabilities, coefs):
            neg_log_likelihood = X.shape[0] * log_loss(y, predicted_probabilities)
            sum_squares_coefs = np.dot(coefs.reshape(-1), coefs.reshape(-1))
            return C * neg_log_likelihood + 0.5 * sum_squares_coefs

        fn_sklearn = logistic_model_function(
            model_sklearn.predict_proba(X), model_sklearn.coef_
        )
        fn_sklearnex = logistic_model_function(
            model_sklearnex.predict_proba(X), model_sklearnex.coef_
        )
        assert fn_sklearnex <= fn_sklearn


@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues(device_filter_="gpu")
)
def test_gpu_logreg_prediction_shapes(dataframe, queue):
    if not queue or not queue.sycl_device.is_gpu:
        pytest.skip("Test for GPU-only code branch")
    from sklearnex.linear_model import LogisticRegression

    X, y = make_classification(random_state=123)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

    model = LogisticRegression(solver="newton-cg").fit(X, y)
    pred = model.predict(X)
    pred_proba = model.predict_proba(X)
    pred_log_proba = model.predict_log_proba(X)

    np.testing.assert_array_equal(pred.shape, (X.shape[0],))
    np.testing.assert_array_equal(pred_proba.shape, (X.shape[0], 2))
    np.testing.assert_array_equal(pred_log_proba.shape, (X.shape[0], 2))
