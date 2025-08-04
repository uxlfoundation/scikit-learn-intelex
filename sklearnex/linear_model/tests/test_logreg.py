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

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        logreg = LogisticRegression(
            fit_intercept=True, solver="newton-cg", max_iter=100
        ).fit(X_train, y_train)

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        model_sklearn = _sklearn_LogisticRegression(C=C, multi_class="multinomial").fit(
            X, y
        )
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


# Here, scikit-learn does a theoretically incorrect calculation in which
# they set the predictions for the 'negative' class as the negative of the
# predictions for the positive class instead of all-zeros. The idea is to
# match theirs, which is done by falling back. This test ensures that the
# predictions match with sklearn in case it isn't done during conformance tests.
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("C", [1, 0.1])
@pytest.mark.allow_sklearn_fallback
def test_binary_multinomial_probabilities(fit_intercept, C):
    from sklearnex.linear_model import LogisticRegression

    # Adapted from this test:
    # https://github.com/scikit-learn/scikit-learn/blob/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/utils/estimator_checks.py#L963
    X = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.float64,
    )
    y_binary = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_sklearnex = LogisticRegression(
            C=C, fit_intercept=fit_intercept, multi_class="multinomial"
        ).fit(X, y_binary)
        model_sklearn = _sklearn_LogisticRegression(
            C=C, fit_intercept=fit_intercept, multi_class="multinomial"
        ).fit(X, y_binary)
    np.testing.assert_allclose(
        model_sklearnex.predict_proba(X),
        model_sklearn.predict_proba(X),
        rtol=1e-2,
        atol=1e-3,
    )


# Note: some solvers have an internal state, such as previous gradients,
# which is not preserved across warm starts and which influences the
# optimization routines. For these, a warm-started call with the coefficients
# from a previous iterations will not be equal to a cold-start call with
# one more iteration.
# Note2: usually, passing weights will cause the procedure to fall back to
# stock scikit-learn. We want to check that fallbacks also handle warm starts
# correctly when falling back.
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("multi_class", ["auto", "multinomial"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.allow_sklearn_fallback
def test_warm_start_stateful(fit_intercept, n_classes, multi_class, weighted):
    from sklearnex.linear_model import LogisticRegression

    X, y = make_classification(
        random_state=123,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_features=2,
        n_redundant=0,
        # Note: oneDAL and scikit-learn deal with large numbers differently
        # in the calculations, so when comparing against sklearn, we want
        # to avoid ending up with large numbers in the computations.
        class_sep=0.25,
    )

    # Note1: these will throw warnings due to reaching the maximum
    # number of iterations without converging, which is expected
    # given that those are being severely limited for the tests.
    # Note2: this will first compare the results after one iteration, and
    # if those already differ too much (which can be the case given numerical
    # differences), will then skip the rest of test that checks the warm starts.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model1 = _sklearn_LogisticRegression(
            solver="lbfgs",
            fit_intercept=fit_intercept,
            multi_class=multi_class,
            max_iter=1,
            warm_start=True,
        ).fit(
            X,
            y,
            np.ones(X.shape[0]) if weighted else None,
        )
        model2 = LogisticRegression(
            solver="lbfgs",
            fit_intercept=fit_intercept,
            multi_class=multi_class,
            max_iter=1,
            warm_start=True,
        ).fit(
            X,
            y,
            np.ones(X.shape[0]) if weighted else None,
        )

    try:
        np.testing.assert_allclose(model1.coef_, model2.coef_)
    except AssertionError:
        pytest.skip("Too large numerical differences for further comparisons")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model1.fit(
            X,
            y,
            np.ones(X.shape[0]) if weighted else None,
        )
        model2.fit(
            X,
            y,
            np.ones(X.shape[0]) if weighted else None,
        )

    np.testing.assert_allclose(model1.coef_, model2.coef_)
    if fit_intercept:
        if n_classes == 2:
            np.testing.assert_allclose(model1.intercept_, model2.intercept_)
        else:
            # Note: softmax function is invariable to shifting by a constant
            intercepts1 = model1.intercept_ - model1.intercept_.mean()
            intercepts2 = model2.intercept_ - model2.intercept_.mean()
            np.testing.assert_allclose(intercepts1, intercepts2)


# Note: other solvers do not have any internal state and are supposed to yield the
# same result after one iteration given the current coefficients.
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("multi_class", ["ovr", "multinomial"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.allow_sklearn_fallback
def test_warm_start_binary(fit_intercept, multi_class, weighted):
    from sklearnex.linear_model import LogisticRegression

    X, y = make_classification(
        random_state=123,
        n_classes=2,
        n_clusters_per_class=1,
        n_features=2,
        n_redundant=0,
        class_sep=0.5,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model1 = LogisticRegression(
            random_state=123,
            solver="newton-cg",
            fit_intercept=fit_intercept,
            multi_class=multi_class,
            max_iter=2,
        ).fit(
            X,
            y,
            np.ones(X.shape[0]) if weighted else None,
        )
        model2 = (
            LogisticRegression(
                random_state=123,
                solver="newton-cg",
                fit_intercept=fit_intercept,
                multi_class=multi_class,
                max_iter=1,
                warm_start=True,
            )
            .fit(
                X,
                y,
                np.ones(X.shape[0]) if weighted else None,
            )
            .fit(
                X,
                y,
                np.ones(X.shape[0]) if weighted else None,
            )
        )

    np.testing.assert_allclose(model1.coef_, model2.coef_)
    if fit_intercept:
        np.testing.assert_allclose(model1.intercept_, model2.intercept_)


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("multi_class", ["ovr", "multinomial"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.allow_sklearn_fallback
def test_warm_start_multinomial(fit_intercept, multi_class, weighted):
    from sklearnex.linear_model import LogisticRegression

    X, y = make_classification(
        random_state=123,
        n_classes=3,
        n_clusters_per_class=1,
        n_features=2,
        n_redundant=0,
        class_sep=0.5,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model1 = LogisticRegression(
            random_state=123,
            solver="newton-cg",
            fit_intercept=fit_intercept,
            multi_class=multi_class,
            max_iter=2,
        ).fit(
            X,
            y,
            np.ones(X.shape[0]) if weighted else None,
        )
        model2 = (
            LogisticRegression(
                random_state=123,
                solver="newton-cg",
                fit_intercept=fit_intercept,
                multi_class=multi_class,
                max_iter=1,
                warm_start=True,
            )
            .fit(
                X,
                y,
                np.ones(X.shape[0]) if weighted else None,
            )
            .fit(
                X,
                y,
                np.ones(X.shape[0]) if weighted else None,
            )
        )

    np.testing.assert_allclose(model1.coef_, model2.coef_)
    if fit_intercept:
        # Note: softmax function is invariable to shifting by a constant
        intercepts1 = model1.intercept_ - model1.intercept_.mean()
        intercepts2 = model2.intercept_ - model2.intercept_.mean()
        np.testing.assert_allclose(intercepts1, intercepts2)


# This is a bit different from the others - it just aims to test that it
# is processing the regularization correctly under all circumstances, and
# that it is not multiplying or dividing the coefficients by two when it
# shouldn't do it.
# It has some overlap with the tests at the beginning, but it is only tested
# with oneDAL>=2025.8. After that version has been released and CIs updated
# to use it, the earlier tests 'test_logistic_regression_is_correct' and
# 'test_multinomial_logistic_regression_is_correct' can be removed.
@pytest.mark.skipif(
    not daal_check_version((2025, "P", 800)), reason="Bugs fixed in later oneDAL releases"
)
@pytest.mark.parametrize("multi_class", ["auto", "multinomial"])
@pytest.mark.parametrize("C", [1, 0.2, 20.0])
@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg"])
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.allow_sklearn_fallback
def test_custom_solvers_are_correct(multi_class, C, solver, n_classes):
    from sklearnex.linear_model import LogisticRegression

    X, y = make_classification(
        random_state=123,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_features=2,
        n_redundant=0,
        class_sep=0.25,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_sklearn = _sklearn_LogisticRegression(
            C=C,
            multi_class=multi_class,
        ).fit(X, y)
        model_sklearnex = LogisticRegression(
            C=C, solver=solver, multi_class=multi_class, max_iter=int(1e7), tol=1e-20
        ).fit(X, y)
        model_sklearnex_refitted = (
            LogisticRegression(
                C=C,
                solver=solver,
                multi_class=multi_class,
                max_iter=int(1e7),
                tol=1e-20,
                warm_start=True,
            )
            .fit(X, y)
            .fit(X, y)
        )

    np.testing.assert_allclose(
        model_sklearnex.coef_, model_sklearn.coef_, rtol=1e-3, atol=5e-3
    )
    if n_classes == 2:
        np.testing.assert_allclose(
            model_sklearnex.intercept_, model_sklearn.intercept_, atol=1e-3
        )
    else:
        np.testing.assert_allclose(
            model_sklearnex.intercept_ - model_sklearnex.intercept_.mean(),
            model_sklearn.intercept_ - model_sklearn.intercept_.mean(),
            atol=1e-3,
        )
    np.testing.assert_allclose(
        model_sklearnex_refitted.coef_, model_sklearn.coef_, rtol=1e-3, atol=5e-3
    )
    if n_classes == 2:
        np.testing.assert_allclose(
            model_sklearnex_refitted.intercept_, model_sklearn.intercept_, atol=1e-3
        )
    else:
        np.testing.assert_allclose(
            model_sklearnex_refitted.intercept_
            - model_sklearnex_refitted.intercept_.mean(),
            model_sklearn.intercept_ - model_sklearn.intercept_.mean(),
            atol=1e-3,
        )

    np.testing.assert_allclose(
        model_sklearnex.predict_proba(X),
        model_sklearn.predict_proba(X),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        model_sklearnex_refitted.predict_proba(X),
        model_sklearn.predict_proba(X),
        rtol=1e-3,
        atol=1e-3,
    )
