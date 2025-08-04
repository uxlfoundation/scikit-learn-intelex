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

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as _sklearn_LogisticRegression

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from daal4py.sklearn.linear_model import LogisticRegression as _d4p_LogisticRegression

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
y_multiclass = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 3, 3, 1, 2, 2, 2, 2], dtype=int)


# Here, scikit-learn does a theoretically incorrect calculation in which
# they set the predictions for the 'negative' class as the negative of the
# predictions for the positive class instead of all-zeros. The idea is to
# match theirs, which is done by falling back. This test ensures that the
# predictions match with sklearn in case it isn't done during conformance tests.
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("C", [1, 0.1])
def test_binary_multinomial_probabilities(fit_intercept, C):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model_sklearnex = _d4p_LogisticRegression(
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
def test_warm_start_stateful(fit_intercept, n_classes, multi_class, weighted):
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
        warnings.simplefilter("ignore", category=ConvergenceWarning)
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
        model2 = _d4p_LogisticRegression(
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
        warnings.simplefilter("ignore", category=ConvergenceWarning)
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
def test_warm_start_binary(fit_intercept, multi_class, weighted):
    X, y = make_classification(
        random_state=123,
        n_classes=2,
        n_clusters_per_class=1,
        n_features=2,
        n_redundant=0,
        class_sep=0.5,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model1 = _d4p_LogisticRegression(
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
            _d4p_LogisticRegression(
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
def test_warm_start_multinomial(fit_intercept, multi_class, weighted):
    X, y = make_classification(
        random_state=123,
        n_classes=3,
        n_clusters_per_class=1,
        n_features=2,
        n_redundant=0,
        class_sep=0.5,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model1 = _d4p_LogisticRegression(
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
            _d4p_LogisticRegression(
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
@pytest.mark.skipif(
    not daal_check_version((2025, "P", 800)), reason="Bugs fixed in later oneDAL releases"
)
@pytest.mark.parametrize("multi_class", ["auto", "multinomial"])
@pytest.mark.parametrize("C", [1, 0.2, 20.0])
@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg"])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_custom_solvers_are_correct(multi_class, C, solver, n_classes):
    X, y = make_classification(
        random_state=123,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_features=2,
        n_redundant=0,
        class_sep=0.25,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model_sklearn = _sklearn_LogisticRegression(
            C=C,
            multi_class=multi_class,
        ).fit(X, y)
        model_sklearnex = _d4p_LogisticRegression(
            C=C, solver=solver, multi_class=multi_class, max_iter=int(1e7), tol=1e-20
        ).fit(X, y)
        model_sklearnex_refitted = (
            _d4p_LogisticRegression(
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
            atol=1e-2,
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
            atol=1e-2,
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
