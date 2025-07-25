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

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as _sklearn_LogisticRegression

from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.linear_model import LogisticRegression as _d4p_LogisticRegression

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


# Adapted from this test:
# https://github.com/scikit-learn/scikit-learn/blob/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/utils/estimator_checks.py#L963
# Note: the logger of the daal4py estimator here might report that it falls back
# to scikit-learn, but in many cases it does so by re-defining their code instead
# of importing their classes, and passing arguments as needed. The aim of this
# test is to verify that those still work correctly, as they are not a direct
# fallback the same way it works elsewhere.
# Since the logger will report a fallback and this test is run under a hook that
# makes it fail on fallbacks from patched classes, the test here imports the
# module with an underscore to bypass the check. It doesn't appear to work
# if putting it under 'sklearnex' as by then the patched class will already be
# imported and checked.
@pytest.mark.parametrize(
    "solver",
    ["lbfgs", "newton-cg", "sag", "liblinear"]
    + (["newton-cholesky"] if sklearn_check_version("1.2") else []),
)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("C", [1, 0.1])
def test_logistic_regression_is_correct_with_weights(solver, fit_intercept, C):
    w = np.arange(X.shape[0])

    model_sklearnex = _d4p_LogisticRegression(
        C=C, solver=solver, fit_intercept=fit_intercept, random_state=123
    ).fit(X, y_binary, w)
    model_sklearn = _sklearn_LogisticRegression(
        C=C, solver=solver, fit_intercept=fit_intercept, random_state=123
    ).fit(X, y_binary, w)

    np.testing.assert_allclose(
        model_sklearnex.coef_,
        model_sklearn.coef_,
        atol=1e-5,
    )
    if fit_intercept:
        np.testing.assert_allclose(
            model_sklearnex.intercept_,
            model_sklearn.intercept_,
            atol=1e-5,
        )


@pytest.mark.parametrize(
    "solver",
    ["lbfgs", "newton-cg", "sag", "liblinear"]
    + (["newton-cholesky"] if sklearn_check_version("1.2") else []),
)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("C", [1, 0.1])
def test_multinomial_logistic_regression_is_correct_with_weights(
    solver, fit_intercept, C
):
    w = np.arange(X.shape[0])

    model_sklearnex = _d4p_LogisticRegression(
        C=C, solver=solver, fit_intercept=fit_intercept, random_state=123
    ).fit(X, y_multiclass, w)
    model_sklearn = _sklearn_LogisticRegression(
        C=C, solver=solver, fit_intercept=fit_intercept, random_state=123
    ).fit(X, y_multiclass, w)

    np.testing.assert_allclose(
        model_sklearnex.coef_,
        model_sklearn.coef_,
        atol=1e-4,
    )
    if fit_intercept:
        np.testing.assert_allclose(
            model_sklearnex.intercept_,
            model_sklearn.intercept_,
            atol=1e-3,
        )


# Here, scikit-learn does a theoretically incorrect calculation in which
# they set the predictions for the 'negative' class as the negative of the
# predictions for the positive class instead of all-zeros. The idea is to
# match theirs, which is done by falling back. This test ensures that the
# predictions match with sklearn in case it isn't done during conformance tests.
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("C", [1, 0.1])
def test_binary_multinomial_probabilities(fit_intercept, C):
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
