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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
@pytest.mark.parametrize("solver", ["lbfgs", "sag", "saga"])
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("multi_class", ["auto", "multinomial"])
@pytest.mark.parametrize("weighted", [False, True])
def test_warm_start_stateful(fit_intercept, solver, n_classes, multi_class, weighted):
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
            random_state=123,
            solver=solver,
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
            random_state=123,
            solver=solver,
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
            intercepts1 = model1.intercept_ - model1.intercept_.sum()
            intercepts2 = model2.intercept_ - model2.intercept_.sum()
            np.testing.assert_allclose(intercepts1, intercepts2)


# Note: other solvers do not have any internal state and are supposed to yield the
# same result after one iteration given the current coefficients.
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("multi_class", ["ovr", "multinomial"])
@pytest.mark.parametrize(
    "solver",
    ["newton-cg"] + (["newton-cholesky"] if sklearn_check_version("1.2") else []),
)
@pytest.mark.parametrize("weighted", [False, True])
def test_warm_start_binary(fit_intercept, multi_class, solver, weighted):
    X, y = make_classification(
        random_state=123,
        n_classes=2,
        n_clusters_per_class=1,
        n_features=2,
        n_redundant=0,
        class_sep=0.5,
    )

    # Note: scikit-learn itself has bugs that would make it fail this test
    # under some versions but not others. Hence, before doing the test, this
    # first checks that it would work correctly in scikit-learn.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model1_skl = _sklearn_LogisticRegression(
                random_state=123,
                solver=solver,
                fit_intercept=fit_intercept,
                multi_class=multi_class,
                max_iter=2,
            ).fit(
                X,
                y,
                np.ones(X.shape[0]) if weighted else None,
            )
        except ValueError:
            pytest.skip("Functionality introduced in a later scikit-learn version.")
        model2_skl = (
            _sklearn_LogisticRegression(
                random_state=123,
                solver=solver,
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

    skl_works = True
    try:
        np.testing.assert_allclose(model1_skl.coef_, model2_skl.coef_)
    except AssertionError:
        skl_works = False
    if skl_works and fit_intercept:
        try:
            np.testing.assert_allclose(model1_skl.intercept_, model2_skl.intercept_)
        except AssertionError:
            skl_works = False

    if not skl_works:
        pytest.skip("Bug in scikit-learn in the functionality being tested")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model1 = _d4p_LogisticRegression(
            random_state=123,
            solver=solver,
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
                solver=solver,
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
@pytest.mark.parametrize(
    "solver",
    ["newton-cg"] + (["newton-cholesky"] if sklearn_check_version("1.2") else []),
)
@pytest.mark.parametrize("weighted", [False, True])
def test_warm_start_multinomial(fit_intercept, multi_class, solver, weighted):
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
        try:
            model1_skl = _sklearn_LogisticRegression(
                random_state=123,
                solver=solver,
                fit_intercept=fit_intercept,
                multi_class=multi_class,
                max_iter=2,
            ).fit(
                X,
                y,
                np.ones(X.shape[0]) if weighted else None,
            )
        except ValueError:
            pytest.skip("Functionality introduced in a later scikit-learn version.")
        model2_skl = (
            _sklearn_LogisticRegression(
                random_state=123,
                solver=solver,
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
    skl_works = True
    try:
        np.testing.assert_allclose(model1_skl.coef_, model2_skl.coef_)
    except AssertionError:
        skl_works = False
    if skl_works and fit_intercept:
        try:
            np.testing.assert_allclose(model1_skl.intercept_, model2_skl.intercept_)
        except AssertionError:
            skl_works = False

    if not skl_works:
        pytest.skip("Bug in scikit-learn in the functionality being tested")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model1 = _d4p_LogisticRegression(
            random_state=123,
            solver=solver,
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
                solver=solver,
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
        intercepts1 = model1.intercept_ - model1.intercept_.sum()
        intercepts2 = model2.intercept_ - model2.intercept_.sum()
        np.testing.assert_allclose(intercepts1, intercepts2)


# This is a bit different from the others - it just aims to test that it
# is processing the regularization correctly under all circumstances, and
# that it is not multiplying or dividing the coefficients by two when it
# shouldn't do it.
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
        warnings.simplefilter("ignore")
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
            model_sklearnex.intercept_ - model_sklearnex.intercept_.sum(),
            model_sklearn.intercept_ - model_sklearn.intercept_.sum(),
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
            - model_sklearnex_refitted.intercept_.sum(),
            model_sklearn.intercept_ - model_sklearn.intercept_.sum(),
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
