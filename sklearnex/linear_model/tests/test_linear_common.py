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
import sys
import time
import warnings
from threading import Barrier, Lock, Thread

import array_api_strict
import numpy as np
import pytest

from daal4py.sklearn._utils import (
    _package_check_version,
    daal_check_version,
    sklearn_check_version,
)
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    dpnp_available,
    dpnp_to_strict_working,
    torch_available,
    torch_xpu_available,
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

    from sklearnex.tests.utils.misc import (
        assert_public_attributes_are_in_namespace,
        assert_same_attributes,
        assert_same_namespace,
    )


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
@pytest.mark.parametrize(
    "estimator",
    ["LinearRegression", "Ridge", "IncrementalLinearRegression", "IncrementalRidge"],
)
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
        if torch_available and torch_xpu_available and sklearn_check_version("1.10")
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
        if torch_available and torch_xpu_available and sklearn_check_version("1.10")
        else []
    ),
)
@pytest.mark.parametrize("num_targets", [1, 2])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_move_estimator_to(
    estimator,
    array_input_like,
    array_move_like,
    num_targets,
    fit_intercept,
    with_array_api,
):
    from sklearnex import linear_model

    # TODO: remove this skip once issue in sklearn is fixed:
    # https://github.com/scikit-learn/scikit-learn/issues/35088
    if (
        isinstance(array_move_like, array_api_strict._array_object.Array)
        and dpnp_available
        and isinstance(array_input_like, dpnp.ndarray)
        and not dpnp_to_strict_working
    ):
        pytest.skip()

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(20, 3), dtype=np.float32)
    y = rng.standard_normal(
        size=X.shape[0] if num_targets == 1 else (X.shape[0], num_targets),
        dtype=np.float32,
    )

    xp_in, _, device_in = get_namespace_and_device(array_input_like)
    X_in = move_to(X, xp=xp_in, device=device_in)
    y_in = move_to(y, xp=xp_in, device=device_in)

    xp_move, _, device_move = get_namespace_and_device(array_move_like)
    X_move = move_to(X, xp=xp_move, device=device_move)

    model = getattr(linear_model, estimator)(fit_intercept=fit_intercept).fit(
        X_in[:10, :], y_in[:10, ...]
    )
    pred_orig = model.predict(X_in)

    # Setting parameters after a fit should not affect the moved estimator
    model.set_params(fit_intercept=False)

    model_moved = move_estimator_to(model, xp_move, device_move)

    assert_same_attributes(model_moved, model)
    assert_public_attributes_are_in_namespace(model, xp_in, device_in)
    assert_public_attributes_are_in_namespace(model_moved, xp_move, device_move)

    pred_moved = model_moved.predict(X_move)
    assert_same_namespace(pred_orig, array_input_like)
    assert_same_namespace(pred_moved, array_move_like)
    np.testing.assert_allclose(_as_numpy(pred_moved), _as_numpy(pred_orig), atol=1e-6)

    pred_orig_after_move = model.predict(X_in)
    assert_same_namespace(pred_orig_after_move, array_input_like)
    np.testing.assert_allclose(
        _as_numpy(pred_orig_after_move), _as_numpy(pred_orig), atol=1e-6
    )

    # Warning is thrown whenever oneDAL hyperparameters (like block sizes)
    # are accessed through Python
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        attrs_orig = dir(model)
        attrs_moved = dir(model_moved)
    for attr in attrs_orig:
        assert attr in attrs_moved

    # After the move, the oneDAL object and the sklearnex object should both
    # share the same memory for the coefficients, which involves creating a
    # new array after the move to hold the 'packed' form with the intercepts.
    if isinstance(array_move_like, np.ndarray):
        ptr_coefs_sklearnex = model_moved.coef_.__array_interface__["data"][0]
        ptr_coefs_onedal = model_moved._onedal_estimator.coef_.__array_interface__[
            "data"
        ][0]
        assert ptr_coefs_sklearnex == ptr_coefs_onedal

        # When both the coefficients and intercepts are arrays,
        # they should be allocated in a single memory chunk as
        # part of a broader oneDAL array
        if num_targets > 1:
            ptr_intercept = model_moved.intercept_.__array_interface__["data"][0]
            assert (
                ptr_coefs_sklearnex
                == ptr_intercept + model_moved.intercept_.dtype.itemsize
            )

    if estimator.startswith("Incremental"):

        # Partial fits should still work, and produce independent arrays
        # of coefficients and intercepts
        model_ref = getattr(linear_model, estimator)(fit_intercept=fit_intercept).fit(
            X, y
        )
        model.partial_fit(X_in[10:, :], y_in[10:, ...])
        model_moved.partial_fit(X_move[10:, :], y_in[10:, ...])
        pref_ref = model_ref.predict(X)
        pred_orig = model.predict(X_in)
        pred_moved = model_moved.predict(X_move)

        if isinstance(array_input_like, np.ndarray) and isinstance(
            array_move_like, np.ndarray
        ):
            ptr_orig = model.coef_.__array_interface__["data"][0]
            ptr_moved = model_moved.coef_.__array_interface__["data"][0]
            assert ptr_orig != ptr_moved

        assert_same_namespace(pred_orig, array_input_like)
        assert_same_namespace(pred_moved, array_move_like)

        # Note: previous versions of oneDAL had a bug that made the tests fail
        if daal_check_version((2026, "P", 300)):
            np.testing.assert_allclose(
                _as_numpy(pred_orig), _as_numpy(pref_ref), atol=1e-6
            )
            np.testing.assert_allclose(
                _as_numpy(pred_moved), _as_numpy(pref_ref), atol=1e-6
            )


@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize(
    "estimator,sklearn_estimator",
    [
        ("LinearRegression", "LinearRegression"),
        ("Ridge", "Ridge"),
        # Incremental estimators are not in scikit-learn and do no
        # strictly need to conform to all patterns there.
        ("IncrementalLinearRegression", "LinearRegression"),
        ("IncrementalRidge", "LinearRegression"),
    ],
)
@pytest.mark.parametrize("X_shape", [(10, 1), (10, 2)])
@pytest.mark.parametrize("y_shape", [10, (10, 2), (10, 1)])
@pytest.mark.parametrize("fit_intercept", [True])  # TODO: make sklearnex use 'float' type
def test_coefs_shape_matches_sklearn(
    estimator, sklearn_estimator, X_shape, y_shape, fit_intercept
):
    from sklearn import linear_model as sklearn_linear_model

    from sklearnex import linear_model

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=X_shape)
    y = rng.standard_normal(size=y_shape)

    model = getattr(linear_model, estimator)(fit_intercept=fit_intercept).fit(X, y)
    sklearn_model = getattr(sklearn_linear_model, sklearn_estimator)(
        fit_intercept=fit_intercept
    ).fit(X, y)

    if not estimator.startswith("Incremental"):
        w = np.ones(X.shape[0])
        fallback_model = getattr(linear_model, estimator)(
            fit_intercept=fit_intercept
        ).fit(X, y, w)
        assert not hasattr(fallback_model, "_onedal_estimator")
        pred_fallback = fallback_model.predict(X)
        assert hasattr(fallback_model, "_onedal_estimator")

    np.testing.assert_array_equal(model.coef_.shape, sklearn_model.coef_.shape)
    if not hasattr(sklearn_model.intercept_, "shape"):
        assert not hasattr(model.intercept_, "shape")
    else:
        np.testing.assert_array_equal(
            model.intercept_.shape, sklearn_model.intercept_.shape
        )

    pred = model.predict(X)
    pred_sklearn = sklearn_model.predict(X)
    np.testing.assert_array_equal(pred.shape, pred_sklearn.shape)

    if not estimator.startswith("Incremental"):
        np.testing.assert_array_equal(
            fallback_model.coef_.shape, sklearn_model.coef_.shape
        )
        if not hasattr(sklearn_model.intercept_, "shape"):
            assert not hasattr(fallback_model.intercept_, "shape")
        else:
            np.testing.assert_array_equal(
                fallback_model.intercept_.shape, sklearn_model.intercept_.shape
            )
        np.testing.assert_array_equal(pred_fallback.shape, pred_sklearn.shape)
        np.testing.assert_allclose(pred_fallback, pred_sklearn)

    if sklearn_check_version("1.9"):
        xp, _, device = get_namespace_and_device(X)
        model_moved = move_estimator_to(model, xp=xp, device=device)
        sklearn_model_moved = move_estimator_to(sklearn_model, xp=xp, device=device)
        np.testing.assert_array_equal(model_moved.coef_.shape, sklearn_model.coef_.shape)
        if not hasattr(sklearn_model_moved.intercept_, "shape"):
            assert not hasattr(model_moved.intercept_, "shape")
        else:
            np.testing.assert_array_equal(
                model_moved.intercept_.shape, sklearn_model_moved.intercept_.shape
            )

        pred = model_moved.predict(X)
        pred_sklearn = sklearn_model_moved.predict(X)
        np.testing.assert_array_equal(pred.shape, pred_sklearn.shape)


@pytest.mark.parametrize(
    "estimator",
    ["LinearRegression", "Ridge", "IncrementalLinearRegression", "IncrementalRidge"],
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_manually_setting_coefficients(estimator, fit_intercept):
    from sklearnex import linear_model

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(10, 3))
    y = rng.standard_normal(size=X.shape[0])

    model = getattr(linear_model, estimator)(fit_intercept=fit_intercept).fit(X, y)
    model.coef_ = model.coef_ * 2
    model.intercept_ = model.intercept_ * 2

    pred_mult2 = model.predict(X)

    model_fresh = getattr(linear_model, estimator)(fit_intercept=fit_intercept).fit(X, y)
    pred = model_fresh.predict(X)

    np.testing.assert_almost_equal(pred_mult2, pred * 2)


@pytest.mark.parametrize(
    "estimator",
    ["LinearRegression", "Ridge", "IncrementalLinearRegression", "IncrementalRidge"],
)
def test_manually_setting_intercept(estimator):
    from sklearnex import linear_model

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(10, 3))
    y = rng.standard_normal(size=X.shape[0])

    model = getattr(linear_model, estimator)(fit_intercept=False).fit(X, y)
    if estimator.startswith("Incremental"):
        with pytest.raises(ValueError):
            model.intercept_ = model.intercept_ + 1
        return

    model.intercept_ = model.intercept_ + 1

    pred_plus1 = model.predict(X)

    model_fresh = getattr(linear_model, estimator)(fit_intercept=False).fit(X, y)
    pred = model_fresh.predict(X)

    np.testing.assert_almost_equal(pred_plus1, pred + 1)


# This test tries to check that if two threads simultaneously attempt
# to finalize an object, there will be a mutex somewhere preventing
# that from happening, allowing only one thread to do the finalization
# while the others wait. It has some hard-coded numbers for sleeping
# time that should trigger data races given the times it takes to
# do the finalization, but the test is stochastic in nature - it is
# not guaranteed that the conditions will trigger a scenario where
# this would fail without mechanisms to prevent it.
@pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="test requires free-threaded python",
)
def test_finalization_is_thread_safe(monkeypatch):
    from onedal.linear_model.incremental_linear_model import BaseIncrementalLinear
    from sklearnex.linear_model import IncrementalRidge

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(100, 500))
    y = rng.standard_normal(size=X.shape[0])

    n_threads = 4
    expected = IncrementalRidge().partial_fit(X, y).coef_
    ests = [IncrementalRidge().partial_fit(X, y)] * n_threads

    orig = BaseIncrementalLinear.finalize_fit
    guard, state = Lock(), {"now": 0, "max": 0}

    def probed(self, *args, **kwargs):
        with guard:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)  # hold the critical section open
        try:
            return orig(self, *args, **kwargs)
        finally:
            with guard:
                state["now"] -= 1

    monkeypatch.setattr(BaseIncrementalLinear, "finalize_fit", probed)

    start, results, errors = Barrier(n_threads), [None] * n_threads, []

    def run(i):
        try:
            start.wait()
            results[i] = ests[i].coef_  # lazy finalize
        except Exception as e:
            errors.append(e)

    threads = [Thread(target=run, args=(i,)) for i in range(n_threads)]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]

    assert not any(t.is_alive() for t in threads)  # no deadlock
    assert not errors
    assert state["max"] == 1  # the lock's actual guarantee
    for r in results:
        assert r.shape == expected.shape
        np.testing.assert_allclose(r, expected)
