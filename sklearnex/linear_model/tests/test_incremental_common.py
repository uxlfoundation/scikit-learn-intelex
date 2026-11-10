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
from sklearnex import linear_model

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
@pytest.mark.parametrize("estimator", ["IncrementalLinearRegression", "IncrementalRidge"])
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
def test_linear_move_estimator_to(
    estimator, array_input_like, array_move_like, num_targets, with_array_api
):
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

    model = getattr(linear_model, estimator)().fit(X_in[:10, :], y_in[:10, ...])
    pred_orig = model.predict(X_in)

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

    # Partial fits should still work, and produce independent arrays
    # of coefficients and intercepts
    model_ref = getattr(linear_model, estimator)().fit(X, y)
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
        np.testing.assert_allclose(_as_numpy(pred_orig), _as_numpy(pref_ref), atol=1e-6)
        np.testing.assert_allclose(_as_numpy(pred_moved), _as_numpy(pref_ref), atol=1e-6)
