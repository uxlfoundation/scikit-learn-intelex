# ===============================================================================
# Copyright 2022 Intel Corporation
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

import numbers

import scipy.sparse as sp
from sklearn.utils.validation import _assert_all_finite as _sklearn_assert_all_finite
from sklearn.utils.validation import _num_samples, check_array, check_non_negative

from daal4py.sklearn._utils import sklearn_check_version
from onedal.utils.validation import _assert_all_finite as _onedal_assert_all_finite

from ._array_api import get_namespace

if sklearn_check_version("1.6"):
    from sklearn.utils.validation import validate_data as _sklearn_validate_data

    _finite_keyword = "ensure_all_finite"

else:
    from sklearn.base import BaseEstimator

    _sklearn_validate_data = BaseEstimator._validate_data
    _finite_keyword = "force_all_finite"


def _is_contiguous(X):
    # array_api does not have a `strides` or `flags` attribute for testing memory
    # order. When dlpack support is brought in for oneDAL, the dlpack python capsule
    # can then be inspected for strides and this must be updated. _is_contiguous is
    # therefore conservative in verifying attributes and does not support array_api.
    # This will block onedal_assert_all_finite from being used for array_api inputs.
    return hasattr(X, "flags") and (X.flags["C_CONTIGUOUS"] or X.flags["F_CONTIGUOUS"])


def _sklearnex_assert_all_finite(
    X,
    *,
    allow_nan=False,
    input_name="",
):
    # size check is an initial match to daal4py for performance reasons, can be
    # optimized later
    xp, _ = get_namespace(X)
    if X.size < 32768 or X.dtype not in [xp.float32, xp.float64] or not _is_contiguous(X):
        if sklearn_check_version("1.1"):
            _sklearn_assert_all_finite(X, allow_nan=allow_nan, input_name=input_name)
        else:
            _sklearn_assert_all_finite(X, allow_nan=allow_nan)
    else:
        _onedal_assert_all_finite(X, allow_nan=allow_nan, input_name=input_name)


def assert_all_finite(
    X,
    *,
    allow_nan=False,
    input_name="",
):
    _sklearnex_assert_all_finite(
        X.data if sp.issparse(X) else X,
        allow_nan=allow_nan,
        input_name=input_name,
    )


def validate_data(
    _estimator,
    /,
    X="no_validation",
    y="no_validation",
    **kwargs,
):
    # force finite check to not occur in sklearn, default is True
    # `ensure_all_finite` is the most up-to-date keyword name in sklearn
    # _finite_keyword provides backward compatability for `force_all_finite`
    ensure_all_finite = kwargs.pop("ensure_all_finite", True)
    kwargs[_finite_keyword] = False

    out = _sklearn_validate_data(
        _estimator,
        X=X,
        y=y,
        **kwargs,
    )
    if ensure_all_finite:
        # run local finite check
        allow_nan = ensure_all_finite == "allow-nan"
        arg = iter(out if isinstance(out, tuple) else (out,))
        if not isinstance(X, str) or X != "no_validation":
            assert_all_finite(next(arg), allow_nan=allow_nan, input_name="X")
        if not (y is None or isinstance(y, str) and y == "no_validation"):
            assert_all_finite(next(arg), allow_nan=allow_nan, input_name="y")
    return out


def _check_sample_weight(
    sample_weight, X, dtype=None, copy=False, only_non_negative=False
):

    n_samples = _num_samples(X)
    xp, _ = get_namespace(X)

    if dtype is not None and dtype not in [xp.float32, xp.float64]:
        dtype = xp.float64

    if sample_weight is None:
        sample_weight = xp.ones(n_samples, dtype=dtype)
    elif isinstance(sample_weight, numbers.Number):
        sample_weight = xp.full(n_samples, sample_weight, dtype=dtype)
    else:
        if dtype is None:
            dtype = [xp.float64, xp.float32]

        params = {
            "accept_sparse": False,
            "ensure_2d": False,
            "dtype": dtype,
            "order": "C",
            "copy": copy,
            _finite_keyword: False,
        }
        if sklearn_check_version("1.1"):
            params["input_name"] = "sample_weight"

        sample_weight = check_array(sample_weight, **params)
        assert_all_finite(sample_weight, input_name="sample_weight")

        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape != (n_samples,):
            raise ValueError(
                "sample_weight.shape == {}, expected {}!".format(
                    sample_weight.shape, (n_samples,)
                )
            )

    if only_non_negative:
        check_non_negative(sample_weight, "`sample_weight`")

    return sample_weight
