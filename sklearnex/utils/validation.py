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

import math
import numbers

import scipy.sparse as sp
from sklearn.utils.validation import _assert_all_finite as _sklearn_assert_all_finite
from sklearn.utils.validation import _num_samples, check_array, check_non_negative

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from daal4py.sklearn.utils.validation import (
    add_dispatcher_docstring,
    check_feature_names,
    check_n_features,
)
from onedal.utils.validation import is_contiguous

from ._array_api import get_namespace

if sklearn_check_version("1.6"):
    from sklearn.utils.validation import validate_data as _sklearn_validate_data

    _finite_keyword = "ensure_all_finite"
else:
    from sklearn.base import BaseEstimator

    _sklearn_validate_data = BaseEstimator._validate_data
    _finite_keyword = "force_all_finite"


if daal_check_version((2024, "P", 700)):
    from onedal.utils.validation import _assert_all_finite as _onedal_assert_all_finite

    def _onedal_supported_format(X, xp):
        # data should be checked if contiguous, as oneDAL will only use contiguous
        # data from sklearnex. Unlike other oneDAL offloading, copying the data is
        # specifically avoided as it has a non-negligible impact on speed. In that
        # case use native sklearn ``_assert_all_finite``
        return X.dtype in [xp.float32, xp.float64] and is_contiguous(X)

else:
    from daal4py.utils.validation import _assert_all_finite as _onedal_assert_all_finite
    from onedal.utils._array_api import _is_numpy_namespace

    def _onedal_supported_format(X, xp):
        # daal4py _assert_all_finite only supports numpy namespaces, use internally-
        # defined check to validate inputs, otherwise offload to sklearn
        return X.dtype in [xp.float32, xp.float64] and _is_numpy_namespace(xp)


def _sklearnex_assert_all_finite(
    X,
    *,
    allow_nan=False,
    input_name="",
):
    # size check is an initial match to daal4py for performance reasons, can be
    # optimized later
    xp, _ = get_namespace(X)
    # this is a PyTorch-specific fix, as Tensor.size is a function. It replicates `.size`
    too_small = math.prod(X.shape) < 32768

    if too_small or not _onedal_supported_format(X, xp):
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


@add_dispatcher_docstring(_sklearn_validate_data)
def validate_data(
    _estimator,
    /,
    X="no_validation",
    y="no_validation",
    **kwargs,
):
    # force finite check to not occur in sklearn, default is True
    # `ensure_all_finite` is the most up-to-date keyword name in sklearn
    # _finite_keyword provides backward compatibility for `force_all_finite`
    ensure_all_finite = kwargs.pop("ensure_all_finite", True)
    kwargs[_finite_keyword] = False

    out = _sklearn_validate_data(
        _estimator,
        X=X,
        y=y,
        **kwargs,
    )

    check_x = not isinstance(X, str) or X != "no_validation"
    check_y = not (y is None or isinstance(y, str) and y == "no_validation")

    if ensure_all_finite:
        # run local finite check
        allow_nan = ensure_all_finite == "allow-nan"
        # the return object from validate_data can be a single

        # element (either x or y) or both (as a tuple). An iterator along with
        # check_x and check_y can go through the output properly without
        # stacking layers of if statements to make sure the proper input_name
        # is used
        arg = iter(out if isinstance(out, tuple) else (out,))
        if check_x:
            assert_all_finite(next(arg), allow_nan=allow_nan, input_name="X")
        if check_y:
            assert_all_finite(next(arg), allow_nan=allow_nan, input_name="y")

    return out


def _check_sample_weight(
    sample_weight, X, dtype=None, copy=False, ensure_non_negative=False
):

    n_samples = _num_samples(X)
    xp, _ = get_namespace(X)

    if dtype is not None and dtype not in [xp.float32, xp.float64]:
        dtype = xp.float64

    if sample_weight is None:
        if hasattr(X, "device"):
            sample_weight = xp.ones(n_samples, dtype=dtype, device=X.device)
        else:
            sample_weight = xp.ones(n_samples, dtype=dtype)
    elif isinstance(sample_weight, numbers.Number):
        if hasattr(X, "device"):
            sample_weight = xp.full(
                n_samples, sample_weight, dtype=dtype, device=X.device
            )
        else:
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

    if ensure_non_negative:
        check_non_negative(sample_weight, "`sample_weight`")

    return sample_weight
