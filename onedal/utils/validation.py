# ==============================================================================
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
# ==============================================================================

import inspect
import warnings
from collections.abc import Sequence
from numbers import Integral

import numpy as np
from scipy import sparse as sp

from onedal.common._backend import BackendFunction
from onedal.utils import _sycl_queue_manager as QM

if np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion("2.0.0a0"):
    # numpy_version >= 2.0
    from numpy.exceptions import VisibleDeprecationWarning
else:
    # numpy_version < 2.0
    from numpy import VisibleDeprecationWarning

from sklearn.preprocessing import LabelEncoder

from daal4py.sklearn.utils.validation import (
    _assert_all_finite as _daal4py_assert_all_finite,
)
from onedal import _default_backend as backend
from onedal.datatypes import to_table


class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code."""


def _is_arraylike(x):
    """Returns whether the input is array-like."""
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _is_arraylike_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return _is_arraylike(array) and not np.isscalar(array)


def _is_integral_float(y):
    return y.dtype.kind == "f" and np.all(y.astype(int) == y)


def _is_multilabel(y):
    if hasattr(y, "__array__") or isinstance(y, Sequence):
        # DeprecationWarning will be replaced by ValueError, see NEP 34
        # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
        with warnings.catch_warnings():
            warnings.simplefilter("error", VisibleDeprecationWarning)
            try:
                y = np.asarray(y)
            except VisibleDeprecationWarning:
                # dtype=object should be provided explicitly for ragged arrays,
                # see NEP 34
                y = np.array(y, dtype=object)

    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    if sp.issparse(y):
        if isinstance(y, (sp.dok_matrix, sp.lil_matrix)):
            y = y.tocsr()
        return (
            len(y.data) == 0
            or np.unique(y.data).size == 1
            and (y.dtype.kind in "biu" or _is_integral_float(np.unique(y.data)))
        )
    labels = np.unique(y)

    return len(labels) < 3 and (y.dtype.kind in "biu" or _is_integral_float(labels))


def _check_n_features(self, X, reset):
    try:
        n_features = _num_features(X)
    except TypeError as e:
        if not reset and hasattr(self, "n_features_in_"):
            raise ValueError(
                "X does not contain any features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features"
            ) from e
        # If the number of features is not defined and reset=True,
        # then we skip this check
        return

    if reset:
        self.n_features_in_ = n_features
        return

    if not hasattr(self, "n_features_in_"):
        # Skip this check if the expected number of expected input features
        # was not recorded by calling fit first. This is typically the case
        # for stateless transformers.
        return

    if n_features != self.n_features_in_:
        raise ValueError(
            f"X has {n_features} features, but {self.__class__.__name__} "
            f"is expecting {self.n_features_in_} features as input."
        )


def _num_features(X, fallback_1d=False):
    if X is None:
        raise ValueError("Expected array-like (array or non-string sequence), got None")
    type_ = type(X)
    if type_.__module__ == "builtins":
        type_name = type_.__qualname__
    else:
        type_name = f"{type_.__module__}.{type_.__qualname__}"
    message = "Unable to find the number of features from X of type " f"{type_name}"
    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if not hasattr(X, "__array__"):
            raise ValueError(message)
        # Only convert X to a numpy array if there is no cheaper, heuristic
        # option.
        X = np.asarray(X)

    if hasattr(X, "shape"):
        ndim_thr = 1 if fallback_1d else 2
        if not hasattr(X.shape, "__len__") or len(X.shape) < ndim_thr:
            message += f" with shape {X.shape}"
            raise ValueError(message)
        if len(X.shape) <= 1:
            return 1
        else:
            return X.shape[-1]

    try:
        first_sample = X[0]
    except IndexError:
        raise ValueError("Passed empty data.")

    # Do not consider an array-like of strings or dicts to be a 2D array
    if isinstance(first_sample, (str, bytes, dict)):
        message += f" where the samples are of type " f"{type(first_sample).__qualname__}"
        raise ValueError(message)

    try:
        # If X is a list of lists, for instance, we assume that all nested
        # lists have the same length without checking or converting to
        # a numpy array to keep this function call as cheap as possible.
        if (not fallback_1d) or hasattr(first_sample, "__len__"):
            return len(first_sample)
        else:
            return 1
    except Exception as err:
        raise ValueError(message) from err


def _num_samples(x):
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
    # Check that shape is returning an integer or default to len
    # Dask dataframes may not return numeric shape[0] value
    if hasattr(x, "shape") and isinstance(x.shape[0], Integral):
        return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def _is_csr(x):
    """Return True if x is scipy.sparse.csr_matrix or scipy.sparse.csr_array"""
    return isinstance(x, sp.csr_matrix) or (
        hasattr(sp, "csr_array") and isinstance(x, sp.csr_array)
    )


def check_all_finite(
    X,
    allow_nan: bool = False,
) -> bool:
    if sp.issparse(X):
        X = X.data
    backend_method = BackendFunction(
        backend.finiteness_checker.compute.compute, backend, "compute", no_policy=False
    )
    X_t = to_table(X)
    params = {
        "fptype": X_t.dtype,
        "method": "dense",
        "allow_nan": allow_nan,
    }
    with QM.manage_global_queue(None, X):
        # Must use the queue provided by X
        return bool(backend_method(params, X_t).finite)


def is_contiguous(X):
    if hasattr(X, "flags"):
        return X.flags["C_CONTIGUOUS"] or X.flags["F_CONTIGUOUS"]
    elif hasattr(X, "__dlpack__"):
        return backend.dlpack_memory_order(X) is not None
    else:
        return False
