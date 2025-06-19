# ==============================================================================
# Copyright 2014 Intel Corporation
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
from contextlib import suppress
from functools import wraps

import numpy as np
import scipy.sparse as sp

if np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion("2.0.0a0"):
    # numpy_version >= 2.0
    from numpy.exceptions import ComplexWarning
else:
    # numpy_version < 2.0
    from numpy.core.numeric import ComplexWarning

from sklearn import get_config as _get_config
from sklearn.utils.extmath import _safe_accumulator_op
from sklearn.utils.fixes import _object_dtype_isnan
from sklearn.utils.validation import _assert_all_finite as _sklearn_assert_all_finite
from sklearn.utils.validation import (
    _ensure_no_complex_data,
    _ensure_sparse_format,
    _num_samples,
    check_consistent_length,
    column_or_1d,
)

import daal4py as d4p

from .._utils import (
    PatchingConditionsChain,
    get_dtype,
    get_number_of_types,
    is_DataFrame,
    sklearn_check_version,
)

if sklearn_check_version("1.6"):
    from sklearn.utils.validation import (
        _check_feature_names as _sklearn_check_feature_names,
    )
    from sklearn.utils.validation import _check_n_features as _sklearn_check_n_features
    from sklearn.utils.validation import validate_data as _sklearn_validate_data
else:
    from sklearn.base import BaseEstimator

    _sklearn_validate_data = BaseEstimator._validate_data
    _sklearn_check_feature_names = BaseEstimator._check_feature_names
    _sklearn_check_n_features = BaseEstimator._check_n_features


def _assert_all_finite(
    X, allow_nan=False, msg_dtype=None, estimator_name=None, input_name=""
):
    if _get_config()["assume_finite"]:
        return

    # Data with small size has too big relative overhead
    # TODO: tune threshold size
    is_df = is_DataFrame(X)
    if not (is_df or isinstance(X, np.ndarray)) or X.size < 32768:
        if sklearn_check_version("1.1"):
            _sklearn_assert_all_finite(
                X,
                allow_nan=allow_nan,
                msg_dtype=msg_dtype,
                estimator_name=estimator_name,
                input_name=input_name,
            )
        else:
            _sklearn_assert_all_finite(X, allow_nan=allow_nan, msg_dtype=msg_dtype)
        return

    num_of_types = get_number_of_types(X)

    # if X is heterogeneous pandas.DataFrame then
    # convert it to a list of arrays
    if is_df and num_of_types > 1:
        lst = []
        for idx in X:
            arr = X[idx].to_numpy()
            lst.append(arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr))
    else:
        X = np.asanyarray(X)
        is_df = False

    dt = np.dtype(get_dtype(X))
    is_float = dt.kind in "fc"

    msg_err = "Input {} contains {} or a value too large for {!r}."
    type_err = "infinity" if allow_nan else "NaN, infinity"
    err = msg_err.format(input_name, type_err, msg_dtype if msg_dtype is not None else dt)

    _patching_status = PatchingConditionsChain(
        "sklearn.utils.validation._assert_all_finite"
    )
    _dal_ready = _patching_status.and_conditions(
        [
            (X.ndim in [1, 2], f"Input {input_name} does not have 1 or 2 dimensions."),
            (not np.any(np.equal(X.shape, 0)), f"Input {input_name} shape contains a 0."),
            (
                dt in [np.float32, np.float64],
                f"Input {input_name} dtype is not float32 or float64.",
            ),
        ]
    )
    _patching_status.write_log()
    if _dal_ready:
        if X.ndim == 1:
            X = X.reshape((-1, 1))

        x_for_daal = lst if is_df and num_of_types > 1 else X

        if dt == np.float64:
            if not d4p.daal_assert_all_finite(x_for_daal, allow_nan, 0):
                raise ValueError(err)
        elif dt == np.float32:
            if not d4p.daal_assert_all_finite(x_for_daal, allow_nan, 1):
                raise ValueError(err)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    elif is_float and (np.isfinite(_safe_accumulator_op(np.sum, X))):
        pass
    elif is_float:
        if allow_nan and np.isinf(X).any() or not allow_nan and not np.isfinite(X).all():
            raise ValueError(err)
    # for object dtype data, we only check for NaNs (GH-13254)
    elif dt == np.dtype("object") and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError(f"Input {input_name} contains NaN")


def _pandas_check_array(
    array,
    array_orig,
    force_all_finite,
    ensure_min_samples,
    ensure_min_features,
    copy,
    context,
):
    if force_all_finite:
        _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError(
                "Found array with %d sample(s) (shape=%s) while a"
                " minimum of %d is required%s."
                % (n_samples, array.shape, ensure_min_samples, context)
            )

    if ensure_min_features > 0:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required%s."
                % (n_features, array.shape, ensure_min_features, context)
            )

    if copy and np.may_share_memory(array, array_orig):
        array = array.copy()

    return array


def _daal_check_array(
    array,
    accept_sparse=False,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    ensure_2d : boolean (default=True)
        Whether to raise a value error if array is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow array.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    if force_all_finite not in (True, False, "allow-nan"):
        raise ValueError(
            'force_all_finite should be a bool or "allow-nan"'
            ". Got {!r} instead".format(force_all_finite)
        )

    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    array_orig = array

    # a branch for heterogeneous pandas.DataFrame
    if is_DataFrame(array) and get_number_of_types(array) > 1:
        from pandas import SparseDtype

        def is_sparse(dtype):
            return isinstance(dtype, SparseDtype)

        if hasattr(array, "sparse") or not array.dtypes.apply(is_sparse).any():
            return _pandas_check_array(
                array,
                array_orig,
                force_all_finite,
                ensure_min_samples,
                ensure_min_features,
                copy,
                context,
            )

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, "kind"):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    has_pd_integer_array = False
    if hasattr(array, "dtypes") and hasattr(array.dtypes, "__array__"):
        # throw warning if columns are sparse. If all columns are sparse, then
        # array.sparse exists and sparsity will be preserved (later).
        with suppress(ImportError):
            from pandas import SparseDtype

            def is_sparse(dtype):
                return isinstance(dtype, SparseDtype)

            if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
                warnings.warn(
                    "pandas.DataFrame with sparse columns found."
                    "It will be converted to a dense numpy array."
                )

        dtypes_orig = list(array.dtypes)
        # pandas boolean dtype __array__ interface coerces bools to objects
        for i, dtype_iter in enumerate(dtypes_orig):
            if dtype_iter.kind == "b":
                dtypes_orig[i] = np.dtype(np.object)
            elif dtype_iter.name.startswith(("Int", "UInt")):
                # name looks like an Integer Extension Array, now check for
                # the dtype
                with suppress(ImportError):
                    from pandas import (
                        Int8Dtype,
                        Int16Dtype,
                        Int32Dtype,
                        Int64Dtype,
                        UInt8Dtype,
                        UInt16Dtype,
                        UInt32Dtype,
                        UInt64Dtype,
                    )

                    if isinstance(
                        dtype_iter,
                        (
                            Int8Dtype,
                            Int16Dtype,
                            Int32Dtype,
                            Int64Dtype,
                            UInt8Dtype,
                            UInt16Dtype,
                            UInt32Dtype,
                            UInt64Dtype,
                        ),
                    ):
                        has_pd_integer_array = True

        if all(isinstance(dtype, np.dtype) for dtype in dtypes_orig):
            dtype_orig = np.result_type(*dtypes_orig)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if has_pd_integer_array:
        # If there are any pandas integer extension arrays,
        array = array.astype(dtype)

    # When all dataframe columns are sparse, convert to a sparse array
    if hasattr(array, "sparse") and array.ndim > 1:
        # DataFrame.sparse only supports `to_coo`
        array = array.sparse.to_coo()

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        kwargs = {
            "accept_sparse": accept_sparse,
            "dtype": dtype,
            "copy": copy,
            "accept_large_sparse": accept_large_sparse,
        }
        if sklearn_check_version("1.6"):
            kwargs["ensure_all_finite"] = force_all_finite
        else:
            kwargs["force_all_finite"] = force_all_finite
        array = _ensure_sparse_format(
            array,
            **kwargs,
        )
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter("error", ComplexWarning)
                if dtype is not None and np.dtype(dtype).kind in "iu":
                    # Conversion float -> int should not contain NaN or
                    # inf (numpy#14412). We cannot use casting='safe' because
                    # then conversion float -> int would be disallowed.
                    array = np.asarray(array, order=order)
                    if array.dtype.kind == "f":
                        _assert_all_finite(array, allow_nan=False, msg_dtype=dtype)
                    array = array.astype(dtype, casting="unsafe", copy=False)
                else:
                    array = np.asarray(array, order=order, dtype=dtype)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n" "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)  # doing nothing for DataFrame

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array)
                )
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array)
                )

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
                FutureWarning,
                stacklevel=2,
            )

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError(
                "Found array with dim %d. %s expected <= 2."
                % (array.ndim, estimator_name)
            )

        if force_all_finite:
            _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError(
                "Found array with %d sample(s) (shape=%s) while a"
                " minimum of %d is required%s."
                % (n_samples, array.shape, ensure_min_samples, context)
            )

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required%s."
                % (n_features, array.shape, ensure_min_features, context)
            )

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    return array


def _daal_check_X_y(
    X,
    y,
    accept_sparse=False,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    multi_output=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    y_numeric=False,
    estimator=None,
):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf, np.nan, pd.NA in X. This parameter
        does not influence whether y can have np.inf, np.nan, pd.NA values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan or pd.NA values in X. Values cannot
          be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    if y is None:
        raise ValueError("y cannot be None")

    X = _daal_check_array(
        X,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        order=order,
        copy=copy,
        force_all_finite=force_all_finite,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        estimator=estimator,
    )
    if multi_output:
        y = _daal_check_array(
            y, accept_sparse="csr", force_all_finite=True, ensure_2d=False, dtype=None
        )
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and hasattr(y, "dtype") and y.dtype.kind == "O":
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y


def _daal_num_features(X):
    """Return the number of features in an array-like X.
    This helper function tries hard to avoid to materialize an array version
    of X unless necessary. For instance, if X is a list of lists,
    this function will return the length of the first element, assuming
    that subsequent elements are all lists of the same length without
    checking.
    Parameters
    ----------
    X : array-like
        array-like to get the number of features.
    Returns
    -------
    features : int
        Number of features
    """
    type_ = type(X)
    if type_.__module__ == "builtins":
        type_name = type_.__qualname__
    else:
        type_name = f"{type_.__module__}.{type_.__qualname__}"
    message = f"Unable to find the number of features from X of type {type_name}"
    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if not hasattr(X, "__array__"):
            raise TypeError(message)
        # Only convert X to a numpy array if there is no cheaper, heuristic
        # option.
        X = np.asarray(X)

    if hasattr(X, "shape"):
        if not hasattr(X.shape, "__len__") or len(X.shape) <= 1:
            message += f" with shape {X.shape}"
            raise TypeError(message)
        return X.shape[1]

    first_sample = X[0]

    # Do not consider an array-like of strings or dicts to be a 2D array
    if isinstance(first_sample, (str, bytes, dict)):
        message += f" where the samples are of type {type(first_sample).__qualname__}"
        raise TypeError(message)

    try:
        # If X is a list of lists, for instance, we assume that all nested
        # lists have the same length without checking or converting to
        # a numpy array to keep this function call as cheap as possible.
        return len(first_sample)
    except Exception as err:
        raise TypeError(message) from err


def get_requires_y_tag(estimator):
    """Gets the value of the 'requires_y' tag from the estimator
    using correct code path depending on the scikit-learn version."""
    if sklearn_check_version("1.6"):
        requires_y = estimator.__sklearn_tags__().target_tags.required
    else:
        try:
            requires_y = estimator._get_tags()["requires_y"]
        except KeyError:
            requires_y = False
    return requires_y


def add_dispatcher_docstring(original_function):
    """Adds a note about the dispatcher function to the docstring of the original function."""

    def wrapper(dispatcher_function):
        @wraps(
            original_function,
            ["__name__", "__doc__", "__annotations__", "__type_params__"],
        )
        def new_function(*args, **kwargs):
            return dispatcher_function(*args, **kwargs)

        new_function.__doc__ = (
            f"Sklearnex dispatcher for '{original_function.__qualname__}' "
            f"from '{original_function.__module__}' module supporting "
            "it's multiple implementations from different scikit-learn versions.\n\n"
            + original_function.__doc__
        )

        return new_function

    return wrapper


# simplified copy of similar function from sklearnex.utils.validation,
# ensures that the correct finiteness check argument is used
@add_dispatcher_docstring(_sklearn_validate_data)
def validate_data(*args, **kwargs):
    if not sklearn_check_version("1.6") and "ensure_all_finite" in kwargs:
        kwargs["force_all_finite"] = kwargs.pop("ensure_all_finite")
    return _sklearn_validate_data(*args, **kwargs)


# dispatcher functions which use correct `check_feature_names`/`check_n_features` import
# depending on the scikit-learn version
@add_dispatcher_docstring(_sklearn_check_feature_names)
def check_feature_names(*args, **kwargs):
    _sklearn_check_feature_names(*args, **kwargs)


@add_dispatcher_docstring(_sklearn_check_n_features)
def check_n_features(*args, **kwargs):
    _sklearn_check_n_features(*args, **kwargs)
