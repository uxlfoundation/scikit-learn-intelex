# ===============================================================================
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
# ===============================================================================

import platform

import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

import daal4py as d4p
from daal4py.sklearn._utils import PatchingConditionsChain

from .._utils import sklearn_check_version

try:
    from sklearn.utils import _safe_indexing as safe_indexing
except ImportError:
    from sklearn.utils import safe_indexing

try:
    import mkl_random

    mkl_random_is_imported = True
except (ImportError, ModuleNotFoundError):
    mkl_random_is_imported = False

try:
    import pandas as pd

    pandas_is_imported = True
except (ImportError, ModuleNotFoundError):
    pandas_is_imported = False

if sklearn_check_version("1.3"):
    import numbers

    from sklearn.utils._param_validation import Interval, RealNotInt, validate_params


def get_dtypes(data):
    if hasattr(data, "dtype"):
        return [data.dtype]
    if hasattr(data, "dtypes"):
        return list(data.dtypes)
    if hasattr(data, "values"):
        return [data.values.dtype]
    return None


def train_test_split(*arrays, **options):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop("test_size", None)
    train_size = options.pop("train_size", None)
    random_state = options.pop("random_state", None)
    stratify = options.pop("stratify", None)
    shuffle = options.pop("shuffle", True)
    rng = options.pop("rng", "OPTIMIZED_MT19937")

    available_rngs = [
        "default",
        "MT19937",
        "SFMT19937",
        "MT2203",
        "R250",
        "WH",
        "MCG31",
        "MCG59",
        "MRG32K3A",
        "PHILOX4X32X10",
        "NONDETERM",
        "OPTIMIZED_MT19937",
    ]
    if rng not in available_rngs:
        raise ValueError(
            "Wrong random numbers generator is chosen. "
            "Available generators: %s" % str(available_rngs)[1:-1]
        )

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )
    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)
    else:
        if stratify is not None:
            cv = StratifiedShuffleSplit(
                test_size=n_test, train_size=n_train, random_state=random_state
            )
            train, test = next(cv.split(X=arrays[0], y=stratify))
        else:
            if (
                mkl_random_is_imported
                and rng not in ["default", "OPTIMIZED_MT19937"]
                and (isinstance(random_state, int) or random_state is None)
            ):
                random_state = mkl_random.RandomState(random_state, rng)
                indexes = random_state.permutation(n_samples)
                test, train = indexes[:n_test], indexes[n_test : (n_test + n_train)]
            elif (
                rng == "OPTIMIZED_MT19937"
                and (isinstance(random_state, int) or random_state is None)
                and platform.system() != "Windows"
            ):
                indexes = np.empty(
                    shape=(n_samples,),
                    dtype=np.int64 if n_train + n_test > 2**31 - 1 else np.int32,
                )
                random_state = np.random.RandomState(random_state)
                random_state = random_state.get_state()[1]
                d4p.daal_generate_shuffled_indices([indexes], [random_state])
                test, train = indexes[:n_test], indexes[n_test : (n_test + n_train)]
            else:
                cv = ShuffleSplit(
                    test_size=n_test, train_size=n_train, random_state=random_state
                )
                train, test = next(cv.split(X=arrays[0], y=stratify))

    res = []
    for arr in arrays:
        _patching_status = PatchingConditionsChain(
            "sklearn.model_selection.train_test_split"
        )

        # input format check
        _patching_status.and_conditions(
            [(isinstance(arr, np.ndarray), "The input is not a np.ndarray object.")]
        )
        if pandas_is_imported:
            _patching_status.or_conditions(
                [
                    (
                        isinstance(arr, pd.core.frame.DataFrame),
                        "The input is not a pd.DataFrame object.",
                    ),
                    (
                        isinstance(arr, pd.core.series.Series),
                        "The input is not a pd.Series object.",
                    ),
                ],
                conditions_merging=any,
            )

        # dimensions check
        _dal_ready = _patching_status.and_conditions(
            [(hasattr(arr, "ndim"), "The input does not have 'ndim' attribute.")]
        )
        if hasattr(arr, "ndim"):
            _patching_status.and_conditions(
                [(arr.ndim <= 2, "The input has more than 2 dimensions.")]
            )

        # data types check
        dtypes = get_dtypes(arr)
        _dal_ready = _patching_status.and_conditions(
            [(dtypes is not None, "Unable to parse input data types.")]
        )
        if dtypes is not None:
            incorrect_dtype = None
            for i, dtype in enumerate(dtypes):
                if str(dtype) not in ["float32", "float64"] and "int" not in str(dtype):
                    incorrect_dtype = str(dtype)
                    break
            _dal_ready = _patching_status.and_conditions(
                [
                    (
                        incorrect_dtype is None,
                        f"Input has incorrect data type '{incorrect_dtype}'. "
                        "Only integer and 32/64-bits floating point types are supported.",
                    )
                ]
            )

        _patching_status.write_log()
        if not _dal_ready:
            res.append(safe_indexing(arr, train))
            res.append(safe_indexing(arr, test))
        else:
            if len(arr.shape) == 2:
                n_cols = arr.shape[1]
                reshape_later = False
            else:
                n_cols = 1
                reshape_later = True

            arr_copy = d4p.get_data(arr)
            if not isinstance(arr_copy, list):
                arr_copy = arr_copy.reshape(
                    (arr_copy.shape[0], n_cols),
                    order="A",
                )
            if isinstance(arr_copy, np.ndarray):
                order = "C" if arr_copy.flags["C_CONTIGUOUS"] else "F"
                train_arr = np.empty(
                    shape=(n_train, n_cols),
                    dtype=arr_copy.dtype,
                    order=order,
                )
                test_arr = np.empty(
                    shape=(n_test, n_cols),
                    dtype=arr_copy.dtype,
                    order=order,
                )
                d4p.daal_train_test_split(arr_copy, train_arr, test_arr, [train], [test])
                if reshape_later:
                    train_arr, test_arr = train_arr.reshape((n_train,)), test_arr.reshape(
                        (n_test,)
                    )
            elif isinstance(arr_copy, list):
                train_arr = [
                    np.empty(
                        shape=(n_train,),
                        dtype=el.dtype,
                        order="C" if el.flags["C_CONTIGUOUS"] else "F",
                    )
                    for el in arr_copy
                ]
                test_arr = [
                    np.empty(
                        shape=(n_test,),
                        dtype=el.dtype,
                        order="C" if el.flags["C_CONTIGUOUS"] else "F",
                    )
                    for el in arr_copy
                ]
                d4p.daal_train_test_split(arr_copy, train_arr, test_arr, [train], [test])
                train_arr = {col: train_arr[i] for i, col in enumerate(arr.columns)}
                test_arr = {col: test_arr[i] for i, col in enumerate(arr.columns)}
            else:
                raise ValueError("Array can't be converted to needed format")

            if pandas_is_imported:
                if isinstance(arr, pd.core.frame.DataFrame):
                    train_arr, test_arr = pd.DataFrame(
                        train_arr, columns=arr.columns
                    ), pd.DataFrame(test_arr, columns=arr.columns)
                if isinstance(arr, pd.core.series.Series):
                    train_arr, test_arr = train_arr.reshape(n_train), test_arr.reshape(
                        n_test
                    )
                    train_arr, test_arr = pd.Series(train_arr, name=arr.name), pd.Series(
                        test_arr, name=arr.name
                    )

            if hasattr(arr, "index"):
                train_arr.index = arr.index.take(train)
                test_arr.index = arr.index.take(test)

            if hasattr(arr, "columns"):
                train_arr.columns = arr.columns
                test_arr.columns = arr.columns

            if hasattr(arr, "name"):
                train_arr.name = arr.name
                test_arr.name = arr.name

            res.append(train_arr)
            res.append(test_arr)

    return res


if sklearn_check_version("1.3"):
    train_test_split = validate_params(
        {
            "test_size": [
                Interval(RealNotInt, 0, 1, closed="neither"),
                Interval(numbers.Integral, 1, None, closed="left"),
                None,
            ],
            "train_size": [
                Interval(RealNotInt, 0, 1, closed="neither"),
                Interval(numbers.Integral, 1, None, closed="left"),
                None,
            ],
            "random_state": ["random_state"],
            "shuffle": ["boolean"],
            "stratify": ["array-like", None],
        },
        prefer_skip_nested_validation=True,
    )(train_test_split)
