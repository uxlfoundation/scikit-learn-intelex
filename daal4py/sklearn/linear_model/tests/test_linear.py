# ==============================================================================
# Copyright 2020 Intel Corporation
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


from os import environ

from daal4py.sklearn._utils import sklearn_check_version

# sklearn requires manual enabling of Scipy array API support
# if `array-api-compat` package is present in environment
# TODO: create generic approach to handle this for all tests
if sklearn_check_version("1.6"):
    environ["SCIPY_ARRAY_API"] = "1"


import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.utils._testing import assert_array_almost_equal


def make_dataset(n_samples, n_features, kind=np.array, random_state=0, types=None):
    try:
        from pandas import DataFrame

        if kind not in (list, np.array, DataFrame):
            kind = np.array
    except ImportError:
        if kind not in (list, np.array):
            kind = np.array

    x, y = make_regression(n_samples, n_features, random_state=random_state)

    if kind == list:
        x = list(x)
        for i, row in enumerate(x):
            x[i] = list(row)
        y = list(y)

        if types:
            n_types = len(types)
            for i, row in enumerate(x):
                for j, cell in enumerate(row):
                    x[i][j] = types[j % n_types](cell)

    try:
        from pandas import DataFrame

        if kind == DataFrame:
            x = DataFrame(data=x, index=None, columns=None)
            y = DataFrame(y)

            if types:
                n_types = len(types)
                dir_dtypes = {col: types[i % n_types] for i, col in enumerate(x.columns)}
                x = x.astype(dir_dtypes)
    except ImportError:
        pass

    return x, y


def test_linear_array_vs_dataframe_homogen():
    pd = pytest.importorskip("pandas")

    x_train, y_train = make_dataset(100, 20)
    x_test, _ = make_dataset(100, 20, random_state=1)

    df_x_train, df_y_train = make_dataset(100, 20, pd.DataFrame)
    df_x_test, _ = make_dataset(100, 20, pd.DataFrame, random_state=1)

    array_reg = LinearRegression()
    array_reg.fit(x_train, y_train)

    df_reg = LinearRegression()
    df_reg.fit(df_x_train, df_y_train)

    assert_array_almost_equal(
        array_reg.coef_.reshape((-1, 1)), df_reg.coef_.reshape((-1, 1))
    )
    assert_array_almost_equal(array_reg.intercept_, df_reg.intercept_)
    assert_array_almost_equal(
        array_reg.predict(x_test).reshape((-1, 1)),
        df_reg.predict(df_x_test).reshape((-1, 1)),
    )


def test_linear_array_vs_dataframe_heterogen():
    pd = pytest.importorskip("pandas")

    types = (np.float64, np.float32)

    x_train, y_train = make_dataset(100, 20)
    x_test, _ = make_dataset(100, 20, random_state=1)

    df_x_train, df_y_train = make_dataset(100, 20, pd.DataFrame, types=types)
    df_x_test, _ = make_dataset(100, 20, pd.DataFrame, random_state=1, types=types)

    array_reg = LinearRegression()
    array_reg.fit(x_train, y_train)

    df_reg = LinearRegression()
    df_reg.fit(df_x_train, df_y_train)

    assert_array_almost_equal(
        array_reg.coef_.reshape((-1, 1)), df_reg.coef_.reshape((-1, 1))
    )
    assert_array_almost_equal(array_reg.intercept_, df_reg.intercept_)
    assert_array_almost_equal(
        array_reg.predict(x_test).reshape((-1, 1)),
        df_reg.predict(df_x_test).reshape((-1, 1)),
        decimal=5,
    )


def test_linear_array_vs_dataframe_heterogen_double_float():
    pd = pytest.importorskip("pandas")

    types = (np.float64, np.float32)

    x_train, y_train = make_dataset(100, 20, list, types=types)
    x_test, _ = make_dataset(100, 20, list, random_state=1, types=types)

    df_x_train, df_y_train = make_dataset(100, 20, pd.DataFrame, types=types)
    df_x_test, _ = make_dataset(100, 20, pd.DataFrame, random_state=1, types=types)

    array_reg = LinearRegression()
    array_reg.fit(x_train, y_train)

    df_reg = LinearRegression()
    df_reg.fit(df_x_train, df_y_train)

    assert_array_almost_equal(
        array_reg.coef_.reshape((-1, 1)), df_reg.coef_.reshape((-1, 1))
    )
    assert_array_almost_equal(array_reg.intercept_, df_reg.intercept_)
    assert_array_almost_equal(
        array_reg.predict(x_test).reshape((-1, 1)),
        df_reg.predict(df_x_test).reshape((-1, 1)),
    )


def test_linear_array_vs_dataframe_heterogen_double_int():
    pd = pytest.importorskip("pandas")

    types = (np.float64, np.int32)

    x_train, y_train = make_dataset(100, 20, list, types=types)
    x_test, _ = make_dataset(100, 20, list, random_state=1, types=types)

    df_x_train, df_y_train = make_dataset(100, 20, pd.DataFrame, types=types)
    df_x_test, _ = make_dataset(100, 20, pd.DataFrame, random_state=1, types=types)

    array_reg = LinearRegression()
    array_reg.fit(x_train, y_train)

    df_reg = LinearRegression()
    df_reg.fit(df_x_train, df_y_train)

    assert_array_almost_equal(
        array_reg.coef_.reshape((-1, 1)), df_reg.coef_.reshape((-1, 1))
    )
    assert_array_almost_equal(array_reg.intercept_, df_reg.intercept_)
    assert_array_almost_equal(
        array_reg.predict(x_test).reshape((-1, 1)),
        df_reg.predict(df_x_test).reshape((-1, 1)),
    )


def test_linear_array_vs_dataframe_heterogen_float_int():
    pd = pytest.importorskip("pandas")

    types = (np.float32, np.int32)

    x_train, y_train = make_dataset(100, 20, list, types=types)
    x_test, _ = make_dataset(100, 20, list, random_state=1, types=types)

    df_x_train, df_y_train = make_dataset(100, 20, pd.DataFrame, types=types)
    df_x_test, _ = make_dataset(100, 20, pd.DataFrame, random_state=1, types=types)

    array_reg = LinearRegression()
    array_reg.fit(x_train, y_train)

    df_reg = LinearRegression()
    df_reg.fit(df_x_train, df_y_train)

    assert_array_almost_equal(
        array_reg.coef_.reshape((-1, 1)), df_reg.coef_.reshape((-1, 1))
    )
    assert_array_almost_equal(array_reg.intercept_, df_reg.intercept_)
    assert_array_almost_equal(
        array_reg.predict(x_test).reshape((-1, 1)),
        df_reg.predict(df_x_test).reshape((-1, 1)),
    )
