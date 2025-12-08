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
import pandas as pd
import pytest
import scipy
import scipy.sparse as sp

import daal4py  # Note: this is used  through 'eval'
import onedal
from sklearnex.basic_statistics import BasicStatistics
from sklearnex.decomposition import PCA
from sklearnex.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearnex.manifold import TSNE
from sklearnex.svm import SVR

# Note here: 'TSNE', 'LogisticRegression', and 'Lasso' are implemented
# in daal4py, and thus work differently from the rest and need separate
# testing. Other estimators like KMeans and SVM which are implemented through
# the 'onedal' module all follow the same logic, so testing two different
# instances should be enough.
# Note also that 'TSNE' is even more special than the rest, because it has
# multiple stages (initialization, calculation of distances, TSNE optimization)
# and the oneDAL implementation covers only the last one, but the earlier
# stages call sklearn estimators / functions that are patched.


# DataFrames of different types. They should follow the following logic
# which mimics scikit-learn's handling of the same classes:
# - Normal inputs should be converted to numpy arrays.
# - Inputs where all columns are sparse should be converted to sparse
#   arrays or matrices.
# - Inputs where some columns are sparse and some are dense should be
#   converted to dense, regardless of how sparse they are.
# - Sparse inputs should be passed as CSR to oneDAL when supported,
#   regardless of their original type.
def make_dense_df():
    rng = np.random.default_rng(seed=123)
    X = rng.random(size=(50, 4))
    return pd.DataFrame(X)


def make_sparse_df():
    X = sp.random(50, 4, 0.5, format="coo", random_state=123)
    return pd.DataFrame(X.toarray()).astype(pd.SparseDtype("float", 0))


def make_mixed_df():
    rng = np.random.default_rng(seed=123)
    col1 = rng.random(size=(50, 1))
    X_sp = sp.random(50, 3, 0.5, format="coo", random_state=123)
    X = np.c_[col1, X_sp.toarray()]
    df = pd.DataFrame(X)
    for col in range(1, 4):
        df[col] = df[col].astype(pd.SparseDtype("float", 0))
    return df


# Sparse matrices from SciPy can come in different classes
def make_sparse_matrix():
    out = sp.random(50, 4, 0.5, format="csc", random_state=123)
    return sp.csc_matrix(out)


def make_sparse_array():
    out = sp.random(50, 4, 0.5, format="csc", random_state=123)
    return sp.csc_array(out)


# Note: sparse pandas data frames have version requirements on scipy.
# This skips the tests if they are incompatible.
def check_sparse_df_is_supported():
    scipy_version = scipy.__version__.split(".")
    if int(scipy_version[0]) > 1:
        return True
    if int(scipy_version[0]) == 1:
        if int(scipy_version[1]) > 8:
            return True
        if int(scipy_version[1]) == 8 and int(scipy_version[2]) > 1:
            return True
    return False


SPARSE_DF_SUPPORTED = check_sparse_df_is_supported()
MSG_UNSUPPORTED_SP_DF = "Requires higher SciPy version"


@pytest.fixture(
    params=[make_sparse_matrix(), make_sparse_df()]
    + ([make_sparse_array()] if hasattr(sp, "csc_array") else [])
)
def sparse_X(request):
    return request.param


@pytest.fixture(params=[make_dense_df(), make_mixed_df()])
def dense_X(request):
    return request.param


# If the estimator doesn't support sparse data, passing either sparse data frames
# or sparse arrays/matrices should result in falling back to scikit-learn.
@pytest.mark.allow_sklearn_fallback
@pytest.mark.skipif(not SPARSE_DF_SUPPORTED, reason=MSG_UNSUPPORTED_SP_DF)
@pytest.mark.parametrize("estimator", [LinearRegression, PCA])
def test_no_sparse_support_falls_back_to_sklearn(estimator, sparse_X, mocker):
    mocker.patch("onedal.datatypes._data_conversion._convert_one_to_table")
    estimator().fit(sparse_X, np.r_[np.zeros(25), np.ones(25)])
    assert not onedal.datatypes._data_conversion._convert_one_to_table.called


# Note that some estimators that are implemented through daal4py do
# not end up using oneDAL tables, so they require a separate test.
@pytest.mark.allow_sklearn_fallback
@pytest.mark.skipif(not SPARSE_DF_SUPPORTED, reason=MSG_UNSUPPORTED_SP_DF)
@pytest.mark.parametrize(
    "estimator,params,internal_function",
    [
        (
            LogisticRegression,
            {},
            "daal4py.sklearn.linear_model.logistic_path.__logistic_regression_path",
        ),
        (
            Lasso,
            {},
            "daal4py.sklearn.linear_model._coordinate_descent._daal4py_fit_lasso",
        ),
        (
            TSNE,
            {"init": "random", "n_components": 2, "method": "barnes_hut"},
            "daal4py.sklearn.neighbors._base.daal4py_fit",
        ),
    ],
)
def test_no_sparse_support_falls_back_to_sklearn_daal4py(
    estimator, params, internal_function, sparse_X, mocker
):
    mocker.patch(internal_function)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estimator().set_params(**params).fit(sparse_X, np.r_[np.zeros(25), np.ones(25)])
    assert not eval(internal_function).called


def is_sparse_csr(x):
    return sp.issparse(x) and x.format == "csr"


# Passing data in any sparse format should result in oneDAL receiving a
# CSR matrix, regardless of what input it comes in.
@pytest.mark.parametrize("estimator", [SVR, BasicStatistics])
@pytest.mark.skipif(not SPARSE_DF_SUPPORTED, reason=MSG_UNSUPPORTED_SP_DF)
def test_sparse_input_is_passed_as_csr_to_onedal(estimator, sparse_X, mocker):
    # First check that it works without crashing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estimator().fit(sparse_X, np.arange(sparse_X.shape[0]))

    # Now check for inputs it receives
    mocker.patch("onedal.datatypes._data_conversion._convert_one_to_table")
    # Note: the call is expected to fail due to the mocking preventing
    # calls to C++, but the function that converts to a oneDAL table
    # should nevertheless be called regardless.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            estimator().fit(sparse_X, np.arange(sparse_X.shape[0]))
    except Exception:
        pass
    called_with_csr = False
    for call in onedal.datatypes._data_conversion._convert_one_to_table.mock_calls:
        if is_sparse_csr(call.args[0]):
            called_with_csr = True
            break
    assert called_with_csr


# All estimators should be able to support dense data.
@pytest.mark.parametrize("estimator", [LinearRegression, SVR, BasicStatistics])
@pytest.mark.skipif(not SPARSE_DF_SUPPORTED, reason=MSG_UNSUPPORTED_SP_DF)
def test_dense_data_is_not_converted_to_sparse(estimator, dense_X, mocker):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estimator().fit(dense_X, np.arange(dense_X.shape[0]))
    mocker.patch("onedal.datatypes._data_conversion._convert_one_to_table")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            estimator().fit(dense_X, np.arange(dense_X.shape[0]))
    except Exception:
        pass
    assert onedal.datatypes._data_conversion._convert_one_to_table.called
    called_with_csr = False
    called_with_numpy = False
    for call in onedal.datatypes._data_conversion._convert_one_to_table.mock_calls:
        if is_sparse_csr(call.args[0]):
            called_with_csr = True
        elif isinstance(call.args[0], np.ndarray):
            called_with_numpy = True
    assert not called_with_csr
    assert called_with_numpy


@pytest.mark.parametrize(
    "estimator,params,internal_function,position_X",
    [
        (
            LogisticRegression,
            {},
            "daal4py.sklearn.linear_model.logistic_path.__logistic_regression_path",
            0,
        ),
        (
            Lasso,
            {},
            "daal4py.sklearn.linear_model._coordinate_descent._daal4py_fit_lasso",
            1,
        ),
        (
            TSNE,
            {"init": "random", "n_components": 2, "method": "barnes_hut"},
            "daal4py.sklearn.neighbors._base.daal4py_fit",
            1,
        ),
    ],
)
@pytest.mark.skipif(not SPARSE_DF_SUPPORTED, reason=MSG_UNSUPPORTED_SP_DF)
def test_dense_data_is_not_converted_to_sparse_daal4py(
    estimator, params, internal_function, position_X, dense_X, mocker
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estimator().set_params(**params).fit(dense_X, np.r_[np.zeros(25), np.ones(25)])
    mocker.patch(internal_function)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            estimator().set_params(**params).fit(
                dense_X, np.r_[np.zeros(25), np.ones(25)]
            )
    except Exception:
        pass
    assert eval(internal_function).called
    called_with_csr = False
    called_with_numpy = False
    for call in eval(internal_function).mock_calls:
        if len(call.args) <= position_X:
            continue
        if is_sparse_csr(call.args[position_X]):
            called_with_csr = True
        elif isinstance(call.args[position_X], np.ndarray):
            called_with_numpy = True
    assert not called_with_csr
    assert called_with_numpy
