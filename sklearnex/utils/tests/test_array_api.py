# ==============================================================================
# Copyright 2024 Intel Corporation
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

import numpy as np
import pytest
from numpy.testing import assert_allclose

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

array_api_dataframes_and_namespaces = {
    "dpctl": "dpctl.tensor",
}

# TODO:
# add test suit for dpctl.tensor, dpnp.ndarray, numpy.ndarray without config_context(array_api_dispatch=True)).


@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(
        dataframe_filter_="numpy,dpctl,array_api", device_filter_="cpu,gpu"
    ),
)
def test_get_namespace_with_config_context(dataframe, queue):
    """Test get_namespace TBD"""
    from sklearnex import config_context
    from sklearnex.utils._array_api import get_namespace

    array_api_compat = pytest.importorskip("array_api_compat")

    X_np = np.asarray([[1, 2, 3]])
    X = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X)
        assert is_array_api_compliant
        if not dataframe in "numpy,array_api":
            # Rather than array_api_compat.get_namespace raw output
            # `get_namespace` has specific wrapper classes for `numpy.ndarray`
            # or `array-api-strict`.
            assert xp_out == array_api_compat.get_namespace(X)


@pytest.mark.skipif(
    not sklearn_check_version("1.2"),
    reason="Array API dispatch requires sklearn 1.2 version",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(
        dataframe_filter_="numpy,dpctl,array_api", device_filter_="cpu,gpu"
    ),
)
def test_get_namespace_with_patching(dataframe, queue):
    """Test get_namespace TBD
    with `patch_sklearn`
    """
    array_api_compat = pytest.importorskip("array_api_compat")

    from sklearnex import patch_sklearn

    patch_sklearn()

    from sklearn import config_context
    from sklearn.utils._array_api import get_namespace

    X_np = np.asarray([[1, 2, 3]])
    X = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X)
        assert is_array_api_compliant
        if not dataframe in "numpy,array_api":
            # Rather than array_api_compat.get_namespace raw output
            # `get_namespace` has specific wrapper classes for `numpy.ndarray`
            # or `array-api-strict`.
            assert xp_out == array_api_compat.get_namespace(X)


@pytest.mark.skipif(
    not sklearn_check_version("1.2"),
    reason="Array API dispatch requires sklearn 1.2 version",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(
        dataframe_filter_="dpctl,array_api", device_filter_="cpu,gpu"
    ),
)
def test_convert_to_numpy_with_patching(dataframe, queue):
    """Test _convert_to_numpy TBD with `patch_sklearn`"""
    pytest.importorskip("array_api_compat")

    from sklearnex import patch_sklearn

    patch_sklearn()

    from sklearn import config_context
    from sklearn.utils._array_api import _convert_to_numpy, get_namespace

    X_np = np.asarray([[1, 2, 3]])
    X = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    with config_context(array_api_dispatch=True):
        xp, _ = get_namespace(X)
        x_np = _convert_to_numpy(X, xp)
        assert type(X_np) == type(x_np)
        assert_allclose(X_np, x_np)


@pytest.mark.skipif(
    not sklearn_check_version("1.2"),
    reason="Array API dispatch requires sklearn 1.2 version",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(
        dataframe_filter_="numpy,dpctl,array_api", device_filter_="cpu,gpu"
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.float32, id=np.dtype(np.float32).name),
        pytest.param(np.float64, id=np.dtype(np.float64).name),
    ],
)
def test_validate_data_with_patching(dataframe, queue, dtype):
    """Test validate_data TBD with `patch_sklearn`"""
    pytest.importorskip("array_api_compat")

    from sklearnex import patch_sklearn

    patch_sklearn()

    from sklearn import config_context
    from sklearn.base import BaseEstimator

    if sklearn_check_version("1.6"):
        from sklearn.utils.validation import validate_data
    else:
        validate_data = BaseEstimator._validate_data

    from sklearn.utils._array_api import _convert_to_numpy, get_namespace

    X_np = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    X_df = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)
    with config_context(array_api_dispatch=True):
        est = BaseEstimator()
        xp, _ = get_namespace(X_df)
        X_df_res = validate_data(
            est, X_df, accept_sparse="csr", dtype=[xp.float64, xp.float32]
        )
        assert type(X_df) == type(X_df_res)
        if dataframe != "numpy":
            # _convert_to_numpy not designed for numpy.ndarray inputs.
            assert_allclose(_convert_to_numpy(X_df, xp), _convert_to_numpy(X_df_res, xp))
        else:
            assert_allclose(X_df, X_df_res)
