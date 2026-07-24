# ===============================================================================
# Copyright 2023 Intel Corporation
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
import os

import pytest
import scipy.sparse as sp

from daal4py.sklearn._utils import _package_check_version
from sklearnex import get_config

try:
    import dpnp

    dpnp_available = True
except ImportError:
    dpnp_available = False

torch_xpu_available = False
try:
    import torch

    torch_available = True

    torch_xpu_available = torch.xpu.is_available()
except ImportError:
    torch_available = False

try:
    # This should be lazy imported in the
    # future along with other popular
    # array_api libraries when testing
    # GPU-no-copy.
    import array_api_strict

    array_api_enabled = lambda: get_config()["array_api_dispatch"]
    array_api_enabled()
    array_api_modules = {"array_api": array_api_strict}


except (ImportError, KeyError):
    array_api_enabled = lambda: False
    array_api_modules = {}


import numpy as np
import pandas as pd

from onedal.datatypes._dlpack import dlpack_to_numpy
from onedal.tests.utils._device_selection import get_queues

test_frameworks = os.environ.get(
    "ONEDAL_PYTEST_FRAMEWORKS", "numpy,pandas,dpnp,array_api"
)


def get_dataframes_and_queues(dataframe_filter_=None, device_filter_="cpu,gpu"):
    """Get supported dataframes for testing.

    This is meant to be used for testing purposes only.
    It is to be used only for pytest test collection.

    Parameters
    ----------
    dataframe_filter_ : str, default=None
        Configure output pytest.params for the certain dataframe formats.
        When it evaluates False, it will default to value of ``test_frameworks``.
    device_filter_ : str, default="cpu,gpu"
        Configure output pytest.params with certain sycl queue for the dataframe,
        where it is applicable.

    Returns
    -------
    list[pytest.param]
        The list of pytest params, included dataframe name (str),
        sycl queue, if applicable for the test case, and test
        case id (str).

    Notes
    -----
        Do not use filters for the test cases disabling. Use `pytest.skip`
        or `pytest.xfail` instead.

    See Also
    --------
    _convert_to_dataframe : Converted input object to certain dataframe format.
    """
    dataframes_and_queues = []

    # filter dataframe_filter_ based on available test frameworks
    if dataframe_filter_:
        dataframe_filter_ = ",".join(
            [i for i in dataframe_filter_.split(",") if i in test_frameworks]
        )
    else:
        dataframe_filter_ = test_frameworks

    if "numpy" in dataframe_filter_:
        dataframes_and_queues.append(pytest.param("numpy", None, id="numpy"))
    if "pandas" in dataframe_filter_:
        dataframes_and_queues.append(pytest.param("pandas", None, id="pandas"))

    def get_df_and_q(dataframe: str):
        df_and_q = []
        for queue in get_queues(device_filter_):
            if queue:
                id = "{}-{}".format(dataframe, queue.id)
                df_and_q.append(pytest.param(dataframe, queue.values[0], id=id))
        return df_and_q

    if dpnp_available and "dpnp" in dataframe_filter_:
        dataframes_and_queues.extend(get_df_and_q("dpnp"))
    if (
        "array_api" in dataframe_filter_
        and "array_api" in array_api_modules
        or array_api_enabled()
    ):
        dataframes_and_queues.append(pytest.param("array_api", None, id="array_api"))

    return dataframes_and_queues


def _as_numpy(obj, *args, **kwargs):
    """Converted input object to numpy.ndarray format."""
    if dpnp_available and isinstance(obj, dpnp.ndarray):
        return obj.asnumpy(*args, **kwargs)
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        return obj.to_numpy(*args, **kwargs)
    if sp.issparse(obj):
        return obj.toarray(*args, **kwargs)
    try:
        return np.asarray(obj, *args, **kwargs)
    except (TypeError, RuntimeError, ValueError, BufferError):
        # np.asarray can't read a non-CPU device tensor (e.g. torch on xpu);
        # fall back to the library's standard dlpack host converter (which uses
        # np.from_dlpack). array_api libraries that np.asarray already handles
        # never reach this path.
        return dlpack_to_numpy(obj)


def _assert_in_namespace(obj, dataframe):
    """Assert obj belongs to the array namespace implied by ``dataframe``.

    Under array_api_dispatch, sklearnex outputs stay in the input namespace,
    so an on-device dpnp/array_api input should yield an on-device result
    (dpnp-in -> dpnp-out). Scalars are namespace-agnostic and are ignored.
    """
    if np.isscalar(obj):
        return
    if dataframe == "dpnp":
        assert dpnp_available and isinstance(
            obj, dpnp.ndarray
        ), f"expected dpnp output, got {type(obj)}"
    elif dataframe in array_api_modules:
        xp = array_api_modules[dataframe]
        assert (
            hasattr(obj, "__array_namespace__") and obj.__array_namespace__() is xp
        ), f"expected {dataframe} output, got {type(obj)}"


def _as_numpy_checked(obj, dataframe, *args, **kwargs):
    """Assert obj is in ``dataframe``'s namespace, then convert to numpy.

    Drop-in for ``_as_numpy`` on result values: verifies dpnp-in -> dpnp-out
    (on-device outputs are not silently host-transferred) before converting so
    the value can be compared against a numpy expected result.
    """
    _assert_in_namespace(obj, dataframe)
    return _as_numpy(obj, *args, **kwargs)


def skip_array_api_strict_readonly(dataframe):
    """Skip if ``dataframe`` is array_api_strict and numpy is older than 2.2.5.

    Estimators that rebuild a oneDAL model from fitted arrays (PCA/IncrementalPCA
    components_, DummyRegressor constant_) route them back through ``to_table``.
    numpy < 2.2.5 returns those arrays read-only, which ``to_table`` cannot export
    through DLPack, so array_api_strict inputs raise a BufferError / read-only
    assignment error under forced array_api_dispatch. numpy >= 2.2.5 returns
    writeable arrays.
    TODO: remove once the oneDAL data conversion handles read-only arrays.
    """
    if dataframe == "array_api" and not _package_check_version("2.2.5", np.__version__):
        pytest.skip("TODO: sklearnex read-only DLPack conversion fails on numpy<2.2.5")


def _convert_to_dataframe(obj, sycl_queue=None, target_df=None, *args, **kwargs):
    """Converted input object to certain dataframe format."""
    if target_df is None:
        return obj
    elif target_df == "numpy":
        # Numpy ndarray.
        # `sycl_queue` arg is ignored.
        return np.asarray(obj, *args, **kwargs)
    # Pandas Dataframe
    elif target_df == "pandas":
        if (
            "dtype" in kwargs
            and hasattr(obj, "astype")
            and np.issubdtype(kwargs["dtype"], np.integer)
        ):
            # Pandas float to int not allowed
            obj = obj.astype(kwargs["dtype"])
        if hasattr(obj, "ndim") and obj.ndim == 1:
            return pd.Series(obj, *args, **kwargs)
        else:
            return pd.DataFrame(obj, *args, **kwargs)
    # DPNP ndarray.
    elif target_df == "dpnp":
        return dpnp.asarray(
            obj, usm_type="device", sycl_queue=sycl_queue, *args, **kwargs
        )
    elif target_df in array_api_modules:
        # Array API input other than DPNP ndarray or Numpy ndarray.

        xp = array_api_modules[target_df]
        return xp.asarray(obj)

    raise RuntimeError("Unsupported dataframe conversion")
